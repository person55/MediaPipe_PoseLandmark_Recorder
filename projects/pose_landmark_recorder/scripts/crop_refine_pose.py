#!/usr/bin/env python
"""Crop-based post-cleaning pose refinement."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dance_pose_recorder.cleaned_preview_renderer import render_corrected_preview
from dance_pose_recorder.coordinate_transform import CoordinateTransformer
from dance_pose_recorder.crop_bbox import (
    CropBBox,
    compute_crop_bbox,
    crop_to_original_norm,
    is_near_crop_edge,
    smooth_bboxes,
)
from dance_pose_recorder.crop_apply import apply_crop_candidates
from dance_pose_recorder.crop_debug_renderer import render_crop_debug_images
from dance_pose_recorder.crop_refiner import (
    CropSegment,
    crop_segments_to_dataframe,
    detect_crop_segments,
    parse_crop_target_landmarks,
    parse_flags,
)
from dance_pose_recorder.crop_enhancement import detection_is_weak, enhance_crop, mean_visibility
from dance_pose_recorder.crop_mirror import (
    CONFUSION_COLUMNS,
    confusion_row,
    mirror_crop,
    pass_disagreement,
    unmirror_pose_landmarks,
    unmirror_world_landmarks,
)
from dance_pose_recorder.crop_rotation import (
    build_inverted_segments,
    detect_width_for_z,
    frame_rotations,
    rotate_crop,
    unrotate_pose_landmarks,
    unrotate_world_landmarks,
)
from dance_pose_recorder.data_writer import frame_to_csv_rows, make_frame_record
from dance_pose_recorder.output_layout import (
    CLEANED_FRAME_STATUS_CSV,
    CROP_REFINE_DIR,
    CROP_REFINE_CANDIDATE_SCORES_CSV,
    CROP_REFINE_CANDIDATES_CSV,
    CROP_REFINE_DEBUG_IMAGES_DIR,
    CROP_REFINE_POSE_CSV,
    CROP_REFINE_POSE_JSONL,
    CROP_REFINE_PREVIEW_MP4,
    CROP_REFINE_REPORT_JSON,
    CROP_REFINE_SEGMENTS_CSV,
    normalize_stage_output_dir,
    resolve_existing_file,
)
from dance_pose_recorder.pose_extractor import PoseExtractor
from dance_pose_recorder.stage_schema import COORD_FIELDS, CROP_SCORE_COLUMNS
from dance_pose_recorder.video_input import VideoFileReader


@dataclass(frozen=True)
class CropRefinementOptions:
    input_video: Path
    input_cleaned_csv: Path
    metadata: Path
    output: Path
    quality_report: Path | None = None
    interpolation_report: Path | None = None
    model: Path = Path("models/pose_landmarker.task")
    delegate: str = "cpu"
    running_mode: str = "video"
    crop_source: str = "torso"
    crop_margin_ratio: float = 1.65
    full_body_margin_ratio: float | None = 1.45
    crop_square: bool = True
    crop_min_size: int = 480
    target_flags: str = "unreliable,interpolated_outlier_removed,estimated_occluded_arm"
    target_landmarks: str = "arms,feet,hands_proxy"
    max_segment_length: int = 100
    segment_margin: int = 12
    target_segment_types: str = "mixed_problem_segment"
    include_short_invalid_cluster: bool = False
    allow_review_only: bool = False
    allow_missing_long_gap: bool = False
    accept_score_margin: float = 0.04
    max_segments: int | None = None
    bbox_smoothing_window: int = 5
    bbox_size_shrink_limit: float = 0.85
    save_candidates: bool = False
    save_refined: bool = False
    save_report: bool = False
    save_jsonl: bool = False
    save_preview: bool = False
    save_debug_images: bool = False
    allow_long_segments: bool = False
    # Rotation-augmented re-detection for inverted poses (Loop 8). Crops whose
    # cleaned body axis deviates far from upright are rotated in 90-degree
    # steps before detection, and landmarks are rotated back afterwards.
    rotate_inverted: bool = True
    rotation_min_angle_deg: float = 60.0
    # Low-light rescue (Loop 9): when every detection for a frame is weak,
    # retry once on a CLAHE-enhanced crop as an extra competing candidate.
    enhance_low_light: bool = True
    enhance_visibility_threshold: float = 0.5
    # Mirror/reverse passes and confusion diagnostics (Loop 10). The mirror
    # pass probes detector left-right asymmetry; the reverse pass approaches
    # each segment from the opposite temporal direction to escape tracker
    # inertia. Forward/mirror disagreement is recorded as a target-switch /
    # confusion diagnostic (metadata only, never auto-corrected).
    mirror_pass: bool = True
    reverse_pass: bool = True
    confusion_threshold: float = 0.03


@dataclass(frozen=True)
class CropRefinementResult:
    crop_refined_csv: Path | None
    crop_refined_jsonl: Path | None
    crop_candidates_csv: Path | None
    crop_candidate_scores_csv: Path
    crop_segments_csv: Path
    crop_refine_report: Path
    crop_refined_preview: Path | None
    crop_debug_dir: Path | None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Refine cleaned pose data using torso-centered crop re-detection.")
    parser.add_argument("--input-video", required=True, type=Path)
    parser.add_argument("--input-cleaned-csv", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--quality-report", type=Path, default=None)
    parser.add_argument("--interpolation-report", type=Path, default=None)
    parser.add_argument("--model", type=Path, default=Path("models/pose_landmarker.task"))
    parser.add_argument("--delegate", default="cpu", choices=["cpu", "gpu"])
    parser.add_argument("--running-mode", default="video", choices=["video", "image", "both"])
    parser.add_argument("--crop-source", default="torso", choices=["torso", "full_body"])
    parser.add_argument("--crop-margin-ratio", type=float, default=1.65)
    parser.add_argument(
        "--full-body-margin-ratio",
        type=float,
        default=1.45,
        help="Margin applied to full-body bbox size.",
    )
    parser.add_argument("--crop-square", action="store_true", default=True)
    parser.add_argument("--no-crop-square", action="store_false", dest="crop_square")
    parser.add_argument("--crop-min-size", type=int, default=480)
    parser.add_argument("--target-flags", default="unreliable,interpolated_outlier_removed,estimated_occluded_arm")
    parser.add_argument("--target-landmarks", default="arms,feet,hands_proxy")
    parser.add_argument("--max-segment-length", type=int, default=100)
    parser.add_argument("--segment-margin", type=int, default=12)
    parser.add_argument("--target-segment-types", default="mixed_problem_segment")
    parser.add_argument("--include-short-invalid-cluster", action="store_true")
    parser.add_argument("--allow-review-only", action="store_true")
    parser.add_argument("--allow-missing-long-gap", action="store_true")
    parser.add_argument("--accept-score-margin", type=float, default=0.04)
    parser.add_argument("--max-segments", type=int, default=None)
    parser.add_argument("--bbox-smoothing-window", type=int, default=5)
    parser.add_argument("--bbox-size-shrink-limit", type=float, default=0.85)
    parser.add_argument("--save-candidates", action="store_true")
    parser.add_argument("--save-refined", action="store_true")
    parser.add_argument("--save-report", action="store_true")
    parser.add_argument("--save-jsonl", action="store_true")
    parser.add_argument("--save-preview", action="store_true")
    parser.add_argument("--save-debug-images", action="store_true")
    parser.add_argument("--allow-long-segments", action="store_true")
    parser.add_argument("--rotate-inverted", action="store_true", default=True)
    parser.add_argument("--no-rotate-inverted", action="store_false", dest="rotate_inverted")
    parser.add_argument("--rotation-min-angle-deg", type=float, default=60.0)
    parser.add_argument("--enhance-low-light", action="store_true", default=True)
    parser.add_argument("--no-enhance-low-light", action="store_false", dest="enhance_low_light")
    parser.add_argument("--enhance-visibility-threshold", type=float, default=0.5)
    parser.add_argument("--mirror-pass", action="store_true", default=True)
    parser.add_argument("--no-mirror-pass", action="store_false", dest="mirror_pass")
    parser.add_argument("--reverse-pass", action="store_true", default=True)
    parser.add_argument("--no-reverse-pass", action="store_false", dest="reverse_pass")
    parser.add_argument("--confusion-threshold", type=float, default=0.03)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = crop_refine_pose(
        CropRefinementOptions(
            input_video=args.input_video,
            input_cleaned_csv=args.input_cleaned_csv,
            metadata=args.metadata,
            output=args.output,
            quality_report=args.quality_report,
            interpolation_report=args.interpolation_report,
            model=args.model,
            delegate=args.delegate,
            running_mode=args.running_mode,
            crop_source=args.crop_source,
            crop_margin_ratio=args.crop_margin_ratio,
            full_body_margin_ratio=args.full_body_margin_ratio,
            crop_square=args.crop_square,
            crop_min_size=args.crop_min_size,
            target_flags=args.target_flags,
            target_landmarks=args.target_landmarks,
            max_segment_length=args.max_segment_length,
            segment_margin=args.segment_margin,
            target_segment_types=args.target_segment_types,
            include_short_invalid_cluster=args.include_short_invalid_cluster,
            allow_review_only=args.allow_review_only,
            allow_missing_long_gap=args.allow_missing_long_gap,
            accept_score_margin=args.accept_score_margin,
            max_segments=args.max_segments,
            bbox_smoothing_window=args.bbox_smoothing_window,
            bbox_size_shrink_limit=args.bbox_size_shrink_limit,
            save_candidates=args.save_candidates,
            save_refined=args.save_refined,
            save_report=args.save_report,
            save_jsonl=args.save_jsonl,
            save_preview=args.save_preview,
            save_debug_images=args.save_debug_images,
            allow_long_segments=args.allow_long_segments,
            rotate_inverted=args.rotate_inverted,
            rotation_min_angle_deg=args.rotation_min_angle_deg,
            enhance_low_light=args.enhance_low_light,
            enhance_visibility_threshold=args.enhance_visibility_threshold,
            mirror_pass=args.mirror_pass,
            reverse_pass=args.reverse_pass,
            confusion_threshold=args.confusion_threshold,
        )
    )
    print(f"Wrote crop refinement outputs to {result.crop_segments_csv.parent}")
    for path in (
        result.crop_refined_csv,
        result.crop_refined_jsonl,
        result.crop_candidates_csv,
        result.crop_candidate_scores_csv,
        result.crop_segments_csv,
        result.crop_refine_report,
        result.crop_refined_preview,
        result.crop_debug_dir,
    ):
        if path:
            print(path)


def crop_refine_pose(options: CropRefinementOptions) -> CropRefinementResult:
    if options.running_mode != "video":
        print(f"warning: running_mode={options.running_mode} requested; v1 uses video mode")

    metadata = json.loads(options.metadata.read_text(encoding="utf-8"))
    cleaned = pd.read_csv(options.input_cleaned_csv, low_memory=False)
    output_dir = normalize_stage_output_dir(options.output, CROP_REFINE_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    fps = float(metadata.get("fps") or 30.0)
    total_frames = int(metadata.get("frame_count_written") or cleaned["frame"].max() + 1)
    total_frames = min(total_frames, int(cleaned["frame"].max()) + 1)

    segments = detect_crop_segments(
        cleaned,
        total_frames=total_frames,
        target_landmarks=options.target_landmarks,
        target_flags=options.target_flags,
        max_segment_length=options.max_segment_length,
        segment_margin=options.segment_margin,
        allow_long_segments=options.allow_long_segments,
        target_segment_types=options.target_segment_types,
        include_short_invalid_cluster=options.include_short_invalid_cluster,
        exclude_missing_long_gap=not options.allow_missing_long_gap,
        exclude_review_only=not options.allow_review_only,
    )
    if options.max_segments is not None:
        segments = segments[: options.max_segments]

    if options.rotate_inverted:
        pose_frames = {
            int(frame): group for frame, group in cleaned[cleaned["source"] == "pose"].groupby("frame", sort=False)
        }
        segments = segments + build_inverted_segments(
            pose_frames,
            total_frames,
            int(metadata.get("width") or 1920),
            int(metadata.get("height") or 1080),
            options.rotation_min_angle_deg,
            segments,
            options.max_segment_length,
        )

    crop_segments_csv = output_dir / CROP_REFINE_SEGMENTS_CSV
    crop_segments_to_dataframe(segments, fps=fps).to_csv(crop_segments_csv, index=False)

    bboxes, candidates, confusion_diagnostics = _redetect_crop_candidates(options, metadata, cleaned, segments)
    refined, score_rows, segment_summaries = apply_crop_candidates(
        cleaned, candidates, segments, options.accept_score_margin
    )
    confusion_csv = output_dir / "crop_confusion_diagnostics.csv"
    confusion_diagnostics.to_csv(confusion_csv, index=False)

    crop_candidate_scores_csv = output_dir / CROP_REFINE_CANDIDATE_SCORES_CSV
    pd.DataFrame(score_rows, columns=CROP_SCORE_COLUMNS).to_csv(crop_candidate_scores_csv, index=False)

    crop_candidates_csv = None
    if options.save_candidates:
        crop_candidates_csv = output_dir / CROP_REFINE_CANDIDATES_CSV
        candidates.to_csv(crop_candidates_csv, index=False)

    crop_refined_csv = None
    if options.save_refined:
        crop_refined_csv = output_dir / CROP_REFINE_POSE_CSV
        refined.to_csv(crop_refined_csv, index=False)

    crop_refined_jsonl = None
    if options.save_jsonl:
        crop_refined_jsonl = output_dir / CROP_REFINE_POSE_JSONL
        write_crop_refined_jsonl(refined, crop_refined_jsonl, metadata)

    report = _build_report(
        metadata=metadata,
        options=options,
        frames_total=total_frames,
        segment_summaries=segment_summaries,
        candidates=candidates,
        refined=refined,
        confusion_diagnostics=confusion_diagnostics,
    )
    crop_refine_report = output_dir / CROP_REFINE_REPORT_JSON
    crop_refine_report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    crop_refined_preview = None
    if options.save_preview:
        frame_status = _frame_status_for_preview(options, refined, total_frames, fps)
        crop_refined_preview = output_dir / CROP_REFINE_PREVIEW_MP4
        render_corrected_preview(options.input_video, crop_refined_preview, refined, frame_status, metadata)

    crop_debug_dir = None
    if options.save_debug_images:
        crop_debug_dir = output_dir / CROP_REFINE_DEBUG_IMAGES_DIR
        render_crop_debug_images(
            options.input_video,
            crop_debug_dir,
            segments,
            bboxes,
            cleaned,
            candidates,
            pd.DataFrame(score_rows, columns=CROP_SCORE_COLUMNS),
        )

    return CropRefinementResult(
        crop_refined_csv=crop_refined_csv,
        crop_refined_jsonl=crop_refined_jsonl,
        crop_candidates_csv=crop_candidates_csv,
        crop_candidate_scores_csv=crop_candidate_scores_csv,
        crop_segments_csv=crop_segments_csv,
        crop_refine_report=crop_refine_report,
        crop_refined_preview=crop_refined_preview,
        crop_debug_dir=crop_debug_dir,
    )


def _redetect_crop_candidates(
    options: CropRefinementOptions,
    metadata: dict,
    cleaned: pd.DataFrame,
    segments: list[CropSegment],
) -> tuple[dict[int, CropBBox], pd.DataFrame]:
    attempted = [segment for segment in segments if segment.crop_attempted]
    if not attempted:
        return {}, pd.DataFrame(), pd.DataFrame(columns=CONFUSION_COLUMNS)

    total_frames = int(metadata.get("frame_count_written") or cleaned["frame"].max() + 1)
    pose_by_frame = {
        int(frame): group.copy()
        for frame, group in cleaned[cleaned["source"] == "pose"].groupby("frame", sort=False)
    }
    frame_to_segment = _frame_segment_map(attempted)

    with VideoFileReader(options.input_video) as reader:
        raw_bboxes = _build_bboxes(
            pose_by_frame=pose_by_frame,
            frames=sorted(frame_to_segment),
            frame_width=reader.info.width,
            frame_height=reader.info.height,
            options=options,
        )
        bboxes = smooth_bboxes(raw_bboxes, options.bbox_smoothing_window, options.bbox_size_shrink_limit)
        rotations: dict[int, int] = {}
        if options.rotate_inverted:
            rotations = frame_rotations(
                pose_by_frame,
                sorted(frame_to_segment),
                reader.info.width,
                reader.info.height,
                options.rotation_min_angle_deg,
            )
        rows, diagnostic_rows = _detect_crops(
            reader=reader,
            bboxes=bboxes,
            frame_to_segment=frame_to_segment,
            options=options,
            metadata=metadata,
            total_frames=total_frames,
            rotations=rotations,
        )
    return bboxes, pd.DataFrame(rows), pd.DataFrame(diagnostic_rows, columns=CONFUSION_COLUMNS)


def _build_bboxes(
    pose_by_frame: dict[int, pd.DataFrame],
    frames: list[int],
    frame_width: int,
    frame_height: int,
    options: CropRefinementOptions,
) -> dict[int, CropBBox]:
    bboxes: dict[int, CropBBox] = {}
    previous: CropBBox | None = None
    for frame in frames:
        frame_rows = pose_by_frame.get(frame, pd.DataFrame())
        bbox = compute_crop_bbox(
            frame_rows,
            frame_width=frame_width,
            frame_height=frame_height,
            frame=frame,
            previous_bbox=previous,
            crop_margin_ratio=options.crop_margin_ratio,
            full_body_margin_ratio=options.full_body_margin_ratio,
            crop_square=options.crop_square,
            crop_min_size=options.crop_min_size,
            crop_source=options.crop_source,
        )
        if bbox is not None:
            bboxes[frame] = bbox
            previous = bbox
    return bboxes


def _detect_crops(
    reader: VideoFileReader,
    bboxes: dict[int, CropBBox],
    frame_to_segment: dict[int, CropSegment],
    options: CropRefinementOptions,
    metadata: dict,
    total_frames: int,
    rotations: dict[int, int] | None = None,
) -> tuple[list[dict], list[dict]]:
    rotations = rotations or {}
    rows: list[dict] = []
    diagnostic_rows: list[dict] = []
    session_id = str(metadata.get("session_id") or options.output.name)
    transformer = CoordinateTransformer(origin_policy=str(metadata.get("origin_policy") or "raw"))
    if transformer.origin_policy != "raw":
        print("warning: crop refinement v1 is calibrated for origin_policy=raw")

    def restored_pose(detection, bbox, snap: int) -> list[dict]:
        raw_pose = unrotate_pose_landmarks(detection["pose_landmarks"], snap)
        return _restore_pose_landmarks(
            raw_pose, bbox, z_detect_width=detect_width_for_z(bbox.w, bbox.h, snap)
        )

    def add_candidate_rows(
        frame_index: int,
        timestamp_ms: int,
        bbox,
        segment,
        snap: int,
        pose_landmarks: list[dict],
        world_landmarks: list[dict],
        enhanced: bool = False,
        pass_name: str = "forward",
    ) -> None:
        transformed = transformer.transform_landmarks(world_landmarks)
        record = make_frame_record(
            session_id=session_id,
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
            pose_landmarks=pose_landmarks,
            pose_world_landmarks=world_landmarks,
            transformed_landmarks=transformed,
        )
        for row in frame_to_csv_rows(record):
            row.update(
                {
                    "crop_segment_id": segment.crop_segment_id,
                    "crop_x0": bbox.x0,
                    "crop_y0": bbox.y0,
                    "crop_w": bbox.w,
                    "crop_h": bbox.h,
                    "crop_margin_ratio": bbox.margin_ratio,
                    "crop_running_mode": "video",
                    "crop_rotation_deg": snap,
                    "crop_enhanced": enhanced,
                    "crop_pass": pass_name,
                }
            )
            if row["source"] == "pose":
                restored = pose_landmarks[int(row["landmark_id"])] if pose_landmarks else {}
                row["crop_x_norm"] = restored.get("crop_x_norm")
                row["crop_y_norm"] = restored.get("crop_y_norm")
                row["crop_edge_risk"] = restored.get("crop_edge_risk")
            else:
                row["crop_x_norm"] = np.nan
                row["crop_y_norm"] = np.nan
                row["crop_edge_risk"] = False
            rows.append(row)

    # Separate video-mode trackers keep the streams isolated:
    #  - baseline: exactly the frames the pre-Loop-8 stage processed, so its
    #    candidates (and therefore prior acceptances) are unaffected. Mixing
    #    new frames or rotated crops into this stream was observed to disturb
    #    tracking on later baseline frames.
    #  - inverted-upright: upright detections on new inverted-pose segments.
    #  - rotated / enhanced / mirror / reverse: one tracker per extra pass.
    extra_extractors: list[PoseExtractor] = []

    def open_extractor() -> PoseExtractor:
        extractor = PoseExtractor(options.model, delegate=options.delegate)
        extractor.__enter__()
        extra_extractors.append(extractor)
        return extractor

    inverted_upright_extractor: PoseExtractor | None = None
    rotated_extractor: PoseExtractor | None = None
    enhanced_extractor: PoseExtractor | None = None
    mirror_extractor: PoseExtractor | None = None
    reverse_extractor: PoseExtractor | None = None
    reverse_buffer: list[tuple[int, CropBBox, CropSegment, np.ndarray]] = []
    reverse_state = {"segment_id": None, "timestamp_ms": 0}
    frame_duration_ms = max(1, int(round(1000.0 / float(metadata.get("fps") or 30.0))))

    def flush_reverse_buffer() -> None:
        # Re-detect the buffered segment in reverse frame order on its own
        # tracker so it approaches the segment from the opposite temporal
        # direction. Timestamps are synthetic but monotonic for the tracker.
        if reverse_extractor is None or not reverse_buffer:
            reverse_buffer.clear()
            return
        for frame_index, bbox, segment, crop in reversed(reverse_buffer):
            reverse_state["timestamp_ms"] += frame_duration_ms
            detection = reverse_extractor.detect(crop, reverse_state["timestamp_ms"])
            add_candidate_rows(
                frame_index,
                reverse_state["timestamp_ms"],
                bbox,
                segment,
                0,
                restored_pose(detection, bbox, 0),
                detection["pose_world_landmarks"],
                pass_name="reverse",
            )
        reverse_buffer.clear()

    try:
        with PoseExtractor(options.model, delegate=options.delegate) as baseline_extractor:
            has_inverted = any(
                segment.segment_type == "inverted_pose_segment" for segment in frame_to_segment.values()
            )
            if has_inverted:
                inverted_upright_extractor = open_extractor()
            if rotations:
                rotated_extractor = open_extractor()
            if options.enhance_low_light:
                enhanced_extractor = open_extractor()
            if options.mirror_pass:
                mirror_extractor = open_extractor()
            if options.reverse_pass:
                reverse_extractor = open_extractor()
            for frame in tqdm(reader.frames(max_frames=total_frames), total=total_frames, unit="frame"):
                bbox = bboxes.get(frame.frame_index)
                if bbox is None:
                    continue
                x0, y0, x1, y1 = bbox.to_int_tuple()
                if x1 <= x0 or y1 <= y0:
                    continue
                crop = frame.image_bgr[y0:y1, x0:x1]
                if crop.size == 0:
                    continue
                segment = frame_to_segment[frame.frame_index]
                if reverse_extractor is not None and reverse_state["segment_id"] != segment.crop_segment_id:
                    flush_reverse_buffer()
                    reverse_state["segment_id"] = segment.crop_segment_id
                if segment.segment_type == "inverted_pose_segment" and inverted_upright_extractor is not None:
                    upright_extractor = inverted_upright_extractor
                else:
                    upright_extractor = baseline_extractor
                result = upright_extractor.detect(crop, frame.timestamp_ms)
                forward_pose = restored_pose(result, bbox, 0)
                add_candidate_rows(
                    frame.frame_index,
                    frame.timestamp_ms,
                    bbox,
                    segment,
                    0,
                    forward_pose,
                    result["pose_world_landmarks"],
                )
                snap = rotations.get(frame.frame_index, 0)
                rotated = None
                rotated_pose = None
                if snap and rotated_extractor is not None:
                    rotated = rotated_extractor.detect(rotate_crop(crop, snap), frame.timestamp_ms)
                    rotated_pose = restored_pose(rotated, bbox, snap)
                    add_candidate_rows(
                        frame.frame_index,
                        frame.timestamp_ms,
                        bbox,
                        segment,
                        snap,
                        rotated_pose,
                        unrotate_world_landmarks(rotated["pose_world_landmarks"], snap),
                    )
                if enhanced_extractor is not None:
                    # Rescue pass: only when every detection for the frame is
                    # weak does an enhanced-contrast retry become a candidate.
                    weak = detection_is_weak(result["pose_landmarks"], options.enhance_visibility_threshold) and (
                        rotated is None
                        or detection_is_weak(rotated["pose_landmarks"], options.enhance_visibility_threshold)
                    )
                    if weak:
                        enhanced = enhanced_extractor.detect(enhance_crop(crop), frame.timestamp_ms)
                        add_candidate_rows(
                            frame.frame_index,
                            frame.timestamp_ms,
                            bbox,
                            segment,
                            0,
                            restored_pose(enhanced, bbox, 0),
                            enhanced["pose_world_landmarks"],
                            enhanced=True,
                        )
                mirror_pose = None
                if mirror_extractor is not None:
                    mirrored = mirror_extractor.detect(mirror_crop(crop), frame.timestamp_ms)
                    unmirrored = unmirror_pose_landmarks(mirrored["pose_landmarks"])
                    if unmirrored:
                        mirror_pose = _restore_pose_landmarks(unmirrored, bbox)
                        add_candidate_rows(
                            frame.frame_index,
                            frame.timestamp_ms,
                            bbox,
                            segment,
                            0,
                            mirror_pose,
                            unmirror_world_landmarks(mirrored["pose_world_landmarks"]),
                            pass_name="mirror",
                        )
                if reverse_extractor is not None:
                    reverse_buffer.append((frame.frame_index, bbox, segment, crop.copy()))
                diagnostic_rows.append(
                    confusion_row(
                        frame.frame_index,
                        segment.crop_segment_id,
                        mean_visibility(result["pose_landmarks"]),
                        pass_disagreement(forward_pose, mirror_pose) if mirror_pose else None,
                        pass_disagreement(forward_pose, rotated_pose) if rotated_pose else None,
                        options.confusion_threshold,
                    )
                )
            flush_reverse_buffer()
    finally:
        for extractor in extra_extractors:
            extractor.__exit__(None, None, None)
    return rows, diagnostic_rows


def _restore_pose_landmarks(
    landmarks: list[dict], bbox: CropBBox, z_detect_width: float | None = None
) -> list[dict]:
    # MediaPipe normalizes z on the same scale as x (image width), so z from a
    # crop detection must be rescaled by detected-image width / frame width
    # alongside x/y. For 90-degree rotated crops the detector saw the crop
    # height as its width.
    z_scale = float(z_detect_width if z_detect_width is not None else bbox.w) / float(bbox.frame_width)
    restored = []
    for landmark in landmarks:
        item = dict(landmark)
        x_crop = float(landmark["x"])
        y_crop = float(landmark["y"])
        x_original, y_original = crop_to_original_norm(x_crop, y_crop, bbox)
        item["x"] = x_original
        item["y"] = y_original
        z_crop = landmark.get("z")
        if z_crop is not None and isfinite(float(z_crop)):
            item["z"] = float(z_crop) * z_scale
        item["crop_x_norm"] = x_crop
        item["crop_y_norm"] = y_crop
        item["crop_edge_risk"] = is_near_crop_edge(x_crop, y_crop)
        restored.append(item)
    return restored


def _frame_segment_map(segments: list[CropSegment]) -> dict[int, CropSegment]:
    mapping: dict[int, CropSegment] = {}
    for segment in segments:
        for frame in range(segment.start_frame, segment.end_frame + 1):
            mapping[frame] = segment
    return mapping


def _build_report(
    metadata: dict,
    options: CropRefinementOptions,
    frames_total: int,
    segment_summaries: list[dict],
    candidates: pd.DataFrame,
    refined: pd.DataFrame,
    confusion_diagnostics: pd.DataFrame | None = None,
) -> dict:
    counts = refined["crop_refine_status"].value_counts().to_dict()
    return {
        "session_id": metadata.get("session_id"),
        "input_video": str(options.input_video),
        "input_cleaned_csv": str(options.input_cleaned_csv),
        "frames_total": frames_total,
        "fps": float(metadata.get("fps") or 30.0),
        "settings": {
            "delegate": options.delegate,
            "running_mode": "video",
            "crop_source": options.crop_source,
            "crop_margin_ratio": options.crop_margin_ratio,
            "full_body_margin_ratio": options.full_body_margin_ratio
            if options.full_body_margin_ratio is not None
            else options.crop_margin_ratio,
            "crop_square": options.crop_square,
            "crop_min_size": options.crop_min_size,
            "target_flags": sorted(parse_flags(options.target_flags)),
            "target_landmarks": sorted(parse_crop_target_landmarks(options.target_landmarks)),
            "max_segment_length": options.max_segment_length,
            "segment_margin": options.segment_margin,
            "target_segment_types": sorted(parse_flags(options.target_segment_types)),
            "include_short_invalid_cluster": options.include_short_invalid_cluster,
            "allow_review_only": options.allow_review_only,
            "allow_missing_long_gap": options.allow_missing_long_gap,
            "accept_score_margin": options.accept_score_margin,
            "allow_long_segments": options.allow_long_segments,
            "rotate_inverted": options.rotate_inverted,
            "rotation_min_angle_deg": options.rotation_min_angle_deg,
            "enhance_low_light": options.enhance_low_light,
            "enhance_visibility_threshold": options.enhance_visibility_threshold,
        },
        "enhancement": {
            "enhanced_frames": int(
                candidates.loc[candidates["crop_enhanced"] == True, "frame"].nunique()  # noqa: E712
            )
            if not candidates.empty and "crop_enhanced" in candidates.columns
            else 0,
        },
        "passes": candidates.groupby("crop_pass")["frame"].nunique().to_dict()
        if not candidates.empty and "crop_pass" in candidates.columns
        else {},
        "confusion_diagnostics": {
            "frames_checked": int(len(confusion_diagnostics))
            if confusion_diagnostics is not None
            else 0,
            "possible_confusion_frames": int(confusion_diagnostics["possible_confusion"].sum())
            if confusion_diagnostics is not None and not confusion_diagnostics.empty
            else 0,
            "threshold": options.confusion_threshold,
            "note": "diagnostic metadata only; confusion is surfaced, never auto-corrected",
        },
        "rotation": {
            "rotated_frames": int(candidates["crop_rotation_deg"].gt(0).groupby(candidates["frame"]).any().sum())
            if not candidates.empty and "crop_rotation_deg" in candidates.columns
            else 0,
            "rotation_degrees": candidates[candidates["crop_rotation_deg"] > 0]
            .groupby("crop_rotation_deg")["frame"]
            .nunique()
            .to_dict()
            if not candidates.empty and "crop_rotation_deg" in candidates.columns
            else {},
        },
        "segment_selection": _segment_selection_summary(segment_summaries),
        "counts": {
            "crop_segment_count": len(segment_summaries),
            "crop_attempted_segment_count": int(sum(1 for segment in segment_summaries if segment.get("crop_attempted"))),
            "crop_candidate_row_count": int(len(candidates)),
            "accepted_row_count": int(counts.get("crop_accepted", 0)),
            "rejected_row_count": int(counts.get("crop_rejected", 0)),
            "unavailable_row_count": int(counts.get("crop_unavailable", 0)),
        },
        "segments": segment_summaries,
        "notes": "Crop refinement does not generate motion. It creates crop-based MediaPipe candidates and accepts them only when they score better than cleaned values.",
    }


def _segment_selection_summary(segment_summaries: list[dict]) -> dict:
    reasons = pd.Series([str(segment.get("selection_reason", "")) for segment in segment_summaries])
    reason_counts = reasons.value_counts().to_dict() if not reasons.empty else {}
    return {
        "total_candidate_segments": int(len(segment_summaries)),
        "selected_segment_count": int(sum(1 for segment in segment_summaries if bool(segment.get("selected_for_crop")))),
        "excluded_review_only_count": int(reason_counts.get("excluded_review_only", 0)),
        "excluded_missing_long_gap_count": int(reason_counts.get("excluded_missing_long_gap", 0)),
        "excluded_long_unreliable_count": int(reason_counts.get("excluded_long_unreliable_run", 0)),
        "excluded_too_long_count": int(reason_counts.get("excluded_too_long", 0)),
    }


def _frame_status_for_preview(options: CropRefinementOptions, refined: pd.DataFrame, total_frames: int, fps: float) -> pd.DataFrame:
    candidate = resolve_existing_file(
        options.input_cleaned_csv.parent,
        CLEANED_FRAME_STATUS_CSV,
        ("frame_status.csv",),
    )
    if candidate.exists():
        return pd.read_csv(candidate)
    rows = []
    for frame in range(total_frames):
        frame_df = refined[refined["frame"] == frame]
        pose = frame_df[frame_df["source"] == "pose"]
        long_missing = bool((pose["quality_flag"] == "missing_long_gap").all()) if not pose.empty else True
        rows.append(
            {
                "frame": frame,
                "time_sec": frame / fps,
                "has_pose": bool((pose["quality_flag"] != "missing_long_gap").any()),
                "is_inside_long_missing_range": long_missing,
            }
        )
    return pd.DataFrame(rows)


def write_crop_refined_jsonl(refined: pd.DataFrame, output_path: str | Path, metadata: dict) -> None:
    output = Path(output_path)
    session_id = metadata.get("session_id")
    with output.open("w", encoding="utf-8") as file:
        for frame, frame_df in refined.groupby("frame", sort=True):
            record = {
                "session_id": session_id,
                "frame": int(frame),
                "time_sec": float(frame_df["time_sec"].iloc[0]),
                "pose_landmarks": _jsonl_landmarks(frame_df[frame_df["source"] == "pose"]),
                "pose_world_landmarks": _jsonl_landmarks(frame_df[frame_df["source"] == "pose_world"]),
                "quality_summary": frame_df["quality_flag"].value_counts().to_dict(),
                "crop_refine_summary": frame_df["crop_refine_status"].value_counts().to_dict(),
            }
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def _jsonl_landmarks(frame_df: pd.DataFrame) -> list[dict]:
    landmarks = []
    for row in frame_df.itertuples(index=False):
        if not bool(row.is_valid) and not bool(row.is_interpolated):
            continue
        item = {
            "id": int(row.landmark_id),
            "name": row.landmark_name,
            "quality_flag": row.quality_flag,
            "crop_refine_status": row.crop_refine_status,
        }
        for field in COORD_FIELDS:
            value = getattr(row, field)
            if pd.notna(value):
                item[field] = float(value)
        landmarks.append(item)
    return landmarks


if __name__ == "__main__":
    main()
