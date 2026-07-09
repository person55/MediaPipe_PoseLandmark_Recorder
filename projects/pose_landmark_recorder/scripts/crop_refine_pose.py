#!/usr/bin/env python
"""Crop-based post-cleaning pose refinement."""

from __future__ import annotations

from dataclasses import dataclass
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
from dance_pose_recorder.crop_candidate_scorer import (
    crop_valid_score,
    decide_crop_candidate,
    visibility_gain_score,
)
from dance_pose_recorder.crop_debug_renderer import render_crop_debug_images
from dance_pose_recorder.crop_refiner import (
    CropSegment,
    crop_segments_to_dataframe,
    detect_crop_segments,
    parse_crop_target_landmarks,
    parse_flags,
)
from dance_pose_recorder.data_writer import frame_to_csv_rows, make_frame_record
from dance_pose_recorder.landmark_schema import POSE_LANDMARK_NAMES
from dance_pose_recorder.pose_candidate_scorer import confidence_score
from dance_pose_recorder.pose_extractor import PoseExtractor
from dance_pose_recorder.video_input import VideoFileReader


COORD_FIELDS = ["x", "y", "z", "visibility", "presence", "tx", "ty", "tz"]
CROP_COLUMNS = [
    "crop_refine_status",
    "crop_refine_source",
    "crop_score_before",
    "crop_score_after",
    "crop_score_delta",
    "crop_segment_id",
    "crop_reason",
    "crop_x0",
    "crop_y0",
    "crop_w",
    "crop_h",
    "crop_margin_ratio",
    "crop_running_mode",
]
SCORE_COLUMNS = [
    "crop_segment_id",
    "frame",
    "source",
    "landmark_id",
    "landmark_name",
    "quality_flag_before",
    "crop_refine_status",
    "crop_reason",
    "score_before",
    "score_after",
    "score_delta",
]


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
    accept_score_margin: float = 0.06
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
    parser.add_argument("--accept-score-margin", type=float, default=0.06)
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
        )
    )
    print(f"Wrote crop refinement outputs to {args.output}")
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
    output_dir = options.output
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

    crop_segments_csv = output_dir / "crop_segments.csv"
    crop_segments_to_dataframe(segments, fps=fps).to_csv(crop_segments_csv, index=False)

    bboxes, candidates = _redetect_crop_candidates(options, metadata, cleaned, segments)
    refined, score_rows, segment_summaries = _apply_crop_candidates(cleaned, candidates, segments, options)

    crop_candidate_scores_csv = output_dir / "crop_candidate_scores.csv"
    pd.DataFrame(score_rows, columns=SCORE_COLUMNS).to_csv(crop_candidate_scores_csv, index=False)

    crop_candidates_csv = None
    if options.save_candidates:
        crop_candidates_csv = output_dir / "crop_candidates.csv"
        candidates.to_csv(crop_candidates_csv, index=False)

    crop_refined_csv = None
    if options.save_refined:
        crop_refined_csv = output_dir / "crop_refined_pose.csv"
        refined.to_csv(crop_refined_csv, index=False)

    crop_refined_jsonl = None
    if options.save_jsonl:
        crop_refined_jsonl = output_dir / "crop_refined_pose.jsonl"
        write_crop_refined_jsonl(refined, crop_refined_jsonl, metadata)

    report = _build_report(
        metadata=metadata,
        options=options,
        frames_total=total_frames,
        segment_summaries=segment_summaries,
        candidates=candidates,
        refined=refined,
    )
    crop_refine_report = output_dir / "crop_refine_report.json"
    crop_refine_report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    crop_refined_preview = None
    if options.save_preview:
        frame_status = _frame_status_for_preview(options, refined, total_frames, fps)
        crop_refined_preview = output_dir / "crop_refined_preview.mp4"
        render_corrected_preview(options.input_video, crop_refined_preview, refined, frame_status, metadata)

    crop_debug_dir = None
    if options.save_debug_images:
        crop_debug_dir = output_dir / "crop_debug_images"
        render_crop_debug_images(
            options.input_video,
            crop_debug_dir,
            segments,
            bboxes,
            cleaned,
            candidates,
            pd.DataFrame(score_rows, columns=SCORE_COLUMNS),
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
        return {}, pd.DataFrame()

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
        rows = _detect_crops(
            reader=reader,
            bboxes=bboxes,
            frame_to_segment=frame_to_segment,
            options=options,
            metadata=metadata,
            total_frames=total_frames,
        )
    return bboxes, pd.DataFrame(rows)


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
) -> list[dict]:
    rows: list[dict] = []
    session_id = str(metadata.get("session_id") or options.output.name)
    transformer = CoordinateTransformer(origin_policy=str(metadata.get("origin_policy") or "raw"))
    if transformer.origin_policy != "raw":
        print("warning: crop refinement v1 is calibrated for origin_policy=raw")

    with PoseExtractor(options.model, delegate=options.delegate) as extractor:
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
            result = extractor.detect(crop, frame.timestamp_ms)
            pose_landmarks = _restore_pose_landmarks(result["pose_landmarks"], bbox)
            transformed = transformer.transform_landmarks(result["pose_world_landmarks"])
            record = make_frame_record(
                session_id=session_id,
                frame_index=frame.frame_index,
                timestamp_ms=frame.timestamp_ms,
                pose_landmarks=pose_landmarks,
                pose_world_landmarks=result["pose_world_landmarks"],
                transformed_landmarks=transformed,
            )
            segment = frame_to_segment[frame.frame_index]
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
                    }
                )
                if row["source"] == "pose":
                    restored = pose_landmarks[int(row["landmark_id"])]
                    row["crop_x_norm"] = restored.get("crop_x_norm")
                    row["crop_y_norm"] = restored.get("crop_y_norm")
                    row["crop_edge_risk"] = restored.get("crop_edge_risk")
                else:
                    row["crop_x_norm"] = np.nan
                    row["crop_y_norm"] = np.nan
                    row["crop_edge_risk"] = False
                rows.append(row)
    return rows


def _restore_pose_landmarks(landmarks: list[dict], bbox: CropBBox) -> list[dict]:
    restored = []
    for landmark in landmarks:
        item = dict(landmark)
        x_crop = float(landmark["x"])
        y_crop = float(landmark["y"])
        x_original, y_original = crop_to_original_norm(x_crop, y_crop, bbox)
        item["x"] = x_original
        item["y"] = y_original
        item["crop_x_norm"] = x_crop
        item["crop_y_norm"] = y_crop
        item["crop_edge_risk"] = is_near_crop_edge(x_crop, y_crop)
        restored.append(item)
    return restored


def _apply_crop_candidates(
    cleaned: pd.DataFrame,
    candidates: pd.DataFrame,
    segments: list[CropSegment],
    options: CropRefinementOptions,
) -> tuple[pd.DataFrame, list[dict], list[dict]]:
    refined = cleaned.copy()
    _initialize_crop_columns(refined)

    landmark_name_to_id = {name: index for index, name in enumerate(POSE_LANDMARK_NAMES)}
    candidate_lookup = _candidate_lookup(candidates)
    candidate_frame_cache = _frame_source_cache(candidates)
    cleaned_frame_cache = _frame_source_cache(cleaned)
    temporal_refs = _temporal_references(cleaned)
    score_rows: list[dict] = []
    segment_summaries: list[dict] = []

    for segment in segments:
        target_ids = {landmark_name_to_id[name] for name in segment.target_landmarks if name in landmark_name_to_id}
        if not segment.selected_for_crop:
            summary = segment.to_dict()
            summary.update(
                {
                    "accepted_rows": 0,
                    "rejected_rows": 0,
                    "unavailable_rows": 0,
                }
            )
            segment_summaries.append(summary)
            continue
        segment_mask = (
            refined["frame"].between(segment.start_frame, segment.end_frame)
            & refined["landmark_id"].isin(target_ids)
            & refined["source"].isin(["pose", "pose_world"])
            & refined["quality_flag"].isin(segment.problem_flags)
        )
        segment_indices = refined[segment_mask].index.tolist()
        counts = {"accepted": 0, "rejected": 0, "unavailable": 0}

        for index in segment_indices:
            row = refined.loc[index]
            source = str(row["source"])
            frame = int(row["frame"])
            landmark_id = int(row["landmark_id"])
            candidate_row = candidate_lookup.get((frame, source, landmark_id))

            before_score = _fast_cleaned_score(
                row,
                temporal_refs.get((source, landmark_id)),
                source=source,
            )

            if candidate_row is None:
                _mark_crop_row(refined, index, "crop_unavailable", "none", before_score, before_score, segment, "rejected_missing_candidate")
                counts["unavailable"] += 1
                score_rows.append(_score_row(row, segment, "crop_unavailable", "rejected_missing_candidate", before_score, before_score))
                continue

            after_score = _fast_crop_score(
                row,
                candidate_row,
                temporal_refs.get((source, landmark_id)),
                source=source,
            )
            decision = decide_crop_candidate(
                before_score,
                after_score,
                accept_score_margin=options.accept_score_margin,
                candidate_row=candidate_row,
                review_only=segment.review_only,
            )
            if decision.accepted:
                _accept_crop_candidate(refined, index, candidate_row, decision, segment)
                counts["accepted"] += 1
                status = "crop_accepted"
            else:
                _mark_crop_row(
                    refined,
                    index,
                    "crop_rejected",
                    "cleaned",
                    decision.score_before,
                    decision.score_after,
                    segment,
                    decision.reason,
                )
                counts["rejected"] += 1
                status = "crop_rejected"
            score_rows.append(_score_row(row, segment, status, decision.reason, decision.score_before, decision.score_after))

        summary = segment.to_dict()
        summary.update(
            {
                "accepted_rows": counts["accepted"],
                "rejected_rows": counts["rejected"],
                "unavailable_rows": counts["unavailable"],
            }
        )
        segment_summaries.append(summary)

    return refined, score_rows, segment_summaries


def _initialize_crop_columns(refined: pd.DataFrame) -> None:
    refined["crop_refine_status"] = "unchanged"
    refined["crop_refine_source"] = "cleaned"
    refined["crop_score_before"] = np.nan
    refined["crop_score_after"] = np.nan
    refined["crop_score_delta"] = np.nan
    refined["crop_segment_id"] = np.nan
    refined["crop_reason"] = "unchanged_not_target"
    refined["crop_x0"] = np.nan
    refined["crop_y0"] = np.nan
    refined["crop_w"] = np.nan
    refined["crop_h"] = np.nan
    refined["crop_margin_ratio"] = np.nan
    refined["crop_running_mode"] = ""


def _accept_crop_candidate(
    refined: pd.DataFrame,
    index: int,
    candidate_row,
    decision,
    segment: CropSegment,
) -> None:
    for field in COORD_FIELDS:
        if hasattr(candidate_row, field):
            refined.at[index, field] = getattr(candidate_row, field)
    refined.at[index, "is_valid"] = True
    refined.at[index, "is_interpolated"] = False
    refined.at[index, "quality_flag"] = "crop_refined_measured"
    refined.at[index, "invalid_reason"] = ""
    refined.at[index, "interpolation_method"] = ""
    refined.at[index, "gap_length"] = np.nan
    refined.at[index, "source_frame_prev"] = np.nan
    refined.at[index, "source_frame_next"] = np.nan
    _mark_crop_row(
        refined,
        index,
        "crop_accepted",
        "crop_video",
        decision.score_before,
        decision.score_after,
        segment,
        decision.reason,
        candidate_row,
    )


def _mark_crop_row(
    refined: pd.DataFrame,
    index: int,
    status: str,
    source: str,
    before: float,
    after: float,
    segment: CropSegment,
    reason: str,
    candidate_row=None,
) -> None:
    refined.at[index, "crop_refine_status"] = status
    refined.at[index, "crop_refine_source"] = source
    refined.at[index, "crop_score_before"] = before
    refined.at[index, "crop_score_after"] = after
    refined.at[index, "crop_score_delta"] = after - before
    refined.at[index, "crop_segment_id"] = segment.crop_segment_id
    refined.at[index, "crop_reason"] = reason
    if candidate_row is not None:
        for target, attr in (
            ("crop_x0", "crop_x0"),
            ("crop_y0", "crop_y0"),
            ("crop_w", "crop_w"),
            ("crop_h", "crop_h"),
            ("crop_margin_ratio", "crop_margin_ratio"),
            ("crop_running_mode", "crop_running_mode"),
        ):
            if hasattr(candidate_row, attr):
                refined.at[index, target] = getattr(candidate_row, attr)


def _score_row(row: pd.Series, segment: CropSegment, status: str, reason: str, before: float, after: float) -> dict:
    return {
        "crop_segment_id": segment.crop_segment_id,
        "frame": int(row["frame"]),
        "source": row["source"],
        "landmark_id": int(row["landmark_id"]),
        "landmark_name": row["landmark_name"],
        "quality_flag_before": row["quality_flag"],
        "crop_refine_status": status,
        "crop_reason": reason,
        "score_before": before,
        "score_after": after,
        "score_delta": after - before,
    }


def _candidate_lookup(candidates: pd.DataFrame) -> dict[tuple[int, str, int], object]:
    if candidates.empty:
        return {}
    return {
        (int(row.frame), str(row.source), int(row.landmark_id)): row
        for row in candidates.itertuples(index=False)
    }


def _frame_source_cache(frame_rows: pd.DataFrame) -> dict[tuple[int, str], pd.DataFrame]:
    if frame_rows.empty:
        return {}
    return {
        (int(frame), str(source)): group.copy()
        for (frame, source), group in frame_rows.groupby(["frame", "source"], sort=False)
    }


def _temporal_references(cleaned: pd.DataFrame) -> dict[tuple[str, int], dict]:
    refs: dict[tuple[str, int], dict] = {}
    stable_flags = {"measured", "low_visibility_leg_kept", "refined_measured", "crop_refined_measured"}
    for (source, landmark_id), group in cleaned[cleaned["quality_flag"].isin(stable_flags)].groupby(
        ["source", "landmark_id"], sort=False
    ):
        fields = _coord_fields(str(source))
        usable = group.dropna(subset=fields).sort_values("frame")
        if usable.empty:
            continue
        frames = usable["frame"].to_numpy(dtype=np.int64)
        coords = usable[fields].to_numpy(dtype=float)
        distances = np.linalg.norm(np.diff(coords, axis=0), axis=1) if len(coords) > 1 else np.array([])
        median_motion = float(np.median(distances)) if len(distances) else 1.0
        refs[(str(source), int(landmark_id))] = {
            "frames": frames,
            "coords": coords,
            "median_motion": median_motion if median_motion > 0 else 1.0,
        }
    return refs


def _fast_cleaned_score(row, temporal_ref: dict | None, source: str) -> float:
    return float(
        confidence_score(row) * 0.30
        + _fast_temporal_score(row, temporal_ref, source) * 0.30
        + 1.0 * 0.15
        + 0.5 * 0.15
        + 0.5 * 0.10
    )


def _fast_crop_score(cleaned_row, candidate_row, temporal_ref: dict | None, source: str) -> float:
    return float(
        confidence_score(candidate_row) * 0.30
        + _fast_temporal_score(candidate_row, temporal_ref, source) * 0.30
        + crop_valid_score(candidate_row) * 0.15
        + 0.5 * 0.15
        + visibility_gain_score(cleaned_row, candidate_row) * 0.10
    )


def _fast_temporal_score(row, temporal_ref: dict | None, source: str) -> float:
    coords = _row_coords(row, source)
    if temporal_ref is None or coords is None:
        return 0.5
    frames = temporal_ref["frames"]
    stable_coords = temporal_ref["coords"]
    if len(frames) == 0:
        return 0.5
    frame = int(_row_get(row, "frame"))
    pos = int(np.searchsorted(frames, frame))
    distances = []
    if pos > 0:
        distances.append(float(np.linalg.norm(coords - stable_coords[pos - 1])))
    if pos < len(frames):
        distances.append(float(np.linalg.norm(coords - stable_coords[pos])))
    if not distances:
        return 0.5
    jump = float(sum(distances) / len(distances))
    ratio = jump / (float(temporal_ref["median_motion"]) + 1e-6)
    return float(min(1.0, max(0.0, 1.0 / (1.0 + ratio))))


def _row_coords(row, source: str) -> np.ndarray | None:
    values = []
    for field in _coord_fields(source):
        value = _row_get(row, field)
        if value is None or pd.isna(value):
            return None
        values.append(float(value))
    return np.array(values, dtype=float)


def _coord_fields(source: str) -> list[str]:
    if source == "pose_world":
        return ["tx", "ty", "tz"]
    return ["x", "y"]


def _row_get(row, key: str):
    if isinstance(row, dict):
        return row.get(key)
    return getattr(row, key, row.get(key) if hasattr(row, "get") else None)


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
    candidate = options.input_cleaned_csv.parent / "frame_status.csv"
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
