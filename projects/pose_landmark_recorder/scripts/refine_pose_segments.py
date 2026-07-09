#!/usr/bin/env python
"""Refine cleaned pose data by re-detecting problematic frame segments."""

from __future__ import annotations

from dataclasses import dataclass
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dance_pose_recorder.cleaned_preview_renderer import render_corrected_preview
from dance_pose_recorder.coordinate_transform import CoordinateTransformer
from dance_pose_recorder.data_writer import frame_to_csv_rows, make_frame_record
from dance_pose_recorder.landmark_schema import POSE_LANDMARK_NAMES
from dance_pose_recorder.pose_candidate_scorer import (
    decide_candidate,
    median_bone_lengths,
    score_row,
    stable_landmark_series,
)
from dance_pose_recorder.pose_extractor import PoseExtractor
from dance_pose_recorder.refine_report import build_refine_report, write_json
from dance_pose_recorder.segment_refiner import (
    CandidateSegment,
    detect_candidate_segments,
    parse_target_landmarks,
    segments_to_dataframe,
)
from dance_pose_recorder.video_input import VideoFileReader


REFINE_COLUMNS = [
    "refine_status",
    "refine_source",
    "refine_score_before",
    "refine_score_after",
    "refine_score_delta",
    "refine_segment_id",
    "refine_reason",
]

SCORE_COLUMNS = [
    "segment_id",
    "frame",
    "source",
    "landmark_id",
    "landmark_name",
    "quality_flag_before",
    "refine_status",
    "refine_reason",
    "score_before",
    "score_after",
    "score_delta",
]

COORD_FIELDS = ["x", "y", "z", "visibility", "presence", "tx", "ty", "tz"]


@dataclass(frozen=True)
class RefinementOptions:
    input_video: Path
    input_cleaned_csv: Path
    metadata: Path
    output: Path
    input_raw_csv: Path | None = None
    input_jsonl: Path | None = None
    frame_status: Path | None = None
    quality_report: Path | None = None
    interpolation_report: Path | None = None
    delegate: str = "cpu"
    model: Path = Path("models/pose_landmarker.task")
    target_landmarks: str = "arms,hands,feet"
    min_cluster_length: int = 2
    max_cluster_length: int = 90
    segment_margin: int = 12
    accept_score_margin: float = 0.08
    save_csv: bool = True
    save_jsonl: bool = True
    save_preview: bool = False
    max_segments: int | None = None


@dataclass(frozen=True)
class RefinementResult:
    refined_csv: Path | None
    refined_jsonl: Path | None
    refine_report: Path
    refined_preview: Path | None
    candidate_segments_csv: Path
    candidate_scores_csv: Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Re-detect problematic pose segments and refine cleaned data.")
    parser.add_argument("--input-video", required=True, type=Path)
    parser.add_argument("--input-cleaned-csv", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--input-raw-csv", type=Path, default=None)
    parser.add_argument("--input-jsonl", type=Path, default=None)
    parser.add_argument("--frame-status", type=Path, default=None)
    parser.add_argument("--quality-report", type=Path, default=None)
    parser.add_argument("--interpolation-report", type=Path, default=None)
    parser.add_argument("--delegate", default="cpu", choices=["cpu", "gpu"])
    parser.add_argument("--model", type=Path, default=Path("models/pose_landmarker.task"))
    parser.add_argument("--target-landmarks", default="arms,hands,feet")
    parser.add_argument("--min-cluster-length", type=int, default=2)
    parser.add_argument("--max-cluster-length", type=int, default=90)
    parser.add_argument("--segment-margin", type=int, default=12)
    parser.add_argument("--accept-score-margin", type=float, default=0.08)
    parser.add_argument("--save-csv", action="store_true")
    parser.add_argument("--save-jsonl", action="store_true")
    parser.add_argument("--save-preview", action="store_true")
    parser.add_argument("--max-segments", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = refine_pose_segments(
        RefinementOptions(
            input_video=args.input_video,
            input_cleaned_csv=args.input_cleaned_csv,
            input_raw_csv=args.input_raw_csv,
            input_jsonl=args.input_jsonl,
            metadata=args.metadata,
            frame_status=args.frame_status,
            quality_report=args.quality_report,
            interpolation_report=args.interpolation_report,
            output=args.output,
            delegate=args.delegate,
            model=args.model,
            target_landmarks=args.target_landmarks,
            min_cluster_length=args.min_cluster_length,
            max_cluster_length=args.max_cluster_length,
            segment_margin=args.segment_margin,
            accept_score_margin=args.accept_score_margin,
            save_csv=args.save_csv or not args.save_jsonl,
            save_jsonl=args.save_jsonl or not args.save_csv,
            save_preview=args.save_preview,
            max_segments=args.max_segments,
        )
    )
    print(f"Wrote refined outputs to {args.output}")
    for path in (
        result.refined_csv,
        result.refined_jsonl,
        result.refine_report,
        result.refined_preview,
        result.candidate_segments_csv,
        result.candidate_scores_csv,
    ):
        if path:
            print(path)


def refine_pose_segments(options: RefinementOptions) -> RefinementResult:
    metadata = json.loads(options.metadata.read_text(encoding="utf-8"))
    cleaned = pd.read_csv(options.input_cleaned_csv, low_memory=False)
    output_dir = options.output
    output_dir.mkdir(parents=True, exist_ok=True)

    total_frames = int(cleaned["frame"].max()) + 1
    segments = detect_candidate_segments(
        cleaned,
        total_frames=total_frames,
        target_landmarks=options.target_landmarks,
        min_cluster_length=options.min_cluster_length,
        max_cluster_length=options.max_cluster_length,
        segment_margin=options.segment_margin,
    )
    if options.max_segments is not None:
        segments = segments[: options.max_segments]

    candidate_segments_csv = output_dir / "candidate_segments.csv"
    segments_to_dataframe(segments).to_csv(candidate_segments_csv, index=False)

    candidates = _redetect_segments(options, metadata, segments)
    refined, score_rows, segment_summaries = _apply_candidates(cleaned, candidates, segments, options)

    candidate_scores_csv = output_dir / "candidate_scores.csv"
    pd.DataFrame(score_rows, columns=SCORE_COLUMNS).to_csv(candidate_scores_csv, index=False)

    refined_csv = None
    if options.save_csv:
        refined_csv = output_dir / "refined_pose.csv"
        refined.to_csv(refined_csv, index=False)

    refined_jsonl = None
    if options.save_jsonl:
        refined_jsonl = output_dir / "refined_pose.jsonl"
        write_refined_jsonl(refined, refined_jsonl, metadata)

    report_path = output_dir / "refine_report.json"
    report = build_refine_report(
        metadata=metadata,
        input_cleaned_csv=options.input_cleaned_csv,
        input_video=options.input_video,
        frames_total=total_frames,
        target_landmarks=sorted(parse_target_landmarks(options.target_landmarks)),
        settings={
            "delegate": options.delegate,
            "min_cluster_length": options.min_cluster_length,
            "max_cluster_length": options.max_cluster_length,
            "segment_margin": options.segment_margin,
            "accept_score_margin": options.accept_score_margin,
        },
        segments=segment_summaries,
        refined=refined,
    )
    refine_report = write_json(report_path, report)

    refined_preview = None
    if options.save_preview:
        if options.frame_status and options.frame_status.exists():
            frame_status = pd.read_csv(options.frame_status)
        else:
            frame_status = _build_minimal_frame_status(refined, total_frames, float(metadata.get("fps") or 30.0))
        refined_preview = output_dir / "refined_preview.mp4"
        render_corrected_preview(options.input_video, refined_preview, refined, frame_status, metadata)

    return RefinementResult(
        refined_csv=refined_csv,
        refined_jsonl=refined_jsonl,
        refine_report=refine_report,
        refined_preview=refined_preview,
        candidate_segments_csv=candidate_segments_csv,
        candidate_scores_csv=candidate_scores_csv,
    )


def _redetect_segments(options: RefinementOptions, metadata: dict, segments: list[CandidateSegment]) -> pd.DataFrame:
    if not segments:
        return pd.DataFrame()

    transformer = CoordinateTransformer(origin_policy=str(metadata.get("origin_policy") or "raw"))
    _warm_transformer(transformer, options.input_raw_csv)
    rows = []
    session_id = str(metadata.get("session_id") or options.output.name)

    with VideoFileReader(options.input_video) as reader, PoseExtractor(options.model, delegate=options.delegate) as extractor:
        segment_index = 0
        active = segments[segment_index]
        total = int(metadata.get("frame_count_written") or reader.info.frame_count)
        for frame in tqdm(reader.frames(max_frames=total), total=total, unit="frame"):
            while active and frame.frame_index > active.end_frame:
                segment_index += 1
                active = segments[segment_index] if segment_index < len(segments) else None
            if active is None:
                break
            if frame.frame_index < active.start_frame:
                continue

            result = extractor.detect(frame.image_bgr, frame.timestamp_ms)
            transformed = transformer.transform_landmarks(result["pose_world_landmarks"])
            record = make_frame_record(
                session_id=session_id,
                frame_index=frame.frame_index,
                timestamp_ms=frame.timestamp_ms,
                pose_landmarks=result["pose_landmarks"],
                pose_world_landmarks=result["pose_world_landmarks"],
                transformed_landmarks=transformed,
            )
            rows.extend(frame_to_csv_rows(record))

    return pd.DataFrame(rows)


def _apply_candidates(
    cleaned: pd.DataFrame,
    candidates: pd.DataFrame,
    segments: list[CandidateSegment],
    options: RefinementOptions,
) -> tuple[pd.DataFrame, list[dict], list[dict]]:
    refined = cleaned.copy()
    refined["refine_status"] = "unchanged"
    refined["refine_source"] = "cleaned"
    refined["refine_score_before"] = np.nan
    refined["refine_score_after"] = np.nan
    refined["refine_score_delta"] = np.nan
    refined["refine_segment_id"] = np.nan
    refined["refine_reason"] = "unchanged_not_target"

    landmark_name_to_id = {
        name: index for index, name in enumerate(POSE_LANDMARK_NAMES)
    }
    candidate_lookup = _candidate_lookup(candidates)
    candidate_frame_cache = _frame_source_cache(candidates)
    cleaned_frame_cache = _frame_source_cache(cleaned)
    median_bones = {
        "pose": median_bone_lengths(cleaned, "pose"),
        "pose_world": median_bone_lengths(cleaned, "pose_world"),
    }
    stable_cache: dict[tuple[str, int], pd.DataFrame] = {}
    median_motion_cache: dict[tuple[str, int], float] = {}
    score_rows: list[dict] = []
    segment_summaries: list[dict] = []

    for segment in segments:
        target_ids = {landmark_name_to_id[name] for name in segment.target_landmarks if name in landmark_name_to_id}
        segment_mask = (
            refined["frame"].between(segment.start_frame, segment.end_frame)
            & refined["landmark_id"].isin(target_ids)
            & refined["source"].isin(["pose", "pose_world"])
            & refined["quality_flag"].isin(segment.problem_flags)
        )
        segment_indices = refined[segment_mask].index.tolist()
        counts = {"accepted": 0, "rejected": 0, "unavailable": 0}
        redetected = False

        for index in segment_indices:
            row = refined.loc[index]
            source = str(row["source"])
            frame = int(row["frame"])
            landmark_id = int(row["landmark_id"])
            key = (frame, source, landmark_id)
            candidate_row = candidate_lookup.get(key)
            if candidate_row is not None:
                redetected = True

            stable_key = (source, landmark_id)
            if stable_key not in stable_cache:
                stable_cache[stable_key] = stable_landmark_series(cleaned, source, landmark_id)
                median_motion_cache[stable_key] = _median_motion_for_series(stable_cache[stable_key], source)

            before_frame_rows = cleaned_frame_cache.get((frame, source), pd.DataFrame())
            before_score = score_row(
                row,
                stable_cache[stable_key],
                before_frame_rows,
                median_bones.get(source, {}),
                source=source,
                median_motion=median_motion_cache[stable_key],
            ).total

            if candidate_row is None:
                _mark_refine_row(refined, index, "refined_unavailable", "none", before_score, before_score, segment, "rejected_missing_candidate")
                counts["unavailable"] += 1
                score_rows.append(_score_row(row, segment, "refined_unavailable", "rejected_missing_candidate", before_score, before_score))
                continue

            candidate_frame_rows = candidate_frame_cache.get((frame, source), pd.DataFrame())
            after_score = score_row(
                candidate_row,
                stable_cache[stable_key],
                candidate_frame_rows,
                median_bones.get(source, {}),
                source=source,
                median_motion=median_motion_cache[stable_key],
            ).total
            decision = decide_candidate(
                cleaned_score=before_score,
                candidate_score=after_score,
                accept_score_margin=options.accept_score_margin,
                candidate_row=candidate_row,
                review_only=segment.review_only,
            )
            if decision.accepted:
                _accept_candidate(refined, index, candidate_row, decision, segment)
                counts["accepted"] += 1
                status = "refined_accepted"
            else:
                _mark_refine_row(
                    refined,
                    index,
                    "refined_rejected",
                    "redetect_full_frame",
                    decision.score_before,
                    decision.score_after,
                    segment,
                    decision.reason,
                )
                counts["rejected"] += 1
                status = "refined_rejected"
            score_rows.append(
                _score_row(
                    row,
                    segment,
                    status,
                    decision.reason,
                    decision.score_before,
                    decision.score_after,
                )
            )

        summary = segment.to_dict()
        summary.update(
            {
                "accepted_rows": counts["accepted"],
                "rejected_rows": counts["rejected"],
                "unavailable_rows": counts["unavailable"],
                "redetected": redetected,
            }
        )
        segment_summaries.append(summary)

    return refined, score_rows, segment_summaries


def _candidate_lookup(candidates: pd.DataFrame) -> dict[tuple[int, str, int], pd.Series]:
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


def _median_motion_for_series(stable_series: pd.DataFrame, source: str) -> float:
    if stable_series.empty:
        return 1.0
    fields = ["tx", "ty", "tz"] if source == "pose_world" else ["x", "y"]
    usable = stable_series.dropna(subset=fields).sort_values("frame")
    if len(usable) < 2:
        return 1.0
    coords = usable[fields].to_numpy(dtype=float)
    distances = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    if len(distances) == 0:
        return 1.0
    median = float(np.median(distances))
    return median if median > 0 else 1.0


def _accept_candidate(
    refined: pd.DataFrame,
    index: int,
    candidate_row: pd.Series,
    decision,
    segment: CandidateSegment,
) -> None:
    for field in COORD_FIELDS:
        if hasattr(candidate_row, field):
            refined.at[index, field] = getattr(candidate_row, field)
    refined.at[index, "is_valid"] = True
    refined.at[index, "is_interpolated"] = False
    refined.at[index, "quality_flag"] = "refined_measured"
    refined.at[index, "invalid_reason"] = ""
    refined.at[index, "interpolation_method"] = ""
    refined.at[index, "gap_length"] = np.nan
    refined.at[index, "source_frame_prev"] = np.nan
    refined.at[index, "source_frame_next"] = np.nan
    _mark_refine_row(
        refined,
        index,
        "refined_accepted",
        "redetect_full_frame",
        decision.score_before,
        decision.score_after,
        segment,
        decision.reason,
    )


def _mark_refine_row(
    refined: pd.DataFrame,
    index: int,
    status: str,
    source: str,
    before: float,
    after: float,
    segment: CandidateSegment,
    reason: str,
) -> None:
    refined.at[index, "refine_status"] = status
    refined.at[index, "refine_source"] = source
    refined.at[index, "refine_score_before"] = before
    refined.at[index, "refine_score_after"] = after
    refined.at[index, "refine_score_delta"] = after - before
    refined.at[index, "refine_segment_id"] = segment.segment_id
    refined.at[index, "refine_reason"] = reason


def _score_row(row: pd.Series, segment: CandidateSegment, status: str, reason: str, before: float, after: float) -> dict:
    return {
        "segment_id": segment.segment_id,
        "frame": int(row["frame"]),
        "source": row["source"],
        "landmark_id": int(row["landmark_id"]),
        "landmark_name": row["landmark_name"],
        "quality_flag_before": row["quality_flag"],
        "refine_status": status,
        "refine_reason": reason,
        "score_before": before,
        "score_after": after,
        "score_delta": after - before,
    }


def _warm_transformer(transformer: CoordinateTransformer, input_raw_csv: Path | None) -> None:
    if transformer.origin_policy != "first_frame_pelvis" or input_raw_csv is None or not input_raw_csv.exists():
        return
    raw = pd.read_csv(input_raw_csv)
    first_frame = raw[(raw["source"] == "pose_world") & (raw["frame"] == raw["frame"].min())]
    if first_frame.empty:
        return
    landmarks = [
        {
            "id": int(row.landmark_id),
            "name": row.landmark_name,
            "x": float(row.x),
            "y": float(row.y),
            "z": float(row.z),
        }
        for row in first_frame.itertuples(index=False)
    ]
    transformer.transform_landmarks(landmarks)


def _build_minimal_frame_status(refined: pd.DataFrame, total_frames: int, fps: float) -> pd.DataFrame:
    rows = []
    for frame in range(total_frames):
        frame_df = refined[refined["frame"] == frame]
        pose = frame_df[frame_df["source"] == "pose"]
        pose_world = frame_df[frame_df["source"] == "pose_world"]
        long_missing = bool((pose["quality_flag"] == "missing_long_gap").all()) if not pose.empty else True
        rows.append(
            {
                "frame": frame,
                "time_sec": frame / fps,
                "has_pose": bool((pose["quality_flag"] != "missing_long_gap").any()),
                "has_pose_world": bool((pose_world["quality_flag"] != "missing_long_gap").any()),
                "pose_landmark_count": int(len(pose)),
                "pose_world_landmark_count": int(len(pose_world)),
                "empty_frame": pose.empty,
                "is_inside_long_missing_range": long_missing,
                "interpolated_landmark_count": int(frame_df["is_interpolated"].sum()),
                "invalid_landmark_count": int((~frame_df["is_valid"]).sum()),
            }
        )
    return pd.DataFrame(rows)


def write_refined_jsonl(refined: pd.DataFrame, output_path: str | Path, metadata: dict) -> None:
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
            "refine_status": row.refine_status,
        }
        for field in COORD_FIELDS:
            value = getattr(row, field)
            if pd.notna(value):
                item[field] = float(value)
        landmarks.append(item)
    return landmarks


if __name__ == "__main__":
    main()
