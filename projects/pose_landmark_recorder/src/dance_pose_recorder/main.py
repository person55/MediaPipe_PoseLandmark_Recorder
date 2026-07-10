"""Command implementation for recording pose data from video files."""

from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from dance_pose_recorder.coordinate_transform import CoordinateTransformer
from dance_pose_recorder.data_writer import SessionWriters, make_frame_record
from dance_pose_recorder.metadata import build_metadata, write_metadata
from dance_pose_recorder.output_layout import RAW_DIR, RAW_PREVIEW_MP4, normalize_stage_output_dir
from dance_pose_recorder.video_input import VideoFileReader


def record_from_video(args: argparse.Namespace) -> dict:
    from dance_pose_recorder.pose_extractor import PoseExtractor
    from dance_pose_recorder.preview_renderer import PreviewRenderer

    input_path = Path(args.input)
    session_output_dir = Path(args.output)
    output_dir = normalize_stage_output_dir(session_output_dir, RAW_DIR)
    model_path = Path(args.model)
    session_id_dir = session_output_dir.parent if session_output_dir.name == RAW_DIR else session_output_dir
    session_id = args.session_id or session_id_dir.name

    save_jsonl = args.save_jsonl or not args.save_csv
    save_csv = args.save_csv or not args.save_jsonl
    output_formats = []
    if save_jsonl:
        output_formats.append("jsonl")
    if save_csv:
        output_formats.append("csv")
    if args.save_preview:
        output_formats.append("preview_video")

    transformer = CoordinateTransformer(origin_policy=args.origin, scale=args.scale)
    frame_count_written = 0

    with VideoFileReader(input_path) as reader, PoseExtractor(model_path, delegate=args.delegate) as extractor, SessionWriters(
        output_dir=output_dir,
        session_id=session_id,
        save_jsonl=save_jsonl,
        save_csv=save_csv,
    ) as writers:
        preview = None
        if args.save_preview:
            preview = PreviewRenderer(output_dir / RAW_PREVIEW_MP4, reader.info.fps, reader.info.width, reader.info.height)

        try:
            progress_total = reader.info.frame_count if reader.info.frame_count > 0 else None
            if args.max_frames is not None:
                progress_total = min(progress_total or args.max_frames, args.max_frames)
            for frame in tqdm(reader.frames(max_frames=args.max_frames), total=progress_total, unit="frame"):
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
                writers.write_frame(record)
                if preview:
                    preview.write(frame.image_bgr, result["pose_landmarks"])
                frame_count_written += 1
        finally:
            if preview:
                preview.close()

        metadata = build_metadata(
            session_id=session_id,
            source_type="video_file",
            source_path=str(input_path),
            video_info=reader.info,
            model_path=str(model_path),
            delegate=args.delegate,
            origin_policy=args.origin,
            output_formats=output_formats,
            frame_count_written=frame_count_written,
        )

    write_metadata(output_dir, metadata)
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Record MediaPipe pose landmarks from a video file.")
    parser.add_argument("--input", required=True, help="Input video path.")
    parser.add_argument("--output", required=True, help="Output session directory.")
    parser.add_argument("--model", default="models/pose_landmarker.task", help="Pose Landmarker .task model path.")
    parser.add_argument("--delegate", default="cpu", choices=["cpu", "gpu"], help="MediaPipe inference delegate.")
    parser.add_argument("--session-id", default=None, help="Optional session id. Defaults to output folder name.")
    parser.add_argument(
        "--origin",
        default="raw",
        choices=["raw", "first_frame_pelvis", "per_frame_pelvis"],
        help="Origin policy for transformed landmarks.",
    )
    parser.add_argument("--scale", type=float, default=1.0, help="Scale factor for transformed landmarks.")
    parser.add_argument("--save-jsonl", action="store_true", help="Write raw_pose.jsonl.")
    parser.add_argument("--save-csv", action="store_true", help="Write raw_pose.csv.")
    parser.add_argument("--save-preview", action="store_true", help="Write raw_preview.mp4.")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional frame limit for smoke tests.")
    return parser


def main() -> None:
    metadata = record_from_video(build_parser().parse_args())
    print(f"Wrote {metadata['frame_count_written']} frames to session {metadata['session_id']}")


if __name__ == "__main__":
    main()
