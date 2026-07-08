"""OpenCV video input helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2


@dataclass(frozen=True)
class VideoInfo:
    source_path: str
    fps: float
    width: int
    height: int
    frame_count: int


@dataclass(frozen=True)
class VideoFrame:
    frame_index: int
    timestamp_ms: int
    image_bgr: object


class VideoFileReader:
    def __init__(self, input_path: str | Path) -> None:
        self.input_path = Path(input_path)
        if not self.input_path.exists():
            raise FileNotFoundError(f"Input video not found: {self.input_path}")

        self.capture = cv2.VideoCapture(str(self.input_path))
        if not self.capture.isOpened():
            raise RuntimeError(f"Could not open input video: {self.input_path}")

        fps = float(self.capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 0:
            fps = 30.0

        self.info = VideoInfo(
            source_path=str(self.input_path),
            fps=fps,
            width=int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
            height=int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
            frame_count=int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
        )

    def frames(self, max_frames: int | None = None) -> Iterator[VideoFrame]:
        frame_index = 0
        while True:
            if max_frames is not None and frame_index >= max_frames:
                break
            ok, frame = self.capture.read()
            if not ok:
                break
            timestamp_ms = int(round(frame_index * 1000.0 / self.info.fps))
            yield VideoFrame(frame_index=frame_index, timestamp_ms=timestamp_ms, image_bgr=frame)
            frame_index += 1

    def close(self) -> None:
        self.capture.release()

    def __enter__(self) -> "VideoFileReader":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
