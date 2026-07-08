"""OpenCV overlay preview writer."""

from __future__ import annotations

from pathlib import Path

import cv2

from dance_pose_recorder.landmark_schema import POSE_CONNECTIONS


class PreviewRenderer:
    def __init__(self, output_path: str | Path, fps: float, width: int, height: int) -> None:
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(str(self.output_path), fourcc, fps, (width, height))
        if not self._writer.isOpened():
            raise RuntimeError(f"Could not create preview video: {self.output_path}")
        self.width = width
        self.height = height

    def write(self, image_bgr: object, pose_landmarks: list[dict]) -> None:
        frame = image_bgr.copy()
        points = {}
        for landmark in pose_landmarks:
            x = int(round(float(landmark["x"]) * self.width))
            y = int(round(float(landmark["y"]) * self.height))
            points[landmark["id"]] = (x, y)

        for start, end in POSE_CONNECTIONS:
            if start in points and end in points:
                cv2.line(frame, points[start], points[end], (0, 210, 255), 2)
        for point in points.values():
            cv2.circle(frame, point, 3, (40, 255, 120), -1)
        if not points:
            cv2.putText(frame, "NO POSE", (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        self._writer.write(frame)

    def close(self) -> None:
        self._writer.release()

    def __enter__(self) -> "PreviewRenderer":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
