"""Landmark presets for trajectory export."""

from __future__ import annotations


POSE_LANDMARK_NAMES = [
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
]

DEFAULT_EXCLUDED_FOR_BLENDER = {
    "left_ear",
    "right_ear",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
}

BODY_CORE_LANDMARKS = {
    "nose",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
}


def get_landmark_names(
    preset: str = "blender_default",
    include_landmarks: list[str] | None = None,
    exclude_landmarks: list[str] | None = None,
) -> list[str]:
    """Return ordered landmark names for an export preset."""

    preset = str(preset or "blender_default")
    if preset == "all":
        names = set(POSE_LANDMARK_NAMES)
    elif preset == "body_core":
        names = set(BODY_CORE_LANDMARKS)
    elif preset == "custom":
        names = set(include_landmarks or [])
    elif preset == "blender_default":
        names = set(POSE_LANDMARK_NAMES) - DEFAULT_EXCLUDED_FOR_BLENDER
    else:
        raise ValueError(f"Unsupported landmark preset: {preset}")

    if include_landmarks:
        names.update(include_landmarks)
    if exclude_landmarks:
        names.difference_update(exclude_landmarks)

    return [name for name in POSE_LANDMARK_NAMES if name in names]


def landmark_group_for_export(landmark_name: str) -> str:
    """Return a Blender/trajectory-oriented landmark group name."""

    if landmark_name == "nose":
        return "head_proxy"
    if landmark_name in {
        "left_eye_inner",
        "left_eye",
        "left_eye_outer",
        "right_eye_inner",
        "right_eye",
        "right_eye_outer",
        "left_ear",
        "right_ear",
        "mouth_left",
        "mouth_right",
    }:
        return "face_detail"
    if landmark_name in {"left_shoulder", "right_shoulder", "left_hip", "right_hip"}:
        return "torso"
    if landmark_name in {"left_elbow", "right_elbow", "left_wrist", "right_wrist"}:
        return "arms"
    if landmark_name in {"left_pinky", "right_pinky", "left_index", "right_index", "left_thumb", "right_thumb"}:
        return "hands_proxy"
    if landmark_name in {"left_knee", "right_knee", "left_ankle", "right_ankle"}:
        return "legs"
    if landmark_name in {"left_heel", "right_heel", "left_foot_index", "right_foot_index"}:
        return "feet"
    return "other"
