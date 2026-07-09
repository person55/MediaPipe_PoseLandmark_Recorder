"""Visualization policy for pose trajectory points and line connections."""

from __future__ import annotations

from dataclasses import dataclass


LANDMARK_GROUPS = {
    "torso": {
        "nose",
        "left_shoulder",
        "right_shoulder",
        "left_hip",
        "right_hip",
    },
    "arms": {
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
    },
    "legs": {
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    },
    "feet": {
        "left_heel",
        "right_heel",
        "left_foot_index",
        "right_foot_index",
    },
    "hands_proxy": {
        "left_thumb",
        "right_thumb",
        "left_index",
        "right_index",
        "left_pinky",
        "right_pinky",
    },
    "face_detail": {
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
    },
}


@dataclass(frozen=True)
class TrajectoryDisplayPolicy:
    visible: bool
    connect: bool
    alpha: float
    width: float
    reason: str


def landmark_group(landmark_name: str) -> str:
    for group_name, names in LANDMARK_GROUPS.items():
        if landmark_name in names:
            return group_name
    return "other"


def is_correctable_landmark(landmark_name: str) -> bool:
    return landmark_group(landmark_name) in {"arms", "legs", "feet"}


def default_trajectory_policy(quality_flag: str, landmark_name: str) -> TrajectoryDisplayPolicy:
    flag = str(quality_flag or "")
    group = landmark_group(str(landmark_name))

    if flag == "missing_long_gap":
        return TrajectoryDisplayPolicy(False, False, 0.0, 0.0, "hidden_missing_long_gap")
    if flag == "review_only":
        return TrajectoryDisplayPolicy(True, False, 0.15, 0.5, "protected_review_only")
    if flag == "unreliable":
        reason = "hands_proxy_unreliable" if group == "hands_proxy" else "faded_unreliable"
        return TrajectoryDisplayPolicy(True, False, 0.2, 0.6, reason)
    if flag == "estimated_occluded_arm":
        return TrajectoryDisplayPolicy(True, False, 0.35, 0.7, "faded_occluded_arm")
    if flag == "low_visibility_leg_kept":
        return TrajectoryDisplayPolicy(True, True, 0.5, 0.8, "stable_low_visibility_leg")
    if flag == "interpolated_outlier_removed":
        return TrajectoryDisplayPolicy(True, True, 0.7, 0.8, "stable_interpolated_outlier")
    if flag == "interpolated_short_gap":
        return TrajectoryDisplayPolicy(True, True, 0.85, 0.9, "stable_short_gap")
    if flag in {"crop_refined_measured", "refined_measured"}:
        return TrajectoryDisplayPolicy(True, True, 1.0, 1.1, "stable_refined")
    if group == "face_detail":
        return TrajectoryDisplayPolicy(True, False, 0.5, 0.5, "face_detail_no_connect")
    return TrajectoryDisplayPolicy(True, True, 1.0, 1.0, "stable")
