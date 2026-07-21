import numpy as np
import pandas as pd

from dance_pose_recorder.motion_profile import (
    build_motion_profile,
    build_session_motion_samples,
    summarize,
)


def _world_rows(landmark_id, landmark_name, positions, quality_flag="measured", frame_step=1):
    rows = []
    for index, (tx, ty, tz) in enumerate(positions):
        rows.append(
            {
                "frame": index * frame_step,
                "landmark_id": landmark_id,
                "landmark_name": landmark_name,
                "source": "pose_world",
                "tx": tx,
                "ty": ty,
                "tz": tz,
                "quality_flag": quality_flag,
            }
        )
    return rows


def _linear_motion(n, speed_m_per_s, fps):
    step = speed_m_per_s / fps
    return [(i * step, 0.0, 0.0) for i in range(n)]


def test_velocity_in_physical_units():
    fps = 30.0
    pose = pd.DataFrame(_world_rows(15, "left_wrist", _linear_motion(10, 2.0, fps)))
    samples, _ = build_session_motion_samples(pose, fps)
    velocity = samples["left_wrist"]["velocity_m_per_s"]
    assert velocity.size == 9
    assert np.allclose(velocity, 2.0)


def test_fps_invariance_of_physical_stats():
    # The same 2 m/s motion sampled at 30fps and 60fps must produce the same m/s stats.
    stats = {}
    for fps in (30.0, 60.0):
        pose = pd.DataFrame(_world_rows(15, "left_wrist", _linear_motion(int(fps) + 1, 2.0, fps)))
        samples, _ = build_session_motion_samples(pose, fps)
        stats[fps] = summarize(samples["left_wrist"]["velocity_m_per_s"])
    assert stats[30.0]["median"] == stats[60.0]["median"] == 2.0
    assert stats[30.0]["p99"] == stats[60.0]["p99"]


def test_frame_gap_resets_chains():
    fps = 30.0
    positions = _linear_motion(6, 1.0, fps)
    pose = pd.DataFrame(_world_rows(15, "left_wrist", positions, frame_step=2))
    samples, _ = build_session_motion_samples(pose, fps)
    # frames are 0,2,4,... — no consecutive frames, so no velocity samples at all
    assert samples["left_wrist"]["velocity_m_per_s"].size == 0


def test_unstable_flags_excluded():
    fps = 30.0
    rows = _world_rows(15, "left_wrist", _linear_motion(5, 1.0, fps))
    rows += _world_rows(16, "right_wrist", _linear_motion(5, 1.0, fps), quality_flag="unreliable")
    samples, _ = build_session_motion_samples(pd.DataFrame(rows), fps)
    assert samples["left_wrist"]["velocity_m_per_s"].size == 4
    assert "right_wrist" not in samples


def test_bone_lengths_measured_per_frame():
    fps = 30.0
    rows = _world_rows(11, "left_shoulder", [(0.0, 0.0, 0.0)] * 4)
    rows += _world_rows(13, "left_elbow", [(0.3, 0.0, 0.0)] * 4)
    _, bones = build_session_motion_samples(pd.DataFrame(rows), fps)
    upper_arm = bones["left_upper_arm"]
    assert upper_arm.size == 4
    assert np.allclose(upper_arm, 0.3)


def test_profile_pools_sessions_and_keeps_note():
    fps_a, fps_b = 30.0, 60.0
    pose_a = pd.DataFrame(_world_rows(15, "left_wrist", _linear_motion(31, 2.0, fps_a)))
    pose_b = pd.DataFrame(_world_rows(15, "left_wrist", _linear_motion(61, 2.0, fps_b)))
    profile = build_motion_profile([("a", pose_a, fps_a), ("b", pose_b, fps_b)])
    pooled = profile["landmarks"]["left_wrist"]["velocity_m_per_s"]
    assert pooled["count"] == 30 + 60
    assert pooled["median"] == 2.0
    assert profile["sessions"] == {"a": 30.0, "b": 60.0}
    assert "read-only" in profile["note"].lower() or "Read-only" in profile["note"]
