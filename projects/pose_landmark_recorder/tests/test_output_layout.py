from pathlib import Path

from dance_pose_recorder.output_layout import (
    CLEANED_DIR,
    blender_blend_filename,
    normalize_stage_output_dir,
)


def test_normalize_stage_output_dir_adds_stage_for_session_root():
    assert normalize_stage_output_dir(Path("session_gpu_005"), CLEANED_DIR) == Path(
        "session_gpu_005/cleaned"
    )


def test_normalize_stage_output_dir_preserves_stage_folder():
    assert normalize_stage_output_dir(Path("session_gpu_005/cleaned"), CLEANED_DIR) == Path(
        "session_gpu_005/cleaned"
    )


def test_normalize_stage_output_dir_preserves_stage_variant():
    assert normalize_stage_output_dir(
        Path("session_gpu_005/cleaned_test_v1"), CLEANED_DIR
    ) == Path("session_gpu_005/cleaned_test_v1")


def test_blender_blend_filename_has_stage_prefix():
    assert blender_blend_filename("session_gpu_005") == "blender_session_gpu_005_trajectory.blend"
