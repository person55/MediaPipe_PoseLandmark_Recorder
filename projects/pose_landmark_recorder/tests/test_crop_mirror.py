import numpy as np

from dance_pose_recorder.crop_mirror import (
    SWAP_INDEX,
    confusion_row,
    mirror_crop,
    pass_disagreement,
    unmirror_pose_landmarks,
    unmirror_world_landmarks,
)


def _landmarks():
    return [{"x": 0.1 + 0.01 * i, "y": 0.2 + 0.01 * i, "z": 0.0, "visibility": 0.9} for i in range(33)]


def test_swap_index_is_an_involution():
    assert sorted(SWAP_INDEX) == list(range(33))
    for landmark_id, partner in enumerate(SWAP_INDEX):
        assert SWAP_INDEX[partner] == landmark_id
    assert SWAP_INDEX[0] == 0  # nose is central
    assert SWAP_INDEX[15] == 16  # wrists swap


def test_unmirror_round_trip_restores_landmarks():
    original = _landmarks()
    mirrored = [dict(original[SWAP_INDEX[i]]) for i in range(33)]
    for item in mirrored:
        item["x"] = 1.0 - item["x"]

    restored = unmirror_pose_landmarks(mirrored)

    for a, b in zip(restored, original):
        assert abs(a["x"] - b["x"]) < 1e-12
        assert abs(a["y"] - b["y"]) < 1e-12


def test_unmirror_world_flips_x_sign_and_swaps():
    world = [{"x": 0.1 * i, "y": 0.5, "z": 0.0} for i in range(33)]
    restored = unmirror_world_landmarks(world)

    assert abs(restored[15]["x"] - (-0.1 * 16)) < 1e-12  # left wrist takes right slot, sign flipped
    assert restored[0]["x"] == 0.0


def test_mirror_crop_flips_horizontally():
    image = np.zeros((4, 6, 3), dtype=np.uint8)
    image[:, 0] = 255
    assert (mirror_crop(image)[:, -1] == 255).all()


def test_pass_disagreement_and_confusion_flag():
    a = _landmarks()
    b = [dict(item) for item in a]
    assert pass_disagreement(a, b) == 0.0
    for item in b:
        item["x"] += 0.1
    disagreement = pass_disagreement(a, b)
    assert abs(disagreement - 0.1) < 1e-9

    row = confusion_row(5, 2, 0.8, disagreement, None, threshold=0.05)
    assert row["possible_confusion"] is True
    row = confusion_row(5, 2, 0.8, 0.01, None, threshold=0.05)
    assert row["possible_confusion"] is False
