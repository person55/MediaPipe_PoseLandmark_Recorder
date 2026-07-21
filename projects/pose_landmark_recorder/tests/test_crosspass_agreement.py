import numpy as np
import pandas as pd

from dance_pose_recorder.crosspass_agreement import AGREEMENT_COLUMNS, build_crosspass_agreement


def _refined_row(frame, landmark_id, x, y, status="crop_accepted", crop_pass="mirror", source="pose"):
    return {
        "frame": frame,
        "landmark_id": landmark_id,
        "landmark_name": f"lm_{landmark_id}",
        "source": source,
        "x": x,
        "y": y,
        "crop_refine_status": status,
        "crop_pass": crop_pass,
    }


def _candidate_row(frame, landmark_id, x, y, crop_pass, source="pose"):
    return {
        "frame": frame,
        "landmark_id": landmark_id,
        "source": source,
        "x": x,
        "y": y,
        "crop_pass": crop_pass,
    }


def _cleaned_row(frame, landmark_id, x, y, source="pose"):
    return {"frame": frame, "landmark_id": landmark_id, "source": source, "x": x, "y": y}


def test_nearest_independent_distance_and_summary():
    refined = pd.DataFrame([_refined_row(10, 1, 0.50, 0.50, crop_pass="mirror")])
    candidates = pd.DataFrame(
        [
            _candidate_row(10, 1, 0.50, 0.50, "mirror"),      # same pass: excluded
            _candidate_row(10, 1, 0.51, 0.50, "forward"),     # distance 0.01 (nearest)
            _candidate_row(10, 1, 0.60, 0.50, "reverse"),     # distance 0.10
        ]
    )
    cleaned = pd.DataFrame([_cleaned_row(10, 1, 0.58, 0.50)])

    rows, summary = build_crosspass_agreement(refined, candidates, cleaned)

    assert list(rows.columns) == AGREEMENT_COLUMNS
    assert len(rows) == 1
    row = rows.iloc[0]
    assert row.n_independent_passes == 2
    assert np.isclose(row.nearest_independent_distance, 0.01)
    # cleaned (0.58) is nearest to the reverse candidate (0.60): 0.02
    assert np.isclose(row.cleaned_nearest_independent_distance, 0.02)
    assert np.isclose(row.displacement, 0.08)
    assert summary["rows_with_independent_pass"] == 1
    assert summary["agreement_within_threshold"] == 1.0
    assert summary["accepted_closer_than_cleaned_fraction"] == 1.0
    assert summary["per_pass"]["mirror"]["rows"] == 1


def test_no_independent_pass_is_nan_not_agreement():
    refined = pd.DataFrame([_refined_row(5, 2, 0.4, 0.4, crop_pass="forward")])
    candidates = pd.DataFrame([_candidate_row(5, 2, 0.4, 0.4, "forward")])
    cleaned = pd.DataFrame([_cleaned_row(5, 2, 0.4, 0.4)])

    rows, summary = build_crosspass_agreement(refined, candidates, cleaned)

    assert len(rows) == 1
    assert rows.iloc[0].n_independent_passes == 0
    assert np.isnan(rows.iloc[0].nearest_independent_distance)
    assert summary["accepted_pose_rows"] == 1
    assert summary["rows_with_independent_pass"] == 0
    assert "agreement_within_threshold" not in summary


def test_only_accepted_pose_rows_are_measured():
    refined = pd.DataFrame(
        [
            _refined_row(1, 1, 0.5, 0.5, status="crop_rejected"),
            _refined_row(1, 1, 0.5, 0.5, source="pose_world"),
            _refined_row(2, 1, 0.5, 0.5),
        ]
    )
    candidates = pd.DataFrame(
        [
            _candidate_row(1, 1, 0.5, 0.5, "forward"),
            _candidate_row(2, 1, 0.505, 0.5, "forward"),
        ]
    )
    cleaned = pd.DataFrame([_cleaned_row(1, 1, 0.5, 0.5), _cleaned_row(2, 1, 0.5, 0.5)])

    rows, summary = build_crosspass_agreement(refined, candidates, cleaned)

    assert len(rows) == 1
    assert rows.iloc[0].frame == 2
    assert summary["accepted_pose_rows"] == 1


def test_empty_inputs_return_empty_contract():
    empty_refined = pd.DataFrame(columns=["frame", "landmark_id", "landmark_name", "source", "x", "y", "crop_refine_status", "crop_pass"])
    rows, summary = build_crosspass_agreement(empty_refined, pd.DataFrame(), pd.DataFrame())
    assert list(rows.columns) == AGREEMENT_COLUMNS
    assert len(rows) == 0
    assert summary["accepted_pose_rows"] == 0
    assert "note" in summary
