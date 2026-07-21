"""Cross-pass agreement diagnostics for accepted crop candidates.

For every accepted crop row (pose source), measure how close the accepted
coordinate lies to the nearest detection from an independent crop pass
(forward/mirror/reverse/...) at the same frame and landmark, and compare that
against the cleaned value the acceptance replaced.

This is observational metadata only: it never changes acceptance decisions.
Interpretation caveat (validated on session_cpu_008): inside left-right
confusion segments the independent passes can share the cleaned value's
confusion, so low agreement there does not imply a wrong acceptance and this
metric must not be used as a standalone acceptance criterion.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_AGREEMENT_THRESHOLD = 0.02

AGREEMENT_COLUMNS = [
    "frame",
    "landmark_id",
    "landmark_name",
    "crop_pass",
    "n_independent_passes",
    "nearest_independent_distance",
    "cleaned_nearest_independent_distance",
    "accepted_x",
    "accepted_y",
    "cleaned_x",
    "cleaned_y",
    "displacement",
]

_INTERPRETATION_NOTE = (
    "Diagnostic metadata only; never used for acceptance. Low agreement inside "
    "left-right confusion segments does not imply a wrong acceptance because "
    "independent passes can share the cleaned value's confusion."
)


def build_crosspass_agreement(
    refined: pd.DataFrame,
    candidates: pd.DataFrame,
    cleaned: pd.DataFrame,
    agreement_threshold: float = DEFAULT_AGREEMENT_THRESHOLD,
) -> tuple[pd.DataFrame, dict]:
    """Return (per-acceptance agreement rows, summary dict)."""
    accepted = refined[
        (refined["crop_refine_status"] == "crop_accepted") & (refined["source"] == "pose")
    ]
    if accepted.empty or candidates.empty:
        return pd.DataFrame(columns=AGREEMENT_COLUMNS), _summarize(
            pd.DataFrame(columns=AGREEMENT_COLUMNS), agreement_threshold
        )

    cleaned_xy = (
        cleaned[cleaned["source"] == "pose"]
        .set_index(["frame", "landmark_id"])[["x", "y"]]
        .rename(columns={"x": "cleaned_x", "y": "cleaned_y"})
    )
    candidate_groups = {
        key: group
        for key, group in candidates[candidates["source"] == "pose"].groupby(["frame", "landmark_id"], sort=False)
    }

    rows = []
    for row in accepted.itertuples(index=False):
        key = (int(row.frame), int(row.landmark_id))
        try:
            cleaned_row = cleaned_xy.loc[key]
        except KeyError:
            continue
        cleaned_x = float(cleaned_row.iloc[0]) if isinstance(cleaned_row, pd.DataFrame) else float(cleaned_row.cleaned_x)
        cleaned_y = float(cleaned_row.iloc[1]) if isinstance(cleaned_row, pd.DataFrame) else float(cleaned_row.cleaned_y)
        group = candidate_groups.get(key)
        independent = group[group["crop_pass"] != row.crop_pass] if group is not None else None
        if independent is None or independent.empty:
            nearest = np.nan
            cleaned_nearest = np.nan
            n_passes = 0
        else:
            nearest = float(
                np.sqrt((independent["x"] - row.x) ** 2 + (independent["y"] - row.y) ** 2).min()
            )
            cleaned_nearest = float(
                np.sqrt((independent["x"] - cleaned_x) ** 2 + (independent["y"] - cleaned_y) ** 2).min()
            )
            n_passes = int(independent["crop_pass"].nunique())
        rows.append(
            {
                "frame": int(row.frame),
                "landmark_id": int(row.landmark_id),
                "landmark_name": row.landmark_name,
                "crop_pass": row.crop_pass,
                "n_independent_passes": n_passes,
                "nearest_independent_distance": nearest,
                "cleaned_nearest_independent_distance": cleaned_nearest,
                "accepted_x": float(row.x),
                "accepted_y": float(row.y),
                "cleaned_x": cleaned_x,
                "cleaned_y": cleaned_y,
                "displacement": float(np.hypot(float(row.x) - cleaned_x, float(row.y) - cleaned_y)),
            }
        )

    agreement = pd.DataFrame(rows, columns=AGREEMENT_COLUMNS)
    return agreement, _summarize(agreement, agreement_threshold)


def _summarize(agreement: pd.DataFrame, agreement_threshold: float) -> dict:
    summary = {
        "agreement_threshold": agreement_threshold,
        "accepted_pose_rows": int(len(agreement)),
        "rows_with_independent_pass": 0,
        "note": _INTERPRETATION_NOTE,
    }
    measured = agreement.dropna(subset=["nearest_independent_distance"])
    summary["rows_with_independent_pass"] = int(len(measured))
    if measured.empty:
        return summary
    distances = measured["nearest_independent_distance"]
    summary.update(
        {
            "nearest_independent_distance_median": round(float(distances.median()), 4),
            "nearest_independent_distance_p90": round(float(distances.quantile(0.9)), 4),
            "agreement_within_threshold": round(float((distances <= agreement_threshold).mean()), 3),
            "accepted_closer_than_cleaned_fraction": round(
                float((distances < measured["cleaned_nearest_independent_distance"]).mean()), 3
            ),
            "displacement_median": round(float(measured["displacement"].median()), 4),
            "displacement_max": round(float(measured["displacement"].max()), 4),
        }
    )
    per_pass = {}
    for pass_name, group in measured.groupby("crop_pass"):
        per_pass[str(pass_name)] = {
            "rows": int(len(group)),
            "nearest_independent_distance_median": round(float(group["nearest_independent_distance"].median()), 4),
            "agreement_within_threshold": round(
                float((group["nearest_independent_distance"] <= agreement_threshold).mean()), 3
            ),
        }
    summary["per_pass"] = per_pass
    per_group: dict[str, list] = {}
    for name, group in measured.groupby("landmark_name"):
        per_group[str(name)] = [
            int(len(group)),
            round(float((group["nearest_independent_distance"] <= agreement_threshold).mean()), 3),
        ]
    summary["per_landmark_rows_and_agreement"] = per_group
    return summary
