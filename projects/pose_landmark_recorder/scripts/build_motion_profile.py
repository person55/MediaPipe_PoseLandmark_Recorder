"""Build the read-only statistical motion profile from session outputs.

Aggregates stable-frame motion statistics (physical per-second units) and bone
lengths across sessions into configs/motion_profile_default.json plus a
human-readable markdown report. Observational only: no pipeline stage reads
this profile for thresholds or acceptance decisions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from dance_pose_recorder.motion_profile import build_motion_profile
from dance_pose_recorder.output_layout import (
    OUTLIER_MINIMIZED_DIR,
    OUTLIER_MINIMIZED_POSE_CSV,
    RAW_DIR,
    RAW_METADATA_JSON,
)

KEY_LANDMARKS = ["left_wrist", "right_wrist", "left_ankle", "right_ankle", "left_hip", "right_hip", "nose"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--session-dir",
        action="append",
        required=True,
        help=(
            "Session output directory (repeatable). Append '::<relpath>' to "
            "override the pose CSV path for that session, e.g. "
            "'examples/output/session_x::outlier_minimized_v3/outlier_minimized_pose.csv'."
        ),
    )
    parser.add_argument(
        "--pose-csv-relpath",
        default=f"{OUTLIER_MINIMIZED_DIR}/{OUTLIER_MINIMIZED_POSE_CSV}",
        help="Default pose CSV path relative to each session dir.",
    )
    parser.add_argument("--output-json", type=Path, default=Path("configs/motion_profile_default.json"))
    parser.add_argument("--output-report", type=Path, default=Path("configs/motion_profile_report.md"))
    args = parser.parse_args()

    sessions = []
    for entry in args.session_dir:
        raw_path, _, override = str(entry).partition("::")
        session_dir = Path(raw_path)
        relpath = override or args.pose_csv_relpath
        metadata = json.loads((session_dir / RAW_DIR / RAW_METADATA_JSON).read_text(encoding="utf-8"))
        fps = float(metadata.get("fps") or 30.0)
        pose = pd.read_csv(session_dir / relpath, low_memory=False)
        sessions.append((session_dir.name, pose, fps))

    profile = build_motion_profile(sessions)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Motion Profile Report",
        "",
        f"> {profile['note']}",
        "",
        "## Sessions",
        "",
        "| session | fps |",
        "|---|---|",
    ]
    for sid, fps in profile["sessions"].items():
        lines.append(f"| {sid} | {fps} |")
    lines += [
        "",
        "## Pooled velocity (m/s) for key landmarks",
        "",
        "| landmark | count | median | p95 | p99 | max |",
        "|---|---|---|---|---|---|",
    ]
    for landmark in KEY_LANDMARKS:
        stats = profile["landmarks"].get(landmark, {}).get("velocity_m_per_s", {})
        if stats.get("count"):
            lines.append(
                f"| {landmark} | {stats['count']} | {stats['median']} | {stats['p95']} | {stats['p99']} | {stats['max']} |"
            )
    lines += [
        "",
        "## Cross-session velocity p95 (m/s) — fps-normalization consistency",
        "",
        "| landmark | " + " | ".join(profile["sessions"]) + " |",
        "|---|" + "|".join(["---"] * len(profile["sessions"])) + "|",
    ]
    for landmark in KEY_LANDMARKS:
        cells = []
        for sid in profile["sessions"]:
            stats = profile["per_session"][sid]["landmarks"].get(landmark, {}).get("velocity_m_per_s", {})
            cells.append(str(stats.get("p95", "-")))
        lines.append(f"| {landmark} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Pooled bone lengths (m)",
        "",
        "| bone | count | median | p95 | p99 | max |",
        "|---|---|---|---|---|---|",
    ]
    for bone, stats in profile["bones"].items():
        if stats.get("count"):
            lines.append(
                f"| {bone} | {stats['count']} | {stats['median']} | {stats['p95']} | {stats['p99']} | {stats['max']} |"
            )
    lines.append("")
    args.output_report.write_text("\n".join(lines), encoding="utf-8")

    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_report}")
    print(json.dumps({"sessions": profile["sessions"]}, indent=2))


if __name__ == "__main__":
    main()
