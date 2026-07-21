"""Generate the cross-pass agreement diagnostic for an existing session.

Backfills `crop_crosspass_agreement.csv` (and the `crosspass_agreement` key in
`crop_refine_report.json`) from saved crop refine outputs without re-running
detection. New pipeline runs produce these automatically in the crop stage.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from dance_pose_recorder.crosspass_agreement import DEFAULT_AGREEMENT_THRESHOLD, build_crosspass_agreement
from dance_pose_recorder.output_layout import (
    CLEANED_DIR,
    CLEANED_POSE_CSV,
    CROP_REFINE_CANDIDATES_CSV,
    CROP_REFINE_CROSSPASS_CSV,
    CROP_REFINE_DIR,
    CROP_REFINE_POSE_CSV,
    CROP_REFINE_REPORT_JSON,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-dir", type=Path, required=True, help="Session output directory")
    parser.add_argument("--crop-refine-dir", type=Path, default=None, help="Override crop refine directory (e.g. a crop_refine_loop10 variant)")
    parser.add_argument("--agreement-threshold", type=float, default=DEFAULT_AGREEMENT_THRESHOLD)
    parser.add_argument("--no-update-report", action="store_true", help="Do not write the summary into crop_refine_report.json")
    args = parser.parse_args()

    crop_dir = args.crop_refine_dir or (args.session_dir / CROP_REFINE_DIR)
    refined = pd.read_csv(crop_dir / CROP_REFINE_POSE_CSV, low_memory=False)
    candidates = pd.read_csv(crop_dir / CROP_REFINE_CANDIDATES_CSV, low_memory=False)
    cleaned = pd.read_csv(args.session_dir / CLEANED_DIR / CLEANED_POSE_CSV, low_memory=False)

    rows, summary = build_crosspass_agreement(refined, candidates, cleaned, args.agreement_threshold)
    out_csv = crop_dir / CROP_REFINE_CROSSPASS_CSV
    rows.to_csv(out_csv, index=False)

    report_path = crop_dir / CROP_REFINE_REPORT_JSON
    if not args.no_update_report and report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["crosspass_agreement"] = summary
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"wrote {out_csv} ({len(rows)} rows)")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
