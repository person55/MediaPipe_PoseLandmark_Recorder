# Crop Refinement

Crop refinement is an optional post-cleaning step.

It uses `cleaned_pose.csv` to detect problematic frame segments, creates torso-centered person crops from the original video, re-runs MediaPipe on those crop inputs, and accepts crop candidates only when they score better than the cleaned values.

## Pipeline

```text
raw_pose.csv
-> cleaned_pose.csv
-> crop refinement candidates
-> crop_refine_pose.csv
-> outlier minimization
-> trajectory export
```

## Current Baseline

The current baseline keeps MediaPipe Full as the model and limits crop refinement to short or mixed problem regions.

Default crop sizing:

```text
crop_margin_ratio: 1.65
full_body_margin_ratio: 1.45
crop_min_size: 480 px
```

Default segment selection:

```text
target_segment_types: mixed_problem_segment
max_segment_length: 100
exclude_review_only: true
exclude_missing_long_gap: true
include_short_invalid_cluster: false
```

Long unreliable runs and `missing_long_gap` regions are reported but are not crop-refined by default.

## Why Torso-Centered Crop

Dance poses can include extended limbs, crossed arms/legs, and unstable hand/foot landmarks.

For this reason, crop center should be based on torso landmarks:

- shoulders
- hips
- pelvis midpoint
- shoulder midpoint

Hands and feet may expand the crop size but should not control the crop center.

## What This Step Does

- creates crop-based MediaPipe candidates
- restores crop coordinates to original frame coordinates
- compares cleaned and crop candidates
- accepts better candidates only
- records selected and excluded crop segments with selection reasons

## What This Step Does Not Do

- does not generate missing motion
- does not fill long missing ranges
- does not reconstruct frame-out limbs
- does not replace cleaned data blindly
- does not use skeleton optimization as the final visualization layer

## Recommended Use

Start with short or mixed problematic segments.

Avoid automatic acceptance in long unreliable runs. Those regions should move to outlier minimization as trajectory breaks or to later manually reviewed/generated layers if the project explicitly needs them.
