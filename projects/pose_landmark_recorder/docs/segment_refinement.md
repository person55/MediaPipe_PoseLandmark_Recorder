# Segment Refinement

Segment refinement is an optional second-pass step after pose cleaning.

It does not generate new motion. It re-runs pose detection only on problematic frame ranges and accepts the re-detected candidate only when it scores better than the existing cleaned value.

## Pipeline

```plain text
raw_pose.csv
-> cleaned_pose.csv
-> segment refinement
-> refined_pose.csv
-> Blender / downstream visualization
```

In the recorder workflow this maps to three stages:

```plain text
1. record_from_video.py
   raw MediaPipe detection and raw preview

2. clean_pose_data.py
   conservative validation, interpolation, smoothing, and quality flags

3. refine_pose_segments.py
   targeted re-detection of problematic segments and score-gated replacement
```

## Candidate Segments

Candidate segments are detected from quality flags such as:

- `unreliable`
- `interpolated_outlier_removed`
- `estimated_occluded_arm`
- `low_visibility_leg_kept`
- `missing_long_gap`

Long unreliable or missing runs are marked as review-only by default. They can be re-detected to confirm whether MediaPipe can recover the pose, but their values are not automatically accepted.

## Acceptance Policy

A re-detected candidate is accepted only when it improves confidence, temporal continuity, and bone-length consistency.

The current Phase 1 implementation uses full-frame MediaPipe re-detection. Crop-based re-detection is left as a later extension.

## Important Limitation

This step cannot reconstruct motion that is not visible in the source video. Long occlusion, frame-out motion, black-screen endings, or hand proxy instability may remain unreliable. Those regions should be hidden, visualized as uncertain, or handled later by a separate generated-motion layer.
