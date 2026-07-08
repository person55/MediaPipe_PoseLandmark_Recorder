# session_gpu_003 Refine Summary

This is a compact summary for Codex context.
Do not read full output CSV files unless explicitly needed.

## Session

```text
session_id: session_gpu_003
input_video: examples/input/dance_take_003.mp4
frames_total: 902
```

## Segment Re-Detection Result

```text
candidate_segment_count: 4
redetected_segment_count: 4
accepted_row_count: 7
rejected_row_count: 3382
unavailable_row_count: 3143
```

## Segments

### Segment 1

```text
frames: 86-162
length: 77
type: mixed_problem_segment
review_only: false
accepted_rows: 7
rejected_rows: 227
unavailable_rows: 0
```

Interpretation:

```text
This is the only segment where re-detection accepted a small number of rows.
Short outlier minimization may still help here.
```

### Segment 2

```text
frames: 176-279
length: 104
type: long_unreliable_run
review_only: true
accepted_rows: 0
rejected_rows: 682
unavailable_rows: 7
```

Interpretation:

```text
Do not automatically fill this segment.
```

### Segment 3

```text
frames: 319-644
length: 326
type: long_unreliable_run
review_only: true
accepted_rows: 0
rejected_rows: 1953
unavailable_rows: 4
```

Interpretation:

```text
This is a long unreliable region.
Re-detection did not recover useful measurements.
Treat as review-only or hidden in Blender.
```

### Segment 4

```text
frames: 659-901
length: 243
type: long_unreliable_run
review_only: true
accepted_rows: 0
rejected_rows: 520
unavailable_rows: 3132
problem_flags include missing_long_gap
```

Interpretation:

```text
This segment contains large unavailable/missing regions.
Do not reconstruct automatically in the lightweight pipeline.
```

## Conclusion

Segment re-detection is useful but insufficient for session_gpu_003.

The next useful step is not stronger skeleton correction.
The next useful step is Outlier Minimizer v2:

```text
confidence-aware filtering
velocity / acceleration / jerk spike detection
landmark-group-specific policies
trajectory break handling
no generated motion
```
