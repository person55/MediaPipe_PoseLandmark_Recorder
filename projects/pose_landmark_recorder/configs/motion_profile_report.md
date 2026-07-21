# Motion Profile Report

> Read-only statistical profile. Not wired into any pipeline threshold or acceptance decision; values are observational.

## Sessions

| session | fps |
|---|---|
| session_cpu_006_v2 | 23.976 |
| session_cpu_007_v2 | 23.976 |
| session_cpu_008 | 23.976 |
| session_cpu_009 | 60.0 |
| session_cpu_010 | 59.969 |

## Pooled velocity (m/s) for key landmarks

| landmark | count | median | p95 | p99 | max |
|---|---|---|---|---|---|
| left_wrist | 11496 | 1.0919 | 3.5336 | 5.3298 | 20.354 |
| right_wrist | 14068 | 1.0699 | 3.6187 | 5.2445 | 30.6329 |
| left_ankle | 14523 | 0.5093 | 2.4597 | 4.5512 | 19.9992 |
| right_ankle | 14567 | 0.5397 | 2.7866 | 4.8767 | 31.1679 |
| left_hip | 14723 | 0.0949 | 0.4641 | 0.7242 | 1.3652 |
| right_hip | 14708 | 0.0942 | 0.4654 | 0.7266 | 1.3661 |
| nose | 14763 | 0.5202 | 1.9861 | 2.8836 | 5.959 |

## Cross-session velocity p95 (m/s) — fps-normalization consistency

| landmark | session_cpu_006_v2 | session_cpu_007_v2 | session_cpu_008 | session_cpu_009 | session_cpu_010 |
|---|---|---|---|---|---|
| left_wrist | 3.6693 | 3.6588 | 3.438 | 3.3846 | 3.5371 |
| right_wrist | 3.6456 | 3.5687 | 3.3387 | 3.8131 | 3.5299 |
| left_ankle | 2.8767 | 2.6452 | 2.4199 | 2.0594 | 2.6163 |
| right_ankle | 3.0049 | 2.7482 | 2.7994 | 2.2266 | 5.4287 |
| left_hip | 0.4826 | 0.5201 | 0.4536 | 0.4361 | 0.4084 |
| right_hip | 0.4755 | 0.5199 | 0.4546 | 0.4362 | 0.441 |
| nose | 2.2306 | 2.1971 | 2.0189 | 1.6442 | 1.4631 |

## Pooled bone lengths (m)

| bone | count | median | p95 | p99 | max |
|---|---|---|---|---|---|
| left_lower_arm | 11251 | 0.2116 | 0.2621 | 0.3005 | 0.7089 |
| left_lower_leg | 14506 | 0.3547 | 0.4091 | 0.4452 | 0.8519 |
| left_upper_arm | 11771 | 0.2162 | 0.2719 | 0.3441 | 0.5834 |
| left_upper_leg | 14561 | 0.395 | 0.4395 | 0.4537 | 0.471 |
| right_lower_arm | 14124 | 0.2153 | 0.2658 | 0.2901 | 0.6933 |
| right_lower_leg | 14617 | 0.3675 | 0.4005 | 0.4211 | 0.4948 |
| right_upper_arm | 14360 | 0.227 | 0.27 | 0.2898 | 0.5668 |
| right_upper_leg | 14668 | 0.385 | 0.4161 | 0.4289 | 0.4642 |
