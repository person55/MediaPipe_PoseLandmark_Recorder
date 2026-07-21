# Motion Profile Report

> Read-only statistical profile. Not wired into any pipeline threshold or acceptance decision; values are observational.

## Sessions

| session | fps |
|---|---|
| session_cpu_006_v2 | 23.976 |
| session_cpu_007_v2 | 23.976 |
| session_cpu_008 | 23.976 |
| session_cpu_009 | 60.0 |

## Pooled velocity (m/s) for key landmarks

| landmark | count | median | p95 | p99 | max |
|---|---|---|---|---|---|
| left_wrist | 10943 | 1.0987 | 3.5322 | 5.3373 | 20.354 |
| right_wrist | 13508 | 1.0871 | 3.6287 | 5.2397 | 30.6329 |
| left_ankle | 13964 | 0.5074 | 2.4586 | 4.5008 | 19.9992 |
| right_ankle | 14028 | 0.5367 | 2.5847 | 4.6728 | 31.1679 |
| left_hip | 14145 | 0.0951 | 0.4658 | 0.7269 | 1.3652 |
| right_hip | 14133 | 0.0945 | 0.4663 | 0.7231 | 1.3661 |
| nose | 14183 | 0.5298 | 1.9958 | 2.8855 | 5.959 |

## Cross-session velocity p95 (m/s) — fps-normalization consistency

| landmark | session_cpu_006_v2 | session_cpu_007_v2 | session_cpu_008 | session_cpu_009 |
|---|---|---|---|---|
| left_wrist | 3.6693 | 3.6588 | 3.438 | 3.3846 |
| right_wrist | 3.6456 | 3.5687 | 3.3387 | 3.8131 |
| left_ankle | 2.8767 | 2.6452 | 2.4199 | 2.0594 |
| right_ankle | 3.0049 | 2.7482 | 2.7994 | 2.2266 |
| left_hip | 0.4826 | 0.5201 | 0.4536 | 0.4361 |
| right_hip | 0.4755 | 0.5199 | 0.4546 | 0.4362 |
| nose | 2.2306 | 2.1971 | 2.0189 | 1.6442 |

## Pooled bone lengths (m)

| bone | count | median | p95 | p99 | max |
|---|---|---|---|---|---|
| left_lower_arm | 10708 | 0.2101 | 0.2623 | 0.3047 | 0.7089 |
| left_lower_leg | 13948 | 0.3549 | 0.4102 | 0.4461 | 0.8519 |
| left_upper_arm | 11221 | 0.2161 | 0.2725 | 0.3574 | 0.5834 |
| left_upper_leg | 13985 | 0.3944 | 0.4397 | 0.454 | 0.471 |
| right_lower_arm | 13562 | 0.2152 | 0.266 | 0.2902 | 0.6933 |
| right_lower_leg | 14067 | 0.3671 | 0.4009 | 0.4221 | 0.4948 |
| right_upper_arm | 13795 | 0.2266 | 0.2702 | 0.29 | 0.5668 |
| right_upper_leg | 14097 | 0.3851 | 0.4166 | 0.429 | 0.4642 |
