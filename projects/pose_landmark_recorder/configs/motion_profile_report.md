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
| session_cpu_011 | 60.0 |

## Pooled velocity (m/s) for key landmarks

| landmark | count | median | p95 | p99 | max |
|---|---|---|---|---|---|
| left_wrist | 11861 | 1.1042 | 3.5502 | 5.4971 | 20.354 |
| right_wrist | 14436 | 1.0677 | 3.6575 | 5.5419 | 30.6329 |
| left_ankle | 14892 | 0.5158 | 2.5197 | 4.5839 | 19.9992 |
| right_ankle | 14942 | 0.5453 | 2.7492 | 4.8592 | 31.1679 |
| left_hip | 15098 | 0.0956 | 0.4713 | 0.7937 | 1.6794 |
| right_hip | 15083 | 0.0949 | 0.4724 | 0.7996 | 1.6733 |
| nose | 15131 | 0.515 | 2.0011 | 2.8955 | 5.959 |

## Cross-session velocity p95 (m/s) — fps-normalization consistency

| landmark | session_cpu_006_v2 | session_cpu_007_v2 | session_cpu_008 | session_cpu_009 | session_cpu_010 | session_cpu_011 |
|---|---|---|---|---|---|---|
| left_wrist | 3.6693 | 3.6588 | 3.438 | 3.3846 | 3.5371 | 4.1218 |
| right_wrist | 3.6456 | 3.5687 | 3.3387 | 3.8131 | 3.5299 | 6.5124 |
| left_ankle | 2.8767 | 2.6452 | 2.4199 | 2.0594 | 2.6163 | 3.6456 |
| right_ankle | 3.0049 | 2.7482 | 2.7994 | 2.2266 | 5.4287 | 2.169 |
| left_hip | 0.4826 | 0.5201 | 0.4536 | 0.4361 | 0.4084 | 1.3105 |
| right_hip | 0.4755 | 0.5199 | 0.4546 | 0.4362 | 0.441 | 1.3016 |
| nose | 2.2306 | 2.1971 | 2.0189 | 1.6442 | 1.4631 | 2.6613 |

## Pooled bone lengths (m)

| bone | count | median | p95 | p99 | max |
|---|---|---|---|---|---|
| left_lower_arm | 11621 | 0.2129 | 0.2656 | 0.3015 | 0.7089 |
| left_lower_leg | 14875 | 0.3551 | 0.4088 | 0.4441 | 0.8519 |
| left_upper_arm | 12142 | 0.2168 | 0.2716 | 0.3397 | 0.5834 |
| left_upper_leg | 14930 | 0.3951 | 0.4394 | 0.4535 | 0.471 |
| right_lower_arm | 14494 | 0.2162 | 0.2658 | 0.2893 | 0.6933 |
| right_lower_leg | 14993 | 0.3678 | 0.4017 | 0.4229 | 0.4948 |
| right_upper_arm | 14734 | 0.2278 | 0.27 | 0.2893 | 0.5668 |
| right_upper_leg | 15044 | 0.3853 | 0.4167 | 0.4294 | 0.4642 |
