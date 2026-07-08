# Quality Flags

The cleaning pipeline writes `quality_flag` values so downstream tools can distinguish measured landmarks from corrected or unreliable values.

## Flags

| quality_flag | Meaning | Suggested Blender display |
|---|---|---|
| `measured` | Original measurement kept after validation and optional smoothing. | Solid point / solid trajectory |
| `interpolated_short_gap` | Landmark value filled across a short missing-frame gap. | Yellow point or dotted segment |
| `interpolated_outlier_removed` | Short recoverable spike/outlier removed and linearly interpolated. | Blue point or dotted segment |
| `estimated_occluded_arm` | Bounded elbow/wrist occlusion estimated from nearby arm structure. | Translucent arm point / faint trajectory |
| `low_visibility_leg_kept` | Low-visibility leg measurement kept because motion and bone checks were stable. | Faint point / low-alpha trajectory |
| `unreliable` | Invalid or uncertain landmark that should not be treated as reliable motion. | Hidden by default |
| `missing_long_gap` | Long missing range that is not interpolated. | Hidden by default |

## Policy

The cleaned output is not a replacement for raw MediaPipe output. It is a downstream visualization layer.

- `raw_pose.csv` and `raw_pose.jsonl` preserve direct MediaPipe measurements.
- `cleaned_pose.csv` and `cleaned_pose.jsonl` add validation, interpolation, smoothing, and quality flags.
- Blender/C4D/After Effects importers should read `quality_flag` and avoid displaying all corrected values as equally reliable.
- Long missing ranges and long outlier runs should not be converted into plausible-looking motion by default.
