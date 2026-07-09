# Trajectory Export

Trajectory export converts `outlier_minimized_pose.csv` into Blender/TouchDesigner-ready CSV files.

It does not correct pose data.
It converts existing outlier and trajectory display policies into points and line segments.

## Default Landmark Preset

The default Blender preset excludes:

- `left_ear`
- `right_ear`
- `left_index`
- `right_index`
- `left_thumb`
- `right_thumb`

It keeps:

- `left_foot_index`
- `right_foot_index`
- `left_pinky`
- `right_pinky`

There is no official MediaPipe Pose `head` landmark.
The default head proxy is `nose`.

## Default Coordinate Mode

The default coordinate mode is `screen_bottom_origin`.

It uses the screen bottom center as the Blender origin:

```text
screen_origin_x = 0.5
screen_origin_y = 1.0
```

The default mapping is:

```text
blender_x = (x - screen_origin_x) * screen_width_scale
blender_z = (screen_origin_y - y) * screen_height_scale
blender_y = -z * depth_scale
```

This is a visualization coordinate system, not real camera calibration or real-world 3D reconstruction.

## Output Files

```text
blender_trajectory_points.csv
blender_trajectory_segments.csv
trajectory_export_report.json
```

## Visualization Policy

- `trajectory_visible=false` rows are skipped by default.
- `trajectory_connect=false` rows can appear as points but do not create line segments.
- `trajectory_alpha` and `trajectory_width` should be passed to Blender materials or curve settings.

## V1 Limits

This version does not implement `screen_floor_hybrid`, foot-contact root solving, pelvis-local skeleton reconstruction, camera calibration, homography, generated motion, or derived head landmarks.
