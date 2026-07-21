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
blender_y = z * depth_scale
```

`blender_y` keeps the MediaPipe `pose_z` sign: z decreases toward the camera, and
the default Blender camera looks from -Y, so a closer landmark maps to a smaller
`blender_y` and renders closer to the camera.

This is a visualization coordinate system, not real camera calibration or real-world 3D reconstruction.

## Output Files

```text
trajectory_export_points.csv
trajectory_export_segments.csv
trajectory_export_report.json
```

## Visualization Policy

- `trajectory_visible=false` rows are skipped by default.
- `trajectory_connect=false` rows can appear as points but do not create line segments.
- `trajectory_alpha` and `trajectory_width` should be passed to Blender materials or curve settings.

## Blender Auto Import

Use `scripts/open_blender_trajectory.py` to open the exported CSV files directly in Blender:

```bash
python scripts/open_blender_trajectory.py \
  --trajectory-dir examples/output/session_gpu_005/trajectory_export
```

The script can run as a normal Python launcher or inside Blender's Python runtime.
When run outside Blender, it opens Blender and executes itself there.

Default visualization settings:

```text
scene reset: fresh Blender startup scene with default Cube removed before import
camera location: 0,-5,3.4
camera rotation: 90,0,0 degrees
x_factor: 2.2
y_factor: 0.36
approximate displayed depth: 0-1.8m
marker: small emissive core
halo: smaller high-emission sphere
left/right colors: left landmarks orange, right landmarks cyan
face markers: white nose, normal-size left/right eyes, inner/outer eyes and mouth hidden
paused state: full overview trajectories
playback state: progressive draw trajectories
metadata labels: hidden by default
```

Use `--show-camera-summary` only when a compact session summary should appear in the lower-left camera view. The script creates a single optional `MPLR_CAMERA_VIEW_SUMMARY` text object and hides other MPLR metadata/debug labels by default.

The script saves `blender/blender_<session_id>_trajectory.blend` under the session output folder unless `--no-save-blend` is used.
Generated `.blend` files remain output artifacts and should not be committed.

The import status JSON reports `fresh_startup_scene=true` and `startup_cube_removed` so the generated scene setup can be checked from logs.

Video rendering is intentionally not part of the default output path. The default final artifact is the saved Blender scene.

## V1 Limits

This version does not implement `screen_floor_hybrid`, foot-contact root solving, pelvis-local skeleton reconstruction, camera calibration, homography, generated motion, or derived head landmarks.
