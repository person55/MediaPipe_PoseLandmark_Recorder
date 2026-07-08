# Coordinate System

The recorder preserves MediaPipe output values and also writes transformed values for downstream 3D tools.

## Raw values

Raw values are written without modification:

- `x`
- `y`
- `z`
- `visibility`
- `presence`

## Transformed values

The current axis mapping is:

```python
tx = world_x
ty = -world_z
tz = -world_y
```

This is the first Blender-oriented mapping and may need adjustment after visual inspection in Blender or C4D.

## Origin policies

- `raw`: keep the MediaPipe origin.
- `first_frame_pelvis`: use the first detected pelvis center as the fixed session origin.
- `per_frame_pelvis`: use each frame's pelvis center as that frame's local origin.

The pelvis center is the midpoint between `left_hip` and `right_hip`.
