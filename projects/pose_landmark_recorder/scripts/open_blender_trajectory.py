#!/usr/bin/env python3
"""Open Blender and import exported MediaPipe trajectory CSV files.

This file is intentionally self-contained so it can run both as a normal
Python launcher and as the script executed inside Blender's Python runtime.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dance_pose_recorder.output_layout import (  # noqa: E402
    BLENDER_DIR,
    TRAJECTORY_EXPORT_DIR,
    TRAJECTORY_EXPORT_POINTS_CSV,
    TRAJECTORY_EXPORT_REPORT_JSON,
    TRAJECTORY_EXPORT_SEGMENTS_CSV,
    blender_blend_filename,
    resolve_existing_file,
    session_id_from_report,
)


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_DIR = SCRIPT_PATH.parents[1]
DEFAULT_SESSION_ID = "session_gpu_005"
DEFAULT_TRAJECTORY_DIR = (
    PROJECT_DIR / "examples" / "output" / DEFAULT_SESSION_ID / TRAJECTORY_EXPORT_DIR
)


def _parse_vec3(value: str, *, kind: str) -> tuple[float, float, float]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"{kind} must be formatted as x,y,z")
    try:
        return (float(parts[0]), float(parts[1]), float(parts[2]))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{kind} must contain numeric values") from exc


def _default_save_blend_path(trajectory_dir: Path, report_json: Path) -> Path:
    session_id = session_id_from_report(report_json, DEFAULT_SESSION_ID)
    session_dir = trajectory_dir.parent
    return session_dir / BLENDER_DIR / blender_blend_filename(session_id)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Open Blender and import trajectory-export CSV files."
    )
    parser.add_argument(
        "--trajectory-dir",
        type=Path,
        default=DEFAULT_TRAJECTORY_DIR,
        help="Directory containing trajectory_export_points.csv, "
        "trajectory_export_segments.csv, and trajectory_export_report.json.",
    )
    parser.add_argument("--points-csv", type=Path)
    parser.add_argument("--segments-csv", type=Path)
    parser.add_argument("--report-json", type=Path)
    parser.add_argument("--save-blend", type=Path)
    parser.add_argument(
        "--no-save-blend",
        action="store_true",
        help="Import into Blender without saving a .blend file.",
    )
    parser.add_argument(
        "--blender-bin",
        type=Path,
        default=os.environ.get("BLENDER_BIN"),
        help="Blender executable path. Defaults to BLENDER_BIN, common macOS path, "
        "or blender on PATH.",
    )
    parser.add_argument(
        "--inside-blender",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--x-factor",
        type=float,
        default=2.2,
        help="Visual scale applied to Blender X so screen width is not collapsed.",
    )
    parser.add_argument(
        "--y-factor",
        type=float,
        default=0.36,
        help="Visual scale applied to estimated Blender Y depth.",
    )
    parser.add_argument(
        "--depth-meters",
        type=float,
        default=5.0,
        help="Approximate fixed-camera depth span before y-factor is applied.",
    )
    parser.add_argument(
        "--local-depth-limit",
        type=float,
        default=0.16,
        help="Maximum local pose_z offset in Blender meters before y-factor.",
    )
    parser.add_argument(
        "--local-depth-scale",
        type=float,
        default=0.35,
        help="Scale for local pose_z offset around frame-level body depth.",
    )
    parser.add_argument(
        "--depth-smoothing-window",
        type=int,
        default=15,
        help="Rolling median window for fixed-camera depth estimation.",
    )
    parser.add_argument(
        "--camera-location",
        type=lambda value: _parse_vec3(value, kind="camera-location"),
        default=(0.0, -5.0, 3.4),
        help="Default Blender camera location as x,y,z.",
    )
    parser.add_argument(
        "--camera-rotation",
        type=lambda value: _parse_vec3(value, kind="camera-rotation"),
        default=(90.0, 0.0, 0.0),
        help="Default Blender camera rotation in degrees as x,y,z.",
    )
    parser.add_argument(
        "--no-set-camera",
        action="store_true",
        help="Do not set the Blender camera transform.",
    )
    parser.add_argument(
        "--show-camera-summary",
        action="store_true",
        help="Show one compact import summary in the lower-left camera view. "
        "Hidden by default.",
    )
    parser.add_argument(
        "--camera-summary-location",
        type=lambda value: _parse_vec3(value, kind="camera-summary-location"),
        default=(-1.78, -0.98, -3.0),
        help="Camera-local x,y,z position for the optional summary label.",
    )
    parser.add_argument(
        "--camera-summary-size",
        type=float,
        default=0.075,
        help="Font size for the optional camera-view summary label.",
    )
    parser.add_argument(
        "--marker-radius",
        type=float,
        default=0.044,
        help="Base marker core radius.",
    )
    parser.add_argument(
        "--halo-radius",
        type=float,
        default=0.072,
        help="Base glow sphere radius.",
    )
    parser.add_argument(
        "--marker-emission-strength",
        type=float,
        default=45.0,
    )
    parser.add_argument(
        "--halo-emission-strength",
        type=float,
        default=90.0,
    )
    parser.add_argument(
        "--overview-trail-alpha",
        type=float,
        default=0.30,
    )
    parser.add_argument(
        "--root-collection",
        default=None,
        help="Blender root collection name. Defaults to MPLR_<session_id>.",
    )
    return parser


def resolve_paths(args: argparse.Namespace) -> None:
    trajectory_dir = args.trajectory_dir
    args.points_csv = args.points_csv or resolve_existing_file(
        trajectory_dir,
        TRAJECTORY_EXPORT_POINTS_CSV,
        ("blender_trajectory_points.csv",),
    )
    args.segments_csv = args.segments_csv or resolve_existing_file(
        trajectory_dir,
        TRAJECTORY_EXPORT_SEGMENTS_CSV,
        ("blender_trajectory_segments.csv",),
    )
    args.report_json = args.report_json or resolve_existing_file(
        trajectory_dir,
        TRAJECTORY_EXPORT_REPORT_JSON,
    )
    if args.save_blend is None and not args.no_save_blend:
        args.save_blend = _default_save_blend_path(trajectory_dir, args.report_json)


def find_blender_executable(args: argparse.Namespace) -> Path:
    if args.blender_bin:
        path = Path(args.blender_bin).expanduser()
        if path.exists():
            return path
        raise FileNotFoundError(f"Blender executable not found: {path}")

    mac_path = Path("/Applications/Blender.app/Contents/MacOS/Blender")
    if mac_path.exists():
        return mac_path

    found = shutil.which("blender")
    if found:
        return Path(found)

    raise FileNotFoundError(
        "Blender executable not found. Set BLENDER_BIN or pass --blender-bin."
    )


def running_inside_blender() -> bool:
    try:
        import bpy  # type: ignore  # noqa: F401
    except Exception:
        return False
    return True


def launch_blender(args: argparse.Namespace) -> int:
    blender = find_blender_executable(args)
    passthrough = [
        str(blender),
        "--python",
        str(SCRIPT_PATH),
        "--",
        "--inside-blender",
    ]
    original_args = sys.argv[1:]
    passthrough.extend(arg for arg in original_args if arg != "--inside-blender")
    print("Opening Blender trajectory scene:")
    print(" ".join(passthrough))
    return subprocess.run(passthrough, check=False).returncode


def read_report(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing trajectory export report: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def median(values: list[float]) -> float | None:
    values = sorted(values)
    if not values:
        return None
    middle = len(values) // 2
    if len(values) % 2:
        return values[middle]
    return (values[middle - 1] + values[middle]) / 2.0


def percentile(values: list[float], pct: float) -> float:
    values = sorted(values)
    if not values:
        return 0.0
    index = (len(values) - 1) * pct / 100.0
    low = int(math.floor(index))
    high = int(math.ceil(index))
    if low == high:
        return values[low]
    return values[low] * (high - index) + values[high] * (index - low)


def import_in_blender(args: argparse.Namespace) -> None:
    import bpy  # type: ignore

    for path in (args.points_csv, args.segments_csv, args.report_json):
        if not path.exists():
            raise FileNotFoundError(f"Missing required trajectory file: {path}")

    # Start from a fresh Blender scene so each generated .blend begins from the
    # default new-file state, then remove the startup cube before importing CSV data.
    try:
        bpy.ops.wm.read_homefile(use_empty=False)
    except Exception:
        bpy.ops.wm.read_factory_settings(use_empty=False)

    startup_cube_removed = False
    startup_cube = bpy.data.objects.get("Cube")
    if startup_cube is not None and startup_cube.type == "MESH":
        bpy.data.objects.remove(startup_cube, do_unlink=True)
        startup_cube_removed = True

    report = read_report(args.report_json)
    session_id = report.get("session_id") or DEFAULT_SESSION_ID
    root_name = args.root_collection or f"MPLR_{session_id}"
    frames_total = int(report.get("frames_total") or 0)
    fps = float(report.get("fps") or 30.0)
    frame_start = 0
    frame_end = max(0, frames_total - 1)

    scene = bpy.context.scene
    scene.frame_start = frame_start
    scene.frame_end = frame_end
    scene.render.fps = int(round(fps)) if fps else 30

    if not args.no_set_camera:
        camera = scene.camera or bpy.data.objects.get("Camera")
        if camera is None:
            camera_data = bpy.data.cameras.new("Camera")
            camera = bpy.data.objects.new("Camera", camera_data)
            bpy.context.scene.collection.objects.link(camera)
            scene.camera = camera
        camera.location = args.camera_location
        camera.rotation_euler = tuple(math.radians(value) for value in args.camera_rotation)
        scene.camera = camera

    def ensure_collection(name: str, parent: Any | None = None) -> Any:
        collection = bpy.data.collections.get(name) or bpy.data.collections.new(name)
        try:
            if parent is None:
                bpy.context.scene.collection.children.link(collection)
            else:
                parent.children.link(collection)
        except RuntimeError:
            pass
        return collection

    def clear_collection(collection: Any) -> int:
        objects = list(collection.objects)
        count = len(objects)
        if objects:
            try:
                bpy.data.batch_remove(objects)
            except Exception:
                for obj in objects:
                    bpy.data.objects.remove(obj, do_unlink=True)
        for child in list(collection.children):
            clear_collection(child)
            bpy.data.collections.remove(child)
        return count

    root = ensure_collection(root_name)
    curves_col = ensure_collection("Trajectory_Curves", root)
    progress_col = ensure_collection("Progressive_Trajectory_Curves", root)
    points_col = ensure_collection("Point_Cloud", root)
    helpers_col = ensure_collection("Helpers", root)
    anim_col = ensure_collection("Animated_Markers", root)
    halo_col = ensure_collection("Animated_Marker_Halos", root)

    removed = {
        "overview_curves": clear_collection(curves_col),
        "progressive_curves": clear_collection(progress_col),
        "point_cloud": clear_collection(points_col),
        "helpers": clear_collection(helpers_col),
        "animated_markers": clear_collection(anim_col),
        "halos": clear_collection(halo_col),
    }

    def hide_mplr_text_labels() -> int:
        hidden = 0
        for obj in bpy.data.objects:
            if obj.type == "FONT" and obj.name.startswith("MPLR_"):
                obj.hide_viewport = True
                obj.hide_render = True
                hidden += 1
        return hidden

    hidden_metadata_labels = hide_mplr_text_labels()

    rows_by_lm: dict[str, list[tuple[int, str, float, float, float, float, float, dict[str, str]]]] = defaultdict(list)
    rows_by_frame: dict[int, list[tuple[int, str, float, float, float, float, float, dict[str, str]]]] = defaultdict(list)
    raw_y_by_frame: dict[int, list[float]] = defaultdict(list)
    core_for_scale = {
        "nose",
        "left_shoulder",
        "right_shoulder",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
        "left_heel",
        "right_heel",
        "left_foot_index",
        "right_foot_index",
    }
    foot_names = {
        "left_ankle",
        "right_ankle",
        "left_heel",
        "right_heel",
        "left_foot_index",
        "right_foot_index",
    }

    with args.points_csv.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if str(row.get("trajectory_visible", "")).lower() not in {"true", "1", "yes"}:
                continue
            try:
                frame = int(float(row["frame"]))
                landmark = row["landmark_name"]
                x = float(row["blender_x"])
                raw_y = float(row["blender_y"])
                z = float(row["blender_z"])
                screen_x = float(row.get("screen_x") or row.get("x") or 0.0)
                screen_y = float(row.get("screen_y") or row.get("y") or 0.0)
                if any(math.isnan(v) or math.isinf(v) for v in (x, raw_y, z, screen_x, screen_y)):
                    continue
            except Exception:
                continue
            record = (frame, landmark, x, raw_y, z, screen_x, screen_y, row)
            rows_by_lm[landmark].append(record)
            rows_by_frame[frame].append(record)
            raw_y_by_frame[frame].append(raw_y)

    if not rows_by_frame:
        raise RuntimeError("No visible trajectory points were read from points CSV.")

    metrics: dict[int, dict[str, float]] = {}
    for frame, records in rows_by_frame.items():
        core = [record for record in records if record[1] in core_for_scale] or records
        y_values = [record[6] for record in core]
        x_values = [record[5] for record in core]
        foot_values = [record[6] for record in records if record[1] in foot_names]
        metrics[frame] = {
            "body_h": max(y_values) - min(y_values) if y_values else 0.0,
            "body_w": max(x_values) - min(x_values) if x_values else 0.0,
            "foot_y": median(foot_values) if foot_values else (max(y_values) if y_values else 0.0),
        }

    sorted_frames = sorted(metrics)
    body_h_values = [metrics[frame]["body_h"] for frame in sorted_frames if metrics[frame]["body_h"] > 0]
    foot_y_values = [metrics[frame]["foot_y"] for frame in sorted_frames]
    h_low = percentile(body_h_values, 5)
    h_high = percentile(body_h_values, 95)
    f_low = percentile(foot_y_values, 5)
    f_high = percentile(foot_y_values, 95)
    if abs(h_high - h_low) < 1e-6:
        h_high = h_low + 1.0
    if abs(f_high - f_low) < 1e-6:
        f_high = f_low + 1.0

    raw_closeness: dict[int, float] = {}
    for frame in sorted_frames:
        h_norm = max(0.0, min(1.0, (metrics[frame]["body_h"] - h_low) / (h_high - h_low)))
        f_norm = max(0.0, min(1.0, (metrics[frame]["foot_y"] - f_low) / (f_high - f_low)))
        raw_closeness[frame] = 0.75 * h_norm + 0.25 * f_norm

    smoothed_closeness: dict[int, float] = {}
    half_window = max(0, args.depth_smoothing_window // 2)
    for index, frame in enumerate(sorted_frames):
        low = max(0, index - half_window)
        high = min(len(sorted_frames), index + half_window + 1)
        smoothed_closeness[frame] = median(
            [raw_closeness[sorted_frames[i]] for i in range(low, high)]
        ) or 0.0

    def raw_y_root(frame: int) -> float:
        return median(raw_y_by_frame.get(frame, [0.0])) or 0.0

    def frame_depth(frame: int) -> float:
        closeness = smoothed_closeness.get(frame, 0.0)
        return max(0.0, min(args.depth_meters, (1.0 - closeness) * args.depth_meters))

    def mapped_location(frame: int, landmark: str, x: float, raw_y: float, z: float) -> tuple[float, float, float]:
        del landmark
        local = (raw_y - raw_y_root(frame)) * args.local_depth_scale
        local = max(-args.local_depth_limit, min(args.local_depth_limit, local))
        y = max(0.0, min(args.depth_meters, frame_depth(frame) + local))
        return (x * args.x_factor, y * args.y_factor, z)

    colors = {
        "head_proxy": (1.0, 1.0, 1.0, 1.0),
        "face_detail": (0.72, 0.76, 0.90, 0.82),
        "torso": (1.0, 0.86, 0.04, 1.0),
        "arms": (0.0, 0.92, 1.0, 1.0),
        "hands_proxy": (1.0, 0.16, 1.0, 1.0),
        "legs": (0.25, 1.0, 0.24, 1.0),
        "feet": (1.0, 0.43, 0.0, 1.0),
        "other": (0.86, 0.86, 0.86, 0.88),
    }
    lr_colors = {
        "left": (1.0, 0.36, 0.08, 1.0),
        "right": (0.0, 0.72, 1.0, 1.0),
        "center": (0.95, 0.95, 0.95, 1.0),
    }
    face_marker_names_to_hide = {
        "left_eye_inner",
        "left_eye_outer",
        "right_eye_inner",
        "right_eye_outer",
        "mouth_left",
        "mouth_right",
    }

    def landmark_group(name: str) -> str:
        if name == "nose":
            return "head_proxy"
        if "eye" in name or "mouth" in name:
            return "face_detail"
        if name.endswith("shoulder") or name.endswith("hip"):
            return "torso"
        if name.endswith("elbow") or name.endswith("wrist"):
            return "arms"
        if name.endswith("pinky"):
            return "hands_proxy"
        if name.endswith("knee") or name.endswith("ankle"):
            return "legs"
        if name.endswith("heel") or name.endswith("foot_index"):
            return "feet"
        return "other"

    def landmark_side(name: str) -> str:
        if name.startswith("left_"):
            return "left"
        if name.startswith("right_"):
            return "right"
        return "center"

    def make_material(name: str, color: tuple[float, float, float, float], *, emission: bool, strength: float, alpha: float | None = None) -> Any:
        alpha = color[3] if alpha is None else alpha
        material = bpy.data.materials.get(name) or bpy.data.materials.new(name)
        material.diffuse_color = (color[0], color[1], color[2], alpha)
        material.use_nodes = True
        material.node_tree.nodes.clear()
        output = material.node_tree.nodes.new(type="ShaderNodeOutputMaterial")
        if emission:
            node = material.node_tree.nodes.new(type="ShaderNodeEmission")
            node.inputs["Color"].default_value = (color[0], color[1], color[2], alpha)
            node.inputs["Strength"].default_value = strength
            material.node_tree.links.new(node.outputs["Emission"], output.inputs["Surface"])
        else:
            node = material.node_tree.nodes.new(type="ShaderNodeBsdfPrincipled")
            node.inputs["Base Color"].default_value = (color[0], color[1], color[2], alpha)
            node.inputs["Alpha"].default_value = alpha
            material.node_tree.links.new(node.outputs["BSDF"], output.inputs["Surface"])
        material.blend_method = "BLEND"
        material.show_transparent_back = True
        return material

    marker_materials = {
        group: make_material(
            f"MPLR_MARKER_AUTO_{group}",
            color,
            emission=True,
            strength=args.marker_emission_strength,
            alpha=color[3],
        )
        for group, color in colors.items()
    }
    halo_materials = {
        group: make_material(
            f"MPLR_HALO_AUTO_{group}",
            color,
            emission=True,
            strength=args.halo_emission_strength,
            alpha=0.22,
        )
        for group, color in colors.items()
    }
    overview_materials = {
        group: make_material(
            f"MPLR_TRAIL_AUTO_OVERVIEW_{group}",
            color,
            emission=False,
            strength=1.0,
            alpha=args.overview_trail_alpha,
        )
        for group, color in colors.items()
    }
    progress_materials = {
        group: make_material(
            f"MPLR_TRAIL_AUTO_PROGRESS_{group}",
            color,
            emission=True,
            strength=4.0,
            alpha=0.95,
        )
        for group, color in colors.items()
    }
    point_materials = {
        group: make_material(
            f"MPLR_POINT_AUTO_{group}",
            color,
            emission=True,
            strength=2.0,
            alpha=0.60,
        )
        for group, color in colors.items()
    }
    lr_overview_materials = {
        side: make_material(
            f"MPLR_LR_{side.upper()}_OVERVIEW",
            color,
            emission=False,
            strength=1.0,
            alpha=args.overview_trail_alpha if side != "center" else 0.36,
        )
        for side, color in lr_colors.items()
    }
    lr_progress_materials = {
        side: make_material(
            f"MPLR_LR_{side.upper()}_DRAW",
            color,
            emission=True,
            strength=5.5 if side != "center" else 2.2,
            alpha=0.95 if side != "center" else 0.65,
        )
        for side, color in lr_colors.items()
    }
    lr_marker_materials = {
        side: make_material(
            f"MPLR_LR_{side.upper()}_MARKER",
            color,
            emission=True,
            strength=args.marker_emission_strength if side != "center" else 70.0,
            alpha=color[3],
        )
        for side, color in lr_colors.items()
    }
    lr_halo_materials = {
        side: make_material(
            f"MPLR_LR_{side.upper()}_HALO",
            color,
            emission=True,
            strength=args.halo_emission_strength if side != "center" else 130.0,
            alpha=0.22 if side != "center" else 0.26,
        )
        for side, color in lr_colors.items()
    }

    def object_material(obj: Any, material: Any) -> None:
        if not obj.material_slots:
            obj.data.materials.append(material)
        slot = obj.material_slots[0]
        try:
            slot.link = "OBJECT"
        except Exception:
            pass
        slot.material = material

    def add_polyline(curve: Any, points: list[tuple[float, float, float]]) -> bool:
        if len(points) < 2:
            return False
        spline = curve.splines.new("POLY")
        spline.points.add(len(points) - 1)
        for point, co in zip(spline.points, points):
            point.co = (co[0], co[1], co[2], 1.0)
        return True

    segments_by_lm: dict[str, list[tuple[int, int, tuple[float, float, float], tuple[float, float, float]]]] = defaultdict(list)
    with args.segments_csv.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            try:
                landmark = row["landmark_name"]
                frame_start_row = int(float(row["frame_start"]))
                frame_end_row = int(float(row["frame_end"]))
                p1 = mapped_location(frame_start_row, landmark, float(row["x1"]), float(row["y1"]), float(row["z1"]))
                p2 = mapped_location(frame_end_row, landmark, float(row["x2"]), float(row["y2"]), float(row["z2"]))
            except Exception:
                continue
            segments_by_lm[landmark].append((frame_start_row, frame_end_row, p1, p2))

    overview_count = 0
    progress_count = 0
    for landmark, segments in sorted(segments_by_lm.items()):
        segments.sort(key=lambda item: (item[0], item[1]))
        group = landmark_group(landmark)
        overview_curve = bpy.data.curves.new(f"MPLR_TRAJ_{landmark}", "CURVE")
        overview_curve.dimensions = "3D"
        overview_curve.resolution_u = 2
        overview_curve.bevel_depth = 0.0038
        overview_curve.bevel_resolution = 1
        paths: list[tuple[int, int, list[tuple[float, float, float]]]] = []
        current: list[tuple[float, float, float]] = []
        start = None
        end = None
        prev_end = None
        for start_frame_row, end_frame_row, p1, p2 in segments:
            if prev_end is not None and start_frame_row == prev_end:
                current.append(p2)
                end = end_frame_row
            else:
                if len(current) >= 2 and start is not None and end is not None:
                    paths.append((start, end, list(current)))
                    add_polyline(overview_curve, current)
                current = [p1, p2]
                start = start_frame_row
                end = end_frame_row
            prev_end = end_frame_row
        if len(current) >= 2 and start is not None and end is not None:
            paths.append((start, end, list(current)))
            add_polyline(overview_curve, current)
        overview_obj = bpy.data.objects.new(f"MPLR_TRAJ_{landmark}", overview_curve)
        overview_obj.data.materials.append(
            lr_overview_materials.get(
                landmark_side(landmark),
                overview_materials.get(group, overview_materials["other"]),
            )
        )
        overview_obj["mplr_overview_curve"] = True
        curves_col.objects.link(overview_obj)
        overview_count += 1

        for index, (path_start, path_end, points) in enumerate(paths):
            curve = bpy.data.curves.new(f"MPLR_DRAWPATH_{landmark}_{index:03d}", "CURVE")
            curve.dimensions = "3D"
            curve.resolution_u = 2
            curve.bevel_depth = 0.0075
            curve.bevel_resolution = 1
            curve.bevel_factor_start = 0.0
            curve.bevel_factor_end = 0.0
            try:
                curve.bevel_factor_mapping = "RESOLUTION"
            except Exception:
                pass
            add_polyline(curve, points)
            obj = bpy.data.objects.new(f"MPLR_DRAWPATH_{landmark}_{index:03d}", curve)
            obj.data.materials.append(
                lr_progress_materials.get(
                    landmark_side(landmark),
                    progress_materials.get(group, progress_materials["other"]),
                )
            )
            obj["mplr_progress_curve"] = True
            obj["mplr_start_frame"] = int(path_start)
            obj["mplr_end_frame"] = int(path_end)
            obj.hide_viewport = True
            obj.hide_render = True
            progress_col.objects.link(obj)
            progress_count += 1

    for landmark, records in sorted(rows_by_lm.items()):
        group = landmark_group(landmark)
        verts = [
            mapped_location(frame, landmark, x, raw_y, z)
            for frame, _, x, raw_y, z, _, _, _ in records
        ]
        mesh = bpy.data.meshes.new(f"MPLR_POINTS_{landmark}")
        mesh.from_pydata(verts, [], [])
        mesh.update()
        obj = bpy.data.objects.new(f"MPLR_POINTS_{landmark}", mesh)
        obj.data.materials.append(point_materials.get(group, point_materials["other"]))
        obj.hide_viewport = True
        obj.hide_render = True
        points_col.objects.link(obj)

    def octa_mesh(name: str, radius: float) -> Any:
        mesh = bpy.data.meshes.get(name)
        if mesh:
            return mesh
        verts = [
            (0, 0, radius),
            (radius, 0, 0),
            (0, radius, 0),
            (-radius, 0, 0),
            (0, -radius, 0),
            (0, 0, -radius),
        ]
        faces = [
            (0, 1, 2),
            (0, 2, 3),
            (0, 3, 4),
            (0, 4, 1),
            (5, 2, 1),
            (5, 3, 2),
            (5, 4, 3),
            (5, 1, 4),
        ]
        mesh = bpy.data.meshes.new(name)
        mesh.from_pydata(verts, [], faces)
        mesh.update()
        return mesh

    def sphere_mesh(name: str, radius: float) -> Any:
        mesh = bpy.data.meshes.get(name)
        if mesh:
            return mesh
        bpy.ops.mesh.primitive_uv_sphere_add(
            segments=24,
            ring_count=12,
            radius=radius,
            location=(0, 0, 0),
        )
        tmp = bpy.context.object
        mesh = tmp.data
        mesh.name = name
        bpy.data.objects.remove(tmp, do_unlink=True)
        return mesh

    def marker_radius(group: str) -> float:
        if group in {"feet", "hands_proxy"}:
            return args.marker_radius * 1.15
        if group in {"torso", "head_proxy"}:
            return args.marker_radius * 1.08
        return args.marker_radius

    def halo_radius(group: str) -> float:
        if group in {"feet", "hands_proxy"}:
            return args.halo_radius * 1.16
        if group in {"torso", "head_proxy"}:
            return args.halo_radius * 1.08
        return args.halo_radius

    core_meshes: dict[str, Any] = {}
    halo_meshes: dict[str, Any] = {}

    def get_core_mesh(group: str) -> Any:
        if group not in core_meshes:
            core_meshes[group] = octa_mesh(
                f"MPLR_AUTO_MARKER_CORE_{group}",
                marker_radius(group),
            )
        return core_meshes[group]

    def get_halo_mesh(group: str) -> Any:
        if group not in halo_meshes:
            halo_meshes[group] = sphere_mesh(
                f"MPLR_AUTO_GLOW_SPHERE_{group}",
                halo_radius(group),
            )
        return halo_meshes[group]

    def marker_visibility(obj: Any, frames: list[int]) -> None:
        frames = sorted(set(frames))
        if not frames:
            return
        events: dict[int, bool] = {}

        def add_event(frame: int, hidden: bool) -> None:
            if frame_start <= frame <= frame_end:
                events[int(frame)] = bool(hidden)

        first = frames[0]
        last = frames[-1]
        add_event(frame_start, first > frame_start)
        add_event(first, False)
        prev = first
        for frame in frames[1:]:
            if frame - prev > 1:
                add_event(prev, False)
                add_event(prev + 1, True)
                add_event(frame - 1, True)
                add_event(frame, False)
            prev = frame
        if last < frame_end:
            add_event(last, False)
            add_event(last + 1, True)
            add_event(frame_end, True)
        for frame, hidden in sorted(events.items()):
            obj.hide_viewport = hidden
            obj.hide_render = hidden
            obj.keyframe_insert(data_path="hide_viewport", frame=frame)
            obj.keyframe_insert(data_path="hide_render", frame=frame)

    marker_count = 0
    halo_count = 0
    for landmark, records in sorted(rows_by_lm.items()):
        if landmark in face_marker_names_to_hide:
            continue
        records.sort(key=lambda item: item[0])
        group = landmark_group(landmark)
        first = records[0]
        obj = bpy.data.objects.new(f"MPLR_ANIM_{landmark}", get_core_mesh(group))
        obj.location = mapped_location(first[0], landmark, first[2], first[3], first[4])
        object_material(
            obj,
            lr_marker_materials.get(
                landmark_side(landmark),
                marker_materials.get(group, marker_materials["other"]),
            ),
        )
        obj.show_in_front = True
        obj["mplr_landmark_name"] = landmark
        obj["mplr_landmark_group"] = group
        anim_col.objects.link(obj)
        for frame, _, x, raw_y, z, _, _, _ in records:
            obj.location = mapped_location(frame, landmark, x, raw_y, z)
            obj.keyframe_insert(data_path="location", frame=frame)
        marker_visibility(obj, [record[0] for record in records])
        if obj.animation_data and obj.animation_data.action:
            try:
                for fcurve in obj.animation_data.action.fcurves:
                    for keyframe in fcurve.keyframe_points:
                        keyframe.interpolation = (
                            "CONSTANT"
                            if fcurve.data_path in {"hide_viewport", "hide_render"}
                            else "LINEAR"
                        )
            except Exception:
                pass
        halo = bpy.data.objects.new(f"MPLR_HALO_{landmark}", get_halo_mesh(group))
        halo.parent = obj
        halo.location = (0.0, 0.0, 0.0)
        object_material(
            halo,
            lr_halo_materials.get(
                landmark_side(landmark),
                halo_materials.get(group, halo_materials["other"]),
            ),
        )
        halo.show_in_front = True
        halo_col.objects.link(halo)
        marker_count += 1
        halo_count += 1

    def mplr_set_playback_draw_state(playing: bool) -> None:
        for obj in bpy.data.objects:
            if obj.get("mplr_overview_curve"):
                obj.hide_viewport = bool(playing)
                obj.hide_render = bool(playing)
            elif obj.get("mplr_progress_curve"):
                obj.hide_viewport = not bool(playing)
                obj.hide_render = not bool(playing)

    def mplr_update_progress_factors(scene_arg: Any) -> None:
        frame = float(scene_arg.frame_current)
        for obj in bpy.data.objects:
            if not obj.get("mplr_progress_curve"):
                continue
            start_frame_value = float(obj.get("mplr_start_frame", 0.0))
            end_frame_value = float(obj.get("mplr_end_frame", start_frame_value + 1.0))
            if frame <= start_frame_value:
                factor = 0.0
            elif frame >= end_frame_value:
                factor = 1.0
            else:
                factor = (frame - start_frame_value) / (
                    end_frame_value - start_frame_value
                )
            obj.data.bevel_factor_start = 0.0
            obj.data.bevel_factor_end = factor

    def mplr_playback_pre(scene_arg: Any) -> None:
        mplr_set_playback_draw_state(True)
        mplr_update_progress_factors(scene_arg)

    def mplr_playback_post(scene_arg: Any) -> None:
        mplr_set_playback_draw_state(False)
        mplr_update_progress_factors(scene_arg)

    def mplr_frame_change_post(scene_arg: Any) -> None:
        try:
            playing = any(
                getattr(screen, "is_animation_playing", False)
                for screen in bpy.data.screens
            )
        except Exception:
            playing = scene_arg.frame_current > 0
        mplr_set_playback_draw_state(playing)
        mplr_update_progress_factors(scene_arg)

    for handler_list in (
        bpy.app.handlers.animation_playback_pre,
        bpy.app.handlers.animation_playback_post,
        bpy.app.handlers.frame_change_post,
    ):
        for handler in list(handler_list):
            if getattr(handler, "__name__", "").startswith("mplr_"):
                handler_list.remove(handler)
    bpy.app.handlers.animation_playback_pre.append(mplr_playback_pre)
    bpy.app.handlers.animation_playback_post.append(mplr_playback_post)
    bpy.app.handlers.frame_change_post.append(mplr_frame_change_post)

    font = bpy.data.curves.new("MPLR_CAMERA_VIEW_SUMMARY", "FONT")
    font.body = (
        f"MPLR {session_id}\\n"
        f"points: {sum(len(records) for records in rows_by_lm.values()):,}\\n"
        f"curves: {overview_count} / draw paths: {progress_count}\\n"
        "origin: screen bottom center\\n"
        f"scale: X {args.x_factor:.2f}, Y {args.y_factor:.2f}"
    )
    font.size = args.camera_summary_size
    font.align_x = "LEFT"
    font.align_y = "BOTTOM"
    text = bpy.data.objects.new("MPLR_CAMERA_VIEW_SUMMARY", font)
    text["mplr_optional_camera_summary"] = True
    text["mplr_note"] = (
        "Hidden by default. Use --show-camera-summary or enable visibility "
        "when a camera-view summary is needed."
    )
    summary_material = make_material(
        "MPLR_CAMERA_SUMMARY_MAT",
        (1.0, 0.58, 0.12, 0.92),
        emission=True,
        strength=1.4,
        alpha=0.92,
    )
    text.data.materials.append(summary_material)
    camera = scene.camera or bpy.data.objects.get("Camera")
    if camera is not None:
        text.parent = camera
        text.location = args.camera_summary_location
        text.rotation_euler = (0.0, 0.0, 0.0)
    else:
        text.location = (-3.0, -0.15, 5.35)
    text.hide_viewport = not args.show_camera_summary
    text.hide_render = not args.show_camera_summary
    helpers_col.objects.link(text)

    guide_material = make_material(
        "MPLR_AUTO_DEPTH_GUIDE_MAT",
        (0.1, 0.6, 1.0, 0.9),
        emission=True,
        strength=3.0,
        alpha=0.9,
    )
    guide = bpy.data.curves.new("MPLR_AUTO_DEPTH_GUIDE", "CURVE")
    guide.dimensions = "3D"
    guide.bevel_depth = 0.012
    add_polyline(guide, [(0.0, 0.0, 0.0), (0.0, args.depth_meters * args.y_factor, 0.0)])
    guide_obj = bpy.data.objects.new("MPLR_AUTO_DEPTH_GUIDE", guide)
    guide_obj.data.materials.append(guide_material)
    helpers_col.objects.link(guide_obj)

    scene.frame_set(0)
    mplr_set_playback_draw_state(False)
    mplr_update_progress_factors(scene)

    if args.save_blend and not args.no_save_blend:
        args.save_blend.parent.mkdir(parents=True, exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=str(args.save_blend))

    print(
        json.dumps(
            {
                "status": "imported_blender_trajectory",
                "session_id": session_id,
                "fresh_startup_scene": True,
                "startup_cube_removed": startup_cube_removed,
                "removed_previous_objects": removed,
                "points_rows": sum(len(records) for records in rows_by_lm.values()),
                "overview_curve_objects": overview_count,
                "progressive_curve_objects": progress_count,
                "animated_markers": marker_count,
                "halos": halo_count,
                "hidden_metadata_labels": hidden_metadata_labels,
                "camera_summary_object": text.name,
                "camera_summary_visible": args.show_camera_summary,
                "left_right_visualization": {
                    "left": "orange",
                    "right": "cyan",
                    "center": "white",
                },
                "face_marker_policy": {
                    "nose": "visible_white_center_marker",
                    "left_eye": "visible_left_marker_size",
                    "right_eye": "visible_right_marker_size",
                    "hidden": sorted(face_marker_names_to_hide),
                },
                "x_factor": args.x_factor,
                "y_factor": args.y_factor,
                "depth_range_m": [0.0, args.depth_meters * args.y_factor],
                "camera_location": args.camera_location,
                "camera_rotation_degrees": args.camera_rotation,
                "save_blend": str(args.save_blend) if args.save_blend else None,
                "points_csv": str(args.points_csv),
                "segments_csv": str(args.segments_csv),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def main(argv: list[str] | None = None) -> int:
    if argv is None and "--" in sys.argv:
        argv = sys.argv[sys.argv.index("--") + 1 :]
    parser = build_parser()
    args, unknown = parser.parse_known_args(argv)
    resolve_paths(args)
    if not args.inside_blender and not running_inside_blender():
        return launch_blender(args)
    import_in_blender(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
