"""Camera pose estimation, up-axis alignment and scene normalisation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import colmap as colmap_mod
from . import sfm as sfm_mod
from .config import Preset
from .types import Camera, Scene


class PoseError(RuntimeError):
    pass


def _rotation_from_to(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / max(np.linalg.norm(a), 1e-12)
    b = b / max(np.linalg.norm(b), 1e-12)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-8:
        if c > 0:
            return np.eye(3)
        axis = np.array([1.0, 0.0, 0.0])
        v = axis
        return _axis_angle(v, np.pi)
    s = np.linalg.norm(v)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))


def _axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / max(np.linalg.norm(axis), 1e-12)
    x, y, z = axis
    c, s = np.cos(angle), np.sin(angle)
    C = 1 - c
    return np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ])


def align_up(scene: Scene) -> Scene:
    """Rotate the reconstruction so the average camera-up becomes +Y."""
    if not scene.cameras:
        return scene
    ups = np.mean([-c.R[:, 1] for c in scene.cameras], axis=0)
    if np.linalg.norm(ups) < 0.15:
        return scene
    up = ups / np.linalg.norm(ups)
    target = np.array([0.0, 1.0, 0.0])
    if np.dot(up, target) > 0.9995:
        return scene
    Rw = _rotation_from_to(up, target)
    for cam in scene.cameras:
        cam.R = Rw @ cam.R
        cam.t = Rw @ cam.t
    if len(scene.points):
        scene.points = (scene.points @ Rw.T).astype(np.float32)
    return scene


def normalize_scene(scene: Scene, target_radius: float = 2.0) -> Scene:
    """Center and uniformly rescale the world for stable optimisation/rendering."""
    centers = np.array([c.position for c in scene.cameras], dtype=np.float64)
    center = np.median(centers, axis=0)
    parts = [np.linalg.norm(centers - center, axis=1)]
    if len(scene.points):
        # Keep camera + point distribution balanced.
        sample = scene.points[::max(1, len(scene.points) // 20000)]
        parts.append(np.linalg.norm(sample - center, axis=1) * 1.1)
    dists = np.concatenate(parts)
    radius = float(np.percentile(dists, 92))
    scale = target_radius / max(radius, 1e-6)
    scale = float(np.clip(scale, 1e-4, 1e4))
    for cam in scene.cameras:
        cam.t = (cam.t - center) * scale
    if len(scene.points):
        scene.points = ((scene.points - center) * scale).astype(np.float32)
    # Depths sampled in the original world frame must be rescaled to match the
    # normalised camera positions (see pipeline bundled-depth handling).
    scene.scale *= scale
    return scene


def _authored_scene(images_dir: Path, log=print) -> Scene | None:
    """Load a ``ground_truth.json`` camera path bundled next to the source images.

    Synthetic demos and calibrated captures ship with known cameras; using them
    removes any pose noise from the showcase pipeline.
    """
    meta_path = images_dir / "frames_meta.json"
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    gt_cache: dict[Path, dict] = {}
    cameras: list[Camera] = []
    for rec in meta.get("frames", []):
        source = Path(rec.get("source", ""))
        gt_path = source.parent / "ground_truth.json"
        if not source.name or not gt_path.exists():
            continue
        if gt_path not in gt_cache:
            try:
                gt_cache[gt_path] = json.loads(gt_path.read_text(encoding="utf-8"))
            except Exception:
                gt_cache[gt_path] = {}
        by_name = {c.get("name"): c for c in gt_cache[gt_path].get("cameras", [])}
        c = by_name.get(source.name)
        if c is None:
            continue
        try:
            out_w = int(rec.get("width") or c["width"])
            out_h = int(rec.get("height") or c["height"])
            sx = out_w / float(c["width"])
            sy = out_h / float(c["height"])
            depth_path = ""
            if c.get("depth"):
                candidate = (source.parent / c["depth"]).resolve()
                if candidate.exists():
                    depth_path = str(candidate)
            cameras.append(Camera(
                name=rec["output"], width=out_w, height=out_h,
                fx=float(c["fx"]) * sx, fy=float(c["fy"]) * sy,
                cx=float(c["cx"]) * sx, cy=float(c["cy"]) * sy,
                R=np.asarray(c["R"], np.float64), t=np.asarray(c["t"], np.float64),
                image_path=str(images_dir / rec["output"]),
                depth_path=depth_path,
            ))
        except (KeyError, TypeError, ValueError):
            continue

    if len(cameras) >= 3:
        log(f"Using authored camera path ({len(cameras)} calibrated frames).")
        return Scene(cameras=cameras, source="authored")
    return None


def estimate(
    images_dir: Path,
    workspace: Path,
    preset: Preset,
    backend: str = "auto",
    allow_colmap_download: bool = False,
    log=print,
    progress=None,
    cancel_event=None,
) -> Scene:
    """Estimate camera poses with calibrated data, COLMAP, or built-in SfM."""
    images_dir = Path(images_dir)
    scene: Scene | None = None

    focal: float | None = None
    meta_path = images_dir / "frames_meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            focal = meta.get("focal_px")
        except Exception:
            pass

    if backend in ("auto", "authored"):
        scene = _authored_scene(images_dir, log=log)

    want_colmap = backend in ("auto", "colmap")
    exe = colmap_mod.find_colmap() if want_colmap and scene is None else None
    if exe is not None:
        try:
            log("Estimating camera poses with COLMAP ...")
            scene = colmap_mod.reconstruct(
                images_dir, workspace / "colmap", log=log, progress=progress,
                cancel_event=cancel_event,
            )
        except Exception as exc:
            if backend == "colmap":
                raise PoseError(f"COLMAP failed: {exc}") from exc
            log(f"COLMAP unavailable/failed ({str(exc).splitlines()[0]}); using built-in SfM.")
    elif backend == "colmap" and allow_colmap_download and scene is None:
        try:
            colmap_mod.install_colmap(
                progress=(lambda p, m: progress(0.02 + 0.1 * p, m)) if progress else None, log=log
            )
            scene = colmap_mod.reconstruct(
                images_dir, workspace / "colmap", log=log, progress=progress,
                cancel_event=cancel_event,
            )
        except Exception as exc:
            raise PoseError(f"COLMAP failed: {exc}") from exc

    if scene is None:
        log("Estimating camera poses with built-in SfM ...")
        try:
            scene = sfm_mod.reconstruct(
                images_dir, workspace / "sfm", focal=focal, log=log, progress=progress,
                cancel_event=cancel_event,
            )
        except Exception as exc:
            raise PoseError(f"Pose estimation failed: {exc}") from exc

    if len(scene.cameras) < 3:
        raise PoseError(
            f"Only {len(scene.cameras)} of the frames could be registered. "
            "Move the camera more slowly around a textured scene, or install COLMAP."
        )

    scene = align_up(scene)
    scene = normalize_scene(scene)
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "poses_aligned.json").write_text(
        json.dumps({
            "source": scene.source,
            "cameras": [c.to_dict() for c in scene.cameras],
            "points": len(scene.points),
        }, indent=2),
        encoding="utf-8",
    )
    if progress:
        progress(0.34, f"Poses ready ({len(scene.cameras)} cameras)")
    log(f"Pose estimation complete: {len(scene.cameras)} cameras, {len(scene.points):,} points.")
    return scene
