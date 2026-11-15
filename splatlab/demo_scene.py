"""Procedural demo scene: renders a small 'toy courtyard' with known cameras.

This doubles as (a) a zero-setup demo and (b) a synthetic test range for the
pose estimator / splat trainer.  No external assets are required.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .capture import imwrite_unicode
from .geometry import look_at_c2w, rotation_to_quaternion


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-9)


def _noise(p: np.ndarray, seed: float = 0.0) -> np.ndarray:
    """Cheap deterministic procedural noise in [0, 1]."""
    a = np.sin(p[..., 0] * 12.9898 + p[..., 1] * 78.233 + p[..., 2] * 37.719 + seed) * 43758.5453
    b = np.sin(p[..., 0] * 43.332 + p[..., 1] * 19.117 + p[..., 2] * 71.531 + seed * 1.7) * 24634.6345
    na = a - np.floor(a)
    nb = b - np.floor(b)
    return 0.6 * na + 0.4 * nb


def _checker(p: np.ndarray, scale: float) -> np.ndarray:
    x = np.floor(p[..., 0] * scale).astype(np.int32)
    z = np.floor(p[..., 2] * scale).astype(np.int32)
    return ((x + z) & 1).astype(np.float32)


def _mix(a, b, t):
    t = np.asarray(t)[..., None] if np.ndim(t) == a.ndim - 1 else t
    return a * (1.0 - t) + b * t


def _tone(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.power(x, 1.0 / 2.2)


# --------------------------------------------------------------------------- #
# scene description
# --------------------------------------------------------------------------- #

BOXES = [
    # lo, hi, base colour, stripe colour, stripe frequency
    ((-0.72, 0.0, -0.58), (0.72, 0.72, 0.58), (0.86, 0.30, 0.24), (0.98, 0.86, 0.62), 6.0),
    ((-2.15, 0.0, 0.55), (-1.55, 1.55, 1.15), (0.20, 0.48, 0.82), (0.92, 0.94, 0.98), 4.0),
    ((1.35, 0.0, 0.85), (1.85, 1.10, 1.35), (0.95, 0.66, 0.20), (0.25, 0.28, 0.32), 5.0),
    ((1.05, 0.0, -1.65), (1.65, 0.55, -1.05), (0.40, 0.72, 0.45), (0.96, 0.95, 0.86), 3.0),
]

SPHERES = [
    ((0.0, 1.15, 0.0), 0.42, (0.98, 0.80, 0.30)),
    ((-1.85, 1.75, 0.85), 0.20, (0.90, 0.92, 0.96)),
    ((0.55, 0.13, 1.65), 0.13, (0.85, 0.42, 0.55)),
    ((-0.95, 0.10, 1.95), 0.10, (0.55, 0.82, 0.88)),
    ((2.15, 0.09, 1.85), 0.09, (0.94, 0.90, 0.80)),
]

GROUND_RADIUS = 3.35
LIGHT_DIR = _normalize(np.array([-0.45, 0.82, 0.36]))


def _shade(albedo: np.ndarray, normal: np.ndarray, view_to_camera: np.ndarray,
           hit: np.ndarray, specular: float = 0.08) -> np.ndarray:
    ndl = np.clip((normal * LIGHT_DIR).sum(-1), 0.0, 1.0)
    half = _normalize(LIGHT_DIR + view_to_camera)
    ndh = np.clip((normal * half).sum(-1), 0.0, 1.0)
    diffuse = 0.38 + 0.78 * ndl
    spec = specular * np.power(ndh, 42.0)
    # Tiny procedural grain keeps SIFT interested even on flat sides.
    grain = 0.055 * (_noise(hit * 38.0, 3.3) - 0.5)
    return np.clip(albedo * (diffuse[..., None] + grain[..., None]) + spec[..., None], 0.0, 1.0)


def _ground_albedo(p: np.ndarray) -> np.ndarray:
    par = _checker(p, 2.0)[..., None]
    teal = np.array([0.16, 0.52, 0.50], dtype=np.float32)
    cream = np.array([0.92, 0.87, 0.73], dtype=np.float32)
    albedo = teal * (1 - par) + cream * par

    # Asymmetric accent tiles make matching unambiguous.
    ix = np.floor(p[..., 0] * 2.0).astype(np.int32)
    iz = np.floor(p[..., 2] * 2.0).astype(np.int32)
    accent = ((ix * 17 + iz * 31) % 11 == 0)
    albedo = np.where(accent[..., None], np.array([0.85, 0.45, 0.25], np.float32), albedo)
    accent2 = ((ix * 7 - iz * 13) % 17 == 0)
    albedo = np.where(accent2[..., None], np.array([0.24, 0.32, 0.58], np.float32), albedo)

    grain = (_noise(p * 55.0, 9.1) - 0.5)[..., None] * 0.09
    albedo = np.clip(albedo + grain, 0.0, 1.0)
    return albedo


def _box_albedo(p: np.ndarray, normal: np.ndarray, base: np.ndarray, stripe: np.ndarray,
                freq: float) -> np.ndarray:
    # Stripes based on the dominant horizontal axis of the face.
    horiz = np.where(np.abs(normal[..., 1]) > 0.5, p[..., 0] + p[..., 2], p[..., 2] - p[..., 0])
    band = (np.sin(horiz * freq * np.pi) > 0.25)[..., None].astype(np.float32)
    albedo = base * (1 - band) + stripe * band
    grain = (_noise(p * 44.0, 5.7) - 0.5)[..., None] * 0.08
    return np.clip(albedo + grain, 0.0, 1.0)


def _sphere_albedo(p: np.ndarray, centre: np.ndarray, radius: float, base: np.ndarray) -> np.ndarray:
    n = (p - centre) / max(radius, 1e-6)
    bands = (np.sin(n[..., 1] * 9.0) > 0.2)[..., None].astype(np.float32)
    light = base * 0.35 + 0.95
    albedo = base * (1 - bands) + np.clip(base * 0.55 + 0.35, 0, 1) * bands
    grain = (_noise(p * 60.0, 2.2) - 0.5)[..., None] * 0.05
    return np.clip(albedo * (0.75 + 0.25 * light) + grain, 0.0, 1.0)


def _intersect_disk(origin: np.ndarray, direction: np.ndarray, radius: float):
    t = -origin[..., 1] / np.where(np.abs(direction[..., 1]) < 1e-8, -1e-8, direction[..., 1])
    p = origin + direction * t[..., None]
    hit = (t > 0) & ((p[..., 0] ** 2 + p[..., 2] ** 2) <= radius ** 2)
    normal = np.zeros_like(direction)
    normal[..., 1] = 1.0
    return hit, t, p, normal


def _intersect_sphere(origin: np.ndarray, direction: np.ndarray, centre: np.ndarray, radius: float):
    oc = origin - centre
    b = 2.0 * (direction * oc).sum(-1)
    c = (oc * oc).sum(-1) - radius * radius
    disc = b * b - 4.0 * c
    ok = disc > 0
    sq = np.sqrt(np.maximum(disc, 0.0))
    t0 = (-b - sq) / 2.0
    t1 = (-b + sq) / 2.0
    t = np.where(t0 > 1e-5, t0, t1)
    hit = ok & (t > 1e-5)
    p = origin + direction * np.where(hit, t, 0.0)[..., None]
    normal = _normalize(p - centre)
    return hit, t, p, normal


def _intersect_box(origin: np.ndarray, direction: np.ndarray, lo: np.ndarray, hi: np.ndarray):
    inv = 1.0 / np.where(np.abs(direction) < 1e-9, 1e-9, direction)
    t0 = (lo - origin) * inv
    t1 = (hi - origin) * inv
    tmin = np.minimum(t0, t1)
    tmax = np.maximum(t0, t1)
    near = tmin.max(-1)
    far = tmax.min(-1)
    hit = (far >= np.maximum(near, 0.0)) & (near > 1e-5)
    axis = np.argmax(tmin, axis=-1)
    p = origin + direction * np.where(hit, near, 0.0)[..., None]
    normal = np.zeros_like(direction)
    for k in range(3):
        sel = axis == k
        normal[..., k] = np.where(direction[..., k] > 0, -1.0, 1.0) * sel
    return hit, near, p, normal


def render_depth(camera_R: np.ndarray, camera_t: np.ndarray, width: int, height: int,
                 fx: float, fy: float, cx: float, cy: float,
                 sky_depth: float = 22.0) -> np.ndarray:
    """Analytic z-depth of the toy scene for known cameras.

    Used by the bundled synthetic demo to show what a perfectly aligned depth
    prior looks like; real captures keep using Depth Anything V2.
    """
    xs = (np.arange(width, dtype=np.float32) + 0.5 - cx) / fx
    ys = (np.arange(height, dtype=np.float32) + 0.5 - cy) / fy
    grid_x, grid_y = np.meshgrid(xs, ys)
    dir_cam = np.stack([grid_x, grid_y, np.ones_like(grid_x)], axis=-1)
    dir_world = dir_cam @ camera_R.T
    length = np.linalg.norm(dir_world, axis=-1, keepdims=True)
    dir_world = dir_world / np.maximum(length, 1e-9)
    origin = np.broadcast_to(camera_t.astype(np.float32), dir_world.shape).copy()

    t_best = np.full((height, width), np.inf, dtype=np.float32)
    hit, t, _p, _n = _intersect_disk(origin, dir_world, GROUND_RADIUS)
    t_best = np.where(hit & (t < t_best), t, t_best)
    for lo, hi, *_rest in BOXES:
        hit, t, _p, _n = _intersect_box(origin, dir_world,
                                        np.array(lo, np.float32), np.array(hi, np.float32))
        t_best = np.where(hit & (t < t_best), t, t_best)
    for centre, radius, _colour in SPHERES:
        hit, t, _p, _n = _intersect_sphere(origin, dir_world,
                                           np.array(centre, np.float32), float(radius))
        t_best = np.where(hit & (t < t_best), t, t_best)

    # Perspective z (not Euclidean distance) because the pipeline unprojects z.
    z_scale = (1.0 / np.maximum(length[..., 0], 1e-9)).astype(np.float32)
    depth = np.where(np.isfinite(t_best), t_best * z_scale, sky_depth)
    return np.clip(depth, 0.05, sky_depth).astype(np.float32)


def render_view(camera_R: np.ndarray, camera_t: np.ndarray, width: int, height: int,
                fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    xs = (np.arange(width, dtype=np.float32) + 0.5 - cx) / fx
    ys = (np.arange(height, dtype=np.float32) + 0.5 - cy) / fy
    grid_x, grid_y = np.meshgrid(xs, ys)
    dir_cam = np.stack([grid_x, grid_y, np.ones_like(grid_x)], axis=-1)
    dir_world = dir_cam @ camera_R.T
    dir_world = _normalize(dir_world)
    origin = np.broadcast_to(camera_t.astype(np.float32), dir_world.shape).copy()

    t_best = np.full((height, width), np.inf, dtype=np.float32)
    prim = np.full((height, width), -1, dtype=np.int32)
    point = np.zeros_like(dir_world)
    normal = np.zeros_like(dir_world)
    albedo = np.zeros_like(dir_world)

    # Ground.
    hit, t, p, n = _intersect_disk(origin, dir_world, GROUND_RADIUS)
    mask = hit & (t < t_best)
    t_best[mask] = t[mask]
    prim[mask] = 0
    point[mask] = p[mask]
    normal[mask] = n[mask]
    albedo[mask] = _ground_albedo(p)[mask]

    pid = 1
    for lo, hi, base, stripe, freq in BOXES:
        lo_v, hi_v = np.array(lo, np.float32), np.array(hi, np.float32)
        hit, t, p, n = _intersect_box(origin, dir_world, lo_v, hi_v)
        mask = hit & (t < t_best)
        t_best[mask] = t[mask]
        prim[mask] = pid
        point[mask] = p[mask]
        normal[mask] = n[mask]
        albedo[mask] = _box_albedo(
            p, n, np.array(base, np.float32), np.array(stripe, np.float32), freq
        )[mask]
        pid += 1

    for centre, radius, colour in SPHERES:
        c = np.array(centre, np.float32)
        hit, t, p, n = _intersect_sphere(origin, dir_world, c, float(radius))
        mask = hit & (t < t_best)
        t_best[mask] = t[mask]
        prim[mask] = pid
        point[mask] = p[mask]
        normal[mask] = n[mask]
        albedo[mask] = _sphere_albedo(p, c, float(radius), np.array(colour, np.float32))[mask]
        pid += 1

    sky = np.clip(0.5 + 0.5 * dir_world[..., 1], 0.0, 1.0)
    horizon = np.array([0.72, 0.80, 0.84], np.float32)
    zenith = np.array([0.14, 0.35, 0.58], np.float32)
    sky_col = horizon * (1 - sky[..., None]) + zenith * sky[..., None]
    sun = np.clip((dir_world * LIGHT_DIR).sum(-1), 0.0, 1.0)
    sky_col = np.clip(sky_col + 0.22 * np.power(sun, 18.0)[..., None], 0.0, 1.0)
    # soft cloud bands
    cloud = np.zeros_like(sky)
    band = np.exp(-((dir_world[..., 1] - 0.33) / 0.09) ** 2)
    ripple = 0.5 + 0.5 * np.sin(dir_world[..., 0] * 18.0 + dir_world[..., 2] * 7.0)
    cloud = band * ripple * 0.16
    sky_col = np.clip(sky_col * (1 - cloud[..., None]) + cloud[..., None], 0.0, 1.0)

    cam_forwardish = dir_world * -1.0
    lit = _shade(albedo, normal, cam_forwardish, point, specular=0.10)
    lit = np.where((normal[..., 1] > 0.5)[..., None] & (prim[..., None] == 0), lit * 0.92, lit)

    # Soft contact shadows on the ground.
    ground = (prim == 0) & (normal[..., 1] > 0.5)
    shadow = np.zeros((height, width), np.float32)
    for (lo, hi, *_rest) in BOXES:
        c = np.array([(lo[0] + hi[0]) * 0.5, 0.0, (lo[2] + hi[2]) * 0.5], np.float32)
        rx = max(0.45, (hi[0] - lo[0]) * 0.75)
        rz = max(0.45, (hi[2] - lo[2]) * 0.75)
        d = ((point[..., 0] - c[0]) / rx) ** 2 + ((point[..., 2] - c[2]) / rz) ** 2
        shadow = np.maximum(shadow, np.exp(-d * 2.2) * 0.38)
    for centre, radius, _c in SPHERES:
        if centre[1] > 0.5:
            d = ((point[..., 0] - centre[0]) / (radius * 1.6)) ** 2 + \
                ((point[..., 2] - centre[2]) / (radius * 1.6)) ** 2
            shadow = np.maximum(shadow, np.exp(-d * 1.7) * 0.30)
    lit = np.where(ground[..., None], lit * (1.0 - shadow[..., None]), lit)

    # Distance haze.
    haze = np.clip((t_best - 6.0) / 10.0, 0.0, 1.0)
    colour = lit * (1 - haze[..., None]) + sky_col * haze[..., None]
    colour = np.where((prim < 0)[..., None], sky_col, colour)
    return _tone(colour)


def render_demo(out_dir: Path, n_views: int = 24, width: int = 640, height: int = 480,
                seed: int = 7) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    fov = np.deg2rad(46.0)
    fx = 0.5 * width / np.tan(fov / 2.0)
    fy = fx
    cx, cy = width / 2.0, height / 2.0

    cameras = []
    depth_dir = out_dir / "depth"
    depth_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n_views):
        angle = 2.0 * np.pi * i / n_views + rng.uniform(-0.03, 0.03)
        radius = 5.1 + rng.uniform(-0.25, 0.25)
        eye = np.array([np.cos(angle) * radius, 1.95 + rng.uniform(-0.25, 0.25),
                        np.sin(angle) * radius], dtype=np.float64)
        target = np.array([0.0, 0.98, 0.0], dtype=np.float64)
        R = look_at_c2w(eye, target)
        image = render_view(R, eye, width, height, fx, fy, cx, cy)
        name = f"demo_{i:03d}.jpg"
        imwrite_unicode(out_dir / name, (np.clip(image, 0, 1) * 255).astype(np.uint8)[..., ::-1],
                        quality=94)
        depth = render_depth(R, eye, width, height, fx, fy, cx, cy)
        depth_name = f"depth/demo_{i:03d}.npy"
        np.save(depth_dir / f"demo_{i:03d}.npy", depth.astype(np.float16))
        q = rotation_to_quaternion(R)
        cameras.append({
            "name": name,
            "width": width,
            "height": height,
            "fx": float(fx), "fy": float(fy), "cx": float(cx), "cy": float(cy),
            "R": R.tolist(),
            "t": eye.tolist(),
            "q": q.tolist(),
            "depth": depth_name,
        })

    meta = {
        "name": "Toy Courtyard",
        "width": width,
        "height": height,
        "fov_deg": 46.0,
        "cameras": cameras,
        "note": "Synthetic demo scene rendered by splatlab.demo_scene",
    }
    (out_dir / "ground_truth.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


if __name__ == "__main__":
    import sys
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("assets/demo/source")
    render_demo(target)
    print(f"Rendered demo images to {target}")
