"""Export splats to standard formats for the web viewer and other tools."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .types import Camera

C0 = 0.28209479177387814
PLY_PROPS = (
    ["x", "y", "z", "nx", "ny", "nz"]
    + [f"f_dc_{i}" for i in range(3)]
    + [f"f_rest_{i}" for i in range(45)]
    + ["opacity"]
    + [f"scale_{i}" for i in range(3)]
    + [f"rot_{i}" for i in range(4)]
)


def _as_numpy(model) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    means = model.means.detach().cpu().numpy().astype(np.float32)
    log_scales = model.log_scales.detach().cpu().numpy().astype(np.float32)
    quats = model.quats.detach().cpu().numpy().astype(np.float32)
    opacity_logits = model.opacity_logits.detach().cpu().numpy().astype(np.float32).reshape(-1)
    colors = model.colors().detach().cpu().numpy().astype(np.float32)
    qn = np.linalg.norm(quats, axis=1, keepdims=True)
    quats = quats / np.maximum(qn, 1e-8)
    return means, log_scales, quats, opacity_logits, colors


def write_splat(path: Path, model) -> None:
    """Write the compact 32-byte/splat format used by web viewers."""
    means, log_scales, quats, opacity_logits, colors = _as_numpy(model)
    n = len(means)
    scales = np.exp(np.clip(log_scales, -12.0, 0.7)).astype(np.float32)
    opacity = 1.0 / (1.0 + np.exp(-opacity_logits))
    rgba = np.empty((n, 4), np.uint8)
    rgba[:, :3] = np.clip(colors * 255.0, 0, 255).astype(np.uint8)
    rgba[:, 3] = np.clip(opacity * 255.0, 0, 255).astype(np.uint8)
    rot = np.clip(quats * 128.0 + 128.0, 0, 255).astype(np.uint8)
    dtype = np.dtype([
        ("position", "<f4", 3), ("scale", "<f4", 3),
        ("color", "u1", 4), ("rotation", "u1", 4),
    ])
    data = np.empty(n, dtype=dtype)
    data["position"] = means
    data["scale"] = scales
    data["color"] = rgba
    data["rotation"] = rot
    path.parent.mkdir(parents=True, exist_ok=True)
    data.tofile(str(path))


def write_ply(path: Path, model) -> None:
    """Write a 3DGS-compatible binary PLY (DC colour, zero higher-order SH)."""
    means, log_scales, quats, opacity_logits, colors = _as_numpy(model)
    n = len(means)
    fields = [("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
              ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4")]
    fields += [(f"f_dc_{i}", "<f4") for i in range(3)]
    fields += [(f"f_rest_{i}", "<f4") for i in range(45)]
    fields += [("opacity", "<f4")]
    fields += [(f"scale_{i}", "<f4") for i in range(3)]
    fields += [(f"rot_{i}", "<f4") for i in range(4)]
    dtype = np.dtype(fields)
    data = np.zeros(n, dtype=dtype)
    data["x"], data["y"], data["z"] = means[:, 0], means[:, 1], means[:, 2]
    sh = (colors - 0.5) / C0
    for i in range(3):
        data[f"f_dc_{i}"] = sh[:, i]
    data["opacity"] = opacity_logits
    for i in range(3):
        data[f"scale_{i}"] = log_scales[:, i]
    for i in range(4):
        data[f"rot_{i}"] = quats[:, i]

    header = [
        "ply",
        "format binary_little_endian 1.0",
        "comment kinesplat CPU Gaussian Splatting export",
        f"element vertex {n}",
    ] + [f"property float {p}" for p in PLY_PROPS] + ["end_header", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(("\n".join(header)).encode("ascii"))
        data.tofile(handle)


def scene_payload(model, cameras: list[Camera], metrics: dict, name: str,
                  background: list[float] | None = None) -> dict:
    means = model.means.detach().cpu().numpy()
    lo = np.percentile(means, 1.0, axis=0)
    hi = np.percentile(means, 99.0, axis=0)
    center = (lo + hi) * 0.5
    radius = float(np.linalg.norm(hi - lo) * 0.5)
    if not math.isfinite(radius) or radius < 1e-6:
        radius = 2.0

    cams = []
    for cam in cameras:
        pos = cam.position
        forward = cam.R[:, 2]
        cams.append({
            "name": cam.name,
            "position": [float(x) for x in pos],
            "target": [float(x) for x in (pos + forward * 0.5)],
            "up": [float(-x) for x in cam.R[:, 1]],
            "width": cam.width,
            "height": cam.height,
            "fx": float(cam.fx),
            "fy": float(cam.fy),
            "cx": float(cam.cx),
            "cy": float(cam.cy),
        })

    return {
        "format": "splatlab-scene-v1",
        "name": name,
        "created": datetime.now(timezone.utc).isoformat(),
        "splat": "model.splat",
        "ply": "model.ply",
        "preview": "preview.png",
        "up": [0, 1, 0],
        "background": background or [0.05, 0.06, 0.07],
        "bounds": {
            "min": [float(x) for x in lo],
            "max": [float(x) for x in hi],
            "center": [float(x) for x in center],
            "radius": radius,
        },
        "cameras": cams,
        "stats": metrics,
    }
