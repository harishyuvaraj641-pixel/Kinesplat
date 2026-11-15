"""Gaussian parameter container and depth-driven initialisation."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

from .config import Preset
from .geometry import unproject_pixels
from .types import Camera, Scene


class GaussianModel(nn.Module):
    def __init__(self, means: torch.Tensor, log_scales: torch.Tensor, quats: torch.Tensor,
                 opacity_logits: torch.Tensor, color_logits: torch.Tensor):
        super().__init__()
        self.means = nn.Parameter(means)
        self.log_scales = nn.Parameter(log_scales)
        self.quats = nn.Parameter(quats)
        self.opacity_logits = nn.Parameter(opacity_logits)
        self.color_logits = nn.Parameter(color_logits)

    @property
    def count(self) -> int:
        return int(self.means.shape[0])

    def scales(self) -> torch.Tensor:
        return torch.exp(torch.clamp(self.log_scales, -12.0, 0.7))

    def colors(self) -> torch.Tensor:
        return torch.sigmoid(self.color_logits)

    def opacity(self) -> torch.Tensor:
        return torch.sigmoid(self.opacity_logits)

    def save(self, path: Path) -> None:
        np.savez(
            path,
            means=self.means.detach().cpu().numpy(),
            log_scales=self.log_scales.detach().cpu().numpy(),
            quats=self.quats.detach().cpu().numpy(),
            opacity_logits=self.opacity_logits.detach().cpu().numpy(),
            color_logits=self.color_logits.detach().cpu().numpy(),
        )

    @classmethod
    def load(cls, path: Path) -> "GaussianModel":
        data = np.load(path)
        return cls(
            torch.from_numpy(data["means"]).float(),
            torch.from_numpy(data["log_scales"]).float(),
            torch.from_numpy(data["quats"]).float(),
            torch.from_numpy(data["opacity_logits"]).float(),
            torch.from_numpy(data["color_logits"]).float(),
        )


def _logit(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.01, 0.99)
    return np.log(x / (1.0 - x))


def _voxel_prune(points: np.ndarray, colors: np.ndarray, scales: np.ndarray,
                 max_points: int, rng: np.random.Generator, log=print) -> tuple:
    """Uniformly thin the cloud with progressively coarser voxels.

    Randomly dropping points leaves holes, and the optimiser then inflates a
    few splats to cover them.  Voxel thinning keeps an even surface sample and
    makes the reconstruction much sharper at the same splat budget.
    """
    def downsample(pts, cols, scs, voxel):
        keys = np.floor(pts / max(voxel, 1e-6)).astype(np.int64)
        hashed = ((keys[:, 0] * 73856093) ^ (keys[:, 1] * 19349663)
                  ^ (keys[:, 2] * 83492791))
        _, unique = np.unique(hashed, return_index=True)
        return pts[unique], cols[unique], scs[unique]

    if len(points) > max_points:
        span = np.percentile(points, 98, axis=0) - np.percentile(points, 2, axis=0)
        diag = float(np.linalg.norm(span))
        voxel = max(diag / 240.0, 1e-4)
        points, colors, scales = downsample(points, colors, scales, voxel)
        for _ in range(8):
            if len(points) <= max_points:
                break
            voxel *= max(1.06, (len(points) / max_points) ** (1.0 / 3.0))
            points, colors, scales = downsample(points, colors, scales, voxel)
        log(f"  voxel merge -> {len(points)} candidate splats")

    if len(points) > max_points:
        pick = rng.choice(len(points), size=max_points, replace=False)
        points, colors, scales = points[pick], colors[pick], scales[pick]
    return points, colors, scales


def initialise(
    scene: Scene,
    cameras: list[Camera],
    images: list[np.ndarray],
    depths: list[np.ndarray],
    preset: Preset,
    seed: int = 4,
    backgrounds: list[np.ndarray] | None = None,
    log=print,
    device: torch.device | str | None = None,
) -> GaussianModel:
    """Unproject depth samples from every frame into an initial splat cloud."""
    rng = np.random.default_rng(seed)
    all_points: list[np.ndarray] = []
    all_colors: list[np.ndarray] = []
    all_scales: list[np.ndarray] = []

    for fi, (cam, img, depth) in enumerate(zip(cameras, images, depths)):
        h, w = depth.shape
        # A depth pixel on a silhouette mixes foreground and background, so its
        # unprojected point hangs in mid air.  Every view then contributes a
        # cloud of those "flying pixels", which is what makes a dense
        # reconstruction look speckled.  Reject pixels whose 3x3 neighbourhood
        # spans a large depth jump.
        zfull = np.nan_to_num(np.asarray(depth, dtype=np.float32),
                              nan=0.0, posinf=0.0, neginf=0.0)
        kernel = np.ones((3, 3), np.uint8)
        near = cv2.erode(zfull, kernel)
        far = cv2.dilate(zfull, kernel)
        smooth = (far - near) <= np.maximum(0.06 * far, 0.02)

        stride = max(1, preset.init_stride)
        ys = np.arange(stride // 2, h, stride)
        xs = np.arange(stride // 2, w, stride)
        if len(xs) == 0 or len(ys) == 0:
            continue
        gx, gy = np.meshgrid(xs, ys)
        gx = gx.reshape(-1)
        gy = gy.reshape(-1)
        z = depth[gy, gx].astype(np.float64)
        col = img[gy, gx][:, ::-1].astype(np.float32) / 255.0  # BGR -> RGB

        good = np.isfinite(z) & (z > 0.02) & smooth[gy, gx]
        if good.sum() < 8:
            continue
        lo, hi = np.percentile(z[good], [1.0, 98.5])
        good &= (z >= max(0.02, lo)) & (z <= hi)
        # A monocular depth map often has a flat far plane (sky, ceiling, an
        # open doorway) that is far behind everything else.  Unprojecting that
        # plane from several viewpoints fills the world with sheets that cut
        # through the real geometry, so drop a large plateau at the far end of
        # the depth histogram.
        if good.sum() >= 64:
            hist, edges = np.histogram(z[good], bins=48)
            peak = int(hist.argmax())
            if peak >= len(hist) - 3 and hist[peak] > 0.10 * good.sum():
                good &= z < edges[peak]
        med_z = float(np.median(z[good])) if good.any() else 0.0
        if med_z > 0:
            good &= z <= med_z * 5.0
        # Drop sky / flat background: far pixels whose colour matches the
        # estimated frame background.  Rendering the background colour is
        # cheaper and cleaner than a shell of giant distant splats.
        if backgrounds is not None and fi < len(backgrounds):
            bg_col = np.asarray(backgrounds[fi], dtype=np.float64).reshape(3)
            far = z > np.percentile(z[good], 86.0)
            close = np.linalg.norm(col - bg_col[None, :], axis=1) < 0.30
            good &= ~(far & close)
        gx, gy, z, col = gx[good], gy[good], z[good], col[good]
        if len(z) < 8:
            continue

        u = gx.astype(np.float64) + 0.5
        v = gy.astype(np.float64) + 0.5
        world = unproject_pixels(u, v, z, cam.K, cam.R, cam.t)
        # A splat should cover roughly the pixel footprint it was sampled from.
        # Oversized initial splats let neighbouring views smear over each other,
        # so stay a little below one pixel instead of well above it.
        base = np.clip(z / max(1e-6, cam.fx) * stride * 0.7, 1e-4, 0.25)
        scale = np.full((len(z), 3), 1.0, np.float32) * base[:, None].astype(np.float32)

        all_points.append(world.astype(np.float32))
        all_colors.append(col)
        all_scales.append(scale)

    if not all_points:
        raise RuntimeError("Depth initialisation produced no points. "
                           "Try a capture with more texture and overlap.")

    points = np.concatenate(all_points, 0)
    colors = np.concatenate(all_colors, 0)
    scales = np.concatenate(all_scales, 0)
    log(f"  {len(points):,} raw depth samples from {len(all_points)} frames")

    points, colors, scales = _voxel_prune(
        points, colors, scales, preset.max_gaussians, rng, log=log
    )
    if len(points) < 100:
        raise RuntimeError("Too few valid depth samples for reconstruction.")

    # Tiny jitter avoids a perfectly regular lattice.
    points = points + rng.normal(0, 0.15, points.shape).astype(np.float32) * scales
    quats = np.zeros((len(points), 4), np.float32)
    quats[:, 0] = 1.0
    opacity = np.full((len(points), 1), 1.6, np.float32)

    log(f"  initialised {len(points):,} Gaussians")
    model = GaussianModel(
        means=torch.from_numpy(points).float(),
        log_scales=torch.from_numpy(np.log(np.clip(scales, 1e-5, 1.0))).float(),
        quats=torch.from_numpy(quats).float(),
        opacity_logits=torch.from_numpy(_logit(np.full((len(points), 1), 0.82))).float(),
        color_logits=torch.from_numpy(_logit(colors)).float(),
    )
    if device is not None:
        model = model.to(torch.device(device))
    return model
