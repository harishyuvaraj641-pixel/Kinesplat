"""Differentiable CPU Gaussian-splat rasteriser.

Rather than looping over screen tiles, every Gaussian is scattered into the
pixels inside its 3-sigma footprint.  Entries are then depth-sorted per pixel
and blended front-to-back with the standard 3DGS alpha-compositing equation.
This keeps the number of evaluated pixels proportional to the actual splat
area, which is what makes CPU optimisation practical.
"""

from __future__ import annotations

import math

import torch

from .geometry import quat_to_rotmat_torch
from .types import Camera

EPS = 1e-8


def _bg_tensor(background, device, dtype):
    """Accept either a scalar or an RGB triple for the background colour."""
    if isinstance(background, torch.Tensor):
        bg = background.to(device=device, dtype=dtype).reshape(-1)
    else:
        bg = torch.as_tensor(background, device=device, dtype=dtype).reshape(-1)
    if bg.numel() == 1:
        bg = bg.expand(3)
    return bg[:3]


def _empty_image(h: int, w: int, device, bg):
    bg = _bg_tensor(bg, device, torch.float32)
    img = bg[:, None, None].expand(3, h, w).clone()
    alpha = torch.zeros((h, w), device=device)
    return img, alpha


def render(
    means: torch.Tensor,
    log_scales: torch.Tensor,
    quats: torch.Tensor,
    opacity_logits: torch.Tensor,
    color_logits: torch.Tensor,
    camera: Camera,
    width: int,
    height: int,
    background: float = 0.0,
    max_radius: int = 8,
    k_per_pixel: int = 24,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Render a (3,H,W) image and (H,W) accumulated alpha."""
    device = means.device
    n = means.shape[0]
    if n == 0:
        return _empty_image(height, width, device, background)

    Rcw = torch.as_tensor(camera.R.T, dtype=means.dtype, device=device)
    tcw = torch.as_tensor(-camera.R.T @ camera.t, dtype=means.dtype, device=device)
    fx, fy = float(camera.fx), float(camera.fy)
    cx, cy = float(camera.cx), float(camera.cy)

    x_cam = means @ Rcw.T + tcw
    x, y, z = x_cam[:, 0], x_cam[:, 1], x_cam[:, 2]
    valid = z > 0.06

    zz = torch.clamp(z, min=0.06)
    u = fx * x / zz + cx
    v = fy * y / zz + cy

    Rq = quat_to_rotmat_torch(quats)
    S = torch.exp(torch.clamp(log_scales, -12.0, 0.7))
    M = Rq * S[:, None, :]                       # (N,3,3) columns scaled
    Sigma = M @ M.transpose(1, 2)                # world-space covariance
    Sigma_cam = Rcw @ Sigma @ Rcw.T

    J = torch.zeros((n, 2, 3), dtype=means.dtype, device=device)
    J[:, 0, 0] = fx / zz
    J[:, 0, 2] = -fx * x / (zz * zz)
    J[:, 1, 1] = fy / zz
    J[:, 1, 2] = -fy * y / (zz * zz)
    cov2 = J @ Sigma_cam @ J.transpose(1, 2)
    cov2[:, 0, 0] = cov2[:, 0, 0] + 0.3
    cov2[:, 1, 1] = cov2[:, 1, 1] + 0.3

    a = cov2[:, 0, 0]
    b = cov2[:, 0, 1]
    c = cov2[:, 1, 1]
    det = torch.clamp(a * c - b * b, min=1e-9)
    inv00 = c / det
    inv01 = -b / det
    inv11 = a / det

    mid = 0.5 * (a + c)
    disc = torch.sqrt(torch.clamp(mid * mid - det, min=0.0))
    radius = 3.0 * torch.sqrt(torch.clamp(mid + disc, min=0.0))
    radius = torch.clamp(radius, 0.35, float(max_radius))
    r_int = torch.ceil(radius).long()

    inside = (u > -max_radius - 2) & (u < width + max_radius + 2) & \
             (v > -max_radius - 2) & (v < height + max_radius + 2)
    valid = valid & inside
    if not bool(valid.any()):
        return _empty_image(height, width, device, background)

    op = torch.sigmoid(opacity_logits).squeeze(-1)
    rgb = torch.sigmoid(color_logits)

    pix_parts: list[torch.Tensor] = []
    alpha_parts: list[torch.Tensor] = []
    depth_parts: list[torch.Tensor] = []
    color_parts: list[torch.Tensor] = []

    for r in range(1, max_radius + 1):
        sel = (r_int == r) & valid
        if not bool(sel.any()):
            continue
        idx = torch.nonzero(sel, as_tuple=False).squeeze(1)
        if idx.numel() == 0:
            continue
        # Large splats are rare; chunk them to cap peak memory.
        chunk = max(1, min(16384, int(2_000_000 // ((2 * r + 1) ** 2))))
        offsets = torch.arange(-r, r + 1, device=device, dtype=means.dtype)
        dy, dx = torch.meshgrid(offsets, offsets, indexing="ij")
        dx = dx.reshape(-1)
        dy = dy.reshape(-1)
        off = dx.numel()

        for start in range(0, idx.numel(), chunk):
            ids = idx[start:start + chunk]
            g = ids.numel()
            # Subpixel-exact evaluation: the splat centre stays at its true
            # continuous position instead of being snapped to the pixel grid.
            # Snapping quantises every splat to whole pixels, which produces
            # staircase edges and caps the achievable sharpness; the fractional
            # offset below costs a few extra temporaries and nothing else.
            uu = torch.round(u[ids])
            vv = torch.round(v[ids])
            fx0 = (u[ids] - uu)[:, None]        # in [-0.5, 0.5]
            fy0 = (v[ids] - vv)[:, None]
            px = (uu[:, None] + dx[None, :]).long()
            py = (vv[:, None] + dy[None, :]).long()
            dxc = dx[None, :] - fx0             # continuous offset from centre
            dyc = dy[None, :] - fy0
            power = (
                inv00[ids, None] * (dxc ** 2)
                + 2.0 * inv01[ids, None] * (dxc * dyc)
                + inv11[ids, None] * (dyc ** 2)
            )
            alpha = op[ids, None] * torch.exp(-0.5 * power)
            alpha = torch.clamp(alpha, max=0.995)
            mask = (alpha > 0.002) & (px >= 0) & (px < width) & (py >= 0) & (py < height)
            if not bool(mask.any()):
                continue
            pix = py * width + px
            pix_parts.append(pix[mask])
            alpha_parts.append(alpha[mask])
            depth_parts.append(z[ids, None].expand(g, off)[mask])
            colors = rgb[ids].unsqueeze(1).expand(g, off, 3)
            color_parts.append(colors.reshape(-1, 3)[mask.reshape(-1)])

    if not pix_parts:
        return _empty_image(height, width, device, background)

    pix = torch.cat(pix_parts)
    alpha = torch.cat(alpha_parts)
    depth = torch.cat(depth_parts)
    colors = torch.cat(color_parts, dim=0)

    # Sort by (pixel, depth): near entries first within each pixel.
    dkey = torch.clamp((depth * 1000.0), 0, 99_999).long()
    key = pix.to(torch.int64) * 100_000 + dkey
    order = torch.argsort(key)
    pix_s = pix[order]
    alpha_s = alpha[order]
    color_s = colors[order]

    # Exact front-to-back compositing with no per-pixel entry cap.
    #
    # Clipping each pixel to its k nearest entries looks harmless but is not:
    # on a slanted surface the *nearest* entries are splats centred on
    # neighbouring pixels, while the splat centred on this pixel sits a little
    # farther away and gets dropped.  With many frames the pixel then keeps
    # only grazing contributions, the accumulated alpha collapses and the
    # background bleeds through -- the reconstruction looks washed out.
    #
    # Instead we composite every entry: the transmittance before entry i is
    # exp(sum of log(1 - a_j) for j before i within the same pixel), which is a
    # segmented cumulative sum.  Alpha is clamped below 1 so the log is finite.
    m = pix_s.numel()
    pos = torch.arange(m, device=device)
    is_first = torch.ones(m, dtype=torch.bool, device=device)
    is_first[1:] = pix_s[1:] != pix_s[:-1]
    first_idx = torch.where(is_first, pos, torch.zeros_like(pos))
    first_idx = torch.cummax(first_idx, dim=0).values

    a_safe = torch.clamp(alpha_s, max=1.0 - 1e-4)
    log_t = torch.log1p(-a_safe)
    cum = torch.cumsum(log_t, dim=0)
    seg_start = cum[first_idx]
    trans_excl = torch.exp(torch.clamp(cum - log_t - seg_start, max=0.0))
    weights = a_safe * trans_excl

    cells = height * width
    out = torch.zeros((cells, 3), dtype=means.dtype, device=device).index_add(
        0, pix_s, weights[:, None] * color_s
    )
    accum = torch.zeros(cells, dtype=means.dtype, device=device).index_add(
        0, pix_s, weights
    )

    # Entries already store the colour premultiplied by weight, so the residual
    # transmittance simply lets the background through.
    bg = _bg_tensor(background, device, means.dtype)
    out = out + (1.0 - accum)[:, None] * bg[None, :]

    img = out.view(height, width, 3).permute(2, 0, 1)
    return img, accum.view(height, width)


def evaluate_psnr(pred: torch.Tensor, gt: torch.Tensor) -> float:
    mse = torch.mean((pred - gt) ** 2).item()
    return 10.0 * math.log10(1.0 / max(mse, 1e-12))
