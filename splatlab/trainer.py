"""Optimisation loop for 3D Gaussian splats (CUDA or CPU)."""

from __future__ import annotations

import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

from .config import Preset
from .gaussians import GaussianModel
from .geometry import quat_to_rotmat_torch
from .rasterizer import evaluate_psnr, render
from .types import Camera


def _gpu_peak() -> float:
    """Peak allocated VRAM this session, 0.0 when not on CUDA."""
    try:
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1024 ** 3
    except Exception:
        pass
    return 0.0


def configure_cpu(threads: int | None = None) -> int:
    if threads is None:
        threads = max(1, (torch.get_num_threads() or 1))
        try:
            import os
            threads = max(1, min(16, (os.cpu_count() or 4) - 1))
        except Exception:
            pass
    torch.set_num_threads(int(threads))
    try:
        torch.set_flush_denormal(True)
    except Exception:
        pass
    return int(threads)


def _ssim_window(size: int = 7, sigma: float = 1.5, device="cpu", dtype=torch.float32):
    x = torch.arange(size, dtype=dtype, device=device) - size // 2
    g = torch.exp(-(x * x) / (2 * sigma * sigma))
    g = g / g.sum()
    return (g[:, None] * g[None, :]).reshape(1, 1, size, size)


def ssim_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """pred/gt: (3,H,W) in [0,1]."""
    x = pred.unsqueeze(0)
    y = gt.unsqueeze(0)
    win = _ssim_window(device=x.device, dtype=x.dtype).repeat(3, 1, 1, 1)
    pad = win.shape[-1] // 2
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    mu_x = torch.nn.functional.conv2d(x, win, padding=pad, groups=3)
    mu_y = torch.nn.functional.conv2d(y, win, padding=pad, groups=3)
    xx = torch.nn.functional.conv2d(x * x, win, padding=pad, groups=3) - mu_x * mu_x
    yy = torch.nn.functional.conv2d(y * y, win, padding=pad, groups=3) - mu_y * mu_y
    xy = torch.nn.functional.conv2d(x * y, win, padding=pad, groups=3) - mu_x * mu_y
    sx = torch.clamp(xx, min=0.0)
    sy = torch.clamp(yy, min=0.0)
    sxy = torch.clamp(xy, min=-1.0, max=1.0)
    ssim_map = ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / \
               ((mu_x ** 2 + mu_y ** 2 + c1) * (sx + sy + c2))
    return 1.0 - ssim_map.mean()


def _save_uint8(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = np.clip(rgb[..., ::-1] * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(str(path), bgr)


def _make_optim(model: GaussianModel, base_lrs: list[float]):
    """Rebuild the Adam optimiser for the current model parameters."""
    groups = [
        {"params": [model.means], "lr": base_lrs[0], "name": "means"},
        {"params": [model.log_scales], "lr": base_lrs[1], "name": "scales"},
        {"params": [model.quats], "lr": base_lrs[2], "name": "quats"},
        {"params": [model.opacity_logits], "lr": base_lrs[3], "name": "opacity"},
        {"params": [model.color_logits], "lr": base_lrs[4], "name": "colour"},
    ]
    return torch.optim.Adam(groups, eps=1e-8), groups


def _densify(model: GaussianModel, grad_ema: torch.Tensor, cap: int, log=print):
    """One adaptive refinement event: prune transparent splats, then clone or
    split the splats with the strongest accumulated positional gradients.

    Without this the splat set is frozen from the depth initialisation:
    under-covered regions stay smeared by oversized splats and floaters from
    noisy depth samples are never removed.  A percentile threshold
    self-calibrates across scenes instead of a fixed gradient cutoff.

    Returns (grad_ema, pruned, added)."""
    with torch.no_grad():
        op = model.opacity().squeeze(-1)
        keep = op > 0.05
        n_keep = int(keep.sum())
        if n_keep < 100:
            return grad_ema, 0, 0
        pruned = model.count - n_keep

        tensors = [model.means.data, model.log_scales.data, model.quats.data,
                   model.opacity_logits.data, model.color_logits.data]
        parts = [[t[keep]] for t in tensors]
        new_grad = [grad_ema[keep]]
        added = 0

        headroom = cap - n_keep
        if headroom > 0:
            gvals = torch.nan_to_num(grad_ema[keep])
            thr = torch.quantile(gvals, 0.95)
            hot = keep & (grad_ema >= thr)
            scales = model.scales()
            smax = scales.max(dim=1).values
            big_thr = torch.quantile(torch.nan_to_num(smax[keep]), 0.85)
            big = hot & (smax > big_thr)
            small = hot & ~big
            n_small = int(small.sum())

            # Clone: duplicate small hot splats as-is so they can specialise.
            if n_small > 0:
                n_small = min(n_small, headroom)
                idx = small.nonzero(as_tuple=False).squeeze(1)[:n_small]
                for t, p in zip(tensors, parts):
                    p.append(t[idx])
                new_grad.append(grad_ema[idx])
                added += n_small
                headroom -= n_small

            # Split: replace large hot splats by two half-sized children
            # sampled from the parent Gaussian.
            n_big = int(big.sum())
            if n_big > 0 and headroom >= 2:
                n_big = min(n_big, headroom // 2)
                idx = big.nonzero(as_tuple=False).squeeze(1)[:n_big]
                means_b = tensors[0][idx]
                quats_b = tensors[2][idx]
                logs_b = tensors[1][idx]
                S = torch.exp(torch.clamp(logs_b, -12.0, 0.7))
                M = quat_to_rotmat_torch(quats_b) * S[:, None, :]
                delta = torch.einsum(
                    "nij,nj->ni",
                    M,
                    torch.normal(0.0, 0.5, (n_big, 3),
                                 device=M.device, dtype=M.dtype),
                )
                two = logs_b.repeat(2, 1) - math.log(1.5)
                for t, p in zip(tensors, parts):
                    if t is tensors[0]:
                        p.append(torch.cat([means_b + delta, means_b - delta]))
                    elif t is tensors[1]:
                        p.append(two)
                    else:
                        p.append(t[idx].repeat(2, *[1] * (t.ndim - 1)))
                new_grad.append(torch.zeros(2 * n_big, device=grad_ema.device))
                added += 2 * n_big

        if pruned == 0 and added == 0:
            return grad_ema, 0, 0

        model.means = nn.Parameter(torch.cat(parts[0], 0))
        model.log_scales = nn.Parameter(torch.cat(parts[1], 0))
        model.quats = nn.Parameter(torch.cat(parts[2], 0))
        model.opacity_logits = nn.Parameter(torch.cat(parts[3], 0))
        model.color_logits = nn.Parameter(torch.cat(parts[4], 0))
        grad_ema = torch.cat(new_grad, 0)
        if pruned:
            log(f"  densify: pruned {pruned:,} faint splats, added {added:,}")
        elif added:
            log(f"  densify: added {added:,} splats ({model.count:,} total)")
        return grad_ema, pruned, added


def _prune_faint(model: GaussianModel, threshold: float = 0.02) -> int:
    """Remove near-transparent splats; they cost export size and render time."""
    with torch.no_grad():
        op = model.opacity().squeeze(-1)
        keep = op > threshold
        removed = model.count - int(keep.sum())
        if removed:
            model.means = nn.Parameter(model.means.data[keep])
            model.log_scales = nn.Parameter(model.log_scales.data[keep])
            model.quats = nn.Parameter(model.quats.data[keep])
            model.opacity_logits = nn.Parameter(model.opacity_logits.data[keep])
            model.color_logits = nn.Parameter(model.color_logits.data[keep])
    return removed


def train(
    model: GaussianModel,
    cameras: list[Camera],
    images: np.ndarray | list[np.ndarray],
    preset: Preset,
    out_dir: Path,
    log=print,
    on_progress=None,
    cancel_event=None,
    seed: int = 3,
    max_radius: int = 8,
    k_per_pixel: int = 24,
    backgrounds: list[np.ndarray] | None = None,
    device: torch.device | str | None = None,
) -> dict:
    out_dir = Path(out_dir)
    preview_dir = out_dir / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device) if device is not None else torch.device("cpu")
    model.to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    if isinstance(images, list):
        gt_list = [
            torch.from_numpy(img[:, :, ::-1].astype(np.float32) / 255.0).permute(2, 0, 1).contiguous().to(device)
            for img in images
        ]
    else:
        gt_list = [
            torch.from_numpy(img[:, :, ::-1].astype(np.float32) / 255.0).permute(2, 0, 1).contiguous().to(device)
            for img in images
        ]
    height, width = gt_list[0].shape[-2:]
    rng = np.random.default_rng(seed)
    if backgrounds:
        bg_list = [
            torch.from_numpy(np.asarray(bg, dtype=np.float32).reshape(3)).to(device)
            for bg in backgrounds
        ]
        while len(bg_list) < len(gt_list):
            bg_list.append(bg_list[-1])
    else:
        bg_list = [torch.zeros(3, device=device) for _ in gt_list]

    base_lrs = [2.5e-4, 1.2e-3, 4.0e-4, 1.5e-2, 4.0e-3]
    optim, groups = _make_optim(model, base_lrs)
    iterations = max(10, preset.iterations)
    start = time.time()
    ema = None
    history: list[float] = []
    best = 1e9
    # Snapshot of the best weights seen so far.  A single-view gradient with
    # Adam's per-parameter step can drift over thousands of iterations, so the
    # final step is not necessarily the best one.
    params = [model.means, model.log_scales, model.quats,
              model.opacity_logits, model.color_logits]
    best_state: list[torch.Tensor] | None = None
    restored = False
    warmup = max(1, int(getattr(preset, "warmup", 0) or 1))

    # Adaptive refinement: densify clone/split events early in the run, then
    # stop so the surviving splats can settle.
    grad_ema = torch.zeros(model.count, device=device)
    densify_every = max(50, iterations // 12)
    densify_until = int(0.70 * iterations)
    total_pruned = total_added = 0

    frame_indices = np.arange(len(gt_list))
    order: list[int] = []
    step = 0
    while step < iterations:
        if cancel_event is not None and cancel_event.is_set():
            log("Training cancelled; exporting current splats.")
            break
        if not order:
            order = list(rng.permutation(frame_indices))
        fi = order.pop()
        cam = cameras[fi]
        gt = gt_list[fi]
        # Exponential decay, independent of the total step count: cosine holds
        # ~90 % of the LR for the first half of a long run, which lets single-view
        # Adam updates slowly jitter the splats apart.  Reaching 20 % by a
        # quarter of the run keeps the same behaviour whether the preset is 250
        # or 1300 iterations.
        ramp = min(1.0, (step + 1) / warmup)
        decay = 0.05 + 0.95 * math.exp(-4.0 * step / iterations)
        gain = decay * (0.15 + 0.85 * ramp)
        for g, base in zip(groups, base_lrs):
            g["lr"] = base * gain

        pred, _alpha = render(
            model.means, model.log_scales, model.quats, model.opacity_logits,
            model.color_logits, cam, width, height,
            background=bg_list[fi], max_radius=max_radius, k_per_pixel=k_per_pixel,
        )
        l1 = torch.mean(torch.abs(pred - gt))
        ssim = ssim_loss(pred, gt)
        reg = 1e-5 * model.scales().mean() + 1e-5 * model.opacity().mean()
        loss = l1 + 0.22 * ssim + reg
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [model.means, model.log_scales, model.quats,
             model.opacity_logits, model.color_logits], 1.0
        )
        optim.step()

        # Track per-splat positional-gradient magnitude for densification.
        if step < densify_until and model.means.grad is not None:
            with torch.no_grad():
                grad_ema.mul_(0.9).add_(0.1 * model.means.grad.norm(dim=1))

        value = float(loss.detach())
        ema = value if ema is None else 0.96 * ema + 0.04 * value
        if ema < best - 1e-5:
            best = ema
            best_state = [p.detach().clone() for p in params]
        history.append(value)
        step += 1

        if step % densify_every == 0 and step < densify_until:
            grad_ema, n_pruned, n_added = _densify(
                model, grad_ema, preset.max_gaussians, log=log)
            if n_pruned or n_added:
                optim, groups = _make_optim(model, base_lrs)
                params = [model.means, model.log_scales, model.quats,
                          model.opacity_logits, model.color_logits]
                best = 1e9
                best_state = None
                total_pruned += n_pruned
                total_added += n_added

        if step == 1 or step % 20 == 0:
            log(f"  iter {step:4d}/{iterations}  loss {ema:.4f}  "
                f"({time.time() - start:.0f}s)")
        if on_progress:
            on_progress(step / iterations, f"Optimising splats {step}/{iterations}")
        if step % preset.preview_every == 0 or step == iterations or step == 1:
            with torch.no_grad():
                p, _ = render(
                    model.means, model.log_scales, model.quats, model.opacity_logits,
                    model.color_logits, cameras[0], width, height,
                    background=bg_list[0], max_radius=max_radius, k_per_pixel=k_per_pixel,
                )
            gt0 = gt_list[0]
            comp = torch.cat([gt0, p.clamp(0, 1)], dim=2).detach().cpu().numpy()
            comp = np.transpose(comp, (1, 2, 0))
            comp = cv2.resize(comp, (comp.shape[1] * max(1, 256 // max(1, comp.shape[0])),
                                     comp.shape[0] * max(1, 256 // max(1, comp.shape[0]))),
                              interpolation=cv2.INTER_NEAREST)
            _save_uint8(preview_dir / f"step_{step:04d}.png", comp)
            _save_uint8(out_dir / "preview.png", np.transpose(
                p.clamp(0, 1).detach().cpu().numpy(), (1, 2, 0)))
            if hasattr(cancel_event, "preview_path"):
                cancel_event.preview_path = str(out_dir / "preview.png")

    # The last step is not always the best one; keep the strongest snapshot.
    if best_state is not None and ema is not None and ema > best + 1e-4:
        with torch.no_grad():
            for param, saved in zip(params, best_state):
                param.copy_(saved)
        ema = best
        restored = True
        log(f"Restored best weights (loss {best:.4f})")

    # Drop near-transparent splats before export: they bloat the .splat file
    # and slow the viewer down without contributing to the image.
    pruned_final = _prune_faint(model)
    if pruned_final:
        log(f"  final prune: removed {pruned_final:,} splats ({model.count:,} left)")

    # Final quality audit on a few frames.
    psnrs: list[float] = []
    n_eval = min(4, len(gt_list))
    eval_ids = np.linspace(0, len(gt_list) - 1, n_eval).astype(int)
    with torch.no_grad():
        for fi in eval_ids:
            pred, _ = render(
                model.means, model.log_scales, model.quats, model.opacity_logits,
                model.color_logits, cameras[int(fi)], width, height,
                background=bg_list[int(fi)], max_radius=max_radius, k_per_pixel=k_per_pixel,
            )
            psnrs.append(evaluate_psnr(pred.clamp(0, 1), gt_list[int(fi)]))
    elapsed = time.time() - start
    peak = _gpu_peak()
    if peak:
        log(f"  peak GPU memory: {peak:.2f} GB")
    metrics = {
        "iterations": step,
        "final_loss": float(ema if ema is not None else 0.0),
        "best_loss": float(best),
        "psnr": float(np.mean(psnrs)) if psnrs else 0.0,
        "seconds": round(elapsed, 1),
        "gaussians": model.count,
        "resolution": [width, height],
        "loss_curve": [round(float(v), 5) for v in history[::max(1, len(history) // 60)]],
        "restored_best": bool(restored),
        "densify_pruned": int(total_pruned),
        "densify_added": int(total_added),
        "gpu_peak_gb": round(peak, 2) if peak else None,
    }
    model.save(out_dir / "gaussians.npz")
    log(f"Training finished in {elapsed:.0f}s  |  PSNR {metrics['psnr']:.2f} dB  |  "
        f"{model.count:,} splats")
    return metrics
