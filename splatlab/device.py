"""Device selection and memory-aware tuning.

kinesplat runs anywhere PyTorch runs.  A CUDA GPU makes the optimiser 10-40x
faster, but the custom rasteriser materialises one entry per splat/pixel
overlap, so the practical limit is VRAM rather than raw compute.  Everything
here is budgeted for ~4 GB of VRAM or less.
"""

from __future__ import annotations

import dataclasses
import shutil

import torch

from .config import Preset

# Conservative per-entry cost while the autograd graph is alive: pixel index
# (int64), depth + alpha + 3 colour channels (float32), plus the sort/cumsum
# temporaries.  Deliberately pessimistic.
_BYTES_PER_ENTRY = 72
# Leave room for the CUDA context, cuBLAS workspace, the depth model and the
# display driver.  A 4 GB card typically has ~3.4 GB usable.
_HEADROOM = 0.18


def resolve_device(prefer: str | None = "auto") -> torch.device:
    """Turn 'auto' / 'cpu' / 'cuda' / 'cuda:0' into a usable device."""
    prefer = str(prefer or "auto").strip().lower()
    if prefer in ("gpu", "cuda", "auto"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        if prefer != "auto":
            raise RuntimeError(
                "CUDA was requested but this PyTorch build has no CUDA support. "
                "Install the GPU build first:  .\\setup.ps1 -Gpu"
            )
        if shutil.which("nvidia-smi"):
            raise RuntimeError(
                "An NVIDIA GPU is present, but torch.cuda.is_available() is false. "
                "The current environment is using CPU-only PyTorch; run .\\setup.ps1 -Gpu "
                "and verify the CUDA driver before starting kinesplat."
            )
    if prefer == "mps":
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        raise RuntimeError("MPS was requested but is not available.")
    return torch.device("cpu")


def device_report(device: torch.device) -> dict:
    """Small, JSON-safe description for logs and the web UI."""
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        return {
            "type": "cuda",
            "name": str(props.name),
            "vram_gb": round(props.total_memory / 1024 ** 3, 1),
            "cuda": str(torch.version.cuda or ""),
            "index": int(index),
        }
    if device.type == "mps":
        return {"type": "mps", "name": "Apple GPU", "vram_gb": None, "cuda": ""}
    return {"type": "cpu", "name": "CPU", "vram_gb": None, "cuda": ""}


def label(device: torch.device) -> str:
    info = device_report(device)
    if info["type"] == "cuda":
        return f"CUDA · {info['name']} ({info['vram_gb']} GB)"
    if info["type"] == "mps":
        return "Apple GPU"
    return "CPU"


def vram_bytes(device: torch.device) -> int | None:
    if device.type != "cuda":
        return None
    index = device.index if device.index is not None else torch.cuda.current_device()
    return int(torch.cuda.get_device_properties(index).total_memory)


def tune_preset(preset: Preset, device: torch.device) -> tuple[Preset, list[str]]:
    """Clamp a preset so its splat count and resolution fit the available VRAM.

    The bundled presets already fit in 4 GB; this is a safety net for bigger
    custom presets (and keeps users from OOM-ing a 4 GB laptop GPU).
    """
    total = vram_bytes(device)
    if total is None:
        return preset, []

    gb = total / 1024 ** 3
    if gb < 4.5:
        max_side, max_splats = 256, 120_000
    elif gb < 6.5:
        max_side, max_splats = 320, 200_000
    elif gb < 9.0:
        max_side, max_splats = 384, 300_000
    else:
        max_side, max_splats = 512, 400_000

    changes: list[str] = []
    kwargs = {}
    if preset.train_size > max_side:
        kwargs["train_size"] = max_side
        changes.append(f"resolution {preset.train_size}px → {max_side}px")
    if preset.max_gaussians > max_splats:
        kwargs["max_gaussians"] = max_splats
        changes.append(f"splats {preset.max_gaussians:,} → {max_splats:,}")
    if not kwargs:
        return preset, []
    return dataclasses.replace(preset, **kwargs), changes


def fits_budget(device: torch.device, n_splats: int, width: int, height: int,
                radius: int = 6) -> bool:
    """Rough pre-flight check for one rasteriser call.

    Entries scale with the splat footprint, not the image size, so this is the
    quantity worth checking before a long run.
    """
    total = vram_bytes(device)
    if total is None:
        return True
    pixels = (2 * radius + 1) ** 2
    entries = n_splats * pixels * 0.35  # most splats are much smaller than r=6
    need = entries * _BYTES_PER_ENTRY * 6  # graph + temporaries
    return need < total * (1.0 - _HEADROOM)


def gpu_peak_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2)
