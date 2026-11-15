"""Paths, quality presets and shared constants."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

WEB_DIR = ROOT / "web"
ASSETS_DIR = ROOT / "assets"
DEMO_DIR = ASSETS_DIR / "demo"
MODELS_DIR = ROOT / "models"
JOBS_DIR = ROOT / "jobs"
TOOLS_DIR = ROOT / "tools"
CACHE_DIR = ROOT / ".cache"

for _d in (MODELS_DIR, JOBS_DIR, TOOLS_DIR, CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

DEPTH_MODEL_FILE = "depth_anything_v2_small.onnx"
DEPTH_MODEL_PATH = MODELS_DIR / DEPTH_MODEL_FILE
DEPTH_MODEL_URL = (
    "https://huggingface.co/onnx-community/depth-anything-v2-small/"
    "resolve/main/onnx/model.onnx"
)
DEPTH_MODEL_MIN_BYTES = 80_000_000

COLMAP_VERSION = "4.2.1"
COLMAP_ZIP_URL = (
    f"https://github.com/colmap/colmap/releases/download/{COLMAP_VERSION}/"
    "colmap-x64-windows-nocuda.zip"
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".wmv"}


@dataclass(frozen=True)
class Preset:
    """A CPU workload preset.  Everything is intentionally small by default."""

    key: str
    label: str
    description: str
    train_size: int          # longest side of the optimisation images
    max_frames: int          # frames used for reconstruction
    max_gaussians: int       # hard cap on the number of splats
    iterations: int          # optimisation steps
    init_stride: int         # pixel stride when unprojecting depth -> points
    preview_every: int
    warmup: int

    @property
    def train_resolution_label(self) -> str:
        return f"{self.train_size}px"


PRESETS: dict[str, Preset] = {
    "draft": Preset(
        key="draft",
        label="Draft",
        description="Fastest preview.  Great for checking framing and colour.",
        train_size=128,
        max_frames=16,
        max_gaussians=20_000,
        iterations=260,
        init_stride=2,
        preview_every=40,
        warmup=30,
    ),
    "balanced": Preset(
        key="balanced",
        label="Balanced",
        description="Recommended CPU setting.  A good quality / time trade-off.",
        train_size=192,
        max_frames=28,
        max_gaussians=90_000,
        iterations=700,
        init_stride=2,
        preview_every=60,
        warmup=40,
    ),
    "quality": Preset(
        key="quality",
        label="Quality",
        description="Sharper splats.  Takes noticeably longer on a laptop CPU.",
        train_size=256,
        max_frames=40,
        max_gaussians=180_000,
        iterations=1200,
        init_stride=1,
        preview_every=100,
        warmup=50,
    ),
}

DEFAULT_PRESET = "balanced"


def get_preset(key: str | None) -> Preset:
    return PRESETS.get((key or DEFAULT_PRESET).lower(), PRESETS[DEFAULT_PRESET])
