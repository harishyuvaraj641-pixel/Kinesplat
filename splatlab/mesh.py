"""Single-photo -> textured mesh reconstruction.

This is the second reconstruction backend.  The multi-view pipeline in
``pipeline.py`` needs several photographs to triangulate camera poses, so a
single upload can never work there.  TripoSR (MIT licensed, by Tripo AI and
Stability AI -- see ``tsr/LICENSE``) solves that case: a feed-forward network
reads one photograph and predicts a triplane NeRF, which is then converted to a
mesh with marching cubes.

The TripoSR checkpoint is kept local in ``models/triposr`` and the network
runs through the vendored ``tsr`` package.  The official DINO ViT config is
resolved through ``huggingface_hub`` on first model initialization.
"""

from __future__ import annotations

import json
import math
import time
import urllib.request
from pathlib import Path

import numpy as np

from .config import MODELS_DIR

TRIPOSR_DIR = MODELS_DIR / "triposr"
TRIPOSR_CONFIG = "config.yaml"
TRIPOSR_WEIGHTS = "model.ckpt"
TRIPOSR_BASE_URL = "https://huggingface.co/stabilityai/TripoSR/resolve/main"
# The published checkpoint is ~1.7 GB; anything far smaller is a truncated
# download (usually an HTML error page saved by a proxy).
TRIPOSR_MIN_BYTES = 1_000_000_000

DEFAULT_THRESHOLD = 25.0
DEFAULT_FOREGROUND_RATIO = 0.85
DEFAULT_CHUNK_SIZE = 8192
DEFAULT_RESOLUTION = 256

_MODEL = None
_MODEL_DEVICE = None
_MODEL_RESOLUTION = None


# --------------------------------------------------------------------- status
def weights_path() -> Path:
    return TRIPOSR_DIR / TRIPOSR_WEIGHTS


def config_path() -> Path:
    return TRIPOSR_DIR / TRIPOSR_CONFIG


def is_downloaded() -> bool:
    path = weights_path()
    return path.exists() and path.stat().st_size >= TRIPOSR_MIN_BYTES


def missing_packages() -> list[str]:
    """Report which optional packages the mesh backend still needs."""
    missing = []
    for name, module in (
        ("torch", "torch"),
        ("einops", "einops"),
        ("omegaconf", "omegaconf"),
        ("trimesh", "trimesh"),
        ("transformers", "transformers"),
        ("huggingface-hub", "huggingface_hub"),
        ("rembg", "rembg"),
        ("scikit-image", "skimage"),
    ):
        try:
            __import__(module)
        except Exception:
            missing.append(name)
    return missing


def backend_status() -> dict:
    """Small payload for the web UI so it can explain what is available."""
    missing = missing_packages()
    return {
        "name": "TripoSR",
        "available": not missing,
        "downloaded": is_downloaded(),
        "weights": str(weights_path()),
        "missing": missing,
        "license": "MIT",
        "source": "https://github.com/VAST-AI-Research/TripoSR",
    }


# ------------------------------------------------------------------- download
def _download_file(url: str, dest: Path, progress=None, log=print,
                   expected_min: int = 0) -> Path:
    """Resumable download.  ``.part`` files survive interrupted connections."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    if dest.exists() and dest.stat().st_size >= max(expected_min, 1):
        return dest

    have = part.stat().st_size if part.exists() else 0
    headers = {"User-Agent": "kinesplat/1.0"}
    if have:
        headers["Range"] = f"bytes={have}-"
    request = urllib.request.Request(url, headers=headers)
    started = time.time()
    with urllib.request.urlopen(request, timeout=60) as response:
        status = getattr(response, "status", 200)
        if have and status != 206:
            # The server ignored the range request; start over cleanly.
            have = 0
        total = int(response.headers.get("Content-Length") or 0) + have
        mode = "ab" if have else "wb"
        with open(part, mode) as handle:
            done = have
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                done += len(chunk)
                if progress and total:
                    elapsed = max(1e-6, time.time() - started)
                    progress(min(1.0, done / total), done, total, done / elapsed)
    size = part.stat().st_size
    if expected_min and size < expected_min:
        part.unlink(missing_ok=True)
        raise RuntimeError(
            f"Download of {dest.name} looks truncated ({size / 1e6:.0f} MB). "
            "Check the connection and retry; partial data was discarded."
        )
    part.replace(dest)
    log(f"  saved {dest.name} ({size / 1e6:.0f} MB)")
    return dest


def download_weights(progress=None, log=print, force: bool = False) -> Path:
    """Fetch config.yaml + model.ckpt into models/triposr (resumable)."""
    if force and weights_path().exists():
        weights_path().unlink()
    if not config_path().exists():
        _download_file(f"{TRIPOSR_BASE_URL}/{TRIPOSR_CONFIG}", config_path(),
                       log=log, expected_min=200)
    if not is_downloaded():
        _download_file(f"{TRIPOSR_BASE_URL}/{TRIPOSR_WEIGHTS}", weights_path(),
                       progress=progress, log=log, expected_min=TRIPOSR_MIN_BYTES)
    return weights_path()


# ---------------------------------------------------------------------- model
def _import_tsr():
    """Import the vendored network with an actionable error message."""
    try:
        from tsr.system import TSR
    except ModuleNotFoundError as exc:
        missing = missing_packages()
        hint = (
            f" Missing packages: {', '.join(missing)}."
            if missing
            else " Run: pip install -r requirements.txt"
        )
        raise RuntimeError(
            f"The single-image mesh backend needs extra packages ({exc}).{hint}"
        ) from exc
    return TSR


def load_model(device: str = "cpu", resolution: int | None = None,
               chunk_size: int = DEFAULT_CHUNK_SIZE, log=print,
               progress=None):
    """Load (and cache) the TripoSR network."""
    global _MODEL, _MODEL_DEVICE, _MODEL_RESOLUTION

    TSR = _import_tsr()
    if not is_downloaded():
        download_weights(log=log, progress=progress)
    if _MODEL is not None and _MODEL_DEVICE == device:
        _MODEL.renderer.set_chunk_size(chunk_size)
        return _MODEL

    log("Loading TripoSR weights ...")
    model = TSR.from_pretrained(
        str(TRIPOSR_DIR), config_name=TRIPOSR_CONFIG, weight_name=TRIPOSR_WEIGHTS
    )
    model.eval()
    model.to(device)
    # Match the official TripoSR default.  In particular, do not use the old
    # 262,144-point GPU chunk: it exceeds the safe working set on a 4 GB RTX
    # 2050 during density/color queries.
    model.renderer.set_chunk_size(chunk_size)
    _MODEL = model
    _MODEL_DEVICE = device
    _MODEL_RESOLUTION = resolution
    log("TripoSR ready.")
    return model


def reset_model() -> None:
    global _MODEL, _MODEL_DEVICE, _MODEL_RESOLUTION
    _MODEL = None
    _MODEL_DEVICE = None
    _MODEL_RESOLUTION = None


# ------------------------------------------------------------ image preparation
def prepare_condition_image(
    image_path: str | Path,
    rembg_session=None,
    remove_background: bool = True,
    foreground_ratio: float = DEFAULT_FOREGROUND_RATIO,
):
    """Apply the official TripoSR rembg + foreground-ratio preprocessing."""
    from PIL import Image
    from tsr.utils import remove_background as remove_bg, resize_foreground

    image = Image.open(image_path)
    if not remove_background:
        # Official --no-remove-bg mode expects an RGB image already composed
        # on the neutral background and already framed for the network.
        image = image.convert("RGB")
        return image, image
    image = remove_bg(image, rembg_session)
    image = resize_foreground(image, foreground_ratio)
    array = np.array(image).astype(np.float32) / 255.0
    rgb = array[:, :, :3] * array[:, :, 3:4] + (1.0 - array[:, :, 3:4]) * 0.5
    return Image.fromarray((rgb * 255.0).astype(np.uint8)), image


# ------------------------------------------------------------------ GLB writing
def write_glb(path: Path, vertices: np.ndarray, faces: np.ndarray,
              colors: np.ndarray | None = None,
              normals: np.ndarray | None = None) -> None:
    """Write a compact binary glTF with vertex colours.

    Hand-rolled so the exported file has exactly the attributes the web viewer
    reads (POSITION, NORMAL, COLOR_0, uint32 indices) and nothing else.
    """
    positions = np.ascontiguousarray(vertices, dtype=np.float32)
    indices = np.ascontiguousarray(faces.reshape(-1), dtype=np.uint32)
    if normals is None:
        normals = _vertex_normals(positions, indices)
    normals = np.ascontiguousarray(normals, dtype=np.float32)
    if colors is None:
        colors = np.full((len(positions), 3), 0.78, np.float32)
    colors = np.clip(np.asarray(colors, dtype=np.float32), 0.0, 1.0)
    color_bytes = np.ascontiguousarray(
        np.round(colors * 255.0).astype(np.uint8)
    )

    chunks: list[bytes] = []
    views: list[dict] = []

    def add_view(data: bytes, target: int) -> int:
        while len(chunks) and len(chunks[-1]) % 4:  # keep every view 4-aligned
            chunks[-1] += b"\x00"
        offset = sum(len(c) for c in chunks)
        views.append({"buffer": 0, "byteOffset": offset,
                      "byteLength": len(data), "target": target})
        chunks.append(data)
        return len(views) - 1

    pos_view = add_view(positions.tobytes(), 34962)
    nrm_view = add_view(normals.tobytes(), 34962)
    col_view = add_view(color_bytes.tobytes(), 34962)
    idx_view = add_view(indices.tobytes(), 34963)

    def accessor(count: int, kind: str, component: int,
                 view: int, normalized: bool = False) -> dict:
        return {
            "bufferView": view,
            "componentType": component,
            "count": int(count),
            "type": kind,
            **({"normalized": True} if normalized else {}),
        }

    acc = [
        {**accessor(len(positions), "VEC3", 5126, pos_view),
         "min": [float(v) for v in positions.min(axis=0)],
         "max": [float(v) for v in positions.max(axis=0)]},
        accessor(len(normals), "VEC3", 5126, nrm_view),
        accessor(len(color_bytes), "VEC3", 5121, col_view, normalized=True),
        accessor(len(indices), "SCALAR", 5125, idx_view),
    ]

    gltf = {
        "asset": {"version": "2.0", "generator": "kinesplat (TripoSR mesh)"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{
            "name": "reconstruction",
            "primitives": [{
                "attributes": {"POSITION": 0, "NORMAL": 1, "COLOR_0": 2},
                "indices": 3,
                "mode": 4,
            }],
        }],
        "accessors": acc,
        "bufferViews": views,
        "buffers": [{"byteLength": sum(len(c) for c in chunks)}],
    }

    js = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    js += b" " * (-len(js) % 4)
    bin_chunk = b"".join(chunks)
    bin_chunk += b"\x00" * (-len(bin_chunk) % 4)

    total = 12 + 8 + len(js) + 8 + len(bin_chunk)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(b"glTF")
        handle.write(np.array([2, total], dtype=np.uint32).tobytes())
        handle.write(np.array([len(js), 0x4E4F534A], dtype=np.uint32).tobytes())
        handle.write(js)
        handle.write(np.array([len(bin_chunk), 0x004E4942], dtype=np.uint32).tobytes())
        handle.write(bin_chunk)


def _vertex_normals(positions: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Area-weighted vertex normals (fallback when the exporter has none)."""
    tri = indices.reshape(-1, 3)
    v0 = positions[tri[:, 0]]
    v1 = positions[tri[:, 1]]
    v2 = positions[tri[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)
    normals = np.zeros_like(positions)
    for corner in range(3):
        np.add.at(normals, tri[:, corner], face_normals)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.maximum(lengths, 1e-9)


# ------------------------------------------------------------------ mesh space
# TripoSR predicts in a Z-up frame (its renderer uses up = +z).  The web viewer
# is Y-up, and its default camera sits on +Z, so this permutation puts +Z (the
# view the network was conditioned on) at +Z and +Z (up) at +Y.
_ZUP_TO_YUP = np.array([[0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                        [1.0, 0.0, 0.0]], dtype=np.float64)


def _orient(vertices: np.ndarray, normals: np.ndarray | None):
    vertices = vertices @ _ZUP_TO_YUP.T
    if normals is not None:
        normals = normals @ _ZUP_TO_YUP.T
    return vertices, normals


# ------------------------------------------------------------------ main entry
def reconstruct_mesh(
    image_path: str | Path,
    out_dir: str | Path,
    *,
    name: str = "Single photo",
    device: str = "cpu",
    resolution: int | None = None,
    threshold: float = DEFAULT_THRESHOLD,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    preview_views: int = 4,
    preview_size: int | None = None,
    remove_background: bool = True,
    foreground_ratio: float = DEFAULT_FOREGROUND_RATIO,
    log=print,
    progress=None,
    cancel_event=None,
) -> dict:
    """Reconstruct one photo into a coloured mesh and write a scene folder."""
    from PIL import Image
    import torch

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    def report(stage: str, pct: float, msg: str) -> None:
        log(msg)
        if progress:
            progress(stage, max(0.0, min(1.0, pct)), msg)

    def check_cancel():
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("Reconstruction cancelled")

    if resolution is None:
        # Official TripoSR default.  The renderer chunk, not a reduced grid,
        # is what keeps the CUDA working set bounded.
        resolution = DEFAULT_RESOLUTION
    if preview_size is None:
        preview_size = 256

    report("mesh", 0.06, "Cutting the subject out of the photo ...")
    rembg_session = None
    if remove_background:
        try:
            import rembg
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "The official TripoSR preprocessing requires rembg. "
                "Run .\\setup.ps1 -Gpu or install requirements.txt."
            ) from exc
        # Use the lightweight u2net model on CPU to avoid OOM on 4 GB GPUs.
        # BriaRMBG (default) needs ~800 MB for a single forward pass.
        rembg_session = rembg.new_session("u2net", providers=["CPUExecutionProvider"])
    condition, cutout = prepare_condition_image(
        image_path,
        rembg_session=rembg_session,
        remove_background=remove_background,
        foreground_ratio=foreground_ratio,
    )
    condition.save(out_dir / "condition.png")
    check_cancel()

    if not is_downloaded():
        report("mesh", 0.08, "Downloading the TripoSR model (one time, ~1.7 GB) ...")

        def on_download(p, done, total, speed):
            report("mesh", 0.08 + 0.42 * p,
                   f"TripoSR weights {done / 1e6:.0f}/{max(1, total) / 1e6:.0f} MB"
                   f" ({speed / 1e6:.1f} MB/s)")

        download_weights(progress=on_download, log=log)

    report("mesh", 0.52, "Loading the reconstruction network ...")
    model = load_model(device=device, chunk_size=chunk_size, log=log)
    check_cancel()

    report("mesh", 0.58, "Predicting the 3D shape ...")
    with torch.no_grad():
        # Keep the official call shape/type.  The image preprocessor inside
        # TSR handles PIL images and performs the 512px resize.
        scene_codes = model([condition], device=device)
    check_cancel()

    report("mesh", 0.70, f"Extracting the mesh ({resolution}³ grid) ...")
    meshes = model.extract_mesh(scene_codes, True, resolution=resolution,
                                threshold=threshold)
    mesh = meshes[0]
    check_cancel()

    report("mesh", 0.84, "Cleaning up the surface ...")
    try:
        mesh.update_faces(mesh.unique_faces() & mesh.nondegenerate_faces())
        mesh.remove_unreferenced_vertices()
    except Exception:
        pass
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if len(vertices) == 0 or len(faces) == 0:
        raise RuntimeError(
            "The network produced an empty surface. Try a photo with a single "
            "well-lit object against a plain background."
        )
    # Trimesh stores colors on ``visual.vertex_colors``.  Reading the old
    # top-level attribute silently selected the neutral gray fallback.
    visual_colors = getattr(getattr(mesh, "visual", None), "vertex_colors", None)
    colors = None
    if visual_colors is not None and len(visual_colors):
        raw_colors = np.asarray(visual_colors)
        if raw_colors.ndim == 2 and raw_colors.shape[1] >= 3:
            colors = raw_colors[:, :3].astype(np.float32)
            if float(np.nanmax(colors)) > 1.0 + 1e-5:
                colors /= 255.0
            colors = np.clip(colors, 0.0, 1.0)
    if colors is None or len(colors) != len(vertices):
        raise RuntimeError(
            "TripoSR returned no usable vertex colors; refusing to export a gray mesh."
        )
    normals = np.asarray(mesh.vertex_normals, dtype=np.float64)

    vertices, normals = _orient(vertices, normals)
    write_glb(out_dir / "mesh.glb", vertices, faces, colors, normals)
    try:
        import trimesh

        out_mesh = trimesh.Trimesh(
            vertices=vertices, faces=faces,
            vertex_colors=(np.clip(colors, 0, 1) * 255).astype(np.uint8)
            if colors is not None else None,
            process=False,
        )
        out_mesh.export(out_dir / "mesh.ply")
    except Exception as exc:
        log(f"  PLY export skipped ({exc})")

    check_cancel()

    # ------------------------------------------------------------- preview
    report("mesh", 0.90, "Rendering a preview ...")
    preview_ok = False
    try:
        shots = model.render(
            scene_codes, n_views=preview_views, elevation_deg=12.0,
            camera_distance=1.75, fovy_deg=40.0,
            height=preview_size, width=preview_size, return_type="pil",
        )[0]
        cols = 2 if preview_views >= 2 else 1
        rows = int(math.ceil(len(shots) / cols))
        sheet = Image.new("RGB", (preview_size * cols, preview_size * rows),
                          (30, 34, 38))
        for i, shot in enumerate(shots):
            sheet.paste(shot.convert("RGB"),
                        ((i % cols) * preview_size, (i // cols) * preview_size))
        sheet.save(out_dir / "preview.png")
        preview_ok = True
    except Exception as exc:
        log(f"  NeRF preview skipped ({exc})")

    photo = Image.open(image_path).convert("RGB")
    photo.thumbnail((960, 960))
    photo.save(out_dir / "poster.jpg", quality=92)
    if not preview_ok:
        photo.save(out_dir / "preview.png")

    # --------------------------------------------------------------- payload
    lo = vertices.min(axis=0)
    hi = vertices.max(axis=0)
    center = (lo + hi) * 0.5
    radius = float(np.linalg.norm(hi - lo) * 0.5) or 1.0
    metrics = {
        "backend": "triposr",
        "vertices": int(len(vertices)),
        "faces": int(len(faces)),
        "resolution": int(resolution),
        "threshold": float(threshold),
        "cond_image": 512,
        "seconds": round(time.time() - started, 1),
        "device": device,
        "source_image": Path(image_path).name,
        "license": "TripoSR (MIT) - Tripo AI / Stability AI",
    }
    payload = {
        "format": "splatlab-scene-v1",
        "kind": "mesh",
        "name": name,
        "splat": None,
        "mesh": "mesh.glb",
        "ply": "mesh.ply",
        "preview": "preview.png",
        "up": [0, 1, 0],
        "background": [0.12, 0.13, 0.15],
        "bounds": {
            "min": [float(v) for v in lo],
            "max": [float(v) for v in hi],
            "center": [float(v) for v in center],
            "radius": radius,
        },
        "cameras": [],
        "stats": metrics,
    }
    (out_dir / "scene.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    report("done", 1.0,
           f"Done: mesh with {metrics['faces']:,} faces")
    return {
        "kind": "mesh",
        "scene": "output/scene.json",
        "splat": None,
        "mesh": "output/mesh.glb",
        "ply": "output/mesh.ply",
        "preview": "output/preview.png" if preview_ok else "output/poster.jpg",
        "poster": "output/poster.jpg",
        "metrics": metrics,
        "gaussians": 0,
        "cameras": 0,
        "seconds": metrics["seconds"],
    }


def estimate_cost(device: str = "cpu") -> str:
    """Human-readable note used by the CLI/UI when the mesh path is chosen."""
    return ("~2-4 minutes on CPU" if device == "cpu" else "~20-40 seconds on GPU")
