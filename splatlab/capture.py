"""Turn uploaded videos / photos into a clean set of reconstruction frames."""

from __future__ import annotations

import json
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from .config import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS


def _rational(value) -> float | None:
    try:
        if hasattr(value, "numerator") and hasattr(value, "denominator"):
            return float(value.numerator) / float(value.denominator or 1)
        return float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def exif_focal_px(path: str | Path, out_width: int) -> float | None:
    """Best-effort pinhole focal length (pixels) from EXIF, for ``out_width``."""
    try:
        with Image.open(path) as img:
            exif = img.getexif()
            tags: dict = dict(exif)
            try:
                tags.update(exif.get_ifd(34665))  # Exif sub-IFD
            except Exception:
                pass
    except Exception:
        return None

    f35 = _rational(tags.get(41989))  # FocalLengthIn35mmFilm
    if f35 and f35 > 1.0:
        return float(f35) * out_width / 36.0

    focal = _rational(tags.get(37386))  # FocalLength (mm)
    if not focal or focal <= 0:
        return None
    res_x = _rational(tags.get(41486))  # FocalPlaneXResolution
    px_x = _rational(tags.get(40962))   # PixelXDimension
    unit = tags.get(41488)              # 2 = inch, 3 = cm
    if res_x and px_x and unit in (2, 3):
        unit_mm = 25.4 if unit == 2 else 10.0
        sensor_w = px_x * unit_mm / res_x
        if 2.0 < sensor_w < 80.0:
            return focal * out_width / sensor_w
    # Unknown sensor: assume a 36 mm full-frame width (sane for most photos).
    return focal * out_width / 36.0


def imread_unicode(path: str | Path) -> np.ndarray | None:
    """cv2.imread replacement that works with non-ASCII Windows paths."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def imwrite_unicode(path: str | Path, image: np.ndarray, quality: int = 95) -> bool:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower() or ".jpg"
    params = [int(cv2.IMWRITE_JPEG_QUALITY), quality] if ext in (".jpg", ".jpeg") else []
    ok, buf = cv2.imencode(ext, image, params)
    if not ok:
        return False
    buf.tofile(str(path))
    return True


def _load_pil_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        return np.array(img)


def _resize_max(image: np.ndarray, max_side: int) -> np.ndarray:
    h, w = image.shape[:2]
    scale = min(1.0, max_side / float(max(h, w)))
    if scale >= 0.999:
        return image
    size = (max(2, round(w * scale)), max(2, round(h * scale)))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def _safe_stem(path: Path) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", path.stem)
    return stem[:48] or "frame"


def list_inputs(paths: list[str | Path]) -> tuple[list[Path], list[Path]]:
    images: list[Path] = []
    videos: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            for child in sorted(p.iterdir()):
                if child.suffix.lower() in IMAGE_EXTENSIONS:
                    images.append(child)
                elif child.suffix.lower() in VIDEO_EXTENSIONS:
                    videos.append(child)
        elif p.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(p)
        elif p.suffix.lower() in VIDEO_EXTENSIONS:
            videos.append(p)
    # Stable, natural ordering (frame_2 before frame_10)
    images.sort(key=lambda p: [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", p.name)])
    return images, videos


def _sample_indices(count: int, limit: int) -> list[int]:
    if count <= limit:
        return list(range(count))
    # Always include the first and last frame, evenly space the rest.
    return sorted({int(round(i * (count - 1) / (limit - 1))) for i in range(limit)})


def extract_video_frames(
    video_path: Path,
    out_dir: Path,
    max_frames: int,
    max_side: int,
) -> list[Path]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path.name}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0

    frames: list[tuple[int, np.ndarray]] = []
    if total > 1:
        wanted = _sample_indices(total, max_frames)
        for idx in wanted:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if ok and frame is not None:
                frames.append((idx, frame))
        # Seeking can fail on some codecs; fall back to sequential sampling.
        if len(frames) < max(3, min(max_frames, total) // 2):
            frames = []
            cap.release()
            cap = cv2.VideoCapture(str(video_path))
            step = max(1, total // max_frames)
            n = 0
            while len(frames) < max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                if n % step == 0:
                    frames.append((n, frame))
                n += 1
    else:
        n = 0
        while len(frames) < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append((n, frame))
            n += 1
    cap.release()

    if not frames:
        raise RuntimeError(f"No frames could be decoded from {video_path.name}")

    out: list[Path] = []
    for order, (idx, frame) in enumerate(frames):
        frame = _resize_max(frame, max_side)
        dest = out_dir / f"{Path(video_path).stem[:24]}_{order:03d}.jpg"
        if imwrite_unicode(dest, frame, quality=94):
            out.append(dest)
    if fps and total:
        duration = total / max(1e-6, fps)
        print(f"  video {video_path.name}: {total} frames / {duration:.1f}s -> {len(out)} keyframes")
    return out


def prepare_frames(
    input_paths: list[str | Path],
    out_dir: Path,
    max_frames: int,
    max_side: int = 1280,
    log=print,
) -> list[Path]:
    """Extract and normalise frames.  Returns the saved image paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    images, videos = list_inputs(input_paths)
    if not images and not videos:
        raise RuntimeError(
            "No supported images or videos found. Add JPG/PNG photos or an MP4/MOV video."
        )

    produced: list[Path] = []
    records: list[dict] = []
    for video in videos:
        log(f"Decoding video {video.name} ...")
        video_frames = extract_video_frames(video, out_dir, max_frames, max_side)
        produced.extend(video_frames)
        for p in video_frames:
            frame = imread_unicode(p)
            w = int(frame.shape[1]) if frame is not None else 0
            h = int(frame.shape[0]) if frame is not None else 0
            records.append({"output": p.name, "source": str(video),
                            "focal_px": None, "width": w, "height": h})

    if images:
        chosen = [images[i] for i in _sample_indices(len(images), max_frames)]
        for i, src in enumerate(chosen):
            try:
                rgb = _load_pil_rgb(src)
            except Exception as exc:
                log(f"  skipping {src.name}: {exc}")
                continue
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            bgr = _resize_max(bgr, max_side)
            dest = out_dir / f"photo_{i:03d}_{_safe_stem(src)}.jpg"
            if imwrite_unicode(dest, bgr, quality=95):
                produced.append(dest)
                records.append({
                    "output": dest.name,
                    "source": str(src),
                    "focal_px": exif_focal_px(src, bgr.shape[1]),
                    "width": int(bgr.shape[1]),
                    "height": int(bgr.shape[0]),
                })
        log(f"Prepared {len(chosen)} of {len(images)} photos.")

    # Deduplicate / cap total frames.
    if len(produced) > max_frames:
        keep = _sample_indices(len(produced), max_frames)
        produced = [produced[i] for i in keep]
        records = [records[i] for i in keep]

    if len(produced) < 2:
        raise RuntimeError(
            "At least two different views are needed. "
            "Use a short video that walks around the scene, or several photos."
        )

    focals = [r["focal_px"] for r in records if r.get("focal_px")]
    focal_px = float(np.median(focals)) if focals else None
    if focal_px and len(focals) >= 3:
        spread = float(np.std(focals) / max(np.median(focals), 1e-9))
        if spread > 0.25:
            log(f"  EXIF focal lengths vary ({spread:.0%}); ignoring them.")
            focal_px = None
    (out_dir / "frames_meta.json").write_text(
        json.dumps({
            "frames": records,
            "focal_px": focal_px,
            "sources": sorted({r["source"] for r in records}),
        }, indent=2),
        encoding="utf-8",
    )

    log(f"Using {len(produced)} frames for reconstruction.")
    if focal_px:
        log(f"  EXIF focal length: {focal_px:.1f}px")
    return produced
