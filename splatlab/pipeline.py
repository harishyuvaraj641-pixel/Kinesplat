"""End-to-end reconstruction pipeline: frames -> poses -> depth -> splats."""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import exporter, gaussians, poses, trainer
from .capture import imread_unicode, imwrite_unicode, list_inputs, prepare_frames
from .config import Preset, get_preset
from .depth import DepthEstimator, ModelUnavailable, download_depth_model
from .device import device_report, label, resolve_device, tune_preset
from .types import Camera, Scene


class Cancelled(RuntimeError):
    pass


@dataclass
class PipelineConfig:
    name: str = "Untitled scene"
    preset: str = "balanced"
    backend: str = "auto"               # auto | splat | mesh
    pose_backend: str = "auto"          # auto | sfm | colmap
    allow_colmap_download: bool = False
    use_depth: bool = True
    keep_debug: bool = False
    max_side: int = 1280
    threads: int | None = None
    device: str = "auto"              # auto | cpu | cuda
    mesh_resolution: int | None = None
    extra: dict = field(default_factory=dict)


def _fit_depth_alignment(disp: np.ndarray, points: np.ndarray,
                         cam: Camera) -> tuple[float, float, int]:
    """Fit 1/z = a*disp + b using the sparse SfM/COLMAP points visible in a frame."""
    h, w = disp.shape
    if points is not None and len(points) > 0:
        from .geometry import project_points
        u, v, z = project_points(points, cam.R, cam.t, cam.K)
        ok = (z > 0.05) & (u >= 0) & (u < w - 1) & (v >= 0) & (v < h - 1)
        if ok.sum() >= 6:
            ui = np.clip(np.round(u[ok]).astype(int), 0, w - 1)
            vi = np.clip(np.round(v[ok]).astype(int), 0, h - 1)
            d = disp[vi, ui].astype(np.float64)
            target = 1.0 / z[ok].astype(np.float64)
            A = np.stack([d, np.ones_like(d)], axis=1)
            good = np.isfinite(target) & np.isfinite(d)
            a, b = 0.0, 0.0
            for _ in range(4):
                if good.sum() < 5:
                    break
                try:
                    sol, *_ = np.linalg.lstsq(A[good], target[good], rcond=None)
                except np.linalg.LinAlgError:
                    break
                a, b = float(sol[0]), float(sol[1])
                pred = A @ sol
                res = pred - target
                med = np.median(res[good])
                mad = np.median(np.abs(res[good] - med)) * 1.4826
                thresh = max(3.0 * mad, 0.015)
                new_good = np.abs(res - med) < thresh
                if new_good.sum() < 6:
                    break
                good = new_good
            if a > 1e-6 and np.isfinite(a) and np.isfinite(b):
                depth = 1.0 / np.clip(a * disp + b, 1e-6, None)
                if np.median(depth) < 35.0:
                    return a, b, int(good.sum())

    # No usable sparse points: map the disparity percentiles onto a plausible
    # near/far range around the camera's distance from the scene centre.
    lo, hi = np.percentile(disp, [5.0, 98.0])
    d = max(float(np.linalg.norm(cam.t)), 0.4)
    z_near = max(0.25, 0.55 * d)
    z_far = max(z_near * 3.0, 3.5 * d)
    if hi - lo < 1e-6:
        a = 1.0 / (z_near * max(float(np.median(disp)), 1e-6))
        return a, 0.0, 0
    a = (1.0 / z_near - 1.0 / z_far) / (hi - lo)
    b = 1.0 / z_far - a * lo
    return float(max(a, 1e-6)), float(b), 0


def _apply_depth_alignment(disp: np.ndarray, a: float, b: float,
                           max_depth: float = 60.0) -> np.ndarray:
    depth = 1.0 / np.clip(a * disp + b, 1e-6, None)
    return np.clip(depth, 0.015, max_depth).astype(np.float32)


def _estimate_background(bgr: np.ndarray) -> np.ndarray:
    """Median colour of the upper border, in RGB [0,1].

    Most captures put sky, ceiling or wall at the top of the frame.  Using it
    as the render background is much cleaner than asking distant splats to
    cover the whole backdrop.
    """
    h, w = bgr.shape[:2]
    band = max(2, h // 10)
    side = max(2, w // 12)
    parts = [
        bgr[:band].reshape(-1, 3),
        bgr[: h // 2, :side].reshape(-1, 3),
        bgr[: h // 2, w - side:].reshape(-1, 3),
    ]
    med = np.median(np.concatenate(parts, axis=0), axis=0)
    return np.clip(med[::-1].astype(np.float32) / 255.0, 0.0, 1.0)


def _resize_bgr(path: Path, width: int, height: int) -> np.ndarray:
    img = imread_unicode(path)
    if img is None:
        raise RuntimeError(f"Could not read image: {path}")
    if img.shape[1] != width or img.shape[0] != height:
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    return img


def run_reconstruction(
    workdir: Path,
    input_paths: list[str | Path],
    config: PipelineConfig | None = None,
    log=print,
    progress=None,
    cancel_event=None,
) -> dict:
    cfg = config or PipelineConfig()
    device = resolve_device(cfg.device)
    preset = get_preset(cfg.preset)
    preset, tuned = tune_preset(preset, device)
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    def report(stage: str, pct: float, msg: str) -> None:
        log(msg)
        if progress:
            progress(stage, max(0.0, min(1.0, pct)), msg)

    log(f"Compute: {label(device)}")
    if tuned:
        log("  VRAM guard: " + ", ".join(tuned))

    def check_cancel():
        if cancel_event is not None and cancel_event.is_set():
            raise Cancelled("Reconstruction cancelled")

    # ------------------------------------------------- single-image mesh path
    # The multi-view pipeline needs at least two photographs to triangulate
    # anything.  One upload can only be handled by the feed-forward network, so
    # route it there instead of failing with "two views are needed".
    images, videos = list_inputs(input_paths)
    available = len(images) + len(videos)
    if available == 0:
        raise RuntimeError(
            "No supported images or videos found. Add JPG/PNG photos or an MP4/MOV video."
        )

    def run_mesh(reason: str, source: Path) -> dict:
        from .mesh import reconstruct_mesh

        log(f"  {reason} Using the single-image TripoSR backend.")
        report("mesh", 0.05, "Single photo: reconstructing a mesh ...")
        return reconstruct_mesh(
            source, workdir / "output",
            name=cfg.name,
            device=device.type,
            resolution=cfg.mesh_resolution,
            log=log,
            progress=progress,
            cancel_event=cancel_event,
        )

    backend = (cfg.backend or "auto").lower()
    if backend == "mesh" or (backend == "auto" and available < 2):
        source = Path(images[0]) if images else Path(input_paths[0])
        if backend == "mesh":
            reason = "Mesh backend requested."
        else:
            reason = "Only one capture was provided, so multi-view poses are impossible."
        return run_mesh(reason, source)

    # ------------------------------------------------------------------ frames
    report("frames", 0.03, "Preparing frames ...")
    frames_dir = workdir / "input"
    frames = prepare_frames(
        input_paths, frames_dir,
        max_frames=preset.max_frames, max_side=cfg.max_side, log=log,
    )
    check_cancel()
    report("frames", 0.10, f"{len(frames)} frames ready")

    # ------------------------------------------------------------------- poses
    try:
        scene: Scene = poses.estimate(
            frames_dir, workdir / "poses", preset,
            backend=cfg.pose_backend,
            allow_colmap_download=cfg.allow_colmap_download,
            log=log,
            progress=lambda f, msg: report("poses", 0.10 + 0.25 * f, msg),
            cancel_event=cancel_event,
        )
    except Cancelled:
        raise
    except Exception as pose_error:
        if backend == "splat":
            raise
        # A weak capture (blurry frames, no parallax, a single viewpoint) cannot
        # be triangulated.  One clean photo still gives a usable mesh, which is
        # far more helpful than an error dialog.
        source = Path(images[0]) if images else Path(frames[0])
        log(f"  Camera estimation failed: {pose_error}")
        return run_mesh("This capture could not be posed.", source)
    check_cancel()

    # -------------------------------------------------------- training images
    ref = scene.cameras[0]
    long_side = preset.train_size
    if ref.width >= ref.height:
        tw = long_side
        th = max(64, int(round(long_side * ref.height / max(1, ref.width))))
    else:
        th = long_side
        tw = max(64, int(round(long_side * ref.width / max(1, ref.height))))
    train_dir = workdir / "train"
    train_dir.mkdir(parents=True, exist_ok=True)
    train_cams: list[Camera] = []
    train_images: list[np.ndarray] = []
    train_paths: list[Path] = []
    for cam in scene.cameras:
        src = Path(cam.image_path)
        if not src.exists():
            log(f"  Missing registered image {src.name}; skipping.")
            continue
        img = _resize_bgr(src, tw, th)
        cam2 = cam.scaled(tw, th)
        cam2.image_path = str(src)
        dest = train_dir / f"frame_{len(train_cams):03d}.png"
        imwrite_unicode(dest, img)
        train_cams.append(cam2)
        train_images.append(img)
        train_paths.append(dest)
    if len(train_cams) < 3:
        raise RuntimeError("Fewer than three registered frames are usable.")
    backgrounds = [_estimate_background(img) for img in train_images]
    scene_background = np.mean(backgrounds, axis=0)
    report("depth", 0.36, f"Training at {tw}x{th} with {len(train_cams)} frames")

    # ------------------------------------------------------------------- depth
    depths: list[np.ndarray] = []
    if cfg.use_depth:
        estimator: DepthEstimator | None = None
        depth_dir = workdir / "depth"
        depth_dir.mkdir(parents=True, exist_ok=True)

        def ensure_estimator() -> DepthEstimator:
            nonlocal estimator
            if estimator is None:
                estimator = DepthEstimator(threads=cfg.threads)
            if not estimator.available():
                report("depth", 0.37, "Downloading pretrained depth model (one time, ~95 MB) ...")
                try:
                    download_depth_model(
                        progress=lambda p, done, total, speed: report(
                            "depth", 0.37 + 0.06 * p,
                            f"Depth model {done / 1e6:.0f}/{max(1, total) / 1e6:.0f} MB"
                        )
                    )
                except Exception as exc:
                    raise ModelUnavailable(
                        "The pretrained depth model is required for automatic "
                        "initialisation but could not be downloaded. Check your "
                        "internet connection and retry."
                    ) from exc
            estimator.ensure()
            return estimator

        for idx, (cam, img) in enumerate(zip(train_cams, train_images)):
            check_cancel()
            cache = depth_dir / f"depth_{idx:03d}.npy"
            if cam.depth_path:
                try:
                    bundled = np.load(cam.depth_path).astype(np.float32)
                    if bundled.ndim != 2:
                        raise ValueError("depth map must be 2-D")
                    if bundled.shape != (th, tw):
                        # Never interpolate depth across a silhouette: an
                        # averaged edge value unprojects to empty space and
                        # shows up as floating splats from every view.
                        bundled = cv2.resize(bundled, (tw, th), interpolation=cv2.INTER_NEAREST)
                    # Bundled depths live in the capture's original world frame;
                    # pose normalisation rescaled the cameras, so match it.
                    depth = np.clip(bundled * float(scene.scale), 0.01, 80.0)
                    depths.append(depth)
                    report("depth", 0.43 + 0.07 * (idx + 1) / len(train_cams),
                           f"Depth {idx + 1}/{len(train_cams)} (bundled ground truth)")
                    continue
                except Exception as exc:
                    log(f"  Bundled depth unusable for {cam.name} ({exc}); estimating it instead.")
            if cache.exists():
                depths.append(np.load(cache))
                continue
            est = ensure_estimator()
            disp = est.predict(img)
            a, b, npts = _fit_depth_alignment(disp, scene.points, cam)
            max_depth = max(2.0, 4.0 * float(np.linalg.norm(cam.t)))
            depth = _apply_depth_alignment(disp, a, b, max_depth=max_depth)
            np.save(cache, depth)
            depths.append(depth)
            report(
                "depth", 0.43 + 0.07 * (idx + 1) / len(train_cams),
                f"Depth {idx + 1}/{len(train_cams)}"
                + (f" (aligned to {npts} sparse points)" if npts else " (relative scale)"),
            )
    else:
        for cam in train_cams:
            h, w = th, tw
            yy, xx = np.mgrid[0:h, 0:w]
            depths.append((1.8 + 0.4 * (yy / max(1, h))).astype(np.float32))

    # ------------------------------------------------------------------- train
    report("init", 0.51, "Initialising Gaussian splats from depth ...")
    model = gaussians.initialise(
        scene, train_cams, train_images, depths, preset,
        backgrounds=backgrounds, log=log, device=device,
    )
    threads = trainer.configure_cpu(cfg.threads)
    log(f"PyTorch CPU threads: {threads}")
    report("train", 0.56, f"Optimising {model.count:,} Gaussians on {label(device)} ...")
    metrics = trainer.train(
        model, train_cams, train_images, preset, workdir / "output",
        log=log,
        on_progress=lambda f, msg: report("train", 0.56 + 0.36 * f, msg),
        cancel_event=cancel_event,
        backgrounds=backgrounds,
        device=device,
    )
    check_cancel()

    # ------------------------------------------------------------------ export
    report("export", 0.94, "Exporting splats ...")
    out_dir = workdir / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    exporter.write_splat(out_dir / "model.splat", model)
    exporter.write_ply(out_dir / "model.ply", model)
    (out_dir / "cameras.json").write_text(
        json.dumps([c.to_dict() for c in train_cams], indent=2), encoding="utf-8"
    )
    metrics.update({
        "preset": preset.key,
        "pose_source": scene.source,
        "cameras": len(train_cams),
        "sparse_points": int(len(scene.points)),
        "total_seconds": round(time.time() - started, 1),
        "width": tw,
        "height": th,
        "device": device_report(device),
    })
    payload = exporter.scene_payload(
        model, train_cams, metrics, cfg.name,
        background=[float(x) for x in scene_background],
    )
    (out_dir / "scene.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Poster frame for the library UI.
    poster = cv2.resize(train_images[0], (min(960, tw * 3), min(720, th * 3)),
                        interpolation=cv2.INTER_NEAREST)
    imwrite_unicode(out_dir / "poster.jpg", poster, quality=92)
    if (out_dir / "preview.png").exists():
        try:
            prev = cv2.imread(str(out_dir / "preview.png"))
            if prev is not None:
                cv2.imwrite(str(out_dir / "preview.jpg"), prev)
        except Exception:
            pass

    if not cfg.keep_debug:
        shutil.rmtree(workdir / "poses" / "colmap", ignore_errors=True)

    report("done", 1.0, f"Done: {model.count:,} splats, {metrics['psnr']:.1f} dB PSNR")
    return {
        "scene": "output/scene.json",
        "splat": "output/model.splat",
        "ply": "output/model.ply",
        "preview": "output/preview.png" if (out_dir / "preview.png").exists() else "output/poster.jpg",
        "poster": "output/poster.jpg",
        "metrics": metrics,
        "gaussians": model.count,
        "cameras": len(train_cams),
        "seconds": metrics["total_seconds"],
    }
