"""COLMAP integration (auto-installed, CPU-only build on Windows).

COLMAP gives us robust camera poses + a sparse point cloud.  It is optional:
the app also ships a lightweight sequential-SfM fallback for offline use.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

from .config import COLMAP_VERSION, COLMAP_ZIP_URL, TOOLS_DIR
from .types import Camera, Scene


class ColmapError(RuntimeError):
    pass


def find_colmap(root: Path = TOOLS_DIR) -> Path | None:
    env = os.environ.get("COLMAP_BIN")
    if env and Path(env).exists():
        return Path(env)
    which = shutil.which("colmap")
    if which:
        return Path(which)
    for candidate in sorted(root.rglob("colmap.exe")):
        return candidate
    return None


def install_colmap(progress=None, log=print, force: bool = False) -> Path:
    existing = find_colmap()
    if existing and not force:
        return existing
    TOOLS_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = TOOLS_DIR / f"colmap-{COLMAP_VERSION}-nocuda.zip"
    dest = TOOLS_DIR / "colmap"
    if force and dest.exists():
        shutil.rmtree(dest, ignore_errors=True)
    if not zip_path.exists() or zip_path.stat().st_size < 50_000_000:
        log(f"Downloading COLMAP {COLMAP_VERSION} (CPU build, ~122 MB) ...")
        request = urllib.request.Request(
            COLMAP_ZIP_URL, headers={"User-Agent": "kinesplat/1.0"}
        )
        start = time.time()
        with urllib.request.urlopen(request, timeout=60) as response:
            total = int(response.headers.get("Content-Length") or 0)
            tmp = zip_path.with_suffix(".part")
            with open(tmp, "wb") as handle:
                done = 0
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
                    done += len(chunk)
                    if progress and total:
                        progress(done / total, f"COLMAP download {done / 1e6:.0f} MB")
            tmp.replace(zip_path)

    log("Extracting COLMAP ...")
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(dest)
    exe = find_colmap(dest)
    if not exe:
        raise ColmapError("COLMAP was extracted but colmap.exe could not be found.")
    # The zip is no longer needed; keep the extracted tools only.
    zip_path.unlink(missing_ok=True)
    log(f"COLMAP ready: {exe}")
    return exe


def _help_text(exe: Path, command: str) -> str:
    try:
        proc = subprocess.run(
            [str(exe), command, "-h"],
            capture_output=True, text=True, errors="replace", timeout=60,
            env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
        )
        return (proc.stdout or "") + (proc.stderr or "")
    except Exception:
        return ""


def _gpu_flags(help_text: str, kind: str) -> list[str]:
    """Pick the right CPU flags for this COLMAP version."""
    flags: list[str] = []
    if "SiftExtraction.use_gpu" in help_text:
        flags += ["--SiftExtraction.use_gpu", "0"]
    if "FeatureExtraction.use_gpu" in help_text:
        flags += ["--FeatureExtraction.use_gpu", "0"]
    if "SiftMatching.use_gpu" in help_text:
        flags += ["--SiftMatching.use_gpu", "0"]
    if "FeatureMatching.use_gpu" in help_text:
        flags += ["--FeatureMatching.use_gpu", "0"]
    return flags


def _run(exe: Path, args: list[str], log=print, cancel_event=None, timeout: int = 3600) -> str:
    cmd = [str(exe)] + args
    log("  $ " + " ".join(args[:5]) + (" ..." if len(args) > 5 else ""))
    env = {**os.environ, "QT_QPA_PLATFORM": "offscreen"}
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        errors="replace", env=env,
    )
    tail: list[str] = []
    assert proc.stdout is not None
    start = time.time()
    for line in proc.stdout:
        line = line.strip()
        if line:
            tail.append(line)
            if len(tail) > 30:
                tail.pop(0)
        if cancel_event is not None and cancel_event.is_set():
            proc.kill()
            raise ColmapError("Cancelled")
        if time.time() - start > timeout:
            proc.kill()
            raise ColmapError("COLMAP timed out")
    code = proc.wait()
    if code != 0:
        raise ColmapError("COLMAP failed:\n" + "\n".join(tail[-12:]))
    return "\n".join(tail)


def _parse_camera_line(parts: list[str]) -> tuple[str, int, int, float, float, float, float, int]:
    cam_id = int(parts[0])
    model = parts[1]
    width, height = int(parts[2]), int(parts[3])
    params = [float(x) for x in parts[4:]]
    if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
        f, cx, cy = params[0], params[1], params[2]
        return model, width, height, f, f, cx, cy, cam_id
    if model in ("PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV"):
        fx, fy, cx, cy = params[0], params[1], params[2], params[3]
        return model, width, height, fx, fy, cx, cy, cam_id
    raise ColmapError(f"Unsupported COLMAP camera model: {model}")


def _quat_to_rotmat(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    n = qw * qw + qx * qx + qy * qy + qz * qz
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    return np.array(
        [
            [1 - s * (qy * qy + qz * qz), s * (qx * qy - qz * qw), s * (qx * qz + qy * qw)],
            [s * (qx * qy + qz * qw), 1 - s * (qx * qx + qz * qz), s * (qy * qz - qx * qw)],
            [s * (qx * qz - qy * qw), s * (qy * qz + qx * qw), 1 - s * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


def read_model(model_dir: Path) -> tuple[list[Camera], np.ndarray, np.ndarray]:
    cam_file = model_dir / "cameras.txt"
    img_file = model_dir / "images.txt"
    pts_file = model_dir / "points3D.txt"
    if not cam_file.exists():
        raise ColmapError(f"Missing {cam_file}. COLMAP model conversion failed.")

    cameras_meta: dict[int, tuple] = {}
    for line in cam_file.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        try:
            model, w, h, fx, fy, cx, cy, cid = _parse_camera_line(parts)
        except Exception:
            continue
        cameras_meta[cid] = (model, w, h, fx, fy, cx, cy)

    cameras: list[Camera] = []
    lines = img_file.read_text(encoding="utf-8", errors="replace").splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith("#"):
            i += 1
            continue
        parts = line.split()
        if len(parts) < 10:
            i += 1
            continue
        try:
            image_id = int(parts[0])
            qw, qx, qy, qz = map(float, parts[1:5])
            tx, ty, tz = map(float, parts[5:8])
            cam_id = int(parts[8])
            name = " ".join(parts[9:])
        except Exception:
            i += 1
            continue
        i += 2  # skip the POINTS2D line
        if cam_id not in cameras_meta:
            continue
        model, w, h, fx, fy, cx, cy = cameras_meta[cam_id]
        Rwc = _quat_to_rotmat(qw, qx, qy, qz)
        Rcw = Rwc.T
        tcw = np.array([tx, ty, tz], dtype=np.float64)
        c2w_t = -Rcw @ tcw
        cameras.append(
            Camera(
                name=name, width=w, height=h,
                fx=fx, fy=fy, cx=cx, cy=cy,
                R=Rcw, t=c2w_t,
            )
        )

    points: list[list[float]] = []
    colors: list[list[float]] = []
    if pts_file.exists():
        for line in pts_file.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            try:
                points.append([float(parts[1]), float(parts[2]), float(parts[3])])
                colors.append([float(parts[4]) / 255.0, float(parts[5]) / 255.0,
                               float(parts[6]) / 255.0])
            except ValueError:
                continue
    return (
        cameras,
        np.asarray(points, dtype=np.float32).reshape(-1, 3),
        np.asarray(colors, dtype=np.float32).reshape(-1, 3),
    )


def reconstruct(
    images_dir: Path,
    workspace: Path,
    log=print,
    progress=None,
    cancel_event=None,
    matcher: str = "auto",
    install: bool = True,
) -> Scene:
    exe = find_colmap()
    if exe is None and install:
        if progress:
            progress(0.02, "Installing COLMAP (one time) ...")
        exe = install_colmap(
            progress=(lambda p, m: progress(0.02 + 0.18 * p, m)) if progress else None,
            log=log,
        )
    if exe is None:
        raise ColmapError("COLMAP is not installed. Run `python -m splatlab setup` first.")

    workspace.mkdir(parents=True, exist_ok=True)
    db = workspace / "database.db"
    sparse = workspace / "sparse"
    sparse.mkdir(exist_ok=True)
    log_dir = workspace / "colmap.log"

    images = sorted([p for p in Path(images_dir).iterdir() if p.suffix.lower() in
                     {".jpg", ".jpeg", ".png", ".webp"}])
    if len(images) < 2:
        raise ColmapError("Need at least two images for COLMAP.")

    feat_help = _help_text(exe, "feature_extractor")
    match_help = _help_text(exe, "exhaustive_matcher")
    cpu_flags = _gpu_flags(feat_help + match_help, "all")

    def stage(p: float, msg: str):
        log(msg)
        if progress:
            progress(p, msg)

    stage(0.20, f"COLMAP: extracting features from {len(images)} frames ...")
    feat_args = [
        "feature_extractor",
        "--database_path", str(db),
        "--image_path", str(images_dir),
        "--ImageReader.single_camera", "1",
        "--ImageReader.camera_model", "OPENCV",
        "--SiftExtraction.max_image_size", "1600",
        "--SiftExtraction.max_num_features", "8192",
    ] + cpu_flags
    try:
        _run(exe, feat_args, log=log, cancel_event=cancel_event)
    except ColmapError:
        if cancel_event is not None and cancel_event.is_set():
            raise
        # Retry with the most compatible minimal flags.
        _run(exe, ["feature_extractor", "--database_path", str(db),
                   "--image_path", str(images_dir)], log=log, cancel_event=cancel_event)

    stage(0.30, "COLMAP: matching frames ...")
    use_exhaustive = matcher == "exhaustive" or (matcher == "auto" and len(images) <= 60)
    if use_exhaustive:
        args = ["exhaustive_matcher", "--database_path", str(db)]
    else:
        args = ["sequential_matcher", "--database_path", str(db),
                "--SequentialMatching.overlap", "12",
                "--SequentialMatching.loop_detection", "1"]
    args += [f for f in cpu_flags if f.startswith("--SiftMatching") or f.startswith("--FeatureMatching")]
    try:
        _run(exe, args, log=log, cancel_event=cancel_event)
    except ColmapError:
        if cancel_event is not None and cancel_event.is_set():
            raise
        _run(exe, ["exhaustive_matcher", "--database_path", str(db)],
             log=log, cancel_event=cancel_event)

    stage(0.40, "COLMAP: reconstructing camera poses (this is the slow part) ...")
    _run(
        exe,
        ["mapper", "--database_path", str(db), "--image_path", str(images_dir),
         "--output_path", str(sparse),
         "--Mapper.ba_refine_principal_point", "0",
         "--Mapper.min_num_matches", "12"],
        log=log, cancel_event=cancel_event,
    )

    models = sorted([p for p in sparse.iterdir() if p.is_dir()])
    if not models:
        raise ColmapError("COLMAP could not register any cameras. "
                          "Try a slower capture with more overlap, or use the 'draft' preset.")
    # The number of registered images is a better metric than file size.
    def model_size(p: Path) -> int:
        txt = p / "images.txt"
        if txt.exists():
            return sum(1 for line in txt.read_text(errors="replace").splitlines()
                       if line and not line.startswith("#") and len(line.split()) > 9) // 2
        return len(list(p.glob("images.*")))
    best = max(models, key=model_size)

    stage(0.78, "COLMAP: undistorting frames ...")
    undist = workspace / "undistorted"
    if undist.exists():
        shutil.rmtree(undist, ignore_errors=True)
    _run(
        exe,
        ["image_undistorter", "--image_path", str(images_dir),
         "--input_path", str(best), "--output_path", str(undist),
         "--output_type", "COLMAP", "--max_image_size", "1600"],
        log=log, cancel_event=cancel_event,
    )

    # Convert the undistorted model to text.
    model_src = undist / "sparse"
    txt_out = workspace / "model_txt"
    if txt_out.exists():
        shutil.rmtree(txt_out)
    _run(
        exe,
        ["model_converter", "--input_path", str(model_src),
         "--output_path", str(txt_out), "--output_type", "TXT"],
        log=log, cancel_event=cancel_event,
    )

    cameras, points, colors = read_model(txt_out)
    if not cameras:
        raise ColmapError("COLMAP model contained no cameras.")
    image_root = undist / "images"
    for cam in cameras:
        cam.image_path = str(image_root / cam.name)

    # Simplify: if multiple cameras slipped in (rare after single_camera),
    # keep the largest coherent set.
    cameras.sort(key=lambda c: c.name)
    scene = Scene(cameras=cameras, points=points, colors=colors, source="colmap")
    (workspace / "scene.json").write_text(
        json.dumps({
            "source": "colmap",
            "cameras": [c.to_dict() for c in cameras],
            "points": len(points),
        }, indent=2),
        encoding="utf-8",
    )
    (workspace / "colmap.log").write_text(log_dir.read_text(errors="replace")
                                          if log_dir.exists() else "", encoding="utf-8")
    stage(0.90, f"COLMAP: registered {len(cameras)} cameras, {len(points)} sparse points.")
    return scene
