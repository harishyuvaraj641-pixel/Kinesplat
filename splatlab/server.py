"""Local web studio for kinesplat.

A dependency-free HTTP server (standard library only) that serves the browser
viewer, accepts photo/video uploads, runs reconstructions on a background
worker and exposes job progress as JSON.

Run with::

    python -m splatlab.server --port 8000 --open
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import queue
import re
import secrets
import shutil
import threading
import time
import traceback
import webbrowser
from collections import deque
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse

from .config import ASSETS_DIR, CACHE_DIR, JOBS_DIR, PRESETS, WEB_DIR, get_preset
from .device import device_report, label, resolve_device, tune_preset

DEMO_RESULT_DIR = ASSETS_DIR / "demo" / "result"
MAX_UPLOAD_BYTES = 48 * 1024 * 1024          # per file
MAX_JOB_BYTES = 420 * 1024 * 1024             # per reconstruction
MAX_TOTAL_UPLOAD_FILES = 220

MIME_OVERRIDES = {
    ".js": "text/javascript; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".splat": "application/octet-stream",
    ".ply": "application/octet-stream",
    ".glb": "model/gltf-binary",
    ".gltf": "model/gltf+json",
    ".html": "text/html; charset=utf-8",
    ".svg": "image/svg+xml",
    ".webp": "image/webp",
    ".avif": "image/avif",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _job_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S-") + secrets.token_hex(2)


def _slug(text: str, fallback: str = "scene") -> str:
    text = re.sub(r"[^A-Za-z0-9._ -]+", "", text or "").strip()
    text = re.sub(r"\s+", " ", text)
    return (text[:48] or fallback)


def _safe_name(filename: str) -> str:
    name = Path(filename).name
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(name).stem)[:60] or "upload"
    ext = re.sub(r"[^A-Za-z0-9.]+", "", Path(name).suffix).lower()[:10]
    return stem + ext


class Job:
    def __init__(self, job_id: str, name: str, preset: str, created: str | None = None):
        self.id = job_id
        self.name = name
        self.preset = preset
        self.backend = "auto"          # auto | splat | mesh
        self.created = created or _now()
        self.started: str | None = None
        self.finished: str | None = None
        self.status = "draft"          # draft | queued | running | done | failed | cancelled
        self.stage = "idle"
        self.progress = 0.0
        self.message = "Waiting for photos"
        self.error: str | None = None
        self.metrics: dict | None = None
        self.gaussians = 0
        self.cameras = 0
        self.seconds: float | None = None
        self.uploads: list[str] = []
        self.result: dict | None = None
        self.preview_mtime = 0.0
        self.log: deque[str] = deque(maxlen=600)
        self.upload_bytes = 0
        self.cancel = threading.Event()

    @property
    def dir(self) -> Path:
        return JOBS_DIR / self.id

    @property
    def workdir(self) -> Path:
        return self.dir / "work"

    @property
    def upload_dir(self) -> Path:
        return self.dir / "uploads"

    def summary(self) -> dict:
        result = self.result or {}
        kind = result.get("kind") or ("mesh" if result.get("mesh") else "splat")
        return {
            "id": self.id,
            "name": self.name,
            "preset": self.preset,
            "backend": self.backend,
            "kind": kind,
            "created": self.created,
            "started": self.started,
            "finished": self.finished,
            "status": self.status,
            "stage": self.stage,
            "progress": round(float(self.progress), 4),
            "message": self.message,
            "error": self.error,
            "metrics": self.metrics,
            "gaussians": self.gaussians,
            "cameras": self.cameras,
            "faces": (self.metrics or {}).get("faces") or 0,
            "seconds": self.seconds,
            "uploads": len(self.uploads),
            "upload_bytes": self.upload_bytes,
            "preview_mtime": self.preview_mtime,
            "poster_url": self.media_url("poster.jpg"),
            "preview_url": self.media_url("preview.png"),
            "scene_url": self.media_url("scene.json"),
            "splat_url": self.media_url("model.splat"),
            "ply_url": self.media_url("model.ply"),
            "mesh_url": self.media_url("mesh.glb"),
        }

    def detail(self) -> dict:
        payload = self.summary()
        payload["log"] = list(self.log)[-260:]
        payload["result"] = self.result
        return payload

    def media_url(self, filename: str) -> str | None:
        target = self.workdir / "output" / filename
        if target.exists():
            return f"/media/jobs/{self.id}/{quote(filename)}?v={int(target.stat().st_mtime_ns)}"
        return None

    def save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        data = self.summary()
        data["uploads"] = list(self.uploads)
        data["log"] = list(self.log)[-300:]
        tmp = self.dir / "job.json.tmp"
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(tmp, self.dir / "job.json")

    def append_log(self, line: str) -> None:
        line = str(line).rstrip()
        if not line:
            return
        stamp = datetime.now().strftime("%H:%M:%S")
        self.log.append(f"{stamp}  {line}")
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            with open(self.dir / "log.txt", "a", encoding="utf-8") as handle:
                handle.write(f"{stamp}  {line}\n")
        except OSError:
            pass


class JobManager:
    def __init__(self, device: str = "auto") -> None:
        self.device = device
        self.jobs: dict[str, Job] = {}
        self.lock = threading.RLock()
        self.queue: queue.Queue[str] = queue.Queue()
        self.worker = threading.Thread(target=self._worker_loop, name="splatlab-worker", daemon=True)
        self.worker.start()
        self._load_existing()

    # ------------------------------------------------------------------ state
    def _load_existing(self) -> None:
        JOBS_DIR.mkdir(parents=True, exist_ok=True)
        for path in sorted(JOBS_DIR.glob("*/job.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                job = Job(data["id"], data.get("name", "Scene"),
                          data.get("preset", "balanced"), data.get("created"))
                for key in ("started", "finished", "status", "stage", "message",
                            "error", "metrics", "gaussians", "cameras", "seconds",
                            "result", "uploads", "upload_bytes", "backend"):
                    if key in data and data[key] is not None:
                        setattr(job, key, data[key])
                log_path = path.parent / "log.txt"
                if log_path.exists():
                    tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-260:]
                    job.log.extend(tail)
                self._refresh_preview(job)
                if job.status in ("queued", "running"):
                    job.status = "failed"
                    job.error = "The server was restarted while this job was running."
                    job.message = job.error
                    job.finished = _now()
                    job.save()
                self.jobs[job.id] = job
            except Exception:
                continue
        for job in self.jobs.values():
            if job.status == "done" and not job.result:
                job.result = self._result_from_disk(job)

    def _result_from_disk(self, job: Job) -> dict | None:
        out = job.workdir / "output"
        if not (out / "scene.json").exists():
            return None
        mesh = out / "mesh.glb"
        if mesh.exists():
            return {
                "kind": "mesh",
                "scene": "output/scene.json",
                "splat": None,
                "mesh": "output/mesh.glb",
                "ply": "output/mesh.ply" if (out / "mesh.ply").exists() else None,
                "preview": "output/preview.png" if (out / "preview.png").exists() else "output/poster.jpg",
                "poster": "output/poster.jpg",
            }
        return {
            "kind": "splat",
            "scene": "output/scene.json",
            "splat": "output/model.splat" if (out / "model.splat").exists() else None,
            "mesh": None,
            "preview": "output/preview.png" if (out / "preview.png").exists() else "output/poster.jpg",
            "poster": "output/poster.jpg",
        }

    def list(self) -> list[dict]:
        with self.lock:
            items = [job.summary() for job in self.jobs.values()]
        items.sort(key=lambda j: j["created"], reverse=True)
        return items

    def get(self, job_id: str) -> Job | None:
        with self.lock:
            return self.jobs.get(job_id)

    # ------------------------------------------------------------------ jobs
    def create(self, name: str, preset: str, backend: str = "auto") -> Job:
        preset = preset if preset in PRESETS else "balanced"
        backend = backend if backend in ("auto", "splat", "mesh") else "auto"
        job = Job(_job_id(), _slug(name, "Untitled scene"), preset)
        job.backend = backend
        job.dir.mkdir(parents=True, exist_ok=True)
        job.upload_dir.mkdir(parents=True, exist_ok=True)
        job.workdir.mkdir(parents=True, exist_ok=True)
        with self.lock:
            self.jobs[job.id] = job
        job.append_log(f"Created job {job.id} ({preset})")
        job.save()
        return job

    def add_upload(self, job: Job, filename: str, data: bytes) -> dict:
        if len(data) == 0:
            raise ValueError("Empty upload")
        if len(data) > MAX_UPLOAD_BYTES:
            raise ValueError(f"{filename} is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
        if job.upload_bytes + len(data) > MAX_JOB_BYTES:
            raise ValueError("This job has reached the upload size limit")
        if len(job.uploads) >= MAX_TOTAL_UPLOAD_FILES:
            raise ValueError(f"At most {MAX_TOTAL_UPLOAD_FILES} files are allowed")
        name = _safe_name(filename or "upload.jpg")
        if not Path(name).suffix:
            name += ".jpg"
        stem, ext = Path(name).stem, Path(name).suffix
        candidate = job.upload_dir / name
        counter = 1
        while candidate.exists():
            candidate = job.upload_dir / f"{stem}_{counter}{ext}"
            counter += 1
        candidate.write_bytes(data)
        rel = candidate.relative_to(job.dir).as_posix()
        job.uploads.append(rel)
        job.upload_bytes += len(data)
        job.message = f"{len(job.uploads)} file(s) uploaded"
        job.save()
        return {"filename": candidate.name, "uploads": len(job.uploads),
                "bytes": job.upload_bytes}

    def remove_upload(self, job: Job, filename: str) -> None:
        target = (job.upload_dir / Path(filename).name).resolve()
        if job.upload_dir.resolve() not in target.parents:
            raise ValueError("Invalid file")
        if target.exists():
            job.upload_bytes = max(0, job.upload_bytes - target.stat().st_size)
            target.unlink()
        rel = target.relative_to(job.dir).as_posix()
        job.uploads = [u for u in job.uploads if u != rel]
        job.save()

    def start(self, job: Job) -> None:
        if not job.uploads:
            raise ValueError("Add at least a few photos or a video first")
        with self.lock:
            if job.status in ("queued", "running"):
                return
            job.status = "queued"
            job.stage = "queued"
            job.progress = 0.0
            job.message = "Queued"
            job.error = None
            job.finished = None
            job.cancel.clear()
        job.append_log("Queued for reconstruction")
        job.save()
        self.queue.put(job.id)

    def cancel(self, job: Job) -> None:
        job.cancel.set()
        with self.lock:
            if job.status == "queued":
                job.status = "cancelled"
                job.stage = "cancelled"
                job.message = "Cancelled"
                job.finished = _now()
        job.append_log("Cancellation requested")
        job.save()

    def delete(self, job: Job) -> None:
        job.cancel.set()
        with self.lock:
            self.jobs.pop(job.id, None)
        shutil.rmtree(job.dir, ignore_errors=True)

    # ----------------------------------------------------------------- worker
    def _worker_loop(self) -> None:
        while True:
            job_id = self.queue.get()
            try:
                job = self.get(job_id)
                if job is not None and job.status == "queued":
                    self._run(job)
            except Exception:
                pass
            finally:
                self.queue.task_done()

    def _run(self, job: Job) -> None:
        from .pipeline import PipelineConfig, run_reconstruction
        from .pipeline import Cancelled

        with self.lock:
            job.status = "running"
            job.started = _now()
            job.stage = "frames"
            job.progress = 0.01
            job.message = "Starting ..."
        job.append_log(f"Starting reconstruction at preset '{job.preset}'")
        job.save()

        started = time.time()

        def log(msg: str) -> None:
            job.append_log(str(msg))
            if job.cancel.is_set():
                return

        def progress(stage: str, pct: float, msg: str) -> None:
            job.stage = stage
            job.progress = float(pct)
            job.message = msg
            if job.cancel.is_set():
                return
            self._refresh_preview(job)

        cfg = PipelineConfig(
            name=job.name,
            preset=job.preset,
            backend=job.backend,
            pose_backend="auto",
            use_depth=True,
            threads=None,
            device=self.device,
        )
        paths = [str(job.dir / rel) for rel in job.uploads]
        try:
            result = run_reconstruction(
                job.workdir, paths, cfg,
                log=log, progress=progress, cancel_event=job.cancel,
            )
            job.result = result
            job.metrics = result.get("metrics", {})
            job.gaussians = int(result.get("gaussians") or 0)
            job.cameras = int(result.get("cameras") or 0)
            job.seconds = float(result.get("seconds") or (time.time() - started))
            with self.lock:
                job.status = "done"
                job.stage = "done"
                job.progress = 1.0
                if result.get("kind") == "mesh" or result.get("mesh"):
                    faces = int((result.get("metrics") or {}).get("faces") or 0)
                    job.message = f"Ready: mesh with {faces:,} faces"
                else:
                    job.message = f"Ready: {job.gaussians:,} splats"
                job.finished = _now()
            job.append_log(f"Finished in {job.seconds:.1f}s")
            self._refresh_preview(job)
        except Cancelled:
            with self.lock:
                job.status = "cancelled"
                job.stage = "cancelled"
                job.message = "Cancelled"
                job.finished = _now()
            job.append_log("Job cancelled")
        except Exception as exc:
            message = str(exc).strip() or exc.__class__.__name__
            with self.lock:
                job.status = "failed"
                job.stage = "failed"
                job.message = message.splitlines()[0][:300]
                job.error = message
                job.finished = _now()
            job.append_log(f"ERROR: {message}")
            job.append_log(traceback.format_exc(limit=3))
        finally:
            job.save()

    def _refresh_preview(self, job: Job) -> None:
        preview = job.workdir / "output" / "preview.png"
        try:
            if preview.exists():
                job.preview_mtime = preview.stat().st_mtime_ns / 1e9
        except OSError:
            pass

    # ------------------------------------------------------------------- demo
    def demo(self) -> dict | None:
        if not (DEMO_RESULT_DIR / "scene.json").exists():
            return None
        try:
            scene = json.loads((DEMO_RESULT_DIR / "scene.json").read_text(encoding="utf-8"))
        except Exception:
            scene = {}
        metrics = scene.get("stats") or {}
        try:
            metrics = json.loads((DEMO_RESULT_DIR / "metrics.json").read_text(encoding="utf-8"))
        except Exception:
            pass
        return {
            "id": "demo",
            "name": scene.get("name", "Toy Courtyard"),
            "preset": "bundled",
            "kind": "splat",
            "status": "done",
            "stage": "done",
            "progress": 1.0,
            "message": "Bundled demo reconstruction",
            "created": "",
            "finished": "",
            "metrics": metrics,
            "gaussians": metrics.get("gaussians") or 0,
            "cameras": metrics.get("cameras") or 0,
            "faces": 0,
            "seconds": metrics.get("total_seconds") or metrics.get("seconds"),
            "uploads": 0,
            "poster_url": "/demo/poster.jpg",
            "preview_url": "/demo/preview.png",
            "scene_url": "/demo/scene.json",
            "splat_url": "/demo/model.splat",
            "ply_url": "/demo/model.ply",
            "mesh_url": None,
        }


MANAGER = JobManager()


def _demo_dir() -> Path:
    return DEMO_RESULT_DIR


class Handler(BaseHTTPRequestHandler):
    server_version = "kinesplat/1.0"
    protocol_version = "HTTP/1.1"

    # -------------------------------------------------------------- utilities
    def handle(self) -> None:
        """Treat a browser-aborted request as a normal disconnect.

        Large scene files are streamed to the viewer, so navigating away or
        closing a tab can reset the socket while ``BaseHTTPRequestHandler`` is
        still reading the next request line.  That is not an application
        failure and should not produce a traceback in the server console.
        """
        try:
            super().handle()
        except (BrokenPipeError, ConnectionResetError):
            return

    def log_message(self, fmt: str, *args) -> None:  # quieter console
        return

    def _send(self, status: int, body: bytes, content_type: str,
              extra_headers: dict | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        if extra_headers:
            for key, value in extra_headers.items():
                self.send_header(key, value)
        self.end_headers()
        if self.command != "HEAD":
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

    def _json(self, payload, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send(status, body, "application/json; charset=utf-8")

    def _error(self, status: int, message: str) -> None:
        self._json({"error": message}, status=status)

    def _read_body(self) -> bytes:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        if length <= 0:
            return b""
        if length > MAX_UPLOAD_BYTES + 1024:
            raise ValueError("Upload too large")
        chunks = []
        remaining = length
        while remaining > 0:
            chunk = self.rfile.read(min(remaining, 1024 * 1024))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def _read_json(self) -> dict:
        raw = self._read_body()
        if not raw:
            return {}
        try:
            data = json.loads(raw.decode("utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception as exc:
            raise ValueError("Invalid JSON body") from exc

    def _job(self, job_id: str) -> Job | None:
        return MANAGER.get(job_id)

    def _serve_file(self, path: Path) -> None:
        try:
            stat = path.stat()
        except OSError:
            self._error(404, "File not found")
            return
        if not path.is_file():
            self._error(404, "File not found")
            return
        suffix = path.suffix.lower()
        ctype = MIME_OVERRIDES.get(suffix) or mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        etag = f'"{int(stat.st_mtime_ns)}-{stat.st_size}"'
        if self.headers.get("If-None-Match") == etag:
            self.send_response(304)
            self.send_header("ETag", etag)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(stat.st_size))
        self.send_header("ETag", etag)
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        if self.command == "HEAD":
            return
        try:
            with open(path, "rb") as handle:
                shutil.copyfileobj(handle, self.wfile, length=1024 * 512)
        except (BrokenPipeError, ConnectionResetError):
            pass

    # ----------------------------------------------------------------- routes
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        query = parse_qs(parsed.query)

        if path.startswith("/api/"):
            self._api_get(path, query)
            return
        if path.startswith("/media/jobs/"):
            parts = path.split("/")
            if len(parts) >= 5:
                job = self._job(parts[3])
                if job is None:
                    self._error(404, "Unknown job")
                    return
                filename = "/".join(parts[4:])
                target = (job.workdir / "output" / filename).resolve()
                if (job.workdir / "output").resolve() not in target.parents:
                    self._error(403, "Forbidden")
                    return
                self._serve_file(target)
                return
            self._error(404, "Not found")
            return
        if path.startswith("/demo/"):
            filename = path[len("/demo/"):]
            target = (_demo_dir() / filename).resolve()
            if _demo_dir().resolve() not in target.parents:
                self._error(403, "Forbidden")
                return
            self._serve_file(target)
            return

        # static web app
        if path in ("", "/"):
            path = "/index.html"
        if path.startswith("/vendor/"):
            target = (WEB_DIR / path.lstrip("/")).resolve()
        else:
            target = (WEB_DIR / path.lstrip("/")).resolve()
        if WEB_DIR.resolve() not in target.parents:
            self._error(403, "Forbidden")
            return
        if not target.exists():
            # SPA fallback for in-app navigation.
            if "." not in Path(path).name:
                self._serve_file(WEB_DIR / "index.html")
            else:
                self._error(404, "Not found")
            return
        self._serve_file(target)

    def do_HEAD(self) -> None:
        self.do_GET()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        try:
            if path == "/api/jobs":
                payload = self._read_json()
                name = str(payload.get("name") or "Untitled scene")[:80]
                preset = str(payload.get("preset") or "balanced").lower()
                backend = str(payload.get("backend") or "auto").lower()
                job = MANAGER.create(name, preset, backend)
                self._json(job.summary(), status=201)
                return
            match = re.fullmatch(r"/api/jobs/([A-Za-z0-9._-]+)/upload", path)
            if match:
                job = self._job(match.group(1))
                if job is None:
                    self._error(404, "Unknown job")
                    return
                if job.status in ("queued", "running"):
                    self._error(409, "Job is already running")
                    return
                filename = self.headers.get("X-Filename") or "upload.jpg"
                filename = unquote(filename)
                data = self._read_body()
                info = MANAGER.add_upload(job, filename, data)
                self._json(info, status=201)
                return
            match = re.fullmatch(r"/api/jobs/([A-Za-z0-9._-]+)/start", path)
            if match:
                job = self._job(match.group(1))
                if job is None:
                    self._error(404, "Unknown job")
                    return
                MANAGER.start(job)
                self._json(job.summary())
                return
            match = re.fullmatch(r"/api/jobs/([A-Za-z0-9._-]+)/cancel", path)
            if match:
                job = self._job(match.group(1))
                if job is None:
                    self._error(404, "Unknown job")
                    return
                MANAGER.cancel(job)
                self._json(job.summary())
                return
            self._error(404, "Not found")
        except ValueError as exc:
            self._error(400, str(exc))
        except Exception as exc:
            traceback.print_exc()
            self._error(500, str(exc) or "Server error")

    def do_DELETE(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        match = re.fullmatch(r"/api/jobs/([A-Za-z0-9._-]+)(?:/uploads/(.+))?", path)
        if not match:
            self._error(404, "Not found")
            return
        job = self._job(match.group(1))
        if job is None:
            self._error(404, "Unknown job")
            return
        if match.group(2):
            try:
                MANAGER.remove_upload(job, match.group(2))
                self._json(job.summary())
            except ValueError as exc:
                self._error(400, str(exc))
            return
        MANAGER.delete(job)
        self._json({"deleted": True})

    # -------------------------------------------------------------- api reads
    def _api_get(self, path: str, query: dict) -> None:
        if path == "/api/config":
            demo = MANAGER.demo()
            device_error = None
            try:
                dev = resolve_device(MANAGER.device)
                info = device_report(dev)
                device_label = label(dev)
            except Exception as exc:
                dev = None
                device_error = str(exc)
                info = {"type": "unavailable", "name": "Unavailable",
                        "vram_gb": None, "cuda": "", "error": device_error}
                device_label = "GPU unavailable"
            presets = []
            for p in PRESETS.values():
                tuned, _ = tune_preset(p, dev) if dev is not None else (p, [])
                presets.append({
                    "key": tuned.key, "label": tuned.label,
                    "description": tuned.description,
                    "iterations": tuned.iterations,
                    "max_gaussians": tuned.max_gaussians,
                    "max_frames": tuned.max_frames,
                    "train_size": tuned.train_size,
                })
            try:
                from .mesh import backend_status

                mesh_status = backend_status()
            except Exception as exc:  # never let a status probe break the app
                mesh_status = {"name": "TripoSR", "available": False,
                               "downloaded": False, "error": str(exc)}
            self._json({
                "version": "1.0.0",
                "presets": presets,
                "default_preset": "balanced",
                "demo": demo,
                "device": device_label,
                "device_info": info,
                "device_error": device_error,
                "mesh_backend": mesh_status,
            })
            return
        if path == "/api/jobs":
            self._json({"jobs": MANAGER.list(), "demo": MANAGER.demo()})
            return
        match = re.fullmatch(r"/api/jobs/([A-Za-z0-9._-]+)", path)
        if match:
            job = self._job(match.group(1))
            if job is None:
                self._error(404, "Unknown job")
                return
            self._json(job.detail())
            return
        if path == "/api/demo":
            demo = MANAGER.demo()
            if demo is None:
                self._error(404, "No bundled demo")
            else:
                self._json(demo)
            return
        match = re.fullmatch(r"/api/jobs/([A-Za-z0-9._-]+)/preview", path)
        if match:
            job = self._job(match.group(1))
            if job is None:
                self._error(404, "Unknown job")
                return
            preview = job.workdir / "output" / "preview.png"
            if preview.exists():
                self._serve_file(preview)
            else:
                poster = job.workdir / "output" / "poster.jpg"
                if poster.exists():
                    self._serve_file(poster)
                else:
                    self._error(404, "No preview yet")
            return
        self._error(404, "Not found")


def serve(host: str = "127.0.0.1", port: int = 8000, open_browser: bool = False,
          device: str = "auto") -> None:
    # Validate the selected compute path before accepting jobs.  In particular,
    # never start an apparently healthy studio that will silently run a GPU
    # request on CPU-only PyTorch.
    selected = resolve_device(device)
    MANAGER.device = device
    # Pre-load demo and mesh backend status to avoid first-request timeout.
    # These can be slow on first call (disk I/O, model checks).
    _ = MANAGER.demo()
    try:
        from .mesh import backend_status
        _ = backend_status()
    except Exception:
        pass
    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    url = f"http://{host}:{port}/"
    print(f"kinesplat is running at {url} ({label(selected)})")
    print("Press Ctrl+C to stop.")
    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        print("\nShutting down ...")
    finally:
        server.server_close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="kinesplat local web studio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--open", action="store_true", help="open the browser on start")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"),
                        help="compute device (default: auto)")
    args = parser.parse_args(argv)
    serve(args.host, args.port, args.open, args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
