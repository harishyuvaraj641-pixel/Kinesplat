"""Pretrained monocular depth (Depth Anything V2 Small, ONNX).

The model is the "predefined model" that makes reconstruction easy: it turns
every frame into a dense depth map, so Gaussian splats can be initialised
immediately instead of waiting for slow CPU densification.
"""

from __future__ import annotations

import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from .config import (
    DEPTH_MODEL_MIN_BYTES,
    DEPTH_MODEL_PATH,
    DEPTH_MODEL_URL,
    MODELS_DIR,
)


class ModelUnavailable(RuntimeError):
    pass


def download_depth_model(progress=None, force: bool = False) -> Path:
    """Download the 95 MB ONNX depth model with a resumable, verified download."""
    path = DEPTH_MODEL_PATH
    if path.exists() and path.stat().st_size >= DEPTH_MODEL_MIN_BYTES and not force:
        return path
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(".onnx.part")
    if part.exists() and force:
        part.unlink()

    request = urllib.request.Request(
        DEPTH_MODEL_URL,
        headers={"User-Agent": "kinesplat/1.0 (+https://github.com/)"},
    )
    start = time.time()
    with urllib.request.urlopen(request, timeout=60) as response:
        total = int(response.headers.get("Content-Length") or 0)
        with open(part, "wb") as handle:
            done = 0
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                done += len(chunk)
                if progress and total:
                    elapsed = max(1e-6, time.time() - start)
                    progress(done / total, done, total, done / elapsed)
    if part.stat().st_size < DEPTH_MODEL_MIN_BYTES:
        part.unlink(missing_ok=True)
        raise ModelUnavailable("Depth model download was incomplete. Check your connection.")
    part.replace(path)
    return path


class DepthEstimator:
    """Lazy ONNX depth estimator running on the CPU."""

    def __init__(self, model_path: Path | None = None, threads: int | None = None):
        self.model_path = Path(model_path or DEPTH_MODEL_PATH)
        self.session = None
        self.input_name = None
        self.input_size = (518, 518)
        self.threads = threads

    def available(self) -> bool:
        return self.model_path.exists() and self.model_path.stat().st_size >= DEPTH_MODEL_MIN_BYTES

    def ensure(self) -> None:
        if self.session is not None:
            return
        if not self.available():
            download_depth_model()
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ModelUnavailable(
                "onnxruntime is missing. Run: pip install -r requirements.txt"
            ) from exc

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if self.threads:
            options.intra_op_num_threads = int(self.threads)
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(
            str(self.model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        shape = inp.shape
        if isinstance(shape, (list, tuple)) and len(shape) == 4:
            h, w = shape[2], shape[3]
            if isinstance(h, int) and isinstance(w, int):
                self.input_size = (w, h)

    @staticmethod
    def _preprocess(bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (518, 518), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        rgb = (rgb - mean) / std
        return np.ascontiguousarray(rgb.transpose(2, 0, 1)[None, ...])

    def predict(self, bgr: np.ndarray) -> np.ndarray:
        """Return a relative disparity map (larger = closer) at the image size."""
        self.ensure()
        h, w = bgr.shape[:2]
        tensor = self._preprocess(bgr)
        outputs = self.session.run(None, {self.input_name: tensor})
        raw = np.asarray(outputs[0], dtype=np.float32).squeeze()
        if raw.ndim != 2:
            raise ModelUnavailable(f"Unexpected depth output shape: {raw.shape}")
        raw = cv2.resize(raw, (w, h), interpolation=cv2.INTER_LINEAR)
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        # Depth Anything outputs an inverse-depth / disparity-like signal.
        lo, hi = np.percentile(raw, [1.0, 99.0])
        if hi - lo < 1e-6:
            raise ModelUnavailable("Depth model produced a flat map for this frame.")
        raw = np.clip(raw, lo, hi)
        # Normalise around the median for numerical conditioning.
        med = float(np.median(raw))
        return (raw / max(1e-6, abs(med))).astype(np.float32)
