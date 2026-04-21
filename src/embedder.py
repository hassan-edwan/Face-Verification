"""
Embedder abstraction — lets the pipeline swap FaceNet for ArcFace
without touching callers.

Two backends are supported:

    - `ArcFaceEmbedder`   (default): InsightFace `buffalo_l`'s
      `w600k_r50.onnx` recognition model, run via `onnxruntime`.
      Trained on WebFace600K; 512-d L2-normalized output; 112×112
      input aligned to ArcFace's canonical 5-point template.
      Strong on low-quality / surveillance-domain faces.

    - `FaceNetEmbedder`   (legacy fallback): wraps `keras_facenet`'s
      Inception-ResNet-V1 (VGGFace2-pretrained). 512-d output, 160×160
      input. Kept so we can reproduce run_009/010 behavior via
      `FACE_EMBEDDER=facenet`.

Pick with `get_embedder()`. Default backend is controlled by the
`FACE_EMBEDDER` env var ("arcface" | "facenet"). Model weights are
lazy-loaded on first embed call; downloads are cached under
`data/models/`.

Callers use only:
    embedder = get_embedder()
    emb = embedder.embed(aligned_bgr)    # -> np.ndarray, shape (dim,), unit norm
"""

from __future__ import annotations

import os
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional, Protocol, Tuple

import cv2
import numpy as np


_PROJECT_ROOT   = Path(__file__).resolve().parents[1]
_MODELS_DIR     = _PROJECT_ROOT / "data" / "models"
_BUFFALO_L_URL  = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
_BUFFALO_L_DIR  = _MODELS_DIR / "buffalo_l"
_ARCFACE_ONNX   = _BUFFALO_L_DIR / "w600k_r50.onnx"


# ─────────────────────────────────────────────────────────────────────────────
# Backend protocol
# ─────────────────────────────────────────────────────────────────────────────


class Embedder(Protocol):
    """Minimal contract every embedder satisfies."""
    name:       str
    dim:        int
    input_size: Tuple[int, int]   # (H, W) — expected aligned crop shape

    def embed(self, aligned_bgr: np.ndarray) -> np.ndarray:
        """Produce an L2-normalized identity embedding.

        `aligned_bgr` should already be aligned to the embedder's
        expected `input_size` via `src.alignment`. If the shape
        doesn't match, the embedder resizes (with a warning cost —
        prefer the right alignment upstream)."""
        ...


# ─────────────────────────────────────────────────────────────────────────────
# ArcFace via buffalo_l (ONNX)
# ─────────────────────────────────────────────────────────────────────────────


def _ensure_arcface_model(progress: bool = True) -> Path:
    """Download + extract `buffalo_l`'s ArcFace recognition ONNX on
    first use. Idempotent. Returns the path to `w600k_r50.onnx`.

    The full `buffalo_l` zip is ~280 MB; we extract the one file we
    need and delete the rest so the on-disk footprint is ~170 MB."""
    if _ARCFACE_ONNX.exists():
        return _ARCFACE_ONNX

    _BUFFALO_L_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = _BUFFALO_L_DIR / "buffalo_l.zip"

    if progress:
        print(f"[embedder] downloading buffalo_l.zip (~280 MB) from {_BUFFALO_L_URL}")
        print(f"[embedder] caching to {zip_path} ...")

    # Stream to disk rather than hold the whole blob in memory.
    with urllib.request.urlopen(_BUFFALO_L_URL, timeout=60) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        read = 0
        last_pct = -1
        with open(zip_path, "wb") as f:
            while True:
                chunk = resp.read(65536)
                if not chunk:
                    break
                f.write(chunk)
                read += len(chunk)
                if progress and total:
                    pct = read * 100 // total
                    if pct != last_pct and pct % 10 == 0:
                        print(f"[embedder]   download {pct}% ({read}/{total} bytes)")
                        last_pct = pct

    if progress:
        print(f"[embedder] extracting {_ARCFACE_ONNX.name} ...")
    with zipfile.ZipFile(zip_path) as zf:
        # InsightFace ships the model as either "w600k_r50.onnx" at root
        # or nested in a folder with the pack name; handle both layouts.
        candidates = [n for n in zf.namelist() if n.endswith("w600k_r50.onnx")]
        if not candidates:
            raise RuntimeError(
                "buffalo_l.zip didn't contain w600k_r50.onnx — contents: "
                + ", ".join(zf.namelist())
            )
        with zf.open(candidates[0]) as src, open(_ARCFACE_ONNX, "wb") as dst:
            shutil.copyfileobj(src, dst)

    # Free the zip once the model is extracted.
    try:
        zip_path.unlink()
    except OSError:
        pass

    if progress:
        print(f"[embedder] arcface ready at {_ARCFACE_ONNX}")
    return _ARCFACE_ONNX


class ArcFaceEmbedder:
    name       = "arcface"
    dim        = 512
    input_size = (112, 112)

    def __init__(self):
        self._session: Optional[object] = None
        self._input_name: str = ""

    def _ensure_session(self):
        if self._session is not None:
            return
        import onnxruntime as ort

        model_path = _ensure_arcface_model()
        # CPU-only for portability; GPU can be added by the caller.
        self._session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )
        self._input_name = self._session.get_inputs()[0].name

    def embed(self, aligned_bgr: np.ndarray) -> np.ndarray:
        self._ensure_session()

        # ArcFace expects 112×112 RGB. If the caller passed a different
        # size (e.g. FaceNet-era 160×160), resize — cheap on CPU, and
        # we prefer a valid embedding over a crash.
        if aligned_bgr.shape[:2] != self.input_size:
            aligned_bgr = cv2.resize(aligned_bgr, self.input_size,
                                     interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)

        # ArcFace preprocessing: (pixel - 127.5) / 127.5  →  [-1, 1]
        # Transpose from HWC to CHW and add a batch dim.
        x = (rgb.astype(np.float32) - 127.5) / 127.5
        x = x.transpose(2, 0, 1)[None, ...]

        feat = self._session.run(None, {self._input_name: x})[0][0]
        norm = np.linalg.norm(feat) + 1e-9
        return (feat / norm).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# FaceNet (legacy fallback via keras_facenet)
# ─────────────────────────────────────────────────────────────────────────────


class FaceNetEmbedder:
    name       = "facenet"
    dim        = 512   # keras_facenet's Inception-ResNet-V1 emits 512-d
    input_size = (160, 160)

    def __init__(self):
        self._model = None

    def _ensure_model(self):
        if self._model is not None:
            return
        from keras_facenet import FaceNet
        self._model = FaceNet()

    def embed(self, aligned_bgr: np.ndarray) -> np.ndarray:
        self._ensure_model()
        if aligned_bgr.shape[:2] != self.input_size:
            aligned_bgr = cv2.resize(aligned_bgr, self.input_size,
                                     interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)
        emb = self._model.embeddings(np.expand_dims(rgb, 0))[0]
        norm = np.linalg.norm(emb) + 1e-9
        return (emb / norm).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────


_BACKENDS = {
    "arcface": ArcFaceEmbedder,
    "facenet": FaceNetEmbedder,
}


def get_embedder(backend: Optional[str] = None) -> Embedder:
    """Return a (lazy-loaded) embedder. `backend` defaults to the
    `FACE_EMBEDDER` env var ("arcface" | "facenet"); falls back to
    "arcface" when unset."""
    backend = (backend
               or os.environ.get("FACE_EMBEDDER", "arcface")).strip().lower()
    if backend not in _BACKENDS:
        raise ValueError(
            f"unknown embedder backend {backend!r}; "
            f"known: {sorted(_BACKENDS)}"
        )
    return _BACKENDS[backend]()
