"""
Saliency estimation backends.

Given a BGR image (as loaded by OpenCV), produce a single-channel
float32 "importance map" of the same height/width, normalized to [0, 1],
where higher values mean "more visually interesting / worth keeping in
the crop".

Two backends are provided:

- "spectral"    OpenCV's StaticSaliencySpectralResidual. Fast, no model
                download, decent at finding regions that stand out from
                their surroundings (color/contrast/texture anomalies).
                This is the default and requires nothing extra.

- "fine"        OpenCV's StaticSaliencyFineGrained. Tends to highlight
                edges/texture detail more than broad regions. Useful as
                a second opinion or blended with "spectral".

- "u2net"       Optional deep-learning backend (U^2-Net, salient object
                detection) via onnxruntime, IF the user has placed a
                u2netp.onnx model file in smartcrop/models/. This is not
                downloaded automatically (keeps the base install light
                and offline-friendly); see README for how to add it.
                Falls back to "spectral" automatically if the model
                file / onnxruntime is not available.
"""

from __future__ import annotations

import os
from typing import Optional

import cv2
import numpy as np

_MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
_U2NET_PATH = os.path.join(_MODEL_DIR, "u2netp.onnx")


def _normalize(mat: np.ndarray) -> np.ndarray:
    mat = mat.astype(np.float32)
    lo, hi = float(mat.min()), float(mat.max())
    if hi - lo < 1e-6:
        return np.zeros_like(mat, dtype=np.float32)
    return (mat - lo) / (hi - lo)


def _spectral_residual(image_bgr: np.ndarray) -> np.ndarray:
    detector = cv2.saliency.StaticSaliencySpectralResidual_create()
    success, saliency_map = detector.computeSaliency(image_bgr)
    if not success:
        raise RuntimeError("OpenCV spectral residual saliency computation failed")
    return _normalize(saliency_map)


def _fine_grained(image_bgr: np.ndarray) -> np.ndarray:
    detector = cv2.saliency.StaticSaliencyFineGrained_create()
    success, saliency_map = detector.computeSaliency(image_bgr)
    if not success:
        raise RuntimeError("OpenCV fine-grained saliency computation failed")
    return _normalize(saliency_map)


_ORT_SESSION = None  # lazily created, cached across calls


def _u2net_available() -> bool:
    if not os.path.isfile(_U2NET_PATH):
        return False
    try:
        import onnxruntime  # noqa: F401
        return True
    except ImportError:
        return False


def _u2net(image_bgr: np.ndarray) -> np.ndarray:
    global _ORT_SESSION
    import onnxruntime as ort

    if _ORT_SESSION is None:
        _ORT_SESSION = ort.InferenceSession(
            _U2NET_PATH, providers=["CPUExecutionProvider"]
        )

    h, w = image_bgr.shape[:2]
    inp = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    inp = cv2.resize(inp, (320, 320), interpolation=cv2.INTER_AREA)
    inp = inp.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    inp = (inp - mean) / std
    inp = np.transpose(inp, (2, 0, 1))[None, ...].astype(np.float32)

    input_name = _ORT_SESSION.get_inputs()[0].name
    output = _ORT_SESSION.run(None, {input_name: inp})[0]
    pred = np.squeeze(output[0])
    pred = _normalize(pred)
    pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_LINEAR)
    return pred


def compute_saliency(
    image_bgr: np.ndarray,
    backend: str = "auto",
    blend_fine_grained: bool = True,
) -> np.ndarray:
    """
    Compute a normalized [0, 1] saliency map for an image.

    Parameters
    ----------
    image_bgr : np.ndarray
        Input image in BGR order (OpenCV's default), shape (H, W, 3).
    backend : str
        One of "auto", "u2net", "spectral", "fine".
        "auto" uses u2net if a model file has been supplied, otherwise
        falls back to the spectral residual method.
    blend_fine_grained : bool
        When using the "spectral" backend, additionally blend in the
        fine-grained saliency map (average of the two). This tends to
        give smoother, more reliable maps for photos with subtle
        subjects. Ignored for the "u2net" backend.

    Returns
    -------
    np.ndarray, shape (H, W), dtype float32, values in [0, 1]
    """
    backend = backend.lower()

    if backend == "auto":
        backend = "u2net" if _u2net_available() else "spectral"

    if backend == "u2net":
        if not _u2net_available():
            raise RuntimeError(
                f"u2net backend requested but model not found at {_U2NET_PATH} "
                "(or onnxruntime not installed). See README for setup, or use "
                "backend='spectral' instead."
            )
        return _u2net(image_bgr)

    if backend == "spectral":
        sal = _spectral_residual(image_bgr)
        if blend_fine_grained:
            try:
                fine = _fine_grained(image_bgr)
                sal = _normalize(0.6 * sal + 0.4 * fine)
            except Exception:
                pass
        return sal

    if backend == "fine":
        return _fine_grained(image_bgr)

    raise ValueError(f"Unknown saliency backend: {backend!r}")
