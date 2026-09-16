"""
Lightweight, no-training aesthetic quality signals.

These are classic, fast, well-established image-quality heuristics (no
deep model, no download) used both:

1. per-candidate-crop, as extra terms in the crop scorer (so the chosen
   rectangle isn't just "contains the salient pixels" but also "is a
   reasonably well-exposed, sharp, non-flat-looking region"), and

2. whole-image, via `image_aesthetic_score`, to answer "is this a good /
   aesthetically pleasing photo" independent of cropping -- useful for
   filtering or ranking a batch of images.

All per-pixel maps are designed to be summed with integral images so that
scoring any crop rectangle is an O(1) lookup instead of an O(pixels) scan
-- this is what makes scoring thousands of crop candidates fast.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import cv2
import numpy as np

_MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")


def _normalize(mat: np.ndarray) -> np.ndarray:
    mat = mat.astype(np.float32)
    lo, hi = float(mat.min()), float(mat.max())
    if hi - lo < 1e-6:
        return np.zeros_like(mat, dtype=np.float32)
    return (mat - lo) / (hi - lo)


@dataclass
class AestheticMaps:
    sharpness: np.ndarray      # local edge energy (|Laplacian|), normalized 0..1
    colorfulness: np.ndarray   # local chroma magnitude (Hasler-Susstrunk style), 0..1
    saturation: np.ndarray     # local HSV saturation, 0..1 -- "vibrance"/punch
    luminance: np.ndarray      # grayscale, 0..1 (used for contrast/exposure)


def compute_aesthetic_maps(image_bgr: np.ndarray) -> AestheticMaps:
    """
    Compute per-pixel maps used for both crop scoring and global scoring.
    Cheap: a Laplacian, a couple of channel subtractions, one cvtColor.
    Intended to be called on an already-downscaled working image.
    """
    b = image_bgr[:, :, 0].astype(np.float32)
    g = image_bgr[:, :, 1].astype(np.float32)
    r = image_bgr[:, :, 2].astype(np.float32)

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1].astype(np.float32) / 255.0

    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    sharpness = _normalize(np.abs(lap))

    rg = r - g
    yb = 0.5 * (r + g) - b
    colorfulness = _normalize(np.sqrt(rg**2 + yb**2))

    return AestheticMaps(
        sharpness=sharpness, colorfulness=colorfulness, saturation=saturation, luminance=gray
    )


def image_aesthetic_score(image_bgr: np.ndarray, max_dim: int = 512) -> dict:
    """
    A fast, whole-image "is this an aesthetically decent photo" score.
    Not a learned model -- combines four classic, cheap signals:

      - sharpness   : variance of the Laplacian (the standard fast blur
                       detector; low value ~= out of focus / blurry)
      - colorfulness : Hasler & Susstrunk (2003) colorfulness metric
                       (low value ~= flat, desaturated, dull)
      - contrast     : std deviation of luminance (low value ~= flat/hazy)
      - exposure     : penalizes images that are mostly clipped to black
                       or white (over/under-exposed)

    Returns a dict with the four sub-scores (0..100 each) and an overall
    "score" (0..100, simple average). Use this to rank/filter a batch of
    photos, independent of any cropping.
    """
    h, w = image_bgr.shape[:2]
    scale = min(1.0, max_dim / max(h, w))
    small = (
        cv2.resize(image_bgr, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
        if scale < 1.0
        else image_bgr
    )

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # --- sharpness: variance of Laplacian, log-compressed & clipped to a
    # reasonable range so it maps to a 0..100 score without being
    # dominated by extreme outliers on very high-detail photos.
    lap_var = cv2.Laplacian(gray, cv2.CV_32F, ksize=3).var()
    sharpness_score = float(np.clip(np.log1p(lap_var) / np.log1p(2000.0), 0, 1) * 100)

    # --- colorfulness: Hasler & Susstrunk global formula
    b = small[:, :, 0].astype(np.float32)
    g = small[:, :, 1].astype(np.float32)
    r = small[:, :, 2].astype(np.float32)
    rg = r - g
    yb = 0.5 * (r + g) - b
    colorfulness_metric = np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(
        rg.mean() ** 2 + yb.mean() ** 2
    )
    colorfulness_score = float(np.clip(colorfulness_metric / 110.0, 0, 1) * 100)

    # --- contrast: std of luminance, normalized against a typical range
    contrast_score = float(np.clip(gray.std() / 70.0, 0, 1) * 100)

    # --- exposure: penalize heavy clipping at black/white ends
    hist = cv2.calcHist([gray.astype(np.uint8)], [0], None, [256], [0, 256]).ravel()
    hist = hist / max(hist.sum(), 1.0)
    clipped_black = hist[:5].sum()
    clipped_white = hist[-5:].sum()
    exposure_score = float(np.clip(1.0 - (clipped_black + clipped_white) * 4, 0, 1) * 100)

    overall = 0.30 * sharpness_score + 0.25 * colorfulness_score + 0.25 * contrast_score + 0.20 * exposure_score

    return {
        "score": round(overall, 1),
        "sharpness": round(sharpness_score, 1),
        "colorfulness": round(colorfulness_score, 1),
        "contrast": round(contrast_score, 1),
        "exposure": round(exposure_score, 1),
    }


# ---------------------------------------------------------------------------
# Optional: NIMA (Neural Image Assessment) -- a trained aesthetic scorer
# ---------------------------------------------------------------------------
#
# Everything above is hand-built heuristics (sharpness, colorfulness,
# contrast, exposure). They're fast and don't need a model download, but
# they're proxies for "looks good" -- not the thing itself.
#
# NIMA (Talebi & Milanfar, 2018) is a small CNN (MobileNet-sized, ~13MB)
# trained directly on the AVA dataset -- ~250,000 photos each rated 1-10 by
# real people for aesthetic quality. Its output is a distribution over
# scores 1-10, whose mean is a genuine learned "how good does this look"
# score, aligned with what actually gets rated well by humans (composition,
# color harmony, lighting, subject choice) rather than a proxy metric.
#
# smartcrop/models/nima_mobilenet.onnx ships with this repo, converted from
# titu1994/neural-image-assessment's MobileNet weights (trained on AVA) and
# verified against the original Keras model's output before conversion.
# SmartCropper(use_nima=True) uses it automatically if onnxruntime is
# installed (`pip install onnxruntime`); if not, it falls back to
# heuristic-only scoring silently. Swapping in a different NIMA export?
# check nima_score()'s preprocessing below -- this one expects
# (pixel/127.5)-1.0 scaling and NHWC layout, not the more common /255 + NCHW.
_NIMA_PATH = os.path.join(_MODEL_DIR, "nima_mobilenet.onnx")
_NIMA_SESSION = None


def nima_available() -> bool:
    if not os.path.isfile(_NIMA_PATH):
        return False
    try:
        import onnxruntime  # noqa: F401
        return True
    except ImportError:
        return False


def nima_score(image_bgr: np.ndarray) -> float:
    """
    Score a single image/crop with NIMA, returning the mean predicted
    aesthetic rating (roughly 1-10, higher is better). Raises if the model
    isn't available -- check nima_available() first.
    """
    global _NIMA_SESSION
    import onnxruntime as ort

    if _NIMA_SESSION is None:
        _NIMA_SESSION = ort.InferenceSession(_NIMA_PATH, providers=["CPUExecutionProvider"])

    # Matches keras.applications.mobilenet.preprocess_input (scale to
    # [-1, 1]) and NHWC layout, which is what the bundled nima_mobilenet.onnx
    # (converted from titu1994/neural-image-assessment's MobileNet weights)
    # expects. If you swap in a different NIMA export, check its expected
    # preprocessing/layout -- this is a common source of silently-wrong
    # scores (the model still runs, it just scores everything ~equally).
    inp = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    inp = cv2.resize(inp, (224, 224), interpolation=cv2.INTER_AREA).astype(np.float32)
    inp = (inp / 127.5) - 1.0
    inp = inp[None, ...]  # NHWC: (1, 224, 224, 3)

    input_name = _NIMA_SESSION.get_inputs()[0].name
    output = _NIMA_SESSION.run(None, {input_name: inp})[0]
    probs = np.squeeze(output)
    probs = probs / max(probs.sum(), 1e-8)
    scores = np.arange(1, len(probs) + 1)
    return float((probs * scores).sum())
