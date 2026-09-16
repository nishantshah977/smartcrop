"""
High-level API: SmartCropper.

    from smartcrop import SmartCropper, find_best_crop, crop_for_social

    # Just want coordinates, as fast as possible:
    x0, y0, x1, y1 = find_best_crop("photo.jpg", aspect_ratio="4:5")

    # Crop for a specific platform by name instead of remembering ratios:
    x0, y0, x1, y1 = crop_for_social("photo.jpg", platform="instagram_story")

    # Full result object (image, score, debug view, aesthetic breakdown):
    cropper = SmartCropper()
    result = cropper.crop("photo.jpg", aspect_ratio="4:5")
    result.box                                # (x0, y0, x1, y1) in source pixels
    result.image.save("photo_cropped.jpg")    # PIL Image
    result.aesthetic                          # whole-image aesthetic score breakdown
    result.debug_visualization().save("debug.jpg")

Goal: "looks good posted on social media", not just "technically correct"
----------------------------------------------------------------------
Default scoring weights (see fast_score.py) favor punchy, tight, colorful
crops over merely "contains the subject" ones -- aesthetic terms
(colorfulness, saturation, contrast, sharpness) are weighted comparably to
or above raw subject coverage. If you have a NIMA ONNX model available
(see aesthetics.py docstring), pass use_nima=True to additionally
re-rank the top heuristic candidates by a learned aesthetic-quality score
trained on real human ratings, rather than relying on heuristics alone.

Speed strategy
--------------
All saliency/aesthetic map computation and candidate scoring happens on a
downscaled "working" copy of the image (default: longest side capped at
`analysis_size`, 480px). Scoring is fully vectorized (see fast_score.py)
rather than looping per-candidate in Python. The winning box's coordinates
are then rescaled back to the original image's resolution.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Union

import cv2
import numpy as np
from PIL import Image

from .aesthetics import compute_aesthetic_maps, image_aesthetic_score, nima_available, nima_score
from .candidates import Box, generate_candidates
from .fast_score import FastScorer
from .saliency import compute_saliency
from .social import resolve_aspect_ratio


def _load_bgr(source: Union[str, np.ndarray, Image.Image]) -> np.ndarray:
    if isinstance(source, np.ndarray):
        return source
    if isinstance(source, Image.Image):
        return cv2.cvtColor(np.array(source.convert("RGB")), cv2.COLOR_RGB2BGR)
    # assume path
    img = cv2.imread(source, cv2.IMREAD_COLOR)
    if img is None:
        # Fall back to PIL for formats OpenCV can't read (e.g. some PNG/HEIC
        # variants), then convert.
        pil_img = Image.open(source).convert("RGB")
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return img


@dataclass
class CropResult:
    box: Box                       # (x0, y0, x1, y1) in ORIGINAL image pixel coords
    score: float
    aesthetic: dict                # whole-image aesthetic score breakdown (0-100 each)
    elapsed_seconds: float
    saliency_map: np.ndarray       # working-resolution, float32 HxW, 0..1
    source_bgr: np.ndarray         # original, full-resolution image
    nima_score: Optional[float] = None   # learned aesthetic score of the chosen crop, if used
    _top_boxes_working: np.ndarray = field(repr=False, default=None)  # (N,4) working-res
    _scale: float = field(repr=False, default=1.0)

    @property
    def image(self) -> Image.Image:
        """The cropped result as a PIL Image (RGB), at full source resolution."""
        x0, y0, x1, y1 = self.box
        crop = self.source_bgr[y0:y1, x0:x1]
        return Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

    def debug_visualization(self, top_n: int = 5) -> Image.Image:
        """
        Side-by-side visualization: saliency heatmap overlay (with the
        winning box in yellow, runner-ups in gray) next to the final crop.
        """
        img = self.source_bgr.copy()
        h, w = img.shape[:2]

        sal_full = cv2.resize(self.saliency_map, (w, h), interpolation=cv2.INTER_LINEAR)
        heat = cv2.applyColorMap((sal_full * 255).astype(np.uint8), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img, 0.55, heat, 0.45, 0)

        if self._top_boxes_working is not None and self._scale:
            inv = 1.0 / self._scale
            for bx in self._top_boxes_working[1:top_n]:
                x0, y0, x1, y1 = (bx * inv).astype(int)
                cv2.rectangle(overlay, (x0, y0), (x1, y1), (160, 160, 160), 1)

        bx0, by0, bx1, by1 = self.box
        cv2.rectangle(overlay, (bx0, by0), (bx1, by1), (0, 215, 255), 3)

        crop_img = img[by0:by1, bx0:bx1]
        crop_resized = (
            cv2.resize(crop_img, (int(crop_img.shape[1] * h / crop_img.shape[0]), h))
            if crop_img.shape[0] > 0
            else crop_img
        )

        gap = np.full((h, 12, 3), 255, dtype=np.uint8)
        canvas = np.hstack([overlay, gap, crop_resized])
        return Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))


class SmartCropper:
    """
    Fast, saliency + aesthetic-quality based cropping, tuned for "looks
    good posted on social media" rather than just "technically correct".
    No training, no GPU required for the default heuristic mode.

    Parameters
    ----------
    saliency_backend : str
        "auto" | "spectral" | "fine" | "u2net". See saliency.py.
    weights : dict, optional
        Override scoring term weights; see fast_score.FastScorer.score.
    analysis_size : int
        Longest-side cap (in pixels) for the working image used during
        saliency/aesthetic analysis and candidate scoring. Smaller = faster,
        larger = slightly more precise box placement. 480 is a good default
        (sub-100ms on most photos); bump to ~800 if you need finer
        placement on very detailed images and can spend a bit more time.
    use_nima : bool
        If True and a NIMA ONNX model is available (see aesthetics.py),
        re-rank the top `nima_top_k` heuristic candidates by a learned
        aesthetic-quality score instead of trusting the heuristic top
        pick outright. Silently falls back to heuristic-only if no NIMA
        model is present. Adds one small-model inference call per
        shortlisted candidate (typically well under 200ms total for
        nima_top_k=8 on CPU).
    nima_top_k : int
        How many top heuristic candidates to re-rank with NIMA.
    """

    def __init__(
        self,
        saliency_backend: str = "auto",
        weights: Optional[dict] = None,
        analysis_size: int = 480,
        use_nima: bool = False,
        nima_top_k: int = 8,
    ):
        self.saliency_backend = saliency_backend
        self.weights = weights
        self.analysis_size = analysis_size
        self.use_nima = use_nima
        self.nima_top_k = nima_top_k

    def crop(
        self,
        source: Union[str, np.ndarray, Image.Image],
        aspect_ratio: Union[str, float, None] = None,
        aspect_ratios: Optional[List] = None,
        scales: Optional[List[float]] = None,
        x_steps: int = 9,
        y_steps: int = 9,
    ) -> CropResult:
        """
        Find the best crop for an image. See module docstring for the
        speed strategy. Returns a CropResult; use `.box` for just the
        rectangle coordinates, or `find_best_crop()` for a bare tuple.
        """
        t0 = time.perf_counter()
        img_bgr = _load_bgr(source)
        h, w = img_bgr.shape[:2]

        scale = min(1.0, self.analysis_size / max(h, w))
        if scale < 1.0:
            work = cv2.resize(
                img_bgr, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                interpolation=cv2.INTER_AREA,
            )
        else:
            work = img_bgr
            scale = 1.0
        wh, ww = work.shape[:2]

        if aspect_ratio is not None:
            ar_list = [aspect_ratio]
        elif aspect_ratios is not None:
            ar_list = list(aspect_ratios)
        else:
            ar_list = ["1:1", "4:5", "3:4", "16:9", "9:16", f"{w}:{h}"]

        scales = scales or [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

        saliency = compute_saliency(work, backend=self.saliency_backend)
        aes_maps = compute_aesthetic_maps(work)

        boxes = generate_candidates(
            ww, wh, aspect_ratios=ar_list, scales=scales, x_steps=x_steps, y_steps=y_steps,
        )
        if not boxes:
            boxes = [(0, 0, ww, wh)]

        scorer = FastScorer(
            saliency=saliency,
            sharpness=aes_maps.sharpness,
            colorfulness=aes_maps.colorfulness,
            saturation=aes_maps.saturation,
            luminance=aes_maps.luminance,
        )
        scores, box_arr = scorer.score(boxes, weights=self.weights)
        order = np.argsort(scores)[::-1]

        best_idx = order[0]
        best_nima = None

        # Optional: re-rank the top-K heuristic candidates with a learned
        # aesthetic model instead of trusting the heuristic #1 outright.
        if self.use_nima and nima_available():
            top_k_idx = order[: self.nima_top_k]
            inv = 1.0 / scale
            nima_scores = []
            for idx in top_k_idx:
                bx = box_arr[idx]
                x0, y0, x1, y1 = (
                    max(0, int(bx[0] * inv)), max(0, int(bx[1] * inv)),
                    min(w, int(bx[2] * inv)), min(h, int(bx[3] * inv)),
                )
                crop_full = img_bgr[y0:y1, x0:x1]
                if crop_full.size == 0:
                    nima_scores.append(-1.0)
                    continue
                nima_scores.append(nima_score(crop_full))
            best_local = int(np.argmax(nima_scores))
            best_idx = top_k_idx[best_local]
            best_nima = nima_scores[best_local]

        best_box_working = box_arr[best_idx]

        inv = 1.0 / scale
        best_box_full = tuple(
            int(round(v))
            for v in (
                best_box_working[0] * inv,
                best_box_working[1] * inv,
                best_box_working[2] * inv,
                best_box_working[3] * inv,
            )
        )
        best_box_full = (
            max(0, min(best_box_full[0], w - 1)),
            max(0, min(best_box_full[1], h - 1)),
            max(1, min(best_box_full[2], w)),
            max(1, min(best_box_full[3], h)),
        )

        aesthetic = image_aesthetic_score(img_bgr)
        elapsed = time.perf_counter() - t0

        return CropResult(
            box=best_box_full,
            score=float(scores[best_idx]),
            aesthetic=aesthetic,
            elapsed_seconds=elapsed,
            saliency_map=saliency,
            source_bgr=img_bgr,
            nima_score=best_nima,
            _top_boxes_working=box_arr[order],
            _scale=scale,
        )


def find_best_crop(
    source: Union[str, np.ndarray, Image.Image],
    aspect_ratio: Union[str, float, None] = None,
    analysis_size: int = 480,
) -> tuple:
    """
    Convenience function: return ONLY the rectangle coordinates
    (x0, y0, x1, y1), in the source image's original pixel space, for the
    best crop.

    Example
    -------
    >>> find_best_crop("photo.jpg", aspect_ratio="4:5")
    (0, 840, 3024, 3864)
    """
    cropper = SmartCropper(analysis_size=analysis_size)
    return cropper.crop(source, aspect_ratio=aspect_ratio).box


def crop_for_social(
    source: Union[str, np.ndarray, Image.Image],
    platform: str = "instagram_post",
    analysis_size: int = 480,
) -> tuple:
    """
    Convenience function: return the best crop rectangle for a named
    social platform/placement, e.g. "instagram_story", "twitter_post".
    See smartcrop.social.SOCIAL_PRESETS for the full list.

    Example
    -------
    >>> crop_for_social("photo.jpg", platform="instagram_story")
    (302, 0, 3073, 3685)
    """
    ratio = resolve_aspect_ratio(platform)
    cropper = SmartCropper(analysis_size=analysis_size)
    return cropper.crop(source, aspect_ratio=ratio).box
