"""
Vectorized candidate scoring.

The original scorer (scorer.py) computes each candidate's score with a
Python-level function call per box, including a np.mgrid-based centroid
calculation over the full box region. That's fine for a handful of boxes
but scales badly: on a 4032x3024 photo with ~300 candidates it dominates
runtime (order of 15-20s), because several of those centroid regions are
millions of pixels and get rebuilt from scratch, in Python, per box.

This module computes the same signals (saliency coverage, rule-of-thirds
centroid, boundary-cut penalty, size penalty) PLUS aesthetic sub-scores
(sharpness, colorfulness, saturation, contrast, exposure), for ALL
candidates at once using integral images (summed-area tables) and numpy
fancy indexing -- no Python loop over candidates, no per-box pixel
iteration.

Combined with running on a downscaled working image (see cropper.py),
this brings scoring from ~20s down to well under 100ms on typical photos.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .candidates import Box

# Small epsilon to avoid divide-by-zero in ratio computations.
_EPS = 1e-8


def _integral(mat: np.ndarray) -> np.ndarray:
    return np.pad(mat.astype(np.float64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)


def _box_sum(integral: np.ndarray, x0, y0, x1, y1):
    """Vectorized summed-area-table lookup for arrays of box coordinates."""
    return integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]


class FastScorer:
    """
    Precomputes integral images once per photo, then scores any number of
    candidate boxes in a single vectorized pass.
    """

    def __init__(
        self,
        saliency: np.ndarray,
        sharpness: np.ndarray,
        colorfulness: np.ndarray,
        saturation: np.ndarray,
        luminance: np.ndarray,
    ):
        h, w = saliency.shape
        ys, xs = np.mgrid[0:h, 0:w]

        self.h, self.w = h, w
        self.total_saliency = max(float(saliency.sum()), _EPS)

        self.I_sal = _integral(saliency)
        self.I_sal_x = _integral(saliency * xs)
        self.I_sal_y = _integral(saliency * ys)
        self.I_sharp = _integral(sharpness)
        self.I_color = _integral(colorfulness)
        self.I_sat = _integral(saturation)
        self.I_lum = _integral(luminance)
        self.I_lum2 = _integral(luminance * luminance)

    def score(self, boxes: List[Box], weights: dict = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Score all candidate boxes at once.

        Returns
        -------
        (scores, box_array) : (np.ndarray shape (N,), np.ndarray shape (N,4) int)
        """
        # Defaults tuned for "looks good posted on social media", not just
        # "technically well-composed" -- but aesthetic terms (colorfulness,
        # saturation, contrast, sharpness) are kept as TIE-BREAKERS among
        # subject-relevant candidates, not able to outweigh coverage/thirds/
        # boundary. An earlier version weighted them too aggressively and it
        # would pick a saturated, sharp-but-boring patch (e.g. a blanket)
        # over a genuinely striking but less colorful subject (e.g. a moody
        # window silhouette) -- "vibrant" isn't the same as "good", and a
        # crop with no notion of *subject* will happily chase raw pixel
        # saturation into nonsense. Composition terms dominate; aesthetics
        # nudge between otherwise-reasonable candidates.
        default_weights = {
            "coverage": 1.0,        # reward for retained saliency mass (recall)
            "area_cost": 1.0,       # cost per unit of frame area used -- the main
                                     # lever for how aggressively it crops
            "thirds": 0.6,
            "boundary_penalty": 1.2,
            "tiny_penalty": 0.6,    # floor: discourage degenerate near-empty crops
            "sharpness": 0.35,
            "colorfulness": 0.2,
            "saturation": 0.15,     # vibrance / "pop" -- distinct from colorfulness
            "contrast": 0.3,
            "exposure": 0.25,
        }
        if weights:
            default_weights.update(weights)
        wts = default_weights

        arr = np.asarray(boxes, dtype=np.int32)  # (N, 4): x0,y0,x1,y1
        x0, y0, x1, y1 = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
        area = np.maximum((x1 - x0) * (y1 - y0), 1).astype(np.float64)
        img_area = float(self.h * self.w)
        area_frac = area / img_area

        # --- saliency recall: how much of the total salient mass is kept ---
        sal_sum = _box_sum(self.I_sal, x0, y0, x1, y1)
        recall = sal_sum / self.total_saliency

        # --- rule-of-thirds centroid (vectorized, no mgrid per box) ---
        sal_x_sum = _box_sum(self.I_sal_x, x0, y0, x1, y1)
        sal_y_sum = _box_sum(self.I_sal_y, x0, y0, x1, y1)
        has_mass = sal_sum > _EPS
        cx = np.where(has_mass, sal_x_sum / np.maximum(sal_sum, _EPS), (x0 + x1) / 2.0)
        cy = np.where(has_mass, sal_y_sum / np.maximum(sal_sum, _EPS), (y0 + y1) / 2.0)

        w_box = np.maximum(x1 - x0, 1)
        h_box = np.maximum(y1 - y0, 1)
        nx = (cx - x0) / w_box
        ny = (cy - y0) / h_box
        dx = np.minimum(np.abs(nx - 1 / 3), np.abs(nx - 2 / 3))
        dy = np.minimum(np.abs(ny - 1 / 3), np.abs(ny - 2 / 3))
        dist = np.sqrt(dx**2 + dy**2)
        thirds = np.clip(1.0 - dist / 0.35, 0.0, 1.0)

        # --- boundary-cut penalty: mean saliency in a thin inner band ---
        band = max(1, int(round(0.015 * min(self.h, self.w))))
        ix0 = np.minimum(x0 + band, x1)
        iy0 = np.minimum(y0 + band, y1)
        ix1 = np.maximum(x1 - band, x0)
        iy1 = np.maximum(y1 - band, y0)
        outer_sum = sal_sum
        inner_sum = _box_sum(self.I_sal, ix0, iy0, ix1, iy1)
        inner_area = np.maximum((ix1 - ix0) * (iy1 - iy0), 0).astype(np.float64)
        band_area = np.maximum(area - inner_area, 1.0)
        boundary_penalty = np.clip((outer_sum - inner_sum) / band_area, 0.0, None)

        # --- tiny-crop floor: only guards against degenerate near-empty crops.
        # (The old version also penalized *large* crops here, but that's now
        # handled properly by area_cost below -- a linear-in-area term is a
        # much more direct lever than a threshold that only bites above 92%.)
        tiny_penalty = np.where(area_frac < 0.12, (0.12 - area_frac) / 0.12, 0.0)

        # --- aesthetic sub-scores (all O(1) lookups via integral images;
        # these are already per-pixel AVERAGES, so they don't themselves
        # bias toward larger boxes the way a raw sum would) ---
        sharp_avg = _box_sum(self.I_sharp, x0, y0, x1, y1) / area
        color_avg = _box_sum(self.I_color, x0, y0, x1, y1) / area
        sat_avg = _box_sum(self.I_sat, x0, y0, x1, y1) / area
        lum_avg = _box_sum(self.I_lum, x0, y0, x1, y1) / area
        lum2_avg = _box_sum(self.I_lum2, x0, y0, x1, y1) / area
        variance = np.clip(lum2_avg - lum_avg**2, 0.0, None)
        contrast = np.sqrt(variance)  # 0..~0.5 for 0..1-scaled luminance
        contrast_norm = np.clip(contrast / 0.25, 0.0, 1.0)
        # exposure: reward mid-range average brightness, penalize near-black/near-white crops
        exposure = np.clip(1.0 - np.abs(lum_avg - 0.5) / 0.5, 0.0, 1.0)

        total = (
            wts["coverage"] * recall
            - wts["area_cost"] * area_frac
            + wts["thirds"] * thirds
            - wts["boundary_penalty"] * boundary_penalty
            - wts["tiny_penalty"] * tiny_penalty
            + wts["sharpness"] * sharp_avg
            + wts["colorfulness"] * color_avg
            + wts["saturation"] * sat_avg
            + wts["contrast"] * contrast_norm
            + wts["exposure"] * exposure
        )

        return total, arr
