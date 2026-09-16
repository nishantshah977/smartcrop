"""
Score candidate crop boxes against a saliency map.

The total score for a box is a weighted sum of four signals:

1. coverage        -- how much of the total saliency "mass" in the full
                       image ends up inside the box (we want to keep the
                       interesting stuff, not crop it out).
2. thirds           -- how close the saliency-weighted centroid of the box
                       is to a rule-of-thirds intersection point, rewarding
                       compositions that don't just center the subject.
3. boundary_penalty -- penalizes boxes whose edge slices through a region
                       of high saliency (i.e. cuts an object in half).
4. size_penalty     -- mild penalty for crops that are almost the entire
                       image (no real framing decision) or very tiny
                       (loses context).

All four are computed on the saliency map directly with integral images,
so scoring thousands of candidates is cheap (no per-pixel Python loops).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .candidates import Box


@dataclass
class ScoredBox:
    box: Box
    score: float
    coverage: float
    thirds: float
    boundary_penalty: float
    size_penalty: float


def _integral(saliency: np.ndarray) -> np.ndarray:
    """cv2/np integral image with a leading zero row/col for easy lookup."""
    return np.pad(saliency, ((1, 0), (1, 0))).cumsum(0).cumsum(1)


def _box_sum(integral: np.ndarray, box: Box) -> float:
    x0, y0, x1, y1 = box
    return float(
        integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]
    )


def _centroid(saliency: np.ndarray, box: Box) -> Tuple[float, float]:
    x0, y0, x1, y1 = box
    region = saliency[y0:y1, x0:x1]
    total = region.sum()
    if total <= 1e-8:
        # no salient content -> geometric center of the box
        return (x0 + x1) / 2.0, (y0 + y1) / 2.0
    ys, xs = np.mgrid[y0:y1, x0:x1]
    cx = float((xs * region).sum() / total)
    cy = float((ys * region).sum() / total)
    return cx, cy


def _thirds_score(cx: float, cy: float, box: Box) -> float:
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    if w <= 0 or h <= 0:
        return 0.0
    # normalized position within the box, 0..1
    nx, ny = (cx - x0) / w, (cy - y0) / h
    third_lines = (1 / 3, 2 / 3)
    dx = min(abs(nx - t) for t in third_lines)
    dy = min(abs(ny - t) for t in third_lines)
    dist = (dx**2 + dy**2) ** 0.5
    # dist ranges 0 (on an intersection) to ~0.47 (dead center); convert to
    # a 0..1 reward, higher is better.
    return max(0.0, 1.0 - dist / 0.35)


def _boundary_penalty(saliency: np.ndarray, box: Box, band: int = 3) -> float:
    """
    Average saliency in a thin band just inside each edge of the box.
    High values here mean the crop is slicing through something salient.
    """
    x0, y0, x1, y1 = box
    h_img, w_img = saliency.shape
    band = max(1, band)

    strips = []
    if y0 + band <= y1:
        strips.append(saliency[y0 : y0 + band, x0:x1])
    if y1 - band >= y0 and y1 < h_img:
        strips.append(saliency[max(y0, y1 - band) : y1, x0:x1])
    if x0 + band <= x1:
        strips.append(saliency[y0:y1, x0 : x0 + band])
    if x1 - band >= x0 and x1 < w_img:
        strips.append(saliency[y0:y1, max(x0, x1 - band) : x1])

    if not strips:
        return 0.0
    vals = np.concatenate([s.ravel() for s in strips if s.size])
    if vals.size == 0:
        return 0.0
    return float(vals.mean())


def score_candidates(
    saliency: np.ndarray,
    boxes: List[Box],
    weights: dict = None,
) -> List[ScoredBox]:
    """
    Score every candidate box against the saliency map.

    weights : dict, optional
        Override the default weighting of the four sub-scores. Keys:
        "coverage", "thirds", "boundary_penalty", "size_penalty".
    """
    default_weights = {
        "coverage": 1.0,
        "thirds": 0.5,
        "boundary_penalty": 1.2,
        "size_penalty": 0.4,
    }
    if weights:
        default_weights.update(weights)
    w = default_weights

    h_img, w_img = saliency.shape
    total_mass = max(float(saliency.sum()), 1e-8)
    integral = _integral(saliency)
    img_area = float(h_img * w_img)

    results = []
    for box in boxes:
        x0, y0, x1, y1 = box
        box_mass = _box_sum(integral, box)
        coverage = box_mass / total_mass  # 0..1, higher = keeps more salient content

        cx, cy = _centroid(saliency, box)
        thirds = _thirds_score(cx, cy, box)

        boundary = _boundary_penalty(saliency, box)

        area_frac = ((x1 - x0) * (y1 - y0)) / img_area
        # penalize both "basically the whole image" and "tiny sliver"
        size_penalty = 0.0
        if area_frac > 0.92:
            size_penalty = (area_frac - 0.92) / 0.08
        elif area_frac < 0.15:
            size_penalty = (0.15 - area_frac) / 0.15

        total = (
            w["coverage"] * coverage
            + w["thirds"] * thirds
            - w["boundary_penalty"] * boundary
            - w["size_penalty"] * size_penalty
        )

        results.append(
            ScoredBox(
                box=box,
                score=total,
                coverage=coverage,
                thirds=thirds,
                boundary_penalty=boundary,
                size_penalty=size_penalty,
            )
        )

    results.sort(key=lambda r: r.score, reverse=True)
    return results
