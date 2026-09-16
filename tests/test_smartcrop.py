"""
Lightweight sanity tests (no pytest fixtures needed, no test-image
dependency beyond a synthetic array). Run with:

    python -m pytest tests/
or
    python tests/test_smartcrop.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time

import cv2
import numpy as np

from smartcrop.candidates import generate_candidates
from smartcrop.saliency import compute_saliency
from smartcrop.scorer import score_candidates
from smartcrop.cropper import SmartCropper, find_best_crop
from smartcrop.aesthetics import image_aesthetic_score


def _synthetic_image(w=400, h=300):
    """A dark image with one bright rectangle -> unambiguous saliency target."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[100:200, 250:350] = 255  # bright block, off-center to the right
    return img


def test_generate_candidates_nonempty():
    boxes = generate_candidates(400, 300)
    assert len(boxes) > 0
    for x0, y0, x1, y1 in boxes:
        assert 0 <= x0 < x1 <= 400
        assert 0 <= y0 < y1 <= 300


def test_saliency_map_shape_and_range():
    img = _synthetic_image()
    sal = compute_saliency(img, backend="spectral")
    assert sal.shape == (300, 400)
    assert sal.min() >= 0.0
    assert sal.max() <= 1.0 + 1e-6


def test_scorer_picks_box_containing_bright_region():
    img = _synthetic_image()
    sal = compute_saliency(img, backend="spectral")
    boxes = generate_candidates(400, 300, aspect_ratios=["4:3"], scales=[0.5])
    scored = score_candidates(sal, boxes)
    best = scored[0].box
    x0, y0, x1, y1 = best
    # the best box should at least overlap the bright block region
    overlap = not (x1 <= 250 or x0 >= 350 or y1 <= 100 or y0 >= 200)
    assert overlap, f"Best box {best} does not overlap the salient region"


def test_smartcropper_end_to_end():
    img = _synthetic_image()
    cropper = SmartCropper()
    result = cropper.crop(img, aspect_ratio="1:1")
    x0, y0, x1, y1 = result.box
    assert x1 - x0 == y1 - y0  # square, as requested
    assert result.image.size == (x1 - x0, y1 - y0)


def test_find_best_crop_returns_plain_int_tuple():
    img = _synthetic_image()
    box = find_best_crop(img, aspect_ratio="1:1")
    assert isinstance(box, tuple) and len(box) == 4
    assert all(isinstance(v, int) for v in box)
    x0, y0, x1, y1 = box
    assert 0 <= x0 < x1 <= 400
    assert 0 <= y0 < y1 <= 300


def test_large_image_is_fast():
    """
    Regression test for the original bottleneck: scoring hundreds of
    candidates on a large image must stay well under a second (it used
    to take ~20s before vectorizing + downscaling the analysis step).
    """
    large = np.random.randint(0, 255, (4032, 3024, 3), dtype=np.uint8)
    cropper = SmartCropper()
    t0 = time.perf_counter()
    result = cropper.crop(large, aspect_ratio="4:5")
    elapsed = time.perf_counter() - t0
    assert elapsed < 2.0, f"crop() took {elapsed:.2f}s on a large image, expected < 2s"
    x0, y0, x1, y1 = result.box
    assert 0 <= x0 < x1 <= 3024
    assert 0 <= y0 < y1 <= 4032


def test_image_aesthetic_score_shape():
    img = _synthetic_image()
    scores = image_aesthetic_score(img)
    for key in ("score", "sharpness", "colorfulness", "contrast", "exposure"):
        assert key in scores
        assert 0 <= scores[key] <= 100


def test_result_includes_aesthetic_and_timing():
    img = _synthetic_image()
    cropper = SmartCropper()
    result = cropper.crop(img, aspect_ratio="1:1")
    assert "score" in result.aesthetic
    assert result.elapsed_seconds >= 0


def test_does_not_just_return_the_whole_image():
    """
    Regression test for the "barely crops anything" bug: on an image with
    one small, clearly localized bright region against a large dim area,
    the chosen crop must meaningfully tighten around it rather than
    defaulting to ~the whole frame. (The old `coverage` formula was a raw
    saliency sum, which is monotonically non-decreasing with box size and
    so almost always favored the biggest possible box.)
    """
    h, w = 1200, 900
    img = np.full((h, w, 3), 30, dtype=np.uint8)  # dim background
    img[150:260, 600:760] = 245  # small bright/salient block, off-center

    cropper = SmartCropper()
    result = cropper.crop(img)
    x0, y0, x1, y1 = result.box
    area_frac = ((x1 - x0) * (y1 - y0)) / (w * h)
    assert area_frac < 0.85, (
        f"crop kept {area_frac:.0%} of the frame; expected meaningful "
        "tightening around the small salient region"
    )


def test_aesthetic_terms_dont_override_subject_choice():
    """
    Regression test for a real bug found during tuning: weighting
    colorfulness/saturation too aggressively caused the scorer to prefer
    a saturated-but-empty pink blanket over the much more subject-relevant
    bright window in test_photo2.jpg, just because the blanket scored
    higher on raw saturation/colorfulness. Aesthetic terms should refine
    among subject-relevant candidates, not override subject choice
    entirely -- this checks the crop lands on the upper (window) portion
    of the frame, not the lower (bed) portion.
    """
    photo_path = os.path.join(os.path.dirname(__file__), "..", "examples", "test_photo2.jpg")
    cropper = SmartCropper()
    result = cropper.crop(photo_path, aspect_ratio="1:1")
    x0, y0, x1, y1 = result.box
    cy = (y0 + y1) / 2
    assert cy < 2000, (
        f"crop centered at y={cy:.0f} (image height 4032); expected it to "
        "land on the window (upper portion) rather than the bed (lower "
        "portion) -- looks like aesthetic weights are overpowering subject "
        "selection again"
    )


def test_bundled_nima_model_loads_and_scores():
    """
    The repo ships an actual nima_mobilenet.onnx (converted from
    titu1994/neural-image-assessment's AVA-trained MobileNet weights).
    Confirm it's detected and produces a score in the valid ~1-10 range
    on a real photo, and that use_nima=True runs end to end.
    """
    from smartcrop.aesthetics import nima_available, nima_score

    if not nima_available():
        print("  (skipping: onnxruntime not installed or model missing)")
        return

    photo_path = os.path.join(os.path.dirname(__file__), "..", "examples", "test_photo2.jpg")
    img = cv2.imread(photo_path)
    s = nima_score(img)
    assert 1.0 <= s <= 10.0, f"NIMA score {s} outside valid range"

    cropper = SmartCropper(use_nima=True, nima_top_k=4)
    result = cropper.crop(photo_path, aspect_ratio="4:5")
    assert result.nima_score is not None
    assert 1.0 <= result.nima_score <= 10.0


if __name__ == "__main__":
    test_generate_candidates_nonempty()
    test_saliency_map_shape_and_range()
    test_scorer_picks_box_containing_bright_region()
    test_smartcropper_end_to_end()
    test_find_best_crop_returns_plain_int_tuple()
    test_large_image_is_fast()
    test_image_aesthetic_score_shape()
    test_result_includes_aesthetic_and_timing()
    test_does_not_just_return_the_whole_image()
    test_aesthetic_terms_dont_override_subject_choice()
    test_bundled_nima_model_loads_and_scores()
    print("All tests passed.")
