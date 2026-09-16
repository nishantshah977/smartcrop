"""
Generate candidate crop rectangles to score.

A candidate is a tuple (x0, y0, x1, y1) of integer pixel coordinates,
x0 < x1 <= width, y0 < y1 <= height.
"""

from __future__ import annotations

from typing import List, Tuple

Box = Tuple[int, int, int, int]


def _parse_aspect_ratio(ar) -> float:
    """Accept a float, or a string like '4:5' / '16:9'."""
    if isinstance(ar, (int, float)):
        return float(ar)
    if isinstance(ar, str) and ":" in ar:
        w, h = ar.split(":")
        return float(w) / float(h)
    return float(ar)


def generate_candidates(
    image_width: int,
    image_height: int,
    aspect_ratios: List = ("1:1", "4:5", "3:4", "16:9", "9:16"),
    scales: List[float] = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    x_steps: int = 7,
    y_steps: int = 7,
) -> List[Box]:
    """
    Slide candidate windows of varying aspect ratio and scale across the
    image at a grid of offsets.

    Parameters
    ----------
    image_width, image_height : int
        Dimensions of the source image in pixels.
    aspect_ratios : list of float or "W:H" strings
        Which output aspect ratios to consider. Include the image's own
        aspect ratio if you want "no crop" to be a possible outcome.
    scales : list of float in (0, 1]
        Fraction of the image's limiting dimension the crop window
        should occupy, before aspect-ratio fitting.
    x_steps, y_steps : int
        Number of horizontal / vertical offset positions to try for each
        (aspect_ratio, scale) combination.

    Returns
    -------
    list of (x0, y0, x1, y1) integer boxes, deduplicated.
    """
    img_ar = image_width / image_height
    seen = set()
    boxes: List[Box] = []

    for ar_raw in aspect_ratios:
        ar = _parse_aspect_ratio(ar_raw)

        for scale in scales:
            # Fit a box of this aspect ratio inside scale * image, anchored
            # to the limiting dimension so it never exceeds the source image.
            if ar >= img_ar:
                # wider than the image -> width is the limiting factor
                w = image_width * scale
                h = w / ar
            else:
                h = image_height * scale
                w = h * ar

            w = min(w, image_width)
            h = min(h, image_height)
            if w < 8 or h < 8:
                continue

            max_x0 = image_width - w
            max_y0 = image_height - h

            xs = [0.0] if max_x0 <= 0 else [
                i / (x_steps - 1) * max_x0 for i in range(x_steps)
            ]
            ys = [0.0] if max_y0 <= 0 else [
                i / (y_steps - 1) * max_y0 for i in range(y_steps)
            ]

            for x0 in xs:
                for y0 in ys:
                    box = (
                        int(round(x0)),
                        int(round(y0)),
                        int(round(x0 + w)),
                        int(round(y0 + h)),
                    )
                    box = (
                        max(0, box[0]),
                        max(0, box[1]),
                        min(image_width, box[2]),
                        min(image_height, box[3]),
                    )
                    if box[2] - box[0] < 8 or box[3] - box[1] < 8:
                        continue
                    if box not in seen:
                        seen.add(box)
                        boxes.append(box)

    return boxes
