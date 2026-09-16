"""
Command-line interface for smartcrop.

Examples
--------
    # Just print the crop rectangle (fast path, no file written):
    python -m smartcrop.cli photo.jpg --aspect 4:5 --coords-only

    # Crop for a named social platform instead of an aspect ratio:
    python -m smartcrop.cli photo.jpg --platform instagram_story --coords-only

    # Full run: save cropped image + debug visualization
    python -m smartcrop.cli photo.jpg --out cropped.jpg --aspect 4:5 --debug debug.jpg

    # Re-rank with the bundled learned aesthetic model (NIMA):
    python -m smartcrop.cli photo.jpg --aspect 4:5 --use-nima --coords-only
"""

from __future__ import annotations

import argparse
import json
import sys

from .cropper import SmartCropper


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="smartcrop",
        description="Find the best crop rectangle for an image, fast.",
    )
    parser.add_argument("input", help="Path to the source image")
    parser.add_argument(
        "--out", "-o", default=None,
        help="Where to save the cropped result. If omitted and --coords-only "
             "is not set, defaults to <input>_cropped.<ext>",
    )
    parser.add_argument(
        "--aspect", "-a", default=None,
        help="Target aspect ratio, e.g. '1:1', '4:5', '16:9'. "
             "If omitted, several ratios are tried and the best overall is picked.",
    )
    parser.add_argument(
        "--platform", "-p", default=None,
        help="Crop for a named social platform instead of --aspect, e.g. "
             "'instagram_post', 'instagram_story', 'twitter_post'. "
             "See smartcrop.social.SOCIAL_PRESETS for the full list.",
    )
    parser.add_argument(
        "--backend", default="auto", choices=["auto", "spectral", "fine", "u2net"],
        help="Saliency backend to use (default: auto)",
    )
    parser.add_argument(
        "--analysis-size", type=int, default=480,
        help="Longest-side cap (px) for internal analysis resolution. "
             "Lower = faster, higher = slightly more precise. Default: 480",
    )
    parser.add_argument(
        "--use-nima", action="store_true",
        help="Re-rank the top heuristic candidates with the bundled NIMA "
             "learned aesthetic-quality model (smartcrop/models/"
             "nima_mobilenet.onnx). Requires `pip install onnxruntime`. "
             "Adds ~200-400ms. Silently ignored if onnxruntime isn't "
             "installed.",
    )
    parser.add_argument(
        "--nima-top-k", type=int, default=8,
        help="How many top heuristic candidates to re-rank with NIMA "
             "when --use-nima is set. Default: 8",
    )
    parser.add_argument(
        "--coords-only", action="store_true",
        help="Only compute and print the crop rectangle (and aesthetic "
             "score) as JSON; do not read/write image files for output.",
    )
    parser.add_argument(
        "--debug", default=None,
        help="If set, also save a debug visualization image to this path "
             "(saliency heatmap + candidate boxes + final crop side by side).",
    )
    args = parser.parse_args(argv)

    aspect = args.aspect
    if args.platform:
        from .social import resolve_aspect_ratio
        aspect = resolve_aspect_ratio(args.platform)

    cropper = SmartCropper(
        saliency_backend=args.backend,
        analysis_size=args.analysis_size,
        use_nima=args.use_nima,
        nima_top_k=args.nima_top_k,
    )
    result = cropper.crop(args.input, aspect_ratio=aspect)

    payload = {
        "box": list(result.box),
        "score": round(result.score, 4),
        "elapsed_seconds": round(result.elapsed_seconds, 4),
        "aesthetic": result.aesthetic,
    }
    if result.nima_score is not None:
        payload["nima_score"] = round(result.nima_score, 3)

    if args.coords_only:
        print(json.dumps(payload, indent=2))
        return 0

    out_path = args.out
    if out_path is None:
        if "." in args.input:
            stem, ext = args.input.rsplit(".", 1)
            out_path = f"{stem}_cropped.{ext}"
        else:
            out_path = args.input + "_cropped.jpg"

    result.image.save(out_path)
    print(json.dumps(payload, indent=2))
    print(f"Saved cropped image to: {out_path}")

    if args.debug:
        result.debug_visualization().save(args.debug)
        print(f"Saved debug visualization to: {args.debug}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
