"""
Minimal example: run SmartCropper on an image and save both the crop and
a debug visualization.

Usage:
    python examples/run_example.py path/to/photo.jpg
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smartcrop import SmartCropper  # noqa: E402


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_example.py <path_to_image> [aspect_ratio]")
        sys.exit(1)

    image_path = sys.argv[1]
    aspect = sys.argv[2] if len(sys.argv) > 2 else None

    out_dir = Path(__file__).resolve().parent.parent / "outputs"
    out_dir.mkdir(exist_ok=True)

    cropper = SmartCropper()  # uses spectral-residual saliency by default
    result = cropper.crop(image_path, aspect_ratio=aspect)

    stem = Path(image_path).stem
    crop_path = out_dir / f"{stem}_cropped.jpg"
    debug_path = out_dir / f"{stem}_debug.jpg"

    result.image.save(crop_path)
    result.debug_visualization().save(debug_path)

    print(f"Input:  {image_path}")
    print(f"Chosen box: {result.box}  (score={result.score:.4f})")
    print(f"Cropped image saved to: {crop_path}")
    print(f"Debug visualization saved to: {debug_path}")


if __name__ == "__main__":
    main()
