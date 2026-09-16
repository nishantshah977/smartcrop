from .cropper import SmartCropper, CropResult, find_best_crop, crop_for_social
from .saliency import compute_saliency
from .aesthetics import image_aesthetic_score
from .social import SOCIAL_PRESETS

__all__ = [
    "SmartCropper",
    "CropResult",
    "find_best_crop",
    "crop_for_social",
    "compute_saliency",
    "image_aesthetic_score",
    "SOCIAL_PRESETS",
]
__version__ = "0.3.0"
