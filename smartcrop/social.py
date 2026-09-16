"""
Aspect-ratio presets matching common social platforms/placements, so you
can ask for "the Instagram post crop" instead of remembering "4:5".

Platforms sometimes support a range (Instagram accepts 1.91:1 to 4:5, for
example) -- these presets pick the commonly-recommended single ratio for
each, not the full allowed range.
"""

from __future__ import annotations

SOCIAL_PRESETS = {
    # square + portrait feed posts
    "instagram_post": "1:1",
    "instagram_portrait": "4:5",     # taller feed post -- takes up more screen space
    "facebook_post": "1:1",
    "linkedin_post": "1:1",
    # full-screen vertical (stories / reels / shorts / tiktok)
    "instagram_story": "9:16",
    "instagram_reel": "9:16",
    "tiktok": "9:16",
    "youtube_shorts": "9:16",
    "facebook_story": "9:16",
    # landscape
    "twitter_post": "16:9",
    "x_post": "16:9",
    "youtube_thumbnail": "16:9",
    "linkedin_banner": "16:9",
    # other
    "pinterest_pin": "2:3",
}


def resolve_aspect_ratio(platform: str) -> str:
    """
    Look up a platform name (case/space/dash-insensitive) in SOCIAL_PRESETS.
    Raises KeyError with the list of valid names if not found.
    """
    key = platform.strip().lower().replace(" ", "_").replace("-", "_")
    if key not in SOCIAL_PRESETS:
        raise KeyError(
            f"Unknown platform {platform!r}. Valid options: {sorted(SOCIAL_PRESETS)}"
        )
    return SOCIAL_PRESETS[key]
