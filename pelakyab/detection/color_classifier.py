"""Classify the background color of a plate crop -> {white, yellow, green,
red, blue, khaki, brown}.

The plate text is dark, so we mask out dark pixels and judge the dominant
background color from the remaining (bright-ish) pixels in HSV space. This
disambiguates plate types the letter alone can't (and confirms the letter).
"""
from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

# (hue_low, hue_high) in OpenCV's 0..179 hue scale, for saturated colors.
_HUE_RANGES = {
    "red":    [(0, 10), (170, 179)],
    "brown":  [(10, 20)],          # low-value orange/brown handled below
    "yellow": [(20, 38)],
    "green":  [(38, 85)],
    "blue":   [(85, 130)],
}


def classify_color(plate_crop: np.ndarray) -> Tuple[str, float]:
    """Return (color_name, confidence 0..1)."""
    if plate_crop is None or plate_crop.size == 0:
        return "white", 0.0

    img = cv2.resize(plate_crop, (160, 50))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    # Background = not-too-dark pixels (drop the black glyphs and shadows).
    bg_mask = v > 60
    total = int(bg_mask.sum())
    if total < 50:
        return "white", 0.0

    sat = s[bg_mask]
    val = v[bg_mask]
    hue = h[bg_mask]

    # Low saturation + high value => white (private / most common).
    white_frac = float(((sat < 45) & (val > 130)).sum()) / total
    if white_frac > 0.55:
        return "white", min(1.0, white_frac)

    # Otherwise pick the colored bucket with the most saturated pixels.
    colored = sat >= 60
    if colored.sum() < 0.15 * total:
        return "white", 0.5

    ch, cs, cv_ = hue[colored], sat[colored], val[colored]
    counts: dict[str, int] = {}
    for name, ranges in _HUE_RANGES.items():
        m = np.zeros_like(ch, dtype=bool)
        for lo, hi in ranges:
            m |= (ch >= lo) & (ch <= hi)
        counts[name] = int(m.sum())

    # khaki = desaturated olive/yellow with mid value (army plates)
    khaki_mask = ((ch >= 18) & (ch <= 40) & (cs >= 40) & (cs <= 110) & (cv_ < 170))
    counts["khaki"] = int(khaki_mask.sum())

    # brown = orange hue but darker value (historic plates)
    brown_mask = ((ch >= 8) & (ch <= 22) & (cv_ < 130))
    counts["brown"] = int(brown_mask.sum())

    best = max(counts, key=counts.get)
    conf = counts[best] / max(1, int(colored.sum()))
    if counts[best] == 0:
        return "white", 0.4
    return best, float(min(1.0, conf))
