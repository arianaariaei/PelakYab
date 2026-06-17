"""Drawing helpers, including Persian (RTL) text rendering onto BGR frames.

OpenCV's ``cv2.putText`` cannot render Persian glyphs. We render through PIL
with a Persian-capable TTF font, shaping the text with arabic_reshaper +
python-bidi so letters join correctly and read right-to-left.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Optional, Tuple

import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
    _PIL_OK = True
except Exception:  # pragma: no cover
    _PIL_OK = False

try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    _SHAPER_OK = True
except Exception:  # pragma: no cover
    _SHAPER_OK = False


def shape_persian(text: str) -> str:
    """Reshape + reorder Persian text for correct visual display (RTL)."""
    if not _SHAPER_OK:
        return text
    try:
        return get_display(arabic_reshaper.reshape(text))
    except Exception:
        return text


@lru_cache(maxsize=8)
def _load_font(path: str, size: int):
    if not _PIL_OK:
        return None
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        try:
            return ImageFont.load_default()
        except Exception:
            return None


def draw_persian_text(frame: np.ndarray, text: str, org: Tuple[int, int],
                      font_path: str, font_size: int = 28,
                      color: Tuple[int, int, int] = (255, 255, 255),
                      bg: Optional[Tuple[int, int, int]] = (0, 0, 0)) -> np.ndarray:
    """Draw shaped Persian ``text`` at ``org`` (top-left) on a BGR frame.

    Falls back to cv2.putText (latin only) if PIL/font is unavailable.
    """
    if not _PIL_OK:
        cv2.putText(frame, text, (org[0], org[1] + font_size),
                    cv2.FONT_HERSHEY_SIMPLEX, font_size / 32.0, color, 2)
        return frame

    font = _load_font(font_path, font_size)
    shaped = shape_persian(text)

    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    rgb = (color[2], color[1], color[0])

    if bg is not None and font is not None:
        try:
            l, t, r, b = draw.textbbox(org, shaped, font=font)
            pad = 4
            draw.rectangle([l - pad, t - pad, r + pad, b + pad],
                           fill=(bg[2], bg[1], bg[0]))
        except Exception:
            pass

    draw.text(org, shaped, font=font, fill=rgb)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


# Color used for each plate category's bounding box (BGR).
CATEGORY_COLORS = {
    "private": (60, 200, 60),
    "public": (0, 215, 255),
    "police": (0, 160, 0),
    "military": (50, 90, 130),
    "government": (40, 40, 220),
    "diplomatic": (200, 120, 0),
    "agricultural": (0, 200, 200),
    "special": (200, 0, 200),
    "temporary": (180, 180, 180),
    "unknown": (160, 160, 160),
}


def draw_plate_box(frame: np.ndarray, bbox: Tuple[int, int, int, int],
                   label: str, category: str, font_path: str,
                   confidence: float = 0.0) -> np.ndarray:
    """Draw a bounding box + a Persian/EN label tag for a detected plate."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    color = CATEGORY_COLORS.get(category, (160, 160, 160))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    tag_y = max(0, y1 - 34)
    frame = draw_persian_text(frame, label, (x1, tag_y), font_path,
                              font_size=26, color=(255, 255, 255), bg=color)
    if confidence:
        cv2.putText(frame, f"{confidence:.2f}", (x1, y2 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return frame
