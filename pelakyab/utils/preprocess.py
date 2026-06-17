"""Optional plate-crop preprocessing to help the recognizer on real-world shots.

Borrowed (and hardened) from the sample project: CLAHE contrast boost and a
conservative deskew. Both are OFF by default — they change the input
distribution our recognizer trained on, so enable + A/B test on real captures.
"""
from __future__ import annotations

import cv2
import numpy as np


def clahe(bgr: np.ndarray, clip: float = 2.0, tile: int = 8) -> np.ndarray:
    """Contrast-limited adaptive histogram equalisation on the L channel only,
    so colour (used for plate-type disambiguation) is preserved."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def deskew(bgr: np.ndarray, max_angle: float = 12.0) -> np.ndarray:
    """Rotate a mildly-tilted plate back to horizontal. Conservative: only uses
    near-horizontal long lines, takes the median angle, and no-ops when it isn't
    confident — so it can't scramble an already-straight plate."""
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 180, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80,
                            minLineLength=int(0.4 * w), maxLineGap=10)
    if lines is None or len(lines) < 3:
        return bgr
    angles = []
    for x1, y1, x2, y2 in lines[:, 0]:
        a = np.degrees(np.arctan2(float(y2 - y1), float(x2 - x1)))
        if abs(a) <= max_angle:                 # plate edges / text baseline
            angles.append(a)
    if len(angles) < 3:
        return bgr
    angle = float(np.median(angles))
    if abs(angle) < 1.0:                         # already straight enough
        return bgr
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(bgr, m, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def preprocess_plate(bgr: np.ndarray, do_deskew: bool = False,
                     do_clahe: bool = False) -> np.ndarray:
    """Apply the enabled steps (deskew then contrast) to a plate crop."""
    if bgr is None or bgr.size == 0:
        return bgr
    out = bgr
    if do_deskew:
        out = deskew(out)
    if do_clahe:
        out = clahe(out)
    return out
