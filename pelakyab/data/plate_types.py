"""Plate TYPE decoding: the Persian letter (and background color) -> category.

Source: Ghabzino Iranian plate guide. The letter is the primary signal; the
background color disambiguates the few overlaps (e.g. green is shared by Police
and IRGC but the letters differ; red is shared by Government and Protocol).
"""
from __future__ import annotations

from typing import Optional, TypedDict


class PlateType(TypedDict, total=False):
    type: str          # English label
    type_fa: str       # Persian label
    color: str         # canonical background color (see COLORS)
    category: str      # coarse bucket for filtering/stats


COLORS = ("white", "yellow", "green", "red", "blue", "khaki", "brown", "black")

# Private cars all share white background; the specific letter is just a series.
_PRIVATE_LETTERS = ["ب", "ج", "د", "س", "ص", "ط", "ق", "ل", "م", "ن", "و", "ه", "ی"]

LETTER_TYPE: dict[str, PlateType] = {
    # ---- private (white) ----
    **{ltr: {"type": "Private", "type_fa": "شخصی", "color": "white",
             "category": "private"} for ltr in _PRIVATE_LETTERS},

    # ---- public / commercial (yellow) ----
    "ت": {"type": "Taxi", "type_fa": "تاکسی", "color": "yellow", "category": "public"},
    "ع": {"type": "Public", "type_fa": "عمومی", "color": "yellow", "category": "public"},
    "ک": {"type": "Agricultural", "type_fa": "کشاورزی", "color": "yellow",
          "category": "agricultural"},

    # ---- security / military ----
    "پ": {"type": "Police", "type_fa": "پلیس (انتظامی)", "color": "green",
          "category": "police"},
    "ث": {"type": "IRGC", "type_fa": "سپاه", "color": "green", "category": "military"},
    "ش": {"type": "Army", "type_fa": "ارتش (نظامی)", "color": "khaki",
          "category": "military"},
    "ز": {"type": "Defense Ministry", "type_fa": "وزارت دفاع", "color": "blue",
          "category": "military"},
    "ف": {"type": "Armed Forces Command", "type_fa": "ستاد کل نیروهای مسلح",
          "color": "blue", "category": "military"},

    # ---- government ----
    "الف": {"type": "Government", "type_fa": "دولتی", "color": "red",
            "category": "government"},

    # ---- special ----
    "ژ": {"type": "Disabled / Veterans", "type_fa": "معلولین و جانبازان",
          "color": "white", "category": "special"},
    "گ": {"type": "Temporary / Transit", "type_fa": "گذر موقت", "color": "white",
          "category": "temporary"},

    # ---- diplomatic (Latin letters appear on these) ----
    "D": {"type": "Diplomat", "type_fa": "سیاسی", "color": "blue", "category": "diplomatic"},
    "S": {"type": "Service (Diplomatic)", "type_fa": "خدمت (سیاسی)", "color": "blue",
          "category": "diplomatic"},
}

# Multi-character / colored-context plate kinds that have no single letter
# (kept for completeness / manual tagging from color):
COLOR_ONLY_TYPES: dict[str, PlateType] = {
    "red": {"type": "Government", "type_fa": "دولتی", "color": "red", "category": "government"},
    "green": {"type": "Police/IRGC", "type_fa": "انتظامی/سپاه", "color": "green",
              "category": "military"},
    "yellow": {"type": "Public", "type_fa": "عمومی", "color": "yellow", "category": "public"},
    "khaki": {"type": "Army", "type_fa": "نظامی", "color": "khaki", "category": "military"},
    "brown": {"type": "Historic", "type_fa": "تاریخی", "color": "brown", "category": "special"},
    "white": {"type": "Private", "type_fa": "شخصی", "color": "white", "category": "private"},
    "blue": {"type": "Diplomatic/Defense", "type_fa": "سیاسی/دفاعی", "color": "blue",
             "category": "diplomatic"},
}


# ---------------------------------------------------------------------------
# Normalization: map whatever tokens a character-detection model emits to the
# canonical Persian glyphs above. YOLO datasets for Iranian plates label the
# 28 classes in many ways (English transliteration, Persian, alef/A, etc.).
# Extend this map to match YOUR weights' `model.names`.
# ---------------------------------------------------------------------------
DIGIT_NORMALIZATION = {
    # western, persian and arabic-indic digits all -> ascii
    **{str(d): str(d) for d in range(10)},
    **{"۰": "0", "۱": "1", "۲": "2", "۳": "3", "۴": "4",
       "۵": "5", "۶": "6", "۷": "7", "۸": "8", "۹": "9"},
    **{"٠": "0", "١": "1", "٢": "2", "٣": "3", "٤": "4",
       "٥": "5", "٦": "6", "٧": "7", "٨": "8", "٩": "9"},
}

LETTER_NORMALIZATION = {
    # canonical persian letters map to themselves
    **{ltr: ltr for ltr in LETTER_TYPE},
    # common english/transliterated class names -> persian letter
    "alef": "الف", "a": "الف", "aleph": "الف",
    "be": "ب", "b": "ب",
    "pe": "پ", "p": "پ",
    "te": "ت", "t": "ت",
    "se": "ث", "the": "ث",
    "jim": "ج", "j": "ج",
    "dal": "د", "d_letter": "د",
    "sin": "س", "s_letter": "س",
    "sad": "ص",
    "ta": "ط", "taa": "ط",
    "ein": "ع", "ain": "ع", "ayn": "ع",
    "qaf": "ق", "ghaf": "ق",
    "kaf": "ک", "k": "ک",
    "gaf": "گ", "g": "گ",
    "lam": "ل", "l": "ل",
    "mim": "م", "m": "م",
    "noon": "ن", "nun": "ن", "n": "ن",
    "vav": "و", "v": "و", "w": "و",
    "he": "ه", "h": "ه",
    "ye": "ی", "y": "ی",
    "zhe": "ژ", "zh": "ژ",
    "ze": "ز", "z": "ز",
    "fe": "ف", "f": "ف",
    # diplomatic latin letters kept as-is
    "D": "D", "S": "S",
    # disabled wheelchair symbol sometimes a class
    "wheelchair": "ژ", "disabled": "ژ", "malul": "ژ",
}


def normalize_token(token: str) -> Optional[str]:
    """Map a single recognized class label to a canonical digit or letter."""
    if token is None:
        return None
    t = str(token).strip()
    if t in DIGIT_NORMALIZATION:
        return DIGIT_NORMALIZATION[t]
    low = t.lower()
    if low in LETTER_NORMALIZATION:
        return LETTER_NORMALIZATION[low]
    if t in LETTER_NORMALIZATION:
        return LETTER_NORMALIZATION[t]
    # last resort: a single persian char we already know
    if t in LETTER_TYPE:
        return t
    return None


def classify_plate_type(letter: Optional[str], color: Optional[str] = None) -> PlateType:
    """Decide the plate type from the letter, falling back to color."""
    if letter and letter in LETTER_TYPE:
        info = dict(LETTER_TYPE[letter])
        # If the detected color disagrees with the letter's expected color,
        # trust the letter but keep the observed color for the record.
        if color:
            info["observed_color"] = color
        return info  # type: ignore[return-value]
    if color and color in COLOR_ONLY_TYPES:
        info = dict(COLOR_ONLY_TYPES[color])
        info["observed_color"] = color
        return info  # type: ignore[return-value]
    return {"type": "Unknown", "type_fa": "نامشخص", "color": color or "white",
            "category": "unknown"}


def is_letter(token: str) -> bool:
    return normalize_token(token) in LETTER_TYPE


# Persian plate letter -> Latin transliteration, for an all-ASCII "English"
# plate string (avoids bidi reordering and is readable in Latin contexts).
LETTER_LATIN = {
    "الف": "alef", "ب": "be", "پ": "pe", "ت": "te", "ث": "se", "ج": "jim",
    "د": "dal", "ز": "ze", "ژ": "zhe", "س": "sin", "ص": "sad", "ط": "ta",
    "ع": "ein", "ف": "fe", "ق": "qaf", "ک": "kaf", "گ": "gaf", "ل": "lam",
    "م": "mim", "ن": "nun", "و": "vav", "ه": "he", "ی": "ye", "ش": "shin",
    "D": "D", "S": "S",
}


def latin_letter(letter: str) -> str:
    """Latin transliteration of a Persian plate letter (e.g. ع -> 'ein')."""
    if not letter:
        return ""
    return LETTER_LATIN.get(letter, letter)
