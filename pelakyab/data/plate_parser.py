"""Turn a left-to-right list of recognized character tokens into a Plate.

Standard Iranian car plate (8 glyphs, read L->R):

        ┌────┬───┬─────┬──────────┐
        │ NN │ L │ NNN │  ایران   │   <- "Iran" + region code at far right
        │ 12 │ ب │ 345 │    68    │
        └────┴───┴─────┴──────────┘
          ^^   ^    ^^^      ^^
        left  letter serial  region code

Display order in Persian goes from the two left digits to the region code.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

from . import plate_types as pt
from .provinces import describe_region


@dataclass
class Plate:
    # raw recognized tokens (canonicalized)
    tokens: list[str] = field(default_factory=list)
    left: str = ""            # 2 digits
    letter: str = ""          # 1 persian letter (series / type)
    serial: str = ""          # 3 digits
    region_code: str = ""     # 2 digits

    # decoded metadata
    province: str = ""
    province_fa: str = ""
    city: str = ""
    cities: list[str] = field(default_factory=list)
    plate_type: str = ""
    plate_type_fa: str = ""
    color: str = ""
    category: str = ""
    ambiguous_region: bool = False

    confidence: float = 0.0
    valid: bool = False

    # ---- presentation ----
    @property
    def key(self) -> str:
        """Normalized identity used for de-duplication."""
        return f"{self.left}{self.letter}{self.serial}-{self.region_code}"

    @property
    def letter_latin(self) -> str:
        return pt.latin_letter(self.letter)

    @property
    def display_en(self) -> str:
        # all-ASCII (e.g. "43 ein 971 - 51") so no RTL char to reorder
        return f"{self.left} {self.letter_latin} {self.serial} - {self.region_code}"

    @property
    def display_fa(self) -> str:
        # Prefix EVERY token with a Left-to-Right Mark so the bidi algorithm
        # keeps the groups in physical plate order (two digits on the LEFT,
        # region code on the RIGHT) while each Persian word still shapes RTL
        # internally. A single leading LRM is NOT enough — the strong-RTL
        # letter and "ایران" otherwise flip the whole group order.
        lrm = "‎"
        parts = [self.left, self.letter, self.serial, "ایران", self.region_code]
        return " ".join(lrm + p for p in parts if p)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["key"] = self.key
        d["display_en"] = self.display_en
        d["display_fa"] = self.display_fa
        return d


def parse_plate(tokens: list[str], color: Optional[str] = None,
                confidence: float = 0.0, expected: int = 8) -> Plate:
    """Build a Plate from canonical tokens (already normalized digits/letters).

    Tolerant: locates the single letter token and slots digits around it, so a
    missing/extra digit degrades gracefully instead of throwing.
    """
    plate = Plate(tokens=list(tokens), confidence=float(confidence))

    # split into the letter (first non-digit) and the digit stream
    letter_idx = next((i for i, t in enumerate(tokens) if t in pt.LETTER_TYPE), None)
    digits = [t for t in tokens if t.isdigit()]

    if letter_idx is not None:
        plate.letter = tokens[letter_idx]
        # digits before the letter are the "left" pair; rest are serial+region
        before = [t for t in tokens[:letter_idx] if t.isdigit()]
        after = [t for t in tokens[letter_idx + 1:] if t.isdigit()]
    else:
        before, after = digits[:2], digits[2:]

    plate.left = "".join(before[:2])
    # of the digits after the letter, the LAST two are the region code,
    # the ones before that (usually 3) are the serial.
    if len(after) >= 2:
        plate.region_code = "".join(after[-2:])
        plate.serial = "".join(after[:-2])
    else:
        plate.serial = "".join(after)

    # ---- decode type (letter + color) ----
    type_info = pt.classify_plate_type(plate.letter or None, color)
    plate.plate_type = type_info.get("type", "Unknown")
    plate.plate_type_fa = type_info.get("type_fa", "نامشخص")
    plate.color = color or type_info.get("color", "")
    plate.category = type_info.get("category", "unknown")

    # ---- decode region (province / city) ----
    if len(plate.region_code) == 2:
        reg = describe_region(plate.region_code)
        plate.province = reg["province"]
        plate.province_fa = reg["province_fa"]
        plate.city = reg["city"]
        plate.cities = reg["cities"]
        plate.ambiguous_region = reg.get("ambiguous", False)

    # ---- validity check ----
    plate.valid = (
        len(plate.left) == 2
        and bool(plate.letter)
        and len(plate.serial) == 3
        and len(plate.region_code) == 2
    )
    return plate
