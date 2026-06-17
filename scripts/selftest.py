#!/usr/bin/env python
"""Pure-Python self-test of the decoding brain (no cv2/torch needed).

Validates token normalization, plate parsing, type decoding, and province
lookup. Run:  python scripts/selftest.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Windows consoles default to cp1252 and choke on Persian glyphs in output.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pelakyab.data.plate_parser import parse_plate
from pelakyab.data.plate_types import normalize_token
from pelakyab.data.provinces import lookup_region, describe_region

PASS, FAIL = 0, 0


def check(name, got, want):
    global PASS, FAIL
    if got == want:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}: got {got!r}, want {want!r}")


def main() -> int:
    print("normalize_token")
    check("persian digit ۵", normalize_token("۵"), "5")
    check("english 'be' -> ب", normalize_token("be"), "ب")
    check("english 'alef' -> الف", normalize_token("alef"), "الف")
    check("ascii digit 7", normalize_token("7"), "7")

    print("\nprovince lookup")
    check("63 -> Fars", lookup_region("63")["province"], "Fars")
    check("12 -> Khorasan Razavi", lookup_region("12")["province"], "Khorasan Razavi")
    check("14 city Ahvaz", describe_region("14")["city"], "Ahvaz")
    check("unknown 00", lookup_region("00"), None)

    print("\nplate parsing (private, Tehran 11)")
    p = parse_plate(["1", "2", "ب", "3", "4", "5", "1", "1"], color="white",
                    confidence=0.9)
    check("valid", p.valid, True)
    check("left", p.left, "12")
    check("letter", p.letter, "ب")
    check("serial", p.serial, "345")
    check("region", p.region_code, "11")
    check("type", p.plate_type, "Private")
    check("province", p.province, "Tehran")
    check("key", p.key, "12ب345-11")
    check("letter_latin", p.letter_latin, "be")
    check("display_en", p.display_en, "12 be 345 - 11")

    print("\nplate parsing (public, yellow, ع)")
    pub = parse_plate(["4", "5", "ع", "6", "7", "8", "6", "3"], color="yellow",
                      confidence=0.8)
    check("public type", pub.plate_type, "Public")
    check("public category", pub.category, "public")
    check("Fars region", pub.province, "Fars")

    print("\nplate parsing (taxi, yellow, ت)")
    taxi = parse_plate(["2", "2", "ت", "1", "1", "1", "6", "3"], color="yellow",
                       confidence=0.8)
    check("taxi type", taxi.plate_type, "Taxi")
    check("taxi category", taxi.category, "public")

    print("\nplate parsing (government, الف, red)")
    g = parse_plate(["1", "0", "الف", "2", "0", "0", "1", "1"], color="red",
                    confidence=0.7)
    check("gov type", g.plate_type, "Government")
    check("gov category", g.category, "government")

    print(f"\n{PASS} passed, {FAIL} failed")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
