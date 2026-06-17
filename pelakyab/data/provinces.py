"""Iranian license-plate region codes  ->  province (+ city where known).

The two right-hand digits of a plate are the *region code*. One province has
several codes (one per city/registration office). This table is compiled from
the Ghabzino city guide and cross-checked against the common public NAJA list.

NOTE ON AMBIGUITY
-----------------
A handful of codes are reused across provinces in different public listings
(most notably 32, and the Tehran/Alborz split codes 21/38/68/78). Where a code
is genuinely contested, ``ambiguous=True`` is set and ``alt`` lists the other
candidate(s). Treat those as best-effort and adjust to taste — the structure
makes single-line edits trivial.

Each entry:  code -> {province, province_fa, cities:[...], (ambiguous), (alt)}
"""
from __future__ import annotations

from typing import Optional, TypedDict


class Region(TypedDict, total=False):
    province: str
    province_fa: str
    cities: list[str]
    ambiguous: bool
    alt: list[str]


def _r(province: str, province_fa: str, cities: list[str], **extra) -> Region:
    return {"province": province, "province_fa": province_fa, "cities": cities, **extra}


PROVINCE_CODES: dict[str, Region] = {
    # ---------------- Tehran ----------------
    "10": _r("Tehran", "تهران", ["Tehran"]),
    "11": _r("Tehran", "تهران", ["Tehran"]),
    "20": _r("Tehran", "تهران", ["Tehran"]),
    "22": _r("Tehran", "تهران", ["Tehran"]),
    "33": _r("Tehran", "تهران", ["Tehran"]),
    "40": _r("Tehran", "تهران", ["Tehran"]),
    "44": _r("Tehran", "تهران", ["Tehran"]),
    "55": _r("Tehran", "تهران", ["Tehran"]),
    "66": _r("Tehran", "تهران", ["Tehran"]),
    "77": _r("Tehran", "تهران", ["Tehran"]),
    "88": _r("Tehran", "تهران", ["Tehran"]),
    "99": _r("Tehran", "تهران", ["Tehran"]),
    "30": _r("Tehran", "تهران", ["Shahriar", "Robat Karim"]),

    # ------------- Alborz (Karaj) -----------
    # Split from Tehran in 2010; these codes appear under both in older lists.
    "21": _r("Alborz", "البرز", ["Karaj", "Eslamshahr"], ambiguous=True, alt=["Tehran"]),
    "38": _r("Alborz", "البرز", ["Karaj", "Shahriar"], ambiguous=True, alt=["Tehran"]),
    "68": _r("Alborz", "البرز", ["Karaj"], ambiguous=True, alt=["Tehran"]),
    "78": _r("Alborz", "البرز", ["Karaj", "Nazarabad"], ambiguous=True, alt=["Tehran"]),

    # ---------------- Isfahan ---------------
    "13": _r("Isfahan", "اصفهان", ["Isfahan"]),
    "23": _r("Isfahan", "اصفهان", ["Kashan", "Najafabad", "Shahreza"]),
    "43": _r("Isfahan", "اصفهان", ["Fereydan", "Mobarakeh", "Shahin Shahr"]),
    "53": _r("Isfahan", "اصفهان", ["Isfahan"]),
    "67": _r("Isfahan", "اصفهان", ["Isfahan"]),

    # ------------- Khorasan Razavi ----------
    "12": _r("Khorasan Razavi", "خراسان رضوی", ["Mashhad"]),
    "36": _r("Khorasan Razavi", "خراسان رضوی", ["Mashhad"]),
    "74": _r("Khorasan Razavi", "خراسان رضوی", ["Mashhad", "Quchan"]),
    "42": _r("Khorasan Razavi", "خراسان رضوی", ["Torbat-e Jam", "Fariman", "Sarakhs"]),
    "32": _r("Khorasan Razavi", "خراسان رضوی", ["Neyshabur", "Sabzevar"],
             ambiguous=True, alt=["Khorasan Shomali (Bojnurd)", "Khorasan Jonoubi (Birjand)"]),

    # ------------- Khorasan Shomali ---------
    "26": _r("Khorasan Shomali", "خراسان شمالی", ["Bojnurd", "Shirvan", "Esfarayen"]),

    # ------------- Khorasan Jonoubi ---------
    "52": _r("Khorasan Jonoubi", "خراسان جنوبی", ["Birjand", "Qaen", "Ferdows"]),

    # ----------------- Fars -----------------
    "63": _r("Fars", "فارس", ["Shiraz"]),
    "73": _r("Fars", "فارس", ["Jahrom", "Larestan", "Darab"]),
    "83": _r("Fars", "فارس", ["Marvdasht", "Kazerun", "Fasa"]),
    "93": _r("Fars", "فارس", ["Shiraz"]),

    # --------------- Khuzestan --------------
    "14": _r("Khuzestan", "خوزستان", ["Ahvaz"]),
    "24": _r("Khuzestan", "خوزستان", ["Abadan", "Khorramshahr", "Dezful", "Behbahan"]),

    # ------------ East Azerbaijan -----------
    "15": _r("East Azerbaijan", "آذربایجان شرقی", ["Tabriz"]),
    "25": _r("East Azerbaijan", "آذربایجان شرقی", ["Tabriz", "Maragheh", "Marand"]),
    "35": _r("East Azerbaijan", "آذربایجان شرقی", ["Tabriz", "Ahar", "Mianeh"]),

    # ------------ West Azerbaijan -----------
    "17": _r("West Azerbaijan", "آذربایجان غربی", ["Urmia"]),
    "27": _r("West Azerbaijan", "آذربایجان غربی", ["Khoy", "Mahabad", "Miandoab"]),
    "37": _r("West Azerbaijan", "آذربایجان غربی", ["Urmia", "Bukan", "Salmas"]),

    # ---------------- Ardabil ---------------
    "91": _r("Ardabil", "اردبیل", ["Ardabil", "Parsabad", "Meshgin Shahr"]),

    # ----------------- Gilan ----------------
    "46": _r("Gilan", "گیلان", ["Rasht"]),
    "56": _r("Gilan", "گیلان", ["Bandar-e Anzali", "Lahijan", "Astara"]),
    "76": _r("Gilan", "گیلان", ["Langarud", "Siahkal", "Lahijan"]),

    # -------------- Mazandaran --------------
    "62": _r("Mazandaran", "مازندران", ["Sari"]),
    "72": _r("Mazandaran", "مازندران", ["Babol", "Amol", "Qaemshahr"]),
    "82": _r("Mazandaran", "مازندران", ["Sari", "Behshahr", "Tonekabon"]),
    "92": _r("Mazandaran", "مازندران", ["Chalus", "Nowshahr", "Ramsar"]),

    # ---------------- Golestan --------------
    "59": _r("Golestan", "گلستان", ["Gorgan"]),
    "69": _r("Golestan", "گلستان", ["Gonbad-e Kavus", "Bandar-e Torkaman", "Azadshahr"]),

    # ----------------- Qazvin ---------------
    "79": _r("Qazvin", "قزوین", ["Qazvin"]),
    "89": _r("Qazvin", "قزوین", ["Takestan", "Buin Zahra", "Abyek"]),

    # ------------------ Qom -----------------
    "16": _r("Qom", "قم", ["Qom"]),

    # ---------------- Markazi ---------------
    "47": _r("Markazi", "مرکزی", ["Arak"]),
    "57": _r("Markazi", "مرکزی", ["Saveh", "Khomein", "Mahallat"]),

    # ---------------- Hamadan ---------------
    "18": _r("Hamadan", "همدان", ["Hamadan"]),
    "28": _r("Hamadan", "همدان", ["Nahavand", "Malayer", "Tuyserkan"]),

    # -------------- Kermanshah --------------
    "19": _r("Kermanshah", "کرمانشاه", ["Kermanshah"]),
    "29": _r("Kermanshah", "کرمانشاه", ["Eslamabad-e Gharb", "Gilan-e Gharb", "Sonqor"]),

    # --------------- Kurdistan --------------
    "51": _r("Kurdistan", "کردستان", ["Sanandaj"]),
    "61": _r("Kurdistan", "کردستان", ["Sanandaj", "Bijar", "Baneh", "Saqqez"]),

    # --------------- Lorestan ---------------
    "31": _r("Lorestan", "لرستان", ["Khorramabad"]),
    "41": _r("Lorestan", "لرستان", ["Borujerd", "Aligudarz", "Dorud"]),

    # ------------------ Ilam ----------------
    "98": _r("Ilam", "ایلام", ["Ilam", "Mehran", "Dehloran"]),

    # ---------------- Bushehr ---------------
    "48": _r("Bushehr", "بوشهر", ["Bushehr"]),
    "58": _r("Bushehr", "بوشهر", ["Dashtestan", "Ganaveh", "Kangan"]),

    # ----------------- Kerman ---------------
    "45": _r("Kerman", "کرمان", ["Kerman"]),
    "65": _r("Kerman", "کرمان", ["Rafsanjan", "Bam", "Sirjan"]),
    "75": _r("Kerman", "کرمان", ["Sirjan", "Bam", "Jiroft"]),

    # --------------- Hormozgan --------------
    "84": _r("Hormozgan", "هرمزگان", ["Bandar Abbas"]),
    "94": _r("Hormozgan", "هرمزگان", ["Minab", "Bandar Lengeh", "Jask", "Qeshm"]),

    # --------- Sistan va Baluchestan --------
    "85": _r("Sistan va Baluchestan", "سیستان و بلوچستان", ["Zahedan"]),
    "95": _r("Sistan va Baluchestan", "سیستان و بلوچستان",
             ["Zabol", "Iranshahr", "Khash", "Nikshahr", "Chabahar"]),

    # ------------------ Yazd ----------------
    "54": _r("Yazd", "یزد", ["Yazd"]),
    "64": _r("Yazd", "یزد", ["Ardakan", "Taft", "Meybod", "Bafq"],
             ambiguous=True, alt=["Khorasan Jonoubi (Tabas)"]),

    # ----------------- Semnan ---------------
    "86": _r("Semnan", "سمنان", ["Semnan"]),
    "96": _r("Semnan", "سمنان", ["Damghan", "Shahrud", "Garmsar"]),

    # ----------------- Zanjan ---------------
    "87": _r("Zanjan", "زنجان", ["Zanjan"]),
    "97": _r("Zanjan", "زنجان", ["Abhar", "Khodabandeh", "Ijrud"]),

    # ------- Kohgiluyeh va Boyer-Ahmad ------
    "49": _r("Kohgiluyeh va Boyer-Ahmad", "کهگیلویه و بویراحمد",
             ["Yasuj", "Kohgiluyeh", "Dogonbadan"]),

    # ------- Chaharmahal va Bakhtiari -------
    "71": _r("Chaharmahal va Bakhtiari", "چهارمحال و بختیاری", ["Shahrekord"]),
    "81": _r("Chaharmahal va Bakhtiari", "چهارمحال و بختیاری", ["Borujen", "Ardal", "Farsan"]),
}


def lookup_region(code: str) -> Optional[Region]:
    """Return the region record for a two-digit code, or None if unknown."""
    if code is None:
        return None
    code = str(code).strip()
    if len(code) == 1:
        code = "0" + code
    return PROVINCE_CODES.get(code)


def describe_region(code: str) -> dict:
    """Flatten a region into display-ready fields (always returns a dict)."""
    reg = lookup_region(code)
    if not reg:
        return {
            "region_code": code,
            "province": "Unknown",
            "province_fa": "نامشخص",
            "city": "",
            "cities": [],
            "ambiguous": False,
        }
    cities = reg.get("cities", [])
    return {
        "region_code": code,
        "province": reg["province"],
        "province_fa": reg["province_fa"],
        "city": cities[0] if cities else "",
        "cities": cities,
        "ambiguous": reg.get("ambiguous", False),
        "alt": reg.get("alt", []),
    }
