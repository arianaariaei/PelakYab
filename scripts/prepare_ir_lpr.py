#!/usr/bin/env python
"""Convert the IR-LPR "License Plate Images" set (Pascal VOC XML, one box per
glyph) into a YOLO character-detection dataset + data.yaml.

IR-LPR ships each split as a zip of paired ``NNNNN.jpg`` / ``NNNNN.xml`` files.
Each XML has per-character ``<object><name>..</name><bndbox>..</bndbox></object>``
entries but **no <size>**, so image dimensions are read from the JPEGs here.

Class names are normalized to the canonical glyphs the parser understands
(``pelakyab.data.plate_types``):
  * ZWJ/ZWNJ stripped            (e.g. "ه‍" -> "ه")
  * verbose special-plate label  ("ژ (معلولین و جانبازان)" -> "ژ")
Digits stay ASCII; single Persian letters pass through unchanged.

Usage:
    python scripts/prepare_ir_lpr.py
    python scripts/prepare_ir_lpr.py --zips-dir datasets/ir-lpr/_zips \
        --out datasets/ir-lpr/yolo
"""
from __future__ import annotations

import argparse
import io
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Windows console is cp1252 and can't encode Persian; print UTF-8 (or replace).
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from PIL import Image  # noqa: E402

from pelakyab.data.plate_types import LETTER_LATIN, normalize_token  # noqa: E402

# zip filename (in --zips-dir) -> YOLO split name
SPLITS = {"lp_train.zip": "train", "lp_val.zip": "val", "lp_test.zip": "test"}

_OBJ_RE = re.compile(
    r"<object>.*?<name>(?P<name>.*?)</name>.*?"
    r"<xmin>(?P<xmin>-?\d+)</xmin>.*?<ymin>(?P<ymin>-?\d+)</ymin>.*?"
    r"<xmax>(?P<xmax>-?\d+)</xmax>.*?<ymax>(?P<ymax>-?\d+)</ymax>.*?</object>",
    re.DOTALL,
)

ZW = {"‌", "‍"}  # zero-width non-joiner / joiner


def clean_name(raw: str) -> str:
    """Map a raw IR-LPR class name to a canonical glyph/digit token."""
    t = raw.strip()
    if t.startswith("ژ"):           # "ژ (معلولین و جانبازان)" special plate
        return "ژ"
    for z in ZW:
        t = t.replace(z, "")
    return t


def parse_objects(xml_text: str) -> list[tuple[str, int, int, int, int]]:
    out = []
    for m in _OBJ_RE.finditer(xml_text):
        out.append((clean_name(m.group("name")),
                    int(m.group("xmin")), int(m.group("ymin")),
                    int(m.group("xmax")), int(m.group("ymax"))))
    return out


def jpg_xml_pairs(zf: zipfile.ZipFile) -> list[tuple[str, str]]:
    members = set(zf.namelist())
    pairs = []
    for n in members:
        if n.lower().endswith(".jpg"):
            xml = n[:-4] + ".xml"
            if xml in members:
                pairs.append((n, xml))
    return sorted(pairs)


def build_class_order(found: set[str]) -> list[str]:
    """Deterministic class list: digits 0-9, then letters in canonical order."""
    digits = [str(d) for d in range(10)]
    letters_present = found - set(digits)
    master = list(LETTER_LATIN.keys())            # canonical-ish letter order
    ordered = [l for l in master if l in letters_present]
    ordered += sorted(letters_present - set(master), key=lambda s: [ord(c) for c in s])
    return digits + ordered


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zips-dir", default="datasets/ir-lpr/_zips")
    ap.add_argument("--out", default="datasets/ir-lpr/yolo")
    args = ap.parse_args()

    zips_dir = (ROOT / args.zips_dir) if not Path(args.zips_dir).is_absolute() \
        else Path(args.zips_dir)
    out = (ROOT / args.out) if not Path(args.out).is_absolute() else Path(args.out)

    present = [z for z in SPLITS if (zips_dir / z).exists()]
    if not present:
        print(f"No IR-LPR zips found in {zips_dir} (expected {list(SPLITS)})")
        return 1
    print(f"Found splits: {[SPLITS[z] for z in present]}")

    # ---- pass A: collect the full class set across every split ----
    found: set[str] = set()
    unknown: set[str] = set()
    for zname in present:
        with zipfile.ZipFile(zips_dir / zname) as zf:
            for _, xml in jpg_xml_pairs(zf):
                for name, *_ in parse_objects(zf.read(xml).decode("utf-8")):
                    found.add(name)
                    if normalize_token(name) is None:
                        unknown.add(name)
    if unknown:
        print(f"  WARNING: {len(unknown)} class name(s) the parser can't map: "
              f"{sorted(unknown)}")
    names = build_class_order(found)
    name2id = {n: i for i, n in enumerate(names)}
    print(f"  {len(names)} classes: {names}")

    # ---- pass B: extract images + write YOLO labels ----
    for zname in present:
        split = SPLITS[zname]
        img_dir = out / "images" / split
        lbl_dir = out / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        n_img = n_box = n_skip = 0
        with zipfile.ZipFile(zips_dir / zname) as zf:
            for jpg, xml in jpg_xml_pairs(zf):
                img_bytes = zf.read(jpg)
                try:
                    with Image.open(io.BytesIO(img_bytes)) as im:
                        W, H = im.size
                except Exception:
                    n_skip += 1
                    continue
                objs = parse_objects(zf.read(xml).decode("utf-8"))
                lines = []
                for name, xmin, ymin, xmax, ymax in objs:
                    cid = name2id.get(name)
                    if cid is None:
                        continue
                    # clamp BOTH corners into the image (some boxes are
                    # annotated partly/fully outside it) so coords stay in [0,1]
                    x1, x2 = sorted((min(max(0, xmin), W), min(max(0, xmax), W)))
                    y1, y2 = sorted((min(max(0, ymin), H), min(max(0, ymax), H)))
                    bw, bh = x2 - x1, y2 - y1
                    if bw <= 1 or bh <= 1:
                        continue
                    cx, cy = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H
                    lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw / W:.6f} {bh / H:.6f}")
                stem = Path(jpg).stem
                (img_dir / f"{stem}.jpg").write_bytes(img_bytes)
                (lbl_dir / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")
                n_img += 1
                n_box += len(lines)
        print(f"  [{split:5s}] {n_img} images, {n_box} boxes"
              + (f", {n_skip} unreadable" if n_skip else ""))

    # ---- data.yaml ----
    val_split = "val" if (out / "images" / "val").exists() else "train"
    yaml_lines = [f"path: {out.as_posix()}",
                  "train: images/train",
                  f"val: images/{val_split}"]
    if (out / "images" / "test").exists():
        yaml_lines.append("test: images/test")
    yaml_lines.append("names:")
    yaml_lines += [f"  {i}: '{n}'" for i, n in enumerate(names)]
    data_yaml = out / "data.yaml"
    data_yaml.write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")
    print(f"\nWrote {data_yaml}")
    print("Next: python scripts/train_char_recognizer.py "
          f"--data {data_yaml.as_posix()} --device 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
