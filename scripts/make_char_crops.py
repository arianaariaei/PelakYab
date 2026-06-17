#!/usr/bin/env python
"""Build a single-character image dataset from the IR-LPR YOLO labels.

Crops every labeled glyph box out of the plate images and saves balanced,
48x48 BGR crops as one compressed .npz per split — training data for the
two-stage CNN character classifier (scripts/train_char_classifier.py).

Usage:
    python scripts/make_char_crops.py
"""
from __future__ import annotations

import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

SIZE = 48
PAD = 0.12               # fraction of box size added around each glyph
CAPS = {"train": 3500, "val": 1200, "test": 1500}   # max crops per class


def crop_glyph(img, cx, cy, w, h, W, H):
    bw, bh = w * W, h * H
    x1 = int((cx * W) - bw / 2 - bw * PAD)
    y1 = int((cy * H) - bh / 2 - bh * PAD)
    x2 = int((cx * W) + bw / 2 + bw * PAD)
    y2 = int((cy * H) + bh / 2 + bh * PAD)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    if x2 - x1 < 3 or y2 - y1 < 3:
        return None
    g = img[y1:y2, x1:x2]
    return cv2.resize(g, (SIZE, SIZE), interpolation=cv2.INTER_AREA)


def build_split(base: Path, split: str, names: dict) -> None:
    img_dir, lbl_dir = base / "images" / split, base / "labels" / split
    cap = CAPS.get(split, 1500)
    buckets: dict[int, list] = defaultdict(list)
    imgs = sorted(img_dir.glob("*.jpg"))
    random.shuffle(imgs)                      # so caps sample variety, not first-N
    for n, p in enumerate(imgs):
        lbl = lbl_dir / f"{p.stem}.txt"
        if not lbl.exists():
            continue
        rows = [r.split() for r in lbl.read_text().splitlines() if len(r.split()) == 5]
        if not rows:
            continue
        # only read the image if it has a glyph for a not-yet-full class
        if all(len(buckets[int(r[0])]) >= cap for r in rows):
            continue
        img = cv2.imread(str(p))
        if img is None:
            continue
        H, W = img.shape[:2]
        for cid, cx, cy, w, h in rows:
            cid = int(cid)
            if len(buckets[cid]) >= cap:
                continue
            g = crop_glyph(img, float(cx), float(cy), float(w), float(h), W, H)
            if g is not None:
                buckets[cid].append(g)
    X, y = [], []
    for cid, crops in buckets.items():
        for g in crops:
            X.append(g)
            y.append(cid)
    X = np.asarray(X, dtype=np.uint8)
    y = np.asarray(y, dtype=np.int64)
    out = base / f"char_crops_{split}.npz"
    np.savez_compressed(out, X=X, y=y,
                        names=np.array([names[i] for i in range(len(names))], dtype=object))
    per = Counter(int(v) for v in y)
    print(f"[{split}] {len(y)} crops -> {out.name}  "
          f"(min/class {min(per.values())}, max/class {max(per.values())})")


def main() -> int:
    random.seed(0)
    base = ROOT / "datasets" / "ir-lpr" / "yolo"
    names = {int(k): v for k, v in
             yaml.safe_load(open(base / "data.yaml", encoding="utf-8"))["names"].items()}
    print(f"{len(names)} classes")
    for split in ("train", "val", "test"):
        if (base / "images" / split).exists():
            build_split(base, split, names)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
