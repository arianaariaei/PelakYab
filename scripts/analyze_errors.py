#!/usr/bin/env python
"""Error analysis for the character recognizer on a YOLO split.

Breaks down where plate reads go wrong so improvements can be targeted:
  * GT plate-length distribution (8 = standard car plate)
  * plate-exact accuracy stratified by GT length
  * top character substitutions (gt -> pred), measured on equal-length reads
  * no-read characteristics (image resolution)
  * per-class recall (how often each true glyph is read correctly)

Usage:
    python scripts/analyze_errors.py --limit 2000
"""
from __future__ import annotations

import argparse
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from pelakyab.data.plate_types import normalize_token, LETTER_TYPE  # noqa: E402
from pelakyab.detection.char_recognizer import CharRecognizer  # noqa: E402


def gt_tokens(label_path: Path, names: dict) -> list[str]:
    rows = []
    for ln in label_path.read_text(encoding="utf-8").splitlines():
        p = ln.split()
        if len(p) != 5:
            continue
        tok = normalize_token(names.get(int(p[0]), ""))
        if tok:
            rows.append((float(p[1]), tok))
    return [t for _, t in sorted(rows, key=lambda r: r[0])]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="datasets/ir-lpr/yolo")
    ap.add_argument("--split", default="test")
    ap.add_argument("--model", default="models/char_recognizer.pt")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=2000)
    args = ap.parse_args()

    base = ROOT / args.data
    img_dir, lbl_dir = base / "images" / args.split, base / "labels" / args.split
    import yaml
    names = {int(k): v for k, v in
             yaml.safe_load(open(base / "data.yaml", encoding="utf-8"))["names"].items()}

    imgs = sorted(img_dir.glob("*.jpg"))
    random.seed(0)
    if args.limit and args.limit < len(imgs):
        imgs = random.sample(imgs, args.limit)

    cr = CharRecognizer(str(ROOT / args.model), device=args.device, conf=0.30,
                        imgsz=320, half=str(args.device).startswith("cuda"),
                        min_chars=7)
    cr.load()

    len_total, len_exact = Counter(), Counter()
    confus = Counter()
    noread_minside, read_minside = [], []
    cls_total, cls_hit = Counter(), Counter()
    n = 0
    for path in imgs:
        lbl = lbl_dir / f"{path.stem}.txt"
        if not lbl.exists():
            continue
        gt = gt_tokens(lbl, names)
        if not gt:
            continue
        n += 1
        img = cv2.imread(str(path))
        h, w = img.shape[:2]
        res = cr.recognize(img)
        pred = res.tokens if res else []
        len_total[len(gt)] += 1
        if pred == gt:
            len_exact[len(gt)] += 1
        if not pred:
            noread_minside.append(min(w, h))
        else:
            read_minside.append(min(w, h))
        if len(pred) == len(gt):                       # alignable
            for g, p in zip(gt, pred):
                cls_total[g] += 1
                if g == p:
                    cls_hit[g] += 1
                else:
                    confus[(g, p)] += 1
        else:
            for g in gt:
                cls_total[g] += 1

    def pct(a, b):
        return f"{a / b:6.2%}" if b else "   n/a"

    print(f"\nAnalyzed {n} plates from '{args.split}'\n" + "=" * 52)

    print("\nGT plate-length distribution & exact accuracy:")
    for L in sorted(len_total):
        print(f"  len {L:2d}: {len_total[L]:5d} plates   exact "
              f"{pct(len_exact[L], len_total[L])}")
    std = len_total[8]
    print(f"\n  standard 8-glyph plates: {std} "
          f"({std / n:.1%} of set), exact {pct(len_exact[8], std)}")

    print("\nTop character substitutions (true -> read):")
    for (g, p), c in confus.most_common(15):
        gl = "letter" if g in LETTER_TYPE else "digit"
        print(f"  {g} -> {p}   x{c:<4d} ({gl})")

    print("\nWorst per-class recall (>=20 samples):")
    rec = [(g, cls_hit[g] / cls_total[g], cls_total[g])
           for g in cls_total if cls_total[g] >= 20]
    for g, r, t in sorted(rec, key=lambda x: x[1])[:12]:
        gl = "letter" if g in LETTER_TYPE else "digit"
        print(f"  {g} ({gl}): recall {r:6.2%}  (n={t})")

    nr = noread_minside
    if nr:
        nr.sort()
        rd = sorted(read_minside)
        print(f"\nNo-reads: {len(nr)}   median min-side {nr[len(nr)//2]}px "
              f"(read plates median {rd[len(rd)//2]}px)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
