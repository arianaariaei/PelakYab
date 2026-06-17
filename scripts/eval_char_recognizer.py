#!/usr/bin/env python
"""Evaluate the character recognizer on a YOLO split (default: IR-LPR test).

Runs the *production* ``CharRecognizer`` (same ordering + normalization +
letter-dedupe the live pipeline uses) over each plate image and compares the
read token sequence against the ground-truth label, reporting:

  * plate-exact accuracy  (every glyph correct, in order)
  * character accuracy    (Levenshtein-based, order-sensitive)
  * no-read rate          (fewer than min_chars glyphs detected)

Usage:
    python scripts/eval_char_recognizer.py
    python scripts/eval_char_recognizer.py --split test --limit 500 --device 0
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from pelakyab.data.plate_types import normalize_token  # noqa: E402
from pelakyab.detection.char_recognizer import CharRecognizer  # noqa: E402


def gt_tokens(label_path: Path, names: dict[int, str]) -> list[str]:
    """Ground-truth glyphs, ordered left-to-right by box center-x."""
    rows = []
    for ln in label_path.read_text(encoding="utf-8").splitlines():
        p = ln.split()
        if len(p) != 5:
            continue
        cid, cx = int(p[0]), float(p[1])
        tok = normalize_token(names.get(cid, ""))
        if tok:
            rows.append((cx, tok))
    return [t for _, t in sorted(rows, key=lambda r: r[0])]


def edit_distance(a: list[str], b: list[str]) -> int:
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, dp[0] = dp[0], i
        for j, cb in enumerate(b, 1):
            prev, dp[j] = dp[j], min(dp[j] + 1, dp[j - 1] + 1,
                                     prev + (ca != cb))
    return dp[-1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="datasets/ir-lpr/yolo")
    ap.add_argument("--split", default="test")
    ap.add_argument("--model", default="models/char_recognizer.pt")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--limit", type=int, default=500,
                    help="evaluate a random N images (0 = all)")
    ap.add_argument("--examples", type=int, default=8)
    ap.add_argument("--two-stage", action="store_true",
                    help="re-decide the letter with the CNN classifier")
    ap.add_argument("--classifier", default="models/char_classifier.pt")
    args = ap.parse_args()

    base = (ROOT / args.data) if not Path(args.data).is_absolute() else Path(args.data)
    img_dir = base / "images" / args.split
    lbl_dir = base / "labels" / args.split
    imgs = sorted(img_dir.glob("*.jpg"))
    if not imgs:
        print(f"No images in {img_dir}")
        return 1
    random.seed(0)
    if args.limit and args.limit < len(imgs):
        imgs = random.sample(imgs, args.limit)

    # GT classes come from data.yaml (the dataset's own definition) so the
    # evaluation is independent of whichever model is under test.
    import yaml
    with open(base / "data.yaml", encoding="utf-8") as f:
        names = {int(k): v for k, v in yaml.safe_load(f)["names"].items()}

    cr = CharRecognizer(str((ROOT / args.model)), device=args.device,
                        conf=args.conf, imgsz=args.imgsz,
                        half=str(args.device).startswith("cuda"), min_chars=7,
                        classifier_path=str(ROOT / args.classifier),
                        two_stage_letter=args.two_stage)
    cr.load()

    n = exact = noread = 0
    tot_chars = tot_err = 0
    examples = []
    for path in imgs:
        lbl = lbl_dir / f"{path.stem}.txt"
        if not lbl.exists():
            continue
        gt = gt_tokens(lbl, names)
        if not gt:
            continue
        n += 1
        img = cv2.imread(str(path))
        res = cr.recognize(img)
        pred = res.tokens if res else []
        if not pred:
            noread += 1
        tot_chars += len(gt)
        tot_err += edit_distance(pred, gt)
        ok = pred == gt
        exact += int(ok)
        if len(examples) < args.examples and not ok and pred:
            examples.append((path.name, "".join(gt), "".join(pred)))

    print(f"\nEvaluated {n} plates from '{args.split}' "
          f"(model: {args.model})")
    print("-" * 48)
    print(f"  plate-exact accuracy : {exact / n:6.2%}  ({exact}/{n})")
    print(f"  character accuracy   : {1 - tot_err / max(1, tot_chars):6.2%}")
    print(f"  no-read rate         : {noread / n:6.2%}  ({noread}/{n})")
    if examples:
        print("\n  sample mismatches (gt -> pred):")
        for fn, g, p in examples:
            print(f"    {fn:14s} {g}  ->  {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
