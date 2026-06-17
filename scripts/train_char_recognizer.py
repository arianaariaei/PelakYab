#!/usr/bin/env python
"""Fine-tune a YOLO CHARACTER recognizer for Iranian plates.

The dataset should contain cropped plate images with one bounding box per glyph,
28 classes (10 digits + 18 letters). IR-LPR provides per-character annotations;
Iranis can be used to augment/balance rare letters.

IMPORTANT: note your dataset's class names (model.names). If they aren't already
canonical Persian glyphs/ascii-digits, extend LETTER_NORMALIZATION /
DIGIT_NORMALIZATION in pelakyab/data/plate_types.py so the parser understands
them. Print them after training with:  print(YOLO('best.pt').names)

Usage:
    python scripts/train_char_recognizer.py --data path/to/data.yaml \
        --base yolov8s.pt --epochs 150 --imgsz 320 --device 0
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dataset data.yaml")
    ap.add_argument("--base", default="yolov8s.pt")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", default="0")
    ap.add_argument("--name", default="char_recognizer")
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--cache", default="ram",
                    help="image cache: 'ram', 'disk', or 'none' to disable")
    ap.add_argument("--workers", type=int, default=8,
                    help="dataloader workers; lower this if you hit cuDNN "
                    "HOST_ALLOCATION_FAILED (pinned-memory exhaustion)")
    ap.add_argument("--strong-aug", action="store_true",
                    help="stronger geometric+photometric augmentation")
    ap.add_argument("--mosaic", type=float, default=None,
                    help="override mosaic prob (mosaic=1.0 is CPU/host-memory "
                    "heavy; lower it if the GPU is starved or host RAM is tight)")
    ap.add_argument("--deploy", action=argparse.BooleanOptionalAction,
                    default=True, help="copy best.pt to models/ (use "
                    "--no-deploy to train a candidate without overwriting)")
    ap.add_argument("--resume", default=None,
                    help="resume an interrupted run from its last.pt "
                    "(continues with the run's saved settings)")
    args = ap.parse_args()

    from ultralytics import YOLO

    # Flips stay OFF in both recipes: glyphs are directional and order matters.
    if args.strong_aug:
        # mild skew/scale/lighting variation to harden look-alike glyphs
        # (2<->3, س<->ص) against real-world capture conditions.
        aug = dict(degrees=5.0, shear=2.0, perspective=0.0005,
                   translate=0.1, scale=0.5, mosaic=1.0, close_mosaic=10,
                   hsv_h=0.015, hsv_s=0.7, hsv_v=0.5,
                   fliplr=0.0, flipud=0.0)
    else:
        aug = dict(degrees=0, fliplr=0.0, flipud=0.0, mosaic=0.5, hsv_v=0.4)
    if args.mosaic is not None:
        aug["mosaic"] = args.mosaic

    if args.resume:
        # continue an interrupted run; ultralytics restores epoch/optimizer/LR
        # and the original training args from the run's checkpoint.
        model = YOLO(args.resume)
        results = model.train(resume=True)
    else:
        model = YOLO(args.base)
        results = model.train(
            data=args.data, epochs=args.epochs, imgsz=args.imgsz,
            batch=args.batch, device=args.device, name=args.name,
            patience=args.patience, cos_lr=True, workers=args.workers,
            cache=(args.cache if args.cache in ("ram", "disk") else False), **aug,
        )
    best = Path(results.save_dir) / "weights" / "best.pt"
    print(f"\nDone. Best weights -> {best}")
    print("Class names:", model.names)
    if args.deploy:
        dst = ROOT / "models" / "char_recognizer.pt"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(best, dst)
        print(f"Deployed -> {dst}")
    else:
        print("Not deployed (--no-deploy). Evaluate, then copy manually if better.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
