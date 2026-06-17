#!/usr/bin/env python
"""Fine-tune a YOLO plate DETECTOR on an Iranian plate dataset.

Expects an Ultralytics-format dataset (data.yaml pointing at train/val images
+ labels) where the class of interest is the whole license plate, e.g. from
IR-LPR or the ANPR-Iranian Roboflow export.

Usage:
    python scripts/train_plate_detector.py --data path/to/data.yaml \
        --base yolov8s.pt --epochs 120 --imgsz 960 --device 0

The best weights are copied to models/plate_detector.pt at the end.
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
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    ap.add_argument("--name", default="plate_detector")
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.base)
    results = model.train(
        data=args.data, epochs=args.epochs, imgsz=args.imgsz,
        batch=args.batch, device=args.device, name=args.name,
        patience=25, cos_lr=True,
        # augmentation that helps small/angled plates at distance:
        degrees=5, translate=0.1, scale=0.5, fliplr=0.0, mosaic=1.0,
    )
    best = Path(results.save_dir) / "weights" / "best.pt"
    dst = ROOT / "models" / "plate_detector.pt"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(best, dst)
    print(f"\nDone. Best weights -> {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
