#!/usr/bin/env python
"""Helper to obtain the two YOLO weight files PelakYab needs:

    models/plate_detector.pt   - detects the plate region in a frame
    models/char_recognizer.pt  - detects+classifies the 8 glyphs in a plate

There is no single "official" public download, so this script does two things:
  1) Prints exactly where to get good Iranian-plate weights/datasets.
  2) Optionally fetches a generic YOLO base you can fine-tune (``--base``),
     so the app can at least start while you train the real models.

Recommended sources (free, Iranian-specific):
  * ANPR-YOLOv8 (weights + Roboflow dataset):
        https://github.com/barzansaeedpour/ANPR-YOLOv8
        https://universe.roboflow.com/barzansaeedpour/anpr-iranian-2
  * IR-LPR dataset (20,967 imgs, plate + per-character annotations):
        https://github.com/mut-deep/IR-LPR
  * Iranis dataset (~83k Persian plate-character crops, 28 classes):
        search "Iranis dataset license plate characters"

Place the resulting .pt files at the two paths above (see config.yaml).
"""
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"

SOURCES = """
================  WHERE TO GET IRANIAN PLATE WEIGHTS  ================
Plate detector + character recognizer (pretrained, Iranian):
  - https://github.com/barzansaeedpour/ANPR-YOLOv8   (YOLOv8, ready weights)
  - Roboflow: https://universe.roboflow.com/barzansaeedpour/anpr-iranian-2

Datasets to train/fine-tune for production accuracy:
  - IR-LPR  : https://github.com/mut-deep/IR-LPR    (plate + char boxes)
  - Iranis  : ~83k character crops, 28 classes

After downloading, save as:
  models/plate_detector.pt
  models/char_recognizer.pt
=====================================================================
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", choices=["yolov8n", "yolov8s", "yolov8m",
                                       "yolo11n", "yolo11s"],
                    help="download a generic YOLO base (to fine-tune later)")
    args = ap.parse_args()

    MODELS.mkdir(parents=True, exist_ok=True)
    print(SOURCES)

    if args.base:
        try:
            from ultralytics import YOLO
        except ImportError:
            print("Install ultralytics first: pip install ultralytics")
            return 1
        print(f"Downloading base model {args.base} (ultralytics cache)…")
        YOLO(f"{args.base}.pt")  # triggers download into the ultralytics cache
        print("Base model cached. Use it as the starting point in the train "
              "scripts (scripts/train_*.py). It is NOT a plate model yet.")
    else:
        print("Tip: run with e.g. --base yolov8s to fetch a base for training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
