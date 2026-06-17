#!/usr/bin/env python
"""Run the full detect->read->decode pipeline on a single image file.

Great for validating your weights and the province/type decoding without a
phone or the GUI.

    python scripts/test_image.py path/to/car.jpg [--save out.jpg]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pelakyab.config import load_config
from pelakyab.detection import PlateDetector, CharRecognizer, classify_color
from pelakyab.data.plate_parser import parse_plate
from pelakyab.utils.draw import draw_plate_box


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--save", default=None)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    frame = cv2.imread(args.image)
    if frame is None:
        print(f"Could not read image: {args.image}")
        return 1

    det = cfg.detection
    pd = PlateDetector(det.plate_model, device=det.device, conf=det.plate_conf,
                       imgsz=det.plate_imgsz, half=det.half)
    cr = CharRecognizer(det.char_model, device=det.device, conf=det.char_conf,
                        imgsz=det.char_imgsz, half=det.half, min_chars=det.min_chars)

    plates = pd.detect(frame)
    print(f"Detected {len(plates)} plate region(s).")
    for i, p in enumerate(plates):
        chars = cr.recognize(p.crop)
        if chars is None:
            print(f"  [{i}] could not read characters")
            continue
        color, cconf = classify_color(p.crop)
        plate = parse_plate(chars.tokens, color=color,
                            confidence=(p.confidence + chars.mean_conf) / 2)
        print(f"  [{i}] {plate.display_en:18s} valid={plate.valid} "
              f"type={plate.plate_type} color={color}({cconf:.2f}) "
              f"prov={plate.province}/{plate.city} conf={plate.confidence:.2f}")
        print(f"       raw labels: {chars.raw_labels}")
        draw_plate_box(frame, p.bbox, plate.display_fa if plate.valid else "?",
                       plate.category, cfg.gui.persian_font, plate.confidence)

    if args.save:
        cv2.imwrite(args.save, frame)
        print(f"Annotated image saved to {args.save}")
    else:
        cv2.imshow("PelakYab test", frame)
        cv2.waitKey(0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
