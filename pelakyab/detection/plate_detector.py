"""Stage 1 — locate license plates in a frame with YOLO.

Wraps an Ultralytics YOLO model. Any YOLOv8/YOLO11 detection weights trained to
output a "plate" (or "license_plate") class work here. ``ultralytics`` is
imported lazily so the GUI can start and show a clear error if it's missing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class PlateDetection:
    bbox: tuple[int, int, int, int]   # x1, y1, x2, y2 in frame coords
    confidence: float
    crop: np.ndarray                  # the cropped plate image (BGR)

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]


class PlateDetector:
    def __init__(self, model_path: str, device: str = "cuda:0",
                 conf: float = 0.35, imgsz: int = 960, half: bool = True,
                 max_plates: int = 6, pad: float = 0.06):
        self.model_path = model_path
        self.device = device
        self.conf = conf
        self.imgsz = imgsz
        self.half = half and str(device).startswith("cuda")
        self.max_plates = max_plates
        self.pad = pad
        self._model = None

    def load(self) -> None:
        if self._model is not None:
            return
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "ultralytics is not installed. Run: pip install ultralytics"
            ) from exc
        self._model = YOLO(self.model_path)
        try:
            self._model.to(self.device)
        except Exception as exc:
            print(f"[PlateDetector] could not move model to {self.device}: {exc}")

    @property
    def ready(self) -> bool:
        return self._model is not None

    def detect(self, frame: np.ndarray) -> list[PlateDetection]:
        if self._model is None:
            self.load()
        h, w = frame.shape[:2]
        results = self._model.predict(
            frame, imgsz=self.imgsz, conf=self.conf, device=self.device,
            half=self.half, verbose=False,
        )
        out: list[PlateDetection] = []
        if not results:
            return out
        boxes = results[0].boxes
        if boxes is None or boxes.xyxy is None:
            return out

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        order = confs.argsort()[::-1][: self.max_plates]
        for i in order:
            x1, y1, x2, y2 = xyxy[i]
            # pad a little so we don't clip edge characters
            pw = (x2 - x1) * self.pad
            ph = (y2 - y1) * self.pad
            x1 = max(0, int(x1 - pw)); y1 = max(0, int(y1 - ph))
            x2 = min(w, int(x2 + pw)); y2 = min(h, int(y2 + ph))
            if x2 <= x1 or y2 <= y1:
                continue
            crop = frame[y1:y2, x1:x2].copy()
            out.append(PlateDetection((x1, y1, x2, y2), float(confs[i]), crop))
        return out
