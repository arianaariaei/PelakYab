"""Stage 2 — read the characters inside a plate crop with YOLO.

Iranian plates have a fixed 8-glyph layout, so character *detection* (one box
per glyph) + left-to-right sorting is both accurate and robust. This wraps an
Ultralytics YOLO model whose classes are the 28 plate glyphs (10 digits + 18
letters). Class labels are normalized to canonical Persian glyphs/digits via
``data.plate_types.normalize_token`` so it works regardless of how the weights
name their classes.

If your weights instead label classes as plain digits/letters that don't need
mapping, normalization is a no-op and everything still works.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..data.plate_types import normalize_token, LETTER_TYPE


@dataclass
class CharResult:
    tokens: list[str] = field(default_factory=list)      # canonical, L->R
    confidences: list[float] = field(default_factory=list)
    mean_conf: float = 0.0
    raw_labels: list[str] = field(default_factory=list)  # what the model emitted

    @property
    def text(self) -> str:
        return "".join(self.tokens)


class CharRecognizer:
    def __init__(self, model_path: str, device: str = "cuda:0",
                 conf: float = 0.30, imgsz: int = 320, half: bool = True,
                 min_chars: int = 7, two_row: bool = False,
                 classifier_path: Optional[str] = None,
                 two_stage_letter: bool = False,
                 two_stage_min_conf: float = 0.5):
        self.model_path = model_path
        self.device = device
        self.conf = conf
        self.imgsz = imgsz
        self.half = half and str(device).startswith("cuda")
        self.min_chars = min_chars
        self.two_row = two_row     # set True for motorcycle/2-row plates
        # stage-2 letter re-scorer (optional)
        self.classifier_path = classifier_path
        self.two_stage_letter = two_stage_letter
        self.two_stage_min_conf = two_stage_min_conf
        self._model = None
        self._classifier = None
        self._names: dict[int, str] = {}

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
            print(f"[CharRecognizer] could not move model to {self.device}: {exc}")
        self._names = dict(self._model.names)

        if self.two_stage_letter and self.classifier_path and self._classifier is None:
            import os
            if os.path.exists(self.classifier_path):
                try:
                    from .char_classifier import CharClassifier
                    self._classifier = CharClassifier(self.classifier_path, self.device)
                    print("[CharRecognizer] stage-2 letter classifier loaded")
                except Exception as exc:
                    print(f"[CharRecognizer] could not load letter classifier: {exc}")
            else:
                print(f"[CharRecognizer] classifier not found: {self.classifier_path}")

    @property
    def ready(self) -> bool:
        return self._model is not None

    def recognize(self, plate_crop: np.ndarray) -> Optional[CharResult]:
        if self._model is None:
            self.load()
        if plate_crop is None or plate_crop.size == 0:
            return None

        results = self._model.predict(
            plate_crop, imgsz=self.imgsz, conf=self.conf, device=self.device,
            half=self.half, verbose=False,
        )
        if not results:
            return None
        boxes = results[0].boxes
        if boxes is None or boxes.xyxy is None or len(boxes) == 0:
            return None

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)

        # Build per-character records with center coords for ordering.
        chars = []
        for (x1, y1, x2, y2), cf, cl in zip(xyxy, confs, clss):
            label = self._names.get(int(cl), str(int(cl)))
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            chars.append({"label": label, "conf": float(cf), "cx": cx, "cy": cy,
                          "h": float(y2 - y1), "box": (x1, y1, x2, y2)})

        chars = self._order(chars)
        if not chars:
            return None

        raw_labels, tokens, confidences, boxes_kept = [], [], [], []
        for c in chars:
            raw_labels.append(c["label"])
            norm = normalize_token(c["label"])
            if norm is None:
                continue
            tokens.append(norm)
            confidences.append(c["conf"])
            boxes_kept.append(c["box"])

        if len(tokens) < self.min_chars:
            return None

        # stage 2: re-decide the LETTER glyph with the dedicated CNN classifier
        if self._classifier is not None:
            li = (2 if len(tokens) == 8 else
                  next((i for i, t in enumerate(tokens) if t in LETTER_TYPE), None))
            if li is not None and li < len(boxes_kept):
                x1, y1, x2, y2 = boxes_kept[li]
                crop = plate_crop[int(y1):int(y2), int(x1):int(x2)]
                letter, lconf = self._classifier.classify_letter(crop)
                yolo_conf = confidences[li] if li < len(confidences) else 0.0
                # only override when the CNN is confident AND more sure than YOLO,
                # so confident-correct YOLO letters are left untouched.
                if (letter and lconf >= self.two_stage_min_conf
                        and lconf > yolo_conf + 0.05):
                    tokens[li] = letter

        tokens, confidences = self._structure_decode(tokens, confidences)
        mean_conf = float(np.mean(confidences)) if confidences else 0.0
        return CharResult(tokens=tokens, confidences=confidences,
                          mean_conf=mean_conf, raw_labels=raw_labels)

    # -------------------------------------------------------------- helpers
    def _order(self, chars: list[dict]) -> list[dict]:
        """Sort glyphs left-to-right (and top row first for 2-row plates)."""
        if not chars:
            return chars
        if self.two_row:
            ys = np.array([c["cy"] for c in chars])
            mid = (ys.min() + ys.max()) / 2.0
            top = sorted([c for c in chars if c["cy"] <= mid], key=lambda c: c["cx"])
            bot = sorted([c for c in chars if c["cy"] > mid], key=lambda c: c["cx"])
            return top + bot
        return sorted(chars, key=lambda c: c["cx"])

    @staticmethod
    def _structure_decode(tokens: list[str],
                          confs: list[float]) -> tuple[list[str], list[float]]:
        """Clean a raw left-to-right read against the rigid Iranian car-plate
        layout: 2 digits, 1 letter, 3 digits, 2 region digits = 8 glyphs.

        Conservative on purpose — it only repairs clear *structural* violations,
        so a clean 8-glyph read is never altered:
          1. exactly one letter: drop spurious extra letters, keeping the most
             confident one (better than blindly keeping the first);
          2. over-detection: if more than 8 glyphs, drop the lowest-confidence
             DIGITS (never the letter) down to 8 — fixes split/duplicate boxes.

        Same-type confusions (2<->3, س<->ص) are visual, not structural, and are
        deliberately left untouched.
        """
        EXPECTED = 8
        drop: set[int] = set()

        letter_idx = [i for i, t in enumerate(tokens) if t in LETTER_TYPE]
        if len(letter_idx) > 1:
            keep = max(letter_idx, key=lambda i: confs[i])
            drop |= set(letter_idx) - {keep}

        kept = [i for i in range(len(tokens)) if i not in drop]
        if len(kept) > EXPECTED:
            digits = [i for i in kept if tokens[i].isdigit()]
            n_extra = len(kept) - EXPECTED
            drop |= set(sorted(digits, key=lambda i: confs[i])[:n_extra])

        if not drop:
            return tokens, confs
        tokens = [t for i, t in enumerate(tokens) if i not in drop]
        confs = [c for i, c in enumerate(confs) if i not in drop]
        return tokens, confs
