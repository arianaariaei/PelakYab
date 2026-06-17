"""Stage-2 single-glyph CNN classifier (see scripts/train_char_classifier.py).

Used to re-decide the noisy LETTER glyph that the YOLO char detector emits: a
dedicated classifier on a centred crop is sharper on the fine letter shapes.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ..data.plate_types import LETTER_TYPE


class CharClassifier:
    """Loads models/char_classifier.pt and classifies a BGR glyph crop."""

    def __init__(self, weights_path: str, device: str = "cuda:0"):
        import torch
        import torch.nn as nn

        class CharCNN(nn.Module):                      # must match the trainer
            def __init__(self, num_classes: int):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
                    nn.MaxPool2d(2),
                )
                self.head = nn.Sequential(
                    nn.Flatten(), nn.Linear(128 * 6 * 6, 256), nn.ReLU(),
                    nn.Dropout(0.3), nn.Linear(256, num_classes),
                )

            def forward(self, x):
                return self.head(self.features(x))

        self._torch = torch
        self.device = device
        ckpt = torch.load(weights_path, map_location=device)
        self.names = list(ckpt["names"])
        self.size = int(ckpt.get("size", 48))
        self.model = CharCNN(len(self.names))
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(device).eval()
        self._letter_idx = [i for i, n in enumerate(self.names) if n in LETTER_TYPE]

    def _logits(self, bgr_crop: np.ndarray):
        import cv2
        g = cv2.resize(bgr_crop, (self.size, self.size),
                       interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        t = self._torch.from_numpy(g).permute(2, 0, 1).unsqueeze(0).to(self.device)
        with self._torch.no_grad():
            return self.model(t).softmax(1)[0]

    def classify_letter(self, bgr_crop: np.ndarray) -> Tuple[Optional[str], float]:
        """Best LETTER for this crop (restricted to letter classes), + prob."""
        if bgr_crop is None or bgr_crop.size == 0 or not self._letter_idx:
            return None, 0.0
        probs = self._logits(bgr_crop)
        best_i, best_p = None, -1.0
        for i in self._letter_idx:
            p = float(probs[i])
            if p > best_p:
                best_i, best_p = i, p
        return (self.names[best_i] if best_i is not None else None), best_p
