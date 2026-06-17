#!/usr/bin/env python
"""Train a small CNN single-character classifier (stage 2 of the recognizer).

Reads the 48x48 glyph crops from scripts/make_char_crops.py and trains a compact
conv net to classify the 31 plate glyphs. The live pipeline can then re-decide
the (noisy) LETTER with this classifier — a dedicated, centred-crop classifier is
sharper on the fine letter shapes than the shared YOLO detection head.

Saves models/char_classifier.pt = {state_dict, names, size}.

Usage:
    python scripts/train_char_classifier.py --epochs 20 --device cuda
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from pelakyab.data.plate_types import LETTER_TYPE  # noqa: E402


class CharCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),                                    # 48 -> 24
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),                                    # 24 -> 12
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),                                    # 12 -> 6
        )
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(128 * 6 * 6, 256), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))


class CropDS(Dataset):
    def __init__(self, X, y, train=False):
        self.X, self.y, self.train = X, y, train

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        img = self.X[i].astype(np.float32) / 255.0          # HWC BGR 0..1
        t = torch.from_numpy(img).permute(2, 0, 1)
        if self.train:
            if torch.rand(1).item() < 0.5:                  # brightness jitter
                t = (t * (0.8 + 0.4 * torch.rand(1).item())).clamp(0, 1)
            if torch.rand(1).item() < 0.3:                  # small translate
                t = torch.roll(t, shifts=(int(torch.randint(-3, 4, (1,))),
                                          int(torch.randint(-3, 4, (1,)))), dims=(1, 2))
        return t, int(self.y[i])


def load(split):
    d = np.load(ROOT / "datasets" / "ir-lpr" / "yolo" / f"char_crops_{split}.npz",
                allow_pickle=True)
    return d["X"], d["y"], [str(n) for n in d["names"]]


def evaluate(model, loader, device, names):
    model.eval()
    correct = total = 0
    lcorrect = ltotal = 0
    letter_ids = {i for i, n in enumerate(names) if n in LETTER_TYPE}
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            pred = model(x).argmax(1).cpu()
            correct += (pred == y).sum().item()
            total += len(y)
            for p, t in zip(pred.tolist(), y.tolist()):
                if t in letter_ids:
                    ltotal += 1
                    lcorrect += int(p == t)
    return correct / max(1, total), lcorrect / max(1, ltotal)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    Xtr, ytr, names = load("train")
    Xva, yva, _ = load("val")
    print(f"train {len(ytr)}  val {len(yva)}  classes {len(names)}  device {args.device}")

    tr = DataLoader(CropDS(Xtr, ytr, train=True), batch_size=args.batch,
                    shuffle=True, num_workers=0)
    va = DataLoader(CropDS(Xva, yva), batch_size=512, num_workers=0)

    device = torch.device(args.device)
    model = CharCNN(len(names)).to(device)
    # class-balanced loss (letters are rarer than digits)
    counts = np.bincount(ytr, minlength=len(names)).astype(np.float32)
    weight = torch.tensor(counts.sum() / (counts + 1e-6), dtype=torch.float32)
    weight = (weight / weight.mean()).to(device)
    crit = nn.CrossEntropyLoss(weight=weight)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    best = 0.0
    for ep in range(1, args.epochs + 1):
        model.train()
        for x, y in tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            crit(model(x), y).backward()
            opt.step()
        sched.step()
        acc, lacc = evaluate(model, va, device, names)
        print(f"epoch {ep:2d}  val_acc {acc:.4f}  letter_acc {lacc:.4f}")
        if acc > best:
            best = acc
            dst = ROOT / "models" / "char_classifier.pt"
            torch.save({"state_dict": model.state_dict(), "names": names, "size": 48}, dst)
    print(f"\nBest val acc {best:.4f} -> models/char_classifier.pt")

    Xte, yte, _ = load("test")
    te = DataLoader(CropDS(Xte, yte), batch_size=512, num_workers=0)
    model.load_state_dict(torch.load(ROOT / "models" / "char_classifier.pt")["state_dict"])
    acc, lacc = evaluate(model, te, device, names)
    print(f"TEST  overall {acc:.4f}  letters {lacc:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
