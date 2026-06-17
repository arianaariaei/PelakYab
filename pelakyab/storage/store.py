"""Thread-safe JSON store for detected plates.

Layout on disk:
    data/plates.json          # dict keyed by normalized plate (e.g. "12ب345-68")
    data/images/<key>/<ts>_full.jpg
    data/images/<key>/<ts>_plate.jpg

De-duplication: the same plate seen again within ``dedup_cooldown`` seconds
updates the existing record (bumps ``last_seen`` / ``count``, appends a sighting
only after the cooldown) instead of creating a new car. A new car => a new key.
"""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from ..data.plate_parser import Plate


def _safe_key(key: str) -> str:
    """Filesystem-safe folder name for a plate key (Persian letters kept)."""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in key)


class PlateStore:
    def __init__(self, json_path: str, images_dir: str,
                 dedup_cooldown: float = 60.0,
                 max_sightings: int = 50,
                 save_full_frame: bool = True,
                 save_plate_crop: bool = True,
                 jpeg_quality: int = 90):
        self.json_path = Path(json_path)
        self.images_dir = Path(images_dir)
        self.dedup_cooldown = dedup_cooldown
        self.max_sightings = max_sightings
        self.save_full_frame = save_full_frame
        self.save_plate_crop = save_plate_crop
        self.jpeg_quality = int(jpeg_quality)

        self._lock = threading.RLock()
        self.json_path.parent.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self._records: dict[str, dict] = self._load()

    # ------------------------------------------------------------- load/save
    def _load(self) -> dict[str, dict]:
        if self.json_path.exists():
            try:
                with open(self.json_path, "r", encoding="utf-8") as fh:
                    return json.load(fh)
            except Exception as exc:
                print(f"[PlateStore] could not read {self.json_path}: {exc}")
        return {}

    def _flush(self) -> None:
        tmp = self.json_path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self._records, fh, ensure_ascii=False, indent=2)
        os.replace(tmp, self.json_path)   # atomic on the same filesystem

    # ---------------------------------------------------------------- query
    def all_records(self) -> list[dict]:
        with self._lock:
            return sorted(self._records.values(),
                          key=lambda r: r.get("last_seen", ""), reverse=True)

    def get(self, key: str) -> Optional[dict]:
        with self._lock:
            return self._records.get(key)

    def __len__(self) -> int:
        return len(self._records)

    # --------------------------------------------------------------- record
    def record(self, plate: Plate,
               full_frame: Optional[np.ndarray] = None,
               plate_crop: Optional[np.ndarray] = None) -> Tuple[dict, bool]:
        """Insert or update a plate. Returns (record, is_new_sighting_event)."""
        now = datetime.now()
        now_iso = now.isoformat(timespec="seconds")
        key = plate.key

        with self._lock:
            rec = self._records.get(key)
            is_new_car = rec is None

            if rec is None:
                rec = {
                    "key": key,
                    "plate_en": plate.display_en,
                    "plate_fa": plate.display_fa,
                    "left": plate.left,
                    "letter": plate.letter,
                    "serial": plate.serial,
                    "region_code": plate.region_code,
                    "type": plate.plate_type,
                    "type_fa": plate.plate_type_fa,
                    "color": plate.color,
                    "category": plate.category,
                    "province": plate.province,
                    "province_fa": plate.province_fa,
                    "city": plate.city,
                    "cities": plate.cities,
                    "ambiguous_region": plate.ambiguous_region,
                    "first_seen": now_iso,
                    "last_seen": now_iso,
                    "count": 0,
                    "best_confidence": 0.0,
                    "sightings": [],
                }
                self._records[key] = rec

            # cooldown: only log a *new sighting* if enough time has passed
            last_event = rec.get("_last_event_ts", 0.0)
            event_now = now.timestamp()
            is_event = is_new_car or (event_now - last_event) >= self.dedup_cooldown

            rec["last_seen"] = now_iso
            rec["best_confidence"] = round(
                max(rec.get("best_confidence", 0.0), plate.confidence), 3)

            if is_event:
                rec["count"] = rec.get("count", 0) + 1
                rec["_last_event_ts"] = event_now
                full_path = plate_path = None
                if full_frame is not None and self.save_full_frame:
                    full_path = self._save_image(key, now, "full", full_frame)
                if plate_crop is not None and self.save_plate_crop:
                    plate_path = self._save_image(key, now, "plate", plate_crop)
                sighting = {
                    "time": now_iso,
                    "confidence": round(plate.confidence, 3),
                    "image": full_path,
                    "plate_image": plate_path,
                }
                rec.setdefault("sightings", []).append(sighting)
                # cap history
                if len(rec["sightings"]) > self.max_sightings:
                    rec["sightings"] = rec["sightings"][-self.max_sightings:]
                self._flush()

            return rec, is_event

    # --------------------------------------------------------------- images
    def _save_image(self, key: str, ts: datetime, kind: str,
                    img: np.ndarray) -> Optional[str]:
        folder = self.images_dir / _safe_key(key)
        folder.mkdir(parents=True, exist_ok=True)
        fname = f"{ts.strftime('%Y%m%d_%H%M%S')}_{kind}.jpg"
        path = folder / fname
        try:
            cv2.imwrite(str(path), img,
                        [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        except Exception as exc:
            print(f"[PlateStore] failed to save image {path}: {exc}")
            return None
        # store path relative to the project for portability
        try:
            return str(path.relative_to(self.images_dir.parent.parent))
        except ValueError:
            return str(path)
