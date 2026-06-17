"""The end-to-end pipeline: frame -> detect -> read -> vote -> decode
-> store -> annotated result.

The GUI runs ``Pipeline.process_once`` in a worker thread; a headless CLI can
call ``Pipeline.run``. Construction is cheap; heavy model loading happens on
first detect (or via ``warmup``).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .config import Config
from .camera import CameraStream, BrowserCameraStream
from .detection import PlateDetector, CharRecognizer, classify_color
from .data.plate_parser import parse_plate, Plate
from .storage import PlateStore
from .stabilizer import PlateStabilizer
from .utils.draw import draw_plate_box
from .utils.preprocess import preprocess_plate


@dataclass
class PlateEvent:
    plate: Plate
    record: dict
    is_new_event: bool
    plate_crop: np.ndarray


@dataclass
class FrameResult:
    frame: np.ndarray                       # annotated BGR frame
    events: list[PlateEvent] = field(default_factory=list)
    n_plates: int = 0
    infer_ms: float = 0.0


class Pipeline:
    def __init__(self, cfg: Config):
        self.cfg = cfg

        cam = cfg.camera
        if str(cam.source).strip().lower() == "browser":
            # App-free phone camera: PC serves a QR-linked HTTPS page that
            # streams the phone's camera over WebSocket (see camera.browser_cam).
            if BrowserCameraStream is None:
                raise RuntimeError(
                    "camera.source='browser' needs aiohttp, qrcode and "
                    "cryptography. Install them: pip install -r requirements.txt")
            browser = getattr(cam, "browser", None)
            self.camera = BrowserCameraStream(
                port=getattr(browser, "port", 8443),
                rotate=getattr(cam, "rotate", 0),
                flip=getattr(cam, "flip", ""),
                host_ip=getattr(browser, "host_ip", None),
                tunnel=getattr(browser, "tunnel", False),
                tunnel_provider=getattr(browser, "tunnel_provider", "auto"),
                relay_url=getattr(browser, "relay_url", None),
            )
        else:
            self.camera = CameraStream(
                source=cam.source,
                stream_path=cam.stream_path,
                snapshot_path=cam.snapshot_path,
                torch_on_path=cam.torch_on_path,
                torch_off_path=cam.torch_off_path,
                on_connect_requests=getattr(cam, "on_connect_requests", []),
                reconnect_delay=cam.reconnect_delay,
                read_timeout=cam.read_timeout,
                rotate=getattr(cam, "rotate", 0),
                flip=getattr(cam, "flip", ""),
            )


        det = cfg.detection
        self.pp_clahe = getattr(det, "preprocess_clahe", False)
        self.pp_deskew = getattr(det, "preprocess_deskew", False)
        self.plate_detector = PlateDetector(
            det.plate_model, device=det.device, conf=det.plate_conf,
            imgsz=det.plate_imgsz, half=det.half, max_plates=det.max_plates_per_frame,
        )
        self.char_recognizer = CharRecognizer(
            det.char_model, device=det.device, conf=det.char_conf,
            imgsz=det.char_imgsz, half=det.half, min_chars=det.min_chars,
            classifier_path=getattr(det, "char_classifier", None),
            two_stage_letter=getattr(det, "two_stage_letter", False),
        )

        st = cfg.storage
        self.store = PlateStore(
            json_path=st.json_path, images_dir=st.images_dir,
            dedup_cooldown=st.dedup_cooldown, max_sightings=st.max_sightings_per_plate,
            save_full_frame=st.save_full_frame, save_plate_crop=st.save_plate_crop,
            jpeg_quality=st.jpeg_quality,
        )

        rc = cfg.recognition
        self.stabilizer = PlateStabilizer(
            window=getattr(rc, "vote_window", 2.0),
            min_votes=getattr(rc, "min_votes", 4),
            emit_interval=getattr(rc, "reemit_interval", 2.0),
            retain=getattr(rc, "track_retain", 20.0),
        ) if getattr(rc, "stabilize", True) else None

        self.font_path = cfg.gui.persian_font
        self._stop = False

    # ------------------------------------------------------------- lifecycle
    def start_camera(self) -> None:
        self.camera.start()

    def warmup(self) -> None:
        """Load both models up front (so the first real frame isn't slow)."""
        self.plate_detector.load()
        self.char_recognizer.load()

    def stop(self) -> None:
        self._stop = True
        self.camera.stop()

    # ---------------------------------------------------------------- core
    def process_once(self, draw: bool = True) -> Optional[FrameResult]:
        ok, frame = self.camera.read()
        if not ok or frame is None:
            return None
        return self.process_frame(frame, draw=draw)

    def process_frame(self, frame: np.ndarray, draw: bool = True) -> FrameResult:
        t0 = time.perf_counter()

        # detect plates
        detections = self.plate_detector.detect(frame)

        annotated = frame.copy() if draw else frame
        events: list[PlateEvent] = []

        # 3) read every detected plate this frame (live, per-frame)
        reads = []
        for det in detections:
            rec_crop = det.crop
            if self.pp_clahe or self.pp_deskew:
                rec_crop = preprocess_plate(det.crop, do_deskew=self.pp_deskew,
                                            do_clahe=self.pp_clahe)
            chars = self.char_recognizer.recognize(rec_crop)
            if chars is None:
                continue
            color, _ = classify_color(det.crop)
            live = parse_plate(
                chars.tokens, color=color,
                confidence=(det.confidence + chars.mean_conf) / 2.0,
                expected=self.cfg.recognition.expected_chars,
            )
            if draw:
                # live feedback box (the raw per-frame read); ASCII label keeps
                # physical L-to-R order with no fragile bidi shaping.
                draw_plate_box(annotated, det.bbox, live.display_en or "?",
                               live.category if live.valid else "unknown",
                               self.font_path,
                               live.confidence if live.valid else det.confidence)
            x1, y1, x2, y2 = det.bbox
            reads.append({"cx": (x1 + x2) / 2.0, "cy": (y1 + y2) / 2.0,
                          "tokens": chars.tokens, "confs": chars.confidences,
                          "color": color, "crop": det.crop,
                          "conf": live.confidence})

        # 4) vote across frames -> only store STABLE plates (one per car), which
        # also corrects most per-frame letter/digit jitter.
        if self.stabilizer is not None:
            confirmed = self.stabilizer.update(reads, frame.shape)
        else:
            confirmed = [{"tokens": r["tokens"], "color": r["color"],
                          "crop": r["crop"], "conf": r["conf"]} for r in reads]

        for c in confirmed:
            plate = parse_plate(c["tokens"], color=c["color"], confidence=c["conf"],
                                expected=self.cfg.recognition.expected_chars)
            if not plate.valid:
                continue
            record, is_event = self.store.record(plate, frame, c["crop"])
            events.append(PlateEvent(plate, record, is_event, c["crop"]))

        infer_ms = (time.perf_counter() - t0) * 1000.0
        return FrameResult(
            frame=annotated,
            events=events,
            n_plates=len(detections),
            infer_ms=infer_ms,
        )

    # ------------------------------------------------------------- headless
    def run(self, on_event=None, on_frame=None) -> None:
        """Blocking headless loop (no GUI). Ctrl+C to stop."""
        self.start_camera()
        self.warmup()
        try:
            while not self._stop:
                result = self.process_once(draw=False)
                if result is None:
                    time.sleep(0.05)
                    continue
                for ev in result.events:
                    if ev.is_new_event and on_event:
                        on_event(ev)
                if on_frame:
                    on_frame(result)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()
