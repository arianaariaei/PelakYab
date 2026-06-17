"""Camera stream abstraction.

Primary target: the Android **IP Webcam** app, which exposes
  - an MJPEG video stream at  <base>/video
  - single JPEG snapshots at   <base>/shot.jpg
  - torch control at           <base>/enabletorch  /  <base>/disabletorch
  - many tunables at           <base>/settings/<name>?set=<value>

A plain integer source ("0", "1", ...) opens a local webcam instead, so you can
develop without a phone (torch control becomes a no-op in that case).
"""
from __future__ import annotations

import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np

try:
    import requests
except Exception:  # pragma: no cover
    requests = None  # type: ignore


class CameraStream:
    """Threaded reader that always hands you the most recent frame."""

    def __init__(self, source: str,
                 stream_path: str = "/video",
                 snapshot_path: str = "/shot.jpg",
                 torch_on_path: str = "/enabletorch",
                 torch_off_path: str = "/disabletorch",
                 on_connect_requests: Optional[list[str]] = None,
                 reconnect_delay: float = 2.0,
                 read_timeout: float = 5.0,
                 rotate: int = 0,
                 flip: str = ""):
        self.rotate = self._norm_rotate(rotate)
        self.flip = flip if flip in ("h", "v", "hv") else ""
        self.source = str(source).strip()
        self.is_local = self.source.isdigit()
        self.base = self.source.rstrip("/") if not self.is_local else None

        self.stream_url = (
            int(self.source) if self.is_local else f"{self.base}{stream_path}"
        )
        self.snapshot_url = None if self.is_local else f"{self.base}{snapshot_path}"
        self.torch_on_url = None if self.is_local else f"{self.base}{torch_on_path}"
        self.torch_off_url = None if self.is_local else f"{self.base}{torch_off_path}"
        self.on_connect_requests = on_connect_requests or []
        self.reconnect_delay = reconnect_delay
        self.read_timeout = read_timeout

        self._cap: Optional[cv2.VideoCapture] = None
        self._frame: Optional[np.ndarray] = None
        self._frame_ts: float = 0.0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._connected = False

    # ----------------------------------------------------------------- open
    def _open_capture(self) -> bool:
        cap = cv2.VideoCapture(self.stream_url)
        # keep latency low for live processing
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        if cap.isOpened():
            self._cap = cap
            self._connected = True
            self._run_connect_requests()
            return True
        cap.release()
        return False

    def _run_connect_requests(self) -> None:
        if self.is_local or requests is None:
            return
        for path in self.on_connect_requests:
            url = path if path.startswith("http") else f"{self.base}{path}"
            try:
                requests.get(url, timeout=self.read_timeout)
            except Exception as exc:
                print(f"[CameraStream] on_connect request failed ({url}): {exc}")

    # --------------------------------------------------------------- thread
    def start(self) -> "CameraStream":
        if self._thread and self._thread.is_alive():
            return self
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="CameraStream")
        self._thread.start()
        return self

    def _loop(self) -> None:
        while not self._stop.is_set():
            if self._cap is None or not self._cap.isOpened():
                if not self._open_capture():
                    self._connected = False
                    time.sleep(self.reconnect_delay)
                    continue
            ok, frame = self._cap.read()
            if not ok or frame is None:
                # stream hiccup -> drop and reconnect
                self._connected = False
                try:
                    self._cap.release()
                except Exception:
                    pass
                self._cap = None
                # try a snapshot as a fallback so we still get *something*
                snap = self._grab_snapshot()
                if snap is not None:
                    self._store(snap)
                time.sleep(self.reconnect_delay)
                continue
            self._store(frame)

    def _store(self, frame: np.ndarray) -> None:
        frame = self._orient(frame)
        with self._lock:
            self._frame = frame
            self._frame_ts = time.monotonic()
            self._connected = True

    # -------------------------------------------------------- orientation
    @staticmethod
    def _norm_rotate(deg) -> int:
        try:
            deg = int(deg) % 360
        except Exception:
            return 0
        return deg if deg in (0, 90, 180, 270) else (round(deg / 90) * 90) % 360

    def set_rotation(self, deg: int) -> None:
        """Live-set the rotation applied to incoming frames (0/90/180/270)."""
        self.rotate = self._norm_rotate(deg)

    def rotate_by(self, delta: int) -> int:
        """Rotate by +/-90 and return the new absolute rotation."""
        self.rotate = self._norm_rotate(self.rotate + delta)
        return self.rotate

    def _orient(self, frame: np.ndarray) -> np.ndarray:
        if self.rotate == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif self.rotate == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif self.rotate == 270:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        if self.flip == "h":
            frame = cv2.flip(frame, 1)
        elif self.flip == "v":
            frame = cv2.flip(frame, 0)
        elif self.flip == "hv":
            frame = cv2.flip(frame, -1)
        return frame

    def _grab_snapshot(self) -> Optional[np.ndarray]:
        if self.is_local or requests is None or not self.snapshot_url:
            return None
        try:
            resp = requests.get(self.snapshot_url, timeout=self.read_timeout)
            if resp.status_code == 200:
                arr = np.frombuffer(resp.content, dtype=np.uint8)
                return cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception:
            return None
        return None

    # ----------------------------------------------------------------- read
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Return (ok, latest_frame_copy)."""
        with self._lock:
            if self._frame is None:
                return False, None
            return True, self._frame.copy()

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def frame_age(self) -> float:
        with self._lock:
            if self._frame_ts == 0:
                return float("inf")
            return time.monotonic() - self._frame_ts

    # ---------------------------------------------------------------- torch
    def set_torch(self, on: bool) -> bool:
        """Turn the phone torch on/off. No-op for local webcams."""
        if self.is_local or requests is None:
            return False
        url = self.torch_on_url if on else self.torch_off_url
        if not url:
            return False
        resp = requests.get(url, timeout=self.read_timeout)
        return resp.status_code == 200

    # ---------------------------------------------------------------- close
    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
        self._connected = False

    def __enter__(self) -> "CameraStream":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()
