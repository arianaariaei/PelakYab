"""Camera input: Android IP Webcam stream, or app-free browser camera (QR)."""
from .ip_webcam import CameraStream

# BrowserCameraStream pulls in aiohttp/cryptography/qrcode; import lazily so the
# package still imports if those optional deps are missing.
try:
    from .browser_cam import BrowserCameraStream
except Exception:  # pragma: no cover
    BrowserCameraStream = None  # type: ignore

__all__ = ["CameraStream", "BrowserCameraStream"]
