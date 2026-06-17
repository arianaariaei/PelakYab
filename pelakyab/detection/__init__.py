"""Detection & recognition: plate detector, char recognizer, color classifier."""
from .plate_detector import PlateDetector, PlateDetection
from .char_recognizer import CharRecognizer, CharResult
from .color_classifier import classify_color

__all__ = [
    "PlateDetector", "PlateDetection",
    "CharRecognizer", "CharResult",
    "classify_color",
]
