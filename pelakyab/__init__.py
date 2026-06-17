"""PelakYab — real-time Iranian (Persian) license-plate recognition.

Pipeline:  phone camera (IP Webcam)  ->  ambient-light / auto-flash  ->
           YOLO plate detector  ->  YOLO character recognizer  ->
           color + letter + region decoding  ->  JSON store  ->  Qt GUI.
"""

__version__ = "0.1.0"
__all__ = ["__version__"]
