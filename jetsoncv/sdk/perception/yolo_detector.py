"""YOLODetector stub with graceful degradation.

If real YOLO dependencies and weights are available, replace the stub
implementation with a proper model loader and inference wrapper.

# TODO: Integrate real YOLO model loading when dependencies and weights are available.
"""
from typing import List
from ..core.types import Detection
import warnings


class YOLODetector:
    """Stub YOLO detector. Returns empty detections and logs a warning.

    The class tries to import typical YOLO libraries but will fall back to a
    no-op implementation if not available.
    """

    def __init__(self, model_path: str = None):
        # Try to import heavy dependencies lazily
        self._available = False
        try:
            import torch  # type: ignore
            self._available = True
        except Exception:
            warnings.warn("YOLO dependencies not available; using stub detector")

    def detect(self, frame) -> List[Detection]:
        """Return list of Detection; empty list when no model present."""
        if not self._available:
            return []
        # Real implementation goes here once dependencies are added.
        return []
