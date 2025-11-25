"""Haar cascade face detector.

This module provides a lightweight face detector using OpenCV's Haar cascade
classifier. It is always available and does not require heavy ML libraries.
"""
from typing import List
import os
import cv2

from ..core.types import Detection


class HaarFaceDetector:
    """Simple face detector using Haar cascades.

    Always available; no heavy dependencies.
    """

    def __init__(self, cascade_path: str = None):
        if cascade_path is None:
            # Try to find bundled cascade
            base = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            cascade_path = os.path.join(base, "Follow-me-DJI-Tello", "haarcascades", "haarcascade_frontalface_default.xml")
            if not os.path.exists(cascade_path):
                # Fallback to OpenCV's installed cascade
                cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        
        self.cascade = cv2.CascadeClassifier(cascade_path)
        self.scale_factor = 1.2
        self.min_neighbors = 8

    def detect(self, frame) -> List[Detection]:
        """Return list of Detection objects from Haar cascade."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors
        )

        detections = []
        for (x, y, w, h) in faces:
            bbox = (int(x), int(y), int(x + w), int(y + h))
            detections.append(Detection(bbox=bbox, label="face", confidence=0.9))

        return detections

    def preprocess(self, frame) -> np.ndarray:
        """Return the image as the model sees it (grayscale, histogram-equalized).

        Useful for displaying the exact input fed to the Haar cascade.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        try:
            gray = cv2.equalizeHist(gray)
        except Exception:
            pass
        return gray
