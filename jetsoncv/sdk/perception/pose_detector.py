"""MediaPipe pose detector.

This module provides pose detection using MediaPipe, which tracks full-body
skeleton keypoints (17 landmarks). Always available if mediapipe is installed.
"""
from typing import List, Optional, Tuple
import mediapipe as mp
import cv2
import numpy as np

from ..core.types import Detection


class MediaPipePoseDetector:
    """Pose detector using MediaPipe Pose.

    Detects full-body pose with 17 keypoints (nose, shoulders, elbows, wrists,
    hips, knees, ankles, etc.).
    """

    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def detect(self, frame) -> Tuple[Optional[List[Tuple[float, float]]], Optional[float]]:
        """Detect pose landmarks.
        
        Returns:
            (landmarks, confidence) where landmarks is a list of (x, y) normalized coords
            and confidence is overall detection confidence, or (None, None) if no pose detected.
        """
        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)

        if not results.pose_landmarks:
            return None, None

        # Extract landmarks as normalized (0-1) coordinates
        landmarks = []
        for lm in results.pose_landmarks.landmark:
            landmarks.append((float(lm.x), float(lm.y)))

        # Average confidence across all landmarks
        confidence = np.mean([lm.visibility for lm in results.pose_landmarks.landmark])

        return landmarks, float(confidence)

    def preprocess(self, frame) -> np.ndarray:
        """Return the image as the model sees it (RGB).

        This is useful for displaying the exact input fed to MediaPipe.
        The returned image is an RGB numpy array with the same shape as input.
        """
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def get_torso_center(self, landmarks: List[Tuple[float, float]], frame_shape) -> Tuple[float, float]:
        """Compute torso center from shoulder and hip landmarks.
        
        Uses landmarks 11, 12 (shoulders) and 23, 24 (hips).
        Returns pixel coordinates (x, y).
        """
        if landmarks is None or len(landmarks) < 25:
            return None, None

        h, w = frame_shape[:2]

        # Shoulder midpoint: (11 + 12) / 2
        shoulder_x = (landmarks[11][0] + landmarks[12][0]) / 2.0
        shoulder_y = (landmarks[11][1] + landmarks[12][1]) / 2.0

        # Hip midpoint: (23 + 24) / 2
        hip_x = (landmarks[23][0] + landmarks[24][0]) / 2.0
        hip_y = (landmarks[23][1] + landmarks[24][1]) / 2.0

        # Torso center is midpoint of shoulder and hip midpoints
        torso_x = ((shoulder_x + hip_x) / 2.0) * w
        torso_y = ((shoulder_y + hip_y) / 2.0) * h

        return torso_x, torso_y

    def draw_landmarks(self, frame, landmarks):
        """Draw pose skeleton on frame."""
        if landmarks is None:
            return

        h, w = frame.shape[:2]

        # Draw connections (skeleton)
        connections = [
            (11, 12),  # shoulders
            (11, 13), (13, 15),  # left arm
            (12, 14), (14, 16),  # right arm
            (11, 23), (12, 24),  # torso to hips
            (23, 24),  # hip line
            (23, 25), (25, 27),  # left leg
            (24, 26), (26, 28),  # right leg
        ]

        for start, end in connections:
            if start < len(landmarks) and end < len(landmarks):
                x1 = int(landmarks[start][0] * w)
                y1 = int(landmarks[start][1] * h)
                x2 = int(landmarks[end][0] * w)
                y2 = int(landmarks[end][1] * h)
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw keypoints
        for i, (x_norm, y_norm) in enumerate(landmarks):
            x = int(x_norm * w)
            y = int(y_norm * h)
            cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)
