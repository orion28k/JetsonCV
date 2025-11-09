import mediapipe as mp

# ---------- MediaPipe Hands setup ----------

def init_hands():
    """
    Initialize and return a MediaPipe Hands instance.
    main.py calls this once and reuses the object.
    """
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return hands