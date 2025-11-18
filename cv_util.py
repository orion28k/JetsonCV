import mediapipe as mp
import cv2

try:
    import tkinter as tk
except Exception:  # pragma: no cover - tkinter not always available
    tk = None

def get_screen_size(default=(1920, 1080)):
    """
    Determine the primary screen resolution.

    Args:
        default: Tuple[int, int] fallback resolution if tkinter is unavailable.

    Returns:
        Tuple[int, int]: Width and height of the primary display.
    """
    if tk is None:
        return default

    root = tk.Tk()
    root.withdraw()
    try:
        width = root.winfo_screenwidth()
        height = root.winfo_screenheight()
    finally:
        root.destroy()
    return width, height

# ---------- MediaPipe Hands ----------

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

def process_hands(img, hands, draw = False,hands_array = [None, None]):
    '''
    Args
    - Image
    - Medipipe Hands object
    - Draw hands?

    Return
    - Hand array [index 0 left hand or index 1 right hand][landmark] or None
    - If draw, draw 
    '''
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    hands_array = [None, None]

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            if draw:
                draw_hands(img, results.multi_hand_landmarks)

            label = handedness.classification[0].label.lower()  # "left" or "right"

            if label == "left":
                hands_array[0] = hand_landmarks
            elif label == "right":
                hands_array[1] = hand_landmarks

    return hands_array

def draw_hands(img, multi_hand_landmarks):
    mp_drawing = mp.solutions.drawing_utils

    for hand_landmarks in multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            img, # image
            hand_landmarks, # hand landmarks
            mp.solutions.hands.HAND_CONNECTIONS, # list of index pairs that define the connections
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3), # customize landmarks
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2) # customize lines
        )

# ---------- MediaPipe Body Pose ----------

def init_pose():
    """
    Initialize and return a MediaPipe Body Pose instance.
    main.py calls this once and reuses the object.
    """
    mp_pose = mp.solutions.pose

    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        smooth_segmentation=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    return pose

def process_pose(img, pose, draw=False):
    '''
    Args
    - Image (BGR)
    - MediaPipe Pose object
    - Draw pose?

    Return
    - pose_landmarks (NormalizedLandmarkList) or None
    '''
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    if results.pose_landmarks:
        if draw:
            draw_pose(img, results.pose_landmarks)
            
        return results.pose_landmarks

    return None

def draw_pose(img, pose_landmarks):
    """
    Draw the full body pose landmarks and connections onto the image.
    """
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    mp_drawing.draw_landmarks(
        img,
        pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=3),
        mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2),
    )


# ---------- MediaPipe Face Mesh (Full Facial Landmarks) ----------

def init_face():
    """
    Initialize and return a MediaPipe Face Mesh instance.
    main.py calls this once and reuses the object.
    """
    mp_face_mesh = mp.solutions.face_mesh

    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,  # includes iris landmarks
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    return face_mesh

def process_face(img, face_mesh, draw=False):
    """
    Run MediaPipe Face Mesh on the given frame and optionally draw the mesh.

    Args:
        img: BGR image (numpy array).
        face_mesh: A MediaPipe FaceMesh instance created by init_face().
        draw: If True, draw the face mesh on the image.

    Returns:
        multi_face_landmarks: list of NormalizedLandmarkList (one per face), or None.
    """
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)

    if results.multi_face_landmarks:
        if draw:
            draw_face(img, results.multi_face_landmarks)
        return results.multi_face_landmarks

    return None

def draw_face(img, multi_face_landmarks):
    """
    Draw face mesh landmarks and connections onto the image.
    """
    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles
    mp_face_mesh = mp.solutions.face_mesh

    for face_landmarks in multi_face_landmarks:
        # Tesselation (dense mesh)
        mp_drawing.draw_landmarks(
            image=img,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style(),
        )
        # Contours (eyes, lips, face outline)
        mp_drawing.draw_landmarks(
            image=img,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_styles.get_default_face_mesh_contours_style(),
        )

# ---------- MediaPipe Holistic ----------

def init_holistic():
    """
    Initialize and return a MediaPipe Holistic instance.
    This detects pose, hands, and face landmarks simultaneously.
    """
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        smooth_segmentation=True,
        refine_face_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    return holistic

def process_holistic(img, holistic, draw=False):
    """Run MediaPipe Holistic on the frame and optionally draw results.

    Args:
        img: BGR image (numpy array).
        holistic: A MediaPipe Holistic instance created by init_holistic().
        draw: If True, draw pose + hand (and optionally face) landmarks.

    Returns:
        results: The MediaPipe Holistic results object, which exposes:
            - results.pose_landmarks
            - results.left_hand_landmarks
            - results.right_hand_landmarks
            - results.face_landmarks
    """
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = holistic.process(img_rgb)

    if draw:
        draw_holistic(img, results)
    if results:
        return results
    else:
        return None


def draw_holistic(img, results):
    """
    Draw all available Holistic landmarks onto the provided image.

    Args:
        img: BGR image (numpy array) that will be drawn on in-place.
        results: MediaPipe Holistic results from process_holistic(), containing pose,
            left/right hand, and face landmarks (each may be None).
    """
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic

    # Draw body pose
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            img,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=3),
            mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2),
        )

    # Draw left hand
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            img,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2),
        )

    # Draw right hand
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            img,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2),
        )

    # Draw face landmarks (contours only, to keep it simple)
    if results.face_landmarks:
        mp_drawing.draw_landmarks(
            img,
            results.face_landmarks,
            mp_holistic.FACEMESH_CONTOURS,
            mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=1, circle_radius=1),
            mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=1),
        )

def init_detection_obj(mode):
    if mode == "holistic":
        # Multiple detections requested -> use Holistic
        print("[INFO]: Using Mediapipe Holistic detection (hands + pose + face).")
        return init_holistic()
    elif mode == "hands":
        print("[INFO]: Using Mediapipe Hand detection.")
        return init_hands()
    elif mode == "pose":
        print("[INFO]: Using Mediapipe Body Pose detection.")
        return init_pose()
    elif mode == "face":
        print("[INFO]: Using Mediapipe Facial Feature detection.")
        return init_face()
    else:
        print("[INFO]: Not using Mediapipe detection")
        return None
