import mediapipe as mp
import cv2


def crop_images():
    pass

def face_detection(image):
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_face_mesh = mp.solutions.face_mesh

    with mp_face_mesh.FaceMesh(max_num_faces=2,refine_landmarks=True,min_detection_confidence=0.5,min_tracking_confidence=0.5,) as face_mesh:
        results = face_mesh.process(image)

        # .......
        pass

def preprocess(image):
    pass