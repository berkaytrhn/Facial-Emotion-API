import mediapipe as mp
import cv2
import config
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np

def face_detection(image):
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_face_mesh = mp.solutions.face_mesh

    with mp_face_mesh.FaceMesh(max_num_faces=config.max_number_of_faces,refine_landmarks=True,min_detection_confidence=0.5,min_tracking_confidence=0.5,) as face_mesh:
        result = face_mesh.process(image)

        # cropping part
        for landmarks in result.multi_face_landmarks:
            height, width, _ = image.shape
            x_min = width
            y_min = height
            x_max = y_max = 0
            for id, lm in enumerate(landmarks.landmark):
                cx, cy = int(lm.x * width), int(lm.y * height)
                if cx < x_min:
                    x_min = cx
                if cy < y_min:
                    y_min = cy
                if cx > x_max:
                    x_max = cx
                if cy > y_max:
                    y_max = cy
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 255, 0), 2)
            cropped = image[y_min:y_max, x_min:x_max].copy()
       
        return cropped, x_min, y_min, x_max, y_max

def preprocess(cropped):
    cropped = cv2.resize(cropped, (config.image_size, config.image_size))
    cropped = img_to_array(cropped)
    cropped = np.expand_dims(cropped, axis=0)
    return cropped