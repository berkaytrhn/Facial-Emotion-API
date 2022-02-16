import os
import config
from utils import *
from predict import *
import argparse
from tensorflow.keras.models import load_model

def load_emotion_model():
    model_path = os.path.join("src","models",f"{config.model_name}") 
    model = load_model(model_path)
    return model 

def facial_emotion(image, model):    
    cropped, x_min, y_min, x_max, y_max = face_detection(image)
    cropped = preprocess(cropped)
    pred, prob = predict(cropped, model)
    
    return pred, prob, (x_min, y_min, x_max, y_max)
  


