import tensorflow as tf
import numpy as np
import config


def predict(cropped, model):
    result = model.predict(cropped)
    prob = round(np.max(result),2)
    prediction = config.emotions[result.argmax()]
    return prediction, prob
