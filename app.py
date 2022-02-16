from ast import Bytes
from fastapi import FastAPI, UploadFile, File, Form
from typing import Optional
import uvicorn
import sys
from PIL import Image
from io import BytesIO
import numpy as np
import cv2
from fastapi.middleware.cors import CORSMiddleware
import sys
sys.path.append("./src/")
from src.main import *
from fastapi.responses import JSONResponse
import argparse



model = load_emotion_model()

application = FastAPI()


application.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]    
)


@application.get("/test/")
async def test():
    return {"Server": "Running!!!"}


@application.post("/upload/", response_class=JSONResponse)
async def upload(file: UploadFile = File(...)):
    img = Image.open(BytesIO(await file.read()))
    img = np.asarray(img)

    try:
        pred, prob, coords = facial_emotion(img, model)
        return {
            "Predicted":  str(pred),
            "Probability": str(prob),
            "x_min": str(coords[0]),
            "y_min": str(coords[1]),
            "x_max": str(coords[2]),
            "y_max": str(coords[3])
        }
    except: 
        # face detection failed!!
        return {"No Faces, Try Again!!"}

    




if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--ip", required=True)
    parser.add_argument("--port", "-p", required=True, type=int)

    args = parser.parse_args()


    uvicorn.run("app:application", host=args.ip, port=args.port, reload=True)
