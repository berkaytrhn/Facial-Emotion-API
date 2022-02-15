# Facial-Emotion-Backend-API
Backend API for Facial Emotion Classification Project, Will be accessed by web and mobile applications.



## Creating and Running Docker Container

```
docker build ./ --tag fastapi-test:01
```

```
winpty docker run --name test-container -d -p 5000:80 fastapi-test:01
```

## Pretrained EfficientNetB0 Models

| Model | Link |
| ------------- | ------------- |
| 5class_emotion_model.h5 | [Drive](https://drive.google.com/file/d/1oF8c23sWTBkyYsXD8KvNGDBDMLrlnx6p/view?usp=sharing) (34MB) |