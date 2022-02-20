# Facial-Emotion-API
<b>FastApi</b> API for Facial Emotion Classification Project, Will be accessed by web and mobile applications.

<br>
<br>
<br>

## Pretrained EfficientNetB0 Models
<br>

| Model | Link |
| ------------- | ------------- |
| 5class_emotion_model.h5 | [Drive](https://drive.google.com/file/d/1oF8c23sWTBkyYsXD8KvNGDBDMLrlnx6p/view?usp=sharing) (34MB) |

<br>
<br>
<br>


## Creating and Running Docker Container
<br>

```
docker build ./ --tag imageName:tag
```

```
docker run --name conatinerName -d -p 5000:80 imageName:tag
```

<br>
<br>
<br>


## DockerHub Built Image Pulling and Running

<br>

<a href="https://hub.docker.com/r/realrioden/facial-emotion-api">Container Link</a>

<br>

```
docker pull realrioden/facial-emotion-api
```

```
docker run --name conatinerName -d -p 5000:80 facial-emotion-api
```



## 
