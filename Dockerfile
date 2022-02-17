FROM ubuntu:latest

RUN mkdir /home/src

COPY requirements.txt /home/src
COPY app.py /home/src
COPY src /home/src/src

WORKDIR /home/src



#Setup
RUN apt-get update && apt-get -y update

RUN apt-get install -y build-essential python3.8 python3-pip python3-dev

RUN pip3 -q install pip --upgrade

RUN pip3 install -r requirements.txt

# to avoid timezone request, set timezone here
ENV TZ=Europe/Istanbul
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone


# avoid cv2 import error
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

CMD python3 app.py --ip 0.0.0.0 --port 80
