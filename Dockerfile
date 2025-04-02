FROM tensorflow/tensorflow:latest-gpu

WORKDIR /netowrk-viz-app

COPY ./requirements.txt .

RUN python3 -m pip install --upgrade pip && \
    pip install --ignore-installed -r requirements.txt