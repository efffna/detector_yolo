FROM python:3.9

ENV APP_DIR=/src
WORKDIR $APP_DIR

COPY requirements.txt $APP_DIR/

RUN apt-get update && apt upgrade -y && \
    apt-get install -y libsm6 libxrender1 libfontconfig1 libxext6 libgl1-mesa-glx && \
    pip install -U pip setuptools wheel && \
    pip install -U --no-cache-dir -r $APP_DIR/requirements.txt
