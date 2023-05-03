FROM tensorflow/tensorflow:2.11.0-gpu
RUN python -m pip install pip --upgrade
WORKDIR /app
ADD ./requirements.txt .
RUN python -m pip install -r requirements.txt
ADD ./algos .
ADD train_headless.py .
