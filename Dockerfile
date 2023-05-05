FROM tensorflow/tensorflow:2.11.0-gpu
RUN python -m pip install pip --upgrade
WORKDIR /app
ADD ./build_requirements.txt .
ADD ./runtime_requirements.txt .
RUN python -m pip install -r build_requirements.txt
RUN python -m pip install -r runtime_requirements.txt
ADD ./algos .
ADD train_headless.py .
