FROM nvcr.io/nvidia/tensorflow:22.11-tf2-py3

WORKDIR /home


RUN apt update && \
    apt -y upgrade && \
    apt install --no-install-recommends -y zip htop screen libgl1-mesa-glx && \
    python -m pip install --upgrade pip wheel && \
    python -m pip install tensorflow-hub tensorflow-text tensorflow-addons
