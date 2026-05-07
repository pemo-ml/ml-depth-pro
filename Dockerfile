# Base image with CUDA 11.8, cuDNN 8, and Ubuntu 20.04
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.9 and essential packages
RUN apt-get update && apt-get install -y \
    software-properties-common \
    python3.9 python3.9-venv python3.9-distutils python3-pip \
    git wget unzip ffmpeg \
    libsm6 libxext6 libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.9 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1 \
 && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Upgrade pip, setuptools, and wheel
RUN pip install --upgrade pip setuptools wheel

# Clone DepthPro repo
WORKDIR /workspace
RUN git clone https://github.com/apple/ml-depth-pro.git
WORKDIR /workspace/ml-depth-pro

# Install using pyproject.toml (editable mode)
RUN pip install -e .

# Optional: download pretrained models
RUN bash get_pretrained_models.sh

# Set default shell
CMD ["/bin/bash"]

