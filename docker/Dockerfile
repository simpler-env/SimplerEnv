ARG OS_VERSION=22.04
ARG CUDA_VERSION=11.7.1

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${OS_VERSION}
ENV NVIDIA_DRIVER_CAPABILITIES all
ARG PYTHON_VERSION=3.9

# Run apt-get (dkpg) without interactive dialogue
ARG DEBIAN_FRONTEND=noninteractive

# For CUDA library and nvidia-container-runtime for OpenGL
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH} \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
COPY 10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json


RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    bash-completion \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    git \
    htop \
    libegl1 \
    libxext6 \
    libjpeg-dev \
    libpng-dev  \
    libvulkan1 \
    rsync \
    tmux \
    unzip \
    vim \
    vulkan-utils \
    wget \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

# Install (mini) conda
RUN curl -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda init && \
    /opt/conda/bin/conda install -y python="$PYTHON_VERSION" && \
    /opt/conda/bin/conda clean -ya

ENV PATH /opt/conda/bin:$PATH
SHELL ["/bin/bash", "-c"]

COPY Real2Sim /Real2Sim
WORKDIR /Real2Sim
RUN pip install tensorflow==2.15.0 && pip cache purge
RUN pip install -e . && pip cache purge

