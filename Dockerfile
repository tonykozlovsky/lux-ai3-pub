FROM nvidia/cuda:12.4.0-devel-ubuntu20.04

# Arguments for user setup
ARG USER=root
ARG UID=1000
ARG GID=1000

# Set timezone non-interactively
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/London
# Change to your desired timezone

# Install base dependencies and set up timezone
RUN apt-get update && apt-get install -y --no-install-recommends \
    apt-utils \
    tzdata \
    sudo \
    software-properties-common \
    curl \
    ncdu \
    nano && \
    ln -fs /usr/share/zoneinfo/$TZ /etc/localtime && \
    dpkg-reconfigure -f noninteractive tzdata && \
    rm -rf /var/lib/apt/lists/*

# Add deadsnakes PPA and install Python 3.11
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-distutils python3.11-dev && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --set python3 /usr/bin/python3.11 && \
    rm -rf /var/lib/apt/lists/*

# Install pip for Python 3.11
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# Install additional Python packages
RUN pip3 install --no-cache-dir hydra-core
RUN pip3 install -U jax
RUN pip3 install -U torch torchvision torchaudio
RUN pip3 install wandb gym matplotlib hydra-colorlog
RUN pip3 install scikit-learn schedule einops python-is-python3

# Create non-root user and set up permissions
RUN groupadd -g $GID $USER && \
    useradd -m -u $UID -g $GID -G sudo -s /bin/bash $USER && \
    echo "$USER ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Switch to the non-root user
USER $USER
WORKDIR /home/$USER
