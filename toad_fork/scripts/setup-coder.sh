#!/bin/bash -e

sudo apt update
sudo apt-get -y install python3-dev python-is-python3

curl -sS https://bootstrap.pypa.io/get-pip.py | python

if [ "$(uname -m)" = "aarch64" ]; then
  cd /workspaces
  [ -f cuda_12.4.0_550.54.14_linux_sbsa.run ] || wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda_12.4.0_550.54.14_linux_sbsa.run
  sudo sh cuda_12.4.0_550.54.14_linux_sbsa.run --toolkit --silent # default debian nvidia-cuda-toolkit does not work on aarch64

  git clone --depth 1 --branch release/3.1.x https://github.com/triton-lang/triton.git || true
  cd triton
  pip install ninja cmake wheel
  pip install -e python

  sudo ln -sf /lib/aarch64-linux-gnu/libcuda.so.1 /lib/aarch64-linux-gnu/libcuda.so # to make -lcuda work. Otherwise torch compile fails.
else
  sudo apt-get -y install nvidia-cuda-toolkit
fi

pip install hydra-core hydra-colorlog wandb gym matplotlib einops
pip install scikit-learn schedule
pip install "jax<0.5"

pip install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 # --index-url is required to work on aarch64
