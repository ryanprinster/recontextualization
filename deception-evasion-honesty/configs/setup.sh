# Tested on the pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel
# docker image, but should work on any recent platform
# with CUDA support
# source activate  # Commented out - only needed if using conda
apt-get update
apt-get install -y rsync
pip install uv

# Safest to install flash attention before other packages
uv pip install --system torch

# For H200 GPUs with PyTorch 2.x, we'll use the built-in SDPA (Scaled Dot Product Attention)
# which includes flash attention kernels and is more compatible with various PyTorch versions
# Flash-attn package often has compatibility issues with nightly PyTorch builds
# SDPA is the recommended approach for PyTorch 2.x and works great with H200s

uv pip install --system -e '.[dev]'
