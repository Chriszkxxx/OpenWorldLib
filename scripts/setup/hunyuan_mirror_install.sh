#!/bin/bash
# scripts/setup/hunyuan_mirror_install.sh
# Description: Setup environment for HunyuanVideo installation of OpenWorldLib
# Usage: bash scripts/setup/hunyuan_mirror_install.sh

echo "=== [1/3] Installing the base environment ==="
pip install torch==2.5.1 torchvision torchaudio
pip install git+https://github.com/openai/CLIP.git

echo "=== [2/3] Installing the requirements ==="

# Install base package with 3d_ply_default extras
# Covers: transformers, tokenizers, peft, diffusers, plyfile (via pyproject.toml)
pip install -e ".[3d_ply_default]"
pip install "plyfile==1.1.3"

# Install HunyuanVideo-specific dependencies not covered by 3d_ply_default
pip install \
  "moviepy==1.0.3" \
  "omegaconf==2.3.0" \
  "pydantic==2.11.10" \
  "scipy==1.10.1" \
  "requests==2.31.0" \
  "trimesh==4.10.1" \
  "matplotlib==3.7.2" \
  "pillow_heif==0.22.0" \
  "onnxruntime==1.23.2" \
  "colorspacious==1.1.2" \
  "numpy==1.24.4" \
  "open3d==0.18.0" \
  "torchmetrics==1.3.2" \
  "pre-commit==3.6.2" \
  "rich==13.3.5" \
  "pytest==6.2.5" \
  "roma==1.5.4" \
  "viser==0.2.11" \
  "huggingface-hub[torch]>=0.22" \
  "ninja==1.11.1" \
  "jaxtyping==0.3.4" \
  "lpips==0.1.4" \
  "pycolmap==3.10.0" \
  "tyro==1.0.3"

echo "=== [3/3] Installing the flash attention ==="
pip install "flash-attn==2.5.9.post1" --no-build-isolation

echo "=== Setup completed! ==="