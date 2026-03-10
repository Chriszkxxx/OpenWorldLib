#!/bin/bash
# scripts/setup/omnivinci_install.sh
# Description: Setup environment for OmniVinci model dependencies
# Usage: bash scripts/setup/omnivinci_install.sh

echo "=== [1/4] Installing core model dependencies ==="
pip install sentencepiece==0.1.99
pip install shortuuid
pip install accelerate==0.34.2
pip install bitsandbytes==0.43.2
pip install einops==0.6.1
pip install einops-exts==0.0.4
pip install timm==0.9.12

echo "=== [2/4] Installing vision and video dependencies ==="
pip install opencv-python-headless==4.8.0.76
pip install pytorchvideo==0.1.5
pip install decord==0.6.0

echo "=== [3/4] Installing audio dependencies ==="
pip install soundfile
pip install librosa
pip install openai-whisper
pip install kaldiio
pip install ffmpeg-python

echo "=== [4/4] Installing utility dependencies ==="
pip install nltk==3.3
pip install pywsd==1.2.4
pip install datasets==2.16.1
pip install requests
pip install httpx
pip install fire
pip install tyro
pip install hydra-core
pip install xgrammar
pip install "protobuf==3.20.*"
pip install beartype
pip install "pydantic==1.10.22"
pip install "s2wrapper@git+https://github.com/bfshi/scaling_on_scales"

echo "=== Setup completed! ==="
