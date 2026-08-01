#!/bin/bash
set -e

cd /home/dev/quant/Kronos
VENV_PYTHON=".venv/bin/python"
VENV_PIP=".venv/bin/pip"

echo "===== Step 1: Create venv and install deps ====="
python3 -m venv .venv
$VENV_PIP install --upgrade pip setuptools wheel
$VENV_PIP install torch --index-url https://download.pytorch.org/whl/cpu
$VENV_PIP install numpy pandas einops huggingface_hub matplotlib tqdm safetensors

echo "===== Step 2: Verify packages ====="
$VENV_PYTHON -c "
import torch; import numpy; import pandas; import einops; import huggingface_hub
print(f'torch={torch.__version__}, numpy={numpy.__version__}, pandas={pandas.__version__}')
print('All packages OK')
"

echo "===== Step 3: Test Kronos model loading ====="
$VENV_PYTHON -c "
import sys
sys.path.insert(0, '/home/dev/quant/Kronos')
from model import Kronos, KronosTokenizer, KronosPredictor
print('Kronos imports OK')
"

echo "===== Setup DONE ====="