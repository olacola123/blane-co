#!/bin/bash
# NM i AI 2026 — Installer alt du trenger
# Kjør: bash setup.sh

echo "=== NM i AI 2026 Setup ==="

# Sjekk Homebrew
if ! command -v brew &> /dev/null; then
    echo "Installerer Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

# Python og Node
echo "Installerer Python og Node..."
brew install python node

# Lag virtualenv
echo "Lager Python-miljø..."
python3 -m venv env
source env/bin/activate

# ML-pakker
echo "Installerer ML-pakker (dette tar et par minutter)..."
pip install --upgrade pip
pip install torch torchvision torchaudio
pip install stable-baselines3[extra]
pip install transformers sentence-transformers
pip install chromadb
pip install opencv-python-headless
pip install scikit-learn
pip install segmentation-models-pytorch
pip install gymnasium
pip install requests python-dotenv websockets

echo ""
echo "=== Ferdig! ==="
echo "Aktiver miljøet med: source env/bin/activate"
echo "Sjekk at alt funker med: python -c 'import torch; print(torch.backends.mps.is_available())'"
