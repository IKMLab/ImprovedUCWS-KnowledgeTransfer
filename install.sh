#!/bin/bash

# Download the dataset and the evaluation script.
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=19IuLtEVB0UC8JloqvGCjMb8bUbF3beOq' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=19IuLtEVB0UC8JloqvGCjMb8bUbF3beOq" -O data.tar && rm -rf /tmp/cookies.txt
tar xvf data.tar && rm -rf data.tar

# Create the directory for experiments.
mkdir -p exp

# Create the Python virtual environment.
python3 -m venv .venv

# Activate the virtual environment.
source .venv/bin/activate

# Install torch.
pip install torch==1.13.0+cu117 -f https://download.pytorch.org/whl/cu117/torch_stable.html

# Install the requirements from the requirements.txt file.
pip install -r requirements.txt
