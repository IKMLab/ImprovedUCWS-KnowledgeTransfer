#!/bin/bash

# Download the dataset and the evaluation script.
uv tool run gdown 19IuLtEVB0UC8JloqvGCjMb8bUbF3beOq -O data.tar
tar xvf data.tar && rm -rf data.tar

# Create the directory for experiments.
mkdir -p exp

# Create the Python virtual environment.
uv venv --python=3.8

# Activate the virtual environment.
source .venv/bin/activate
uv init

# Install torch.
uv pip install torch==1.13.0+cu117 -f https://download.pytorch.org/whl/cu117/torch_stable.html

# Install the requirements from the requirements.txt file.
cat requirements.txt | xargs -n 1 uv add
