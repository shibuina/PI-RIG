#!/bin/bash

# Define the path to the target conda environment
ENV_NAME="final"
ENV_PATH="$HOME/anaconda3/envs/$ENV_NAME"

# Check if the conda environment exists
if [ -d "$ENV_PATH" ]; then
    echo "Conda environment '$ENV_NAME' already exists at $ENV_PATH. Proceeding..."
else
    echo "Conda environment '$ENV_NAME' not found. Creating environment using env.yml..."
    conda env create -f "$PWD/env.yml"
fi

# Activate the environment
# Important: This works if you run with `bash` (not `sh`)
echo "Activating conda environment '$ENV_NAME'..."
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

# Function to check if a package is installed in the current env
is_installed() {
    pip show "$1" > /dev/null 2>&1
}

# Install rlkit if not installed
if is_installed rlkit; then
    echo "rlkit is already installed."
else
    echo "Installing rlkit..."
    cd rlkit || { echo "Failed to enter rlkit directory"; exit 1; }
    pip install -e .
    cd ..
fi

# Install multiworld if not installed
if is_installed multiworld; then
    echo "multiworld is already installed."
else
    echo "Installing multiworld..."
    cd multiworld || { echo "Failed to enter multiworld directory"; exit 1; }
    pip install -e .
    cd ..
fi

echo "Setup complete."