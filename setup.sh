#!/bin/bash

# Define the path to the target conda environment
ENV_NAME="final"
ENV_PATH="$HOME/anaconda3/envs/$ENV_NAME"

# Function to check if a system package is installed
is_system_package_installed() {
    dpkg -l | grep -q "^ii  $1 "
}

# Check and install required system packages
echo "Checking for required system packages..."

REQUIRED_PACKAGES=("libgl1-mesa-glx" "libglu1-mesa" "libosmesa6" "ffmpeg" "patchelf")
PACKAGES_TO_INSTALL=()

for package in "${REQUIRED_PACKAGES[@]}"; do
    if is_system_package_installed "$package"; then
        echo "$package is already installed."
    else
        echo "$package is not installed. Adding to install list..."
        PACKAGES_TO_INSTALL+=("$package")
    fi
done

# Install missing packages if any
if [ ${#PACKAGES_TO_INSTALL[@]} -gt 0 ]; then
    echo "Installing missing packages: ${PACKAGES_TO_INSTALL[*]}"
    sudo apt-get update
    sudo apt-get install -y "${PACKAGES_TO_INSTALL[@]}"
else
    echo "All required system packages are already installed."
fi

# Install MuJoCo 2.1.0
MUJOCO_DIR="$HOME/.mujoco"
MUJOCO_210_DIR="$MUJOCO_DIR/mujoco210"

if [ -d "$MUJOCO_210_DIR" ]; then
    echo "MuJoCo 2.1.0 is already installed at $MUJOCO_210_DIR"
else
    echo "Installing MuJoCo 2.1.0..."
    mkdir -p "$MUJOCO_DIR"
    cd "$MUJOCO_DIR" || { echo "Failed to enter .mujoco directory"; exit 1; }
    
    echo "Downloading MuJoCo 2.1.0..."
    curl -L https://github.com/google-deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz -o mujoco210.tar.gz
    
    echo "Extracting MuJoCo 2.1.0..."
    tar -xzf mujoco210.tar.gz
    rm mujoco210.tar.gz
    
    echo "MuJoCo 2.1.0 installed successfully at $MUJOCO_210_DIR"
    cd - > /dev/null
fi

# Set up environment variables for MuJoCo
echo "Setting up MuJoCo environment variables..."
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

# Add environment variables to .bashrc if not already present
BASHRC="$HOME/.bashrc"
if ! grep -q "LD_LIBRARY_PATH.*mujoco210" "$BASHRC"; then
    echo "" >> "$BASHRC"
    echo "# MuJoCo environment variables" >> "$BASHRC"
    echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:\$HOME/.mujoco/mujoco210/bin" >> "$BASHRC"
    echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/usr/lib/nvidia" >> "$BASHRC"
    echo "MuJoCo environment variables added to .bashrc"
else
    echo "MuJoCo environment variables already exist in .bashrc"
fi

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
echo "Note: Please restart your terminal or run 'source ~/.bashrc' to apply MuJoCo environment variables."