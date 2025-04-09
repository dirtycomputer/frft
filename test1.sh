#!/bin/bash

# Set your virtual environment or conda env here if needed
# source activate your_env

# Define the list of all model types
MODEL_TYPES=(
  # "normal"
  # "frft_1_trainable"
  # "frft_1_fixed"
  # "frft_5_trainable"
  # "frft_5_fixed"
  # "spatial_frft_1_fixed"
  # "spatial_frft_1_trainable"
  # "spatial_frft_5_trainable"
  # "spatial_frft_5_fixed"
  # "mixed_frft_1_fixed"
  "mixed_frft_1_trainable"
  "mixed_frft_5_trainable"
  "mixed_frft_5_fixed"
)

# Directory to store logs
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

# Loop through each model type
for MODEL in "${MODEL_TYPES[@]}"; do
  echo "Running model: $MODEL"
  LOG_FILE="${LOG_DIR}/${MODEL}.log"
  
  python frft.py \
    --model "$MODEL" \
    --device "cuda:1" \
    2>&1 | tee "$LOG_FILE"
done