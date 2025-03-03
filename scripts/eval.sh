#!/bin/bash
# Brain Tumor Detection - Evaluation Script

echo "Starting Brain Tumor Detection Evaluation..."

# Set default parameters
DATA_DIR="data"
IMG_SIZE=240
MODEL_PATH="cnn-parameters-improvement-23-0.91.model"

# Run evaluation
python main.py \
    --mode evaluate \
    --data_dir "$DATA_DIR" \
    --img_size "$IMG_SIZE" \
    --model "$MODEL_PATH"

echo "Evaluation completed." 