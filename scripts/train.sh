#!/bin/bash
# Brain Tumor Detection - Training Script

echo "Starting Brain Tumor Detection Training..."

# Set default parameters
DATA_DIR="data"
EPOCHS=50
BATCH_SIZE=32
LEARNING_RATE=0.001
IMG_SIZE=240
MODEL_PATH="cnn-parameters-improvement-23-0.91.model"

# Run training
python main.py \
    --mode train \
    --data_dir "$DATA_DIR" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LEARNING_RATE" \
    --img_size "$IMG_SIZE" \
    --model "$MODEL_PATH"

echo "Training completed." 