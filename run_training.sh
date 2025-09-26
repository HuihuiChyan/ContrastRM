#!/bin/bash

# Reward Model Training Script
# This script trains a reward model using contrastive learning

export CUDA_VISIBLE_DEVICES=0,5,6,7

echo "Starting Reward Model Training..."
echo "=================================="

TEMPERATURE=0.5
TRAIN_MODE="pairwise"

MODEL_PATH="/workspace/HFModels/Qwen2.5-0.5B-Instruct"
OUTPUT_PATH="./output/Qwen2.5-0.5B-Instruct-$TRAIN_MODE" #-temp$TEMPERATURE"
DATA_PATH="./data/train-unified-feedback/train_data_bi.jsonl"

# Check if the model directory exists
if [ ! -d $MODEL_PATH ]; then
    echo "Error: Model directory $MODEL_PATH not found!"
    echo "Please ensure the model is downloaded to the correct location."
    exit 1
fi

# Check if training data exists
if [ ! -f $DATA_PATH ]; then
    echo "Error: Training data $DATA_PATH not found!"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_PATH

# Run training with mixed precision (BF16 for better stability)
accelerate launch --deepspeed_config_file configs/ds_z2_config.json codes/train/train_reward_model.py \
    --model_name_or_path $MODEL_PATH \
    --train_data_path $DATA_PATH \
    --output_dir $OUTPUT_PATH \
    --training_mode $TRAIN_MODE \
    --num_epochs 2 \
    --batch_size 8 \
    --learning_rate 5e-6 \
    --max_length 1024 \
    --temperature $TEMPERATURE \
    --warmup_steps 100 \
    --eval_steps 500 \
    --save_steps 1000 \
    --max_grad_norm 1.0 \
    --weight_decay 0.01 \
    --seed 42 \
    --bf16

echo "Training completed!"
echo "Model saved to: $OUTPUT_PATH"
