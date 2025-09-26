#!/bin/bash

# Reward-Bench Evaluation Script
# This script evaluates the trained reward model on Reward-Bench dataset


export CUDA_VISIBLE_DEVICES=0

echo "Starting Reward-Bench Evaluation..."
echo "===================================="

TRAIN_MODE="pairwise"

# Default parameters
MODEL_PATH="./output/Qwen2.5-0.5B-Instruct-$TRAIN_MODE"
CATEGORY="all"  # Options: Chat, "Chat Hard", Safety, Reasoning, all
BATCH_SIZE=32

# Check if the model directory exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model directory $MODEL_PATH not found!"
    echo "Please ensure the model is trained and saved to the correct location."
    echo "You can train the model using: ./run_training.sh"
    exit 1
fi


# Create results directory if it doesn't exist
mkdir -p ./results

echo "Configuration:"
echo "  Model Path: $MODEL_PATH"
echo "  Category: $CATEGORY"
echo "  Batch Size: $BATCH_SIZE"
echo ""

# Run evaluation
python codes/evaluate/eval_reward_bench.py "$CATEGORY" \
    --model_path "$MODEL_PATH" \
    --batch_size "$BATCH_SIZE"

if [ $? -eq 0 ]; then
    echo ""
    echo "Reward-Bench evaluation completed successfully!"
    echo "Results saved to: ./results/"
else
    echo ""
    echo "Evaluation failed. Please check the error messages above."
    exit 1
fi
