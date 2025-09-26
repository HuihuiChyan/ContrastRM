#!/bin/bash

# Reward Model Evaluation Script
# This script evaluates the trained reward model on different bias types


export CUDA_VISIBLE_DEVICES=3

echo "Starting Reward Model Evaluation..."
echo "==================================="

TRAIN_MODE="pairwise"

# Default parameters
MODEL_PATH="./output/Qwen2.5-0.5B-Instruct-$TRAIN_MODE"
BIAS_TYPE="all"
BATCH_SIZE=32

# Check if the model directory exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model directory $MODEL_PATH not found!"
    echo "Please ensure the model is trained and saved to the correct location."
    echo "You can train the model using: ./run_training.sh"
    exit 1
fi

# Check if evaluation data directory exists
if [ ! -d "./data" ]; then
    echo "Error: Data directory ./data not found!"
    echo "Please ensure the evaluation data files are available."
    exit 1
fi

# Create results directory if it doesn't exist
mkdir -p ./results

echo "Configuration:"
echo "  Model Path: $MODEL_PATH"
echo "  Bias Type: $BIAS_TYPE"
echo "  Batch Size: $BATCH_SIZE"
echo ""

# Run evaluation
python codes/evaluate/eval_rm.py "$BIAS_TYPE" \
    --model_path "$MODEL_PATH" \
    --batch_size "$BATCH_SIZE"

if [ $? -eq 0 ]; then
    echo ""
    echo "Evaluation completed successfully!"
    echo "Results saved to: ./results/"
else
    echo ""
    echo "Evaluation failed. Please check the error messages above."
    exit 1
fi
