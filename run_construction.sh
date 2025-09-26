#!/bin/bash

# 简单的数据生成脚本
# 用于调用data-construction目录中的Python脚本生成训练数据

set -e  # 遇到错误时退出

# 默认配置
DATA_PATH="./data/train-unified-feedback"
INPUT_FILE="$DATA_PATH/train_data_bi.jsonl"
MODEL_NAME="gemini-2.0-flash"
LIMIT=""

echo "=== 数据生成脚本 ==="
echo "输入文件: $INPUT_FILE"
echo "模型名称: $MODEL_NAME"

export API_KEY="PUT YOUR KEY HERE"

# echo ""
# echo "步骤1: 生成相似指令..."
# python gptinst.py \
#     --input_file "$INPUT_FILE" \
#     --output_file "step1_similar.jsonl" \
#     --model_name "$MODEL_NAME" \
#     $LIMIT_ARG

# echo ""
# echo "步骤2: 判断指令相似性..."
# python judge_similarity.py \
#     --input_file "step1_similar.jsonl" \
#     --output_file "step2_judged.jsonl" \
#     --model_name "$MODEL_NAME" \
#     $LIMIT_ARG

# echo ""
# echo "步骤3: 为不同指令生成新的rejected回答..."
# python generate_sim_rejected.py \
#     --input_file "step2_judged.jsonl" \
#     --output_file "step3_rejected.jsonl" \
#     --model_name "$MODEL_NAME" \
#     $LIMIT_ARG

echo ""
echo "步骤1: 注入偏见回答..."
python -u codes/construct/gptout.py \
    --input_file "$DATA_PATH/train_data_bi.jsonl" \
    --output_file "$DATA_PATH/train_data_bi_biased.jsonl" \
    --model_name "$MODEL_NAME" 

echo ""
echo "步骤2: 验证rejected回答的正确性..."
python codes/construct/verify_correctness.py \
    --input_file "$DATA_PATH/train_data_bi_biased.jsonl" \
    --output_file "$DATA_PATH/train_data_bi_biased_eval.jsonl" \
    --model_name "$MODEL_NAME"