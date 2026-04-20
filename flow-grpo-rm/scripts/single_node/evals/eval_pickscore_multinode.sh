#!/bin/bash
# Multi-GPU PickScore evaluation
# Usage: bash scripts/single_node/evals/eval_pickscore_multinode.sh

# ============ 配置参数 ============

# GPU 数量
NUM_GPUS=8

# checkpoint 目录 (修改为你的路径)
CHECKPOINT_DIR="logs/GRPO-Fast-nocfg-OurRM-0.2_proxy_adv_clip3e-6_lr5e-5-update_2026.04.18_09.48.56/checkpoints"

# 评测范围和间隔
START_STEP=120
END_STEP=2400
EVAL_INTERVAL=120

# 测试数据集
TEST_PROMPTS="dataset/pickscore/test.txt"

# 基础模型
BASE_MODEL_PATH="/models/stable-diffusion-3.5-medium"

# 推理参数 (评测时用，与 train_sd3_fast_rm.py eval 阶段一致)
BATCH_SIZE=1              # 每张 GPU 的 batch size
NUM_INFERENCE_STEPS=40
GUIDANCE_SCALE=4.5
RESOLUTION=512
NOISE_LEVEL=0
SDE_WINDOW_SIZE=0
SDE_TYPE="discrete"

# 输出结果文件
OUTPUT_JSON="eval_results/pickscore_results.json"

# ============ 执行评测 ============

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

cd "${PROJECT_DIR}"

# 设置 PYTHONPATH
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"

# 设置分布式训练端口（默认 29500，被占用时可修改）
export RANK_0_PORT=29501

# 构建 accelerate 命令
CMD="accelerate launch \
    --num_processes=${NUM_GPUS} \
    --main_training_function main \
    scripts/eval_pickscore_checkpoints_multinode.py \
    --checkpoint_dir '${CHECKPOINT_DIR}' \
    --base_model_path '${BASE_MODEL_PATH}' \
    --start_step ${START_STEP} \
    --eval_interval ${EVAL_INTERVAL} \
    --test_prompts_path '${TEST_PROMPTS}' \
    --batch_size ${BATCH_SIZE} \
    --num_inference_steps ${NUM_INFERENCE_STEPS} \
    --guidance_scale ${GUIDANCE_SCALE} \
    --resolution ${RESOLUTION} \
    --noise_level ${NOISE_LEVEL} \
    --sde_window_size ${SDE_WINDOW_SIZE} \
    --sde_type '${SDE_TYPE}' \
    --output_json '${OUTPUT_JSON}'"

# 添加 end_step 如果设置了
if [ -n "${END_STEP}" ]; then
    CMD="${CMD} --end_step ${END_STEP}"
fi

# 执行
echo "Running Multi-GPU PickScore evaluation..."
echo "Number of GPUs: ${NUM_GPUS}"
echo "Checkpoint dir: ${CHECKPOINT_DIR}"
echo "Start step: ${START_STEP}, End step: ${END_STEP:-'last'}, Interval: ${EVAL_INTERVAL}"
echo "Inference: steps=${NUM_INFERENCE_STEPS}, guidance=${GUIDANCE_SCALE}, batch_size=${BATCH_SIZE}/GPU"
echo ""

eval ${CMD}
