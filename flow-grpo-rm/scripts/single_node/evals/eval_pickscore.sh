#!/bin/bash
# Evaluate checkpoints on PickScore
# Usage: bash scripts/single_node/evals/eval_pickscore.sh

# ============ 配置参数 ============

# checkpoint 目录 (修改为你的路径)
CHECKPOINT_DIR="logs/GRPO-Fast-nocfg-OurRM-0.2_proxy_adv_clip3e-6_lr5e-5-update_2026.04.18_09.48.56/checkpoints"

# 评测范围和间隔
START_STEP=120
END_STEP=              # 留空表示评测到最后一个 checkpoint
EVAL_INTERVAL=120

# 测试数据集
TEST_PROMPTS="dataset/pickscore/test.txt"

# 基础模型 (SD3.5 Medium 用于 pickscore_sd3_fast_rm)
BASE_MODEL_PATH="stabilityai/stable-diffusion-3.5-medium"

# 推理参数 (基于 pickscore_sd3_fast_rm 配置)
BATCH_SIZE=4
NUM_INFERENCE_STEPS=40
GUIDANCE_SCALE=4.5
RESOLUTION=512
NOISE_LEVEL=0.8
SDE_WINDOW_SIZE=3
SDE_TYPE="cps"

# 输出结果文件
OUTPUT_JSON="eval_results/pickscore_results.json"

# ============ 执行评测 ============

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Go up to project root: scripts/single_node/[evals/] -> project root
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

cd "${PROJECT_DIR}"

# 构建命令
CMD="python scripts/eval_pickscore_checkpoints.py \
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
echo "Running PickScore evaluation..."
echo "Checkpoint dir: ${CHECKPOINT_DIR}"
echo "Start step: ${START_STEP}, End step: ${END_STEP:-'last'}, Interval: ${EVAL_INTERVAL}"
echo "Inference: steps=${NUM_INFERENCE_STEPS}, guidance=${GUIDANCE_SCALE}, noise=${NOISE_LEVEL}, sde_window=${SDE_WINDOW_SIZE}, sde_type=${SDE_TYPE}"
echo ""

eval ${CMD}
