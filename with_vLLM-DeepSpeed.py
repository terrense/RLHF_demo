#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# RLHF engineering skeleton based on OpenRLHF + vLLM + DeepSpeed + Ray
# ----------------------------------------------------------------------------
# 这不是 toy PPO 的“数学演示版”，而是更接近真实工程的训练脚手架。
#
# 目标：
# 1) 用 DeepSpeed 跑 SFT
# 2) 用 DeepSpeed 跑 Reward Model
# 3) 用 OpenRLHF 的 Ray + vLLM + DeepSpeed 栈跑 PPO
#
# 你真正该理解的是：
# - OpenRLHF 负责 RLHF workflow orchestration
# - vLLM 负责高吞吐 rollout generation
# - DeepSpeed 负责大模型训练侧的显存与分布式优化
# - Ray 负责多角色（actor/ref/reward/critic/vLLM engines）的调度
#
# 官方资料对应关系：
# - OpenRLHF README 里给了 deepspeed train_sft / train_rm，以及 ray job submit + train_ppo_ray 的用法
# - OpenRLHF 强调其架构是 Ray + vLLM + DeepSpeed
# - DeepSpeed ZeRO 的核心是通过 zero_optimization / stage 1/2/3 分片优化器状态、梯度、参数
# - vLLM 的 OpenAI-compatible server 是单独 serving 的常见路径；但 OpenRLHF PPO 训练时通常直接内嵌/调度 vLLM engines
#
# 这份脚手架的定位：
# - 可作为 repo 里的“工程版 starter”
# - 参数需要按你的机器、模型、数据路径做调整
# - 并不保证你 copy-paste 后 100% 在任意环境零修改运行
# - 但结构、职责拆分、命令组织方式是对的
# ============================================================================


# ==============================
# 0. 用户可配置区
# ==============================

# 基础路径
export PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
export DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data}"
export CKPT_ROOT="${CKPT_ROOT:-${PROJECT_ROOT}/checkpoints}"
export LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/logs}"

mkdir -p "${DATA_ROOT}" "${CKPT_ROOT}" "${LOG_ROOT}"

# 模型与数据
export BASE_MODEL="${BASE_MODEL:-meta-llama/Meta-Llama-3-8B-Instruct}"
export SFT_DATASET="${SFT_DATASET:-Open-Orca/OpenOrca}"
export RM_DATASET="${RM_DATASET:-OpenRLHF/preference_dataset_mixture2_and_safe_pku}"
export PPO_PROMPT_DATASET="${PPO_PROMPT_DATASET:-OpenRLHF/prompt-collection-v0.1}"

# 输出路径
export SFT_CKPT="${SFT_CKPT:-${CKPT_ROOT}/llama3-8b-sft}"
export RM_CKPT="${RM_CKPT:-${CKPT_ROOT}/llama3-8b-rm}"
export PPO_CKPT="${PPO_CKPT:-${CKPT_ROOT}/llama3-8b-ppo}"

# 训练精度 / batch / 长度
export PARAM_DTYPE="${PARAM_DTYPE:-bf16}"
export MAX_LEN_SFT="${MAX_LEN_SFT:-4096}"
export MAX_LEN_RM="${MAX_LEN_RM:-4096}"
export PROMPT_MAX_LEN="${PROMPT_MAX_LEN:-1024}"
export GENERATE_MAX_LEN="${GENERATE_MAX_LEN:-512}"

export SFT_TRAIN_BATCH_SIZE="${SFT_TRAIN_BATCH_SIZE:-128}"
export SFT_MICRO_BATCH_SIZE="${SFT_MICRO_BATCH_SIZE:-2}"
export RM_TRAIN_BATCH_SIZE="${RM_TRAIN_BATCH_SIZE:-128}"
export RM_MICRO_BATCH_SIZE="${RM_MICRO_BATCH_SIZE:-1}"

# PPO 训练超参
export PPO_ZERO_STAGE="${PPO_ZERO_STAGE:-3}"
export ACTOR_LR="${ACTOR_LR:-5e-7}"
export CRITIC_LR="${CRITIC_LR:-9e-6}"
export INIT_KL_COEF="${INIT_KL_COEF:-0.01}"
export PPO_ROLLOUT_BATCH_SIZE="${PPO_ROLLOUT_BATCH_SIZE:-256}"
export PPO_MICRO_ROLLOUT_BATCH_SIZE="${PPO_MICRO_ROLLOUT_BATCH_SIZE:-16}"
export PPO_TRAIN_BATCH_SIZE="${PPO_TRAIN_BATCH_SIZE:-128}"
export PPO_MICRO_TRAIN_BATCH_SIZE="${PPO_MICRO_TRAIN_BATCH_SIZE:-2}"
export MAX_SAMPLES="${MAX_SAMPLES:-100000}"
export MAX_EPOCHS="${MAX_EPOCHS:-1}"

# GPU 资源切分（示意值，你必须按机器改）
# OpenRLHF 文档给出的角色：actor / ref / reward / critic / vLLM engines
# 示例中用 8 卡；你自己的机器少于这个数，就按比例缩。
export REF_NUM_NODES="${REF_NUM_NODES:-1}"
export REF_NUM_GPUS_PER_NODE="${REF_NUM_GPUS_PER_NODE:-1}"
export REWARD_NUM_NODES="${REWARD_NUM_NODES:-1}"
export REWARD_NUM_GPUS_PER_NODE="${REWARD_NUM_GPUS_PER_NODE:-1}"
export CRITIC_NUM_NODES="${CRITIC_NUM_NODES:-1}"
export CRITIC_NUM_GPUS_PER_NODE="${CRITIC_NUM_GPUS_PER_NODE:-1}"
export ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-1}"
export ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-1}"
export VLLM_NUM_ENGINES="${VLLM_NUM_ENGINES:-1}"
export VLLM_TP_SIZE="${VLLM_TP_SIZE:-1}"

# Ray
export RAY_HEAD_ADDR="${RAY_HEAD_ADDR:-127.0.0.1}"
export RAY_HEAD_PORT="${RAY_HEAD_PORT:-6379}"
export RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"

# WandB（可选）
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-openrlhf-demo}"

# 你可以把 HF token / datasets cache / model cache 也放这里
export HF_HOME="${HF_HOME:-${PROJECT_ROOT}/.hf_cache}"
mkdir -p "${HF_HOME}"


# ==============================
# 1. 环境检查
# ==============================

function print_banner() {
  echo
  echo "=============================================================="
  echo "$1"
  echo "=============================================================="
}

function require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "[ERROR] Missing command: $1"
    exit 1
  fi
}

function check_env() {
  print_banner "[1/8] Checking environment"
  require_cmd python
  require_cmd deepspeed
  require_cmd ray

  python - <<'PY'
import sys
print('[Python]', sys.version)
try:
    import torch
    print('[Torch ]', torch.__version__)
except Exception as e:
    print('[Torch ] import failed:', e)
try:
    import transformers
    print('[HF    ]', transformers.__version__)
except Exception as e:
    print('[HF    ] import failed:', e)
try:
    import deepspeed
    print('[DS    ]', deepspeed.__version__)
except Exception as e:
    print('[DS    ] import failed:', e)
try:
    import ray
    print('[Ray   ]', ray.__version__)
except Exception as e:
    print('[Ray   ] import failed:', e)
PY

  echo "[INFO] PROJECT_ROOT=${PROJECT_ROOT}"
  echo "[INFO] BASE_MODEL=${BASE_MODEL}"
  echo "[INFO] SFT_CKPT=${SFT_CKPT}"
  echo "[INFO] RM_CKPT=${RM_CKPT}"
  echo "[INFO] PPO_CKPT=${PPO_CKPT}"
}


# ==============================
# 2. 依赖安装示例
# ==============================

function print_install_hint() {
  print_banner "[2/8] Installation hint (printed only, not executed)"
  cat <<'EOF'
# One possible setup sequence:
# pip install --upgrade pip
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
# pip install deepspeed ray[default] transformers datasets sentencepiece accelerate
# pip install vllm openai
# pip install openrlhf
#
# Notes:
# - vLLM / FlashAttention / CUDA build compatibility is environment-sensitive.
# - OpenRLHF often lives best in a dedicated conda env.
# - Do not pretend dependency management is trivial; it is one of the real pain points.
EOF
}


# ==============================
# 3. SFT 阶段（DeepSpeed）
# ==============================

function train_sft() {
  print_banner "[3/8] SFT training with DeepSpeed + OpenRLHF CLI"

  # 这一步对应：
  # deepspeed --module openrlhf.cli.train_sft ... --zero_stage 2/3
  #
  # 为什么先做 SFT：
  # - PPO 不是拿随机 policy 开始玩；那样基本是找死。
  # - 真实 RLHF 里 reference model 通常就是 SFT checkpoint 的 frozen copy。

  deepspeed --module openrlhf.cli.train_sft \
    --max_len "${MAX_LEN_SFT}" \
    --dataset "${SFT_DATASET}" \
    --input_key question \
    --output_key response \
    --input_template $'User: {}\nAssistant: ' \
    --train_batch_size "${SFT_TRAIN_BATCH_SIZE}" \
    --micro_train_batch_size "${SFT_MICRO_BATCH_SIZE}" \
    --max_samples "${MAX_SAMPLES}" \
    --pretrain "${BASE_MODEL}" \
    --save_path "${SFT_CKPT}" \
    --save_steps -1 \
    --logging_steps 1 \
    --eval_steps -1 \
    --zero_stage 2 \
    --max_epochs 1 \
    --packing_samples \
    --param_dtype "${PARAM_DTYPE}" \
    --learning_rate 5e-6 \
    --gradient_checkpointing \
    ${WANDB_API_KEY:+--use_wandb "${WANDB_API_KEY}"} \
    2>&1 | tee "${LOG_ROOT}/train_sft.log"
}


# ==============================
# 4. RM 阶段（DeepSpeed）
# ==============================

function train_rm() {
  print_banner "[4/8] Reward model training with DeepSpeed + OpenRLHF CLI"

  # 奖励模型阶段：
  # - 输入 chosen / rejected preference pairs
  # - 训练一个标量打分器
  # - PPO 阶段里它通常冻结，只负责打分
  #
  # 现实提醒：
  # - 很多人只盯 PPO 算法，忽视 RM 质量。
  # - 这是错的。RM 弱，后面全是 reward hacking 风险。

  deepspeed --module openrlhf.cli.train_rm \
    --save_path "${RM_CKPT}" \
    --save_steps -1 \
    --logging_steps 1 \
    --eval_steps -1 \
    --train_batch_size "${RM_TRAIN_BATCH_SIZE}" \
    --micro_train_batch_size "${RM_MICRO_BATCH_SIZE}" \
    --pretrain "${SFT_CKPT}" \
    --param_dtype "${PARAM_DTYPE}" \
    --max_epochs 1 \
    --max_len "${MAX_LEN_RM}" \
    --zero_stage 3 \
    --learning_rate 9e-6 \
    --dataset "${RM_DATASET}" \
    --apply_chat_template \
    --chosen_key chosen \
    --rejected_key rejected \
    --packing_samples \
    --gradient_checkpointing \
    ${WANDB_API_KEY:+--use_wandb "${WANDB_API_KEY}"} \
    2>&1 | tee "${LOG_ROOT}/train_rm.log"
}


# ==============================
# 5. 自定义 reward 函数示例
# ==============================

function write_custom_reward_func() {
  print_banner "[5/8] Writing a custom reward function example"

  mkdir -p "${PROJECT_ROOT}/examples"

  cat > "${PROJECT_ROOT}/examples/reward_func.py" <<'PY'
"""
Custom reward function example for OpenRLHF PPO.

定位：
- 这不是 preference RM 的替代品，而是你做 task-specific shaping 时的入口。
- 适合：格式约束、rule-based correctness、关键词检查、结构化输出校验。
- 不适合：拿一堆脆弱 heuristics 假装这就是高质量 reward model。

接口说明：
- OpenRLHF README 示例中，`--remote_rm_url /path/to/reward_func.py` 可把数据集某个字段（比如 label_key）传给 reward_func。
- 具体运行时签名和上下文可能随版本变化；这份函数故意写得保守、好改。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def reward_func(
    prompts: List[str],
    responses: List[str],
    labels: Optional[List[str]] = None,
    **kwargs: Dict[str, Any],
) -> List[float]:
    rewards: List[float] = []

    for i, (prompt, response) in enumerate(zip(prompts, responses)):
        score = 0.0

        # 1) 长度太短，通常回答质量不够
        if len(response.strip()) < 8:
            score -= 1.0

        # 2) 回答包含明确结论，给一点正奖励
        lowered = response.lower()
        if "therefore" in lowered or "final answer" in lowered or "answer:" in lowered:
            score += 0.3

        # 3) 如果给了监督标签，可做简单匹配奖励
        if labels is not None and i < len(labels) and labels[i] is not None:
            gold = str(labels[i]).strip().lower()
            if gold and gold in lowered:
                score += 1.0
            else:
                score -= 0.5

        # 4) 避免明显拒答模板泛滥（示意）
        if "i cannot help with that" in lowered:
            score -= 0.3

        rewards.append(score)

    return rewards
PY

  echo "[INFO] Wrote ${PROJECT_ROOT}/examples/reward_func.py"
}


# ==============================
# 6. 启动 Ray
# ==============================

function start_ray_head() {
  print_banner "[6/8] Starting Ray head"

  # OpenRLHF 的 PPO Ray 版靠 Ray 调度 actor/ref/reward/critic/vLLM engines。
  # 单机可以先启动 head。多机时要再加 worker 节点接入。

  ray stop || true
  ray start \
    --head \
    --node-ip-address="${RAY_HEAD_ADDR}" \
    --port="${RAY_HEAD_PORT}" \
    --dashboard-host=0.0.0.0 \
    --dashboard-port="${RAY_DASHBOARD_PORT}"

  echo "[INFO] Ray dashboard: http://${RAY_HEAD_ADDR}:${RAY_DASHBOARD_PORT}"
}


# ==============================
# 7. PPO 阶段（OpenRLHF + Ray + vLLM + DeepSpeed）
# ==============================

function train_ppo() {
  print_banner "[7/8] PPO training with OpenRLHF + Ray + vLLM + DeepSpeed"

  # 这是关键阶段：
  # - Ray 负责多角色调度
  # - OpenRLHF 的 train_ppo_ray 负责 RLHF workflow orchestration
  # - vLLM engines 用来加速 rollout generation
  # - DeepSpeed 负责 actor/critic 的训练内存与优化
  #
  # 你最容易犯的概念错误：
  # - 以为自己要单独再写一个 vLLM server 给 OpenRLHF PPO 用
  # - 不一定。OpenRLHF 的 PPO Ray 模式本身就支持 vLLM engines
  # - 单独的 vLLM serve 更适合评测、独立 rollout 服务或部署

  ray job submit --address="http://${RAY_HEAD_ADDR}:${RAY_DASHBOARD_PORT}" \
    --runtime-env-json="{\"working_dir\": \"${PROJECT_ROOT}\"}" \
    -- python3 -m openrlhf.cli.train_ppo_ray \
    --pretrain "${SFT_CKPT}" \
    --reward_pretrain "${RM_CKPT}" \
    --save_path "${PPO_CKPT}" \
    --prompt_data "${PPO_PROMPT_DATASET}" \
    --input_key context_messages \
    --apply_chat_template \
    --prompt_max_len "${PROMPT_MAX_LEN}" \
    --generate_max_len "${GENERATE_MAX_LEN}" \
    --max_samples "${MAX_SAMPLES}" \
    --max_epochs "${MAX_EPOCHS}" \
    --zero_stage "${PPO_ZERO_STAGE}" \
    --param_dtype "${PARAM_DTYPE}" \
    --actor_learning_rate "${ACTOR_LR}" \
    --critic_learning_rate "${CRITIC_LR}" \
    --init_kl_coef "${INIT_KL_COEF}" \
    --train_batch_size "${PPO_TRAIN_BATCH_SIZE}" \
    --micro_train_batch_size "${PPO_MICRO_TRAIN_BATCH_SIZE}" \
    --rollout_batch_size "${PPO_ROLLOUT_BATCH_SIZE}" \
    --micro_rollout_batch_size "${PPO_MICRO_ROLLOUT_BATCH_SIZE}" \
    --normalize_reward \
    --gradient_checkpointing \
    --packing_samples \
    --vllm_num_engines "${VLLM_NUM_ENGINES}" \
    --vllm_tensor_parallel_size "${VLLM_TP_SIZE}" \
    --vllm_sync_backend nccl \
    --ref_num_nodes "${REF_NUM_NODES}" \
    --ref_num_gpus_per_node "${REF_NUM_GPUS_PER_NODE}" \
    --reward_num_nodes "${REWARD_NUM_NODES}" \
    --reward_num_gpus_per_node "${REWARD_NUM_GPUS_PER_NODE}" \
    --critic_num_nodes "${CRITIC_NUM_NODES}" \
    --critic_num_gpus_per_node "${CRITIC_NUM_GPUS_PER_NODE}" \
    --actor_num_nodes "${ACTOR_NUM_NODES}" \
    --actor_num_gpus_per_node "${ACTOR_NUM_GPUS_PER_NODE}" \
    ${WANDB_API_KEY:+--use_wandb "${WANDB_API_KEY}"} \
    2>&1 | tee "${LOG_ROOT}/train_ppo.log"
}


# ==============================
# 8. 可选：PPO with custom reward function
# ==============================

function train_ppo_with_custom_reward() {
  print_banner "[8/8] PPO with custom reward function (optional)"

  # 当你不想只用 reward_pretrain checkpoint，或者想混入 task-specific shaping 时，
  # 可以用 remote_rm_url / reward_func 路线。
  #
  # 这里最重要的现实提醒：
  # - rule-based reward 可以做补充，但通常不该假装替代真正 RM。
  # - 如果 reward 太脆弱，你得到的不是“更聪明的模型”，而是“更会钻空子的模型”。

  ray job submit --address="http://${RAY_HEAD_ADDR}:${RAY_DASHBOARD_PORT}" \
    --runtime-env-json="{\"working_dir\": \"${PROJECT_ROOT}\"}" \
    -- python3 -m openrlhf.cli.train_ppo_ray \
    --pretrain "${SFT_CKPT}" \
    --save_path "${PPO_CKPT}-custom-reward" \
    --prompt_data "${PPO_PROMPT_DATASET}" \
    --input_key context_messages \
    --apply_chat_template \
    --prompt_max_len "${PROMPT_MAX_LEN}" \
    --generate_max_len "${GENERATE_MAX_LEN}" \
    --max_samples "${MAX_SAMPLES}" \
    --max_epochs "${MAX_EPOCHS}" \
    --zero_stage "${PPO_ZERO_STAGE}" \
    --param_dtype "${PARAM_DTYPE}" \
    --actor_learning_rate "${ACTOR_LR}" \
    --critic_learning_rate "${CRITIC_LR}" \
    --init_kl_coef "${INIT_KL_COEF}" \
    --train_batch_size "${PPO_TRAIN_BATCH_SIZE}" \
    --micro_train_batch_size "${PPO_MICRO_TRAIN_BATCH_SIZE}" \
    --rollout_batch_size "${PPO_ROLLOUT_BATCH_SIZE}" \
    --micro_rollout_batch_size "${PPO_MICRO_ROLLOUT_BATCH_SIZE}" \
    --normalize_reward \
    --gradient_checkpointing \
    --packing_samples \
    --remote_rm_url "${PROJECT_ROOT}/examples/reward_func.py" \
    --label_key answer \
    --vllm_num_engines "${VLLM_NUM_ENGINES}" \
    --vllm_tensor_parallel_size "${VLLM_TP_SIZE}" \
    --vllm_sync_backend nccl \
    --ref_num_nodes "${REF_NUM_NODES}" \
    --ref_num_gpus_per_node "${REF_NUM_GPUS_PER_NODE}" \
    --reward_num_nodes 0 \
    --reward_num_gpus_per_node 0 \
    --critic_num_nodes "${CRITIC_NUM_NODES}" \
    --critic_num_gpus_per_node "${CRITIC_NUM_GPUS_PER_NODE}" \
    --actor_num_nodes "${ACTOR_NUM_NODES}" \
    --actor_num_gpus_per_node "${ACTOR_NUM_GPUS_PER_NODE}" \
    ${WANDB_API_KEY:+--use_wandb "${WANDB_API_KEY}"} \
    2>&1 | tee "${LOG_ROOT}/train_ppo_custom_reward.log"
}


# ==============================
# 9. 可选：独立 vLLM server 用于评测/人工抽样
# ==============================

function print_vllm_eval_hint() {
  print_banner "[Optional] Standalone vLLM serving hint"
  cat <<EOF
# OpenRLHF PPO 训练时通常直接调度 vLLM engines，不一定需要额外手工启动一个 vLLM server。
# 但如果你想单独做评测、抽样、人工检查，可以额外起一个 OpenAI-compatible server：
#
# vllm serve ${PPO_CKPT} \
#   --dtype auto \
#   --api-key token-abc123 \
#   --generation-config vllm
#
# 然后用 OpenAI Python client 连：
# base_url=http://localhost:8000/v1
# api_key=token-abc123
EOF
}


# ==============================
# 10. 命令分发
# ==============================

function usage() {
  cat <<'EOF'
Usage:
  bash train_openrlhf_ppo_stack.sh check
  bash train_openrlhf_ppo_stack.sh install_hint
  bash train_openrlhf_ppo_stack.sh sft
  bash train_openrlhf_ppo_stack.sh rm
  bash train_openrlhf_ppo_stack.sh reward_func
  bash train_openrlhf_ppo_stack.sh ray
  bash train_openrlhf_ppo_stack.sh ppo
  bash train_openrlhf_ppo_stack.sh ppo_custom_reward
  bash train_openrlhf_ppo_stack.sh vllm_hint
  bash train_openrlhf_ppo_stack.sh all

Recommended order:
  1) check
  2) reward_func   (optional)
  3) sft
  4) rm
  5) ray
  6) ppo
EOF
}

CMD="${1:-}"

case "${CMD}" in
  check)
    check_env
    ;;
  install_hint)
    print_install_hint
    ;;
  sft)
    check_env
    train_sft
    ;;
  rm)
    check_env
    train_rm
    ;;
  reward_func)
    write_custom_reward_func
    ;;
  ray)
    check_env
    start_ray_head
    ;;
  ppo)
    check_env
    train_ppo
    ;;
  ppo_custom_reward)
    check_env
    train_ppo_with_custom_reward
    ;;
  vllm_hint)
    print_vllm_eval_hint
    ;;
  all)
    check_env
    write_custom_reward_func
    train_sft
    train_rm
    start_ray_head
    train_ppo
    print_vllm_eval_hint
    ;;
  *)
    usage
    exit 1
    ;;
esac
