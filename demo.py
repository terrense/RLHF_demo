"""
一个“教学级、可运行、单文件”的 RLHF + PPO 示例。

你别自欺欺人：
- 这不是 production-ready 的万亿参数 LLM 训练系统。
- 但它把 PPO 在 RLHF 里的核心数据流、量的定义、时间顺序、teacher-forced update、GAE、KL reward shaping，
  全部从代码层面打通了。
- 你如果把这个文件真正读透、跑通、改明白，你对 PPO-RLHF 的理解会比只背公式强得多。

这个脚本演示的完整流程：
1) 先做一个很小的 SFT warm start（模拟真实 RLHF 里“先 SFT，再 PPO”）
2) 冻结一份 reference model
3) 使用一个“冻结的 reward model”（这里用 rule-based RM 代替训练好的神经 RM）
4) rollout：旧策略逐 token 采样 response，记录 old_logprobs / values / masks
5) reference 打出 ref_logprobs，构造 per-token KL penalty
6) RM 给整段 response 打分，并把序列级 reward 加到“最后一个有效 response token”上
7) 用 rewards + values 计算 GAE advantages 和 returns
8) 用 PPO clip loss + value loss 更新 policy/value
9) 反复 rollout -> update
10) 最后只保留 policy 做推理

这个 toy 任务是什么？
- prompt 只有三种：问你回答 A / B / C
- 理想回答是：对应字母 + PLEASE + EOS
- reward model 会奖励“回答正确、礼貌、别太啰嗦”
- PPO 的目标就是让 policy 学会在 reference 约束下，更稳定地产出高 reward response

为什么不用真正大模型？
- 因为你现在缺的不是显卡烧钱，而是对 PPO 数据流的可解释理解。
- 真大模型的难点主要是：并行、KV cache、packing、distributed optimizer、mixed precision、padding side、
  reward model batching、serving/inference infra。这些是工程放大问题，不是 PPO 数学本体。

怎么运行：
python rlhf_ppo_toy_full_walkthrough.py

依赖：
- Python 3.10+
- PyTorch 2.x

建议：
先用 CPU 跑通。你现在最不该做的，是还没懂就急着上多卡框架。
"""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


# ============================================================
# 0. 基础工具
# ============================================================


def now_str() -> str:
    """返回简单的时间戳字符串，用于日志打印。"""
    return datetime.now().strftime("%H:%M:%S")


def log(msg: str) -> None:
    print(f"[{now_str()}] {msg}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# 1. 配置区
# ============================================================


@dataclass
class Config:
    # 设备
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42

    # 模型尺寸（故意很小，便于教学和 CPU 跑通）
    vocab_size: int = 12
    d_model: int = 64
    hidden_size: int = 96

    # prompt / response 长度
    prompt_len: int = 2
    max_new_tokens: int = 4

    # 训练阶段：SFT warm start
    sft_steps: int = 250
    sft_batch_size: int = 64
    sft_lr: float = 3e-3

    # PPO 训练
    ppo_updates: int = 120
    rollout_batch_size: int = 64
    ppo_epochs: int = 4
    mini_batch_size: int = 16
    ppo_lr: float = 2e-3
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 1.0

    # GAE / return
    gamma: float = 1.0        # LLM RLHF 里常见地会取 1.0
    gae_lambda: float = 0.95

    # KL control
    init_kl_coef: float = 0.08
    target_kl: float = 0.12
    min_kl_coef: float = 0.005
    max_kl_coef: float = 1.0

    # value clipping（很多 PPO 实现会做）
    value_clip_range: float = 0.2
    use_value_clip: bool = True

    # rollout 采样策略
    rollout_temperature: float = 1.0

    # 打印 / 评估
    print_every_sft: int = 25
    print_every_ppo: int = 5


# ============================================================
# 2. 一个极小 tokenizer / vocab
# ============================================================


class TinyVocab:
    """
    我们不用真正 tokenizer，直接手工定义一个极小词表。

    token 含义：
    0: PAD
    1: BOS
    2: EOS
    3: QUERY_A   -> prompt: "请回答 A"
    4: QUERY_B   -> prompt: "请回答 B"
    5: QUERY_C   -> prompt: "请回答 C"
    6: A
    7: B
    8: C
    9: PLEASE
    10: THANKS
    11: NOISE
    """

    def __init__(self) -> None:
        self.id_to_token: Dict[int, str] = {
            0: "<PAD>",
            1: "<BOS>",
            2: "<EOS>",
            3: "QUERY_A",
            4: "QUERY_B",
            5: "QUERY_C",
            6: "A",
            7: "B",
            8: "C",
            9: "PLEASE",
            10: "THANKS",
            11: "NOISE",
        }
        self.token_to_id: Dict[str, int] = {v: k for k, v in self.id_to_token.items()}

        self.pad_id = self.token_to_id["<PAD>"]
        self.bos_id = self.token_to_id["<BOS>"]
        self.eos_id = self.token_to_id["<EOS>"]

        self.query_to_answer = {
            self.token_to_id["QUERY_A"]: self.token_to_id["A"],
            self.token_to_id["QUERY_B"]: self.token_to_id["B"],
            self.token_to_id["QUERY_C"]: self.token_to_id["C"],
        }

    def decode_ids(self, ids: List[int]) -> str:
        tokens = []
        for x in ids:
            if x == self.pad_id:
                continue
            tokens.append(self.id_to_token[int(x)])
        return " ".join(tokens)

    def decode_response_until_stop(self, ids: List[int]) -> str:
        tokens = []
        for x in ids:
            if x == self.pad_id:
                continue
            tokens.append(self.id_to_token[int(x)])
            if x == self.eos_id:
                break
        return " ".join(tokens)


# ============================================================
# 3. toy 数据集：固定长度 prompt，便于你聚焦 PPO 本身
# ============================================================


class ToyPromptDataset:
    """
    prompt 固定格式：
    [BOS, QUERY_X]

    理想 SFT response 固定为：
    [X, PLEASE, EOS, PAD]

    这是故意设计成最小可解释例子。
    真正 LLM 里当然是自然语言 token 序列，而不是这种玩具 token。
    """

    def __init__(self, vocab: TinyVocab, cfg: Config, device: str) -> None:
        self.vocab = vocab
        self.cfg = cfg
        self.device = device
        self.query_tokens = [
            vocab.token_to_id["QUERY_A"],
            vocab.token_to_id["QUERY_B"],
            vocab.token_to_id["QUERY_C"],
        ]

    def sample_prompts(self, batch_size: int) -> torch.Tensor:
        query_ids = torch.tensor(
            random.choices(self.query_tokens, k=batch_size),
            dtype=torch.long,
            device=self.device,
        )
        bos = torch.full((batch_size, 1), self.vocab.bos_id, dtype=torch.long, device=self.device)
        queries = query_ids.unsqueeze(1)
        prompts = torch.cat([bos, queries], dim=1)  # [B, 2]
        return prompts

    def build_sft_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prompts = self.sample_prompts(batch_size)

        responses = torch.full(
            (batch_size, self.cfg.max_new_tokens),
            self.vocab.pad_id,
            dtype=torch.long,
            device=self.device,
        )
        response_mask = torch.zeros_like(responses, dtype=torch.float32)

        for i in range(batch_size):
            query_id = int(prompts[i, 1].item())
            answer_id = self.vocab.query_to_answer[query_id]
            target = [answer_id, self.vocab.token_to_id["PLEASE"], self.vocab.eos_id]
            responses[i, : len(target)] = torch.tensor(target, dtype=torch.long, device=self.device)
            response_mask[i, : len(target)] = 1.0

        return prompts, responses, response_mask


# ============================================================
# 4. 模型：一个极小 GRU-based causal LM + value head
# ============================================================


class TinyCausalLMWithValue(nn.Module):
    """
    你别被“GRU”吓到或者嫌弃。
    这里我们关心的是 RLHF-PPO 数据流，不是 Transformer 架构炫技。

    输出：
    - logits: [B, L, V]
    - values: [B, L]

    values[:, t] 表示“看到第 t 个输入位置后，对当前状态的价值估计”。
    在生成第一个 response token 时，我们会取最后一个 prompt 位置的 value 作为 V(s_1)。
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.rnn = nn.GRU(
            input_size=cfg.d_model,
            hidden_size=cfg.hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size)
        self.value_head = nn.Linear(cfg.hidden_size, 1)

    def forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        input_ids: [B, L]
        returns:
            logits: [B, L, V]
            values: [B, L]
        """
        x = self.embed(input_ids)               # [B, L, d_model]
        h, _ = self.rnn(x)                      # [B, L, hidden]
        logits = self.lm_head(h)               # [B, L, vocab]
        values = self.value_head(h).squeeze(-1)  # [B, L]
        return logits, values


# ============================================================
# 5. Reward Model：这里用 rule-based frozen RM 代替训练好的神经 RM
# ============================================================


class RuleBasedRewardModel:
    """
    真正 RLHF 里，RM 一般是训练出来的模型（常见是 pairwise preference 数据训练）。
    这里我们用一个“冻结的规则奖励器”代替，目的只是把 PPO 数据流讲清楚。

    奖励设计：
    - 第一 response token 回答正确：+2.0，否则 -2.0
    - 出现 PLEASE：+0.5
    - 出现 THANKS：+0.2
    - 出现 NOISE：-0.5
    - 太啰嗦：每多一个非 EOS token，额外 -0.15
    - 如果没有在 max_new_tokens 内结束（没有 EOS）：-0.3
    """

    def __init__(self, vocab: TinyVocab):
        self.vocab = vocab

    def _extract_valid_response(self, response_row: torch.Tensor) -> List[int]:
        tokens: List[int] = []
        for x in response_row.tolist():
            if x == self.vocab.pad_id:
                continue
            tokens.append(int(x))
            if x == self.vocab.eos_id:
                break
        return tokens

    @torch.no_grad()
    def score(self, prompts: torch.Tensor, responses: torch.Tensor) -> torch.Tensor:
        """
        prompts:   [B, prompt_len]
        responses: [B, max_new_tokens]
        returns sequence-level scalar reward: [B]
        """
        batch_size = prompts.size(0)
        scores = torch.zeros(batch_size, device=prompts.device, dtype=torch.float32)

        for i in range(batch_size):
            query_id = int(prompts[i, 1].item())
            gold_answer = self.vocab.query_to_answer[query_id]
            resp = self._extract_valid_response(responses[i])

            if len(resp) == 0:
                scores[i] = -2.5
                continue

            # 1) 第一 token 是否答对
            if resp[0] == gold_answer:
                scores[i] += 2.0
            else:
                scores[i] -= 2.0

            # 2) 礼貌 token
            if self.vocab.token_to_id["PLEASE"] in resp:
                scores[i] += 0.5
            if self.vocab.token_to_id["THANKS"] in resp:
                scores[i] += 0.2

            # 3) 噪声 token
            if self.vocab.token_to_id["NOISE"] in resp:
                scores[i] -= 0.5

            # 4) 长度惩罚：去掉 EOS，只看实际生成内容长度
            content_tokens = [t for t in resp if t != self.vocab.eos_id]
            if len(content_tokens) > 2:
                scores[i] -= 0.15 * (len(content_tokens) - 2)

            # 5) 没有 EOS 的惩罚
            if self.vocab.eos_id not in resp:
                scores[i] -= 0.3

        return scores


# ============================================================
# 6. 核心工具函数：logprob / entropy / mask / flatten
# ============================================================


def gather_logprobs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    logits: [B, L, V]
    labels: [B, L]
    return: 每个位置 target token 的 logprob, shape [B, L]
    """
    log_probs = F.log_softmax(logits, dim=-1)
    return log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)



def token_entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """
    logits: [B, L, V]
    return entropy per token: [B, L]
    """
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)
    return entropy



def masked_mean(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return (x * mask).sum() / (mask.sum() + eps)



def whiten_masked(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    valid = mask.bool()
    if valid.sum() == 0:
        return x
    flat = x[valid]
    mean = flat.mean()
    std = flat.std(unbiased=False)
    out = x.clone()
    out[valid] = (flat - mean) / (std + eps)
    return out



def explained_variance(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    valid = mask.bool()
    if valid.sum() < 2:
        return 0.0
    y = target[valid]
    y_hat = pred[valid]
    var_y = torch.var(y, unbiased=False)
    if var_y.item() < 1e-12:
        return 0.0
    return (1.0 - torch.var(y - y_hat, unbiased=False) / var_y).item()


# ============================================================
# 7. teacher-forced 计算：给定 prompt + response，重新算当前模型的 logprobs / values
# ============================================================


@torch.no_grad()
def compute_ref_logprobs(
    ref_model: TinyCausalLMWithValue,
    prompts: torch.Tensor,
    responses: torch.Tensor,
    prompt_len: int,
) -> torch.Tensor:
    """
    reference model 是冻结的，只需要 logprobs，不需要 values。

    关键点：
    PPO update 阶段不是重新采样，而是对“已采样好的 response”做 teacher-forced forward。

    设 combined = [prompt, response]
    那么第一个 response token 的概率，来自“看完 prompt 最后一个位置”后的预测。

    返回：response 区间每个 token 的 logprob, shape [B, T_resp]
    """
    combined = torch.cat([prompts, responses], dim=1)         # [B, P+T]
    model_input = combined[:, :-1]                            # [B, P+T-1]
    target = combined[:, 1:]                                  # [B, P+T-1]

    logits, _ = ref_model(model_input)
    all_logprobs = gather_logprobs(logits, target)            # [B, P+T-1]

    t_resp = responses.size(1)
    response_logprobs = all_logprobs[:, prompt_len - 1 : prompt_len - 1 + t_resp]
    return response_logprobs



def compute_policy_logprobs_values_entropy(
    policy: TinyCausalLMWithValue,
    prompts: torch.Tensor,
    responses: torch.Tensor,
    prompt_len: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    返回当前 policy 在 response 区间上的：
    - token logprobs: [B, T]
    - values:        [B, T]
    - entropy:       [B, T]
    """
    combined = torch.cat([prompts, responses], dim=1)         # [B, P+T]
    model_input = combined[:, :-1]                            # [B, P+T-1]
    target = combined[:, 1:]                                  # [B, P+T-1]

    logits, values = policy(model_input)                      # logits [B, P+T-1, V], values [B, P+T-1]
    all_logprobs = gather_logprobs(logits, target)            # [B, P+T-1]
    all_entropy = token_entropy_from_logits(logits)           # [B, P+T-1]

    t_resp = responses.size(1)
    start = prompt_len - 1
    end = start + t_resp

    response_logprobs = all_logprobs[:, start:end]            # [B, T]
    response_values = values[:, start:end]                    # [B, T]
    response_entropy = all_entropy[:, start:end]              # [B, T]
    return response_logprobs, response_values, response_entropy


# ============================================================
# 8. rollout：旧策略逐 token 采样，记录 old_logprobs / values / mask
# ============================================================


@torch.no_grad()
def generate_rollout(
    policy: TinyCausalLMWithValue,
    prompts: torch.Tensor,
    cfg: Config,
    vocab: TinyVocab,
    temperature: float,
    print_shapes_once: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    这是 RLHF-PPO 的 rollout 阶段。
    注意：这里只做前向采样，不做梯度更新。

    记录的关键量：
    - responses:     [B, T]
    - old_logprobs:  [B, T]    <-- rollout 时旧策略对 sampled token 的 logprob
    - values:        [B, T]    <-- critic 对每一步状态的价值估计 V(s_t)
    - mask:          [B, T]    <-- 哪些 response token 有效（EOS 后无效）

    这里每一步都重新喂完整前缀，效率不高，但逻辑最直白。
    真正大模型里会用 KV cache，避免重复算历史前缀。
    """
    device = prompts.device
    batch_size = prompts.size(0)
    max_new_tokens = cfg.max_new_tokens

    responses = torch.full(
        (batch_size, max_new_tokens),
        vocab.pad_id,
        dtype=torch.long,
        device=device,
    )
    old_logprobs = torch.zeros(batch_size, max_new_tokens, device=device)
    values = torch.zeros(batch_size, max_new_tokens, device=device)
    mask = torch.zeros(batch_size, max_new_tokens, device=device)

    current = prompts.clone()                  # 当前前缀，初始就是 prompt
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    for t in range(max_new_tokens):
        logits, value_seq = policy(current)
        next_logits = logits[:, -1, :]        # [B, V]，当前状态下“下一个 token”的分布
        state_values = value_seq[:, -1]       # [B]，当前状态价值 V(s_t)

        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        dist = Categorical(logits=next_logits / temperature)
        next_token = dist.sample()            # [B]
        next_logprob = dist.log_prob(next_token)  # [B]

        # 已经 finished 的样本，后续 token 全部设成 PAD，mask=0
        next_token = torch.where(finished, torch.full_like(next_token, vocab.pad_id), next_token)
        next_logprob = torch.where(finished, torch.zeros_like(next_logprob), next_logprob)
        state_values = torch.where(finished, torch.zeros_like(state_values), state_values)
        valid = (~finished).float()

        responses[:, t] = next_token
        old_logprobs[:, t] = next_logprob
        values[:, t] = state_values
        mask[:, t] = valid

        current = torch.cat([current, next_token.unsqueeze(1)], dim=1)
        finished = finished | (next_token == vocab.eos_id)

    if print_shapes_once:
        log("[Shape Check] rollout tensors:")
        log(f"  prompts.shape      = {tuple(prompts.shape)}")
        log(f"  responses.shape    = {tuple(responses.shape)}")
        log(f"  old_logprobs.shape = {tuple(old_logprobs.shape)}")
        log(f"  values.shape       = {tuple(values.shape)}")
        log(f"  mask.shape         = {tuple(mask.shape)}")

    return {
        "responses": responses,
        "old_logprobs": old_logprobs,
        "values": values,
        "mask": mask,
    }


# ============================================================
# 9. reward shaping：RM 分数 + per-token KL penalty
# ============================================================


@torch.no_grad()
def build_token_level_rewards(
    prompts: torch.Tensor,
    responses: torch.Tensor,
    old_logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
    mask: torch.Tensor,
    reward_model: RuleBasedRewardModel,
    kl_coef: float,
) -> Dict[str, torch.Tensor]:
    """
    这一步是 RLHF PPO 最容易被说糊涂的地方。

    我们不是只拿一个 sequence reward 就完事。
    更常见的做法是：

    1) 每个 token 都有一个 KL penalty
       kl_t = logpi_old(a_t|s_t) - logpi_ref(a_t|s_t)
       non_score_reward_t = - beta * kl_t

    2) reward model 给整个 response 一个 scalar 分数 rm_score
       然后把它加到“最后一个有效 token”上

    于是最终：
    - 前面的 token 一般只有 KL reward
    - 最后一个有效 token = KL reward + RM score

    这比“只有最后一步有奖励，其余全是 0”更贴近 RLHF 实践。
    """
    # sampled-token KL proxy
    token_kl = (old_logprobs - ref_logprobs) * mask

    # 每个 token 的非任务奖励（KL 惩罚）
    non_score_reward = -kl_coef * token_kl

    # sequence-level RM reward
    rm_score = reward_model.score(prompts, responses)  # [B]

    rewards = non_score_reward.clone()

    # 把 sequence-level reward 加到最后一个有效 response token 上
    lengths = mask.sum(dim=1).long()   # [B]
    for i in range(prompts.size(0)):
        if lengths[i].item() <= 0:
            continue
        last_idx = lengths[i].item() - 1
        rewards[i, last_idx] += rm_score[i]

    return {
        "token_kl": token_kl,
        "non_score_reward": non_score_reward,
        "rm_score": rm_score,
        "rewards": rewards,
    }


# ============================================================
# 10. GAE + returns
# ============================================================


@torch.no_grad()
def compute_gae_and_returns(
    rewards: torch.Tensor,
    values: torch.Tensor,
    mask: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    rewards: [B, T]
    values:  [B, T]
    mask:    [B, T]

    返回：
    - advantages: [B, T]
    - returns:    [B, T]

    公式：
    delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
    A_t = delta_t + gamma * lambda * A_{t+1}
    G_t = A_t + V(s_t)

    注意：
    next state 是否存在，不是靠你“猜未来”，而是 rollout 完整结束后，
    你已经拿到了整条轨迹，于是可以从后往前做 credit assignment。
    这叫 post-hoc credit assignment，不是算命。
    """
    batch_size, T = rewards.shape
    advantages = torch.zeros_like(rewards)
    lastgaelam = torch.zeros(batch_size, device=rewards.device)

    for t in reversed(range(T)):
        if t == T - 1:
            next_values = torch.zeros(batch_size, device=rewards.device)
            next_nonterminal = torch.zeros(batch_size, device=rewards.device)
        else:
            next_values = values[:, t + 1]
            next_nonterminal = mask[:, t + 1]

        delta = rewards[:, t] + gamma * next_values * next_nonterminal - values[:, t]
        lastgaelam = delta + gamma * gae_lambda * next_nonterminal * lastgaelam
        advantages[:, t] = lastgaelam

    advantages = advantages * mask
    returns = (advantages + values) * mask
    return advantages, returns


# ============================================================
# 11. PPO update
# ============================================================


@dataclass
class PPOStepStats:
    actor_loss: float
    critic_loss: float
    entropy: float
    total_loss: float
    approx_kl: float
    clipfrac: float



def ppo_update(
    policy: TinyCausalLMWithValue,
    optimizer: torch.optim.Optimizer,
    prompts: torch.Tensor,
    responses: torch.Tensor,
    old_logprobs: torch.Tensor,
    old_values: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    mask: torch.Tensor,
    cfg: Config,
) -> PPOStepStats:
    """
    这是 PPO 的“模型更新阶段”。

    关键区分：
    - rollout 阶段：旧策略采样，存 old_logprobs / old_values
    - update 阶段：当前策略对“同一批 response”做 teacher-forced forward，算 new_logprobs / new_values

    然后：
    ratio = exp(new_logprob - old_logprob)

    如果你这里的 old_logprob 不是 rollout 时那份快照，而是拿更新后的模型重算出来的，
    那你就把 PPO 最核心的东西弄坏了。
    """
    batch_size = prompts.size(0)
    actor_losses: List[float] = []
    critic_losses: List[float] = []
    entropies: List[float] = []
    total_losses: List[float] = []
    approx_kls: List[float] = []
    clipfracs: List[float] = []

    # 只对白名单位置（有效 response token）做标准化
    advantages = whiten_masked(advantages, mask)

    for epoch in range(cfg.ppo_epochs):
        perm = torch.randperm(batch_size, device=prompts.device)

        for start in range(0, batch_size, cfg.mini_batch_size):
            idx = perm[start : start + cfg.mini_batch_size]

            mb_prompts = prompts[idx]
            mb_responses = responses[idx]
            mb_old_logprobs = old_logprobs[idx]
            mb_old_values = old_values[idx]
            mb_advantages = advantages[idx]
            mb_returns = returns[idx]
            mb_mask = mask[idx]

            # 当前 policy 对固定 response 做 teacher-forced forward
            new_logprobs, new_values, entropy = compute_policy_logprobs_values_entropy(
                policy,
                mb_prompts,
                mb_responses,
                cfg.prompt_len,
            )

            valid = mb_mask.bool()

            # PPO actor ratio：只针对 sampled token 这一项，不是全 vocab 分布 ratio
            ratio = torch.exp(new_logprobs - mb_old_logprobs)

            surr1 = ratio * mb_advantages
            surr2 = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * mb_advantages
            actor_loss = -torch.min(surr1, surr2)[valid].mean()

            # value loss
            if cfg.use_value_clip:
                value_pred_clipped = mb_old_values + (new_values - mb_old_values).clamp(
                    -cfg.value_clip_range, cfg.value_clip_range
                )
                vf_losses1 = (new_values - mb_returns) ** 2
                vf_losses2 = (value_pred_clipped - mb_returns) ** 2
                critic_loss = 0.5 * torch.max(vf_losses1, vf_losses2)[valid].mean()
            else:
                critic_loss = 0.5 * ((new_values - mb_returns) ** 2)[valid].mean()

            entropy_loss = entropy[valid].mean()

            total_loss = actor_loss + cfg.vf_coef * critic_loss - cfg.ent_coef * entropy_loss

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                approx_kl = (mb_old_logprobs[valid] - new_logprobs[valid]).mean().item()
                clipfrac = ((ratio[valid] - 1.0).abs() > cfg.clip_eps).float().mean().item()

            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())
            entropies.append(entropy_loss.item())
            total_losses.append(total_loss.item())
            approx_kls.append(approx_kl)
            clipfracs.append(clipfrac)

    return PPOStepStats(
        actor_loss=sum(actor_losses) / len(actor_losses),
        critic_loss=sum(critic_losses) / len(critic_losses),
        entropy=sum(entropies) / len(entropies),
        total_loss=sum(total_losses) / len(total_losses),
        approx_kl=sum(approx_kls) / len(approx_kls),
        clipfrac=sum(clipfracs) / len(clipfracs),
    )


# ============================================================
# 12. SFT warm start
# ============================================================


def sft_train(
    policy: TinyCausalLMWithValue,
    dataset: ToyPromptDataset,
    vocab: TinyVocab,
    cfg: Config,
) -> None:
    """
    先做一点 SFT warm start，模拟真实 RLHF 里“先 SFT 再 PPO”。

    这是非常重要的现实经验：
    - 没有 decent warm start，PPO 很容易学得又慢又不稳。
    - 大模型实践里，reference model 通常就是 SFT checkpoint 的 frozen copy。
    """
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.sft_lr)
    policy.train()

    for step in range(1, cfg.sft_steps + 1):
        prompts, responses, response_mask = dataset.build_sft_batch(cfg.sft_batch_size)

        # teacher forcing
        _, _, _ = response_mask.shape if False else (None, None, None)  # 只是提醒你这里有 response mask，不要忘了它
        new_logprobs, _, _ = compute_policy_logprobs_values_entropy(policy, prompts, responses, cfg.prompt_len)

        # SFT NLL = -log p(target)
        loss_per_token = -new_logprobs
        loss = masked_mean(loss_per_token, response_mask)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
        optimizer.step()

        if step % cfg.print_every_sft == 0 or step == 1:
            log(f"[SFT] step={step:04d} loss={loss.item():.4f}")

    policy.eval()


# ============================================================
# 13. 推理 / 评估工具
# ============================================================


@torch.no_grad()
def greedy_generate(
    policy: TinyCausalLMWithValue,
    prompts: torch.Tensor,
    cfg: Config,
    vocab: TinyVocab,
) -> torch.Tensor:
    """部署阶段更接近 greedy / sampling 推理；这里只做 greedy 方便展示。"""
    current = prompts.clone()
    batch_size = prompts.size(0)
    responses = torch.full(
        (batch_size, cfg.max_new_tokens),
        vocab.pad_id,
        dtype=torch.long,
        device=prompts.device,
    )
    finished = torch.zeros(batch_size, dtype=torch.bool, device=prompts.device)

    for t in range(cfg.max_new_tokens):
        logits, _ = policy(current)
        next_token = logits[:, -1, :].argmax(dim=-1)
        next_token = torch.where(finished, torch.full_like(next_token, vocab.pad_id), next_token)
        responses[:, t] = next_token
        current = torch.cat([current, next_token.unsqueeze(1)], dim=1)
        finished = finished | (next_token == vocab.eos_id)

    return responses


@torch.no_grad()
def print_samples(
    policy: TinyCausalLMWithValue,
    vocab: TinyVocab,
    cfg: Config,
    device: str,
    title: str,
) -> None:
    policy.eval()
    prompts = torch.tensor(
        [
            [vocab.bos_id, vocab.token_to_id["QUERY_A"]],
            [vocab.bos_id, vocab.token_to_id["QUERY_B"]],
            [vocab.bos_id, vocab.token_to_id["QUERY_C"]],
        ],
        dtype=torch.long,
        device=device,
    )
    responses = greedy_generate(policy, prompts, cfg, vocab)

    log(title)
    for i in range(prompts.size(0)):
        prompt_txt = vocab.decode_ids(prompts[i].tolist())
        resp_txt = vocab.decode_response_until_stop(responses[i].tolist())
        print(f"  prompt   : {prompt_txt}")
        print(f"  response : {resp_txt}")
        print("  -")


# ============================================================
# 14. adaptive KL control
# ============================================================


def adjust_kl_coef(current_kl_coef: float, measured_kl: float, cfg: Config) -> float:
    """
    一个很粗但很实用的 adaptive KL controller。

    真实工程里：
    - KL 太高：说明 policy 偏离 reference 太猛，容易 reward hacking / 训崩
    - KL 太低：说明 policy 被 reference 绑太死，PPO 基本没学到新行为

    所以 beta / kl_coef 往往不是一成不变的。
    """
    if cfg.target_kl <= 0:
        return current_kl_coef

    new_coef = current_kl_coef
    if measured_kl > 1.5 * cfg.target_kl:
        new_coef *= 1.5
    elif measured_kl < 0.5 * cfg.target_kl:
        new_coef /= 1.5

    new_coef = max(cfg.min_kl_coef, min(cfg.max_kl_coef, new_coef))
    return float(new_coef)


# ============================================================
# 15. 主训练流程：SFT -> freeze ref -> PPO rollout/update
# ============================================================


def train_rlhf_ppo() -> None:
    cfg = Config()
    set_seed(cfg.seed)
    device = cfg.device
    vocab = TinyVocab()
    dataset = ToyPromptDataset(vocab, cfg, device)

    log(f"device = {device}")
    log("Initialize policy / reward model / dataset")

    policy = TinyCausalLMWithValue(cfg).to(device)
    reward_model = RuleBasedRewardModel(vocab)

    # 1) 先做 SFT warm start
    log("Start SFT warm start")
    print_samples(policy, vocab, cfg, device, title="[Before SFT] policy samples")
    sft_train(policy, dataset, vocab, cfg)
    print_samples(policy, vocab, cfg, device, title="[After SFT] policy samples")

    # 2) 拷贝 frozen reference
    reference = copy.deepcopy(policy).to(device)
    reference.eval()
    for p in reference.parameters():
        p.requires_grad = False

    # 3) PPO optimizer
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.ppo_lr)
    kl_coef = cfg.init_kl_coef

    log("Start PPO RLHF training")

    for update_idx in range(1, cfg.ppo_updates + 1):
        policy.eval()

        # ----------------------------------------------------
        # τ0: sample prompt batch
        # ----------------------------------------------------
        prompts = dataset.sample_prompts(cfg.rollout_batch_size)

        # ----------------------------------------------------
        # τ1: rollout with old policy (no grad)
        # ----------------------------------------------------
        rollout = generate_rollout(
            policy=policy,
            prompts=prompts,
            cfg=cfg,
            vocab=vocab,
            temperature=cfg.rollout_temperature,
            print_shapes_once=(update_idx == 1),
        )
        responses = rollout["responses"]
        old_logprobs = rollout["old_logprobs"]
        old_values = rollout["values"]
        mask = rollout["mask"]

        # ----------------------------------------------------
        # τ2: frozen reference logprobs
        # ----------------------------------------------------
        ref_logprobs = compute_ref_logprobs(reference, prompts, responses, cfg.prompt_len)

        # ----------------------------------------------------
        # τ3 + τ4: RM scalar reward + token-level KL shaping
        # ----------------------------------------------------
        reward_pack = build_token_level_rewards(
            prompts=prompts,
            responses=responses,
            old_logprobs=old_logprobs,
            ref_logprobs=ref_logprobs,
            mask=mask,
            reward_model=reward_model,
            kl_coef=kl_coef,
        )
        rewards = reward_pack["rewards"]
        rm_score = reward_pack["rm_score"]
        token_kl = reward_pack["token_kl"]
        non_score_reward = reward_pack["non_score_reward"]

        # ----------------------------------------------------
        # τ5: GAE + returns
        # ----------------------------------------------------
        advantages, returns = compute_gae_and_returns(
            rewards=rewards,
            values=old_values,
            mask=mask,
            gamma=cfg.gamma,
            gae_lambda=cfg.gae_lambda,
        )

        # rollout 侧日志（更新前）
        mean_rm = rm_score.mean().item()
        mean_kl = masked_mean(token_kl, mask).item()
        mean_non_score_reward = masked_mean(non_score_reward, mask).item()
        mean_total_reward = masked_mean(rewards, mask).item()
        ev_before = explained_variance(old_values, returns, mask)
        mean_len = mask.sum(dim=1).float().mean().item()

        # ----------------------------------------------------
        # τ6: PPO update phase
        # ----------------------------------------------------
        policy.train()
        step_stats = ppo_update(
            policy=policy,
            optimizer=optimizer,
            prompts=prompts,
            responses=responses,
            old_logprobs=old_logprobs,
            old_values=old_values,
            advantages=advantages,
            returns=returns,
            mask=mask,
            cfg=cfg,
        )
        policy.eval()

        # adaptive KL control
        kl_coef = adjust_kl_coef(kl_coef, step_stats.approx_kl, cfg)

        # ----------------------------------------------------
        # τ7: 丢弃这批 rollout，进入下一轮
        # ----------------------------------------------------
        if update_idx % cfg.print_every_ppo == 0 or update_idx == 1:
            log(
                "[PPO] "
                f"update={update_idx:04d} | "
                f"len={mean_len:.2f} | "
                f"rm={mean_rm:+.3f} | "
                f"token_kl={mean_kl:+.4f} | "
                f"non_score_reward={mean_non_score_reward:+.4f} | "
                f"reward={mean_total_reward:+.4f} | "
                f"adv_mean(valid)={advantages[mask.bool()].mean().item():+.4f} | "
                f"ret_mean(valid)={returns[mask.bool()].mean().item():+.4f} | "
                f"ev_before={ev_before:+.4f} | "
                f"actor={step_stats.actor_loss:.4f} | "
                f"critic={step_stats.critic_loss:.4f} | "
                f"entropy={step_stats.entropy:.4f} | "
                f"total={step_stats.total_loss:.4f} | "
                f"approx_kl={step_stats.approx_kl:+.4f} | "
                f"clipfrac={step_stats.clipfrac:.3f} | "
                f"kl_coef={kl_coef:.4f}"
            )
            print_samples(policy, vocab, cfg, device, title="[PPO samples]")

    log("Training done. Final policy samples:")
    print_samples(policy, vocab, cfg, device, title="[Final policy]")

    # 部署语义提醒
    log("Deployment reminder: online inference only needs the policy model.")
    log("Reference / Reward Model / GAE / PPO loss are training-time scaffolding, not inference-time components.")


# ============================================================
# 16. 关键理解总结（写在代码尾部，逼你再看一遍）
# ============================================================

"""
你必须真正记住的几点：

1) rollout 阶段无梯度
   - 旧策略采样 response
   - 存 old_logprobs / values / mask
   - 这一步不更新参数

2) reference model 是冻结的
   - 它不参与训练
   - 只负责提供 KL 约束基准

3) reward model 一般给 sequence-level scalar reward
   - 但 PPO 真正训练时经常要把它和 token-level KL reward 组合成 per-token rewards

4) GAE 不是“预知未来”
   - 是 rollout 完整结束后，对整条轨迹做 post-hoc credit assignment

5) PPO 的 ratio 是 sampled token 的概率比
   ratio_t = exp(logpi_new(a_t|s_t) - logpi_old(a_t|s_t))
   不是全 vocab 分布的 ratio

6) teacher-forced update 非常关键
   - update 阶段不是重新采样
   - 是对“已经采好的 response”重新前向，算 new_logprobs / new_values

7) deployment 只需要 policy
   - RM / ref / critic / GAE / PPO loss 都是训练时组件

8) 实战小技巧
   - 只对 response token 算 loss
   - old_logprobs 必须来自 rollout 当时的旧策略快照
   - advantage 通常要 normalize / whiten
   - KL coefficient 最好自适应，不然容易太松或太紧
   - PPO epoch 不要贪多，否则越来越 off-policy
   - 不要傻乎乎存 full logits [B,T,V]，工程上通常只存 sampled token 的 logprob/value 等必要量
"""


if __name__ == "__main__":
    train_rlhf_ppo()
