# RLHF PPO Theory Walkthrough

A single-file, runnable educational implementation of **LLM post-training with RLHF-style PPO**.

This repository is intended to make the **training-time data flow** of PPO-based RLHF explicit and inspectable. It does **not** attempt to reproduce production-scale language model training systems. Instead, it provides a compact, self-contained script that exposes the core mechanics of:

* supervised warm start
* frozen reference policy
* reward modeling
* rollout collection
* token-level KL reward shaping
* generalized advantage estimation (GAE)
* PPO policy/value updates
* adaptive KL control
* the separation between training-time components and deployment-time inference

---

<img src="111.jpg" alt="图1" width="100%" style="display:block;">
<img src="222.png" alt="图2" width="100%" style="display:block;">


Modern large language models are usually developed in multiple stages:

1. **Pretraining**
   A base model is trained with next-token prediction on large-scale corpora to acquire general linguistic and world knowledge.

2. **Supervised Fine-Tuning (SFT)**
   The pretrained model is adapted to instruction following, response formatting, and domain-specific tasks using curated prompt-response pairs.

3. **Post-Training / Alignment**
   The SFT model is further optimized to better match preference, helpfulness, harmlessness, style, or task-specific objectives.

This repository focuses on the **post-training** stage, specifically on **Reinforcement Learning from Human Feedback (RLHF)** using **Proximal Policy Optimization (PPO)**.

---

## RLHF in Brief

RLHF is a framework in which a language model is optimized using a reward signal that approximates human preference.

A typical PPO-based RLHF setup contains four components:

* **Policy model** (`π_θ`)
  The model being optimized and, after training, the model used for deployment.

* **Reference model** (`π_ref`)
  A frozen copy of the SFT policy, used to constrain excessive policy drift through KL regularization.

* **Reward model (RM)** (`r(x, y)`)
  A model or scoring function that evaluates the quality of a generated response.

* **Value model / Critic** (`V(s)`)
  A model that estimates the expected future return of each generation state and reduces variance during policy optimization.

The central idea is straightforward:

1. sample responses from the current policy
2. score them with a reward signal
3. assign token-level credit using value estimates and GAE
4. update the policy with PPO under a KL constraint relative to the reference model

---

## Why PPO for RLHF?

PPO has historically been one of the most widely used policy optimization methods in RLHF because it offers a relatively stable and practical first-order update rule.

Its main advantages include:

* **clipped policy updates** that reduce destructive step sizes
* **compatibility with sequence-level rewards** and token-level credit assignment
* **integration with value-function baselines** for lower-variance training
* **practical on-policy optimization** with minibatch updates over rollout data

In LLM alignment, PPO is not the only option, and newer approaches such as DPO, IPO, and related preference-optimization methods are often simpler. However, PPO remains important because it explicitly exposes the full reinforcement learning pipeline: rollout, reward shaping, value estimation, advantage computation, and constrained policy updates.

---

## What This Repository Implements

This repository provides a small, fully readable training script that mirrors the logical structure of PPO-based RLHF.

The implementation includes:

* a **toy causal language model with a value head**
* a brief **SFT warm start** phase
* a **frozen reference model** copied from the SFT policy
* a **frozen reward function** used to score completed responses
* **autoregressive rollout generation** with storage of old log-probabilities and value estimates
* **token-level KL penalties** computed against the reference model
* **sequence-level reward injection** on the last valid response token
* **GAE-based advantage estimation** and return computation
* **PPO clipped objective** for actor updates
* **value loss** for critic updates
* **entropy regularization**
* **adaptive KL coefficient adjustment**

The goal is educational clarity rather than scale.

---

## PPO-Based RLHF Training Flow

### 1. Supervised Warm Start

Training starts from a policy that has already learned a reasonable response prior through supervised fine-tuning. In practical RLHF systems, PPO is rarely started from a randomly initialized policy.

In this demo, a short SFT stage is included to simulate that warm-start behavior.

### 2. Freeze the Reference Model

A copy of the post-SFT policy is frozen as the **reference model**. During PPO, the policy is discouraged from drifting too far from this reference through a KL-based penalty.

### 3. Rollout Collection

For a batch of prompts, the current policy autoregressively samples response tokens. During rollout, the script records:

* generated response tokens
* old policy log-probabilities for sampled tokens
* value predictions for each generation step
* validity masks for response tokens

This stage is **forward-only** and does not update model parameters.

### 4. Reward Construction

After rollout, two reward-related quantities are computed:

* a **sequence-level reward** from the reward model
* a **token-level KL penalty** relative to the reference model

A common practical construction is:

* each token receives a KL-based shaping reward
* the final valid token additionally receives the sequence-level reward from the reward model

This produces a **token-level reward sequence** suitable for temporal credit assignment.

### 5. Advantage and Return Computation

Using the token-level rewards and stored value predictions, the script computes:

* **TD residuals**
* **generalized advantage estimates (GAE)**
* **returns** used for critic regression

This is the stage where delayed sequence-level reward is propagated backward across the generated trajectory.

### 6. PPO Update

During the update phase, the model does **not** resample new responses. Instead, it performs a **teacher-forced forward pass** over the already collected responses to recompute:

* current policy log-probabilities
* current value estimates
* token entropies

Then it applies:

* PPO clipped actor loss
* critic value loss
* entropy bonus

Only the **policy/value model** is optimized. The **reference model** and **reward model** remain frozen.

### 7. Repeat Rollout → Update

Because PPO is an **on-policy** algorithm, rollout data are only reused for a limited number of optimization epochs. After that, a new rollout batch must be collected from the updated policy.

---

## Key Quantities in PPO-Based RLHF

For a prompt `x` and autoregressive response `y = (y1, y2, ..., yT)`, define the state at time step `t` as `s_t = (x, y_{<t})`, and the sampled action as `a_t = y_t`.

Important quantities include:

* **old policy log-probability**
  `log π_old(a_t | s_t)`
  Recorded during rollout.

* **current policy log-probability**
  `log π_θ(a_t | s_t)`
  Recomputed during the update phase.

* **reference policy log-probability**
  `log π_ref(a_t | s_t)`
  Used for KL regularization.

* **value estimate**
  `V(s_t)`
  Produced by the critic / value head.

* **token-level KL proxy**
  A common practical choice is:

  `kl_t = log π_old(a_t | s_t) - log π_ref(a_t | s_t)`

* **KL shaping reward**
  `r_t^KL = -β * kl_t`

* **sequence-level reward**
  `r_rm(x, y)`

* **final token-level reward**
  In a common RLHF construction:

  * for intermediate tokens: reward is primarily KL shaping
  * for the last valid token: reward includes both KL shaping and the sequence-level RM score

* **GAE advantage**
  `A_t`

* **return**
  `G_t = A_t + V(s_t)`

* **PPO ratio**
  `ratio_t = exp(log π_θ(a_t | s_t) - log π_old(a_t | s_t))`

This distinction between **old**, **current**, and **reference** probabilities is central to understanding PPO-based RLHF correctly.

---

## Training-Time Components vs Deployment-Time Components

One of the most important conceptual distinctions in RLHF is the difference between **training-time scaffolding** and **deployment-time inference**.

### Training time

PPO-based RLHF typically requires:

* policy model
* reference model
* reward model
* value model / critic
* rollout storage
* GAE and PPO loss computation

### Deployment time

After training, inference usually requires only:

* the **final policy model**

The reward model, reference model, critic, and PPO losses are not part of standard online generation.

---

## Implementation Notes

This repository intentionally uses a minimal toy setup so that the full PPO-RLHF logic can be examined in one place.

It is therefore **not** intended as:

* a benchmark implementation
* a distributed training framework
* a large-scale LLM serving stack
* a production RLHF system

Several simplifications are deliberate:

* a small recurrent language model is used instead of a Transformer
* the reward model is implemented as a frozen rule-based scorer rather than a learned neural preference model
* rollout is performed without KV-cache optimization
* the objective is pedagogical readability rather than throughput

Despite those simplifications, the core post-training logic remains faithful to PPO-style RLHF:

* collect on-policy rollouts
* compute reward and KL shaping
* estimate advantages and returns
* update policy and value under a clipped objective

---

## File Structure

* `rlhf_ppo_toy_full_walkthrough.py`
  Single-file runnable implementation of the entire training flow.

---

## Running the Demo

Requirements:

* Python 3.10+
* PyTorch 2.x

Run:

```bash
python rlhf_ppo_toy_full_walkthrough.py
```

The script prints training logs showing quantities such as:

* rollout length
* reward model score
* token-level KL
* total shaped reward
* actor loss
* critic loss
* entropy
* approximate KL
* clip fraction
* adaptive KL coefficient

It also prints sample generations before SFT, after SFT, and during PPO training.

---

## What to Look For When Reading the Code

The most important functions in the script are:

* `generate_rollout`
  Collects autoregressive samples and stores rollout-time quantities.

* `compute_ref_logprobs`
  Computes reference log-probabilities for sampled tokens.

* `build_token_level_rewards`
  Combines KL shaping and sequence-level reward into per-token rewards.

* `compute_gae_and_returns`
  Performs temporal credit assignment.

* `ppo_update`
  Recomputes current-policy log-probabilities and applies PPO/value updates.

* `train_rlhf_ppo`
  Orchestrates the complete training loop.

---

## Common Practical Lessons Reflected in the Demo

Even in a toy implementation, several practical lessons from real PPO-RLHF systems remain visible:

* PPO should generally start from a reasonable **SFT warm start**.
* The **reference model** should remain frozen.
* **Old log-probabilities** must come from the rollout policy snapshot, not from a later recomputation after parameters have changed.
* PPO updates should apply only to **valid response tokens**, not to prompt tokens or padded positions.
* The policy update is based on **teacher-forced evaluation of collected responses**, not on resampling during the optimization step.
* **KL control** is often critical for preventing unstable policy drift.
* The **critic** is not optional bookkeeping; poor value estimates can substantially destabilize policy optimization.
* PPO data are **on-policy** and cannot be reused indefinitely.

---

## Limitations

This repository does not cover many engineering concerns found in large-scale LLM post-training, including:

* distributed rollout and optimization
* mixed precision and memory optimization
* packed sequences and dynamic batching
* KV cache acceleration during generation
* learned reward-model training from preference pairs
* large-vocabulary tokenizers and real instruction datasets
* large-scale monitoring, evaluation, and safety filtering

Those are essential for production systems, but they are separate from the core conceptual structure that this repository aims to clarify.

---

## Intended Use

This repository is intended for:

* readers learning the mechanics of PPO-based RLHF
* engineers who want a minimal end-to-end post-training reference
* interview preparation on RLHF and PPO data flow
* debugging conceptual confusion around rollout, reward shaping, GAE, and policy/value updates

It is best understood as a **didactic implementation** rather than a scalable framework.

---

## Summary

This project presents a compact walkthrough of **LLM post-training with RLHF-style PPO**. Its purpose is to expose the full reinforcement learning pipeline behind alignment-oriented post-training:

* start from an SFT policy
* sample on-policy responses
* score them with reward and KL shaping
* propagate sequence-level reward backward with GAE
* update the policy with PPO and the critic with value regression
* deploy only the final aligned policy

For readers who want to understand not only the formulas but also the concrete training-time flow of PPO-based RLHF, this repository provides a minimal but complete reference.
