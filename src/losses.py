from __future__ import annotations

import torch

from .types import GRPOLossResult


def compute_grpo_loss(
    new_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    token_kl: torch.Tensor,
    clip_eps: float,
    beta: float,
) -> GRPOLossResult:
    """计算 GRPO clipped objective。

    输入形状：
    - `new_logprobs`: `[batch, response_seq_len]`
    - `old_logprobs`: `[batch, response_seq_len]`
    - `advantages`: `[batch]` 或 `[batch, 1]`
    - `response_mask`: `[batch, response_seq_len]`
    - `token_kl`: `[batch, response_seq_len]`

    公式：
        ratio = exp(new_logprobs - old_logprobs)
        clipped_ratio = clip(ratio, 1 - clip_eps, 1 + clip_eps)

        per_token_obj = min(ratio * A, clipped_ratio * A)
        policy_loss = - sum(per_token_obj * response_mask) / sum(response_mask)

        kl_loss = sum(token_kl * response_mask) / sum(response_mask)
        total_loss = policy_loss + beta * kl_loss

    注意：
    - `old_logprobs` 是 rollout 时旧策略的 logprob，不应带当前梯度。
    - advantage 通常按 prompt group 归一化后 detach。
    - A 为负数时，`min` 和 clipping 的行为仍应按 PPO/GRPO objective 逐元素处理。
    """
    raise NotImplementedError("核心 GRPO loss 逻辑留给手写实现。")

