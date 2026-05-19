from __future__ import annotations

import torch

from .types import GroupAdvantage


def compute_group_advantages(
    prompt_id: str,
    rewards: torch.Tensor,
    eps: float = 1e-8,
) -> GroupAdvantage:
    """计算单个 prompt group 内部的 GRPO advantage。

    对同一个 prompt 的 G 个 completion，reward 向量为 r。

    公式：
        mean = (1 / G) * sum_i r_i
        std = sqrt((1 / G) * sum_i (r_i - mean)^2)
        A_i = (r_i - mean) / (std + eps)

    注意事项：
    - `rewards` 应该是一维 tensor，shape 为 `[group_size]`。
    - 当所有 reward 相等时，std 接近 0。为了避免数值噪声，此时应返回全 0
      advantages。
    - advantage 不需要梯度，通常应从 reward 标量构造或显式 detach。
    """
    #raise NotImplementedError("核心 advantage 逻辑留给手写实现。")
    if rewards.ndim != 1:
        raise ValueError(f"Expected a 1D tensor, but got shape {tuple(rewards.shape)}")
    rewards = rewards.detach()
    advantages = torch.zeros_like(rewards)
    mean=rewards.mean(dim=0)
    std=rewards.std(dim=0,correction=0)
    if std <= eps:
        advantages = torch.zeros_like(rewards)
    else:
        advantages = (rewards - mean) / (std + eps)
    return GroupAdvantage(
        prompt_id=prompt_id,
        rewards=rewards,
        mean=mean,
        std=std,
        advantages=advantages,
    )

