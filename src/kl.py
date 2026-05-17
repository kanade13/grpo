from __future__ import annotations

import torch

from .types import KLDivergenceResult


def compute_k3_kl(
    policy_logprobs: torch.Tensor,
    reference_logprobs: torch.Tensor,
    mask: torch.Tensor,
) -> KLDivergenceResult:
    """计算 GRPO 常用的 k3 KL 估计。

    令：
        logr = reference_logprob - policy_logprob

    k3 KL estimator:
        KL_token = exp(logr) - logr - 1

    聚合：
        masked_kl = KL_token * mask
        mean_kl = sum(masked_kl) / sum(mask)

    注意：
    - `policy_logprobs` 和 `reference_logprobs` 必须来自同一批 generated tokens。
    - `mask` 只覆盖 response token。
    - reference model 不参与梯度更新；reference_logprobs 应该 detach。
    """
    raise NotImplementedError("核心 KL 逻辑留给手写实现。")

