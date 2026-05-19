from __future__ import annotations

import torch

from .types import LogProbResult


def compute_token_logprobs(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    mask: torch.Tensor,
) -> LogProbResult:
    """计算目标 token 的逐 token log probability。

    输入形状约定：
    - `logits`: `[batch, seq_len, vocab_size]`
    - `target_ids`: `[batch, seq_len]`
    - `mask`: `[batch, seq_len]`，1 表示参与 loss/reward 的 response token。

    公式：
        log_probs = log_softmax(logits, dim=-1)
        token_logprobs[b, t] = log_probs[b, t, target_ids[b, t]]
        sequence_logprobs[b] = sum_t token_logprobs[b, t] * mask[b, t]

    实现时要注意：
    - 用 `torch.log_softmax` 和 `torch.gather`，不要手写 softmax 后取 log。
    - 通常 logits 和 target_ids 要对齐到“预测下一个 token”的位置。
      如果上游传入的是完整 input_ids，需要先做 shift。
    """
    """
    input_ids: p0 p1 p2 p3 p4 r0 r1 r2
    logits     x0 x1 x2 x3 x4 x5 x6 x7
    应该选:                 1  1  1

    """
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    token_logprobs = log_probs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
    mask = mask.to(dtype=token_logprobs.dtype)
    sequence_logprobs = (token_logprobs * mask).sum(dim=1)#这里因为是log概率，所以是相加，还挺方便
    return LogProbResult(
        token_logprobs=token_logprobs,
        sequence_logprobs=sequence_logprobs,
        mask=mask,
    )
