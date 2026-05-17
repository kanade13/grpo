from __future__ import annotations

import torch


def build_response_mask(
    input_ids: torch.Tensor,
    prompt_lengths: torch.Tensor,
    pad_token_id: int,
) -> torch.Tensor:
    """构造只覆盖 response tokens 的 mask。

    输入形状：
    - `input_ids`: `[batch, seq_len]`
    - `prompt_lengths`: `[batch]`，每条样本 prompt 在当前 padding 策略下的真实长度。

    目标：
    - prompt token 的 mask 为 0。
    - pad token 的 mask 为 0。
    - response token 的 mask 为 1。

    注意：
    - 如果 tokenizer 使用 left padding，prompt 起点和 response 起点要结合
      attention_mask 或真实长度处理，不能简单假设 prompt 从位置 0 开始。
    - 训练 loss、KL、logprob 聚合都必须只在 response mask 上计算。
    """
    raise NotImplementedError("核心 response mask 逻辑留给手写实现。")

