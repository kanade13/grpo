from __future__ import annotations

import torch


def build_response_mask(
    input_ids: torch.Tensor,
    prompt_lengths: torch.Tensor,
    response_lengths: torch.Tensor,
    pad_token_id: int,
) -> torch.Tensor:
    """构造只覆盖 response tokens 的 mask。

    输入形状：
    - `input_ids`: `[batch, seq_len]` 为right padding后的prompt+reponse
    - `prompt_lengths`: `[batch]`，每条样本 prompt 在当前 padding 策略下的真实长度。
    - `response_lengths`: `[batch]`，每条样本 response 的真实长度，包含生成出的
      EOS token。

    目标：
    - prompt token 的 mask 为 0。
    - response 后面的 padding 位置为 0。
    - response token 的 mask 为 1。

    注意：
    - 这里不通过 `pad_token_id` 判断 padding，因为部分模型会让 pad token 同时作为
      EOS token。真实 response 中的 EOS 应该参与训练，只有 response 之后补齐的
      padding 不参与训练。
    - 当前实现假设输入是 right padding，且 prompt 从位置 0 开始。
    - 训练 loss、KL、logprob 聚合都必须只在 response mask 上计算。
    """
    if input_ids.ndim != 2:
        raise ValueError(f"input_ids must be 2D, got shape {tuple(input_ids.shape)}")
    if prompt_lengths.ndim != 1:
        raise ValueError(f"prompt_lengths must be 1D, got shape {tuple(prompt_lengths.shape)}")
    if response_lengths.ndim != 1:
        raise ValueError(f"response_lengths must be 1D, got shape {tuple(response_lengths.shape)}")
    if prompt_lengths.shape[0] != input_ids.shape[0] or response_lengths.shape[0] != input_ids.shape[0]:
        raise ValueError("prompt_lengths and response_lengths must match input_ids batch size.")
    if torch.any(prompt_lengths < 0) or torch.any(response_lengths < 0):
        raise ValueError("prompt_lengths and response_lengths must be non-negative.")

    prompt_lengths = prompt_lengths.to(device=input_ids.device)
    response_lengths = response_lengths.to(device=input_ids.device)
    response_ends = prompt_lengths + response_lengths
    if torch.any(response_ends > input_ids.shape[1]):
        raise ValueError("prompt_length + response_length must not exceed input_ids sequence length.")

    _ = pad_token_id
    positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
    response_starts = prompt_lengths.unsqueeze(1)
    response_ends = response_ends.unsqueeze(1)
    return ((positions >= response_starts) & (positions < response_ends)).to(dtype=input_ids.dtype)
