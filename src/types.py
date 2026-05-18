from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


@dataclass
class GSM8KExample:
    id: str
    question: str
    answer: str
    final_answer: str
    prompt: str


@dataclass
class RewardResult:
    total_reward: float
    answer_reward: float
    format_reward: float
    length_penalty: float
    extracted_answer: str | None
    is_correct: bool


@dataclass
class GeneratedCompletion:
    """单个 prompt 的一次采样结果，不是一整个 group。

    例子：如果一道题设置 `group_size = 4`，会有 4 个 `GeneratedCompletion`。
    这 4 个对象共同放在一个 `PromptRollout.completions` 里。

    因此这些字段都是“单次采样”的值，而不是 list of samples：
    - `response`: 这一次采样的文本回答。
    - `response_token_ids`: 这一次采样生成的 response token ids。
    - `reward`: 这一次采样对应的 reward。

    训练时需要 token 级 logprob，所以还要保存：
    - `input_ids`: 模型实际训练时看到的完整 token 序列，
      即 chat-template prompt tokens + response tokens，未 padding。
    - `prompt_token_count`: `input_ids` 中 prompt 部分的长度。response mask
      会用它来区分 prompt token 和 response token。
    - `attention_mask`: 单条未 padding 序列的 mask，通常就是
      `[1] * len(input_ids)`；真正组成 batch 时，`trainer.tokenize_rollout_batch`
      会重新 padding，并用 `input_ids != pad_token_id` 生成 batch attention mask。
    """

    prompt_id: str
    completion_id: int
    prompt: str
    response: str
    response_token_ids: list[int]
    attention_mask: list[int]
    input_ids: list[int] | None = None
    prompt_token_count: int | None = None
    reward: RewardResult | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PromptRollout:
    """一道题目的整组采样结果。

    `PromptRollout` 才对应 GRPO 里的一个 prompt group：
    - `prompt`: 一道题目的 prompt。
    - `answer`: 这道题的标准答案。
    - `completions`: 同一道题的 G 个 `GeneratedCompletion`。

    后续 advantage 会在 `completions` 这组内部计算，而不是跨题计算。
    """

    prompt_id: str
    prompt: str
    answer: str
    completions: list[GeneratedCompletion]


@dataclass
class GroupAdvantage:
    prompt_id: str
    rewards: torch.Tensor
    mean: torch.Tensor
    std: torch.Tensor
    advantages: torch.Tensor


@dataclass
class LogProbResult:
    token_logprobs: torch.Tensor
    sequence_logprobs: torch.Tensor
    mask: torch.Tensor


@dataclass
class KLDivergenceResult:
    token_kl: torch.Tensor
    masked_kl: torch.Tensor
    mean_kl: torch.Tensor


@dataclass
class GRPOLossResult:
    loss: torch.Tensor
    policy_loss: torch.Tensor
    kl_loss: torch.Tensor
    clip_fraction: torch.Tensor
    mean_ratio: torch.Tensor


@dataclass
class TrainMetrics:
    step: int
    loss: float
    mean_reward: float
    reward_std: float
    mean_kl: float
    grad_norm: float
    learning_rate: float


@dataclass
class TokenizedRolloutBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    prompt_lengths: torch.Tensor
    advantages: torch.Tensor
    rewards: torch.Tensor
    prompt_ids: list[str]
    completion_ids: list[int]
