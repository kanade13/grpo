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
    prompt_id: str
    completion_id: int
    prompt: str
    response: str
    response_token_ids: list[int]
    attention_mask: list[int]
    reward: RewardResult | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PromptRollout:
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
