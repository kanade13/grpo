import pytest
import torch

from src.advantages import compute_group_advantages
from src.kl import compute_k3_kl
from src.logprobs import compute_token_logprobs
from src.losses import compute_grpo_loss
from src.masks import build_response_mask
from src.rollout import generate_group_rollouts


def test_core_algorithm_functions_are_explicit_placeholders() -> None:
    with pytest.raises(NotImplementedError):
        compute_group_advantages("prompt-0", torch.tensor([1.0, 0.0]))

    with pytest.raises(NotImplementedError):
        compute_token_logprobs(
            torch.zeros(1, 2, 3),
            torch.zeros(1, 2, dtype=torch.long),
            torch.ones(1, 2),
        )

    with pytest.raises(NotImplementedError):
        build_response_mask(
            torch.tensor([[1, 2, 0]]),
            torch.tensor([2]),
            pad_token_id=0,
        )

    with pytest.raises(NotImplementedError):
        compute_k3_kl(torch.zeros(1, 2), torch.zeros(1, 2), torch.ones(1, 2))

    with pytest.raises(NotImplementedError):
        compute_grpo_loss(
            new_logprobs=torch.zeros(1, 2),
            old_logprobs=torch.zeros(1, 2),
            advantages=torch.ones(1),
            response_mask=torch.ones(1, 2),
            token_kl=torch.zeros(1, 2),
            clip_eps=0.2,
            beta=0.01,
        )

    with pytest.raises(NotImplementedError):
        generate_group_rollouts(model_runner=None, examples=[], group_size=2)

