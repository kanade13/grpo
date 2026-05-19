import pytest
import torch

from src.kl import compute_k3_kl
from src.losses import compute_grpo_loss


def test_core_algorithm_functions_are_explicit_placeholders() -> None:
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
