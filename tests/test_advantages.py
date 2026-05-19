import pytest
import torch

from src.advantages import compute_group_advantages
from src.types import GroupAdvantage


def test_compute_group_advantages_normalizes_one_prompt_group() -> None:
    rewards = torch.tensor([1.0, 2.0, 3.0])

    result = compute_group_advantages("prompt-0", rewards)

    expected_advantages = torch.tensor([-1.2247448, 0.0, 1.2247448])
    assert isinstance(result, GroupAdvantage)
    assert result.prompt_id == "prompt-0"
    assert torch.equal(result.rewards, rewards)
    assert torch.allclose(result.mean, torch.tensor(2.0))
    assert torch.allclose(result.std, torch.tensor(0.8164966), atol=1e-6)
    assert torch.allclose(result.advantages, expected_advantages, atol=1e-6)


def test_compute_group_advantages_all_equal_rewards_returns_zero_advantages() -> None:
    rewards = torch.tensor([5.0, 5.0, 5.0])

    result = compute_group_advantages("prompt-0", rewards)

    assert torch.allclose(result.mean, torch.tensor(5.0))
    assert torch.allclose(result.std, torch.tensor(0.0))
    assert torch.equal(result.advantages, torch.zeros_like(rewards))


def test_compute_group_advantages_single_completion_returns_zero_advantage() -> None:
    rewards = torch.tensor([7.0])

    result = compute_group_advantages("prompt-0", rewards)

    assert torch.allclose(result.mean, torch.tensor(7.0))
    assert torch.allclose(result.std, torch.tensor(0.0))
    assert torch.equal(result.advantages, torch.zeros_like(rewards))


def test_compute_group_advantages_handles_negative_rewards() -> None:
    rewards = torch.tensor([-1.0, 0.0, 1.0])

    result = compute_group_advantages("prompt-0", rewards)

    expected_advantages = torch.tensor([-1.2247448, 0.0, 1.2247448])
    assert torch.allclose(result.mean, torch.tensor(0.0))
    assert torch.allclose(result.std, torch.tensor(0.8164966), atol=1e-6)
    assert torch.allclose(result.advantages, expected_advantages, atol=1e-6)


def test_compute_group_advantages_detaches_reward_tensor() -> None:
    rewards = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

    result = compute_group_advantages("prompt-0", rewards)

    assert result.rewards.requires_grad is False
    assert result.mean.requires_grad is False
    assert result.std.requires_grad is False
    assert result.advantages.requires_grad is False


def test_compute_group_advantages_rejects_non_1d_rewards() -> None:
    rewards = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    with pytest.raises(ValueError, match="1D"):
        compute_group_advantages("prompt-0", rewards)
