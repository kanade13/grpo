from dataclasses import dataclass

import torch

from src.trainer import TrainingConfig, run_train_step
from src.types import GRPOLossResult, LogProbResult, TokenizedRolloutBatch, TrainMetrics


class FakePolicyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0]))

    @property
    def device(self) -> torch.device:
        return self.weight.device


@dataclass
class FakeTokenizer:
    pad_token_id: int = 0


@dataclass
class FakeRunner:
    model: FakePolicyModel
    tokenizer: FakeTokenizer


def test_training_config_from_config() -> None:
    config = {
        "prompt": {"template": "{question}"},
        "data": {"train_path": "data/gsm8k/train.jsonl", "train_limit": 4},
        "train": {
            "max_steps": 2,
            "prompt_batch_size": 1,
            "group_size": 2,
            "learning_rate": 1e-4,
        },
    }

    train_cfg = TrainingConfig.from_config(config)

    assert train_cfg.train_data_path == "data/gsm8k/train.jsonl"
    assert train_cfg.max_steps == 2
    assert train_cfg.group_size == 2
    assert train_cfg.reference_model_enabled is True


def test_run_train_step_wires_core_functions(monkeypatch) -> None:
    import src.trainer as trainer

    model = FakePolicyModel()
    runner = FakeRunner(model=model, tokenizer=FakeTokenizer())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    train_cfg = TrainingConfig(
        train_data_path="unused",
        prompt_template="{question}",
        output_dir="unused",
        max_steps=1,
        prompt_batch_size=1,
        group_size=2,
        learning_rate=0.1,
        weight_decay=0.0,
        clip_eps=0.2,
        beta=0.01,
        max_grad_norm=None,
        reference_model_enabled=False,
        log_every=1,
    )
    rollout_batch = TokenizedRolloutBatch(
        input_ids=torch.tensor([[1]]),
        attention_mask=torch.tensor([[1]]),
        labels=torch.tensor([[1]]),
        prompt_lengths=torch.tensor([0]),
        advantages=torch.tensor([1.0]),
        rewards=torch.tensor([1.0]),
        prompt_ids=["prompt-0"],
        completion_ids=[0],
    )

    monkeypatch.setattr(trainer, "generate_group_rollouts", lambda *args, **kwargs: [])
    monkeypatch.setattr(trainer, "build_advantage_lookup", lambda rollouts: {})
    monkeypatch.setattr(trainer, "flatten_rollouts", lambda rollouts: [])
    monkeypatch.setattr(trainer, "tokenize_rollout_batch", lambda *args, **kwargs: rollout_batch)

    def fake_compute_model_logprobs(model, batch, pad_token_id):
        token_logprobs = model.weight.reshape(1, 1)
        mask = torch.ones_like(token_logprobs)
        return LogProbResult(token_logprobs=token_logprobs, sequence_logprobs=token_logprobs.sum(dim=1), mask=mask)

    def fake_compute_grpo_loss(**kwargs):
        loss = kwargs["new_logprobs"].sum()
        zero = loss.detach() * 0
        return GRPOLossResult(
            loss=loss,
            policy_loss=loss,
            kl_loss=zero,
            clip_fraction=zero,
            mean_ratio=zero + 1,
        )

    monkeypatch.setattr(trainer, "compute_model_logprobs", fake_compute_model_logprobs)
    monkeypatch.setattr(trainer, "compute_grpo_loss", fake_compute_grpo_loss)

    metrics = run_train_step(
        policy_runner=runner,
        reference_model=None,
        optimizer=optimizer,
        examples=[],
        train_cfg=train_cfg,
        step=1,
    )

    assert isinstance(metrics, TrainMetrics)
    assert metrics.step == 1
    assert metrics.mean_reward == 1.0
    assert model.weight.item() < 1.0
