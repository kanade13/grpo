from dataclasses import dataclass

import torch

from src.trainer import TrainingConfig, build_advantage_lookup, run_train_step, tokenize_rollout_batch
from src.types import (
    GRPOLossResult,
    GeneratedCompletion,
    LogProbResult,
    PromptRollout,
    RewardResult,
    TokenizedRolloutBatch,
    TrainMetrics,
)


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


def test_build_advantage_lookup_rejects_missing_reward() -> None:
    completion = GeneratedCompletion(
        prompt_id="prompt-0",
        completion_id=0,
        prompt="prompt",
        response="response",
        response_token_ids=[3],
        attention_mask=[1],
        input_ids=[1, 2, 3],
        prompt_token_count=2,
        reward=None,
    )
    rollout = PromptRollout(
        prompt_id="prompt-0",
        prompt="prompt",
        answer="3",
        completions=[completion],
    )

    try:
        build_advantage_lookup([rollout])
    except ValueError as exc:
        assert "missing reward" in str(exc)
    else:
        raise AssertionError("missing reward should raise ValueError")


def test_tokenize_rollout_batch_uses_rollout_token_ids() -> None:
    tokenizer = FakeTokenizer()
    completions = [
        GeneratedCompletion(
            prompt_id="prompt-0",
            completion_id=0,
            prompt="chat-template prompt",
            response="response",
            response_token_ids=[3, 4],
            attention_mask=[1, 1],
            input_ids=[1, 2, 3, 4],
            prompt_token_count=2,
            reward=RewardResult(1.0, 1.0, 0.0, 0.0, "4", True),
        ),
        GeneratedCompletion(
            prompt_id="prompt-1",
            completion_id=0,
            prompt="chat-template prompt 2",
            response="response",
            response_token_ids=[6],
            attention_mask=[1],
            input_ids=[5, 6],
            prompt_token_count=1,
            reward=RewardResult(0.0, 0.0, 0.0, 0.0, "6", False),
        ),
    ]
    advantage_lookup = {
        ("prompt-0", 0): (1.0, 0.5),
        ("prompt-1", 0): (0.0, -0.5),
    }

    batch = tokenize_rollout_batch(tokenizer, completions, advantage_lookup, device=torch.device("cpu"))

    assert batch.input_ids.tolist() == [[1, 2, 3, 4], [5, 6, 0, 0]]
    assert batch.attention_mask.tolist() == [[1, 1, 1, 1], [1, 1, 0, 0]]
    assert batch.labels.tolist() == batch.input_ids.tolist()
    assert batch.prompt_lengths.tolist() == [2, 1]
    assert batch.advantages.tolist() == [0.5, -0.5]


def test_compute_model_logprobs_shifts_logits_labels_and_mask(monkeypatch) -> None:
    import src.trainer as trainer

    class FakeOutput:
        def __init__(self, logits: torch.Tensor) -> None:
            self.logits = logits

    class FakeModel(torch.nn.Module):
        def forward(self, input_ids, attention_mask):
            batch_size, seq_len = input_ids.shape
            logits = torch.arange(batch_size * seq_len * 3, dtype=torch.float32).reshape(batch_size, seq_len, 3)
            return FakeOutput(logits)

    rollout_batch = TokenizedRolloutBatch(
        input_ids=torch.tensor([[1, 2, 3, 0]]),
        attention_mask=torch.tensor([[1, 1, 1, 0]]),
        labels=torch.tensor([[1, 2, 3, 0]]),
        prompt_lengths=torch.tensor([1]),
        advantages=torch.tensor([1.0]),
        rewards=torch.tensor([1.0]),
        prompt_ids=["prompt-0"],
        completion_ids=[0],
    )

    monkeypatch.setattr(trainer, "build_response_mask", lambda *args, **kwargs: torch.tensor([[0, 1, 1, 0]]))

    def fake_compute_token_logprobs(logits, target_ids, mask):
        assert logits.shape == (1, 3, 3)
        assert target_ids.tolist() == [[2, 3, 0]]
        assert mask.tolist() == [[1, 1, 0]]
        return LogProbResult(token_logprobs=torch.ones(1, 3), sequence_logprobs=torch.ones(1), mask=mask)

    monkeypatch.setattr(trainer, "compute_token_logprobs", fake_compute_token_logprobs)

    result = trainer.compute_model_logprobs(FakeModel(), rollout_batch, pad_token_id=0)

    assert result.mask.tolist() == [[1, 1, 0]]


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
