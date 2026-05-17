from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml

from .advantages import compute_group_advantages
from .data import batch_iter, load_gsm8k_jsonl
from .kl import compute_k3_kl
from .logprobs import compute_token_logprobs
from .losses import compute_grpo_loss
from .masks import build_response_mask
from .model_runner import ModelRunner
from .optim import build_adam_optimizer
from .rollout import generate_group_rollouts
from .types import GeneratedCompletion, PromptRollout, TokenizedRolloutBatch, TrainMetrics


@dataclass
class TrainingConfig:
    train_data_path: str
    prompt_template: str
    output_dir: str
    max_steps: int
    prompt_batch_size: int
    group_size: int
    learning_rate: float
    weight_decay: float
    clip_eps: float
    beta: float
    max_grad_norm: float | None
    reference_model_enabled: bool
    log_every: int
    train_split: str = "train"
    data_limit: int | None = None

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "TrainingConfig":
        train_cfg = _require_mapping(config, "train")
        data_cfg = _require_mapping(config, "data")
        prompt_cfg = _require_mapping(config, "prompt")

        return cls(
            train_data_path=str(data_cfg.get("train_path") or data_cfg.get("path")),
            prompt_template=str(prompt_cfg["template"]),
            output_dir=str(train_cfg.get("output_dir", "./runs/train_debug")),
            max_steps=int(train_cfg.get("max_steps", 1)),
            prompt_batch_size=int(train_cfg.get("prompt_batch_size", 1)),
            group_size=int(train_cfg.get("group_size", 2)),
            learning_rate=float(train_cfg.get("learning_rate", 1e-5)),
            weight_decay=float(train_cfg.get("weight_decay", 0.0)),
            clip_eps=float(train_cfg.get("clip_eps", 0.2)),
            beta=float(train_cfg.get("beta", 0.01)),
            max_grad_norm=_optional_float(train_cfg.get("max_grad_norm")),
            reference_model_enabled=bool(train_cfg.get("reference_model_enabled", True)),
            log_every=int(train_cfg.get("log_every", 1)),
            train_split=str(data_cfg.get("train_split", "train")),
            data_limit=_optional_int(data_cfg.get("train_limit")),
        )


def load_training_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return config


def build_reference_model(policy_runner: ModelRunner):
    """加载冻结 reference model。

    这是边缘逻辑，不属于你后面要手写的核心算法。默认从同一个 `model.path`
    再加载一份模型，并冻结参数。显存不足时可以在配置里设置
    `train.reference_model_enabled: false`，此时 trainer 会跳过 KL 参考模型路径。
    """
    from transformers import AutoModelForCausalLM

    model_cfg = _require_mapping(policy_runner.config, "model")
    reference_model = AutoModelForCausalLM.from_pretrained(
        model_cfg["path"],
        torch_dtype=model_cfg.get("torch_dtype", "auto"),
        device_map=model_cfg.get("device_map", "auto"),
    )
    reference_model.eval()
    for parameter in reference_model.parameters():
        parameter.requires_grad_(False)
    return reference_model


def flatten_rollouts(rollouts: list[PromptRollout]) -> list[GeneratedCompletion]:
    completions: list[GeneratedCompletion] = []
    for rollout in rollouts:
        completions.extend(rollout.completions)
    return completions


def build_advantage_lookup(rollouts: list[PromptRollout]) -> dict[tuple[str, int], tuple[float, float]]:
    lookup: dict[tuple[str, int], tuple[float, float]] = {}
    for rollout in rollouts:
        rewards = torch.tensor(
            [
                completion.reward.total_reward
                if completion.reward is not None
                else float(completion.metadata.get("reward", 0.0))
                for completion in rollout.completions
            ],
            dtype=torch.float32,
        )
        group_advantage = compute_group_advantages(rollout.prompt_id, rewards)
        for completion, reward, advantage in zip(
            rollout.completions,
            group_advantage.rewards.tolist(),
            group_advantage.advantages.tolist(),
            strict=True,
        ):
            lookup[(completion.prompt_id, completion.completion_id)] = (float(reward), float(advantage))
    return lookup


def tokenize_rollout_batch(
    tokenizer,
    completions: list[GeneratedCompletion],
    advantage_lookup: dict[tuple[str, int], tuple[float, float]],
    device: torch.device,
) -> TokenizedRolloutBatch:
    texts = [completion.prompt + completion.response for completion in completions]
    prompts = [completion.prompt for completion in completions]
    tokenized = tokenizer(texts, return_tensors="pt", padding=True).to(device)
    prompt_tokenized = tokenizer(prompts, return_tensors="pt", padding=True).to(device)

    rewards: list[float] = []
    advantages: list[float] = []
    for completion in completions:
        reward, advantage = advantage_lookup[(completion.prompt_id, completion.completion_id)]
        rewards.append(reward)
        advantages.append(advantage)

    return TokenizedRolloutBatch(
        input_ids=tokenized.input_ids,
        attention_mask=tokenized.attention_mask,
        labels=tokenized.input_ids.clone(),
        prompt_lengths=prompt_tokenized.attention_mask.sum(dim=1),
        advantages=torch.tensor(advantages, dtype=torch.float32, device=device),
        rewards=torch.tensor(rewards, dtype=torch.float32, device=device),
        prompt_ids=[completion.prompt_id for completion in completions],
        completion_ids=[completion.completion_id for completion in completions],
    )


def compute_model_logprobs(model, rollout_batch: TokenizedRolloutBatch, pad_token_id: int):
    response_mask = build_response_mask(
        rollout_batch.input_ids,
        rollout_batch.prompt_lengths,
        pad_token_id=pad_token_id,
    )
    outputs = model(
        input_ids=rollout_batch.input_ids,
        attention_mask=rollout_batch.attention_mask,
    )
    return compute_token_logprobs(outputs.logits, rollout_batch.labels, response_mask)


def run_train_step(
    policy_runner: ModelRunner,
    reference_model,
    optimizer: torch.optim.Optimizer,
    examples,
    train_cfg: TrainingConfig,
    step: int,
) -> TrainMetrics:
    policy_runner.model.train()
    rollouts = generate_group_rollouts(policy_runner, examples, train_cfg.group_size)
    advantage_lookup = build_advantage_lookup(rollouts)
    completions = flatten_rollouts(rollouts)
    device = policy_runner.model.device
    rollout_batch = tokenize_rollout_batch(
        policy_runner.tokenizer,
        completions,
        advantage_lookup,
        device=device,
    )
    pad_token_id = policy_runner.tokenizer.pad_token_id

    with torch.no_grad():
        old_logprobs = compute_model_logprobs(policy_runner.model, rollout_batch, pad_token_id)

    new_logprobs = compute_model_logprobs(policy_runner.model, rollout_batch, pad_token_id)

    if reference_model is not None:
        with torch.no_grad():
            reference_logprobs = compute_model_logprobs(reference_model, rollout_batch, pad_token_id)
        kl_result = compute_k3_kl(
            new_logprobs.token_logprobs,
            reference_logprobs.token_logprobs.detach(),
            new_logprobs.mask,
        )
        token_kl = kl_result.token_kl
        mean_kl = float(kl_result.mean_kl.detach().cpu())
    else:
        token_kl = torch.zeros_like(new_logprobs.token_logprobs)
        mean_kl = 0.0

    loss_result = compute_grpo_loss(
        new_logprobs=new_logprobs.token_logprobs,
        old_logprobs=old_logprobs.token_logprobs.detach(),
        advantages=rollout_batch.advantages.detach(),
        response_mask=new_logprobs.mask,
        token_kl=token_kl,
        clip_eps=train_cfg.clip_eps,
        beta=train_cfg.beta,
    )

    optimizer.zero_grad(set_to_none=True)
    loss_result.loss.backward()
    grad_norm = 0.0
    if train_cfg.max_grad_norm is not None:
        grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
            policy_runner.model.parameters(),
            train_cfg.max_grad_norm,
        )
        grad_norm = float(grad_norm_tensor.detach().cpu())
    optimizer.step()

    return TrainMetrics(
        step=step,
        loss=float(loss_result.loss.detach().cpu()),
        mean_reward=float(rollout_batch.rewards.mean().detach().cpu()),
        reward_std=float(rollout_batch.rewards.std(unbiased=False).detach().cpu()),
        mean_kl=mean_kl,
        grad_norm=grad_norm,
        learning_rate=optimizer.param_groups[0]["lr"],
    )


def train(config: dict[str, Any]) -> list[TrainMetrics]:
    train_cfg = TrainingConfig.from_config(config)
    examples = load_gsm8k_jsonl(
        train_cfg.train_data_path,
        split=train_cfg.train_split,
        prompt_template=train_cfg.prompt_template,
        limit=train_cfg.data_limit,
    )
    if not examples:
        raise ValueError("No training examples loaded.")

    policy_runner = ModelRunner(config)
    reference_model = build_reference_model(policy_runner) if train_cfg.reference_model_enabled else None
    optimizer = build_adam_optimizer(
        policy_runner.model.parameters(),
        lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
    )

    output_dir = Path(train_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"

    metrics: list[TrainMetrics] = []
    example_batches = list(batch_iter(examples, train_cfg.prompt_batch_size))
    with metrics_path.open("w", encoding="utf-8") as f:
        for step in range(1, train_cfg.max_steps + 1):
            batch = example_batches[(step - 1) % len(example_batches)]
            step_metrics = run_train_step(
                policy_runner=policy_runner,
                reference_model=reference_model,
                optimizer=optimizer,
                examples=batch,
                train_cfg=train_cfg,
                step=step,
            )
            metrics.append(step_metrics)
            f.write(json.dumps(step_metrics.__dict__, ensure_ascii=False) + "\n")
            if train_cfg.log_every > 0 and step % train_cfg.log_every == 0:
                print(json.dumps(step_metrics.__dict__, ensure_ascii=False))

    return metrics


def _require_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Config field '{key}' must be a mapping.")
    return value


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a minimal GRPO training loop.")
    parser.add_argument("--config", required=True, help="Path to YAML training config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train(load_training_config(args.config))


if __name__ == "__main__":
    main()

