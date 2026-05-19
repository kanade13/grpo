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
from .types import (
    GSM8KExample,
    GeneratedCompletion,
    LogProbResult,
    PromptRollout,
    TokenizedRolloutBatch,
    TrainMetrics,
)


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
    """读取训练 YAML 配置，并确认顶层结构是 mapping。"""
    with Path(path).open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return config


def build_reference_model(policy_runner: ModelRunner) -> torch.nn.Module:
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
    """把按 prompt 分组的 rollout 展平成 completion 列表。

    `generate_group_rollouts` 的输出结构是 `B` 个 `PromptRollout`，每个里面有
    `G` 个 completion。训练前向需要把这些 completion 拼成一个 batch，所以这里
    返回长度为 `B * G` 的平铺列表，并保留每个 completion 自己的 prompt_id 和
    completion_id。
    """
    completions: list[GeneratedCompletion] = []
    for rollout in rollouts:
        completions.extend(rollout.completions)
    return completions


def build_advantage_lookup(rollouts: list[PromptRollout]) -> dict[tuple[str, int], tuple[float, float]]:
    """为每个 completion 计算并索引 reward/advantage。

    对每个 `PromptRollout` 内部的 G 个 reward 调用
    `compute_group_advantages`。返回的 dict 用
    `(prompt_id, completion_id)` 做 key，value 是 `(reward, advantage)`。

    这里要求每个 completion 已经有 reward；缺 reward 直接报错，避免训练时静默
    使用错误的 0 reward。
    """
    lookup: dict[tuple[str, int], tuple[float, float]] = {}
    for rollout in rollouts:
        reward_values: list[float] = []
        for completion in rollout.completions:
            if completion.reward is None:
                raise ValueError(
                    f"Completion {completion.prompt_id}/{completion.completion_id} is missing reward."
                )
            reward_values.append(completion.reward.total_reward)
        rewards = torch.tensor(reward_values, dtype=torch.float32)
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
    tokenizer: Any,
    completions: list[GeneratedCompletion],
    advantage_lookup: dict[tuple[str, int], tuple[float, float]],
    device: torch.device,
) -> TokenizedRolloutBatch:
    """把 rollout completions 转成模型训练用的 padded tensor batch。

    这里不重新 tokenize `prompt + response` 文本，而是使用 rollout 阶段保存的
    `completion.input_ids`。原因是训练必须复用模型生成时的 chat-template prompt
    token ids 和 response token ids，避免 token 序列与采样时不一致。

    输出内容：
    - `input_ids`: padding 后的完整序列 `[prompt tokens, response tokens, pad...]`。
    - `attention_mask`: 根据真实序列长度构造，真实 token 为 1，padding token 为 0。
    - `labels`: 当前等于 `input_ids`，后续在 `compute_model_logprobs` 中做 causal shift。
    - `prompt_lengths`: 每条样本 prompt 部分的 token 数，用来构造 response mask。
    - `response_lengths`: 每条样本 response 部分的 token 数，包含生成出的 EOS token。
    - `rewards` / `advantages`: 与 completion batch 顺序一一对应。
    """
    if tokenizer.pad_token_id is None:
        raise ValueError("tokenizer.pad_token_id must be set before training.")
    if not completions:
        raise ValueError("Cannot tokenize an empty rollout batch.")

    input_id_rows: list[torch.Tensor] = []
    prompt_lengths: list[int] = []
    response_lengths: list[int] = []
    sequence_lengths: list[int] = []
    for completion in completions:
        if completion.input_ids is None:
            raise ValueError(
                f"Completion {completion.prompt_id}/{completion.completion_id} is missing input_ids."
            )
        if completion.prompt_token_count is None:
            raise ValueError(
                f"Completion {completion.prompt_id}/{completion.completion_id} is missing prompt_token_count."
            )
        input_id_rows.append(torch.tensor(completion.input_ids, dtype=torch.long))
        prompt_lengths.append(completion.prompt_token_count)
        response_lengths.append(len(completion.response_token_ids))
        sequence_lengths.append(len(completion.input_ids))

    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_id_rows,
        batch_first=True,
        padding_value=tokenizer.pad_token_id,
    ).to(device)
    sequence_lengths_tensor = torch.tensor(sequence_lengths, dtype=torch.long, device=device)
    positions = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
    attention_mask = (positions < sequence_lengths_tensor.unsqueeze(1)).long()#形状是[batch_size, max_seq_len]，
                                                                              #对input_ids mask(prompt+response)

    rewards: list[float] = []
    advantages: list[float] = []
    for completion in completions:
        reward, advantage = advantage_lookup[(completion.prompt_id, completion.completion_id)]
        rewards.append(reward)
        advantages.append(advantage)

    return TokenizedRolloutBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=input_ids.clone(),
        prompt_lengths=torch.tensor(prompt_lengths, dtype=torch.long, device=device),
        response_lengths=torch.tensor(response_lengths, dtype=torch.long, device=device),
        advantages=torch.tensor(advantages, dtype=torch.float32, device=device),
        rewards=torch.tensor(rewards, dtype=torch.float32, device=device),
        prompt_ids=[completion.prompt_id for completion in completions],
        completion_ids=[completion.completion_id for completion in completions],
    )


def compute_model_logprobs(
    model: torch.nn.Module,
    rollout_batch: TokenizedRolloutBatch,
    pad_token_id: int,
) -> LogProbResult:
    """计算当前 model 对 rollout response tokens 的 logprob。

    先用 `build_response_mask` 得到完整序列上的 response-only mask，再做 causal LM
    shift：
    - `logits[:, :-1, :]` 预测下一个 token。
    - `labels[:, 1:]` 是被预测的目标 token。
    - `response_mask[:, 1:]` 与 shifted labels 对齐。

    因此 `compute_token_logprobs` 收到的是已经对齐好的 logits/labels/mask。
    """
    response_mask = build_response_mask(
        rollout_batch.input_ids,
        rollout_batch.prompt_lengths,
        rollout_batch.response_lengths,
        pad_token_id=pad_token_id,
    )
    outputs = model(
        input_ids=rollout_batch.input_ids,
        attention_mask=rollout_batch.attention_mask,
    )
    return compute_token_logprobs(
        outputs.logits[:, :-1, :],
        rollout_batch.labels[:, 1:],
        response_mask[:, 1:],
    )


def run_train_step(
    policy_runner: ModelRunner,
    reference_model: torch.nn.Module | None,
    optimizer: torch.optim.Optimizer,
    examples: list[GSM8KExample],
    train_cfg: TrainingConfig,
    step: int,
) -> TrainMetrics:
    """执行一个最小 GRPO update step。

    数据流：
    1. 用当前 policy 生成每个 prompt 的 group rollouts。
    2. 计算每个 prompt group 内的 reward 和 advantage。
    3. 把 completions padding 成一个训练 batch。
    4. 捕获 old policy logprobs，并重新计算带梯度的 new policy logprobs。
    5. 可选计算 reference model 的 k3 KL。
    6. 调用 `compute_grpo_loss`，反向传播并执行一次 Adam step。

    这个函数负责串联流程；GRPO 数学细节仍在各核心函数中实现。
    """
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
    """运行最小训练循环，并把每步 metrics 写入 `metrics.jsonl`。

    该函数负责加载数据、模型、可选 reference model、Adam optimizer，并循环调用
    `run_train_step`。当前版本不做 checkpoint/resume/scheduler，目标是验证核心
    GRPO 训练闭环能跑通。
    """
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
