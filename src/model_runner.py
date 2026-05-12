from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer


THINK_END_TOKEN_ID = 151668


@dataclass
class GenerationResult:
    prompt: str
    raw_text: str
    thinking_content: str
    content: str
    output_token_ids: list[int]
    generated_token_count: int
    reached_max_new_tokens: bool


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(f"Config must be a YAML mapping: {config_path}")
    return config


def _require_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Config field '{key}' must be a mapping.")
    return value


class ModelRunner:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        model_cfg = _require_mapping(config, "model")
        generation_cfg = _require_mapping(config, "generation")

        model_path = model_cfg.get("path")
        if not model_path:
            raise ValueError("Config field 'model.path' is required.")

        self.enable_thinking = bool(generation_cfg.get("enable_thinking", True))
        self.generation_kwargs = self._build_generation_kwargs(generation_cfg)

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.padding_side = model_cfg.get("padding_side", "left")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=model_cfg.get("torch_dtype", "auto"),
            device_map=model_cfg.get("device_map", "auto"),
        )

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.model.generation_config.pad_token_id is None:
            self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
        self.stop_token_ids = self._build_stop_token_ids()

    @classmethod
    def from_config_file(cls, path: str | Path) -> "ModelRunner":
        return cls(load_yaml_config(path))

    def generate(self, prompt: str) -> GenerationResult:
        return self.generate_batch([prompt])[0]

    def generate_batch(self, prompts: list[str]) -> list[GenerationResult]:
        if not prompts:
            return []

        texts = [self._format_chat_prompt(prompt) for prompt in prompts]
        model_inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
        ).to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                **self.generation_kwargs,
            )

        prompt_width = model_inputs.input_ids.shape[1]
        results = []
        for row_index, prompt in enumerate(prompts):
            raw_output_token_ids = generated_ids[row_index][prompt_width:].tolist()
            output_token_ids, stopped_early = self._trim_generated_token_ids(raw_output_token_ids)
            generated_token_count = len(output_token_ids)
            raw_text = self.tokenizer.decode(output_token_ids, skip_special_tokens=True).strip("\n")
            thinking_content, content = self._split_thinking_content(output_token_ids)
            results.append(
                GenerationResult(
                    prompt=prompt,
                    raw_text=raw_text,
                    thinking_content=thinking_content,
                    content=content,
                    output_token_ids=output_token_ids,
                    generated_token_count=generated_token_count,
                    reached_max_new_tokens=(
                        not stopped_early
                        and len(raw_output_token_ids) >= self.generation_kwargs["max_new_tokens"]
                    ),
                )
            )
        return results

    def _format_chat_prompt(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )

    @staticmethod
    def _build_generation_kwargs(generation_cfg: dict[str, Any]) -> dict[str, Any]:
        supported_keys = {
            "max_new_tokens",
            "do_sample",
            "temperature",
            "top_p",
            "top_k",
            "min_p",
            "presence_penalty",
            "repetition_penalty",
            "num_return_sequences",
        }
        kwargs = {
            key: value
            for key, value in generation_cfg.items()
            if key in supported_keys and value is not None
        }
        if "max_new_tokens" not in kwargs:
            raise ValueError("Config field 'generation.max_new_tokens' is required.")
        if kwargs.get("num_return_sequences", 1) != 1:
            raise ValueError("Batched evaluation requires 'generation.num_return_sequences' to be 1.")
        return kwargs

    def _build_stop_token_ids(self) -> set[int]:
        stop_token_ids: set[int] = set()
        eos_token_id = self.model.generation_config.eos_token_id
        if isinstance(eos_token_id, int):
            stop_token_ids.add(eos_token_id)
        elif eos_token_id is not None:
            stop_token_ids.update(eos_token_id)
        if self.tokenizer.pad_token_id is not None:
            stop_token_ids.add(self.tokenizer.pad_token_id)
        return stop_token_ids

    def _trim_generated_token_ids(self, output_token_ids: list[int]) -> tuple[list[int], bool]:
        for index, token_id in enumerate(output_token_ids):
            if token_id in self.stop_token_ids:
                return output_token_ids[: index + 1], True
        return output_token_ids, False

    def _split_thinking_content(self, output_token_ids: list[int]) -> tuple[str, str]:
        try:
            index = len(output_token_ids) - output_token_ids[::-1].index(THINK_END_TOKEN_ID)
        except ValueError:
            index = 0

        thinking_content = self.tokenizer.decode(
            output_token_ids[:index],
            skip_special_tokens=True,
        ).strip("\n")
        content = self.tokenizer.decode(
            output_token_ids[index:],
            skip_special_tokens=True,
        ).strip("\n")
        return thinking_content, content
