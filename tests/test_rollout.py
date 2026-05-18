from dataclasses import dataclass

import pytest

from src.model_runner import GenerationResult
from src.rollout import generate_group_rollouts
from src.types import GSM8KExample


class FakeTokenizer:
    def __call__(self, text: str, add_special_tokens: bool = False):
        del add_special_tokens
        return {"input_ids": [ord(char) for char in text]}


@dataclass
class FakeRunner:
    tokenizer: FakeTokenizer

    def _format_chat_prompt(self, prompt: str) -> str:
        return f"<chat>{prompt}<assistant>"

    def generate_batch(self, prompts: list[str]) -> list[GenerationResult]:
        return [
            GenerationResult(
                prompt=prompt,
                raw_text=f"Reasoning {index}. \\boxed{{{index}}}",
                thinking_content="",
                content=f"\\boxed{{{index}}}",
                output_token_ids=[100 + index],
                generated_token_count=1,
                reached_max_new_tokens=False,
            )
            for index, prompt in enumerate(prompts)
        ]


def test_generate_group_rollouts_batches_and_groups_completions() -> None:
    examples = [
        GSM8KExample(
            id="train-0",
            question="q0",
            answer="a0",
            final_answer="0",
            prompt="prompt 0",
        ),
        GSM8KExample(
            id="train-1",
            question="q1",
            answer="a1",
            final_answer="2",
            prompt="prompt 1",
        ),
    ]

    rollouts = generate_group_rollouts(FakeRunner(FakeTokenizer()), examples, group_size=2)

    assert [rollout.prompt_id for rollout in rollouts] == ["train-0", "train-1"]
    assert [len(rollout.completions) for rollout in rollouts] == [2, 2]

    first_completion = rollouts[0].completions[0]
    prompt_token_ids = [ord(char) for char in "<chat>prompt 0<assistant>"]
    assert first_completion.prompt_id == "train-0"
    assert first_completion.completion_id == 0
    assert first_completion.response == "\\boxed{0}"
    assert first_completion.response_token_ids == [100]
    assert first_completion.input_ids == prompt_token_ids + [100]
    assert first_completion.prompt_token_count == len(prompt_token_ids)
    assert first_completion.attention_mask == [1] * len(first_completion.input_ids)
    assert first_completion.reward is not None
    assert first_completion.reward.is_correct is True
    assert first_completion.metadata["raw_text"].startswith("Reasoning")


def test_generate_group_rollouts_rejects_invalid_group_size() -> None:
    with pytest.raises(ValueError):
        generate_group_rollouts(FakeRunner(FakeTokenizer()), [], group_size=0)
