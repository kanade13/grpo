from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, TypeVar

from .types import GSM8KExample


T = TypeVar("T")


def extract_gsm8k_final_answer(answer_text: str) -> str:
    marker = "####"
    if marker not in answer_text:
        raise ValueError("GSM8K answer does not contain '####'.")
    return answer_text.rsplit(marker, 1)[1].strip()


def format_gsm8k_prompt(question: str, template: str) -> str:
    return template.format(question=question)


def load_gsm8k_jsonl(
    path: str | Path,
    split: str,
    prompt_template: str,
    limit: int | None = None,
) -> list[GSM8KExample]:
    examples: list[GSM8KExample] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for index, line in enumerate(f):
            if limit is not None and len(examples) >= limit:
                break
            record = json.loads(line)
            question = record["question"]
            answer = record["answer"]
            examples.append(
                GSM8KExample(
                    id=f"{split}-{index}",
                    question=question,
                    answer=answer,
                    final_answer=extract_gsm8k_final_answer(answer),
                    prompt=format_gsm8k_prompt(question, prompt_template),
                )
            )
    return examples


def batch_iter(items: list[T], batch_size: int) -> Iterable[list[T]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]

