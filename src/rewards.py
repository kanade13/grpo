from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

from .types import RewardResult


BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")
NUMBER_RE = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")


def extract_prediction_answer(text: str) -> str | None:
    boxed_matches = BOXED_RE.findall(text)
    if boxed_matches:
        boxed_answer = boxed_matches[-1].strip()
        number_matches = NUMBER_RE.findall(boxed_answer)
        return number_matches[-1] if number_matches else boxed_answer

    number_matches = NUMBER_RE.findall(text)
    if number_matches:
        return number_matches[-1]
    return None


def normalize_answer_number(value: str | None) -> Decimal | None:
    if value is None:
        return None
    cleaned = value.strip().replace(",", "").replace("$", "")
    if cleaned.endswith("."):
        cleaned = cleaned[:-1]
    try:
        return Decimal(cleaned)
    except InvalidOperation:
        return None


def answers_equal(prediction: str | None, gold_answer: str) -> bool:
    pred_number = normalize_answer_number(prediction)
    gold_number = normalize_answer_number(extract_prediction_answer(gold_answer) or gold_answer)
    return pred_number is not None and gold_number is not None and pred_number == gold_number


def compute_gsm8k_reward(
    response: str,
    gold_answer: str,
    format_reward_weight: float = 0.1,
    length_penalty_weight: float = 0.0,
    max_response_chars: int | None = None,
) -> RewardResult:
    extracted_answer = extract_prediction_answer(response)
    is_correct = answers_equal(extracted_answer, gold_answer)
    answer_reward = 1.0 if is_correct else 0.0
    format_reward = format_reward_weight if BOXED_RE.search(response) else 0.0

    length_penalty = 0.0
    if max_response_chars is not None and len(response) > max_response_chars:
        overage = len(response) - max_response_chars
        length_penalty = -length_penalty_weight * overage / max_response_chars

    return RewardResult(
        total_reward=answer_reward + format_reward + length_penalty,
        answer_reward=answer_reward,
        format_reward=format_reward,
        length_penalty=length_penalty,
        extracted_answer=extracted_answer,
        is_correct=is_correct,
    )

