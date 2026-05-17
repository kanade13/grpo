from decimal import Decimal

from src.rewards import (
    answers_equal,
    compute_gsm8k_reward,
    extract_prediction_answer,
    normalize_answer_number,
)


def test_extract_prediction_answer_prefers_boxed_answer() -> None:
    assert extract_prediction_answer("First 12, final \\boxed{18}.") == "18"


def test_normalize_answer_number() -> None:
    assert normalize_answer_number("$1,234.") == Decimal("1234")
    assert normalize_answer_number(None) is None


def test_answers_equal() -> None:
    assert answers_equal("1,234", "1234")
    assert not answers_equal("1235", "1234")


def test_compute_gsm8k_reward() -> None:
    reward = compute_gsm8k_reward("\\boxed{18}", "18")
    assert reward.is_correct is True
    assert reward.answer_reward == 1.0
    assert reward.format_reward > 0
    assert reward.total_reward > 1.0


def test_compute_gsm8k_reward_length_penalty() -> None:
    reward = compute_gsm8k_reward(
        "x" * 20,
        "18",
        length_penalty_weight=0.5,
        max_response_chars=10,
    )
    assert reward.is_correct is False
    assert reward.length_penalty < 0

