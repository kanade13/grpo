import json
from dataclasses import dataclass
from decimal import Decimal

from src.gsm8k_eval import (
    answers_match,
    extract_gsm8k_answer,
    extract_prediction,
    format_prompt,
    iter_eval_rows,
    normalize_number,
    write_eval_output,
)


@dataclass
class FakeGenerationResult:
    prompt: str
    raw_text: str
    thinking_content: str
    content: str
    output_token_ids: list[int]
    generated_token_count: int
    reached_max_new_tokens: bool


class FakeRunner:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def generate_batch(self, prompts: list[str]) -> list[FakeGenerationResult]:
        self.batch_sizes.append(len(prompts))
        return [
            FakeGenerationResult(
                prompt=prompt,
                raw_text="\\boxed{18}",
                thinking_content="",
                content="\\boxed{18}",
                output_token_ids=[1, 2, 3],
                generated_token_count=3,
                reached_max_new_tokens=False,
            )
            for prompt in prompts
        ]


def test_extract_gsm8k_answer() -> None:
    answer = "Some reasoning\n#### 70,000"
    assert extract_gsm8k_answer(answer) == "70,000"


def test_extract_prediction_prefers_boxed_answer() -> None:
    text = "The final value is 12, but after checking, \\boxed{18}."
    assert extract_prediction(text) == "18"


def test_extract_prediction_falls_back_to_last_number() -> None:
    text = "First compute 9 eggs. The answer is $18."
    assert extract_prediction(text) == "18"


def test_normalize_number() -> None:
    assert normalize_number("$70,000") == Decimal("70000")
    assert normalize_number("18.") == Decimal("18")
    assert normalize_number(None) is None


def test_answers_match_numeric_formats() -> None:
    assert answers_match("$70,000", "70000")
    assert answers_match(extract_prediction("\\boxed{18}"), "18")
    assert not answers_match("17", "18")


def test_format_prompt() -> None:
    template = "{question}\nPlease reason step by step."
    assert format_prompt(template, "What is 2+3?").startswith("What is 2+3?")


def test_iter_eval_rows_outputs_prediction_and_label() -> None:
    samples = [{"question": "How much?", "answer": "Reasoning\n#### 18"}]
    rows = list(iter_eval_rows(FakeRunner(), samples, "{question}", batch_size=1))
    assert rows[0]["gold_raw_answer"] == "Reasoning\n#### 18"
    assert rows[0]["gold_answer"] == "18"
    assert rows[0]["prediction"] == "18"
    assert rows[0]["is_correct"] is True


def test_iter_eval_rows_batches_samples() -> None:
    samples = [
        {"question": f"Question {index}?", "answer": "Reasoning\n#### 18"}
        for index in range(5)
    ]
    runner = FakeRunner()
    rows = list(iter_eval_rows(runner, samples, "{question}", batch_size=2))
    assert len(rows) == 5
    assert runner.batch_sizes == [2, 2, 1]
    assert [row["index"] for row in rows] == [0, 1, 2, 3, 4]


def test_write_eval_output_summary_mode(tmp_path) -> None:
    rows = [
        {"is_correct": True, "prediction": "18", "reached_max_new_tokens": True},
        {"is_correct": False, "prediction": "17", "reached_max_new_tokens": False},
    ]
    output_file = tmp_path / "summary.json"
    summary = write_eval_output(rows, output_file, total_rows=2, batch_size=16, output_mode="summary", show_progress=False)

    assert summary == {
        "total": 2,
        "correct": 1,
        "accuracy": 0.5,
        "batch_size": 16,
        "max_new_tokens_reached": 1,
    }
    assert json.loads(output_file.read_text(encoding="utf-8")) == summary


def test_write_eval_output_details_mode(tmp_path) -> None:
    rows = [
        {"is_correct": True, "prediction": "18"},
        {"is_correct": False, "prediction": "17"},
    ]
    output_file = tmp_path / "details.jsonl"
    summary = write_eval_output(rows, output_file, total_rows=2, batch_size=8, output_mode="details", show_progress=False)
    lines = output_file.read_text(encoding="utf-8").strip().splitlines()

    assert summary == {"total": 2, "correct": 1, "accuracy": 0.5, "batch_size": 8, "max_new_tokens_reached": 0}
    assert len(lines) == 2
    assert json.loads(lines[0]) == {"is_correct": True, "prediction": "18"}
