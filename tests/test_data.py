import json

from src.data import batch_iter, extract_gsm8k_final_answer, format_gsm8k_prompt, load_gsm8k_jsonl


def test_extract_gsm8k_final_answer() -> None:
    assert extract_gsm8k_final_answer("reasoning\n#### 1,234") == "1,234"


def test_format_gsm8k_prompt() -> None:
    prompt = format_gsm8k_prompt("What is 2+3?", "{question}\nAnswer in \\boxed{{}}.")
    assert prompt == "What is 2+3?\nAnswer in \\boxed{}."


def test_load_gsm8k_jsonl(tmp_path) -> None:
    data_file = tmp_path / "test.jsonl"
    data_file.write_text(
        json.dumps({"question": "What is 2+3?", "answer": "2+3=5\n#### 5"}) + "\n",
        encoding="utf-8",
    )

    examples = load_gsm8k_jsonl(data_file, split="test", prompt_template="{question}", limit=1)

    assert examples[0].id == "test-0"
    assert examples[0].final_answer == "5"
    assert examples[0].prompt == "What is 2+3?"


def test_batch_iter() -> None:
    assert list(batch_iter([1, 2, 3, 4, 5], batch_size=2)) == [[1, 2], [3, 4], [5]]

