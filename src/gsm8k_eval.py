from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable, Iterable

from tqdm import tqdm
import yaml


BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")
NUMBER_RE = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")
OUTPUT_MODES = {"summary", "details"}
BYTES_PER_MIB = 1024 * 1024


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(f"Config must be a YAML mapping: {config_path}")
    return config


def load_jsonl(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if limit is not None and len(records) >= limit:
                break
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def extract_gsm8k_answer(answer_text: str) -> str:
    marker = "####"
    if marker not in answer_text:
        raise ValueError("GSM8K answer does not contain '####'.")
    return answer_text.rsplit(marker, 1)[1].strip()


def extract_prediction(text: str) -> str | None:
    boxed_matches = BOXED_RE.findall(text)
    if boxed_matches:
        boxed_answer = boxed_matches[-1].strip()
        number_matches = NUMBER_RE.findall(boxed_answer)
        return number_matches[-1] if number_matches else boxed_answer

    number_matches = NUMBER_RE.findall(text)
    if number_matches:
        return number_matches[-1]
    return None


def normalize_number(value: str | None) -> Decimal | None:
    if value is None:
        return None
    cleaned = value.strip().replace(",", "").replace("$", "")
    if cleaned.endswith("."):
        cleaned = cleaned[:-1]
    try:
        return Decimal(cleaned)
    except InvalidOperation:
        return None


def answers_match(prediction: str | None, gold_answer: str) -> bool:
    pred_number = normalize_number(prediction)
    gold_number = normalize_number(extract_prediction(gold_answer) or gold_answer)
    return pred_number is not None and gold_number is not None and pred_number == gold_number


def format_prompt(template: str, question: str) -> str:
    return template.format(question=question)


def iter_batches(items: list[dict[str, Any]], batch_size: int) -> Iterable[tuple[int, list[dict[str, Any]]]]:
    if batch_size <= 0:
        raise ValueError("Batch size must be a positive integer.")
    for start in range(0, len(items), batch_size):
        yield start, items[start : start + batch_size]


def iter_eval_rows(
    runner: Any,
    samples: list[dict[str, Any]],
    prompt_template: str,
    batch_size: int,
) -> Iterable[dict[str, Any]]:
    for start, batch in iter_batches(samples, batch_size):
        prompts = [format_prompt(prompt_template, sample["question"]) for sample in batch]
        results = runner.generate_batch(prompts)
        for offset, (sample, result) in enumerate(zip(batch, results, strict=True)):
            index = start + offset
            gold_answer = extract_gsm8k_answer(sample["answer"])
            prediction = extract_prediction(result.content)
            is_correct = answers_match(prediction, gold_answer)

            yield {
                "index": index,
                "question": sample["question"],
                "prompt": result.prompt,
                "gold_raw_answer": sample["answer"],
                "gold_answer": gold_answer,
                "prediction": prediction,
                "is_correct": is_correct,
                **asdict(result),
            }


def reset_cuda_peak_memory() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def get_cuda_peak_memory_summary() -> dict[str, int | float | None]:
    import torch

    if not torch.cuda.is_available():
        return {
            "cuda_peak_memory_allocated_bytes": None,
            "cuda_peak_memory_allocated_mib": None,
            "cuda_peak_memory_reserved_bytes": None,
            "cuda_peak_memory_reserved_mib": None,
        }

    allocated = torch.cuda.max_memory_allocated()
    reserved = torch.cuda.max_memory_reserved()
    return {
        "cuda_peak_memory_allocated_bytes": allocated,
        "cuda_peak_memory_allocated_mib": round(allocated / BYTES_PER_MIB, 2),
        "cuda_peak_memory_reserved_bytes": reserved,
        "cuda_peak_memory_reserved_mib": round(reserved / BYTES_PER_MIB, 2),
    }


def write_eval_output(
    rows: Iterable[dict[str, Any]],
    output_file: Path,
    total_rows: int,
    batch_size: int,
    output_mode: str,
    show_progress: bool = True,
    extra_summary_fn: Callable[[], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if output_mode not in OUTPUT_MODES:
        raise ValueError(f"Config field 'eval.output_mode' must be one of: {sorted(OUTPUT_MODES)}.")

    total = 0
    correct = 0
    max_new_tokens_reached = 0
    with output_file.open("w", encoding="utf-8") as f:
        for row in tqdm(rows, total=total_rows, desc="Evaluating GSM8K", disable=not show_progress):
            total += 1
            correct += int(row["is_correct"])
            max_new_tokens_reached += int(row.get("reached_max_new_tokens", False))
            if output_mode == "details":
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        accuracy = correct / total if total else 0.0
        summary = {
            "total": total,
            "correct": correct,
            "accuracy": accuracy,
            "batch_size": batch_size,
            "max_new_tokens_reached": max_new_tokens_reached,
        }
        if extra_summary_fn is not None:
            summary.update(extra_summary_fn())
        if output_mode == "summary":
            json.dump(summary, f, ensure_ascii=False, indent=2)
            f.write("\n")

    return summary


def run_evaluation(
    config: dict[str, Any],
    limit: int | None = None,
    output_path: str | None = None,
    batch_size: int | None = None,
    output_mode: str | None = None,
) -> dict[str, Any]:
    data_cfg = _require_mapping(config, "data")
    prompt_cfg = _require_mapping(config, "prompt")
    eval_cfg = _require_mapping(config, "eval")

    data_path = data_cfg.get("path")
    prompt_template = prompt_cfg.get("template")
    if not data_path:
        raise ValueError("Config field 'data.path' is required.")
    if not prompt_template:
        raise ValueError("Config field 'prompt.template' is required.")

    eval_limit = limit if limit is not None else eval_cfg.get("limit")
    eval_output = output_path or eval_cfg.get("output_path")
    eval_batch_size = batch_size if batch_size is not None else eval_cfg.get("batch_size", 1)
    eval_output_mode = output_mode or eval_cfg.get("output_mode", "summary")
    if not eval_output:
        raise ValueError("Config field 'eval.output_path' is required.")
    if not isinstance(eval_batch_size, int) or eval_batch_size <= 0:
        raise ValueError("Config field 'eval.batch_size' must be a positive integer.")
    if eval_output_mode not in OUTPUT_MODES:
        raise ValueError(f"Config field 'eval.output_mode' must be one of: {sorted(OUTPUT_MODES)}.")

    samples = load_jsonl(data_path, eval_limit)
    from .model_runner import ModelRunner

    runner = ModelRunner(config)

    output_file = Path(eval_output)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    reset_cuda_peak_memory()
    rows = iter_eval_rows(runner, samples, prompt_template, eval_batch_size)
    summary = write_eval_output(
        rows=rows,
        output_file=output_file,
        total_rows=len(samples),
        batch_size=eval_batch_size,
        output_mode=eval_output_mode,
        extra_summary_fn=get_cuda_peak_memory_summary,
    )
    summary["output_mode"] = eval_output_mode
    summary["output_path"] = str(output_file)
    return summary


def _require_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Config field '{key}' must be a mapping.")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Qwen3 on local GSM8K JSONL.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--limit", type=int, default=None, help="Override eval.limit.")
    parser.add_argument("--output", default=None, help="Override eval.output_path.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override eval.batch_size.")
    parser.add_argument(
        "--output-mode",
        choices=sorted(OUTPUT_MODES),
        default=None,
        help="Override eval.output_mode. 'summary' writes aggregate metrics; 'details' writes per-sample JSONL.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    run_evaluation(
        config,
        limit=args.limit,
        output_path=args.output,
        batch_size=args.batch_size,
        output_mode=args.output_mode,
    )


if __name__ == "__main__":
    main()
