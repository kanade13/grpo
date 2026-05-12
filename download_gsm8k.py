from datasets import load_dataset

dataset = load_dataset("openai/gsm8k", "main")

dataset["train"].to_json("data/gsm8k/train.jsonl", force_ascii=False)
dataset["test"].to_json("data/gsm8k/test.jsonl", force_ascii=False)