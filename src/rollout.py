from __future__ import annotations

from typing import TYPE_CHECKING

from .rewards import compute_gsm8k_reward
from .types import GeneratedCompletion, GSM8KExample, PromptRollout

if TYPE_CHECKING:
    from .model_runner import ModelRunner


def generate_group_rollouts(
    model_runner: ModelRunner,
    examples: list[GSM8KExample],
    group_size: int,
) -> list[PromptRollout]:
    """为每个 prompt 生成一组 completions。

    GRPO 不训练 value model，而是对同一个 prompt 的 G 个回答做组内比较。
    注意层级关系：
    - 一个 `GeneratedCompletion` = 同一道题的一次采样。
    - 一个 `PromptRollout` = 同一道题的 `group_size` 个采样。
    - 返回值 `list[PromptRollout]` = 当前 batch 内多道题的采样结果。

    所以 `response` 和 `reward` 不应该是 list。它们属于某一次采样；
    如果 `group_size = 4`，就创建 4 个 `GeneratedCompletion`，每个对象有自己
    的 `response`、`response_token_ids`、`reward`、`input_ids`。

    这个函数完成：
    1. 对每个 `GSM8KExample.prompt` 重复生成 `group_size` 个 completion。
    2. 把模型输出文本、response token ids、attention mask 和元信息封装为
       `GeneratedCompletion`。
    3. 为每个 completion 填入训练所需的 `input_ids` 和 `prompt_token_count`：
       - `input_ids` 必须是模型实际看到的 chat-template prompt token ids
         加 response token ids，不能用裸文本拼接后重新 tokenize。
       - `prompt_token_count` 是 input_ids 中 prompt 部分的 token 数。
       - `attention_mask` 对单条未 padding 的完整序列通常是
         `[1] * len(input_ids)`。它表示哪些 token 是真实 token。
         组成 batch 时，trainer 会 padding 到同一长度，并重新计算：
         `batch_attention_mask = (input_ids != pad_token_id).long()`，
         真实 token 为 1，padding token 为 0。

        这里的“input”是对训练讲的：训练需要的input，即包含prompt和response，而不是说“这轮模型输入输出中的input”
    4. 调用 reward 函数，把每个 completion 的答案 reward 写入对象。
    5. 返回 `list[PromptRollout]`，其中每个元素对应一个 prompt group。

    期望形状：
    - 输入 prompt 数量：B
    - 每个 prompt 的 completion 数量：G
    - 输出 completion 总数：B * G

    后续 `compute_group_advantages` 会在每个 prompt 内部对 G 个 reward 做归一化。
    """
    if group_size <= 0:
        raise ValueError("group_size must be a positive integer.")
    if not examples:
        return []

    flat_prompts = [example.prompt for example in examples for _ in range(group_size)]
    flat_results = model_runner.generate_batch(flat_prompts)
    if len(flat_results) != len(flat_prompts):
        raise ValueError("model_runner.generate_batch returned an unexpected number of results.")

    prompt_token_ids_by_id = {
        example.id: _encode_prompt_token_ids(model_runner, example.prompt)
        for example in examples
    }

    rollouts: list[PromptRollout] = []
    result_index = 0
    for example in examples:
        prompt_token_ids = prompt_token_ids_by_id[example.id]
        completions: list[GeneratedCompletion] = []
        for completion_id in range(group_size):
            result = flat_results[result_index]
            result_index += 1

            input_ids = prompt_token_ids + result.output_token_ids
            reward = compute_gsm8k_reward(
                response=result.content,
                gold_answer=example.final_answer,
            )
            completions.append(
                GeneratedCompletion(
                    prompt_id=example.id,
                    completion_id=completion_id,
                    prompt=example.prompt,
                    response=result.content,
                    response_token_ids=result.output_token_ids,
                    attention_mask=[1] * len(input_ids),
                    input_ids=input_ids,
                    prompt_token_count=len(prompt_token_ids),
                    reward=reward,
                    metadata={
                        "raw_text": result.raw_text,
                        "thinking_content": result.thinking_content,
                        "generated_token_count": result.generated_token_count,
                        "reached_max_new_tokens": result.reached_max_new_tokens,
                    },
                )
            )

        rollouts.append(
            PromptRollout(
                prompt_id=example.id,
                prompt=example.prompt,
                answer=example.final_answer,
                completions=completions,
            )
        )

    return rollouts


def _encode_prompt_token_ids(model_runner: ModelRunner, prompt: str) -> list[int]:
    formatted_prompt = model_runner._format_chat_prompt(prompt)
    encoded = model_runner.tokenizer(
        formatted_prompt,
        add_special_tokens=False,
    )
    return list(encoded["input_ids"])
