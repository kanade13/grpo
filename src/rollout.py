from __future__ import annotations

from .types import GSM8KExample, PromptRollout


def generate_group_rollouts(
    model_runner,
    examples: list[GSM8KExample],
    group_size: int,
) -> list[PromptRollout]:
    """为每个 prompt 生成一组 completions。

    GRPO 不训练 value model，而是对同一个 prompt 的 G 个回答做组内比较。
    这个函数应该完成：
    1. 对每个 `GSM8KExample.prompt` 重复生成 `group_size` 个 completion。
    2. 把模型输出文本、response token ids、attention mask 和元信息封装为
       `GeneratedCompletion`。
    3. 调用 reward 函数，把每个 completion 的答案 reward 写入对象。
    4. 返回 `list[PromptRollout]`，其中每个元素对应一个 prompt group。

    期望形状：
    - 输入 prompt 数量：B
    - 每个 prompt 的 completion 数量：G
    - 输出 completion 总数：B * G

    后续 `compute_group_advantages` 会在每个 prompt 内部对 G 个 reward 做归一化。
    """
    raise NotImplementedError("核心 rollout 逻辑留给手写实现。")

