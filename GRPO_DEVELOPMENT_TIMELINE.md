# GRPO 手写开发时间表

## 基本假设

- 第一个目标模型：`Qwen/Qwen3-0.6B`，本地路径为 `./models/Qwen3-0.6B`。
- 第一阶段目标任务：加载模型，并在 GSM8K 的 `test` 划分上得到训练前基线。
- 第一个训练模式：单 GPU，小 batch size，不做分布式训练。
- 目标：理解并手写 GRPO 核心逻辑，而不是追求 benchmark 分数。
- 数据优先使用本地 `./data/gsm8k/*.jsonl`，避免训练和评测流程依赖网络。
- 暂时跳过 checkpoint、resume、复杂 scheduler 等边缘能力，优先打通 GRPO 核心数学和最小训练闭环。

## 当前占位函数实现顺序

推荐顺序按依赖关系排列：

1. `generate_group_rollouts`：先能为每个 prompt 生成 G 个回答，并给每个回答算 reward。
2. `compute_group_advantages`：有 reward 后才能做组内归一化 advantage。
3. `build_response_mask`：后续 logprob、KL、loss 都依赖 response-only mask。
4. `compute_token_logprobs`：训练目标和 old-policy logprob 捕获都依赖逐 token logprob。
5. `compute_k3_kl`：有 policy/reference logprob 和 mask 后再实现 KL。
6. `compute_grpo_loss`：最后把 ratio clipping、advantage 和 KL 组合成最终 loss。

## 每日安排

| Day | 实现什么功能 | 如何测试 | 完成情况 |
| --- | --- | --- | --- |
| Day 1 | 完成模型加载与调用代码，与测评代码。包括 `ModelRunner`、Qwen3 thinking/non-thinking 配置、GSM8K test 评测、批量生成、summary/details 输出和基础解析测试。 | 运行 `python3 -m pytest tests/test_gsm8k_eval.py tests/test_model_runner.py -q`、`python3 -m ruff check src tests`，并用 `python3 -m src.gsm8k_eval --config configs/qwen3_gsm8k_non_thinking.yaml --limit 32` 做小样本评测。 | 已完成 |
| Day 2 | 实现 `generate_group_rollouts`。对每个 `GSM8KExample` 生成 `group_size` 个 completion，封装为 `PromptRollout` / `GeneratedCompletion`，并调用 `compute_gsm8k_reward` 写入 reward。 | 新增 rollout 单元测试，使用 fake `ModelRunner` 返回固定 completion；确认 prompt 数量、每组 completion 数量、reward、token ids、metadata 都正确。 | TODO |
| Day 3 | 实现 `compute_group_advantages`。对每个 prompt group 的 reward 做均值、标准差和 advantage 计算。 | 新增 advantage 单元测试，覆盖普通 reward、全相等 reward、单元素 group、负 reward；确认全相等时 advantages 为 0。 | TODO |
| Day 4 | 实现 `build_response_mask`。构造只覆盖 response tokens 的 mask，排除 prompt tokens 和 padding tokens。 | 新增 mask 单元测试，覆盖 right padding、left padding、不同 prompt length、短 response、全 padding 边界。 | TODO |
| Day 5 | 实现 `compute_token_logprobs`，并明确 logits/target shift 约定。用 `log_softmax + gather` 得到生成 token 的逐 token logprob 和 sequence logprob。 | 新增 logprob 单元测试，使用小型 fake logits 和手算结果对齐；覆盖 mask 聚合和 padding token 不参与 sequence logprob。 | TODO |
| Day 6 | 实现 `compute_k3_kl`。基于 policy/reference logprob 和 response mask 计算 k3 KL：`exp(logr) - logr - 1`，其中 `logr = ref_logprob - policy_logprob`。 | 新增 KL 单元测试，使用手算 tensor 验证 token KL、masked KL、mean KL；确认 mask 为 0 的位置不影响均值。 | TODO |
| Day 7 | 实现 `compute_grpo_loss`，把 `ratio = exp(new-old)`、clipped ratio、group advantage、response mask 和 KL penalty 组合成最终 loss。随后接一个最小 fake training smoke test。 | 新增 GRPO loss 单元测试，覆盖正/负 advantage、clip 生效、不生效、mask、KL penalty；运行一个 fake 参数反传测试，确认 loss finite 且参数有梯度。 | TODO |

## 里程碑

| 里程碑 | 目标天数 | 退出标准 | 完成情况 |
| --- | --- | --- | --- |
| M1：Qwen3 + GSM8K 基线评测 | Day 1 | 模型能加载，GSM8K test 能批量评测，summary 能记录 accuracy、max token 截断数和显存峰值。 | 已完成 |
| M2：Rollout 与组内奖励 | Day 2 到 Day 3 | 每个 prompt 能生成 G 个 completion，reward 和 advantage 都有测试覆盖。 | TODO |
| M3：Token 级训练张量 | Day 4 到 Day 5 | response mask 和 token logprob 正确，能稳定产出 loss 所需张量。 | TODO |
| M4：KL 与 GRPO Loss | Day 6 到 Day 7 | k3 KL 和 clipped GRPO objective 有手算单元测试，fake backward 能跑通。 | TODO |

## 推荐执行原则

1. 每实现一个占位函数，就把 `tests/test_core_placeholders.py` 中对应的 `NotImplementedError` 断言替换成数值测试。
2. 所有核心数学先用小 tensor 手算测试，不先接真实 Qwen 模型。
3. 实现顺序不要跳过 mask 和 logprob；GRPO loss 的大多数 bug 都来自 token 对齐和 mask 错位。
4. `old_logprobs`、`advantages`、`reference_logprobs` 默认不带梯度；只有 `new_logprobs` 连接当前 policy 梯度。
5. Day 7 结束前只追求 fake training 反传正确，不急着做 checkpoint、resume、TensorBoard 或长训练。
