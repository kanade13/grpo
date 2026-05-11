# GRPO 手写开发时间表
## 基本假设

- 第一个目标模型：`Qwen/Qwen3-0.6B`
- 第一阶段目标任务：加载 `Qwen/Qwen3-0.6B`，并在 GSM8K 的 `test` 划分上测试 zero-shot 或 few-shot 准确率，得到训练前基线。
- 第一个训练模式：单 GPU，小 batch size，不做分布式训练。
- 目标：理解并手写 GRPO 核心逻辑，而不是追求 benchmark 分数。
- 每日代码量目标：约 100 行。非代码任务约 2 到 3 小时。

## 每日安排

| Day | 实现什么功能 | 如何测试 | 完成情况 |
| --- | --- | --- | --- |
| Day 1 | 创建项目骨架：`src/grpo_handwritten/`、`tests/`、`scripts/`、`configs/`、`runs/`、`data/` 和一份简短的 `README.md`。添加最小 package 文件和 CLI 入口。 | 以 editable 模式安装包后运行 `python -m grpo_handwritten --help`。确认命令能打印帮助文本。 | TODO |
| Day 2 | 添加依赖检查脚本 `scripts/check_env.py`。打印 Python、PyTorch、CUDA、GPU 名称、transformers、datasets、accelerate、peft、可用显存，并确认 `datasets` 能访问 `openai/gsm8k`。 | 运行 `python scripts/check_env.py`。它必须无 import error 地结束，并清楚打印 CUDA 是否可用；再运行一个只加载 1 条 GSM8K 样本的 smoke test。 | TODO |
| Day 3 | 实现 `src/grpo_handwritten/modeling.py` 和 `scripts/smoke_load_qwen3.py`。加载 `Qwen/Qwen3-0.6B` 的 tokenizer 和 causal LM，设置 pad token，选择 `cuda` 或 `cpu`，打印 dtype、参数量和显存占用。 | 运行 `python scripts/smoke_load_qwen3.py --model Qwen/Qwen3-0.6B`。确认模型能加载，并能对一个简短 prompt 完成 tokenize。 | TODO |
| Day 4 | 实现 `src/grpo_handwritten/generate.py`。对 `Qwen/Qwen3-0.6B` 做批量生成封装，支持 `max_new_tokens`、`temperature`、`top_p`、`do_sample` 和 deterministic greedy decoding。 | 运行 `python scripts/smoke_generate.py --model Qwen/Qwen3-0.6B --prompt "What is 2+3?" --max-new-tokens 64`。确认返回非空文本，且 greedy 模式重复运行结果一致。 | TODO |
| Day 5 | 实现 GSM8K test 数据加载与答案解析：`src/grpo_handwritten/gsm8k.py`。加载 `openai/gsm8k` 的 `main` 配置和 `test` 划分，提取标准答案中的最终数字，保存 prompt、raw answer、final answer。 | 添加 `tests/test_gsm8k.py`。运行 `pytest tests/test_gsm8k.py`，并运行 `python scripts/preview_gsm8k.py --split test --limit 5`，确认能看到题目和解析后的答案。 | TODO |
| Day 6 | 实现第一阶段基线评估脚本 `scripts/eval_gsm8k_qwen3.py`。在 GSM8K `test` 划分上调用 `Qwen/Qwen3-0.6B` 生成答案，解析模型最终数字，计算 accuracy，并把样本、预测、标签、是否正确写入 JSONL。 | 先运行 `python scripts/eval_gsm8k_qwen3.py --model Qwen/Qwen3-0.6B --split test --limit 20 --output runs/qwen3_gsm8k_test_20.jsonl`。确认打印 accuracy、总样本数、正确数；再去掉 `--limit` 跑完整 test，记录训练前基线。 | TODO |
| Day 7 | 基于 Day 5 的 GSM8K loader，扩展训练数据接口。支持加载 GSM8K `train` 划分、截取小样本 debug subset，并统一输出 `id`、`prompt`、`answer`、`final_answer`。 | 运行 `pytest tests/test_data.py`。再运行 `python scripts/preview_gsm8k.py --split train --limit 10`，确认打印出 10 条训练 prompts 和解析答案。 | TODO |
| Day 8 | 在 `rewards.py` 中实现简单 reward 函数：精确答案匹配、整数提取、格式 reward 和长度惩罚。 | 运行 `pytest tests/test_rewards.py`。覆盖正确答案、错误答案、没有数字、过长回复等 case。 | TODO |
| Day 9 | 实现 `rollout.py`，为每个 prompt 生成 `G` 个 completions。保存 prompt 文本、response 文本、token ids、attention masks、answer 和 reward。 | 运行 `python scripts/smoke_rollout.py --num-prompts 2 --group-size 4`。确认产出 8 个 completions，且每个 completion 都有一个 reward。 | TODO |
| Day 10 | 实现组内 reward 归一化：对每个 prompt group 计算 mean、std 和 advantage `(reward - mean) / (std + eps)`。 | 运行 `pytest tests/test_advantages.py`。测试 reward 全相等、混合 reward、单 group 和多 group。 | TODO |
| Day 11 | 学习 policy gradient 基础。写 `notes/day11_policy_gradient.md`，解释 log probabilities、advantage、KL penalty，以及为什么 GRPO 不需要 value model。 | 手动阅读笔记。复现一个 5 行 tensor 例子，用来计算 `advantage * logprob`。 | TODO |
| Day 12 | 实现 `logprobs.py`：给定模型 logits 和目标 token ids，使用 `log_softmax` 与 `gather` 计算逐 token log probabilities。 | 运行 `pytest tests/test_logprobs.py`。用一个很小的 fake logits tensor 和手算结果对比。 | TODO |
| Day 13 | 实现 response masking。构建 mask，使 loss 只覆盖生成的 response tokens，并排除 prompt padding 和 prompt tokens。 | 运行 `pytest tests/test_masks.py`。测试 left padding、right padding、短 response 和空 response 边界情况。 | TODO |
| Day 14 | 实现 old-policy log probability 捕获。在 rollout 阶段，优化步骤之前保存当前 policy 的 log probs。 | 运行 `python scripts/smoke_old_logprobs.py`。确认保存的 logprob tensor shape 与生成 response token shape 一致。 | TODO |
| Day 15 | 实现 reference model 加载。加载初始模型的冻结副本，或支持在早期 CPU 测试中关闭 reference KL。 | 运行 `python scripts/smoke_reference.py`。确认所有 reference model 参数都是 `requires_grad=False`。 | TODO |
| Day 16 | 实现 policy 与 reference model 在生成 tokens 上的逐 token KL 估计。只对 response tokens 做 mask。 | 运行 `pytest tests/test_kl.py`。使用小型 fake logprob tensors，并手动验证 masked KL 数值。 | TODO |
| Day 17 | 在 `losses.py` 中实现 GRPO clipped objective：ratio、clipped ratio、advantage weighting、KL penalty、response mask 和最终 scalar loss。 | 运行 `pytest tests/test_grpo_loss.py`。包含 clipping 改变 loss 的测试，以及 mask 把 token 置零的测试。 | TODO |
| Day 18 | 添加一个很小的 fake model training test。使用最小随机初始化 LM 或 mocked logits，确认 loss 能反向传播且参数收到梯度。 | 运行 `pytest tests/test_backward.py`。断言 loss 是 finite，且至少一个可训练参数有非零梯度。 | TODO |
| Day 19 | 实现 optimizer 和 scheduler builder。支持 AdamW、learning rate、weight decay、gradient clipping 和 warmup ratio。 | 运行 `pytest tests/test_optim.py`。确认 optimizer 能更新一个小参数，并且 gradient clipping 会限制 norm。 | TODO |
| Day 20 | 构建第一个训练循环 `trainer.py`：加载 config、加载 model、采样 prompts、rollout、计算 advantages、计算 loss、backward、optimizer step。 | 运行 `python scripts/train_tiny.py --max-steps 1`。必须完整跑完一次 GRPO update，且没有 NaN。 | TODO |
| Day 21 | 添加结构化 metrics：loss、mean reward、reward std、KL、response length、grad norm、learning rate、tokens/sec 和 GPU memory。 | 运行 `python scripts/train_tiny.py --max-steps 3`。确认每一步都打印 metrics，且数值都是 finite。 | TODO |
| Day 22 | 添加 TensorBoard logging。写入 scalar metrics，以及少量采样 prompt/response/reward 文本记录。 | 运行 `python scripts/train_tiny.py --max-steps 3 --tensorboard runs/tb_test`，然后运行 `tensorboard --logdir runs/tb_test` 并在本地检查。 | TODO |
| Day 23 | 实现 checkpoint 保存：model、tokenizer、optimizer、scheduler、step number 和 config snapshot。 | 运行 `python scripts/train_tiny.py --max-steps 2 --save-every 1`。确认 checkpoint 目录存在且包含模型文件。 | TODO |
| Day 24 | 实现 checkpoint resume。加载 model 和 optimizer state，并从保存的 step 继续训练。 | 先训练 2 steps，再 resume 到 4 steps，确认日志从 step 3 继续，而不是重新从头开始。 | TODO |
| Day 25 | 通过 PEFT 添加 LoRA 支持。让 full fine-tuning 和 LoRA 都能在 config 中选择。 | 运行 `python scripts/smoke_lora.py`。确认 trainable parameter count 明显小于 total parameter count。 | TODO |
| Day 26 | 添加显存控制：gradient accumulation、bf16/fp16 选择、max prompt length、max response length 和 batch size 校验。 | 运行 `python scripts/train_tiny.py --max-steps 2 --gradient-accumulation-steps 2`。确认 optimizer steps 按预期间隔发生。 | TODO |
| Day 27 | 添加确定性的离线单元测试路径，使用极小 synthetic tokenizer/model fixture，使核心测试不需要下载真实模型。 | 运行 `pytest`。确认大多数测试不需要网络或 GPU 也能通过。 | TODO |
| Day 28 | 创建第一个真实配置 `configs/grpo_qwen3_0_6b_gsm8k.yaml`，用于在 GSM8K `train` 小样本上做短数学答案 GRPO 训练。 | 运行 `python scripts/train.py --config configs/grpo_qwen3_0_6b_gsm8k.yaml --max-steps 5`。确认训练能启动且显存够用。 | TODO |
| Day 29 | 实现 evaluation script。为 GSM8K held-out prompts 生成答案，并计算 exact-match reward、format reward、average length 和 sample outputs。 | 运行 `python scripts/eval.py --checkpoint <checkpoint> --dataset gsm8k --split test --limit 100`。确认打印 metrics summary 和 sample table。 | TODO |
| Day 30 | 添加训练前后评估 workflow。评估 base model，训练一个小 GRPO run，评估 checkpoint，并写出 comparison JSON。 | 运行 `python scripts/compare_eval.py`。确认报告 base reward、trained reward 和 delta。 | TODO |
| Day 31 | 提升 reward 鲁棒性：处理多种数字格式、空白字符、final-answer tags 和无效 completions。 | 运行 `pytest tests/test_rewards.py`，再对 20 条样例运行 `python scripts/eval.py`。确认 reward parser 不崩溃。 | TODO |
| Day 32 | 添加 sample inspection tool，把 prompts、group completions、rewards、advantages 和 selected tokens 保存到 JSONL，方便调试。 | 运行 `python scripts/inspect_rollouts.py --num-prompts 3 --group-size 4`。手动检查 JSONL，确认 group 可读。 | TODO |
| Day 33 | 添加 NaN 和不稳定性保护：检查 finite loss、finite rewards、max KL threshold warning、gradient norm warning，并在中止前自动保存 checkpoint。 | 创建一个 learning rate 过高的 config 并运行 3 steps。确认会出现 warning 或干净中止，而不是静默坏掉。 | TODO |
| Day 34 | 添加命令行易用性：把 `train`、`eval`、`generate`、`inspect-rollouts` 和 `check-env` 子命令整合到一个 package CLI 下。 | 分别运行每个 CLI 的 `--help` 和一个 smoke command。确认所有命令都能从 repo root 工作。 | TODO |
| Day 35 | 编写算法文档 `docs/grpo_algorithm.md`：data flow、tensors、shapes、loss formula、masks 和常见 bug。 | 对照代码名称检查文档。运行 `pytest`，确认示例或引用的函数仍然存在。 | TODO |
| Day 36 | 为这个代码库编写 PyTorch 新手笔记：`requires_grad`、`backward`、`no_grad`、model train/eval modes、devices 和 mixed precision。 | 阅读笔记并运行其中的 snippets。确认它们能在环境中执行。 | TODO |
| Day 37 | 运行一次更长的 50 到 100 step LoRA 实验。追踪 reward、KL、length 和 sample outputs。 | 运行 `python scripts/train.py --config configs/grpo_qwen3_0_6b_gsm8k.yaml --max-steps 100`。确认 metrics 保持 finite，且 checkpoint 正常保存。 | TODO |
| Day 38 | 分析实验结果。判断 reward 提升是因为答案真的变好，还是因为 reward function 被利用。把笔记写入 `experiments/`。 | 在 held-out prompts 上运行 evaluation，并手动检查至少 30 条 samples。写一份简短实验报告。 | TODO |
| Day 39 | 修复实验中发现的前两个问题，例如输出过长、reward hacking、KL 不稳定或 prompt formatting 较弱。 | 重新运行一次 20-step 实验。确认目标 metric 朝预期方向变化。 | TODO |
| Day 40 | 最终清理：type hints、有用的 docstrings、README commands、config comments，以及删除废弃 scripts。 | 运行 `ruff check .`、`pytest`、`python scripts/check_env.py`，以及一次 1-step train smoke test。 | TODO |

## 里程碑

| 里程碑 | 目标天数 | 退出标准 | 完成情况 |
| --- | --- | --- | --- |
| M1：Qwen3 + GSM8K 基线评估 | Day 1 到 Day 6 | 环境已验证，`Qwen/Qwen3-0.6B` 能加载，GSM8K `test` 划分能读取，并得到完整 test accuracy 基线。 | TODO |
| M2：GSM8K 数据与 rewards | Day 7 到 Day 10 | GSM8K train/test 数据能加载，rewards 能计算，group advantages 有测试覆盖。 | TODO |
| M3：GRPO 数学核心 | Day 11 到 Day 18 | Logprobs、masks、KL 和 GRPO loss 都有单元测试。 | TODO |
| M4：第一个训练循环 | Day 19 到 Day 24 | 一次完整 update 能运行，metrics 能记录，checkpoint 能保存和恢复。 | TODO |
| M5：实用训练能力 | Day 25 到 Day 33 | LoRA、显存控制、eval、rollout inspection 和稳定性保护都能工作。 | TODO |
| M6：可用项目 | Day 34 到 Day 40 | CLI、docs、实验报告、lint、tests 和 smoke training 都完成。 | TODO |

## 推荐的第一阶段编码顺序

1. 不要从 GRPO loss 开始。第一阶段先完成 `Qwen/Qwen3-0.6B` 在 GSM8K `test` 上的准确率基线，因为训练前必须知道原模型水平。
2. 先用 greedy decoding 做可复现评估，再考虑 temperature sampling。否则准确率波动会让后续训练效果难以判断。
3. 先把 GSM8K 答案解析写扎实。数学任务的第一批 bug 通常不是模型问题，而是最终数字提取和格式处理问题。
4. 一开始保持 reward 函数为规则型。Learned reward model 会在基础循环稳定前引入另一个训练问题。
5. 把每个 tensor shape 都写下来。大多数 GRPO bug 都是 mask、padding 和 logprob 对齐问题。
6. 把 TRL 当成参考，而不是依赖路径。卡住时可以对比行为，但实现路径保持手写。
