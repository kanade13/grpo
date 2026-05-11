# GRPO 手写开发时间表

## 环境检查

当前 `environment.yml` 足够支撑一个最小化的单 GPU GRPO 原型：

- `python=3.10`、`pytorch=2.5.*`、`pytorch-cuda=12.1`：如果 NVIDIA 驱动支持 CUDA 12.1 runtime，就足够进行本地 CUDA 训练。
- `transformers`、`datasets`、`accelerate`、`peft`、`safetensors`、`sentencepiece`：足够加载常见的 Hugging Face 因果语言模型和数据集。
- `trl`：可以作为参考实现使用；如果目标是手写算法，项目代码可以不直接调用 TRL 的 GRPO trainer。
- `numpy<2`：这是保守且合理的选择，因为不少机器学习包在 NumPy 2 附近仍可能有兼容性边界。

不过，这份配置对“边学习边手写开发”还不够理想，因为它缺少测试、日志、绘图和配置管理工具。建议补充：

```yaml
  - matplotlib
  - pyyaml
  - psutil
  - pip:
      - pytest>=8,<9
      - ruff>=0.8,<1
      - tensorboard>=2.18,<3
      - rich>=13,<14
      - einops>=0.8,<1
```

后续可选补充：

- `bitsandbytes`：只有在低显存下需要 4-bit 或 8-bit 加载时再加。
- `wandb`：只有需要远程实验追踪时再加。
- `evaluate`：只有使用标准 NLP 指标，而不是自定义 reward 函数时再加。

最小环境验证：

```bash
conda env create -f environment.yml
conda activate grpo
export PYTHONNOUSERSITE=1
python - <<'PY'
import torch, transformers, datasets, accelerate, peft, trl
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("gpu_count", torch.cuda.device_count())
print("transformers", transformers.__version__)
print("datasets", datasets.__version__)
print("accelerate", accelerate.__version__)
print("peft", peft.__version__)
print("trl", trl.__version__)
PY
```

环境检查状态：DONE

## 基本假设

- 第一个目标模型：`Qwen/Qwen2.5-0.5B-Instruct`，或其他 0.5B 到 1.5B 的因果语言模型。
- 第一个目标任务：简单数学题，或带格式要求的提示词，并使用规则型 reward。
- 第一个训练模式：单 GPU，小 batch size，不做分布式训练。
- 目标：理解并手写 GRPO 核心逻辑，而不是追求 benchmark 分数。
- 每日代码量目标：约 100 行。非代码任务约 2 到 3 小时。

## 每日安排

| Day | 实现什么功能 | 如何测试 | 完成情况 |
| --- | --- | --- | --- |
| Day 1 | 创建项目骨架：`src/grpo_handwritten/`、`tests/`、`scripts/`、`configs/`、`runs/` 和一份简短的 `README.md`。添加空 package 文件和最小 CLI 入口。 | 以 editable 模式安装包后运行 `python -m grpo_handwritten --help`。确认命令能打印帮助文本。 | TODO |
| Day 2 | 添加依赖检查脚本 `scripts/check_env.py`。打印 Python、PyTorch、CUDA、GPU 名称、transformers、datasets、accelerate、peft 和可用显存。 | 运行 `python scripts/check_env.py`。它必须无 import error 地结束，并清楚打印 CUDA 是否可用。 | TODO |
| Day 3 | 学习并记录 tensor 基础，写入 `notebooks_or_notes/day03_tensors.md`。编写小例子覆盖 tensor 创建、shape、dtype、device、索引、broadcasting、`mean`、`sum`、`gather` 和 `log_softmax`。 | 作为脚本或 notebook 运行这些例子。确认每个打印出的 tensor shape 都和笔记一致。 | TODO |
| Day 4 | 实现 `src/grpo_handwritten/config.py`，使用 dataclass 表示 model、data、generation、reward 和 training 配置。添加 YAML 加载。 | 添加 `tests/test_config.py`。运行 `pytest tests/test_config.py`，确认合法 YAML 能加载，非法 key 会清楚失败。 | TODO |
| Day 5 | 在 `modeling.py` 中添加 tokenizer 和模型加载工具。加载 tokenizer，如果缺少 pad token 则设置 pad token；加载 causal LM，并把模型移动到目标 device。 | 运行 `python scripts/smoke_load_model.py --model Qwen/Qwen2.5-0.5B-Instruct`。应打印模型 dtype、参数量和一个 tokenized prompt。 | TODO |
| Day 6 | 实现基础生成封装 `generate.py`：批量处理 prompts，tokenize，调用 `model.generate`，decode responses，并返回 prompt/response pairs。 | 运行 `python scripts/smoke_generate.py --prompt "What is 2+3?"`。确认模型返回非空文本。 | TODO |
| Day 7 | 为一个很小的本地 JSONL 文件添加 prompt dataset loader。定义 schema：`id`、`prompt`、可选 `answer`。添加 10 条手写数学 prompts。 | 运行 `pytest tests/test_data.py`。再运行 `python scripts/preview_data.py`，确认打印出 10 条 prompts。 | TODO |
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
| Day 28 | 创建第一个真实配置 `configs/grpo_qwen_0_5b_math.yaml`，用于在小型 JSONL 数据集上做短数学答案训练。 | 运行 `python scripts/train.py --config configs/grpo_qwen_0_5b_math.yaml --max-steps 5`。确认训练能启动且显存够用。 | TODO |
| Day 29 | 实现 evaluation script。为 held-out prompts 生成答案，并计算 exact-match reward、format reward、average length 和 sample outputs。 | 运行 `python scripts/eval.py --checkpoint <checkpoint> --data data/eval_math.jsonl`。确认打印 metrics summary 和 sample table。 | TODO |
| Day 30 | 添加训练前后评估 workflow。评估 base model，训练一个小 GRPO run，评估 checkpoint，并写出 comparison JSON。 | 运行 `python scripts/compare_eval.py`。确认报告 base reward、trained reward 和 delta。 | TODO |
| Day 31 | 提升 reward 鲁棒性：处理多种数字格式、空白字符、final-answer tags 和无效 completions。 | 运行 `pytest tests/test_rewards.py`，再对 20 条样例运行 `python scripts/eval.py`。确认 reward parser 不崩溃。 | TODO |
| Day 32 | 添加 sample inspection tool，把 prompts、group completions、rewards、advantages 和 selected tokens 保存到 JSONL，方便调试。 | 运行 `python scripts/inspect_rollouts.py --num-prompts 3 --group-size 4`。手动检查 JSONL，确认 group 可读。 | TODO |
| Day 33 | 添加 NaN 和不稳定性保护：检查 finite loss、finite rewards、max KL threshold warning、gradient norm warning，并在中止前自动保存 checkpoint。 | 创建一个 learning rate 过高的 config 并运行 3 steps。确认会出现 warning 或干净中止，而不是静默坏掉。 | TODO |
| Day 34 | 添加命令行易用性：把 `train`、`eval`、`generate`、`inspect-rollouts` 和 `check-env` 子命令整合到一个 package CLI 下。 | 分别运行每个 CLI 的 `--help` 和一个 smoke command。确认所有命令都能从 repo root 工作。 | TODO |
| Day 35 | 编写算法文档 `docs/grpo_algorithm.md`：data flow、tensors、shapes、loss formula、masks 和常见 bug。 | 对照代码名称检查文档。运行 `pytest`，确认示例或引用的函数仍然存在。 | TODO |
| Day 36 | 为这个代码库编写 PyTorch 新手笔记：`requires_grad`、`backward`、`no_grad`、model train/eval modes、devices 和 mixed precision。 | 阅读笔记并运行其中的 snippets。确认它们能在环境中执行。 | TODO |
| Day 37 | 运行一次更长的 50 到 100 step LoRA 实验。追踪 reward、KL、length 和 sample outputs。 | 运行 `python scripts/train.py --config configs/grpo_qwen_0_5b_math.yaml --max-steps 100`。确认 metrics 保持 finite，且 checkpoint 正常保存。 | TODO |
| Day 38 | 分析实验结果。判断 reward 提升是因为答案真的变好，还是因为 reward function 被利用。把笔记写入 `experiments/`。 | 在 held-out prompts 上运行 evaluation，并手动检查至少 30 条 samples。写一份简短实验报告。 | TODO |
| Day 39 | 修复实验中发现的前两个问题，例如输出过长、reward hacking、KL 不稳定或 prompt formatting 较弱。 | 重新运行一次 20-step 实验。确认目标 metric 朝预期方向变化。 | TODO |
| Day 40 | 最终清理：type hints、有用的 docstrings、README commands、config comments，以及删除废弃 scripts。 | 运行 `ruff check .`、`pytest`、`python scripts/check_env.py`，以及一次 1-step train smoke test。 | TODO |

## 里程碑

| 里程碑 | 目标天数 | 退出标准 | 完成情况 |
| --- | --- | --- | --- |
| M1：环境与推理 | Day 1 到 Day 6 | 环境已验证，模型能加载，且一个 prompt 能生成回复。 | TODO |
| M2：数据与 rewards | Day 7 到 Day 10 | Prompt dataset 能加载，rewards 能计算，group advantages 有测试覆盖。 | TODO |
| M3：GRPO 数学核心 | Day 11 到 Day 18 | Logprobs、masks、KL 和 GRPO loss 都有单元测试。 | TODO |
| M4：第一个训练循环 | Day 19 到 Day 24 | 一次完整 update 能运行，metrics 能记录，checkpoint 能保存和恢复。 | TODO |
| M5：实用训练能力 | Day 25 到 Day 33 | LoRA、显存控制、eval、rollout inspection 和稳定性保护都能工作。 | TODO |
| M6：可用项目 | Day 34 到 Day 40 | CLI、docs、实验报告、lint、tests 和 smoke training 都完成。 | TODO |

## 推荐的第一阶段编码顺序

1. 不要从 GRPO loss 开始。先做模型加载和生成，因为这能验证环境，并消除最大的未知数。
2. 一开始保持 reward 函数为规则型。Learned reward model 会在基础循环稳定前引入另一个训练问题。
3. 第一个模型保持小。对学习而言，0.5B 模型比一个经常显存溢出的更大模型更合适。
4. 把每个 tensor shape 都写下来。大多数 GRPO bug 都是 mask、padding 和 logprob 对齐问题。
5. 把 TRL 当成参考，而不是依赖路径。卡住时可以对比行为，但实现路径保持手写。
