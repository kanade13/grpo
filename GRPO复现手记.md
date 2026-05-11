# 第一阶段：下载Qwen3-0.6B， 并在GSM8K test 划分上测试准确率

注意测试条件：
model: Qwen/Qwen3-0.6B
dataset: openai/gsm8k, main, test split
mode: enable_thinking=True / False
prompt: 是否要求 \boxed{} / Final answer
decoding:
  temperature
  top_p
  top_k
  do_sample
  max_new_tokens
eval:
  answer extractor 规则
  exact match 规则
  是否忽略逗号、单位、小数格式
hardware:
  GPU
  dtype
  vLLM / transformers