from transformers import AutoModelForCausalLM, AutoTokenizer

import torch, torchvision
print("torch:", torch.__version__, torch.version.cuda, torch.__file__)
print("torchvision:", torchvision.__version__, torchvision.__file__)
print("cuda available:", torch.cuda.is_available())

model_path="./models/Qwen3-0.6B"

tokenizer=AutoTokenizer.from_pretrained(model_path)

print(type(tokenizer))
print("eos_token:", tokenizer.eos_token)
print("eos_token_id:", tokenizer.eos_token_id)
print("pad_token:", tokenizer.pad_token)
print("pad_token_id:", tokenizer.pad_token_id)
print("has chat_template:", tokenizer.chat_template is not None)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto"
)
print("model dtype:",model.dtype)
print("model device on: ",model.device)

prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True # 在思考模式和非思考模式之间切换。默认值为 True。
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# 执行文本补全
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=32768
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 
print(output_ids)
# 解析思考内容
try:
    # rindex 用于寻找 151668（</think>）
    index = len(output_ids) - output_ids[::-1].index(151668)
except ValueError:
    index = 0

thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

print("thinking content:", thinking_content)
print("content:", content)

