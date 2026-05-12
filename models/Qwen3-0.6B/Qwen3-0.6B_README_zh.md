---
library_name: transformers
license: apache-2.0
license_link: https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/LICENSE
pipeline_tag: text-generation
base_model:
- Qwen/Qwen3-0.6B-Base
---

# Qwen3-0.6B
<a href="https://chat.qwen.ai/" target="_blank" style="margin: 2px;">
    <img alt="Chat" src="https://img.shields.io/badge/%F0%9F%92%9C%EF%B8%8F%20Qwen%20Chat%20-536af5" style="display: inline-block; vertical-align: middle;"/>
</a>

## Qwen3 亮点

Qwen3 是 Qwen 系列最新一代大语言模型，提供了一整套稠密模型与混合专家模型（Mixture-of-Experts, MoE）。基于大规模训练，Qwen3 在推理、指令遵循、智能体能力以及多语言支持方面实现了突破性进展，主要特性如下：

- **在单个模型中独特地支持思考模式与非思考模式的无缝切换**：思考模式适用于复杂逻辑推理、数学和编程；非思考模式适用于高效的通用对话。这样可以确保模型在不同场景下都能获得较优表现。
- **推理能力显著增强**：在数学、代码生成和常识逻辑推理方面，超过了此前的 QwQ（思考模式）和 Qwen2.5 Instruct 模型（非思考模式）。
- **更优秀的人类偏好对齐能力**：在创意写作、角色扮演、多轮对话和指令遵循方面表现出色，能够提供更自然、更有参与感、更具沉浸感的对话体验。
- **具备出色的智能体能力**：无论在思考模式还是非思考模式下，都可以准确集成外部工具，并在复杂智能体任务中达到开源模型中的领先水平。
- **支持 100 多种语言和方言**，并具备强大的**多语言指令遵循**与**翻译**能力。

## 模型概览

**Qwen3-0.6B** 具有以下特性：

- 类型：因果语言模型（Causal Language Models）
- 训练阶段：预训练与后训练
- 参数数量：0.6B
- 参数数量（不含 Embedding）：0.44B
- 层数：28
- 注意力头数量（GQA）：Q 为 16，KV 为 8
- 上下文长度：32,768

更多细节，包括基准评测、硬件需求和推理性能，请参考我们的 [博客](https://qwenlm.github.io/blog/qwen3/)、[GitHub](https://github.com/QwenLM/Qwen3) 和 [文档](https://qwen.readthedocs.io/en/latest/)。

> [!TIP]
> 如果你遇到明显的无限重复问题，请参考 [最佳实践](#最佳实践) 一节中的推荐采样参数，并将 `presence_penalty` 设置为 1.5。

## 快速开始

Qwen3 的代码已经集成到最新版 Hugging Face `transformers` 中，我们建议你使用最新版 `transformers`。

如果使用 `transformers<4.51.0`，你会遇到以下错误：

```text
KeyError: 'qwen3'
```

下面的代码片段展示了如何使用该模型根据给定输入生成内容。

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3-0.6B"

# 加载 tokenizer 和模型
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

# 准备模型输入
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
```

部署时，你可以使用 `sglang>=0.4.6.post1` 或 `vllm>=0.8.5` 来创建兼容 OpenAI API 的端点：

- SGLang：

    ```shell
    python -m sglang.launch_server --model-path Qwen/Qwen3-0.6B --reasoning-parser qwen3
    ```

- vLLM：

    ```shell
    vllm serve Qwen/Qwen3-0.6B --enable-reasoning --reasoning-parser deepseek_r1
    ```

对于本地使用，Ollama、LMStudio、MLX-LM、llama.cpp 和 KTransformers 等应用也已经支持 Qwen3。

## 在思考模式和非思考模式之间切换

> [!TIP]
> `enable_thinking` 开关同样可以用于由 SGLang 和 vLLM 创建的 API。
> SGLang 和 vLLM 用户请参考我们的文档：[SGLang 文档](https://qwen.readthedocs.io/en/latest/deployment/sglang.html#thinking-non-thinking-modes) 和 [vLLM 文档](https://qwen.readthedocs.io/en/latest/deployment/vllm.html#thinking-non-thinking-modes)。

### `enable_thinking=True`

默认情况下，Qwen3 会启用思考能力，类似于 QwQ-32B。这意味着模型会使用其推理能力来提升生成回答的质量。例如，当你在 `tokenizer.apply_chat_template` 中显式设置 `enable_thinking=True`，或保留默认值时，模型就会进入思考模式。

```python
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True  # enable_thinking 的默认值就是 True
)
```

在该模式下，模型会生成包裹在 `<think>...</think>` 块中的思考内容，然后再给出最终回答。

> [!NOTE]
> 对于思考模式，请使用 `Temperature=0.6`、`TopP=0.95`、`TopK=20` 和 `MinP=0`（即 `generation_config.json` 中的默认设置）。**不要使用贪婪解码**，因为它可能导致性能下降和无限重复。更详细的指导请参考 [最佳实践](#最佳实践) 一节。

### `enable_thinking=False`

我们提供了一个硬开关，用于严格禁用模型的思考行为，使其功能表现与此前的 Qwen2.5-Instruct 模型对齐。该模式特别适合需要关闭思考以提升效率的场景。

```python
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False  # 设置 enable_thinking=False 会禁用思考模式
)
```

在该模式下，模型不会生成任何思考内容，也不会包含 `<think>...</think>` 块。

> [!NOTE]
> 对于非思考模式，我们建议使用 `Temperature=0.7`、`TopP=0.8`、`TopK=20` 和 `MinP=0`。更详细的指导请参考 [最佳实践](#最佳实践) 一节。

### 高级用法：通过用户输入在思考模式和非思考模式之间切换

我们提供了一种软开关机制：当 `enable_thinking=True` 时，用户可以动态控制模型行为。具体来说，你可以在用户提示词或系统消息中添加 `/think` 和 `/no_think`，从而在不同轮次之间切换模型的思考模式。模型会在多轮对话中遵循最近的一条相关指令。

下面是一个多轮对话示例：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

class QwenChatbot:
    def __init__(self, model_name="Qwen/Qwen3-0.6B"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.history = []

    def generate_response(self, user_input):
        messages = self.history + [{"role": "user", "content": user_input}]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(text, return_tensors="pt")
        response_ids = self.model.generate(**inputs, max_new_tokens=32768)[0][len(inputs.input_ids[0]):].tolist()
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)

        # 更新历史记录
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": response})

        return response

# 使用示例
if __name__ == "__main__":
    chatbot = QwenChatbot()

    # 第一个输入（没有 /think 或 /no_think 标签，默认启用思考模式）
    user_input_1 = "How many r's in strawberries?"
    print(f"User: {user_input_1}")
    response_1 = chatbot.generate_response(user_input_1)
    print(f"Bot: {response_1}")
    print("----------------------")

    # 第二个输入，带有 /no_think
    user_input_2 = "Then, how many r's in blueberries? /no_think"
    print(f"User: {user_input_2}")
    response_2 = chatbot.generate_response(user_input_2)
    print(f"Bot: {response_2}") 
    print("----------------------")

    # 第三个输入，带有 /think
    user_input_3 = "Really? /think"
    print(f"User: {user_input_3}")
    response_3 = chatbot.generate_response(user_input_3)
    print(f"Bot: {response_3}")
```

> [!NOTE]
> 为了兼容 API，当 `enable_thinking=True` 时，无论用户使用 `/think` 还是 `/no_think`，模型都会始终输出一个包裹在 `<think>...</think>` 中的块。不过，如果思考被禁用，这个块内部的内容可能为空。
> 当 `enable_thinking=False` 时，软开关无效。无论用户输入任何 `/think` 或 `/no_think` 标签，模型都不会生成思考内容，也不会包含 `<think>...</think>` 块。

## 智能体用法

Qwen3 在工具调用能力方面表现出色。我们推荐使用 [Qwen-Agent](https://github.com/QwenLM/Qwen-Agent) 来充分发挥 Qwen3 的智能体能力。Qwen-Agent 内部封装了工具调用模板和工具调用解析器，可以大幅降低编码复杂度。

若要定义可用工具，你可以使用 MCP 配置文件，使用 Qwen-Agent 的内置工具，或者自行集成其他工具。

```python
from qwen_agent.agents import Assistant

# 定义 LLM
llm_cfg = {
    'model': 'Qwen3-0.6B',

    # 使用阿里云百炼 Model Studio 提供的端点：
    # 'model_type': 'qwen_dashscope',
    # 'api_key': os.getenv('DASHSCOPE_API_KEY'),

    # 使用兼容 OpenAI API 的自定义端点：
    'model_server': 'http://localhost:8000/v1',  # api_base
    'api_key': 'EMPTY',

    # 其他参数：
    # 'generate_cfg': {
    #         # 添加：当响应内容为 `<think>this is the thought</think>this is the answer` 时使用；
    #         # 不添加：当响应已经被分离为 reasoning_content 和 content 时使用。
    #         'thought_in_content': True,
    #     },
}

# 定义工具
tools = [
    {'mcpServers': {  # 你可以指定 MCP 配置文件
            'time': {
                'command': 'uvx',
                'args': ['mcp-server-time', '--local-timezone=Asia/Shanghai']
            },
            "fetch": {
                "command": "uvx",
                "args": ["mcp-server-fetch"]
            }
        }
    },
  'code_interpreter',  # 内置工具
]

# 定义 Agent
bot = Assistant(llm=llm_cfg, function_list=tools)

# 流式生成
messages = [{'role': 'user', 'content': 'https://qwenlm.github.io/blog/ Introduce the latest developments of Qwen'}]
for responses in bot.run(messages=messages):
    pass
print(responses)
```

## 最佳实践

为了获得最佳性能，我们推荐以下设置：

1. **采样参数**：
   - 对于思考模式（`enable_thinking=True`），使用 `Temperature=0.6`、`TopP=0.95`、`TopK=20` 和 `MinP=0`。**不要使用贪婪解码**，因为它可能导致性能下降和无限重复。
   - 对于非思考模式（`enable_thinking=False`），我们建议使用 `Temperature=0.7`、`TopP=0.8`、`TopK=20` 和 `MinP=0`。
   - 对于支持的框架，你可以将 `presence_penalty` 参数调整到 0 到 2 之间，以减少无限重复。不过，使用更高的值有时可能导致语言混杂，并使模型性能略有下降。

2. **充足的输出长度**：我们建议对大多数查询使用 32,768 token 的输出长度。对于数学和编程竞赛中出现的高复杂度问题的基准测试，我们建议将最大输出长度设置为 38,912 token。这能为模型提供足够的空间生成详细、全面的回答，从而提升整体性能。

3. **标准化输出格式**：我们建议在基准测试时使用提示词来规范模型输出。
   - **数学题**：在提示词中加入：`Please reason step by step, and put your final answer within \boxed{}.`
   - **选择题**：在提示词中加入以下 JSON 结构来规范响应：`Please show your choice in the answer field with only the choice letter, e.g., "answer": "C".`

4. **历史记录中不保留思考内容**：在多轮对话中，历史模型输出应只包含最终输出部分，不需要包含思考内容。所提供的 Jinja2 聊天模板已经实现了这一点。不过，对于没有直接使用 Jinja2 聊天模板的框架，开发者需要自行确保遵循这一最佳实践。

### 引用

如果你觉得我们的工作有帮助，欢迎引用我们。

```bibtex
@misc{qwen3technicalreport,
      title={Qwen3 Technical Report}, 
      author={Qwen Team},
      year={2025},
      eprint={2505.09388},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.09388}, 
}
```
