# OpenAI 到 Gemini 适配器 (自定义提示词、动态变量、正则处理与 Gemini 特性增强)

本项目是一个使用 FastAPI 构建的 API 适配器服务。它接收标准的 OpenAI Chat Completion API 格式请求，将其转换为 Google Gemini API 格式，调用 Gemini API，然后将 Gemini 的响应转换回 OpenAI 格式返回给客户端。
本项目不仅实现了 OpenAI 与 Gemini API 之间的核心格式转换，还集成了强大的自定义提示词注入（支持智能模板选择）、动态变量处理（如 `{{roll}}` 和 `{{random}}`）、灵活的响应后处理（包括正则表达式替换和 JSON 载荷注入），并特别支持 Google Gemini 的专属高级功能，如 `show_thinking`（展示模型内部思考过程）和 `thinking_budget`（控制模型思考资源）。

## ✨ 核心特性

### 🔄 格式转换 (OpenAI <=> Gemini)
-   **请求转换**: 将传入的 OpenAI `/v1/chat/completions` 请求（包括消息、模型名称等）转换为 Gemini API 的 `generateContent` 请求格式。
-   **响应转换**: 将 Gemini API 的响应（包括内容、停止原因、token 使用量等）转换回 OpenAI Chat Completion API 的响应格式。
-   **参数映射**: 智能处理 `temperature`, `max_tokens` (Gemini 中为 `maxOutputTokens`), `top_p`, `top_k` 等生成参数，优先使用服务端配置。

### 🎯 自定义提示词注入
-   **智能模板选择**:
    -   根据用户请求中是否包含实际输入内容，自动从配置文件中指定的不同模板路径加载提示词。
    -   通过 `config/settings.yaml` 中的 `proxy.prompt_template_path_with_input` 和 `proxy.prompt_template_path_without_input` 进行配置。
-   **模板占位符**:
    -   `{{user_input}}`: 将被替换为用户最新一条消息的 `content`。
    -   `type: api_input_placeholder`: 标记历史消息（原始请求 `messages` 数组中除了最后一条用户消息之外的内容）的插入位置。
-   **消息合并**: 自动合并相邻的同角色消息（如多个 `user` 或 `system` 消息），并确保发送给 Gemini 的消息序列是 `user` 和 `model` 角色的严格交替。

### 🎲 动态变量系统
在提示词模板的 `content` 字段中，可以使用以下动态变量：
-   `{{roll XdY}}`: 模拟投掷 X 个 Y 面骰子，并替换为投掷结果的总点数。例如：`{{roll 2d6}}`。
-   `{{random::选项1::选项2::选项3}}`: 从提供的多个选项中随机选择一个并替换。选项之间用 `::` 分隔。例如：`{{random::晴天::阴天::雨天}}`。

### 🔧 响应后处理
在从 Gemini API 获取响应并转换为 OpenAI 格式后，但在最终返回给客户端之前，可以对助手的 `content` 应用一系列后处理规则：
-   **正则表达式替换**:
    -   在模板文件中定义 `type: 正则` 规则块。
    -   包含 `查找` (正则表达式字符串) 和 `替换` (替换字符串，支持捕获组如 `\1`, `\g<name>`)。
    -   `action: "replace"` (可选，默认为替换)。
-   **JSON 载荷注入**:
    -   同样使用 `type: 正则` 规则块，但 `action` 设置为 `"json_payload"`。
    -   `替换` 字段应为一个合法的 JSON 字符串。匹配到的内容（通常是整个响应或特定部分）会被这个 JSON 对象包裹或替换，从而允许向客户端返回结构化的数据。
-   **规则级联**: 所有定义的后处理规则会按照它们在 YAML 模板文件中出现的顺序依次执行。

### ⚙️ 参数配置与 Gemini 专属功能
-   **客户端参数控制**: 服务端强制使用配置文件 (`config/settings.yaml`) 中的生成参数。客户端请求中仅以下参数有效：
    -   `model`: 指定要使用的 Gemini 模型名称 (例如 `gemini-1.5-flash-latest`, `gemini-1.5-pro-latest`)。
    -   `messages`: OpenAI 格式的消息列表。
    -   `stream`: 布尔值，指示是否请求（伪）流式响应。
    -   客户端传递的其他生成参数（如 `temperature`, `max_tokens` 等）将被忽略，并使用配置文件中的默认值。
-   **Gemini 专属功能支持**:
    -   **`show_thinking`**: (布尔值，默认为 `false`) 当设置为 `true` 时，如果模型支持，Gemini API 可能会在响应中包含其内部的思考过程或工具调用计划。这对于理解模型决策和调试非常有用。可在 `config/settings.yaml` 的 `proxy.gemini_generation.show_thinking` 中配置。
    -   **`thinking_budget`**: (整数，范围 0-24576，默认为 `1024`) 此参数指导模型在生成最终答案前可以使用的“思考”token 数量。较高的预算可能允许模型进行更深入的推理或规划。可在 `config/settings.yaml` 的 `proxy.gemini_generation.thinking_budget` 中配置。
-   **其他 Gemini 生成参数**: `temperature`, `max_output_tokens`, `top_p`, `top_k`, `candidate_count`, `stop_sequences` 等均可通过 `config/settings.yaml` 中 `proxy.gemini_generation` 部分进行全局配置。

### 🔑 API 密钥处理
-   Google Gemini API 密钥必须通过客户端请求的 `Authorization` HTTP 头传递：
    `Authorization: Bearer YOUR_GEMINI_API_KEY`

### 🌊 (伪)流式处理
-   后端对 Gemini API 的调用始终是非流式的。
-   如果客户端请求 `stream: true`：
    1.  服务立即与客户端建立连接。
    2.  定期发送心跳注释行 (例如 `: ping`) 以保持连接活跃。
    3.  获取到完整的 Gemini 响应并转换后，将其内容（通常是 `choices[0].delta.content`）模拟 Server-Sent Events (SSE) 分块发送。
    4.  最后发送 `data: [DONE]\n\n` 事件表示流结束。

### 📝 日志记录与环境变量
-   集成详细的日志记录功能，可通过 `config/settings.yaml` 中的 `log_level` (如 `INFO`, `DEBUG`) 控制日志级别。
-   **彩色日志输出**: 在交互式终端中默认启用彩色日志，便于开发和调试。
-   **禁用颜色**: 设置环境变量 `NO_COLOR=1` (或任何非空值) 可以强制禁用日志颜色输出，适用于日志文件记录或不支持 ANSI 颜色的环境。

## 📋 使用场景
-   **平滑迁移**: 帮助已使用 OpenAI API 的应用快速切换或兼容 Google Gemini API。
-   **统一接口**: 为不同的 Gemini 模型提供一个遵循 OpenAI 规范的统一访问点。
-   **提示工程增强**: 通过集中的模板管理、智能选择和动态变量，简化和强化对 Gemini 的提示构建。
-   **响应定制化**: 利用强大的正则和 JSON 注入后处理能力，精确控制和丰富返回给客户端的内容。
-   **Gemini 特性探索**: 方便地实验和利用 Gemini 的 `show_thinking` 和 `thinking_budget` 等高级功能。
-   **调试与分析**: 通过 `show_thinking`深入了解模型行为，或通过日志分析请求与响应的完整流程。

## 🚀 快速开始

### 1. 安装依赖
克隆本仓库到本地后，在项目根目录下执行：
```bash
pip install -r requirements.txt
```
建议在 Python 虚拟环境中使用。

### 2. 配置服务
-   复制配置文件模板：
    ```bash
    cp config/settings.yaml.example config/settings.yaml
    ```
-   编辑 `config/settings.yaml`：
    -   `log_level`: 设置合适的日志级别。
    -   `proxy.prompt_template_path_with_input`: 用户有输入时使用的模板文件路径 (例如 `templates/with_input.yaml`)。
    -   `proxy.prompt_template_path_without_input`: 用户无输入时 (例如仅有系统提示) 使用的模板文件路径 (例如 `templates/without_input.yaml`)。
    -   `proxy.gemini_generation`: 配置各项 Gemini 生成参数的默认值，包括 `temperature`, `max_output_tokens`, `top_p`, `top_k`, `candidate_count`, `show_thinking`, `thinking_budget`, `stop_sequences`。
    -   `proxy.fake_streaming.heartbeat_interval`: （伪）流式响应的心跳间隔（秒）。
    -   `proxy.request_timeout`: 请求 Gemini API 的超时时间（秒）。

### 3. 准备提示词模板
根据在 `config/settings.yaml` 中配置的路径，创建或修改相应的 YAML 模板文件。例如：
-   `templates/with_input.yaml`
-   `templates/without_input.yaml`
模板的具体格式见下文“配置说明”部分。

### 4. 启动服务
-   **开发模式 (使用 Uvicorn, 支持热重载)**:
    ```bash
    uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
    ```
-   **直接运行 (不推荐用于生产)**:
    ```bash
    python -m src.main
    ```

## 📖 API 使用方法

### 请求 URL
将您通常发送给 OpenAI API `/v1/chat/completions` 端点的请求，改为发送到本适配器服务的相同路径：
`POST http://<your-adapter-host>:<port>/v1/chat/completions`

例如，如果服务运行在 `localhost:8000`：
`http://localhost:8000/v1/chat/completions`

### 请求头
必须在请求头中包含您的 **Google Gemini API Key**:
`Authorization: Bearer YOUR_GEMINI_API_KEY`

### 请求体
使用标准的 OpenAI Chat Completion API 请求格式，但**仅需提供以下三个参数**：
```json
{
  "model": "gemini-1.5-flash-latest", // 必须是有效的 Gemini 模型名称
  "messages": [
    {"role": "system", "content": "你是一个乐于助人的AI助手。"},
    {"role": "user", "content": "你好！请用中文介绍一下你自己。"}
  ],
  "stream": false // 设置为 true 以启用伪流式响应
}
```
**重要提示**:
-   客户端在请求体中传递的任何其他参数（如 `temperature`, `max_tokens`, `top_p`, `n`, `stop` 等）都将被**忽略**。
-   所有实际用于调用 Gemini API 的生成参数均来自服务端的 `config/settings.yaml` 文件中 `proxy.gemini_generation` 部分的配置。
-   如果客户端发送了这些被忽略的参数，系统会在日志中记录相关信息。

## 📁 项目结构

```
.
├── .gitignore
├── README.md
├── README_temp.md             # 临时参考文件，可删除
├── requirements.txt
├── config/
│   ├── settings.yaml          # 应用主配置文件 (由 settings.yaml.example 复制而来)
│   └── settings.yaml.example  # 配置文件模板
├── src/
│   ├── __init__.py
│   ├── config.py              # 配置加载与管理
│   ├── conversion_utils.py    # 格式转换与后处理工具
│   ├── gemini_client.py       # Gemini API 客户端逻辑 (假设)
│   ├── main.py                # FastAPI 应用主入口
│   ├── openai_proxy.py        # 核心代理与转换逻辑
│   ├── streaming_utils.py     # (伪)流式处理工具
│   └── template_handler.py    # 提示词模板加载与处理
└── templates/
    ├── with_input.yaml        # 用户有输入时的模板示例
    └── without_input.yaml     # 用户无输入时的模板示例
    # (可能还有其他用户自定义的模板文件)
```
*注意: `gemini_client.py` 是基于功能推测的模块名，实际文件名可能不同。项目结构基于通用模板，请根据实际情况调整。*

## ⚙️ 配置说明

### `config/settings.yaml` 详解
主要的配置项包括：
-   `app_name`: 应用名称 (例如 "GeminiAdapter")。
-   `log_level`: 日志级别 (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`)。
-   `debug_mode`: 是否开启 FastAPI 的调试模式 (布尔值)。
-   `proxy`:
    -   `prompt_template_path_with_input`: 字符串，指向用户有输入时使用的模板文件路径。
    -   `prompt_template_path_without_input`: 字符串，指向用户无输入时使用的模板文件路径。
    -   `request_timeout`: 整数，请求 Gemini API 的超时时间（秒）。
    -   `fake_streaming`:
        -   `heartbeat_interval`: 整数，伪流式心跳间隔（秒）。
    -   `gemini_generation`:
        -   `temperature`: 浮点数 (0.0 - 2.0)。
        -   `max_output_tokens`: 整数 (例如 8192)。
        -   `top_p`: 浮点数 (0.0 - 1.0)。
        -   `top_k`: 整数 (例如 40)。
        -   `candidate_count`: 整数 (1-8)。
        -   `show_thinking`: 布尔值。
        -   `thinking_budget`: 整数 (0-24576)。

### 提示词模板文件格式 (`.yaml`)
提示词模板是一个 YAML 文件，其顶层是一个列表，列表中的每一项代表一个消息对象或一个后处理规则。

-   **消息块**:
    -   标准的 OpenAI 消息结构，包含 `role` (例如 `system`, `user`, `assistant`) 和 `content` (字符串)。
    -   `content` 字段内可使用动态变量 `{{user_input}}`, `{{roll XdY}}`, `{{random::opt1::opt2}}`。
-   **历史消息占位符**:
    -   一个字典对象：`{type: api_input_placeholder}`。它标记了原始请求中历史消息（除最新用户消息外）的插入位置。
-   **后处理规则块 (`type: 正则`)**:
    -   一个字典对象，必须包含 `type: 正则`。
    -   `查找`: 字符串，表示要查找的正则表达式模式。
    -   `替换`: 字符串，表示替换匹配内容的字符串。支持捕获组如 `\1`, `\g<name>`。
    -   `action` (可选):
        -   `"replace"` (默认): 执行查找和替换。
        -   `"json_payload"`: 将 `替换` 字段中的 JSON 字符串注入到响应中。通常用于将模型的纯文本输出包装成结构化数据。

**示例 (`templates/with_input.yaml`):**
```yaml
- role: system
  content: "你是一个AI助手。今天是 {{random::星期一::星期二::星期三}}，幸运数字是 {{roll 1d100}}。"
- type: api_input_placeholder
- role: user
  content: "{{user_input}}"
# 示例后处理规则：将所有 "Gemini" 替换为 "本AI模型"
- type: 正则
  查找: "\\bGemini\\b"
  替换: "本AI模型"
# 示例JSON载荷注入：如果模型输出了被```python ... ```包裹的代码块，则提取代码并构造成JSON
- type: 正则
  查找: ".*?```python\\n(.*?)\\n```.*" # 非贪婪匹配，提取代码
  替换: '{"tool_code_output": {"language": "python", "code": "\\1"}}'
  action: "json_payload"
```

**模板热重载**: 本项目支持模板文件的热重载。当您修改并保存 YAML 模板文件后，服务会在下次处理请求时自动加载最新的模板内容，无需重启服务。

## 🛠️ 开发和部署

### 开发模式
使用 Uvicorn 并开启 `--reload` 标志可以在代码或模板文件更改时自动重启服务：
```bash
python -m src.main
```

或使用 Uvicorn：
```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000
```

## 📝 注意事项
-   **API 密钥安全**: 务必妥善保管您的 Google Gemini API 密钥，不要将其硬编码到代码或版本控制中。
-   **超时配置**: 根据您的网络环境和 Gemini API 的典型响应时间，合理配置 `proxy.request_timeout`。
-   **日志监控**: 定期检查应用日志，以便及时发现和处理潜在问题。
-   **错误处理**: 服务会尽力捕获与 Gemini API 通信或内部处理时发生的错误，并向客户端返回适当的 HTTP 状态码和错误信息。
-   **模板复杂性**: 过度复杂的提示词模板或大量的后处理规则可能会影响性能，请按需使用。

## 🤝 贡献
欢迎通过提交 Issues 或 Pull Requests 来为本项目做出贡献。请确保您的代码遵循项目的编码风格，并提供充分的测试。

## 📄 许可证
本项目采用 MIT 许可证。详情请参阅 `LICENSE` 文件 (如果项目中有)。
(如果项目中没有 LICENSE 文件，可以考虑添加一个，或者直接声明 "本项目采用 MIT 许可证。")