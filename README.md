# OpenAI 到 Gemini 适配器 (自定义提示词、动态变量与正则处理)

本项目是一个使用 FastAPI 构建的 API 适配器服务。它接收 OpenAI Chat Completion API 格式的请求，将其转换为 Google Gemini API 格式，调用 Gemini API，然后将 Gemini 的响应转换回 OpenAI 格式返回给客户端。
本项目保留了原有的自定义提示词注入功能，并增强了动态变量处理（包括 `{{roll}}` 和 `{{random}}`）以及通过YAML模板对助手响应进行正则表达式后处理的能力。

## 核心功能

-   **格式转换**:
    -   将传入的 OpenAI `/v1/chat/completions` 请求（包括消息、模型名称、温度、最大token数、top_p、停止序列等参数）转换为 Gemini API 的请求格式。
    -   将 Gemini API 的响应（包括内容、停止原因、token使用量等）转换回 OpenAI Chat Completion API 的响应格式。
-   **自定义提示词注入**: 从 YAML 文件加载提示词模板，支持 `{{user_input}}` (最新用户消息) 和 `api_input_placeholder` (历史消息) 占位符。
-   **动态变量处理**:
    -   `{{roll XdY}}`: 在提示词模板内容中，此变量会被替换为 X 个 Y 面骰子的投掷总和。例如，`{{roll 2d6}}`。
    -   `{{random::opt1::opt2...}}`: 在提示词模板内容中，此变量会从提供的选项中随机选择一个并替换。选项之间用 `::` 分隔。例如，`{{random::苹果::香蕉::橙子}}`。
-   **正则表达式后处理**:
    -   支持在 YAML 提示词模板文件中定义 `type: 正则` 的规则块。
    -   每个规则块包含 `查找` (正则表达式) 和 `替换` (替换字符串，支持捕获组如 `\1`, `\g<name>`)。
    -   这些规则会按顺序应用于从 Gemini API 获取并已转换为 OpenAI 格式的助手消息的 `content` 字段，在最终返回给客户端之前。
-   **参数配置**: 
    -   **强制使用配置文件默认值**: 系统将忽略客户端传递的所有生成参数，强制使用配置文件中的默认值，包括：
        -   `temperature` (控制生成文本的随机性，范围 0-2)
        -   `max_tokens` (最大输出词元数，Gemini 2.5 Pro 上限 65535)
        -   `top_p` (控制采样时要考虑的词元比例，范围 0-1)
        -   `top_k` (控制采样时要考虑的前 K 个词元)
        -   `candidate_count` (生成候选响应数量，范围 1-8)
        -   `show_thinking` (Gemini 2.5 Flash 特色功能，返回模型内部思考过程)
        -   `thinking_budget` (思考预算，指导模型可使用的思考 token 数量，范围 0-24576)
        -   `stop` (停止序列)
    -   **仅接受的客户端参数**: 只接受 `model`(模型名称)、`messages`(消息内容)、`stream`(是否流式) 三个参数，其他参数都将被忽略并记录日志。
-   **伪流式处理**:
    -   后端对 Gemini API 的调用始终是非流式的。
    -   如果客户端请求流式响应 (`stream: true`)，服务将提供"伪流式"：立即建立连接，定期发送心跳，获取到完整 Gemini 响应并转换后，将其内容分块模拟 SSE 事件发送，最后以 `data: [DONE]\n\n` 结束。
-   **API 密钥处理**: 从客户端请求的 `Authorization: Bearer <YOUR_GEMINI_API_KEY>` 头中获取 Google Gemini API 密钥。
-   **配置文件驱动**: 通过 `config/settings.yaml` 进行日志级别、模板路径等配置。
-   **并发支持**: 基于 FastAPI 和 `asyncio`。
-   **日志记录**: 集成日志功能。

## 环境变量配置

除了配置文件外，还支持以下环境变量：

- `NO_COLOR`: 设置为任意值（如 `1` 或 `true`）可强制禁用日志颜色输出，适用于后台部署环境
- `TERM`: 终端类型，系统会自动检测是否支持颜色输出

### 日志输出模式

系统会自动检测运行环境：
- **交互式终端**: 使用彩色日志输出，便于开发调试
- **后台部署环境**: 自动切换到纯文本模式，避免在日志文件中出现 ANSI 颜色代码（如 `[36m`、`[31m` 等）

如需强制禁用颜色，可设置环境变量：
```bash
export NO_COLOR=1
python -m src.main
```

## 目录结构

```
.
├── src/                          # 源代码目录
│   ├── __init__.py
│   ├── main.py                   # FastAPI 应用主文件
│   ├── config.py                 # 配置加载和管理模块
│   ├── openai_proxy.py           # 核心适配器逻辑模块
├── templates/                    # 提示词模板目录
│   └── default_prompt.yaml       # 默认提示词模板 (路径可在配置中修改)
├── config/                       # 配置文件目录
│   └── settings.yaml.example     # 应用配置文件示例 (请复制为 settings.yaml 并修改)
├── .gitignore
├── README.md
└── requirements.txt
```

## 安装与运行

1.  **克隆仓库**:
    ```bash
    git clone <your-repo-url>
    cd <your-repo-name>
    ```

2.  **创建并激活虚拟环境** (推荐):
    ```bash
    python -m venv venv
    # Linux/macOS:
    source venv/bin/activate
    # Windows (cmd.exe):
    # venv\Scripts\activate.bat
    # Windows (PowerShell):
    # venv\Scripts\Activate.ps1
    ```

3.  **安装依赖**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **配置**:
    -   复制 `config/settings.yaml.example` 为 `config/settings.yaml`。
    -   编辑 `config/settings.yaml` 以配置应用参数，例如：
        -   `log_level`: 日志级别 (如 INFO, DEBUG)。
        -   `proxy.prompt_template_path`: 提示词模板文件的路径。
        -   `proxy.fake_streaming.heartbeat_interval`: （伪）流式心跳间隔（秒）。
        -   `proxy.openai_request_timeout`: 请求 Gemini API 的超时时间（秒）。
        -   `proxy.gemini_generation`: Gemini 2.5 Pro 生成参数的默认配置：
            -   `temperature`: 控制生成文本的随机性 (默认 1.0)
            -   `max_output_tokens`: 最大输出词元数 (默认 65535)
            -   `top_p`: 控制采样时要考虑的词元比例 (默认 1.0)
            -   `top_k`: 控制采样时要考虑的前 K 个词元 (默认 64)
            -   `candidate_count`: 生成候选响应数量 (默认 1)
            -   `show_thinking`: 是否返回模型内部思考过程 (默认 false)
            -   `thinking_budget`: 思考预算，指导模型可使用的思考 token 数量 (默认 1024，范围 0-24576)
    -   编辑 `templates/default_prompt.yaml` (或您在配置中指定的其他模板文件) 来定义您的提示词结构、动态变量和正则规则。

5.  **运行服务** (开发模式):
    ```bash
    python -m uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
    ```
    对于生产环境，建议使用 Gunicorn + Uvicorn workers。

## 使用方法

将您通常发送给 OpenAI API `/v1/chat/completions` 端点的请求，改为发送到本适配器服务的相同路径。

**代理请求 URL**: `http://<your-adapter-host>:<port>/v1/chat/completions`
例如，如果代理服务运行在 `http://localhost:8000`：
`http://localhost:8000/v1/chat/completions`

**请求头**:
确保在请求头中包含您的 **Google Gemini API Key**:
`Authorization: Bearer YOUR_GEMINI_API_KEY`

**请求体**:
使用标准的 OpenAI Chat Completion API 请求格式，但只需要提供以下三个参数：
```json
{
  "model": "gemini-2.5-flash-preview-05-20", // 指定 Gemini 模型名称
  "messages": [
    {"role": "system", "content": "你是一个乐于助人的助手。"},
    {"role": "user", "content": "你好！请介绍一下你自己。"}
  ],
  "stream": false              // 或 true，以启用伪流式响应
}
```

**注意**: 
- 客户端传递的其他参数（如 `temperature`, `max_tokens`, `top_p`, `top_k`, `candidate_count`, `show_thinking`, `thinking_budget`, `stop` 等）将被忽略。
- 所有生成参数都使用配置文件 `config/settings.yaml` 中 `proxy.gemini_generation` 部分定义的默认值。
- 如果客户端发送了这些被忽略的参数，系统会记录日志显示哪些参数被忽略。
- `system` 角色的消息会被合并，并作为普通 `user` 消息包含在发送给 Gemini 的 `contents` 中。

## 提示词模板 (`templates/default_prompt.yaml`)

提示词模板是一个 YAML 文件，定义了一个消息对象列表，也可以包含用于后处理的正则表达式规则。

**消息块**:
-   标准的 OpenAI 消息结构 (`role`, `content`)。
-   **特殊占位符**:
    -   一个字典对象 `type: api_input_placeholder`: 标记历史消息 (原始请求 `messages` 数组中除了最后一条用户消息之外的内容) 的插入位置。
    -   字符串 `{{user_input}}`: 将被替换为原始请求中最后一条用户消息的 `content`。
    -   字符串 `{{roll XdY}}`: 例如 `{{roll 1d20}}` 将被替换为一个1到20之间的随机数。
    -   字符串 `{{random::opt1::opt2}}`: 例如 `{{random::晴天::雨天}}` 将随机替换为 "晴天" 或 "雨天"。

**正则规则块**:
-   一个字典对象 `type: 正则`。
-   包含两个键:
    -   `查找`: 一个字符串，表示要查找的正则表达式模式。
    -   `替换`: 一个字符串，表示替换匹配到的内容的字符串。可以使用捕获组，如 `\1`, `\g<name>`。
-   所有正则规则会按照它们在 YAML 文件中定义的顺序，依次应用于从 Gemini API 返回并转换为 OpenAI 格式后的助手消息的 `content` 字段。

**示例 (`templates/default_prompt.yaml`)**:
```yaml
- role: system
  content: "你是一个友好的聊天机器人。今天的幸运数字是 {{roll 1d100}}。"
- type: api_input_placeholder
- role: user
  content: "用户最新的问题是：{{user_input}}。今天天气怎么样？可能是{{random::晴朗::多云::小雨}}。"
- role: assistant
  content: "我正在思考如何回答..." # 这是一个可选的固定助手前缀
# 示例正则规则：将所有 "Gemini" 替换为 "本AI助手"
- type: 正则
  查找: "Gemini"
  替换: "本AI助手"
# 示例正则规则：移除所有 "抱歉，" 开头的短语
- type: 正则
  查找: "^抱歉，"
  替换: ""
```
根据这个模板和用户的输入，最终发送给 Gemini 的 `contents` 会被动态构建，并且 Gemini 的响应会经过正则规则的处理。