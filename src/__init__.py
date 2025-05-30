"""
src 包

此包包含了 OpenAI-Gemini 代理应用的核心源代码。
主要模块包括：
- main.py: FastAPI 应用入口和 API 端点定义。
- config.py: 应用配置模型和加载逻辑。
- gemini_client.py: 与 Google Gemini API 交互的客户端逻辑。
- conversion_utils.py: OpenAI 和 Gemini 请求/响应格式之间的转换工具。
- template_handler.py: 提示词模板加载和处理逻辑。
- streaming_utils.py: 模拟流式响应的工具。
"""
# 此文件使得 src 目录成为一个 Python 包。