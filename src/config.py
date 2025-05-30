# src/config.py
"""
此模块定义了应用配置的数据模型以及加载配置的逻辑。

它使用 Pydantic库 来定义配置结构、提供数据验证和默认值。
配置可以从 YAML 文件加载，如果文件不存在或格式错误，则会使用默认配置。
"""
import yaml
from pydantic import BaseModel, Field, ValidationError
from typing import Optional

class FakeStreamingConfig(BaseModel):
    """
    模拟流式响应相关的配置。
    """
    enabled: bool = Field(default=True, description="是否启用模拟流式响应功能。")
    heartbeat_interval: int = Field(default=1, ge=1, description="模拟流式响应时发送心跳信号的间隔时间（秒）。最小值应为1秒。") # 心跳间隔至少1秒

class GeminiGenerationConfig(BaseModel):
    """
    Gemini 2.5 Pro 生成参数配置。
    """
    temperature: float = Field(default=1.0, ge=0.0, le=2.0, description="控制生成文本的随机性。范围：0-2。")
    max_output_tokens: int = Field(default=65535, ge=1, le=65535, description="限制模型生成响应的最大词元数量。Gemini 2.5 Pro 上限为 65535。")
    top_p: float = Field(default=1.0, ge=0.0, le=1.0, description="控制采样时要考虑的词元比例。范围：0-1。")
    top_k: int = Field(default=64, ge=1, description="控制采样时要考虑的前 K 个词元。")
    candidate_count: int = Field(default=1, ge=1, le=8, description="指定模型生成多少个候选响应。范围：1-8。")
    show_thinking: bool = Field(default=False, description="Gemini 2.5 Pro 特色功能，设置为 true 时返回模型的内部思考过程。")
    thinking_budget: int = Field(default=1024, ge=0, le=24576, description="思考预算，指导模型在生成回答时可使用的思考 token 数量。范围：0-24576。设置为 0 会停用思考功能。")

class ProxyConfig(BaseModel):
    """
    代理核心功能相关的配置。
    """
    prompt_template_path_with_input: str = Field(default="templates/with_input.yaml", description="当用户输入 {{user_input}} 有内容时使用的提示词模板文件路径。")
    prompt_template_path_without_input: str = Field(default="templates/without_input.yaml", description="当用户输入 {{user_input}} 无内容时使用的提示词模板文件路径。")
    fake_streaming: FakeStreamingConfig = Field(default_factory=FakeStreamingConfig, description="模拟流式响应的特定配置。")
    openai_request_timeout: int = Field(default=60, ge=10, description="代理向目标 OpenAI/Gemini 服务发出请求的超时时间（秒）。最小值应为10秒。") # 超时至少10秒
    gemini_generation: GeminiGenerationConfig = Field(default_factory=GeminiGenerationConfig, description="Gemini 2.5 Pro 生成参数的默认配置。")

class AppSettings(BaseModel):
    """
    应用顶层配置模型，聚合了所有其他配置部分。
    """
    app_name: str = Field(default="geminir", description="应用程序的名称，主要用于日志记录。")
    log_level: str = Field(default="INFO", description="应用程序的日志级别 (例如 DEBUG, INFO, WARNING, ERROR)。")
    proxy: ProxyConfig = Field(default_factory=ProxyConfig, description="代理功能相关的配置。")
    debug_mode: bool = Field(default=False, description="是否启用调试模式。在调试模式下，可能会记录更详细的日志或启用特定的调试功能。")
    server_host: str = Field(default="0.0.0.0", description="Uvicorn 开发服务器监听的主机地址。")
    server_port: int = Field(default=8000, gt=0, lt=65536, description="Uvicorn 开发服务器监听的端口号。")


def load_config(path: str = "config/settings.yaml") -> AppSettings:
    """
    从指定的 YAML 文件加载应用配置。

    如果配置文件未找到、为空、或解析/验证失败，则会打印警告信息到控制台，
    并返回一个使用默认值的 `AppSettings` 实例。

    参数:
        path (str): 配置文件的路径。默认为 "config/settings.yaml"。

    返回:
        AppSettings: 加载并验证后的应用配置实例，或者在出错时返回默认配置实例。
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)
        if not config_data: # 处理空配置文件的情况
            print(f"警告: 配置文件 {path} 为空，将使用默认设置。")
            return AppSettings()
        # Pydantic 会自动处理嵌套字典到模型的转换
        return AppSettings(**config_data)
    except FileNotFoundError:
        print(f"警告: 配置文件 {path} 未找到，将使用默认设置。")
        return AppSettings()
    except yaml.YAMLError as e:
        print(f"警告: 配置文件 {path} 解析错误: {e}，将使用默认设置。")
        return AppSettings()
    except ValidationError as e: # Pydantic 验证错误
        print(f"警告: 配置文件 {path} 验证错误: {e}，将使用默认设置。")
        return AppSettings()
    except Exception as e: # 捕获其他更广泛的潜在错误
        print(f"警告: 加载配置 {path} 时发生未知错误: {e}，将使用默认设置。")
        return AppSettings()

# 全局配置实例：在模块加载时从配置文件或默认值初始化。
# 应用的其他部分可以通过导入 `settings` 来访问配置。
settings: AppSettings = load_config()