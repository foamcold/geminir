# src/config.py
import yaml
from pydantic import BaseModel, Field, ValidationError
from typing import Optional

class FakeStreamingConfig(BaseModel):
    enabled: bool = True
    heartbeat_interval: int = Field(default=1, ge=1) # 心跳间隔至少1秒

class ProxyConfig(BaseModel):
    prompt_template_path: str = "templates/default_prompt.yaml"
    fake_streaming: FakeStreamingConfig = Field(default_factory=FakeStreamingConfig)
    openai_request_timeout: int = Field(default=60, ge=10) # 超时至少10秒

class AppSettings(BaseModel):
    app_name: str = "OpenAI Proxy with Custom Prompts"
    log_level: str = "INFO"
    proxy: ProxyConfig = Field(default_factory=ProxyConfig)

def load_config(path: str = "config/settings.yaml") -> AppSettings:
    try:
        with open(path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)
        if not config_data:
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
    except ValidationError as e:
        print(f"警告: 配置文件 {path} 验证错误: {e}，将使用默认设置。")
        return AppSettings()
    except Exception as e: # 更广泛的错误捕获
        print(f"警告: 加载配置 {path} 时发生未知错误: {e}，将使用默认设置。")
        return AppSettings()

settings = load_config()