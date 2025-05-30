# src/gemini_client.py
"""
此模块负责与 Google Gemini Pro API 进行直接交互。

主要功能包括：
- 初始化 Gemini SDK 客户端 (`genai.Client`)。
- 处理非流式的 Gemini API 请求。
- 管理 API 密钥和授权。
- 构建和发送请求到 Gemini 模型。
- 处理和转换 Gemini API 的响应。
- 应用默认的安全设置。
"""
import asyncio
import json
import logging
import uuid
from typing import Dict, Any, Optional, List

from fastapi import HTTPException

from google import genai
from google.genai import types # 导入 google.genai.types 以便类型注解

from .config import settings
from .template_handler import _prepare_openai_messages
from .conversion_utils import convert_openai_to_gemini_request, convert_gemini_response_to_openai_chat_completion

# 获取基于应用配置的日志记录器
logger = logging.getLogger(settings.app_name)

# 默认的 Gemini 安全设置 (类型化)
# 这些设置将应用于所有发送到 Gemini API 的请求，以控制内容安全策略。
# "BLOCK_NONE" 表示不阻止任何内容，可以根据需要调整。
DEFAULT_GEMINI_SAFETY_SETTINGS_TYPED: List[types.SafetySetting] = [
    types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
    types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
    types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
    types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE")
]

async def execute_non_stream_gemini_request(
    original_body: Dict[str, Any],
    auth_header: str,
    prepared_data: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    处理一个完整的非流式 Gemini API 请求。

    此函数按顺序执行以下操作：
    1. 从 `auth_header` 中提取 Gemini API 密钥。
    2. 使用提取的 API 密钥初始化 Gemini SDK 客户端。
    3. 调用 `template_handler._prepare_openai_messages` 准备 OpenAI 格式的消息体（例如，应用模板）。
    4. 调用 `conversion_utils.convert_openai_to_gemini_request` 将 OpenAI 请求体转换为 Gemini API 所需的格式。
    5. 构建 `types.GenerateContentConfig` 对象，合并用户指定的生成参数和默认的安全设置。
    6. 异步调用 Gemini SDK 的 `client.models.generate_content` 方法，向指定的 Gemini 模型发送非流式请求。
    7. 提取 Gemini 响应中的用量元数据（token 数量）。
    8. 调用 `conversion_utils.convert_gemini_response_to_openai_chat_completion` 将 Gemini 的响应转换回 OpenAI Chat Completion 格式。

    参数:
        original_body (Dict[str, Any]): 从客户端接收到的原始 OpenAI 格式的请求体。
        auth_header (str): HTTP 请求中的 Authorization 头部字符串，应包含 "Bearer <GEMINI_API_KEY>"。
        prepared_data (Optional[Dict[str, Any]]): 预先准备好的 OpenAI 消息体数据。

    返回:
        Dict[str, Any]: 转换回 OpenAI Chat Completion 格式的响应体。

    可能抛出的异常:
        HTTPException:
            - 状态码 401: 如果 `auth_header` 中缺少或格式不正确的 Gemini API 密钥。
            - 状态码 500: 如果 Gemini SDK 客户端初始化失败。
            - 状态码 400: 如果在准备好的 OpenAI 请求数据中找不到模型名称。
            - 状态码 500: 如果在 OpenAI 到 Gemini 请求转换过程中发生错误。
            - 状态码 500: 如果在创建 `types.GenerateContentConfig` 对象时发生错误（尽管会尝试回退）。
            - 状态码 500: 如果调用 Gemini API 时发生错误。
            - 状态码 500: 如果在 Gemini 到 OpenAI 响应转换过程中发生错误。
    """
    gemini_api_key: Optional[str] = None
    # 从 Authorization 头部提取 API 密钥
    if auth_header and auth_header.startswith("Bearer "):
        gemini_api_key = auth_header.split("Bearer ")[1]

    if not gemini_api_key:
        logger.error("在 Authorization 头部中未找到 Gemini API 密钥。")
        raise HTTPException(status_code=401, detail="在 Authorization 头部中缺少 Gemini API 密钥。期望格式为 'Bearer <key>'。")

    try:
        # 初始化 Gemini SDK 客户端
        # API 密钥通过参数传递，或者 SDK 会从 GOOGLE_API_KEY 环境变量中读取
        client = genai.Client(api_key=gemini_api_key)
        logger.debug("Gemini SDK 客户端已使用请求中的 API 密钥初始化。")
    except Exception as e:
        logger.error(f"初始化 Gemini SDK 客户端失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"初始化 Gemini SDK 客户端失败: {e}")

    # 通过模板处理器准备 OpenAI 消息体，并获取处理后的消息和目标 Gemini 模型名称
    if prepared_data is None:
        prepared_data = _prepare_openai_messages(original_body)
    prepared_messages: List[Dict[str, Any]] = prepared_data.get("messages", [])
    target_gemini_model_name: Optional[str] = prepared_data.get("model")
    selected_regex_rules: List[Dict[str, Any]] = prepared_data.get("selected_regex_rules", [])

    if not target_gemini_model_name:
        logger.error("在准备好的 OpenAI 请求数据中未找到模型名称 (可能在模板处理后丢失)。")
        raise HTTPException(status_code=400, detail="模型名称是必需的，但在处理请求后未找到。")

    # 将 OpenAI 请求（使用已处理的消息）转换为 Gemini API 格式
    try:
        # original_body 用于提取如 temperature, top_p 等参数
        # prepared_messages 是已处理过的消息列表
        gemini_request_payload = convert_openai_to_gemini_request(original_body, prepared_messages)
    except Exception as e:
        logger.error(f"将 OpenAI 请求转换为 Gemini 格式时出错: {e}", exc_info=True)
        # 确保抛出原始的 TypeError 或其他具体错误，以便调试
        if isinstance(e, TypeError):
            raise # 直接重新抛出 TypeError，因为它可能包含有用的参数信息
        raise HTTPException(status_code=500, detail=f"将请求转换为 Gemini 格式时出错: {e}")

    contents_for_gemini: Optional[List[Dict[str, Any]]] = gemini_request_payload.get("contents")

    # 获取用户指定的生成配置参数 (这些参数名已在转换函数中映射为 Gemini SDK 的名称)
    user_generation_config_dict: Dict[str, Any] = gemini_request_payload.get("generationConfig", {})

    # 构建最终的 types.GenerateContentConfig 对象
    final_config_params_for_sdk: Dict[str, Any] = {}
    if user_generation_config_dict:
        final_config_params_for_sdk.update(user_generation_config_dict)

    # 应用默认/强制的安全设置
    final_config_params_for_sdk['safety_settings'] = DEFAULT_GEMINI_SAFETY_SETTINGS_TYPED

    # 提取 thinkingConfig（如果存在）并添加到配置中
    thinking_config_dict = gemini_request_payload.get("thinkingConfig")
    if thinking_config_dict:
        try:
            thinking_config_obj = types.ThinkingConfig(**thinking_config_dict)
            final_config_params_for_sdk['thinking_config'] = thinking_config_obj
        except Exception as e:
            logger.error(f"从字典 {thinking_config_dict} 创建 SDK types.ThinkingConfig 失败: {e}", exc_info=True)

    sdk_final_config_obj: Optional[types.GenerateContentConfig] = None
    if final_config_params_for_sdk:
        try:
            sdk_final_config_obj = types.GenerateContentConfig(**final_config_params_for_sdk)
        except Exception as e:
            logger.error(f"从字典 {final_config_params_for_sdk} 创建 SDK types.GenerateContentConfig 失败: {e}", exc_info=True)

    # 调用 Gemini API
    try:
        logger.debug(f"正在向 Gemini 模型 '{target_gemini_model_name}' 发送请求。 "
                     f"内容 (Contents): {json.dumps(contents_for_gemini, ensure_ascii=False, default=str)}, "
                     # 使用 str() 记录 dataclass
                     f"配置 (Config): {str(sdk_final_config_obj) if sdk_final_config_obj else 'None'}")

        # 使用 asyncio.to_thread 在单独的线程中运行同步的 SDK 调用，以避免阻塞事件循环
        gemini_response_obj: types.GenerateContentResponse = await asyncio.to_thread(
            client.models.generate_content,
            model=target_gemini_model_name,
            contents=contents_for_gemini,
            config=sdk_final_config_obj  # 传递单个配置对象或 None
        )

        logger.info(f"已收到来自 Gemini 模型 '{target_gemini_model_name}' 的响应。")

    except Exception as e:
        logger.error(f"调用 Gemini API 模型 '{target_gemini_model_name}' 时出错: {e}", exc_info=True)
        error_detail_str = str(e)
        status_code = 500 # 默认为内部服务器错误
        # 此处可以根据 'e' 的类型进行更具体的错误映射
        raise HTTPException(status_code=status_code, detail=f"调用 Gemini API 时出错: {error_detail_str}")

    # 提取用量元数据（如果可用）
    prompt_tokens: Optional[int] = None
    candidates_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    if hasattr(gemini_response_obj, 'usage_metadata'):
        if gemini_response_obj.usage_metadata:
            prompt_tokens = gemini_response_obj.usage_metadata.prompt_token_count
            candidates_tokens = gemini_response_obj.usage_metadata.candidates_token_count
            total_tokens = gemini_response_obj.usage_metadata.total_token_count
            logger.debug(f"Gemini usage_metadata - prompt: {prompt_tokens}, candidates: {candidates_tokens}, total: {total_tokens}")
        else:
            logger.debug("Gemini 响应包含 usage_metadata，但其值为 None 或为空。")
    else:
        logger.debug("Gemini 响应不包含 usage_metadata 属性。")


    # 为 OpenAI 兼容的响应生成一个唯一的 ID
    openai_response_id = f"chatcmpl-{uuid.uuid4().hex}"

    # 将 Gemini 响应转换为 OpenAI Chat Completion 格式
    try:
        openai_formatted_response = convert_gemini_response_to_openai_chat_completion(
            gemini_response=gemini_response_obj, # 修正关键字参数名称
            original_openai_request_model=target_gemini_model_name, # 使用客户端发送的模型名称
            request_id=openai_response_id,
            prompt_tokens=prompt_tokens,
            candidates_tokens=candidates_tokens,
            total_tokens=total_tokens,
            regex_rules=selected_regex_rules # 传递选定模板的正则规则
            # is_stream 参数已移除，因为它在被调用函数中不存在
        )
    except Exception as e:
        logger.error(f"将 Gemini 响应转换为 OpenAI 格式时出错: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"将 Gemini 响应转换为 OpenAI 格式时出错: {e}")

    logger.info(f"已成功处理模型 '{target_gemini_model_name}' 的请求并转换为 OpenAI 格式。")
    return openai_formatted_response