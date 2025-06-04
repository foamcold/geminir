# src/conversion_utils.py
"""
此模块负责在 OpenAI 和 Gemini API 格式之间进行数据转换。

主要功能包括：
- 将 OpenAI 聊天完成请求格式转换为 Gemini API 请求格式。
- 将 Gemini API 响应格式转换为 OpenAI 聊天完成响应格式。
- 处理参数映射，如 temperature, max_tokens, top_p, stop 等。
- 处理消息格式转换，包括角色映射和内容结构调整。
- 应用正则表达式规则对响应内容进行后处理。
"""
import json
import time
import logging
import re # <--- 新增导入 re 模块
from typing import Dict, Any, List, Optional

from google.genai import types as google_genai_types # 导入 google.genai.types 以便类型注解

from .template_handler import _apply_regex_rules_to_content # 导入正则规则应用函数
from .config import settings # 导入应用配置

logger = logging.getLogger(settings.app_name) # 获取logger实例

# --- OpenAI 请求到 Gemini 请求的转换函数 ---

def convert_openai_to_gemini_request(
    original_openai_request: Dict[str, Any],
    prepared_messages: List[Dict[str, Any]] # 接收已处理的消息列表
    # API 密钥不由请求体直接处理，而是由 SDK 客户端在初始化时处理。
) -> Dict[str, Any]:
    """
    将 OpenAI 兼容的聊天请求体（结合已预处理的消息）转换为 Gemini API 所需的请求体格式。

    此函数使用 `prepared_messages` (已由 `_prepare_openai_messages` 处理过)
    作为 Gemini `contents` 的基础。
    然后，它将原始 OpenAI 请求中的其他参数（如 temperature, top_p, max_tokens, stop）
    映射到 Gemini 的 `generationConfig` 中。

    Args:
        original_openai_request (Dict[str, Any]): 原始的 OpenAI 格式聊天请求体。
            用于提取 temperature, top_p, max_tokens, stop 等参数。
        prepared_messages (List[Dict[str, Any]]): 已经过 `_prepare_openai_messages`
            处理（模板注入、动态变量、合并等）的 OpenAI 格式消息列表。

    Returns:
        Dict[str, Any]: 转换后的 Gemini API 请求体字典。主要包含：
                        - "contents": Gemini 格式的消息列表。
                        - "generationConfig" (optional): 包含生成参数的字典。
    """
    # 记录初始的 OpenAI 请求
    logger.debug(f"初始的 OpenAI 请求: {json.dumps(original_openai_request, ensure_ascii=False, default=str)}")
    
    # 新增日志：打印传入的 prepared_messages
    logger.debug(f"进入 convert_openai_to_gemini_request 时，prepared_messages 内容: {json.dumps(prepared_messages, ensure_ascii=False, default=str)}")

    # 1. 使用传入的已预处理的 OpenAI 消息
    openai_messages = prepared_messages # 直接使用传入的处理后消息
    # original_model_name 通常在调用此函数之前从 prepared_data 中提取，
    # 此函数不直接处理模型名称的传递。

    # 2. 将 OpenAI 消息转换为 Gemini `contents` 格式
    gemini_contents: List[Dict[str, Any]] = []
    # Gemini API v1beta 及更高版本中，system_instruction 已被整合到 contents 中，
    # 通常作为第一个 'user' 角色的消息的一部分，或者通过特定的 'system' 角色（如果API支持）。
    # 当前实现将 'system' 消息转换为 'user' 角色。

    for msg_idx, msg in enumerate(openai_messages): # 添加索引用于日志
        role = msg.get("role")
        raw_content = msg.get("content") # 获取原始 content，可以是 str 或 list
        
        gemini_parts: List[Dict[str, Any]] = []
        
        # 角色映射
        gemini_role = "user" # 默认角色
        if role == "user" or role == "system": # OpenAI system message 转为 Gemini user message
            gemini_role = "user"
        elif role == "assistant":
            gemini_role = "model"
        else:
            logger.warning(f"在 OpenAI 消息中遇到不支持的角色 '{role}' (消息 #{msg_idx})，该消息将被跳过。")
            continue

        # 处理 content
        if isinstance(raw_content, str):
            if raw_content: # 只有非空字符串才添加 text part
                gemini_parts.append({"text": raw_content})
        elif isinstance(raw_content, list): # OpenAI 多模态格式
            for item_idx, item_part in enumerate(raw_content):
                if not isinstance(item_part, dict):
                    logger.warning(f"多模态 content 中的项 (消息 #{msg_idx}, 项 #{item_idx}) 不是字典格式，已跳过: {item_part}")
                    continue
                
                part_type = item_part.get("type")

                if part_type == "text":
                    text_content = item_part.get("text", "")
                    if text_content: # 只有非空文本才添加
                         gemini_parts.append({"text": text_content})
                elif part_type == "image_url":
                    image_url_obj = item_part.get("image_url", {})
                    url_data = image_url_obj.get("url", "")
                    
                    match = re.match(r"^data:(image\/(?:png|jpeg|webp|heic|heif));base64,(.+)$", url_data)
                    if match:
                        mime_type = match.group(1)
                        base64_data = match.group(2)
                        gemini_parts.append({
                            "inline_data": {
                                "mime_type": mime_type,
                                "data": base64_data
                            }
                        })
                    else:
                        logger.warning(f"无法解析 image_url data URI 或不支持的 MIME 类型 (消息 #{msg_idx}, 项 #{item_idx}): {url_data[:100]}...")
                else:
                    logger.warning(f"多模态 content 中 (消息 #{msg_idx}, 项 #{item_idx}) 遇到不支持的 part type: '{part_type}'，已跳过。")
        elif raw_content is None:
            logger.debug(f"消息 role '{role}' (消息 #{msg_idx}) 的 content 为 None，将生成空的 parts 列表。")
        else:
            logger.warning(f"消息 role '{role}' (消息 #{msg_idx}) 的 content 类型未知 ({type(raw_content).__name__})，尝试转换为字符串。")
            try:
                str_content = str(raw_content)
                if str_content:
                    gemini_parts.append({"text": str_content})
            except Exception as e_str_conv:
                logger.error(f"无法将未知类型的 content ({type(raw_content).__name__}) (消息 #{msg_idx}) 转换为字符串: {e_str_conv}")

        if gemini_parts:
            gemini_contents.append({"role": gemini_role, "parts": gemini_parts})
        elif role and raw_content is None:
             logger.info(f"OpenAI 消息 (role: {role}, 消息 #{msg_idx}) content 为 None，转换后无有效 parts，此消息未添加到 Gemini contents。")
        elif not gemini_parts and raw_content:
            logger.warning(f"OpenAI 消息 (role: {role}, 消息 #{msg_idx}) content 无法转换为有效的 Gemini parts，此消息未添加到 Gemini contents。原始 content: {str(raw_content)[:200]}")

    # 构建 Gemini 请求体
    gemini_request: Dict[str, Any] = {
        "contents": gemini_contents,
        # "model" 字段在 Gemini API 请求中通常不是顶级字段，而是在创建 GenerativeModel 客户端时指定，
        # 或者在某些特定 REST API 端点作为路径参数。对于 SDK 的 generate_content 方法，模型名是顶级参数。
        # 此函数返回的是用于 SDK 方法的参数结构，所以不包含 model。
    }
    
    # 3. 准备 `generationConfig`
    generation_config: Dict[str, Any] = {}
    
    # 获取配置文件中的默认 Gemini 生成参数
    default_config = settings.proxy.gemini_generation
    
    # 映射 OpenAI 参数到 Gemini `generationConfig`，强制使用配置文件默认值，忽略客户端传递的参数
    
    # 温度 (temperature) - 强制使用配置文件默认值
    if "temperature" in original_openai_request and original_openai_request["temperature"] is not None:
        logger.info(f"客户端请求的 temperature 值为 {original_openai_request['temperature']}，但将使用配置文件默认值 {default_config.temperature}")
    generation_config["temperature"] = default_config.temperature

    # Top P (top_p -> topP) - 强制使用配置文件默认值
    if "top_p" in original_openai_request and original_openai_request["top_p"] is not None:
        logger.info(f"客户端请求的 top_p 值为 {original_openai_request['top_p']}，但将使用配置文件默认值 {default_config.top_p}")
    generation_config["topP"] = default_config.top_p
            
    # 最大令牌数 (max_tokens -> maxOutputTokens) - 强制使用配置文件默认值
    if "max_tokens" in original_openai_request and original_openai_request["max_tokens"] is not None:
        logger.info(f"客户端请求的 max_tokens 值为 {original_openai_request['max_tokens']}，但将使用配置文件默认值 {default_config.max_output_tokens}")
    generation_config["maxOutputTokens"] = default_config.max_output_tokens

    # Top K - 强制使用配置文件默认值
    if "top_k" in original_openai_request and original_openai_request["top_k"] is not None:
        logger.info(f"客户端请求的 top_k 值为 {original_openai_request['top_k']}，但将使用配置文件默认值 {default_config.top_k}")
    generation_config["topK"] = default_config.top_k

    # 候选数量 - 强制使用配置文件默认值
    if "candidate_count" in original_openai_request and original_openai_request["candidate_count"] is not None:
        logger.info(f"客户端请求的 candidate_count 值为 {original_openai_request['candidate_count']}，但将使用配置文件默认值 {default_config.candidate_count}")
    generation_config["candidateCount"] = default_config.candidate_count

    # 显示思考过程和思考预算 - 强制使用配置文件默认值
    # 处理 show_thinking 参数 - 强制使用配置文件默认值
    if "show_thinking" in original_openai_request and original_openai_request["show_thinking"] is not None:
        logger.info(f"客户端请求的 show_thinking 值为 {original_openai_request['show_thinking']}，但将使用配置文件默认值 {default_config.show_thinking}")
    include_thoughts_value = default_config.show_thinking

    # 处理 thinking_budget 参数 - 强制使用配置文件默认值
    if "thinking_budget" in original_openai_request and original_openai_request["thinking_budget"] is not None:
        logger.info(f"客户端请求的 thinking_budget 值为 {original_openai_request['thinking_budget']}，但将使用配置文件默认值 {default_config.thinking_budget}")
    thinking_budget_value = default_config.thinking_budget

    # 停止序列 (stop -> stopSequences) - 忽略客户端传递的值
    if "stop" in original_openai_request and original_openai_request["stop"] is not None:
        logger.info(f"客户端请求包含 stop 参数 {original_openai_request['stop']}，但将被忽略，不设置停止序列")
    # 不设置 stopSequences，使用 Gemini 默认行为

    # 安全设置 (safety_settings) 通常在调用 Gemini API 时作为顶层参数传递给 SDK 方法，
    # 或者包含在 `generationConfig` 中（取决于具体的 SDK 版本和方法）。
    # 此处转换的 `generation_config` 不直接包含 `safety_settings`，
    # 它们将在 `gemini_client.py` 中被添加到最终的 `types.GenerateContentConfig` 对象中。

    # 总是添加 generation_config 到请求中，因为现在包含了默认值
    gemini_request["generationConfig"] = generation_config
    logger.debug(f"构建的 Gemini generationConfig: {json.dumps(generation_config, ensure_ascii=False)}")
    
    # 添加 thinking_config 到请求中（如果有任何思考相关配置）
    thinking_config = {}
    if include_thoughts_value is not None:
        thinking_config["includeThoughts"] = include_thoughts_value
    if thinking_budget_value is not None:  # 移除 > 0 的条件，因为 0 是有效值
        thinking_config["thinkingBudget"] = thinking_budget_value
    
    if thinking_config:
        gemini_request["thinkingConfig"] = thinking_config
        logger.debug(f"构建的 Gemini thinkingConfig: {json.dumps(thinking_config, ensure_ascii=False)}")
    
    return gemini_request

# --- Gemini 响应到 OpenAI 响应的转换函数 ---

def convert_gemini_response_to_openai_chat_completion(
    gemini_response: Any, # 期望是 google.generativeai.types.GenerateContentResponse 对象
    original_openai_request_model: str, # 来自原始 OpenAI 请求的模型名称
    request_id: str, # 为此聊天完成生成的唯一 ID
    prompt_tokens: Optional[int] = None, # 可选：来自 Gemini usage_metadata 的提示令牌数
    candidates_tokens: Optional[int] = None, # 可选：来自 Gemini usage_metadata 的候选内容令牌数
    total_tokens: Optional[int] = None, # 可选：来自 Gemini usage_metadata 的总令牌数
    regex_rules: Optional[List[Dict[str, Any]]] = None # 可选：用于处理响应内容的正则规则
) -> Dict[str, Any]:
    """
    将 Gemini API 的响应（`GenerateContentResponse` 对象或其字典表示）
    转换为 OpenAI `/v1/chat/completions` API 的响应格式。

    Args:
        gemini_response (Any): Gemini SDK 返回的响应对象，通常是
                               `google.generativeai.types.GenerateContentResponse` 类型，
                               或在某些情况下可能是其字典表示。
        original_openai_request_model (str): 原始 OpenAI 请求中指定的模型名称，
                                             将用于填充 OpenAI 响应中的 "model" 字段。
        request_id (str): 为此 OpenAI 聊天完成生成的唯一请求 ID。
        prompt_tokens (Optional[int]): 从 Gemini 响应的 `usage_metadata` 中提取的提示令牌数。
        candidates_tokens (Optional[int]): 从 Gemini 响应的 `usage_metadata` 中提取的完成令牌数。
        total_tokens (Optional[int]): 从 Gemini 响应的 `usage_metadata` 中提取的总令牌数。
        regex_rules (Optional[List[Dict[str, Any]]]): 用于处理助手响应内容的正则规则列表。

    Returns:
        Dict[str, Any]: 转换后的 OpenAI 聊天完成格式的字典。
    """
    # 记录初始的 Gemini 响应（使用 str() 因为可能是复杂对象）
    logger.debug(f"初始的 Gemini 响应: {str(gemini_response)}")
    
    # Gemini 的 FinishReason 枚举值到 OpenAI finish_reason 字符串的映射
    GEMINI_TO_OPENAI_FINISH_REASON_MAP = {
        google_genai_types.FinishReason.STOP.name: "stop",
        google_genai_types.FinishReason.MAX_TOKENS.name: "length",
        google_genai_types.FinishReason.SAFETY.name: "content_filter",
        google_genai_types.FinishReason.RECITATION.name: "content_filter", # 将 RECITATION 也视为内容过滤
        google_genai_types.FinishReason.OTHER.name: "stop", # 其他未知原因默认为 stop
        google_genai_types.FinishReason.FINISH_REASON_UNSPECIFIED.name: "stop", # 未指定原因默认为 stop
    }

    assistant_content = "" # 初始化助手回复内容
    finish_reason_str = "stop" # 默认的完成原因

    try:
        # 优先处理 SDK 返回的 GenerateContentResponse 对象
        if hasattr(gemini_response, 'candidates') and gemini_response.candidates:
            # 通常我们只关心第一个候选者
            candidate = gemini_response.candidates[0]
            
            # 提取内容
            if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts') and candidate.content.parts:
                # 连接所有 part 的文本内容，尽管对于纯文本模型通常只有一个 part
                assistant_content_parts = [part.text for part in candidate.content.parts if hasattr(part, 'text')]
                assistant_content = "".join(assistant_content_parts)

            # 提取并映射完成原因
            gemini_finish_reason_enum = getattr(candidate, 'finish_reason', None)
            if gemini_finish_reason_enum is not None:
                # 获取枚举成员的名称 (例如 "STOP", "MAX_TOKENS")
                gemini_finish_reason_name = getattr(gemini_finish_reason_enum, 'name', str(gemini_finish_reason_enum))
                finish_reason_str = GEMINI_TO_OPENAI_FINISH_REASON_MAP.get(gemini_finish_reason_name, "stop")
                if gemini_finish_reason_name not in GEMINI_TO_OPENAI_FINISH_REASON_MAP:
                    logger.warning(f"未映射的 Gemini finish_reason: '{gemini_finish_reason_name}'。将默认为 'stop'。")
            else:
                logger.warning("Gemini 响应候选者中未找到 'finish_reason' 属性。")


        # 处理可能的字典格式的响应 (例如，如果响应被 JSON 序列化和反序列化过)
        elif isinstance(gemini_response, dict):
            logger.debug("Gemini 响应是字典类型，尝试按字典结构解析。")
            candidates_list = gemini_response.get("candidates", [])
            if candidates_list:
                candidate_dict = candidates_list[0]
                content_data = candidate_dict.get("content", {})
                parts_data = content_data.get("parts", [])
                assistant_content_parts = [part.get("text", "") for part in parts_data if "text" in part]
                assistant_content = "".join(assistant_content_parts)
                
                # Gemini REST API JSON 可能使用 "finishReason" (驼峰)
                gemini_finish_reason_val = candidate_dict.get("finishReason") 
                if not gemini_finish_reason_val: # 尝试兼容 "finish_reason" (下划线)
                     gemini_finish_reason_val = candidate_dict.get("finish_reason")
                
                # 将 finish reason (可能是字符串或数字) 转为大写字符串以匹配映射表
                finish_reason_str = GEMINI_TO_OPENAI_FINISH_REASON_MAP.get(str(gemini_finish_reason_val).upper(), "stop")
                if gemini_finish_reason_val and str(gemini_finish_reason_val).upper() not in GEMINI_TO_OPENAI_FINISH_REASON_MAP:
                     logger.warning(f"从字典解析时，未映射的 Gemini finish_reason: '{gemini_finish_reason_val}'。将默认为 'stop'。")
            else: 
                # 如果字典中没有候选者，检查是否有 promptFeedback (可能因为安全策略被阻止)
                logger.warning("Gemini 响应 (字典格式) 中没有候选者。")
                prompt_feedback = gemini_response.get("promptFeedback")
                if prompt_feedback:
                    block_reason = prompt_feedback.get("blockReason")
                    if block_reason:
                        logger.warning(f"Gemini promptFeedback 指示请求被阻止。原因: {block_reason}")
                        finish_reason_str = "content_filter" # 标记为内容过滤
                        assistant_content = f"请求因安全策略被阻止。原因: {block_reason}。"
                        safety_ratings_details = prompt_feedback.get("safetyRatings", [])
                        if safety_ratings_details:
                             assistant_content += f" 安全评分详情: {json.dumps(safety_ratings_details)}"
        else: 
            # 响应结构未知
            logger.error(f"预期的 Gemini 响应结构无法解析: 类型为 {type(gemini_response)}。无法提取内容。")
            assistant_content = "[错误：无法解析 Gemini 响应]"
            finish_reason_str = "stop" # 或其他错误指示

    except Exception as e:
        logger.error(f"处理 Gemini 响应时发生错误: {e}", exc_info=settings.debug_mode)
        assistant_content = f"[错误：处理 Gemini 响应时发生异常: {e}]"
        finish_reason_str = "stop" # 或其他错误指示

    # 对提取到的助手内容应用后处理的正则表达式规则
    if isinstance(assistant_content, str) and assistant_content: # 确保是字符串且非空
        # 临时解决方案：传递空的正则规则列表，因为在响应处理阶段我们无法确定使用的模板
        processed_assistant_content = _apply_regex_rules_to_content(assistant_content, regex_rules or [])
        if processed_assistant_content != assistant_content:
            logger.info("助手回复内容已通过 convert_gemini_response_to_openai_chat_completion 中的正则规则处理。")
            assistant_content = processed_assistant_content
        # else:
            # logger.debug("正则规则未改变 convert_gemini_response_to_openai_chat_completion 中的助手回复内容。")

    # 构建 OpenAI 格式的响应字典
    openai_response = {
        "id": request_id, # 使用传入的请求 ID
        "object": "chat.completion",
        "created": int(time.time()), # 当前时间戳
        "model": original_openai_request_model, # 使用原始请求中的模型名称
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": assistant_content,
                },
                "finish_reason": finish_reason_str,
                # "logprobs": null, # OpenAI API 中 logprobs 默认为 null
            }
        ],
        "usage": { 
            "prompt_tokens": prompt_tokens if prompt_tokens is not None else 0,
            "completion_tokens": candidates_tokens if candidates_tokens is not None else 0, # OpenAI 称之为 completion_tokens
            "total_tokens": total_tokens if total_tokens is not None else 0,
        }
    }
    
    # 根据 OpenAI 规范，即使 token 计数未知，usage 字段也通常存在并用0填充。
    # 如果所有 token 计数都为 None，可以选择移除 "usage" 字段或保持用0填充。当前选择保持。
    if prompt_tokens is None and candidates_tokens is None and total_tokens is None:
        logger.debug("未能从 Gemini 响应中获取 token 使用数据，OpenAI 响应中的 usage 将使用0填充。")
        pass # 保持 usage 字段和0值

    logger.debug(f"转换后的 OpenAI 聊天完成响应: {json.dumps(openai_response, ensure_ascii=False, default=str)}")
    return openai_response

# --- End OpenAI to Gemini Conversion Functions ---