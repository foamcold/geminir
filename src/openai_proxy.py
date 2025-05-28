# src/openai_proxy.py
import httpx
import yaml
import copy
import asyncio
import json
import logging # 添加日志模块
import time # 添加 time 模块
import re # 新增：正则表达式模块
import random # 新增：随机模块
from fastapi import HTTPException
from typing import List, Dict, Any, AsyncGenerator, Optional

from .config import settings
import os # 用于检查文件修改时间
import uuid # 用于生成请求 ID

import google.generativeai as genai
from google.generativeai.types import GenerateContentResponse, GenerationConfig # 用于类型提示和访问特定字段
# Consider specific error types from google.api_core.exceptions if needed for granular error handling

logger = logging.getLogger(settings.app_name) # 获取logger实例

# 默认的 Gemini 安全设置 (强制应用到每个请求)
DEFAULT_GEMINI_SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
    # {"category": "HARM_CATEGORY_CIVIC_INTEGRITY", "threshold": "BLOCK_NONE"} # Removed due to KeyError
]

# 全局变量用于缓存模板和跟踪修改时间
_CACHED_PROMPT_BLUEPRINTS: List[Dict[str, Any]] = [] # 用于构建请求的提示
_CACHED_REGEX_RULES: List[Dict[str, str]] = []    # 用于处理响应的正则规则
_LAST_TEMPLATE_MTIME: float = 0.0
_TEMPLATE_PATH: str = settings.proxy.prompt_template_path # 从配置获取路径

def _load_templates(force_reload: bool = False) -> None:
    """
    加载或热加载提示词模板，并分离正则规则。
    如果文件未更改且非强制重载，则不执行操作。
    """
    global _CACHED_PROMPT_BLUEPRINTS, _CACHED_REGEX_RULES, _LAST_TEMPLATE_MTIME, _TEMPLATE_PATH
    
    try:
        current_mtime = os.path.getmtime(_TEMPLATE_PATH)
    except FileNotFoundError:
        if not hasattr(_load_templates, '_logged_not_found_paths'):
            _load_templates._logged_not_found_paths = set()
        if _TEMPLATE_PATH not in _load_templates._logged_not_found_paths:
            logger.error(f"提示词模板文件 '{_TEMPLATE_PATH}' 未找到。")
            _load_templates._logged_not_found_paths.add(_TEMPLATE_PATH)
        _CACHED_PROMPT_BLUEPRINTS = []
        _CACHED_REGEX_RULES = []
        _LAST_TEMPLATE_MTIME = 0.0
        return

    if not force_reload and current_mtime == _LAST_TEMPLATE_MTIME and _LAST_TEMPLATE_MTIME != 0.0:
        return

    logger.info(f"尝试加载/热加载模板文件: '{_TEMPLATE_PATH}' (上次修改时间: {_LAST_TEMPLATE_MTIME}, 当前文件修改时间: {current_mtime})")
    try:
        with open(_TEMPLATE_PATH, "r", encoding="utf-8") as f:
            loaded_yaml_content = yaml.safe_load(f)
        
        if hasattr(_load_templates, '_logged_not_found_paths') and _TEMPLATE_PATH in _load_templates._logged_not_found_paths:
            _load_templates._logged_not_found_paths.remove(_TEMPLATE_PATH)

        if isinstance(loaded_yaml_content, list):
            new_blueprints = []
            new_regex_rules = []
            for item in loaded_yaml_content:
                if isinstance(item, dict):
                    item_type = item.get("type")
                    if item_type == "正则":
                        find_pattern = item.get("查找")
                        replace_pattern = item.get("替换")
                        if find_pattern is not None and replace_pattern is not None:
                            new_regex_rules.append({
                                "查找": str(find_pattern),
                                "替换": str(replace_pattern)
                            })
                        else:
                            logger.warning(f"模板中的 '正则' 类型块缺少 '查找' 或 '替换' 字段，或其值为 None，已忽略: {item}")
                    else:
                        new_blueprints.append(item)
                else:
                    logger.warning(f"模板文件 '{_TEMPLATE_PATH}' 中包含非字典类型的顶层列表项，已忽略: {item}")
            
            _CACHED_PROMPT_BLUEPRINTS = new_blueprints
            _CACHED_REGEX_RULES = new_regex_rules
            _LAST_TEMPLATE_MTIME = current_mtime
            logger.info(f"提示词模板 '{_TEMPLATE_PATH}' 已成功加载/热加载。提示词块数: {len(_CACHED_PROMPT_BLUEPRINTS)}, 正则规则数: {len(_CACHED_REGEX_RULES)}")
        else:
            logger.warning(f"加载/热加载模板 '{_TEMPLATE_PATH}' 失败：文件内容不是一个列表。将保留上一个有效版本（如果有）。")
            if _LAST_TEMPLATE_MTIME == 0.0:
                 _CACHED_PROMPT_BLUEPRINTS = []
                 _CACHED_REGEX_RULES = []
    except yaml.YAMLError as e:
        logger.error(f"解析模板文件 '{_TEMPLATE_PATH}' 失败: {e}。将保留上一个有效版本（如果有）。")
        if _LAST_TEMPLATE_MTIME == 0.0:
            _CACHED_PROMPT_BLUEPRINTS = []
            _CACHED_REGEX_RULES = []
    except Exception as e:
        logger.error(f"加载模板文件 '{_TEMPLATE_PATH}' 时发生未知错误: {e}。将保留上一个有效版本（如果有）。")
        if _LAST_TEMPLATE_MTIME == 0.0:
            _CACHED_PROMPT_BLUEPRINTS = []
            _CACHED_REGEX_RULES = []

# 应用启动时首次加载模板
_load_templates(force_reload=True)


# --- 新增：动态变量处理函数 ---
def _process_dice_rolls(text_content: str) -> str:
    """处理文本中的 {{roll XdY}} 骰子变量"""
    if not isinstance(text_content, str):
        return text_content

    def replace_dice_roll(match):
        try:
            num_dice = int(match.group(1))
            num_sides = int(match.group(2))
            if num_dice <= 0 or num_sides <= 0:
                return f"{{roll {num_dice}d{num_sides} - 无效的骰子参数}}"
            
            total_roll = sum(random.randint(1, num_sides) for _ in range(num_dice))
            logger.debug(f"处理骰子变量: {{roll {num_dice}d{num_sides}}} -> {total_roll}")
            return str(total_roll)
        except ValueError:
            return f"{{roll {match.group(1)}d{match.group(2)} - 参数非整数}}"
        except Exception as e:
            logger.error(f"处理骰子变量 {{roll {match.group(1)}d{match.group(2)}}} 时出错: {e}")
            return f"{{roll {match.group(1)}d{match.group(2)} - 处理错误}}"

    # 正则表达式查找 {{roll XdY}}，允许数字前后有空格
    # 使用非贪婪匹配，以防一个变量处理函数干扰另一个（如果它们有相似的定界符）
    # 但这里 {{...}} 是明确的，所以贪婪/非贪婪影响不大
    return re.sub(r"\{\{roll\s*(\d+)\s*d\s*(\d+)\s*\}\}", replace_dice_roll, text_content)

def _process_random_choices(text_content: str) -> str:
    """处理文本中的 {{random::opt1::opt2...}} 随机选择变量"""
    if not isinstance(text_content, str):
        return text_content

    def replace_random_choice(match):
        try:
            options_str = match.group(1)
            if not options_str: # 处理 {{random::}} 这种情况
                return "{{random:: - 无选项}}"
            options = options_str.split('::')
            if not all(options): # 如果分割后有空字符串选项，例如 {{random::a::::b}}
                 logger.warning(f"随机选择变量 {{random::{options_str}}} 包含空选项。")
                 # 可以选择过滤掉空选项，或者将其视为有效选项
                 options = [opt for opt in options if opt] # 过滤空选项
                 if not options:
                     return "{{random:: - 过滤后无有效选项}}"

            chosen = random.choice(options)
            logger.debug(f"处理随机选择变量: {{random::{options_str}}} -> {chosen}")
            return chosen
        except Exception as e:
            logger.error(f"处理随机选择变量 {{random::{match.group(1)}}} 时出错: {e}")
            return f"{{random::{match.group(1)}}} - 处理错误}}"
            
    # 正则表达式查找 {{random::...}}
    # (.*?)是非贪婪匹配，匹配两个::之间的任何字符，直到第一个}}
    return re.sub(r"\{\{random::(.*?)\}\}", replace_random_choice, text_content)

def _apply_regex_rules_to_content(text_content: str) -> str:
    """
    按顺序将缓存的正则规则应用于给定的文本内容。
    """
    if not _CACHED_REGEX_RULES or not isinstance(text_content, str):
        return text_content

    current_content = text_content
    for rule_idx, rule in enumerate(_CACHED_REGEX_RULES):
        try:
            find_pattern = rule.get("查找", "")
            replace_pattern = rule.get("替换", "")
            # re.sub 支持在 replace_pattern 中使用 \1, \2, \g<0> 等
            processed_content = re.sub(find_pattern, replace_pattern, current_content)
            if processed_content != current_content:
                logger.debug(f"应用正则规则 #{rule_idx + 1}: 查找='{find_pattern}', 替换='{replace_pattern}'. 内容已更改。")
            else:
                logger.debug(f"应用正则规则 #{rule_idx + 1}: 查找='{find_pattern}'. 内容未更改。")
            current_content = processed_content
        except re.error as e:
            logger.error(f"应用正则规则 #{rule_idx + 1} (查找='{find_pattern}') 时发生正则表达式错误: {e}. 该规则被跳过。")
        except Exception as e:
            logger.error(f"应用正则规则 #{rule_idx + 1} 时发生未知错误: {e}. 该规则被跳过。")
    return current_content
# --- 结束：动态变量处理函数 ---


def _prepare_openai_messages(original_body: Dict[str, Any]) -> Dict[str, Any]:
    """
    准备 OpenAI 格式的 messages 列表，进行模板注入、动态变量处理和消息合并。
    返回一个包含 "model" 和处理后 "messages" 的字典。
    """
    _load_templates() # 确保模板已加载/热加载
    current_blueprints = _CACHED_PROMPT_BLUEPRINTS
    
    original_messages: List[Dict[str, Any]] = original_body.get("messages", [])
    if not isinstance(original_messages, list):
        # 如果 messages 不是列表（例如 None 或其他类型），视为空列表处理或根据需要抛出错误
        # 这里我们将其视为空列表，后续逻辑会处理空 messages 的情况
        logger.warning(f"请求体中的 'messages' 不是一个列表 (实际类型: {type(original_messages).__name__})，将视为空消息列表。")
        original_messages = []

    # 1. 模板注入和初步的动态变量处理 (user_input)
    processed_messages: List[Dict[str, Any]] = []
    last_user_input_content: str = ""

    # 提取原始消息中的历史记录和最后一个用户输入 (如果存在)
    historic_messages: List[Dict[str, Any]] = []
    if original_messages:
        if original_messages[-1].get("role") == "user":
            last_user_input_content = original_messages[-1].get("content", "")
            historic_messages = original_messages[:-1]
        else:
            historic_messages = original_messages
    
    if not current_blueprints: # 没有模板的情况
        processed_messages.extend(historic_messages)
        if last_user_input_content: # 确保最后一个用户输入也被加入
             processed_messages.append({"role": "user", "content": last_user_input_content})
        elif not historic_messages and original_messages and original_messages[-1].get("role") == "user":
            # 处理只有一个用户消息且无模板的情况
            processed_messages.append({"role": "user", "content": original_messages[-1].get("content","")})
        
        # 如果 processed_messages 仍然为空 (例如 original_messages 为空)，则保持为空
        # 如果 original_messages 不为空但上述逻辑未填充 processed_messages (例如只有 assistant 消息且无模板)
        # 则直接使用 original_messages
        if not processed_messages and original_messages:
            processed_messages = copy.deepcopy(original_messages)

    else: # 有模板的情况
        for blueprint_msg_template in current_blueprints:
            blueprint_msg = copy.deepcopy(blueprint_msg_template)
            if blueprint_msg.get("type") == "api_input_placeholder":
                processed_messages.extend(copy.deepcopy(historic_messages)) # 深拷贝以防修改原始数据
            else:
                content_template = blueprint_msg.get("content")
                if isinstance(content_template, str):
                    # 替换 {{user_input}}
                    temp_content = content_template.replace("{{user_input}}", last_user_input_content)
                    blueprint_msg["content"] = temp_content # 先赋值，后续统一处理动态变量
                processed_messages.append(blueprint_msg)
        
        # 如果模板中没有 api_input_placeholder，且用户有实际输入，需要将最后一个用户输入添加到末尾
        # 检查模板是否包含占位符
        has_placeholder = any(bp.get("type") == "api_input_placeholder" for bp in current_blueprints)
        if not has_placeholder and last_user_input_content:
            processed_messages.append({"role": "user", "content": last_user_input_content})
        elif not has_placeholder and not historic_messages and original_messages and original_messages[-1].get("role") == "user":
            # 处理只有一个用户消息，且模板无占位符的情况
             processed_messages.append({"role": "user", "content": original_messages[-1].get("content","")})


    # 2. 全局动态变量处理 ({{roll}}, {{random}}) 应用于所有消息
    final_messages_step1: List[Dict[str, Any]] = []
    for msg in processed_messages:
        new_msg = msg.copy() # 浅拷贝，因为 content 会被替换
        content = new_msg.get("content")
        if isinstance(content, str):
            content = _process_dice_rolls(content)
            content = _process_random_choices(content)
            new_msg["content"] = content
        final_messages_step1.append(new_msg)

    # 3. 移除 content 为空或 None 的消息
    final_messages_step2: List[Dict[str, Any]] = []
    if final_messages_step1:
        original_count = len(final_messages_step1)
        for msg in final_messages_step1:
            if isinstance(msg, dict) and msg.get("content") is not None and msg.get("content") != "":
                final_messages_step2.append(msg)
        if len(final_messages_step2) < original_count:
            logger.debug(f"移除了 {original_count - len(final_messages_step2)} 条 content 为空或 None 的消息。")
        if not final_messages_step2:
            logger.debug("所有消息因 content 为空或 None 被移除。")
    
    # 4. 合并相邻的同角色消息 (system, user, assistant)
    if not final_messages_step2:
        merged_messages: List[Dict[str, Any]] = []
    else:
        merged_messages = []
        current_message = copy.deepcopy(final_messages_step2[0]) # 深拷贝第一个消息
        
        for i in range(1, len(final_messages_step2)):
            next_msg = final_messages_step2[i]
            if next_msg.get("role") == current_message.get("role") and \
               isinstance(next_msg.get("content"), str) and \
               isinstance(current_message.get("content"), str): # 确保 content 是字符串才能合并
                current_message["content"] += "\n" + next_msg["content"]
            else:
                merged_messages.append(current_message)
                current_message = copy.deepcopy(next_msg) # 深拷贝下一个消息
        merged_messages.append(current_message) # 添加最后一个处理中的消息
        logger.debug(f"消息合并后，消息数量从 {len(final_messages_step2)} 变为 {len(merged_messages)}。")

    # 准备返回结果
    result = {
        "model": original_body.get("model"),
        "messages": merged_messages
    }
    
    if result.get("model") is None:
        logger.warning("请求中 model 参数为 None 或未提供。")
    if not merged_messages:
        logger.debug("预处理后 messages 列表为空。")
        
    logger.debug(f"预处理后的 OpenAI messages: {json.dumps(result, ensure_ascii=False)}")
    return result


# --- OpenAI to Gemini Conversion Functions ---

def convert_openai_to_gemini_request(
    original_openai_request: Dict[str, Any],
    # gemini_api_key: str # API key will be handled by the SDK client, not directly in the request body
) -> Dict[str, Any]:
    """
    Converts an OpenAI-compatible chat request body to a Gemini API request body.
    """
    prepared_data = _prepare_openai_messages(original_openai_request)
    openai_messages = prepared_data.get("messages", [])
    model_name = prepared_data.get("model")

    gemini_contents: List[Dict[str, Any]] = []
    # system_instructions_parts is removed as system messages are now part of contents

    for msg in openai_messages:
        role = msg.get("role")
        content = msg.get("content", "")
        if not isinstance(content, str): # Ensure content is a string
            content = str(content)

        if role == "system": # System messages are treated as user messages for Gemini contents
            gemini_contents.append({"role": "user", "parts": [{"text": content}]})
        elif role == "user":
            gemini_contents.append({"role": "user", "parts": [{"text": content}]})
        elif role == "assistant":
            gemini_contents.append({"role": "model", "parts": [{"text": content}]})
        else:
            logger.warning(f"Unsupported role '{role}' in OpenAI messages, skipping.")

    gemini_request: Dict[str, Any] = {
        # "model": model_name, # Model is specified when creating the GenerativeModel client
        "contents": gemini_contents,
        # system_instruction field is no longer added to gemini_request
    }
    
    # Prepare generationConfig
    generation_config: Dict[str, Any] = {}
    
    # Map OpenAI parameters to Gemini generationConfig
    # Temperature
    if "temperature" in original_openai_request and original_openai_request["temperature"] is not None:
        generation_config["temperature"] = float(original_openai_request["temperature"])
    
    # Top P
    if "top_p" in original_openai_request and original_openai_request["top_p"] is not None:
        generation_config["topP"] = float(original_openai_request["top_p"]) # Gemini uses topP

    # Max Tokens (max_tokens -> maxOutputTokens)
    if "max_tokens" in original_openai_request and original_openai_request["max_tokens"] is not None:
        generation_config["maxOutputTokens"] = int(original_openai_request["max_tokens"])

    # Stop Sequences (stop -> stopSequences)
    # OpenAI 'stop' can be a string or an array of strings. Gemini 'stopSequences' is an array of strings.
    openai_stop = original_openai_request.get("stop")
    if openai_stop is not None:
        if isinstance(openai_stop, str):
            generation_config["stopSequences"] = [openai_stop]
        elif isinstance(openai_stop, list) and all(isinstance(s, str) for s in openai_stop):
            generation_config["stopSequences"] = openai_stop
        else:
            logger.warning(f"Invalid 'stop' parameter format: {openai_stop}. Expected string or list of strings. Ignoring.")

    # Add hardcoded safety settings
    # These are now part of the generationConfig in the Gemini API
    # However, the Python SDK might handle safety_settings at the model.generate_content level,
    # or within generation_config. Let's assume it's part of generation_config for now.
    # The SDK's `generateContent` method takes `safety_settings` as a direct argument.
    # So, we will prepare it separately and not put it into `generation_config` here.
    # It will be passed to `model.generate_content(..., safety_settings=DEFAULT_GEMINI_SAFETY_SETTINGS)`

    if generation_config: # Only add if there are actual config values
        gemini_request["generationConfig"] = generation_config
        logger.debug(f"Constructed Gemini generationConfig: {json.dumps(generation_config, ensure_ascii=False)}")

    # The API key is not part of the request body for Gemini, it's used to initialize the client.
    # Model name is also typically handled by the client (e.g., genai.GenerativeModel(model_name))
    # So, the returned dict is primarily 'contents', 'system_instruction', and 'generationConfig'.
    
    logger.debug(f"Converted OpenAI request to Gemini request format: {json.dumps(gemini_request, ensure_ascii=False)}")
    return gemini_request

def convert_gemini_response_to_openai_chat_completion(
    gemini_response: Any, # Expecting google.generativeai.types.GenerateContentResponse
    original_openai_request_model: str, # The model name from the original OpenAI request
    request_id: str, # A unique ID for the chat completion
    prompt_tokens: Optional[int] = None, # Optional: from Gemini usage metadata
    candidates_tokens: Optional[int] = None, # Optional: from Gemini usage metadata
    total_tokens: Optional[int] = None # Optional: from Gemini usage metadata
) -> Dict[str, Any]:
    """
    Converts a Gemini API response (GenerateContentResponse object or dict)
    to an OpenAI /v1/chat/completions format.
    """
    # Placeholder for Gemini's FinishReason to OpenAI's finish_reason mapping
    GEMINI_TO_OPENAI_FINISH_REASON_MAP = {
        "STOP": "stop", # FINISH_REASON_STOP
        "MAX_TOKENS": "length", # FINISH_REASON_MAX_TOKENS
        "SAFETY": "content_filter", # FINISH_REASON_SAFETY
        "RECITATION": "content_filter", # FINISH_REASON_RECITATION (Treat as content_filter)
        "OTHER": "stop", # FINISH_REASON_OTHER (Default to stop)
        "FINISH_REASON_UNSPECIFIED": "stop", # Default to stop
        # Add more mappings as needed based on google.generativeai.types.FinishReason
    }

    assistant_content = ""
    finish_reason_str = "stop" # Default finish reason

    try:
        # Assuming gemini_response is the direct response object from SDK
        # or a dict representation of it.
        if hasattr(gemini_response, 'candidates') and gemini_response.candidates:
            candidate = gemini_response.candidates[0]
            
            if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts') and candidate.content.parts:
                # Concatenate text from all parts, though typically there's one for text models
                assistant_content_parts = [part.text for part in candidate.content.parts if hasattr(part, 'text')]
                assistant_content = "".join(assistant_content_parts)

            gemini_finish_reason = None
            if hasattr(candidate, 'finish_reason'):
                # gemini_response.candidates[0].finish_reason is an enum e.g. FinishReason.STOP
                # We need its string name for the map.
                gemini_finish_reason_enum_val = candidate.finish_reason
                # Try to get the name of the enum member
                if hasattr(gemini_finish_reason_enum_val, 'name'):
                    gemini_finish_reason = gemini_finish_reason_enum_val.name
                else: # If it's already a string or int, try to use it (less ideal)
                    gemini_finish_reason = str(gemini_finish_reason_enum_val)
            
            finish_reason_str = GEMINI_TO_OPENAI_FINISH_REASON_MAP.get(gemini_finish_reason, "stop")
            if gemini_finish_reason and gemini_finish_reason not in GEMINI_TO_OPENAI_FINISH_REASON_MAP:
                logger.warning(f"Unmapped Gemini finish_reason: '{gemini_finish_reason}'. Defaulting to 'stop'.")

        elif isinstance(gemini_response, dict): # Handling if it's a dict (e.g. from JSON)
            candidates = gemini_response.get("candidates", [])
            if candidates:
                candidate = candidates[0]
                content_data = candidate.get("content", {})
                parts_data = content_data.get("parts", [])
                assistant_content_parts = [part.get("text", "") for part in parts_data if "text" in part]
                assistant_content = "".join(assistant_content_parts)
                
                gemini_finish_reason = candidate.get("finishReason") # Note: Gemini JSON might use "finishReason"
                if not gemini_finish_reason: # try finish_reason if that's what the dict has
                     gemini_finish_reason = candidate.get("finish_reason")

                finish_reason_str = GEMINI_TO_OPENAI_FINISH_REASON_MAP.get(str(gemini_finish_reason).upper(), "stop")
                if gemini_finish_reason and str(gemini_finish_reason).upper() not in GEMINI_TO_OPENAI_FINISH_REASON_MAP:
                     logger.warning(f"Unmapped Gemini finish_reason from dict: '{gemini_finish_reason}'. Defaulting to 'stop'.")
            else: # No candidates in dict
                logger.warning("Gemini response (dict) has no candidates. Returning empty content.")
                # Check for prompt_feedback for safety issues if no candidates
                prompt_feedback = gemini_response.get("promptFeedback")
                if prompt_feedback:
                    block_reason = prompt_feedback.get("blockReason")
                    if block_reason:
                        logger.warning(f"Gemini prompt blocked. Reason: {block_reason}")
                        finish_reason_str = "content_filter" # Or a more specific error based on blockReason
                        # assistant_content could be an error message here
                        assistant_content = f"Request blocked due to: {block_reason}."
                        # Check safety ratings for details
                        safety_ratings_details = prompt_feedback.get("safetyRatings", [])
                        if safety_ratings_details:
                             assistant_content += f" Details: {json.dumps(safety_ratings_details)}"


        else: # Neither SDK object-like nor dict with candidates
            logger.error(f"Unexpected Gemini response structure: {type(gemini_response)}. Cannot extract content.")
            assistant_content = "[ERROR: Could not parse Gemini response]"
            finish_reason_str = "stop" # Or some error indicator

    except Exception as e:
        logger.error(f"Error processing Gemini response: {e}", exc_info=True)
        assistant_content = f"[ERROR: Exception during Gemini response processing: {e}]"
        finish_reason_str = "stop" # Or some error indicator

    # Apply regex rules to the extracted assistant content
    if isinstance(assistant_content, str) and assistant_content:
        processed_assistant_content = _apply_regex_rules_to_content(assistant_content)
        if processed_assistant_content != assistant_content:
            logger.info("Assistant content processed by regex rules in convert_gemini_response_to_openai_chat_completion.")
            assistant_content = processed_assistant_content
        else:
            logger.debug("Regex rules did not change assistant content in convert_gemini_response_to_openai_chat_completion.")

    openai_response = {
        "id": request_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": original_openai_request_model, # Use the model name from the original request
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": assistant_content,
                },
                "finish_reason": finish_reason_str,
            }
        ],
        "usage": { # Populate if available, otherwise use placeholders or omit
            "prompt_tokens": prompt_tokens if prompt_tokens is not None else 0,
            "completion_tokens": candidates_tokens if candidates_tokens is not None else 0,
            "total_tokens": total_tokens if total_tokens is not None else 0,
        }
    }
    
    # If tokens were not provided (e.g. from usage_metadata), we might want to remove the usage field
    # or ensure it's clear they are estimates/unavailable. For now, 0 is a placeholder.
    if prompt_tokens is None and candidates_tokens is None and total_tokens is None:
        # Optionally, remove usage if all are None, or keep with 0s.
        # Let's keep it with 0s for now, as per OpenAI spec it's often present.
        pass

    logger.debug(f"Converted Gemini response to OpenAI chat completion format: {json.dumps(openai_response, ensure_ascii=False)}")
    return openai_response

# --- End OpenAI to Gemini Conversion Functions ---


async def get_gemini_non_stream_response_and_convert_to_openai(
    original_body: Dict[str, Any], # Original OpenAI request body
    auth_header: str # Client's Authorization header (Bearer <gemini_api_key>)
) -> Dict[str, Any]:
    """
    Handles the full process:
    1. Extracts Gemini API key.
    2. Configures Gemini SDK.
    3. Converts OpenAI request to Gemini format.
    4. Calls Gemini API (non-stream).
    5. Converts Gemini response back to OpenAI format.
    """
    gemini_api_key = None
    if auth_header and auth_header.startswith("Bearer "):
        gemini_api_key = auth_header.split("Bearer ")[1]
    
    if not gemini_api_key:
        logger.error("Gemini API key not found in Authorization header.")
        raise HTTPException(status_code=401, detail="Missing Gemini API key in Authorization header. Expected 'Bearer <key>'.")

    try:
        genai.configure(api_key=gemini_api_key)
        logger.debug("Gemini SDK configured with API key from request.")
    except Exception as e:
        logger.error(f"Failed to configure Gemini SDK: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to configure Gemini SDK: {e}")

    # Get the target Gemini model name from the original OpenAI request
    # _prepare_openai_messages now returns a dict with "model" and "messages"
    prepared_openai_data = _prepare_openai_messages(original_body)
    target_gemini_model_name = prepared_openai_data.get("model")

    if not target_gemini_model_name:
        logger.error("Model name not found in the prepared OpenAI request data.")
        raise HTTPException(status_code=400, detail="Model name is required but was not found after processing the request.")
    
    # Convert OpenAI request to Gemini format
    # The convert_openai_to_gemini_request function itself calls _prepare_openai_messages
    # so we pass the original_body to it.
    try:
        gemini_request_payload = convert_openai_to_gemini_request(original_body) # This will now handle system messages as user messages
    except Exception as e:
        logger.error(f"Error converting OpenAI request to Gemini format: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error converting request to Gemini format: {e}")

    # Initialize the Gemini model (system_instruction is no longer passed here as per new plan)
    try:
        model = genai.GenerativeModel(model_name=target_gemini_model_name)
        logger.info(f"Initialized Gemini model: {target_gemini_model_name} (system messages are now part of 'contents')")
    except Exception as e: # Catching broad exception for model initialization issues
        logger.error(f"Failed to initialize Gemini model '{target_gemini_model_name}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to initialize Gemini model '{target_gemini_model_name}': {e}")

    # Extract parts for generate_content
    contents_for_gemini = gemini_request_payload.get("contents")
    # system_instruction is no longer a separate field in gemini_request_payload from convert_openai_to_gemini_request
    
    # GenerationConfig needs to be an instance of the SDK's GenerationConfig class if provided
    generation_config_dict = gemini_request_payload.get("generationConfig")
    sdk_generation_config: Optional[GenerationConfig] = None
    if generation_config_dict:
        try:
            # Filter out None values before passing to GenerationConfig constructor
            filtered_gc_dict = {k: v for k, v in generation_config_dict.items() if v is not None}
            sdk_generation_config = GenerationConfig(**filtered_gc_dict)
            logger.debug(f"SDK GenerationConfig created: {sdk_generation_config}")
        except Exception as e:
            logger.error(f"Failed to create SDK GenerationConfig from dict {generation_config_dict}: {e}", exc_info=True)
            # Decide: raise error or proceed without generation_config? For now, proceed without.
            sdk_generation_config = None


    gemini_response: Optional[GenerateContentResponse] = None
    try:
        logger.debug(f"Sending request to Gemini model '{target_gemini_model_name}'. "
                     f"Contents: {json.dumps(contents_for_gemini, ensure_ascii=False, default=str)}, "
                     f"SystemInstruction: None (handled as part of 'contents'), " # Corrected log message
                     f"GenerationConfig: {sdk_generation_config}, "
                     f"SafetySettings: {json.dumps(DEFAULT_GEMINI_SAFETY_SETTINGS, ensure_ascii=False)}")

        # Construct arguments for generate_content, only including non-None values
        gen_content_args = {
            "contents": contents_for_gemini,
            "safety_settings": DEFAULT_GEMINI_SAFETY_SETTINGS
        }
        # system_instruction is NOT passed to generate_content, it's part of the model initialization.
        
        if sdk_generation_config:
            gen_content_args["generation_config"] = sdk_generation_config
        
        # Make the API call (non-streaming)
        # Note: The Gemini SDK's generate_content is synchronous by default.
        # To use it in an async FastAPI endpoint, it should be run in a thread pool.
        # For now, let's make it a blocking call and address async execution later if performance issues arise.
        # Alternatively, check if there's an async version of the SDK or use httpx for direct API calls.
        # The `google-generativeai` library does not seem to have an async client out-of-the-box for generate_content.
        # We will run this synchronously for now.
        gemini_response_obj = await asyncio.to_thread(
            model.generate_content,
            **gen_content_args
        )
        
        # gemini_response_obj = model.generate_content(**gen_content_args) # Synchronous call

        logger.info(f"Received response from Gemini model '{target_gemini_model_name}'.")
        # logger.debug(f"Raw Gemini response object: {gemini_response_obj}") # Careful with logging full PII or large objects

    except Exception as e: # Catching broad google.api_core.exceptions or others
        # More specific error handling can be added here, e.g. for
        # google.api_core.exceptions.PermissionDenied (API key issue)
        # google.api_core.exceptions.InvalidArgument (bad request params)
        # google.generativeai.types.BlockedPromptException, google.generativeai.types.StopCandidateException
        logger.error(f"Error calling Gemini API for model '{target_gemini_model_name}': {e}", exc_info=True)
        
        # Try to get more details if it's a known Gemini SDK exception type
        error_detail_str = str(e)
        status_code = 500 # Default to internal server error

        # Example of more specific error handling (needs specific exception types imported)
        # from google.api_core import exceptions as google_exceptions
        # if isinstance(e, google_exceptions.PermissionDenied):
        #     status_code = 401 # Or 403
        #     error_detail_str = f"Gemini API permission denied (check API key or permissions): {e}"
        # elif isinstance(e, google_exceptions.InvalidArgument):
        #     status_code = 400
        #     error_detail_str = f"Invalid argument to Gemini API: {e}"
        # elif hasattr(e, 'message'): # For some SDK-specific exceptions
        # error_detail_str = e.message

        # A common error is when no candidates are returned due to safety filters
        # The SDK might raise specific exceptions for this, or it might be in the response object (handled below)
        # If an exception is raised *before* getting a response object (e.g. API key error),
        # then gemini_response_obj will not be set.

        raise HTTPException(status_code=status_code, detail=f"Error calling Gemini API: {error_detail_str}")

    # Extract usage metadata if available
    prompt_tokens, candidates_tokens, total_tokens = None, None, None
    if hasattr(gemini_response_obj, 'usage_metadata'):
        if gemini_response_obj.usage_metadata:
            prompt_tokens = gemini_response_obj.usage_metadata.prompt_token_count
            candidates_tokens = gemini_response_obj.usage_metadata.candidates_token_count
            total_tokens = gemini_response_obj.usage_metadata.total_token_count
            logger.debug(f"Gemini usage_metadata - prompt: {prompt_tokens}, candidates: {candidates_tokens}, total: {total_tokens}")
        else: # usage_metadata exists but is None or empty
            logger.debug("Gemini response has usage_metadata, but it's None or empty.")
    else: # usage_metadata attribute doesn't exist
        logger.debug("Gemini response does not have usage_metadata attribute.")


    # Generate a unique ID for the OpenAI-compatible response
    openai_response_id = f"chatcmpl-{uuid.uuid4().hex}"

    # Convert Gemini response to OpenAI chat completion format
    try:
        openai_formatted_response = convert_gemini_response_to_openai_chat_completion(
            gemini_response_obj,
            original_openai_request_model=target_gemini_model_name, # Use the model name client sent
            request_id=openai_response_id,
            prompt_tokens=prompt_tokens,
            candidates_tokens=candidates_tokens,
            total_tokens=total_tokens
        )
    except Exception as e:
        logger.error(f"Error converting Gemini response to OpenAI format: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error converting Gemini response to OpenAI format: {e}")

    logger.info(f"Successfully processed request for model '{target_gemini_model_name}' and converted to OpenAI format.")
    return openai_formatted_response


async def stream_openai_response_to_client( # 此函数将被移除，因为 Gemini 后端调用是非流式的
    original_body: Dict[str, Any],
    openai_target_url: str, # 此参数将变为 Gemini API 相关，或移除
    auth_header: str
) -> AsyncGenerator[str, None]:
    """从OpenAI获取流式响应并直接转发给客户端。(此函数将被移除)"""
    logger.warning("stream_openai_response_to_client 被调用，但计划移除此函数，因 Gemini 后端为非流式。将返回错误。")
    error_payload = {"error": {"message": "此流式接口不再支持直接调用后端流式API。", "type": "deprecated_stream_function"}}
    yield f"data: {json.dumps(error_payload)}\n\n"
    yield f"data: [DONE]\n\n"
    # 原始逻辑保留在此处作为参考，但不会被执行
    # new_request_body = _prepare_openai_request_body(original_body, stream_to_openai=True)
    # auth_header_to_log = f"Bearer {auth_header[7:12]}..." if auth_header and auth_header.startswith("Bearer ") and len(auth_header) > 12 else "Not a Bearer token or too short"
    # if not auth_header:
    #     auth_header_to_log = "None"
    # headers_for_openai = {"Authorization": auth_header, "Content-Type": "application/json", "Accept": "text/event-stream"}
    # logger.debug(f"准备发送给目标 API (流式) 的请求体: {json.dumps(new_request_body, ensure_ascii=False)}")
    # logger.debug(f"准备发送给目标 API (流式) 的请求头: {{'Authorization': '{auth_header_to_log}', 'Content-Type': 'application/json', 'Accept': 'text/event-stream'}}")
    # async with httpx.AsyncClient() as client:
    #     try:
    #         async with client.stream(
    #             "POST",
    #             openai_target_url,
    #             json=new_request_body,
    #             headers=headers_for_openai,
    #             timeout=settings.proxy.openai_request_timeout
    #         ) as response:
    #             if response.status_code != 200:
    #                 error_content_bytes = await response.aread()
    #                 error_detail = f"OpenAI API 流式请求初始错误: {response.status_code}"
    #                 try:
    #                     error_json = json.loads(error_content_bytes.decode('utf-8', errors='ignore'))
    #                     error_detail += f" - {error_json}"
    #                 except Exception:
    #                     error_detail += f" - {error_content_bytes.decode('utf-8', errors='ignore')}"
    #                 yield f"data: {json.dumps({'error': {'message': error_detail, 'code': response.status_code, 'type':'openai_stream_error'}})}\n\n"
    #                 yield f"data: [DONE]\n\n"
    #                 return
    #             async for chunk in response.aiter_bytes():
    #                 yield chunk.decode('utf-8', errors='ignore')
    #     except httpx.HTTPStatusError as exc:
    #         error_detail = f"OpenAI API 流式请求状态错误: {exc.response.status_code}"
    #         try: error_detail += f" - {exc.response.json()}"
    #         except Exception: error_detail += f" - {exc.response.text}"
    #         yield f"data: {json.dumps({'error': {'message': error_detail, 'code': exc.response.status_code, 'type':'http_status_error'}})}\n\n"
    #         yield f"data: [DONE]\n\n"
    #     except httpx.RequestError as exc:
    #         error_payload = {"error": {"message": f"请求 OpenAI API 网络错误 (流式): {str(exc)}", "type": "network_error"}}
    #         yield f"data: {json.dumps(error_payload)}\n\n"
    #         yield f"data: [DONE]\n\n"
    #     except Exception as e:
    #         error_payload = {"error": {"message": f"处理对OpenAI的流式请求时发生内部错误: {str(e)}", "type": "internal_stream_error"}}
    #         yield f"data: {json.dumps(error_payload)}\n\n"
    #         yield f"data: [DONE]\n\n"


async def fake_stream_generator_from_non_stream(
    non_stream_task: asyncio.Task, # Task that calls get_openai_non_stream_response
    original_body: Dict[str, Any]
) -> AsyncGenerator[str, None]:
    """
    假流式生成器。使用 asyncio.Event 定期发送心跳，直到非流式任务完成。
    然后处理响应（包括正则替换）并模拟流式 delta 发送。
    """
    heartbeat_interval = settings.proxy.fake_streaming.heartbeat_interval
    task_completion_event = asyncio.Event()
    _wrapped_task_result: Any = None
    _wrapped_task_exception: Optional[BaseException] = None # 捕获 BaseException 以包含 CancelledError

    async def _execute_and_signal_completion():
        nonlocal _wrapped_task_result, _wrapped_task_exception
        try:
            # 使用 asyncio.shield 保护任务不被外部取消，除非生成器本身被取消
            _wrapped_task_result = await asyncio.shield(non_stream_task)
            logger.debug(f"非流式任务在 _execute_and_signal_completion 中成功完成，结果: {_wrapped_task_result}")
        except asyncio.CancelledError as e: # 特别处理 CancelledError
            _wrapped_task_exception = e
            logger.debug(f"非流式任务在 _execute_and_signal_completion 中被取消: {e}")
        except BaseException as e: # 捕获其他所有 BaseException
            _wrapped_task_exception = e
            logger.debug(f"非流式任务在 _execute_and_signal_completion 中遇到异常: {e}")
        finally:
            task_completion_event.set()
            logger.debug("_execute_and_signal_completion 完成，事件已设置。")

    # 创建并启动后台任务来执行非流式请求并设置事件
    background_executor_task = asyncio.create_task(_execute_and_signal_completion())
    logger.debug(f"已创建后台任务 _execute_and_signal_completion: {background_executor_task.get_name()}")

    try:
        while not task_completion_event.is_set():
            try:
                # 等待事件被设置，带有超时
                await asyncio.wait_for(task_completion_event.wait(), timeout=heartbeat_interval)
                # 如果 wait_for 没有超时，说明事件被设置了，任务已完成或出错
                logger.debug("task_completion_event 已设置，退出心跳循环。")
                break
            except asyncio.TimeoutError:
                # 超时，发送心跳
                current_ts = int(time.time())
                model_name = original_body.get("model", "gpt-3.5-turbo-proxy-heartbeat")
                heartbeat_chunk = {
                    "id": f"chatcmpl-fake-hb-{current_ts}",
                    "object": "chat.completion.chunk",
                    "created": current_ts,
                    "model": model_name,
                    "choices": [{"index": 0, "delta": {"content": ""}, "finish_reason": None}]
                }
                heartbeat_message = f"data: {json.dumps(heartbeat_chunk)}\n\n"
                logger.debug(f"发送假流式心跳数据块: {heartbeat_message.strip()}")
                yield heartbeat_message
            except asyncio.CancelledError:
                logger.info("fake_stream_generator_from_non_stream 被取消，正在清理...")
                if not background_executor_task.done():
                    background_executor_task.cancel()
                    try:
                        await background_executor_task # 等待后台任务实际取消
                    except asyncio.CancelledError:
                        logger.debug("后台执行器任务成功取消。")
                    except Exception as e_bg_cancel:
                        logger.error(f"等待后台执行器任务取消时发生错误: {e_bg_cancel}")
                raise # 重新抛出 CancelledError 以便上层处理

        # 任务已完成（成功或失败），处理结果
        if _wrapped_task_exception:
            logger.error(f"非流式任务执行时捕获到异常: {_wrapped_task_exception}")
            if isinstance(_wrapped_task_exception, HTTPException):
                error_payload = {"error": {"message": str(_wrapped_task_exception.detail), "code": _wrapped_task_exception.status_code, "type": "api_error_in_fake_stream"}}
            elif isinstance(_wrapped_task_exception, asyncio.CancelledError):
                 error_payload = {"error": {"message": "请求在服务器端被取消 (fake stream)。", "type": "server_request_cancelled_in_fake_stream"}}
            else:
                error_payload = {"error": {"message": f"非流式任务执行时发生内部错误: {str(_wrapped_task_exception)}", "type": "internal_task_error_in_fake_stream"}}
            error_message = f"data: {json.dumps(error_payload)}\n\n"
            logger.debug(f"发送假流式错误数据 (来自 _wrapped_task_exception): {error_message.strip()}")
            yield error_message
        elif _wrapped_task_result is not None:
            full_response_data = _wrapped_task_result
            logger.debug(f"从非流式任务获取到的原始完整响应: {json.dumps(full_response_data, ensure_ascii=False)}")

            # 对助手消息应用正则规则 (如果存在)
            if isinstance(full_response_data, dict) and "choices" in full_response_data and full_response_data["choices"]:
                if full_response_data["choices"][0].get("message", {}).get("role") == "assistant":
                    original_content = full_response_data["choices"][0].get("message", {}).get("content", "")
                    if isinstance(original_content, str):
                        processed_content = _apply_regex_rules_to_content(original_content)
                        if processed_content != original_content:
                            logger.info("助手消息内容已通过正则规则处理。")
                            full_response_data["choices"][0]["message"]["content"] = processed_content
                        else:
                            logger.debug("正则规则未改变助手消息内容。")

            # 将可能已处理的非流式响应模拟成流式 delta 块发送
            if isinstance(full_response_data, dict) and "choices" in full_response_data and full_response_data["choices"]:
                choice = full_response_data["choices"][0]
                message = choice.get("message", {})
                
                resp_id = full_response_data.get("id", f"chatcmpl-simulated-{int(time.time())}")
                resp_model = full_response_data.get("model", original_body.get("model", "gpt-3.5-turbo-proxy"))
                resp_created = full_response_data.get("created", int(time.time()))
                
                role = message.get("role")
                if role:
                    role_chunk = {
                        "id": resp_id, "object": "chat.completion.chunk", "created": resp_created, "model": resp_model,
                        "choices": [{"index": 0, "delta": {"role": role}, "finish_reason": None}]
                    }
                    yield f"data: {json.dumps(role_chunk)}\n\n"

                content = message.get("content")
                if content is not None: # 即使是空字符串也发送
                    content_chunk = {
                        "id": resp_id, "object": "chat.completion.chunk", "created": resp_created, "model": resp_model,
                        "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}]
                    }
                    yield f"data: {json.dumps(content_chunk)}\n\n"
                
                finish_reason = choice.get("finish_reason")
                if finish_reason:
                    finish_chunk = {
                        "id": resp_id, "object": "chat.completion.chunk", "created": resp_created, "model": resp_model,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}]
                    }
                    yield f"data: {json.dumps(finish_chunk)}\n\n"
            else:
                # 增强日志，说明为什么格式不符合预期
                reason = ""
                if not isinstance(full_response_data, dict):
                    reason = f"因为它不是一个字典 (实际类型: {type(full_response_data).__name__})"
                elif "choices" not in full_response_data:
                    reason = "因为它缺少 'choices' 键"
                elif not full_response_data["choices"]: # choices 存在但是是空列表
                    reason = "因为 'choices' 键对应的值为空列表"
                else:
                    reason = "因为 'choices' 列表的第一个元素结构不符合预期 (例如缺少 'message' 或 'delta')"
                
                logger.warning(
                    f"接收到的 full_response_data 格式不符合预期的 chat.completion 结构。{reason} "
                    f"将尝试直接发送原始数据: {json.dumps(full_response_data, ensure_ascii=False)}"
                )
                yield f"data: {json.dumps(full_response_data)}\n\n"
        else:
            # 这种情况理论上不应该发生，因为 _execute_and_signal_completion 总会设置 _wrapped_task_result 或 _wrapped_task_exception
            logger.error("非流式任务完成但既无结果也无异常，这是一个意外状态。")
            fallback_error_payload = {"error": {"message": "非流式任务完成但状态未知。", "type": "unknown_task_completion_state"}}
            yield f"data: {json.dumps(fallback_error_payload)}\n\n"

    except asyncio.CancelledError: # 捕获生成器自身的取消
        logger.info("fake_stream_generator_from_non_stream 主体被取消。")
        if not background_executor_task.done():
            logger.debug("尝试取消后台执行器任务...")
            background_executor_task.cancel()
            try:
                await background_executor_task # 等待后台任务实际取消
            except asyncio.CancelledError:
                logger.debug("后台执行器任务在生成器取消时成功取消。")
            except Exception as e_bg_cancel_outer:
                logger.error(f"等待后台执行器任务在外层取消时发生错误: {e_bg_cancel_outer}")
        raise # 重新抛出 CancelledError
    except Exception as e_outer: # 捕获生成器主体的其他异常
        logger.error(f"fake_stream_generator_from_non_stream 发生意外错误: {e_outer}")
        error_payload = {"error": {"message": f"假流式生成器发生意外错误: {str(e_outer)}", "type": "fake_stream_generator_unexpected_error"}}
        yield f"data: {json.dumps(error_payload)}\n\n"
    finally:
        # 确保后台任务如果仍在运行且未被取消，则尝试取消它
        if 'background_executor_task' in locals() and not background_executor_task.done():
            logger.debug("fake_stream_generator_from_non_stream 在 finally 块中，后台任务未完成，尝试取消。")
            background_executor_task.cancel()
            try:
                await background_executor_task
            except asyncio.CancelledError:
                logger.debug("后台任务在 finally 块中成功取消。")
            except Exception as e_final_cancel:
                logger.error(f"在 finally 块中取消后台任务时发生错误: {e_final_cancel}")
        
        done_message = "data: [DONE]\n\n"
        logger.debug(f"发送假流式结束标记: {done_message.strip()}")
        yield done_message