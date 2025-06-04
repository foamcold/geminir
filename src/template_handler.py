# src/template_handler.py
"""
此模块负责处理提示词模板的加载、解析、动态变量替换以及最终消息的准备。

主要功能包括：
- 从 YAML 文件加载和缓存提示词模板及正则表达式规则。
- 支持模板的热重载，当模板文件更新时自动重新加载。
- 处理模板内容中的动态变量，例如：
    - 骰子投掷 (`{{roll XdY}}`): 模拟投掷X个Y面骰子，并替换为总点数。
    - 随机选择 (`{{random::opt1::opt2...}}`): 从提供的选项中随机选择一个。
- 应用用户定义的正则表达式规则对生成内容或模板内容进行后处理。
- 根据加载的模板、用户输入历史和最新的用户输入，构建最终用于提交给大语言模型的
  OpenAI 格式消息列表。这包括模板注入、消息合并等逻辑。
"""
import yaml
import copy
import logging
import re
import random
import os
import json # <--- 新增导入 json 模块
from typing import List, Dict, Any

from .config import settings

logger = logging.getLogger(settings.app_name) # 获取logger实例，用于记录模块相关信息

# --- 模块级全局变量 ---
_CACHED_PROMPT_BLUEPRINTS: Dict[str, List[Dict[str, Any]]] = {}  # 按模板路径缓存的提示词蓝图
_CACHED_REGEX_RULES: Dict[str, List[Dict[str, Any]]] = {}        # 按模板路径缓存的正则规则
_LAST_TEMPLATE_MTIME: Dict[str, float] = {}                      # 按模板路径缓存的最后修改时间

def _get_template_path_for_user_input(user_input_content: str) -> str:
    """
    根据用户输入内容选择合适的模板文件路径。
    
    Args:
        user_input_content (str): 用户输入的内容
        
    Returns:
        str: 选择的模板文件路径
    """
    if user_input_content and user_input_content.strip():
        # 用户输入有内容，使用 with_input 模板
        return settings.proxy.prompt_template_path_with_input
    else:
        # 用户输入无内容，使用 without_input 模板
        return settings.proxy.prompt_template_path_without_input

def _load_templates(template_path: str, force_reload: bool = False) -> None:
    """
    加载或热加载在 YAML 文件中定义的提示词模板和正则表达式规则。

    此函数从指定的 YAML 文件中读取模板内容，并将其分类为两种类型：
    - 正则表达式规则：用于在消息内容中查找和替换特定模式的字典项。
      此时应包含 `查找` (find_pattern) 和 `替换` (replace_pattern) 两个键。
    - 其他所有字典项被视为提示词蓝图，通常包含 `role` 和 `content` 键。

    此函数会按模板路径缓存加载的内容。只有在模板文件被修改或 `force_reload` 参数为 True 时，
    才会执行重新加载操作。

    Args:
        template_path (str): 模板文件的路径。
        force_reload (bool, optional): 如果为 True，则强制重新加载模板文件，
                                       忽略文件修改时间的比较。默认为 False。

    Returns:
        None: 此函数不返回任何值，但会更新模块级的全局缓存变量：
              `_CACHED_PROMPT_BLUEPRINTS` (按路径缓存的提示词蓝图字典),
              `_CACHED_REGEX_RULES` (按路径缓存的正则规则字典),
              和 `_LAST_TEMPLATE_MTIME` (按路径缓存的最后加载时文件的时间戳)。
    """
    global _CACHED_PROMPT_BLUEPRINTS, _CACHED_REGEX_RULES, _LAST_TEMPLATE_MTIME
    
    try:
        current_mtime = os.path.getmtime(template_path) # 获取模板文件当前的最后修改时间
    except FileNotFoundError:
        # 文件未找到时的处理逻辑
        if not hasattr(_load_templates, '_logged_not_found_paths'):
            # 使用函数属性（静态变量类似物）来跟踪已记录的未找到路径集合，避免日志刷屏
            _load_templates._logged_not_found_paths = set()
        if template_path not in _load_templates._logged_not_found_paths:
            logger.error(f"提示词模板文件 '{template_path}' 未找到。将使用空模板和规则。")
            _load_templates._logged_not_found_paths.add(template_path) # 将路径加入已记录集合
        # 清空该路径的缓存，确保后续逻辑使用空模板/规则
        _CACHED_PROMPT_BLUEPRINTS[template_path] = []
        _CACHED_REGEX_RULES[template_path] = []
        _LAST_TEMPLATE_MTIME[template_path] = 0.0 # 重置时间戳
        return

    # 条件性跳过加载：
    # 1. 非强制重载
    # 2. 当前文件修改时间与上次加载时相同
    # 3. 上次加载时间戳不为0（表示至少成功加载过一次）
    if not force_reload and current_mtime == _LAST_TEMPLATE_MTIME.get(template_path, 0.0) and _LAST_TEMPLATE_MTIME.get(template_path, 0.0) != 0.0:
        logger.debug(f"模板文件 '{template_path}' 未更改，跳过加载。")
        return

    logger.info(f"尝试加载/热加载模板文件: '{template_path}' (上次修改时间: {_LAST_TEMPLATE_MTIME.get(template_path, 0.0)}, 当前文件修改时间: {current_mtime})")
    try:
        with open(template_path, "r", encoding="utf-8") as f:
            loaded_yaml_content = yaml.safe_load(f) # 从 YAML 文件安全加载内容
        
        # 如果之前记录过此路径未找到，现在找到了就从集合中移除，以便下次找不到时能再次记录
        if hasattr(_load_templates, '_logged_not_found_paths') and template_path in _load_templates._logged_not_found_paths:
            _load_templates._logged_not_found_paths.remove(template_path)

        if isinstance(loaded_yaml_content, list):
            # YAML 内容成功加载且为列表类型
            new_blueprints = [] # 用于存储本次加载的提示词蓝图
            new_regex_rules = []  # 用于存储本次加载的正则规则
            for item_idx, item in enumerate(loaded_yaml_content): # 遍历YAML中的每个顶层列表项
                if isinstance(item, dict):
                    item_type = item.get("type") # 获取项的类型，用于区分正则规则和普通提示
                    if item_type == "正则":
                        # 处理正则表达式规则
                        find_pattern = item.get("查找")
                        replace_pattern = item.get("替换")
                        rule_action = item.get("action", "replace") # 默认为替换操作
                        
                        if find_pattern is not None and replace_pattern is not None:
                            rule_entry = {
                                "查找": str(find_pattern),    # 确保查找模式是字符串
                                "替换": str(replace_pattern),  # 确保替换内容是字符串
                                "action": str(rule_action).lower() # 确保action是小写字符串
                            }
                            # 如果 action 是 json_payload，可以添加额外的验证或处理
                            if rule_entry["action"] == "json_payload":
                                try:
                                    # 尝试解析替换模式（此时应为JSON字符串）以验证其有效性
                                    json.loads(rule_entry["替换"]) 
                                    logger.debug(f"加载 JSON 载荷正则规则 #{len(new_regex_rules)+1}: 查找='{find_pattern}', 动作='json_payload'")
                                except json.JSONDecodeError:
                                    logger.warning(f"模板文件 '{template_path}' 中的 '正则' 类型块 (索引 {item_idx}) action 为 'json_payload' 但 '替换' 字段不是有效的JSON字符串: '{replace_pattern}'。此规则可能无法按预期工作。")
                                    # 仍然添加规则，但记录警告
                            
                            new_regex_rules.append(rule_entry)
                            if rule_entry["action"] != "json_payload": # 避免重复记录已记录的json_payload
                                logger.debug(f"加载正则规则 #{len(new_regex_rules)}: 查找='{find_pattern}', 替换='{replace_pattern}', 动作='{rule_action}'")
                        else:
                            logger.warning(f"模板文件 '{template_path}' 中的 '正则' 类型块 (索引 {item_idx}) 缺少 '查找' 或 '替换' 字段，或其值为 None，已忽略: {item}")
                    else:
                        # 非 "正则" 类型的项视为提示词蓝图
                        new_blueprints.append(item)
                else:
                    logger.warning(f"模板文件 '{template_path}' 中包含非字典类型的顶层列表项 (索引 {item_idx})，已忽略: {item}")
            
            # 成功解析完所有项后，更新该路径的缓存
            _CACHED_PROMPT_BLUEPRINTS[template_path] = new_blueprints
            _CACHED_REGEX_RULES[template_path] = new_regex_rules
            _LAST_TEMPLATE_MTIME[template_path] = current_mtime # 更新最后修改时间戳
            logger.info(f"提示词模板 '{template_path}' 已成功加载/热加载。提示词块数: {len(new_blueprints)}, 正则规则数: {len(new_regex_rules)}")
        else:
            # YAML 文件内容不是列表，加载失败
            logger.warning(f"加载/热加载模板 '{template_path}' 失败：文件内容不是一个列表 (实际类型: {type(loaded_yaml_content).__name__})。将保留上一个有效版本（如果有）。")
            if _LAST_TEMPLATE_MTIME.get(template_path, 0.0) == 0.0: # 如果是首次加载就失败，则确保缓存为空
                 _CACHED_PROMPT_BLUEPRINTS[template_path] = []
                 _CACHED_REGEX_RULES[template_path] = []
    except yaml.YAMLError as e:
        # YAML 解析错误
        logger.error(f"解析模板文件 '{template_path}' 失败: {e}。将保留上一个有效版本（如果有）。")
        if _LAST_TEMPLATE_MTIME.get(template_path, 0.0) == 0.0: # 首次加载失败
            _CACHED_PROMPT_BLUEPRINTS[template_path] = []
            _CACHED_REGEX_RULES[template_path] = []
    except Exception as e:
        # 其他未知错误
        logger.error(f"加载模板文件 '{template_path}' 时发生未知错误: {e}。将保留上一个有效版本（如果有）。", exc_info=settings.debug_mode)
        if _LAST_TEMPLATE_MTIME.get(template_path, 0.0) == 0.0: # 首次加载失败
            _CACHED_PROMPT_BLUEPRINTS[template_path] = []
            _CACHED_REGEX_RULES[template_path] = []

# 应用启动时执行一次模板加载，确保初始状态正确
_load_templates(template_path=settings.proxy.prompt_template_path_with_input, force_reload=True)
_load_templates(template_path=settings.proxy.prompt_template_path_without_input, force_reload=True)

def _process_dice_rolls(text_content: str) -> str:
    """
    处理文本内容中的骰子投掷变量 `{{roll XdY}}`。

    将匹配到的 `{{roll XdY}}` 替换为 X 个 Y 面骰子的投掷结果总和。
    例如 `{{roll 2d6}}` 会被替换为一个表示两次六面骰子投掷总和的数字字符串。

    Args:
        text_content (str): 可能包含骰子变量的原始文本内容。

    Returns:
        str: 处理过骰子变量后的文本内容。如果输入不是字符串，则原样返回。
             如果骰子参数无效或处理出错，会将错误信息嵌入替换后的字符串中。
    """
    if not isinstance(text_content, str):
        logger.debug("输入到 _process_dice_rolls 的内容非字符串，跳过处理。")
        return text_content # 如果输入不是字符串，直接返回

    def replace_dice_roll(match: re.Match) -> str:
        """内部辅助函数，用于替换单个骰子匹配项。"""
        try:
            num_dice = int(match.group(1))    # 骰子数量 X
            num_sides = int(match.group(2))   # 骰子面数 Y

            if num_dice <= 0 or num_sides <= 0:
                logger.warning(f"无效的骰子参数: {{roll {num_dice}d{num_sides}}}，数量和面数必须为正。")
                return f"{{roll {num_dice}d{num_sides} - 无效的骰子参数}}" # 参数错误提示
            
            # 模拟投掷
            total_roll = sum(random.randint(1, num_sides) for _ in range(num_dice))
            logger.debug(f"处理骰子变量: {{roll {num_dice}d{num_sides}}} -> {total_roll}")
            return str(total_roll)
        except ValueError:
            # 参数无法转换为整数
            logger.warning(f"骰子参数无法转换为整数: {{roll {match.group(1)}d{match.group(2)}}}")
            return f"{{roll {match.group(1)}d{match.group(2)} - 参数非整数}}"
        except Exception as e:
            # 其他处理错误
            logger.error(f"处理骰子变量 {{roll {match.group(1)}d{match.group(2)}}} 时出错: {e}", exc_info=settings.debug_mode)
            return f"{{roll {match.group(1)}d{match.group(2)} - 处理错误}}"

    # 使用正则表达式查找所有 {{roll XdY}} 格式的变量，并用其投掷结果替换
    # \s* 允许数字前后有空格，例如 {{roll 2 d 6}}
    return re.sub(r"\{\{roll\s*(\d+)\s*d\s*(\d+)\s*\}\}", replace_dice_roll, text_content)

def _process_random_choices(text_content: str) -> str:
    """
    处理文本内容中的随机选择变量 `{{random::opt1::opt2...}}`。

    将匹配到的 `{{random::opt1::opt2...}}` 替换为从 `opt1`, `opt2` 等选项中
    随机选择的一个。选项之间用 `::` 分隔。

    Args:
        text_content (str): 可能包含随机选择变量的原始文本内容。

    Returns:
        str: 处理过随机选择变量后的文本内容。如果输入不是字符串，则原样返回。
             如果无选项或处理出错，会将错误信息嵌入替换后的字符串中。
    """
    if not isinstance(text_content, str):
        logger.debug("输入到 _process_random_choices 的内容非字符串，跳过处理。")
        return text_content # 如果输入不是字符串，直接返回

    def replace_random_choice(match: re.Match) -> str:
        """内部辅助函数，用于替换单个随机选择匹配项。"""
        try:
            options_str = match.group(1) # 获取 `::` 分隔的选项字符串
            if not options_str: # 例如 {{random::}}
                logger.warning("随机选择变量 {{random::}} 无任何选项。")
                return "{{random:: - 无选项}}"
            
            options = options_str.split('::') # 按 '::' 分割选项
            
            # 检查并处理空选项，例如 {{random::a::::b}} 会产生空字符串
            if not all(options): 
                 logger.warning(f"随机选择变量 {{random::{options_str}}} 包含空选项。将过滤空选项。")
                 options = [opt for opt in options if opt] # 过滤掉空字符串选项
                 if not options: # 如果过滤后没有有效选项了
                     logger.warning(f"随机选择变量 {{random::{options_str}}} 过滤空选项后无有效选项。")
                     return "{{random:: - 过滤后无有效选项}}"

            chosen = random.choice(options) # 从有效选项中随机选择一个
            logger.debug(f"处理随机选择变量: {{random::{options_str}}} -> {chosen}")
            return chosen
        except Exception as e:
            # 处理错误
            logger.error(f"处理随机选择变量 {{random::{match.group(1)}}} 时出错: {e}", exc_info=settings.debug_mode)
            return f"{{random::{match.group(1)}}} - 处理错误}}"
            
    # 使用正则表达式查找所有 {{random::...}} 格式的变量
    # (.*?) 是非贪婪匹配，匹配两个 "random::" 和 "}}" 之间的任何字符
    return re.sub(r"\{\{random::(.*?)\}\}", replace_random_choice, text_content)

def _apply_regex_rules_to_content(text_content: str, regex_rules: List[Dict[str, Any]]) -> str:
    """
    按顺序将指定的正则表达式规则应用于给定的文本内容。

    这些规则从模板文件中加载（类型为 "正则" 的项）。
    每个规则包含 "查找" (正则表达式模式) 和 "替换" (替换字符串) 以及可选的 "action"。
    支持的 action:
        - "replace" (默认): 执行标准的查找和替换。
        - "json_payload": 将 `text_content` 视为 JSON 字符串，将 `rule['替换']` 也视为 JSON 字符串。
                          尝试将 `rule['替换']` 中的 "payload" 键的值，更新或添加到 `text_content` JSON 对象
                          的 "tool_code_interpreter_output" 键下（如果已存在则更新，不存在则创建）。
                          `rule['查找']` 在此 action 下通常不直接用于 re.sub，而是可能用于条件判断（当前未实现）。

    Args:
        text_content (str): 需要应用正则规则的原始文本内容。
        regex_rules (List[Dict[str, Any]]): 要应用的正则规则列表。

    Returns:
        str: 应用所有正则规则处理后的文本内容。
             如果无规则或输入不是字符串，则原样返回。
    """
    if not regex_rules: # 如果没有正则规则
        logger.debug("无正则规则，跳过 _apply_regex_rules_to_content 处理。")
        return text_content
    if not isinstance(text_content, str): # 如果输入不是字符串
        logger.debug("输入到 _apply_regex_rules_to_content 的内容非字符串，跳过处理。")
        return text_content

    current_content = text_content
    logger.debug(f"开始对内容应用 {len(regex_rules)} 条正则规则。")
    for rule_idx, rule in enumerate(regex_rules): # 遍历所有规则
        try:
            find_pattern = rule.get("查找", "")
            replace_pattern = rule.get("替换", "")
            action = rule.get("action", "replace") # 默认为 "replace"

            if action == "json_payload":
                logger.debug(f"处理 JSON 载荷规则 #{rule_idx + 1}: 查找='{find_pattern}'")
                try:
                    # 替换模式此时应为包含 "payload" 键的 JSON 字符串
                    payload_obj_from_rule = json.loads(replace_pattern)
                    payload_to_inject = payload_obj_from_rule.get("payload")

                    if payload_to_inject is None:
                        logger.warning(f"JSON 载荷规则 #{rule_idx + 1} 的 '替换' JSON 中缺少 'payload' 键或其值为 null，规则跳过。替换内容: {replace_pattern}")
                        continue

                    # 原始内容也应为 JSON 字符串，或者至少是可解析为 JSON 的
                    try:
                        current_content_obj = json.loads(current_content)
                        if not isinstance(current_content_obj, dict):
                             logger.warning(f"JSON 载荷规则 #{rule_idx + 1}: 当前内容解析为 JSON 但不是字典类型 (类型: {type(current_content_obj).__name__})，无法注入 payload。规则跳过。")
                             continue
                    except json.JSONDecodeError:
                         # 如果当前内容不是有效的 JSON，则创建一个新的字典结构
                         logger.warning(f"JSON 载荷规则 #{rule_idx + 1}: 当前内容不是有效的 JSON，将创建新的 JSON 结构以注入 payload。原始内容: '{current_content[:100]}...'")
                         current_content_obj = {} # 或者可以决定如何处理这种情况，例如跳过

                    # 将 payload 注入或更新到 current_content_obj 的 "tool_code_interpreter_output" 键
                    # 注意：这里简单地覆盖或设置。如果需要更复杂的合并逻辑，需要相应修改。
                    current_content_obj["tool_code_interpreter_output"] = payload_to_inject
                    
                    # 将修改后的对象转换回 JSON 字符串
                    processed_content = json.dumps(current_content_obj, ensure_ascii=False, indent=2)
                    logger.info(f"JSON 载荷规则 #{rule_idx + 1} 应用成功。'tool_code_interpreter_output' 已更新/设置。")

                except json.JSONDecodeError as jde:
                    logger.error(f"JSON 载荷规则 #{rule_idx + 1} 处理时发生 JSON 解析错误: {jde}. '替换'内容: '{replace_pattern}', 当前内容: '{current_content[:100]}...'. 规则跳过。")
                    continue # 跳过此规则
                except Exception as e_json_op:
                    logger.error(f"JSON 载荷规则 #{rule_idx + 1} 处理时发生未知错误: {e_json_op}. 规则跳过。", exc_info=settings.debug_mode)
                    continue # 跳过此规则
            
            elif action == "replace":
                # 使用 re.sub 应用标准正则表达式替换
                processed_content = re.sub(find_pattern, replace_pattern, current_content)
                if processed_content != current_content:
                    logger.debug(f"应用替换正则规则 #{rule_idx + 1}: 查找='{find_pattern}', 替换='{replace_pattern}'. 内容已更改。")
            else:
                logger.warning(f"未知的正则规则 action: '{action}' (规则 #{rule_idx + 1})。规则跳过。")
                processed_content = current_content # 保持不变

            current_content = processed_content
        except re.error as e_re:
            # 正则表达式本身有错误 (主要针对 action="replace")
            logger.error(f"应用正则规则 #{rule_idx + 1} (查找='{find_pattern}') 时发生正则表达式错误: {e_re}. 该规则被跳过。")
        except Exception as e_outer_rule:
            # 其他未知错误
            logger.error(f"应用正则规则 #{rule_idx + 1} (查找='{find_pattern}', 替换='{replace_pattern}') 时发生未知错误: {e_outer_rule}. 该规则被跳过。", exc_info=settings.debug_mode)
    
    logger.debug("所有正则规则应用完毕。")
    return current_content

def _prepare_openai_messages(original_body: Dict[str, Any]) -> Dict[str, Any]:
    """
    根据原始请求体中的消息、加载的提示词模板和动态变量处理，准备最终的 OpenAI 格式消息列表。

    处理流程包括：
    1. 提取原始消息中的历史记录和最后一个用户输入。
    2. 根据用户输入内容选择合适的模板文件（有输入/无输入）。
    3. 确保选定的模板已加载（调用 `_load_templates`）。
    4. 根据模板注入历史消息和用户输入：
        - 如果模板中有 `type: api_input_placeholder`，则在该位置插入历史消息。
        - 将模板内容中的 `{{user_input}}` 替换为最后一个用户输入。
    5. 对所有生成的消息内容应用全局动态变量处理 ({{roll}}, {{random}})
    6. 移除内容为空或 None 的消息。
    7. 合并相邻的 system 和 user 消息，只与 assistant 消息交替

    Args:
        original_body (Dict[str, Any]): 原始的 OpenAI 格式请求体，
                                       期望包含 "messages" (消息列表) 和 "model" (模型名称) 键。

    Returns:
        Dict[str, Any]: 一个包含三个键的字典：
                        - "model": 从原始请求中获取的模型名称。
                        - "messages": 处理和合并后的最终 OpenAI 格式消息列表。
                        - "selected_regex_rules": 选定模板的正则规则列表，用于响应处理阶段。
    """
    original_messages: List[Dict[str, Any]] = original_body.get("messages", [])
    if not isinstance(original_messages, list):
        logger.warning(f"请求体中的 'messages' 不是一个列表 (实际类型: {type(original_messages).__name__})，将视为空消息列表。")
        original_messages = [] # 如果 messages 无效，视为空列表

    # 1. 提取历史消息和最后一个用户输入
    processed_messages: List[Dict[str, Any]] = [] # 用于构建处理后的消息列表
    raw_last_user_content: Any = ""               # 存储最后一个用户的原始 content (可以是 str 或 list)
    text_for_template_processing: str = ""        # 存储从最后一个用户 content 中提取的纯文本，用于模板逻辑
    historic_messages: List[Dict[str, Any]] = []  # 存储除最后一个用户输入外的历史消息

    if original_messages:
        last_message = original_messages[-1]
        if last_message.get("role") == "user":
            # 如果最后一条消息是用户消息，则将其视为当前用户输入，其余为历史
            raw_last_user_content = last_message.get("content", "")
            historic_messages = original_messages[:-1]
            
            # 从 raw_last_user_content 中提取纯文本用于模板处理
            if isinstance(raw_last_user_content, list):
                # 多模态内容，提取文本部分
                text_parts = [
                    part.get("text", "")
                    for part in raw_last_user_content
                    if isinstance(part, dict) and part.get("type") == "text"
                ]
                text_for_template_processing = " ".join(filter(None, text_parts)).strip()
            elif isinstance(raw_last_user_content, str):
                text_for_template_processing = raw_last_user_content
            else:
                # 其他意外类型，视为空文本
                text_for_template_processing = ""
        else:
            # 如果最后一条不是用户消息，则所有消息都视为历史
            historic_messages = original_messages
            # raw_last_user_content 和 text_for_template_processing 保持空字符串
    
    # 2. 根据提取的纯文本用户输入内容选择合适的模板文件
    selected_template_path = _get_template_path_for_user_input(text_for_template_processing)
    logger.debug(f"根据用户输入选择模板文件: '{selected_template_path}' (提取的文本输入长度: {len(text_for_template_processing.strip()) if text_for_template_processing else 0})")
    
    # 3. 确保选定的模板已加载
    _load_templates(template_path=selected_template_path)
    current_blueprints = _CACHED_PROMPT_BLUEPRINTS.get(selected_template_path, []) # 获取选定模板的缓存
    current_regex_rules = _CACHED_REGEX_RULES.get(selected_template_path, [])     # 获取选定模板的正则规则
    
    # 4. 根据模板注入消息和用户输入
    if not current_blueprints: # 如果没有加载任何模板蓝图
        logger.debug("未加载任何提示词模板，直接使用原始消息（如有）。")
        processed_messages.extend(copy.deepcopy(historic_messages)) # 直接使用历史消息
        # 如果有最后的用户输入 (raw_last_user_content 可能为空字符串或空列表，也可能包含有效内容)
        # 只有当原始消息的最后一条是 user 时，raw_last_user_content 才可能有值
        if original_messages and original_messages[-1].get("role") == "user":
            # 添加包含原始 content (str 或 list) 的用户消息
            processed_messages.append({"role": "user", "content": raw_last_user_content})
        # 特殊情况：如果原始消息只有一条用户消息，且无历史，上面的逻辑已覆盖
        # elif not historic_messages and original_messages and original_messages[-1].get("role") == "user":
            # processed_messages.append(copy.deepcopy(original_messages[-1])) # original_messages[-1] 的 content 也是 raw_last_user_content
        
        # 如果上述处理后 processed_messages 仍为空，但 original_messages 不为空
        # (例如，original_messages 只包含 assistant 消息且无模板)，则直接使用原始消息。
        # 这种情况通常不应该发生，因为期望至少有用户输入或模板。
        if not processed_messages and original_messages:
            logger.debug("无模板且无明确用户输入分离，直接深拷贝原始消息作为基础。")
            processed_messages = copy.deepcopy(original_messages)
    else: # 有模板蓝图的情况
        logger.debug(f"使用 {len(current_blueprints)} 条模板蓝图处理消息。")
        for blueprint_msg_template in current_blueprints:
            blueprint_msg = copy.deepcopy(blueprint_msg_template) # 深拷贝模板项以防修改缓存
            if blueprint_msg.get("type") == "api_input_placeholder":
                # 如果模板项是历史消息占位符，则在此处插入历史消息
                logger.debug(f"在模板中遇到 'api_input_placeholder'，插入 {len(historic_messages)} 条历史消息。")
                processed_messages.extend(copy.deepcopy(historic_messages)) 
            else:
                # 普通模板项，处理 {{user_input}} 占位符
                content_template = blueprint_msg.get("content")
                if isinstance(content_template, str):
                    # 替换模板内容中的 {{user_input}} 为提取的纯文本用户输入
                    temp_content = content_template.replace("{{user_input}}", text_for_template_processing)
                    blueprint_msg["content"] = temp_content
                processed_messages.append(blueprint_msg)
        
        # 检查模板中是否实际包含了 api_input_placeholder
        has_placeholder = any(bp.get("type") == "api_input_placeholder" for bp in current_blueprints)
        # 如果模板中没有历史消息占位符，但原始消息的最后一条是用户消息，
        # 则应将其（包含原始 content）追加到末尾，以确保用户输入不丢失。
        # 之前的逻辑在 "if not current_blueprints:" 分支中已经通过 raw_last_user_content 添加了最后的用户消息，
        # 此处需要确保在有模板但无占位符时，如果最后一条是用户消息，它也被添加。
        # 并且，要避免重复添加。
        # 检查 processed_messages 中是否已经包含了原始的最后一条用户消息（通过比较对象或内容）
        # 一个更简单的方法是，如果最后一条是用户消息，且模板没有占位符，就添加它，
        # 但要确保它不是因为 "if not current_blueprints" 逻辑而被重复添加。
        # 实际上，如果 original_messages[-1] 是 user role，raw_last_user_content 会有值。
        
        # 修正逻辑：如果模板没有占位符，并且原始最后一条消息是用户消息，
        # 并且这条用户消息还没有被加入到 processed_messages 中（例如，模板为空，或者模板不含用户输入占位符）
        # 这种情况通常是：模板只包含系统消息，然后我们期望追加用户的原始输入。
        if not has_placeholder and original_messages and original_messages[-1].get("role") == "user":
            # 检查是否已添加 (避免在无蓝图时重复添加)
            # 如果 processed_messages 为空，或者最后一条不是我们刚提取的 raw_last_user_content
            # (注意：deepcopy 会创建新对象，所以不能直接比较对象引用)
            # 一个简单的检查是，如果 processed_messages 中最后一条用户消息的 content 不是 raw_last_user_content
            
            needs_to_add_raw_user_message = True
            if processed_messages:
                last_added_msg = processed_messages[-1]
                # 检查最后添加的消息是否就是我们想要添加的原始用户消息
                if last_added_msg.get("role") == "user" and last_added_msg.get("content") == raw_last_user_content:
                    needs_to_add_raw_user_message = False
            
            if needs_to_add_raw_user_message:
                logger.debug("模板中无 'api_input_placeholder'，且原始最后一条消息是用户消息，将其（含原始 content）追加。")
                processed_messages.append({"role": "user", "content": raw_last_user_content})
        # elif not has_placeholder and not historic_messages and original_messages and original_messages[-1].get("role") == "user":
             # logger.debug("模板中无 'api_input_placeholder'，原始消息仅一条用户消息，将其追加。")
             # processed_messages.append(copy.deepcopy(original_messages[-1])) # 这部分逻辑被上面的合并了

    # 5. 对所有消息内容应用全局动态变量处理 ({{roll}}, {{random}})
    final_messages_step1: List[Dict[str, Any]] = []
    logger.debug(f"开始对 {len(processed_messages)} 条消息应用动态变量处理。")
    for msg in processed_messages:
        new_msg = msg.copy() # 浅拷贝，因为 content 字段会被完整替换
        content = new_msg.get("content")
        if isinstance(content, str):
            # 先应用动态变量处理
            content_after_dice = _process_dice_rolls(content)
            content_after_random = _process_random_choices(content_after_dice)
            # 注意：正则规则不在此处应用，只在处理 Gemini 响应时应用
            new_msg["content"] = content_after_random
        final_messages_step1.append(new_msg)

    # 6. 移除 content 为空字符串或 None 的消息
    final_messages_step2: List[Dict[str, Any]] = []
    if final_messages_step1:
        original_count = len(final_messages_step1)
        for msg in final_messages_step1:
            # 确保 msg 是字典且 content 存在且非空
            if isinstance(msg, dict) and msg.get("content") is not None and msg.get("content") != "":
                final_messages_step2.append(msg)
        if len(final_messages_step2) < original_count:
            logger.debug(f"移除了 {original_count - len(final_messages_step2)} 条 content 为空或 None 的消息。")
        if not final_messages_step2 and original_count > 0: # 如果所有消息都被移除了
            logger.warning("所有消息因 content 为空或 None 被移除。最终消息列表将为空。")
    
    # 7. 合并相邻的 system 和 user 消息，只与 assistant 消息交替
    if not final_messages_step2:
        merged_messages: List[Dict[str, Any]] = []
        logger.debug("消息列表为空，无需合并。")
    else:
        merged_messages = []
        # 深拷贝第一个消息作为合并起点
        current_message = copy.deepcopy(final_messages_step2[0])
        
        # 如果第一个消息是 system，将其角色改为 user
        if current_message.get("role") == "system":
            current_message["role"] = "user"
        
        for i in range(1, len(final_messages_step2)):
            next_msg = final_messages_step2[i]
            next_role = next_msg.get("role")
            current_role = current_message.get("role")
            
            # 检查是否可以合并：
            # 1. 当前消息是 user 或 system，下一个消息也是 user 或 system
            # 2. 内容都是字符串类型
            can_merge = (
                current_role in ["user", "system"] and 
                next_role in ["user", "system"] and
                isinstance(next_msg.get("content"), str) and 
                isinstance(current_message.get("content"), str)
            )
            
            if can_merge:
                # 合并内容，用换行符分隔
                current_message["content"] += "\n" + next_msg["content"]
                # 确保合并后的消息角色是 user
                current_message["role"] = "user"
            else:
                # 不能合并，将当前已合并的消息加入列表
                merged_messages.append(current_message)
                # 开始新的合并段
                current_message = copy.deepcopy(next_msg)
                # 如果新消息是 system，将其角色改为 user
                if current_message.get("role") == "system":
                    current_message["role"] = "user"
        
        # 添加最后一个处理中的消息段
        merged_messages.append(current_message)
        logger.debug(f"消息合并后，消息数量从 {len(final_messages_step2)} 变为 {len(merged_messages)}。所有相邻的 system 和 user 消息已合并为 user 消息。")

    # --- 开始修正：确保原始多模态用户输入被保留 ---
    # raw_last_user_content 持有原始最后一条用户消息的 content (str 或 list)
    # original_messages 是最原始的输入消息列表
    if original_messages:
        original_last_message = original_messages[-1]
        # 检查原始最后一条消息是否为用户角色，并且其原始 content (raw_last_user_content) 是一个列表
        if original_last_message.get("role") == "user" and isinstance(raw_last_user_content, list):
            logger.debug(f"修正步骤：原始最后一条用户消息是多模态列表。尝试在 merged_messages 中恢复它。raw_last_user_content: {str(raw_last_user_content)[:200]}")
            
            found_and_updated = False
            # 从后向前查找 merged_messages 中的最后一条 user 消息并更新其 content
            # 这是因为经过模板处理和合并后，原始的用户输入（如果存在）通常会体现为最终消息列表中的最后一条用户消息
            if merged_messages: # 确保 merged_messages 不是空的
                for i in range(len(merged_messages) - 1, -1, -1):
                    if merged_messages[i].get("role") == "user":
                        logger.info(f"修正步骤：找到 merged_messages 中的最后一条用户消息 (索引 {i})，将其 content 更新为原始多模态列表。")
                        merged_messages[i]["content"] = raw_last_user_content # 使用原始的多模态列表
                        found_and_updated = True
                        break
            
            if not found_and_updated:
                # 如果 merged_messages 为空，或者在其中没有找到 user 角色的消息来更新（理论上不太可能在正常流程后发生）
                # 则直接将原始的多模态用户消息追加到 merged_messages。
                # 这种情况可能发生在例如：模板完全为空，且历史消息也为空或不含用户消息。
                logger.debug("修正步骤：merged_messages 中未找到用户消息来更新，或者列表为空。将追加原始多模态用户消息。")
                merged_messages.append({"role": "user", "content": raw_last_user_content})
            
            # 调试日志：检查修正后的状态
            if merged_messages and merged_messages[-1].get("role") == "user":
                 logger.debug(f"修正后，merged_messages 最后一条用户消息 content 类型: {type(merged_messages[-1].get('content'))}, content: {str(merged_messages[-1].get('content'))[:200]}")
        elif original_last_message.get("role") == "user": # 原始是用户消息，但 content 不是 list (即是 str)
             logger.debug(f"修正步骤：原始最后一条用户消息是文本（非多模态列表），无需特殊恢复。raw_last_user_content 类型: {type(raw_last_user_content)}")
        else: # 原始最后一条消息不是 user
            logger.debug("修正步骤：原始最后一条消息非用户角色，无需特殊恢复多模态 content。")
    elif not original_messages:
        logger.debug("修正步骤：original_messages 为空，无需进行多模态 content 恢复。")
    # --- 修正结束 ---

    # 准备最终返回结果
    model_name = original_body.get("model")
    # 清理模型名称，去除可能的换行符、回车符和前后空格
    if isinstance(model_name, str):
        model_name = model_name.strip().replace('\r', '').replace('\n', '')
        if not model_name:  # 如果清理后为空
            model_name = None
            logger.warning("模型名称在清理后为空字符串。")
    
    result = {
        "model": model_name,
        "messages": merged_messages,
        "selected_regex_rules": current_regex_rules
    }
    
    if result.get("model") is None:
        logger.warning("请求中 model 参数为 None 或未提供。")
    if not merged_messages:
        logger.info("预处理后最终的 messages 列表为空。") # 使用 info 级别，因为这可能是重要情况
        
    return result