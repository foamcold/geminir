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
    # processed_messages: List[Dict[str, Any]] = [] # 将在模板处理循环中初始化和构建
    raw_last_user_content: Any = ""               # 存储最后一个用户的原始 content (可以是 str 或 list)
    text_for_template_processing: str = ""        # 存储从最后一个用户 content 中提取的纯文本，用于模板逻辑和选择
    historic_messages: List[Dict[str, Any]] = []  # 存储除最后一个用户输入外的历史消息
    is_multimodal_input: bool = False             # 标记最后一个用户输入是否为多模态列表

    if original_messages:
        last_message = original_messages[-1]
        if last_message.get("role") == "user":
            raw_last_user_content = last_message.get("content", "")
            historic_messages = original_messages[:-1]

            if isinstance(raw_last_user_content, list):
                is_multimodal_input = True
                # 从多模态内容中提取所有文本部分的拼接，用于模板选择和非图片注入时的 {{user_input}} 替换
                text_parts = [
                    part.get("text", "")
                    for part in raw_last_user_content
                    if isinstance(part, dict) and part.get("type") == "text"
                ]
                text_for_template_processing = " ".join(filter(None, text_parts)).strip()
                logger.debug(f"原始末尾用户输入是多模态列表。提取的文本部分 (text_for_template_processing): '{text_for_template_processing[:200]}...'")
            elif isinstance(raw_last_user_content, str):
                text_for_template_processing = raw_last_user_content
                # is_multimodal_input 保持 False
            else:
                logger.warning(f"原始末尾用户输入 content 类型未知: {type(raw_last_user_content)}，将视为空文本。")
                text_for_template_processing = ""
                # is_multimodal_input 保持 False
        else:
            # 最后一条消息不是 user，所有消息视为历史
            historic_messages = original_messages
            # raw_last_user_content, text_for_template_processing 保持空, is_multimodal_input 保持 False
    
    # 2. 根据提取的纯文本用户输入内容选择合适的模板文件
    # 模板选择仍然基于 text_for_template_processing
    selected_template_path = _get_template_path_for_user_input(text_for_template_processing)
    logger.debug(f"根据用户输入选择模板文件: '{selected_template_path}' (提取的文本输入长度: {len(text_for_template_processing.strip()) if text_for_template_processing else 0})")
    
    # 3. 确保选定的模板已加载
    _load_templates(template_path=selected_template_path)
    current_blueprints = _CACHED_PROMPT_BLUEPRINTS.get(selected_template_path, []) # 获取选定模板的缓存
    current_regex_rules = _CACHED_REGEX_RULES.get(selected_template_path, [])     # 获取选定模板的正则规则
    
    # 4. 根据模板注入消息和用户输入 (重构此部分)
    processed_messages: List[Dict[str, Any]] = [] # 初始化用于构建处理后消息的列表

    if not current_blueprints:
        logger.debug("未加载任何提示词模板。将直接使用历史消息和最后的用户输入（如果存在）。")
        processed_messages.extend(copy.deepcopy(historic_messages))
        if original_messages and original_messages[-1].get("role") == "user":
            # 如果原始最后一条消息是用户消息，则添加它（其 content 可能是 str 或 list）
            processed_messages.append({"role": "user", "content": raw_last_user_content})
        elif not processed_messages and original_messages: # 无模板，但有非用户角色的原始消息
             processed_messages = copy.deepcopy(original_messages)

    else: # 有模板蓝图的情况
        logger.debug(f"使用 {len(current_blueprints)} 条模板蓝图处理消息。")
        for blueprint_msg_template in current_blueprints:
            blueprint_msg = copy.deepcopy(blueprint_msg_template)
            
            if blueprint_msg.get("type") == "api_input_placeholder":
                logger.debug(f"在模板中遇到 'api_input_placeholder'，插入 {len(historic_messages)} 条历史消息。")
                processed_messages.extend(copy.deepcopy(historic_messages))
                continue

            # 处理普通模板项
            content_template_str = blueprint_msg.get("content")
            if not isinstance(content_template_str, str):
                logger.warning(f"模板消息的 content 不是字符串 (类型: {type(content_template_str).__name__})，将按原样保留并添加。消息: {blueprint_msg}")
                processed_messages.append(blueprint_msg)
                continue

            # A. 先对模板字符串内容应用动态变量处理 ({{roll}}, {{random}})
            content_after_vars = _process_random_choices(_process_dice_rolls(content_template_str))
            logger.debug(f"模板内容应用动态变量后: '{content_after_vars[:200]}...'")

            # B. 处理 {{user_input}} 和多模态注入
            if "{{user_input}}" in content_after_vars:
                if is_multimodal_input and raw_last_user_content: # 用户输入了多模态内容 (raw_last_user_content 是列表)
                    logger.debug(f"模板含 '{{user_input}}' 且用户输入为多模态。原始角色: '{blueprint_msg.get('role')}'")
                    parts = content_after_vars.split("{{user_input}}", 1)
                    prefix_text = parts[0]
                    suffix_text = parts[1] if len(parts) > 1 else ""
                    
                    new_content_list: List[Dict[str, Any]] = []
                    if prefix_text:
                        new_content_list.append({"type": "text", "text": prefix_text})
                    
                    # raw_last_user_content 包含了用户提供的所有 parts (text 和 image_url)
                    if isinstance(raw_last_user_content, list): # 再次确认，以防万一
                        new_content_list.extend(raw_last_user_content)
                    else: # 不应该发生，但作为保险
                        logger.error(f"is_multimodal_input 为 True 但 raw_last_user_content 不是列表: {type(raw_last_user_content)}")
                        if raw_last_user_content: # 如果仍有内容，尝试作为文本添加
                             new_content_list.append({"type": "text", "text": str(raw_last_user_content)})
                    
                    if suffix_text:
                        new_content_list.append({"type": "text", "text": suffix_text})
                    
                    if new_content_list: # 只有当列表非空时才更新 content
                        blueprint_msg["content"] = new_content_list
                        logger.debug(f"多模态注入完成。消息 content 变为列表，包含 {len(new_content_list)} 个 part(s)。")
                    else: # 如果处理后列表为空（例如，模板占位符前后无文本，用户输入也为空列表）
                        blueprint_msg["content"] = "" # 设为空字符串，可能后续被移除
                        logger.debug("多模态注入后 new_content_list 为空，消息 content 设为空字符串。")
                    # 角色按用户指示不在此处强制修改。
                
                else: # 模板含 '{{user_input}}' 但用户输入是纯文本 (或无输入)
                    logger.debug(f"模板含 '{{user_input}}'，用户输入为纯文本。替换为: '{text_for_template_processing[:100]}...'")
                    blueprint_msg["content"] = content_after_vars.replace("{{user_input}}", text_for_template_processing)
            
            else: # 模板不含 '{{user_input}}'
                logger.debug("模板不含 '{{user_input}}'。用户输入（包括图片）将被忽略（如果未通过其他方式如api_input_placeholder处理）。")
                blueprint_msg["content"] = content_after_vars
            
            processed_messages.append(blueprint_msg)

        # 在有模板的情况下，如果模板没有api_input_placeholder，并且原始最后一条消息是用户消息，
        # 且这条用户消息没有通过{{user_input}}被处理（例如模板就没有{{user_input}}），
        # 那么这条原始用户输入（raw_last_user_content）可能会丢失。
        # 此处需要确保，如果用户有最后输入 (raw_last_user_content 非空)，
        # 并且它没有被任何 {{user_input}} 消耗掉，它应该被追加。
        # 一个简单的判断是：如果 is_multimodal_input 为 True，或者 text_for_template_processing 非空，
        # 并且没有任何一个模板消息的 content 变成了列表（对于多模态）或者等于 text_for_template_processing（对于纯文本），
        # 这可能意味着用户输入未被使用。但这个判断比较复杂。

        # 简化处理：如果模板处理完后，原始的 raw_last_user_content （如果是用户角色发出的）没有体现在
        # processed_messages 的最后一条用户消息中，则追加它。
        # 这与旧的 "if not has_placeholder..." 逻辑类似，但需要更精确。
        # 目前的逻辑是，如果模板没有 {{user_input}}，图片（和附带文本）会被忽略。这是用户确认的。
        # 如果模板有 {{user_input}}，图片和文本会被注入。
        # 所以，不需要额外的追加逻辑了，因为用户输入要么被注入，要么按设计被忽略。

    # 5. 步骤名称不变，但内容已在模板处理循环中完成 (动态变量处理)
    final_messages_step1 = processed_messages # 重命名以保持后续步骤变量名一致
    logger.debug(f"模板和用户输入处理完成，进入空消息移除前有 {len(final_messages_step1)} 条消息。")

    # 6. 移除 content 为空或无效的消息
    final_messages_step2: List[Dict[str, Any]] = []
    if final_messages_step1:
        original_count = len(final_messages_step1)
        for msg in final_messages_step1:
            content = msg.get("content")
            is_valid_content = False
            if isinstance(content, str):
                if content.strip(): # 非空字符串
                    is_valid_content = True
            elif isinstance(content, list):
                if content: # 列表非空
                    # 进一步检查列表中的 part 是否都有效可能过于复杂，暂时只要列表非空就认为有效
                    # 或者可以检查是否有任何一个 part 是有效的
                    is_valid_content = any(
                        (isinstance(p, dict) and p.get("type") == "text" and p.get("text","").strip()) or \
                        (isinstance(p, dict) and p.get("type") == "image_url" and p.get("image_url",{}).get("url","").strip())
                        for p in content
                    )
            
            if is_valid_content:
                final_messages_step2.append(msg)
            else:
                logger.debug(f"移除无效 content 的消息: role='{msg.get('role')}', content='{str(content)[:100]}...'")

        if len(final_messages_step2) < original_count:
            logger.debug(f"移除了 {original_count - len(final_messages_step2)} 条 content 为空或无效的消息。")
        if not final_messages_step2 and original_count > 0:
            logger.warning("所有消息因 content 为空或无效被移除。最终消息列表将为空。")
    
    # 7. 合并相邻的 system 和 user 消息
    if not final_messages_step2:
        merged_messages: List[Dict[str, Any]] = []
        logger.debug("消息列表为空（移除空消息后），无需合并。")
    else:
        merged_messages = []
        current_message_to_merge = copy.deepcopy(final_messages_step2[0])

        # 第一个消息如果是 system，角色转为 user (Gemini 要求 user/model 交替)
        if current_message_to_merge.get("role") == "system":
            logger.debug(f"合并前：首条消息角色从 'system' 转为 'user'。Content: {str(current_message_to_merge.get('content'))[:100]}")
            current_message_to_merge["role"] = "user"
        
        for i in range(1, len(final_messages_step2)):
            next_msg_to_process = final_messages_step2[i]
            
            # 检查是否可以合并当前 current_message_to_merge 和 next_msg_to_process
            can_merge_texts = (
                current_message_to_merge.get("role") in ["user", "system"] and # 当前实际上已确保是 user
                next_msg_to_process.get("role") in ["user", "system"] and
                isinstance(current_message_to_merge.get("content"), str) and
                isinstance(next_msg_to_process.get("content"), str)
            )
            
            if can_merge_texts:
                logger.debug(f"合并消息：将 next_msg (role='{next_msg_to_process.get('role')}', content='{str(next_msg_to_process.get('content'))[:100]}...') 合并到 current_message (role='{current_message_to_merge.get('role')}', content='{str(current_message_to_merge.get('content'))[:100]}...')")
                current_message_to_merge["content"] += "\n" + next_msg_to_process.get("content", "")
                current_message_to_merge["role"] = "user" # 合并后确保是 user
            else:
                # 不能合并，将已累积的 current_message_to_merge 添加到结果列表
                merged_messages.append(current_message_to_merge)
                # next_msg 成为新的累积消息起点
                current_message_to_merge = copy.deepcopy(next_msg_to_process)
                # 如果新的起点是 system 消息，也将其角色转为 user
                if current_message_to_merge.get("role") == "system":
                    logger.debug(f"合并中：新段起始消息角色从 'system' 转为 'user'。Content: {str(current_message_to_merge.get('content'))[:100]}")
                    current_message_to_merge["role"] = "user"
        
        # 添加最后一个累积的消息段
        merged_messages.append(current_message_to_merge)
        logger.debug(f"消息合并后，消息数量从 {len(final_messages_step2)} 变为 {len(merged_messages)}。")

    # --- 删除旧的“修正步骤” ---
    # (原 src/template_handler.py:577-616 的内容将被下面的空行取代或直接删除)

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