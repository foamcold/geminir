# src/streaming_utils.py
"""
此模块提供了用于模拟服务器发送事件 (SSE) 流式响应的工具。

主要功能是 `fake_stream_generator_from_non_stream` 异步生成器，
它能够接收一个执行非流式操作的异步任务，并在等待该任务完成期间，
定期发送心跳信号以保持连接活跃。一旦任务完成，它会将其结果
（通常是 OpenAI Chat Completion 格式的字典）转换为一系列模拟的 SSE 数据块 (deltas)
并发送给客户端，最后以 `[DONE]` 标记结束流。
"""
import asyncio
import json
import logging
import time
from typing import Dict, Any, AsyncGenerator, Optional, List # BaseException 已从此行移除，因为它是内置类型

from fastapi import HTTPException

from .config import settings
from .template_handler import _apply_regex_rules_to_content # 用于对最终内容应用正则规则

# 获取基于应用配置的日志记录器
logger = logging.getLogger(settings.app_name)

async def fake_stream_generator_from_non_stream(
    non_stream_task: asyncio.Task,
    original_body: Dict[str, Any],
    regex_rules: Optional[List[Dict[str, Any]]] = None
) -> AsyncGenerator[str, None]:
    """
    从一个非流式异步任务模拟生成 SSE (Server-Sent Events) 流。

    此异步生成器会执行以下操作：
    1. 启动一个内部的后台任务 (`_execute_and_signal_completion`) 来执行传入的 `non_stream_task`。
       `non_stream_task` 应该返回一个类似 OpenAI Chat Completion 格式的字典。
    2. 在等待 `non_stream_task` 完成期间，定期（由 `settings.proxy.fake_streaming.heartbeat_interval` 控制）
       发送 SSE 心跳消息，以防止客户端连接超时。心跳消息是一个空的 content delta。
    3. 当 `non_stream_task` 完成后：
        a. 如果任务成功完成：
            i.  获取其结果（一个字典）。
            ii. (可选) 对结果中 "assistant" 角色的消息内容应用 `_apply_regex_rules_to_content`。
            iii.将完整的响应内容（角色、内容、结束原因）拆分为多个模拟的 SSE 数据块 (delta chunks)，
                并逐个 `yield`。
        b. 如果任务执行过程中发生异常：
            i.  捕获异常。
            ii. 将异常信息格式化为一个 SSE 错误事件并 `yield`。
        c. 如果任务被取消：
            i.  记录取消事件。
            ii. 发送一个表示请求被取消的 SSE 错误事件。
    4. 无论成功、失败还是取消，最后都会 `yield` 一个 `data: [DONE]\n\n` 消息，以符合 OpenAI SSE 流规范。
    5. 妥善处理 `asyncio.CancelledError`，确保在生成器被取消时，其内部的后台任务也能被取消。

    参数:
        non_stream_task (asyncio.Task): 一个已经创建的 `asyncio.Task` 对象，
                                        该任务负责执行实际的非流式请求并返回结果。
                                        此任务的结果预期是一个字典，通常是 OpenAI Chat Completion 格式。
        original_body (Dict[str, Any]): 原始的 OpenAI 请求体，用于在生成心跳或模拟块时
                                        提取如 `model` 名称等信息。
        regex_rules (Optional[List[Dict[str, Any]]]): 可选的正则表达式规则列表，用于处理助手消息内容。

    Yields:
        str: 符合 SSE 格式的字符串消息。例如:
             `data: {"id": "...", "choices": [{"delta": {"content": "Hello"}}]}\n\n`
             `data: [DONE]\n\n`
    """
    heartbeat_interval: float = settings.proxy.fake_streaming.heartbeat_interval
    task_completion_event = asyncio.Event() # 用于通知 non_stream_task 已完成的事件
    _wrapped_task_result: Any = None # 存储 non_stream_task 的成功结果
    _wrapped_task_exception: Optional[BaseException] = None # 存储 non_stream_task 的异常 (BaseException 是内置类型)

    async def _execute_and_signal_completion():
        """
        内部辅助函数：执行被包装的 `non_stream_task` 并设置完成事件。
        此函数在单独的 `asyncio.Task` 中运行。
        """
        nonlocal _wrapped_task_result, _wrapped_task_exception
        try:
            # 使用 asyncio.shield 保护 non_stream_task，使其不被外部直接取消，
            # 除非本生成器 fake_stream_generator_from_non_stream 本身被取消。
            _wrapped_task_result = await asyncio.shield(non_stream_task)
            logger.debug(f"非流式任务在 _execute_and_signal_completion 中成功完成。")
        except asyncio.CancelledError as e:
            # 如果 non_stream_task (被 shield 保护的) 内部自行处理了 CancelledError 并重新抛出，
            # 或者 shield 被取消（通常是因为 fake_stream_generator_from_non_stream 被取消），
            # 则会捕获到此异常。
            _wrapped_task_exception = e
            logger.debug(f"非流式任务在 _execute_and_signal_completion 中被取消: {e}")
        except BaseException as e: # 捕获所有其他类型的异常 (BaseException 是内置类型)
            _wrapped_task_exception = e
            logger.debug(f"非流式任务在 _execute_and_signal_completion 中遇到异常: {e}", exc_info=settings.debug_mode)
        finally:
            task_completion_event.set() # 无论成功、失败或取消，都设置事件，通知主循环
            logger.debug("_execute_and_signal_completion 完成，事件已设置。")

    # 创建并启动后台任务，用于执行非流式请求并设置完成事件
    background_executor_task = asyncio.create_task(_execute_and_signal_completion())
    logger.debug(f"后台任务 _execute_and_signal_completion 已创建: {background_executor_task.get_name()}")

    try:
        # 心跳循环：在后台任务完成前，定期发送心跳
        while not task_completion_event.is_set():
            try:
                # 等待后台任务完成事件，超时时间为心跳间隔
                await asyncio.wait_for(task_completion_event.wait(), timeout=heartbeat_interval)
                logger.debug("task_completion_event 已设置，退出心跳循环。")
                break # 事件已设置，表示后台任务已完成或出错，跳出心跳循环
            except asyncio.TimeoutError:
                # 等待超时，表示后台任务仍在运行，发送心跳信号
                current_ts = int(time.time())
                # 从原始请求体中获取模型名称，若无则使用默认值
                model_name = original_body.get("model", "gpt-3.5-turbo-proxy-heartbeat")
                heartbeat_chunk = {
                    "id": f"chatcmpl-fake-hb-{current_ts}",
                    "object": "chat.completion.chunk",
                    "created": current_ts,
                    "model": model_name,
                    "choices": [{"index": 0, "delta": {"content": ""}, "finish_reason": None}] # 心跳是空内容块
                }
                heartbeat_message = f"data: {json.dumps(heartbeat_chunk)}\n\n"
                logger.debug(f"发送模拟流心跳。")
                yield heartbeat_message
            except asyncio.CancelledError:
                # 如果在心跳循环中本生成器被取消
                logger.info("fake_stream_generator_from_non_stream 在心跳循环期间被取消，正在清理...")
                if not background_executor_task.done():
                    background_executor_task.cancel() # 取消后台执行器任务
                    try:
                        await background_executor_task # 等待后台任务实际取消完成
                    except asyncio.CancelledError:
                        logger.debug("后台执行器任务在生成器取消期间成功取消。")
                    except Exception as e_bg_cancel:
                        logger.error(f"等待后台执行器任务取消时发生错误: {e_bg_cancel}", exc_info=settings.debug_mode)
                raise # 重新抛出 CancelledError，由上层处理

        # 后台任务已完成 (可能成功，也可能带有异常)
        if _wrapped_task_exception:
            # 如果后台任务执行时捕获到异常
            logger.error(f"从非流式任务执行中捕获到异常: {_wrapped_task_exception}", exc_info=settings.debug_mode)
            if isinstance(_wrapped_task_exception, HTTPException):
                # 如果是 FastAPI 的 HTTPException，提取其状态码和详情
                error_payload = {"error": {"message": str(_wrapped_task_exception.detail), "code": _wrapped_task_exception.status_code, "type": "api_error_in_fake_stream"}}
            elif isinstance(_wrapped_task_exception, asyncio.CancelledError):
                # 如果任务是被取消的
                 error_payload = {"error": {"message": "请求在服务器端被取消 (模拟流)。", "type": "server_request_cancelled_in_fake_stream"}}
            else:
                # 其他类型的异常
                error_payload = {"error": {"message": f"非流式任务执行期间发生内部错误: {str(_wrapped_task_exception)}", "type": "internal_task_error_in_fake_stream"}}
            error_message = f"data: {json.dumps(error_payload)}\n\n"
            logger.debug(f"发送模拟流错误数据 (来自 _wrapped_task_exception)。")
            yield error_message
        elif _wrapped_task_result is not None:
            # 如果后台任务成功完成并返回结果
            full_response_data: Dict[str, Any] = _wrapped_task_result
            logger.debug(f"已收到来自非流式任务的原始完整响应。")

            # (可选) 对助手消息内容应用正则表达式规则
            if isinstance(full_response_data, dict) and "choices" in full_response_data and full_response_data["choices"]:
                if full_response_data["choices"][0].get("message", {}).get("role") == "assistant":
                    original_content = full_response_data["choices"][0].get("message", {}).get("content", "")
                    if isinstance(original_content, str):
                        # 临时解决方案：传递空的正则规则列表，因为在响应处理阶段我们无法确定使用的模板
                        processed_content = _apply_regex_rules_to_content(original_content, regex_rules)
                        if processed_content != original_content:
                            logger.info("助手消息内容已通过正则表达式规则处理。")
                            full_response_data["choices"][0]["message"]["content"] = processed_content
                        # else:
                            # logger.debug("正则表达式规则未更改助手消息内容。")

            # 从（可能已处理的）非流式响应中模拟流式数据块
            if isinstance(full_response_data, dict) and "choices" in full_response_data and full_response_data["choices"]:
                choice = full_response_data["choices"][0]
                message = choice.get("message", {})

                resp_id = full_response_data.get("id", f"chatcmpl-simulated-{int(time.time())}")
                resp_model = full_response_data.get("model", original_body.get("model", "gpt-3.5-turbo-proxy"))
                resp_created = full_response_data.get("created", int(time.time()))

                # 1. 发送角色信息块 (如果存在)
                role = message.get("role")
                if role:
                    role_chunk = {
                        "id": resp_id, "object": "chat.completion.chunk", "created": resp_created, "model": resp_model,
                        "choices": [{"index": 0, "delta": {"role": role}, "finish_reason": None}]
                    }
                    yield f"data: {json.dumps(role_chunk)}\n\n"

                # 2. 发送内容块 (如果存在，即使是空字符串也发送)
                content = message.get("content")
                if content is not None:
                    content_chunk = {
                        "id": resp_id, "object": "chat.completion.chunk", "created": resp_created, "model": resp_model,
                        "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}]
                    }
                    yield f"data: {json.dumps(content_chunk)}\n\n"

                # 3. 发送结束原因块 (如果存在)
                finish_reason = choice.get("finish_reason")
                if finish_reason:
                    finish_chunk = {
                        "id": resp_id, "object": "chat.completion.chunk", "created": resp_created, "model": resp_model,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}] # 结束块的 delta 为空
                    }
                    yield f"data: {json.dumps(finish_chunk)}\n\n"
            else:
                # 如果响应结构不符合预期的 Chat Completion 格式，记录警告并尝试发送原始数据
                reason = ""
                if not isinstance(full_response_data, dict):
                    reason = f"因为它不是一个字典 (实际类型: {type(full_response_data).__name__})"
                elif "choices" not in full_response_data:
                    reason = "因为它缺少 'choices' 键"
                elif not full_response_data["choices"]:
                    reason = "因为 'choices' 键对应一个空列表"
                else:
                    reason = "因为 'choices' 列表的第一个元素结构不符合预期"

                logger.warning(
                    f"收到的 full_response_data 与预期的 chat.completion 结构不匹配 {reason}。"
                    f"正在尝试发送原始数据。"
                )
                yield f"data: {json.dumps(full_response_data)}\n\n" # 作为最后的尝试，发送原始数据
        else:
            # 这种极端情况理论上不应发生，因为 _execute_and_signal_completion 总会设置结果或异常
            logger.error("非流式任务已完成，但既未产生结果也未产生异常。这是一个意外状态。")
            fallback_error_payload = {"error": {"message": "非流式任务以未知状态完成。", "type": "unknown_task_completion_state"}}
            yield f"data: {json.dumps(fallback_error_payload)}\n\n"

    except asyncio.CancelledError:
        # 如果 fake_stream_generator_from_non_stream 的主执行体被取消
        logger.info("fake_stream_generator_from_non_stream 主体被取消。")
        if not background_executor_task.done():
            logger.debug("由于主生成器取消，尝试取消后台执行器任务...")
            background_executor_task.cancel()
            try:
                await background_executor_task # 等待后台任务实际取消
            except asyncio.CancelledError:
                logger.debug("后台执行器任务在外部取消期间成功取消。")
            except Exception as e_bg_cancel_outer:
                logger.error(f"在外部取消期间等待后台执行器任务取消时发生错误: {e_bg_cancel_outer}", exc_info=settings.debug_mode)
        raise # 重新抛出 CancelledError
    except Exception as e_outer:
        # 捕获生成器主逻辑中的任何其他意外错误
        logger.error(f"fake_stream_generator_from_non_stream 中发生意外错误: {e_outer}", exc_info=True)
        error_payload = {"error": {"message": f"模拟流生成器发生意外错误: {str(e_outer)}", "type": "fake_stream_generator_unexpected_error"}}
        yield f"data: {json.dumps(error_payload)}\n\n"
    finally:
        # 清理阶段：确保后台任务在生成器退出前被妥善处理
        # 'background_executor_task' in locals() 确保变量已定义
        if 'background_executor_task' in locals() and not background_executor_task.done():
            logger.debug("fake_stream_generator_from_non_stream 进入 finally 块，后台任务未完成，尝试取消。")
            background_executor_task.cancel()
            try:
                await background_executor_task # 等待取消操作完成
            except asyncio.CancelledError:
                logger.debug("后台任务在 finally 块中成功取消。")
            except Exception as e_final_cancel:
                logger.error(f"在 finally 块中取消后台任务时发生错误: {e_final_cancel}", exc_info=settings.debug_mode)

        # 无论如何，最后都发送 [DONE] 标记
        done_message = "data: [DONE]\n\n"
        logger.debug(f"发送模拟流 [DONE] 标记。")
        yield done_message