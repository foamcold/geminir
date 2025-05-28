# src/main.py
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
import asyncio
import logging # 添加日志记录

from .config import settings
from .openai_proxy import (
    get_gemini_non_stream_response_and_convert_to_openai, # Updated function name
    fake_stream_generator_from_non_stream
    # stream_openai_response_to_client is no longer used
    # _apply_regex_rules_to_content is no longer directly used here
)

# 配置日志
logging.basicConfig(
    level=settings.log_level.upper(),
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(settings.app_name)


app = FastAPI(title=settings.app_name)

@app.on_event("startup")
async def startup_event():
    logger.info(f"应用 '{settings.app_name}' 已启动。")
    logger.info(f"日志级别: {settings.log_level.upper()}")
    logger.info(f"提示词模板路径: {settings.proxy.prompt_template_path}")
    logger.info(f"假流式启用配置: {settings.proxy.fake_streaming.enabled}")
    if settings.proxy.fake_streaming.enabled:
        logger.info(f"假流式心跳间隔: {settings.proxy.fake_streaming.heartbeat_interval}s")
    logger.info(f"OpenAI 请求超时: {settings.proxy.openai_request_timeout}s")


@app.post("/v1/chat/completions") # Fixed endpoint path
async def chat_completions_endpoint(request: Request):
    try:
        original_body = await request.json()
        logger.debug(f"收到请求: {request.method} {request.url}")
        logger.debug(f"请求体: {original_body}")
    except Exception as e:
        logger.error(f"解析请求体失败: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"无效的 JSON 请求体: {e}")

    auth_header = request.headers.get("Authorization")
    # API key presence is checked within get_gemini_non_stream_response_and_convert_to_openai

    client_requests_stream = original_body.get("stream", False)
    logger.info(f"客户端请求流式: {client_requests_stream}")

    try:
        if not client_requests_stream:
            logger.info("处理非流式请求 (Gemini Adapter)")
            # The get_gemini_non_stream_response_and_convert_to_openai function
            # now handles the entire conversion and API call process.
            # Regex rules are applied within convert_gemini_response_to_openai_chat_completion.
            response_data = await get_gemini_non_stream_response_and_convert_to_openai(
                original_body, auth_header
            )
            logger.debug(f"准备返回给客户端的最终非流式响应数据 (来自 Gemini Adapter): {response_data}")
            return JSONResponse(content=response_data)
        else:
            # Client requests stream, use fake streaming as backend is non-streaming
            logger.info("处理流式请求 (Gemini Adapter with Fake Streaming)")
            
            # Create a task for the non-streaming Gemini call and conversion
            gemini_task = asyncio.create_task(
                get_gemini_non_stream_response_and_convert_to_openai(original_body, auth_header)
            )
            
            # Use the fake stream generator
            # fake_stream_generator_from_non_stream will handle applying regex rules
            # to the full response before chunking it.
            return StreamingResponse(
                fake_stream_generator_from_non_stream(gemini_task, original_body),
                media_type="text/event-stream"
            )
    except HTTPException as e: # 重新抛出已知的HTTP异常
        logger.error(f"HTTPException: {e.status_code} - {e.detail}")
        raise e
    except Exception as e: # 捕获其他所有意外错误
        logger.error(f"处理请求时发生未知错误: {e}", exc_info=True) # exc_info=True 会记录堆栈跟踪
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    # 这个仅用于本地开发测试，生产环境应使用 Gunicorn + Uvicorn workers
    # uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True, workers=1)
    # 为了能直接运行，可以将 reload=True 改为 reload_dirs=["src"] 如果 uvicorn 支持
    # 或者直接 uvicorn.run(app, ...)
    uvicorn.run(app, host="0.0.0.0", port=8000) # 简化运行命令