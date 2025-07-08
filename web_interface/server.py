"""
FastAPI Web Server for SimpleAgent
符合OpenAI API规范的Web服务器实现
支持多Agent架构，通过model name选择不同的Agent
"""

import json
import time
import uuid
from datetime import datetime
from typing import Dict, Any, AsyncGenerator, Optional
import uvicorn
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from agent import (
    SimpleAgent,
    get_agent_registry,
    register_agent,
    list_available_models
)
from tools import (
    execute_command,
    file_operations,
    sketch_pad_operations,
)

from .models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChunk,
    ChatMessage,
    ChatChoice,
    ChatCompletionChunkChoice,
    DeltaMessage,
    Usage,
    ModelInfo,
    ModelListResponse,
    ErrorResponse,
    ErrorDetail,
    HealthResponse,
    ServerInfoResponse
)


# 全局Agent注册器
agent_registry = None


def initialize_default_agents():
    """初始化默认的Agent"""
    global agent_registry
    agent_registry = get_agent_registry()
    
    # 创建默认工具集
    toolkit = [
        execute_command,
        file_operations,
        sketch_pad_operations,
    ]
    
    # 注册SimpleAgent
    register_agent("simple-agent-v1", SimpleAgent)
    
    # 创建默认Agent实例
    agent_registry.get_or_create_agent(
        "simple-agent-v1",
        name="SimpleAgent Web Service",
        description="Professional CAD modeling assistant with web API",
        toolkit=toolkit,
        context_file="history/conversation_history.json",
        max_history_length=20
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时初始化智能体
    print("🚀 Initializing SimpleAgent Web Server...")
    try:
        initialize_default_agents()
        print("✅ SimpleAgent initialized successfully!")
        
    except Exception as e:
        print(f"❌ Failed to initialize SimpleAgent: {e}")
        raise
    
    yield
    
    # 关闭时清理资源
    print("🔄 Shutting down SimpleAgent Web Server...")


# 创建FastAPI应用
app = FastAPI(
    title="SimpleAgent API",
    description="OpenAI-compatible API for SimpleAgent - A universal agent framework",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def estimate_tokens(text: str) -> int:
    """简单的Token估算（1 token ≈ 4 字符）"""
    return max(1, len(text) // 4)


def get_agent_for_model(model_name: str, agent_registry):
    """
    根据模型名称获取Agent实例（确保单例）
    
    Args:
        model_name: 模型名称
        agent_registry: Agent注册器实例
        
    Returns:
        Agent实例
        
    Raises:
        HTTPException: 如果模型不存在或创建失败
    """
    if not agent_registry:
        raise HTTPException(status_code=500, detail="Agent registry not initialized")
    
    # 首先尝试获取已存在的Agent实例
    agent = agent_registry.get_agent(model_name)
    if agent:
        return agent
    
    # 如果不存在，尝试创建新的Agent实例
    try:
        agent = agent_registry.get_or_create_agent(
            model_name,
            name=f"Agent for {model_name}",
            description=f"Agent instance for model {model_name}",
            toolkit=[execute_command, file_operations, sketch_pad_operations],
        )
        return agent
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model_name}")


def create_error_response(message: str, error_type: str = "invalid_request", 
                         param: Optional[str] = None, code: Optional[str] = None,
                         status_code: int = 400) -> JSONResponse:
    """创建标准错误响应"""
    error_detail = ErrorDetail(
        message=message,
        type=error_type,
        param=param,
        code=code
    )
    error_response = ErrorResponse(error=error_detail)
    return JSONResponse(
        status_code=status_code,
        content=error_response.model_dump()
    )


async def stream_chat_completion(request: ChatCompletionRequest, request_id: str) -> AsyncGenerator[str, None]:
    """流式聊天完成生成器"""
    global agent_registry
    
    # 使用统一的Agent获取逻辑
    agent = get_agent_for_model(request.model, agent_registry)
    
    # 获取用户最后一条消息
    user_messages = [msg for msg in request.messages if msg.role == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="No user message found")
    
    last_user_message = user_messages[-1]
    query = last_user_message.content or ""
    
    created_time = int(time.time())
    
    # 发送开始块
    start_chunk = ChatCompletionChunk(
        id=request_id,
        object="chat.completion.chunk",
        created=created_time,
        model=request.model,
        choices=[
            ChatCompletionChunkChoice(
                index=0,
                delta=DeltaMessage(role="assistant", content=None, tool_calls=None),
                finish_reason=None
            )
        ],
        usage=None,
        system_fingerprint=None
    )
    yield f"data: {start_chunk.model_dump_json()}\n\n"
    
    # 流式处理智能体响应
    content_buffer = ""
    try:
        # 直接迭代AsyncGenerator
        async for chunk in agent.run(query):
            if chunk.strip():
                content_buffer += chunk
                
                # 创建内容块
                content_chunk = ChatCompletionChunk(
                    id=request_id,
                    object="chat.completion.chunk",
                    created=created_time,
                    model=request.model,
                    choices=[
                        ChatCompletionChunkChoice(
                            index=0,
                            delta=DeltaMessage(role=None, content=chunk, tool_calls=None),
                            finish_reason=None
                        )
                    ],
                    usage=None,
                    system_fingerprint=None
                )
                yield f"data: {content_chunk.model_dump_json()}\n\n"
    
    except Exception as e:
        # 错误处理
        error_chunk = ChatCompletionChunk(
            id=request_id,
            object="chat.completion.chunk",
            created=created_time,
            model=request.model,
            choices=[
                ChatCompletionChunkChoice(
                    index=0,
                    delta=DeltaMessage(role=None, content=f"\n\nError: {str(e)}", tool_calls=None),
                    finish_reason="stop"
                )
            ],
            usage=None,
            system_fingerprint=None
        )
        yield f"data: {error_chunk.model_dump_json()}\n\n"
    
    # 发送结束块
    end_chunk = ChatCompletionChunk(
        id=request_id,
        object="chat.completion.chunk",
        created=created_time,
        model=request.model,
        choices=[
            ChatCompletionChunkChoice(
                index=0,
                delta=DeltaMessage(role=None, content=None, tool_calls=None),
                finish_reason="stop"
            )
        ],
        usage=Usage(
            prompt_tokens=estimate_tokens(query),
            completion_tokens=estimate_tokens(content_buffer),
            total_tokens=estimate_tokens(query + content_buffer)
        ),
        system_fingerprint=None
    )
    yield f"data: {end_chunk.model_dump_json()}\n\n"
    yield "data: [DONE]\n\n"


@app.get("/", response_model=ServerInfoResponse)
async def root():
    """服务器根路径信息"""
    return ServerInfoResponse(
        name="SimpleAgent API Server",
        version="1.0.0",
        description="OpenAI-compatible API for SimpleAgent universal framework",
        api_version="v1",
        supported_models=["simple-agent-v1"],
        capabilities=[
            "chat.completions",
            "streaming",
            "tool_calling",
            "conversation_history",
            "sketch_pad_storage"
        ]
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查端点"""
    global agent_registry
    
    # 获取默认Agent
    default_agent = agent_registry.get_agent("simple-agent-v1") if agent_registry else None
    
    return HealthResponse(
        status="ok",
        timestamp=datetime.now().isoformat(),
        version="1.0.0",
        agent_name=default_agent.name if default_agent else "Not initialized"
    )


@app.get("/v1/agents", response_model=Dict[str, Any])
async def list_agents():
    """列出所有活跃的Agent实例（用于调试和监控）"""
    global agent_registry
    
    if not agent_registry:
        return {"error": "Agent registry not initialized"}
    
    agents_info = {}
    for model_name in agent_registry.list_agents():
        agent_info = agent_registry.get_agent_info(model_name)
        if agent_info:
            # 添加实例ID用于验证单例
            agent = agent_registry.get_agent(model_name)
            agent_info["instance_id"] = id(agent) if agent else None
            agents_info[model_name] = agent_info
    
    return {
        "registry_stats": agent_registry.get_agent_stats(),
        "agents": agents_info
    }


@app.get("/v1/models", response_model=ModelListResponse)
async def list_models():
    """列出可用模型"""
    global agent_registry
    
    # 获取所有已注册的模型
    available_models = list_available_models()
    
    models = []
    for model_name in available_models:
        models.append(ModelInfo(
            id=model_name,
            object="model",
            created=int(time.time()),
            owned_by="simple-agent",
            permission=None,
            root=model_name,
            parent=None
        ))
    
    return ModelListResponse(object="list", data=models)


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """聊天完成端点（符合OpenAI规范）"""
    global agent_registry
    
    if not agent_registry:
        return create_error_response(
            message="Agent registry not initialized",
            error_type="server_error",
            status_code=500
        )
    
    # 使用统一的Agent获取逻辑
    try:
        agent = get_agent_for_model(request.model, agent_registry)
    except HTTPException as e:
        return create_error_response(
            message=e.detail,
            error_type="invalid_request" if e.status_code == 400 else "server_error",
            param="model" if e.status_code == 400 else None,
            status_code=e.status_code
        )
    
    # 验证请求
    if not request.messages:
        return create_error_response(
            message="Missing required parameter: messages",
            error_type="invalid_request",
            param="messages"
        )
    
    request_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"
    
    # 流式响应
    if request.stream:
        return StreamingResponse(
            stream_chat_completion(request, request_id),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
            }
        )
    
    # 非流式响应
    try:
        # 获取用户最后一条消息
        user_messages = [msg for msg in request.messages if msg.role == "user"]
        if not user_messages:
            return create_error_response(
                message="No user message found in conversation",
                error_type="invalid_request",
                param="messages"
            )
        
        last_user_message = user_messages[-1]
        query = last_user_message.content or ""
        
        # 收集智能体的完整响应
        full_response = ""
        async for chunk in agent.run(query):
            full_response += chunk
        
        # 创建响应
        created_time = int(time.time())
        response_message = ChatMessage(
            role="assistant",
            content=full_response.strip(),
            name=None,
            tool_calls=None,
            tool_call_id=None
        )
        
        choice = ChatChoice(
            index=0,
            message=response_message,
            finish_reason="stop"
        )
        
        usage = Usage(
            prompt_tokens=estimate_tokens(query),
            completion_tokens=estimate_tokens(full_response),
            total_tokens=estimate_tokens(query + full_response)
        )
        
        return ChatCompletionResponse(
            id=request_id,
            object="chat.completion",
            created=created_time,
            model=request.model,
            choices=[choice],
            usage=usage,
            system_fingerprint=None
        )
        
    except Exception as e:
        return create_error_response(
            message=f"Internal server error: {str(e)}",
            error_type="server_error",
            status_code=500
        )


@app.websocket("/v1/chat/completions/ws")
async def websocket_chat_completions(websocket: WebSocket):
    """通过WebSocket提供流式聊天完成"""
    await websocket.accept()
    global agent_registry

    if not agent_registry:
        await websocket.close(code=1011, reason="Agent registry not initialized")
        return

    try:
        while True:
            # 接收JSON格式的请求数据
            try:
                data = await websocket.receive_json()
                request = ChatCompletionRequest(**data)
            except Exception as e:
                error_detail = ErrorDetail(message=f"Invalid request format: {e}", type="invalid_request", param=None, code=None)
                await websocket.send_json(ErrorResponse(error=error_detail).model_dump())
                continue

            request_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"
            chunk_sequence = 0

            # 使用现有的流式生成器
            try:
                async for chunk_str in stream_chat_completion(request, request_id):
                    if chunk_str.strip():
                        # 解析SSE格式的数据
                        if chunk_str.startswith("data: "):
                            json_data_str = chunk_str[len("data: "):].strip()
                            if json_data_str != "[DONE]":
                                try:
                                    json_data = json.loads(json_data_str)
                                    # 封装数据包，加入序列号
                                    response_packet = {
                                        "sequence": chunk_sequence,
                                        "payload": json_data
                                    }
                                    await websocket.send_json(response_packet)
                                    chunk_sequence += 1
                                except json.JSONDecodeError:
                                    # 如果不是有效的JSON，则按原样发送（这种情况应较少见）
                                    await websocket.send_text(json_data_str)
                
                # 发送一个最终消息表示流结束
                await websocket.send_json({
                    "sequence": chunk_sequence,
                    "payload": {"id": request_id, "choices": [{"finish_reason": "stop"}]}
                })

            except Exception as e:
                error_detail = ErrorDetail(message=f"Error during streaming: {str(e)}", type="server_error", param=None, code=None)
                await websocket.send_json(ErrorResponse(error=error_detail).model_dump())

    except WebSocketDisconnect:
        print("Client disconnected from WebSocket.")
    except Exception as e:
        print(f"An unexpected error occurred in WebSocket: {e}")
        await websocket.close(code=1011, reason=f"Internal Server Error: {e}")


@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    """404错误处理"""
    return create_error_response(
        message=f"Not found: {request.url.path}",
        error_type="not_found",
        status_code=404
    )


@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    """500错误处理"""
    return create_error_response(
        message="Internal server error",
        error_type="server_error",
        status_code=500
    )


def start_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """启动Web服务器"""
    print(f"🌐 Starting SimpleAgent Web Server on http://{host}:{port}")
    print(f"📖 API Documentation: http://{host}:{port}/docs")
    print(f"🔄 ReDoc Documentation: http://{host}:{port}/redoc")
    
    uvicorn.run(
        "web_interface.server:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )


if __name__ == "__main__":
    start_server()
