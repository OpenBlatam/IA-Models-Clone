"""
FastAPI Routes for Unified AI Model
Complete API for chat and LLM functionality
"""

import json
import asyncio
import logging
from typing import Optional, Dict, Any, List
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from fastapi.responses import StreamingResponse

from .schemas import (
    ChatRequest,
    ChatResponse,
    ChatMessageResponse,
    GenerateRequest,
    GenerateResponse,
    StreamRequest,
    ParallelGenerateRequest,
    ParallelGenerateResponse,
    ModelResponse,
    CreateConversationRequest,
    ConversationResponse,
    ConversationListResponse,
    ConversationHistoryResponse,
    ModelsResponse,
    StatsResponse,
    HealthResponse,
    CodeAnalysisRequest,
    CodeAnalysisResponse
)
from ..core.llm_service import LLMService, get_llm_service
from ..core.chat_service import ChatService, get_chat_service
from ..core.performance_monitor import PerformanceMonitor, get_performance_monitor
from ..config import get_config

logger = logging.getLogger(__name__)

# Version defined here to avoid circular import
__version__ = "1.0.0"

router = APIRouter(prefix="/api/v1", tags=["Unified AI"])


# ==================== Dependencies ====================

def get_llm() -> LLMService:
    """Dependency to get LLM service."""
    return get_llm_service()


def get_chat() -> ChatService:
    """Dependency to get chat service."""
    return get_chat_service()


def get_monitor() -> PerformanceMonitor:
    """Dependency to get performance monitor."""
    return get_performance_monitor()


# ==================== Health & Status ====================

@router.get("/health", response_model=HealthResponse)
async def health_check(monitor: PerformanceMonitor = Depends(get_monitor)):
    """Health check endpoint."""
    config = get_config()
    
    return HealthResponse(
        status="healthy",
        version=__version__,
        uptime_seconds=monitor.get_uptime_seconds(),
        models_available=bool(config.openrouter.api_key)
    )


@router.get("/stats", response_model=StatsResponse)
async def get_stats(
    llm: LLMService = Depends(get_llm),
    monitor: PerformanceMonitor = Depends(get_monitor)
):
    """Get service statistics."""
    summary = monitor.get_summary()
    
    return StatsResponse(
        uptime_seconds=summary["uptime_seconds"],
        requests=summary["requests"],
        cache=summary["cache"],
        latency=summary["latency"],
        tokens=summary["tokens"]
    )


@router.post("/stats/reset")
async def reset_stats(
    llm: LLMService = Depends(get_llm),
    monitor: PerformanceMonitor = Depends(get_monitor)
):
    """Reset all statistics."""
    llm.reset_stats()
    monitor.reset()
    return {"message": "Statistics reset successfully"}


# ==================== Chat Endpoints ====================

@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    chat_service: ChatService = Depends(get_chat),
    monitor: PerformanceMonitor = Depends(get_monitor)
):
    """
    Send a chat message and get a response.
    
    Creates a new conversation if conversation_id is not provided.
    Maintains conversation history automatically.
    """
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    try:
        response = await chat_service.chat(
            message=request.message.strip(),
            conversation_id=request.conversation_id,
            system_prompt=request.system_prompt,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        
        if response.latency_ms:
            monitor.record_request(
                model=response.model,
                latency_ms=response.latency_ms,
                tokens=response.usage.get("total_tokens", 0) if response.usage else 0,
                success=response.error is None
            )
        
        message_response = None
        if response.message:
            message_response = ChatMessageResponse(
                role=response.message.role.value,
                content=response.message.content,
                message_id=response.message.message_id,
                timestamp=response.message.timestamp.isoformat()
            )
        
        return ChatResponse(
            success=response.error is None,
            message=message_response,
            conversation_id=response.conversation_id,
            model=response.model,
            usage=response.usage,
            latency_ms=response.latency_ms,
            error=response.error
        )
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        monitor.record_error("chat_error", str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    chat_service: ChatService = Depends(get_chat)
):
    """
    Send a chat message and stream the response.
    
    Returns Server-Sent Events (SSE) with response chunks.
    """
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    async def generate():
        try:
            async for chunk in chat_service.chat_stream(
                message=request.message.strip(),
                conversation_id=request.conversation_id,
                system_prompt=request.system_prompt,
                model=request.model,
                temperature=request.temperature,
                max_tokens=request.max_tokens
            ):
                yield f"data: {json.dumps({'chunk': chunk})}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


# ==================== Generate Endpoints ====================

@router.post("/generate", response_model=GenerateResponse)
async def generate(
    request: GenerateRequest,
    llm: LLMService = Depends(get_llm),
    monitor: PerformanceMonitor = Depends(get_monitor)
):
    """
    Generate text from a prompt.
    
    Single-turn generation without conversation history.
    """
    if not request.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")
    
    try:
        response = await llm.generate(
            prompt=request.prompt.strip(),
            model=request.model,
            system_prompt=request.system_prompt,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        
        if response.latency_ms:
            monitor.record_request(
                model=response.model,
                latency_ms=response.latency_ms,
                tokens=response.usage.get("total_tokens", 0) if response.usage else 0,
                success=response.error is None,
                cached=response.cached
            )
        
        return GenerateResponse(
            success=response.error is None,
            content=response.content,
            model=response.model,
            usage=response.usage,
            latency_ms=response.latency_ms,
            cached=response.cached,
            error=response.error
        )
        
    except Exception as e:
        logger.error(f"Generate error: {e}")
        monitor.record_error("generate_error", str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate/stream")
async def generate_stream(
    request: StreamRequest,
    llm: LLMService = Depends(get_llm)
):
    """
    Generate text with streaming response.
    
    Returns Server-Sent Events (SSE) with response chunks.
    """
    if not request.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")
    
    async def generate():
        try:
            async for chunk in llm.generate_stream(
                prompt=request.prompt.strip(),
                model=request.model,
                system_prompt=request.system_prompt,
                temperature=request.temperature,
                max_tokens=request.max_tokens
            ):
                yield f"data: {json.dumps({'chunk': chunk})}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@router.post("/generate/parallel", response_model=ParallelGenerateResponse)
async def generate_parallel(
    request: ParallelGenerateRequest,
    llm: LLMService = Depends(get_llm),
    monitor: PerformanceMonitor = Depends(get_monitor)
):
    """
    Generate responses from multiple models in parallel.
    
    Useful for comparing models or getting diverse responses.
    """
    if not request.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")
    
    if request.models and len(request.models) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 models allowed")
    
    try:
        responses = await llm.generate_parallel(
            prompt=request.prompt.strip(),
            models=request.models,
            system_prompt=request.system_prompt,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        
        model_responses = {}
        successful = 0
        
        for model, response in responses.items():
            if not response.error:
                successful += 1
            
            model_responses[model] = ModelResponse(
                model=response.model,
                content=response.content,
                usage=response.usage,
                latency_ms=response.latency_ms,
                error=response.error
            )
        
        return ParallelGenerateResponse(
            success=True,
            responses=model_responses,
            total_models=len(responses),
            successful_models=successful
        )
        
    except Exception as e:
        logger.error(f"Parallel generate error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Conversation Endpoints ====================

@router.post("/conversations", response_model=ConversationResponse)
async def create_conversation(
    request: CreateConversationRequest,
    chat_service: ChatService = Depends(get_chat)
):
    """Create a new conversation."""
    conversation = chat_service.create_conversation(
        system_prompt=request.system_prompt,
        model=request.model,
        metadata=request.metadata
    )
    
    return ConversationResponse(
        conversation_id=conversation.conversation_id,
        message_count=len(conversation.messages),
        model=conversation.model,
        created_at=conversation.created_at.isoformat(),
        updated_at=conversation.updated_at.isoformat()
    )


@router.get("/conversations", response_model=ConversationListResponse)
async def list_conversations(
    chat_service: ChatService = Depends(get_chat)
):
    """List all conversations."""
    conversations = chat_service.list_conversations()
    
    return ConversationListResponse(
        conversations=[
            ConversationResponse(
                conversation_id=c["conversation_id"],
                message_count=c["message_count"],
                model=c.get("model"),
                created_at=c["created_at"],
                updated_at=c["updated_at"]
            )
            for c in conversations
        ],
        total=len(conversations)
    )


@router.get("/conversations/{conversation_id}")
async def get_conversation(
    conversation_id: str,
    chat_service: ChatService = Depends(get_chat)
):
    """Get a conversation by ID."""
    conversation = chat_service.get_conversation(conversation_id)
    
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    return conversation.to_dict()


@router.get("/conversations/{conversation_id}/history", response_model=ConversationHistoryResponse)
async def get_conversation_history(
    conversation_id: str,
    limit: Optional[int] = None,
    chat_service: ChatService = Depends(get_chat)
):
    """Get conversation message history."""
    messages = chat_service.get_conversation_history(conversation_id, limit)
    
    if not messages and not chat_service.get_conversation(conversation_id):
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    return ConversationHistoryResponse(
        conversation_id=conversation_id,
        messages=messages,
        total=len(messages)
    )


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    chat_service: ChatService = Depends(get_chat)
):
    """Delete a conversation."""
    if not chat_service.delete_conversation(conversation_id):
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    return {"message": "Conversation deleted successfully"}


@router.post("/conversations/{conversation_id}/clear")
async def clear_conversation_history(
    conversation_id: str,
    chat_service: ChatService = Depends(get_chat)
):
    """Clear conversation history while keeping the conversation."""
    if not chat_service.clear_conversation_history(conversation_id):
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    return {"message": "Conversation history cleared"}


@router.post("/conversations/{conversation_id}/regenerate")
async def regenerate_last_response(
    conversation_id: str,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    chat_service: ChatService = Depends(get_chat)
):
    """Regenerate the last assistant response."""
    response = await chat_service.regenerate_last_response(
        conversation_id=conversation_id,
        model=model,
        temperature=temperature
    )
    
    if response.error:
        raise HTTPException(status_code=400, detail=response.error)
    
    return response.to_dict()


# ==================== Models Endpoints ====================

@router.get("/models", response_model=ModelsResponse)
async def get_models(llm: LLMService = Depends(get_llm)):
    """Get list of available models from OpenRouter."""
    try:
        models = await llm.get_available_models()
        return ModelsResponse(
            models=models,
            total=len(models)
        )
    except Exception as e:
        logger.error(f"Error getting models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/default")
async def get_default_models(llm: LLMService = Depends(get_llm)):
    """Get the default models configured."""
    return {
        "default_model": llm.default_model,
        "default_models": llm.default_models
    }


# ==================== Code Analysis Endpoints ====================

@router.post("/code/analyze", response_model=CodeAnalysisResponse)
async def analyze_code(
    request: CodeAnalysisRequest,
    llm: LLMService = Depends(get_llm)
):
    """
    Analyze code for issues, improvements, or security.
    
    Supported analysis types: general, bugs, performance, security
    """
    try:
        response = await llm.analyze_code(
            code=request.code,
            language=request.language,
            analysis_type=request.analysis_type,
            model=request.model
        )
        
        return CodeAnalysisResponse(
            success=response.error is None,
            analysis=response.content,
            model=response.model,
            latency_ms=response.latency_ms,
            error=response.error
        )
        
    except Exception as e:
        logger.error(f"Code analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Continuous Agent Endpoints ====================

from ..core.continuous_agent import (
    ContinuousAgent,
    create_agent,
    get_agent,
    list_agents,
    stop_agent,
    stop_all_agents
)

# Store for running agent tasks
_agent_tasks: Dict[str, asyncio.Task] = {}


class CreateAgentRequest(BaseModel):
    """Request to create a new continuous agent."""
    name: str = Field("ContinuousAgent", description="Agent name")
    system_prompt: Optional[str] = Field(None, description="System prompt")
    model: Optional[str] = Field(None, description="Model to use")
    loop_interval: float = Field(1.0, ge=0.1, le=60.0, description="Loop interval in seconds")
    idle_interval: float = Field(5.0, ge=1.0, le=300.0, description="Idle interval in seconds")
    # Parallel processing options
    enable_parallel: bool = Field(True, description="Enable parallel task processing")
    max_concurrent_tasks: int = Field(3, ge=1, le=20, description="Max concurrent tasks")
    batch_size: int = Field(8, ge=1, le=64, description="Batch size for parallel processing")
    num_workers: int = Field(3, ge=1, le=10, description="Number of worker threads")


class SubmitTaskRequest(BaseModel):
    """Request to submit a task to an agent."""
    description: str = Field(..., min_length=1, max_length=50000, description="Task description")
    priority: int = Field(5, ge=1, le=10, description="Priority 1-10")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class SubmitBatchRequest(BaseModel):
    """Request to submit multiple tasks at once."""
    tasks: List[SubmitTaskRequest] = Field(..., min_length=1, max_length=100, description="List of tasks")


@router.post("/agents")
async def create_continuous_agent(request: CreateAgentRequest):
    """
    Create a new continuous agent.
    
    The agent runs indefinitely until explicitly stopped.
    Supports parallel processing for high throughput.
    """
    try:
        agent = create_agent(
            name=request.name,
            system_prompt=request.system_prompt,
            model=request.model,
            loop_interval=request.loop_interval,
            idle_interval=request.idle_interval,
            enable_parallel=request.enable_parallel,
            max_concurrent_tasks=request.max_concurrent_tasks,
            batch_size=request.batch_size,
            num_workers=request.num_workers
        )
        
        # Start agent in background
        task = asyncio.create_task(agent.start())
        _agent_tasks[agent.agent_id] = task
        
        return {
            "success": True,
            "agent_id": agent.agent_id,
            "name": agent.name,
            "parallel_config": {
                "enabled": request.enable_parallel,
                "max_concurrent": request.max_concurrent_tasks,
                "batch_size": request.batch_size,
                "num_workers": request.num_workers
            },
            "message": "Agent created and started. It will run until stopped."
        }
        
    except Exception as e:
        logger.error(f"Error creating agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/agents")
async def list_all_agents():
    """List all continuous agents."""
    return {
        "agents": list_agents(),
        "total": len(list_agents())
    }


@router.get("/agents/{agent_id}")
async def get_agent_status(agent_id: str):
    """Get status of a specific agent."""
    agent = get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    return agent.get_status()


@router.post("/agents/{agent_id}/tasks")
async def submit_agent_task(agent_id: str, request: SubmitTaskRequest):
    """Submit a task to a continuous agent."""
    agent = get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    try:
        task_id = agent.submit_task(
            description=request.description,
            priority=request.priority,
            metadata=request.metadata
        )
        
        return {
            "success": True,
            "task_id": task_id,
            "message": "Task submitted successfully"
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/agents/{agent_id}/tasks/batch")
async def submit_batch_tasks(agent_id: str, request: SubmitBatchRequest):
    """
    Submit multiple tasks at once for parallel processing.
    
    More efficient than submitting tasks one by one.
    """
    agent = get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    try:
        tasks_data = [
            {
                "description": t.description,
                "priority": t.priority,
                "metadata": t.metadata
            }
            for t in request.tasks
        ]
        
        task_ids = await agent.submit_batch(tasks_data)
        
        successful = [tid for tid in task_ids if tid is not None]
        failed = len(task_ids) - len(successful)
        
        return {
            "success": True,
            "task_ids": successful,
            "submitted": len(successful),
            "failed": failed,
            "message": f"Batch submitted: {len(successful)} tasks"
        }
        
    except Exception as e:
        logger.error(f"Error submitting batch: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/agents/{agent_id}/tasks")
async def get_agent_tasks(agent_id: str):
    """Get tasks for an agent."""
    agent = get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    return {
        "queue": agent.get_queue(),
        "active": agent.get_active_tasks(),
        "completed": agent.get_completed_tasks(limit=20)
    }


@router.get("/agents/{agent_id}/tasks/{task_id}")
async def get_task_status(agent_id: str, task_id: str):
    """Get status of a specific task."""
    agent = get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    task = agent.get_task_status(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return task


@router.delete("/agents/{agent_id}/tasks/{task_id}")
async def cancel_agent_task(agent_id: str, task_id: str):
    """Cancel a pending task."""
    agent = get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    if agent.cancel_task(task_id):
        return {"success": True, "message": "Task cancelled"}
    else:
        raise HTTPException(status_code=404, detail="Task not found or already processing")


@router.post("/agents/{agent_id}/pause")
async def pause_agent(agent_id: str):
    """Pause an agent (stops processing new tasks)."""
    agent = get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    agent.pause()
    return {"success": True, "message": "Agent paused"}


@router.post("/agents/{agent_id}/resume")
async def resume_agent(agent_id: str):
    """Resume a paused agent."""
    agent = get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    agent.resume()
    return {"success": True, "message": "Agent resumed"}


@router.post("/agents/{agent_id}/stop")
async def stop_continuous_agent(agent_id: str):
    """Stop an agent."""
    agent = get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    agent.stop()
    
    # Cancel the background task
    if agent_id in _agent_tasks:
        _agent_tasks[agent_id].cancel()
        del _agent_tasks[agent_id]
    
    return {"success": True, "message": "Agent stopped"}


@router.post("/agents/stop-all")
async def stop_all_continuous_agents():
    """Stop all agents."""
    count = stop_all_agents()
    
    # Cancel all background tasks
    for task in _agent_tasks.values():
        task.cancel()
    _agent_tasks.clear()
    
    return {"success": True, "message": f"Stopped {count} agents"}



