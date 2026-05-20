"""
Core module for Unified AI Model.
Contains LLM client, services, and utilities.
"""

from .llm_client import OpenRouterClient, get_openrouter_client
from .llm_service import LLMService, LLMResponse, LLMRequest, get_llm_service
from .chat_service import ChatService, Conversation, Message, get_chat_service
from .performance_monitor import PerformanceMonitor, get_performance_monitor
from .continuous_agent import (
    ContinuousAgent,
    AgentTask,
    AgentStatus,
    TaskStatus,
    BatchProcessor,
    WorkerPool,
    PriorityTaskQueue,
    PriorityLevel,
    create_agent,
    get_agent,
    list_agents,
    stop_agent,
    stop_all_agents
)

__all__ = [
    "OpenRouterClient",
    "get_openrouter_client",
    "LLMService",
    "LLMResponse",
    "LLMRequest",
    "get_llm_service",
    "ChatService",
    "Conversation",
    "Message",
    "get_chat_service",
    "PerformanceMonitor",
    "get_performance_monitor",
    "ContinuousAgent",
    "AgentTask",
    "AgentStatus",
    "TaskStatus",
    "BatchProcessor",
    "WorkerPool",
    "PriorityTaskQueue",
    "PriorityLevel",
    "create_agent",
    "get_agent",
    "list_agents",
    "stop_agent",
    "stop_all_agents",
]



