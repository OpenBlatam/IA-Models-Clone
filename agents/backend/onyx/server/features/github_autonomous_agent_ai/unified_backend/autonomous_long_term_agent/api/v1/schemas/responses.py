"""
Response schemas for Autonomous Long-Term Agent API
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime


class AgentStatusResponse(BaseModel):
    """Agent status response"""
    agent_id: str
    status: str
    instruction: str
    metrics: Dict[str, Any]
    queue_size: int
    learning_stats: Dict[str, Any]
    knowledge_stats: Dict[str, Any]
    timestamp: str


class TaskResponse(BaseModel):
    """Task response"""
    task_id: str
    status: str
    instruction: str
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class AgentListResponse(BaseModel):
    """List of agents response"""
    agents: List[Dict[str, Any]]
    total: int


class MessageResponse(BaseModel):
    """Simple message response"""
    message: str
    success: bool = True


class ParallelAgentsResponse(BaseModel):
    """Response for parallel agents creation"""
    agent_ids: List[str]
    total: int
    message: str




