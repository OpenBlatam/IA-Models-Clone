"""
Response schemas for Lovable Community SAM3 API.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime


class TaskResponse(BaseModel):
    """Response schema for task operations."""
    
    task_id: str = Field(..., description="Task ID")
    status: str = Field(..., description="Task status")
    message: str = Field(..., description="Status message")
    created_at: Optional[datetime] = Field(default_factory=datetime.now, description="Creation timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "task_id": "task123",
                "status": "created",
                "message": "Task created successfully",
                "created_at": "2024-01-01T00:00:00"
            }
        }


class ChatResponse(BaseModel):
    """Response schema for chat data."""
    
    id: str = Field(..., description="Chat ID")
    user_id: str = Field(..., description="User ID")
    title: str = Field(..., description="Chat title")
    content: str = Field(..., description="Chat content")
    description: Optional[str] = Field(None, description="Chat description")
    tags: Optional[List[str]] = Field(default_factory=list, description="Chat tags")
    category: Optional[str] = Field(None, description="Chat category")
    is_public: bool = Field(True, description="Whether chat is public")
    is_featured: bool = Field(False, description="Whether chat is featured")
    vote_count: int = Field(0, description="Vote count")
    remix_count: int = Field(0, description="Remix count")
    view_count: int = Field(0, description="View count")
    score: float = Field(0.0, description="Ranking score")
    created_at: Optional[datetime] = Field(None, description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Update timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "chat123",
                "user_id": "user123",
                "title": "My Chat",
                "content": "Chat content",
                "description": "A description",
                "tags": ["tag1", "tag2"],
                "category": "general",
                "is_public": True,
                "is_featured": False,
                "vote_count": 10,
                "remix_count": 5,
                "view_count": 100,
                "score": 85.5,
                "created_at": "2024-01-01T00:00:00",
                "updated_at": "2024-01-01T00:00:00"
            }
        }


class StatsResponse(BaseModel):
    """Response schema for statistics."""
    
    executor: Dict[str, Any] = Field(..., description="Executor statistics")
    agent_running: bool = Field(..., description="Whether agent is running")
    total_tasks: Optional[int] = Field(None, description="Total tasks")
    completed_tasks: Optional[int] = Field(None, description="Completed tasks")
    failed_tasks: Optional[int] = Field(None, description="Failed tasks")
    
    class Config:
        json_schema_extra = {
            "example": {
                "executor": {
                    "total_tasks": 100,
                    "completed_tasks": 90,
                    "failed_tasks": 5,
                    "active_workers": 4
                },
                "agent_running": True,
                "total_tasks": 100,
                "completed_tasks": 90,
                "failed_tasks": 5
            }
        }


class ErrorResponse(BaseModel):
    """Response schema for errors."""
    
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Error details")
    status_code: int = Field(..., description="HTTP status code")
    
    class Config:
        json_schema_extra = {
            "example": {
                "error": "NotFoundError",
                "message": "Resource not found",
                "details": {"resource": "Chat", "id": "chat123"},
                "status_code": 404
            }
        }




