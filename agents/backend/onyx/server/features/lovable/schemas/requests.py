"""
Request schemas for Lovable Community SAM3 API.
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime


class PublishChatRequest(BaseModel):
    """Request schema for publishing a chat."""
    
    user_id: str = Field(..., description="User ID")
    title: str = Field(..., min_length=1, max_length=200, description="Chat title")
    content: str = Field(..., min_length=1, description="Chat content")
    description: Optional[str] = Field(None, max_length=500, description="Chat description")
    tags: Optional[List[str]] = Field(default_factory=list, description="Chat tags")
    category: Optional[str] = Field(None, max_length=50, description="Chat category")
    is_public: bool = Field(True, description="Whether chat is public")
    priority: int = Field(5, ge=1, le=10, description="Task priority")
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "user123",
                "title": "My Chat",
                "content": "This is my chat content",
                "description": "A description",
                "tags": ["tag1", "tag2"],
                "category": "general",
                "is_public": True,
                "priority": 5
            }
        }


class OptimizeContentRequest(BaseModel):
    """Request schema for optimizing content."""
    
    content: str = Field(..., min_length=1, description="Content to optimize")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional context")
    priority: int = Field(5, ge=1, le=10, description="Task priority")
    
    class Config:
        json_schema_extra = {
            "example": {
                "content": "Content to optimize",
                "context": {},
                "priority": 5
            }
        }


class VoteRequest(BaseModel):
    """Request schema for voting on a chat."""
    
    chat_id: Optional[str] = Field(None, description="Chat ID (optional if in path)")
    user_id: str = Field(..., description="User ID")
    vote_type: str = Field(..., pattern="^(upvote|downvote)$", description="Vote type")
    
    class Config:
        json_schema_extra = {
            "example": {
                "chat_id": "chat123",
                "user_id": "user123",
                "vote_type": "upvote"
            }
        }


class RemixRequest(BaseModel):
    """Request schema for creating a remix."""
    
    original_chat_id: Optional[str] = Field(None, description="Original chat ID (optional if in path)")
    user_id: str = Field(..., description="User ID")
    title: str = Field(..., min_length=1, max_length=200, description="Remix title")
    content: str = Field(..., min_length=1, description="Remix content")
    description: Optional[str] = Field(None, max_length=500, description="Remix description")
    tags: Optional[List[str]] = Field(default_factory=list, description="Remix tags")
    
    class Config:
        json_schema_extra = {
            "example": {
                "original_chat_id": "chat123",
                "user_id": "user123",
                "title": "My Remix",
                "content": "Remix content",
                "description": "A remix",
                "tags": ["tag1"]
            }
        }


class UpdateChatRequest(BaseModel):
    """Request schema for updating a chat."""
    
    title: Optional[str] = Field(None, min_length=1, max_length=200, description="Chat title")
    content: Optional[str] = Field(None, min_length=1, description="Chat content")
    description: Optional[str] = Field(None, max_length=500, description="Chat description")
    tags: Optional[List[str]] = Field(None, description="Chat tags")
    category: Optional[str] = Field(None, max_length=50, description="Chat category")
    is_public: Optional[bool] = Field(None, description="Whether chat is public")
    
    class Config:
        json_schema_extra = {
            "example": {
                "title": "Updated Title",
                "content": "Updated content",
                "description": "Updated description",
                "tags": ["tag1", "tag2"],
                "category": "general",
                "is_public": True
            }
        }


class FeatureChatRequest(BaseModel):
    """Request schema for featuring a chat."""
    
    featured: bool = Field(True, description="Whether to feature the chat")
    
    class Config:
        json_schema_extra = {
            "example": {
                "featured": True
            }
        }


class BatchOperationRequest(BaseModel):
    """Request schema for batch operations."""
    
    operation: str = Field(..., pattern="^(delete|update|feature|unfeature)$", description="Operation type")
    chat_ids: List[str] = Field(..., min_items=1, description="List of chat IDs")
    update_data: Optional[Dict[str, Any]] = Field(None, description="Update data (for update operation)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "operation": "delete",
                "chat_ids": ["chat1", "chat2"],
                "update_data": None
            }
        }




