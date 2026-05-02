"""
Request schemas for Autonomous Long-Term Agent API
Enhanced with better validation
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any


class StartAgentRequest(BaseModel):
    """Request to start a new agent instance"""
    instruction: Optional[str] = Field(
        default="Operate autonomously and continuously learn",
        description="Initial instruction for the agent",
        min_length=1,
        max_length=5000
    )
    agent_id: Optional[str] = Field(
        default=None,
        description="Optional custom agent ID",
        min_length=1,
        max_length=100
    )
    enhanced: bool = Field(
        default=True,
        description="Use enhanced version with paper-based optimizations (recommended)"
    )
    
    @validator('instruction')
    def validate_instruction(cls, v):
        """Validate instruction is not empty"""
        if v and len(v.strip()) == 0:
            raise ValueError("Instruction cannot be empty")
        return v


class AddTaskRequest(BaseModel):
    """Request to add a task to an agent"""
    instruction: str = Field(
        ...,
        description="Task instruction",
        min_length=1,
        max_length=5000
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Optional task metadata"
    )
    
    @validator('instruction')
    def validate_instruction(cls, v):
        """Validate instruction is not empty"""
        if not v or len(v.strip()) == 0:
            raise ValueError("Instruction cannot be empty")
        return v.strip()


class UpdateAgentRequest(BaseModel):
    """Request to update agent configuration"""
    instruction: Optional[str] = Field(
        default=None,
        description="Update agent instruction",
        min_length=1,
        max_length=5000
    )
    pause: Optional[bool] = Field(
        default=None,
        description="Pause/resume agent"
    )
    
    @validator('instruction')
    def validate_instruction(cls, v):
        """Validate instruction if provided"""
        if v is not None and len(v.strip()) == 0:
            raise ValueError("Instruction cannot be empty")
        return v.strip() if v else v


class ParallelAgentRequest(BaseModel):
    """Request to start multiple agents in parallel"""
    count: int = Field(
        ...,
        ge=1,
        le=10,
        description="Number of parallel agents to start (1-10)"
    )
    instruction: Optional[str] = Field(
        default="Operate autonomously and continuously learn",
        description="Initial instruction for all agents",
        min_length=1,
        max_length=5000
    )
    enhanced: bool = Field(
        default=True,
        description="Use enhanced version with paper-based optimizations (recommended)"
    )
    
    @validator('instruction')
    def validate_instruction(cls, v):
        """Validate instruction is not empty"""
        if v and len(v.strip()) == 0:
            raise ValueError("Instruction cannot be empty")
        return v
