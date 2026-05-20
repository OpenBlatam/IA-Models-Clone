"""
Agent Schemas
=============

Pydantic models for business agent operations.
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime

from ..business_agents import BusinessArea

class AgentCapabilityRequest(BaseModel):
    """Request schema for agent capability creation."""
    
    name: str = Field(..., min_length=1, max_length=100, description="Capability name")
    description: str = Field(..., min_length=1, max_length=500, description="Capability description")
    input_types: List[str] = Field(..., description="Expected input types")
    output_types: List[str] = Field(..., description="Expected output types")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Capability parameters")
    estimated_duration: int = Field(300, ge=1, le=3600, description="Estimated duration in seconds")

class BusinessAgentRequest(BaseModel):
    """Request schema for business agent creation."""
    
    name: str = Field(..., min_length=1, max_length=100, description="Agent name")
    business_area: BusinessArea = Field(..., description="Business area")
    description: str = Field(..., min_length=1, max_length=1000, description="Agent description")
    capabilities: List[AgentCapabilityRequest] = Field(..., description="Agent capabilities")
    is_active: bool = Field(True, description="Whether agent is active")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class AgentCapabilityResponse(BaseModel):
    """Response schema for agent capability."""
    
    name: str = Field(..., description="Capability name")
    description: str = Field(..., description="Capability description")
    input_types: List[str] = Field(..., description="Input types")
    output_types: List[str] = Field(..., description="Output types")
    parameters: Dict[str, Any] = Field(..., description="Capability parameters")
    estimated_duration: int = Field(..., description="Estimated duration in seconds")

class AgentResponse(BaseModel):
    """Response schema for business agent."""
    
    id: str = Field(..., description="Agent ID")
    name: str = Field(..., description="Agent name")
    business_area: str = Field(..., description="Business area")
    description: str = Field(..., description="Agent description")
    capabilities: List[AgentCapabilityResponse] = Field(..., description="Agent capabilities")
    is_active: bool = Field(..., description="Whether agent is active")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

class CapabilityExecutionRequest(BaseModel):
    """Request schema for capability execution."""
    
    agent_id: str = Field(..., description="Agent ID")
    capability_name: str = Field(..., description="Capability name")
    inputs: Dict[str, Any] = Field(..., description="Execution inputs")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Execution parameters")

class CapabilityExecutionResponse(BaseModel):
    """Response schema for capability execution."""
    
    status: str = Field(..., description="Execution status")
    agent_id: str = Field(..., description="Agent ID")
    capability: str = Field(..., description="Capability name")
    result: Optional[Dict[str, Any]] = Field(None, description="Execution result")
    error: Optional[str] = Field(None, description="Error message if failed")
    execution_time: Optional[int] = Field(None, description="Execution time in seconds")
    timestamp: datetime = Field(default_factory=datetime.now, description="Execution timestamp")

class AgentListResponse(BaseModel):
    """Response schema for agent list."""
    
    agents: List[AgentResponse] = Field(..., description="List of agents")
    total: int = Field(..., description="Total number of agents")
    business_area: Optional[str] = Field(None, description="Filtered business area")
    is_active: Optional[bool] = Field(None, description="Filtered active status")

class BusinessAreaResponse(BaseModel):
    """Response schema for business area information."""
    
    value: str = Field(..., description="Business area value")
    name: str = Field(..., description="Business area display name")
    agents_count: int = Field(..., description="Number of agents in this area")
    description: Optional[str] = Field(None, description="Business area description")
