"""
Agent Definitions
=================

Core agent definitions and data structures.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime

class BusinessArea(Enum):
    """Business area enumeration"""
    MARKETING = "marketing"
    SALES = "sales"
    OPERATIONS = "operations"
    HR = "hr"
    FINANCE = "finance"
    LEGAL = "legal"
    TECHNICAL = "technical"
    CONTENT = "content"
    CUSTOMER_SERVICE = "customer_service"
    PRODUCT_DEVELOPMENT = "product_development"
    STRATEGY = "strategy"
    COMPLIANCE = "compliance"

@dataclass
class AgentCapability:
    """Agent capability definition"""
    name: str
    description: str
    input_types: List[str]
    output_types: List[str]
    parameters: Dict[str, Any]
    estimated_duration: int  # seconds
    required_permissions: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

@dataclass
class BusinessAgent:
    """Business agent definition"""
    id: str
    name: str
    business_area: BusinessArea
    description: str
    capabilities: List[AgentCapability]
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "1.0.0"
    author: str = "system"
    dependencies: List[str] = field(default_factory=list)
    configuration: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AgentExecution:
    """Agent execution result"""
    execution_id: str
    agent_id: str
    capability_name: str
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    status: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class AgentStatus(Enum):
    """Agent status enumeration"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    MAINTENANCE = "maintenance"
    ERROR = "error"

class ExecutionStatus(Enum):
    """Execution status enumeration"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
