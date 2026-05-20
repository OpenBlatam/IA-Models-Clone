"""
Secondary LLM Flows Base Classes and Interfaces
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any, Optional
from datetime import datetime
from uuid import uuid4
from dataclasses import dataclass


class FlowType(str, Enum):
    """Flow types"""
    VALIDATION = "validation"
    REFINEMENT = "refinement"
    POST_PROCESSING = "post_processing"
    FALLBACK = "fallback"


@dataclass
class FlowResult:
    """Flow result"""
    success: bool
    output: Any
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class LLMFlow:
    """LLM flow definition"""
    
    def __init__(
        self,
        name: str,
        flow_type: FlowType,
        config: Optional[Dict[str, Any]] = None
    ):
        self.id = str(uuid4())
        self.name = name
        self.flow_type = flow_type
        self.config = config or {}
        self.created_at = datetime.utcnow()


class SecondaryFlowBase(ABC):
    """Base interface for secondary LLM flows"""
    
    @abstractmethod
    async def validate(
        self,
        content: str,
        criteria: Optional[Dict[str, Any]] = None
    ) -> FlowResult:
        """Validate content"""
        pass
    
    @abstractmethod
    async def refine(
        self,
        content: str,
        instructions: str
    ) -> FlowResult:
        """Refine content"""
        pass
    
    @abstractmethod
    async def post_process(
        self,
        content: str,
        processing_type: str
    ) -> FlowResult:
        """Post-process content"""
        pass

