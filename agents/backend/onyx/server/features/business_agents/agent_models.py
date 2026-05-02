"""
Data Models for Business Agents

Core data structures including BusinessArea, AgentCapability, and BusinessAgent.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class BusinessArea(str, Enum):
    """Business areas for agent specialization."""

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

    def __str__(self) -> str:
        return self.value


@dataclass
class AgentCapability:
    """Capability definition for a business agent."""

    name: str
    description: str
    input_types: List[str] = field(default_factory=list)
    output_types: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    estimated_duration: int = 0  # In milliseconds


@dataclass
class BusinessAgent:
    """Core entity representing a business agent."""

    id: str
    name: str
    business_area: BusinessArea | str  # Accept str for backward compat
    description: str
    capabilities: List[AgentCapability] = field(default_factory=list)
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Sanitize fields after creation."""
        if not self.id:
            self.id = str(uuid.uuid4())
        
        # Ensure business_area is enum (if it matches)
        if isinstance(self.business_area, str):
            try:
                self.business_area = BusinessArea(self.business_area)
            except ValueError:
                pass  # Keep as string if custom area

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        from dataclasses import asdict
        return asdict(self)
