from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum
from dataclasses import dataclass, field
from uuid import UUID, uuid4
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Template domain entity for LinkedIn Posts system.
"""



class TemplateCategory(str, Enum):
    """Template category enumeration."""
    BUSINESS = "business"
    MARKETING = "marketing"
    PERSONAL = "personal"
    EDUCATIONAL = "educational"
    INSPIRATIONAL = "inspirational"
    PROMOTIONAL = "promotional"


class TemplateType(str, Enum):
    """Template type enumeration."""
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    ARTICLE = "article"
    POLL = "poll"


@dataclass
class TemplateVariable:
    """Template variable definition."""
    name: str
    description: str
    type: str = "string"
    required: bool = True
    default_value: Optional[str] = None
    options: Optional[List[str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "type": self.type,
            "required": self.required,
            "default_value": self.default_value,
            "options": self.options
        }


@dataclass
class Template:
    """
    Template domain entity for post generation.
    
    Features:
    - Variable substitution
    - Category organization
    - Usage tracking
    - AI optimization
    """
    
    # Core fields
    id: UUID = field(default_factory=uuid4)
    name: str = ""
    description: str = ""
    category: TemplateCategory = TemplateCategory.BUSINESS
    template_type: TemplateType = TemplateType.TEXT
    
    # Content
    content_template: str = ""
    variables: List[TemplateVariable] = field(default_factory=list)
    
    # Metadata
    tags: List[str] = field(default_factory=list)
    industry: Optional[str] = None
    target_audience: Optional[str] = None
    
    # Usage and performance
    usage_count: int = 0
    success_rate: float = 0.0
    average_engagement: float = 0.0
    
    # Status
    is_active: bool = True
    is_premium: bool = False
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    # AI and optimization
    ai_score: float = 0.0
    optimization_suggestions: List[str] = field(default_factory=list)
    
    # Creator info
    created_by: Optional[UUID] = None
    version: str = "1.0.0"
    
    def __post_init__(self) -> Any:
        """Post-initialization processing."""
        if isinstance(self.variables, list):
            self.variables = [
                TemplateVariable(**var) if isinstance(var, dict) else var
                for var in self.variables
            ]
    
    @property
    def variable_names(self) -> List[str]:
        """Get list of variable names."""
        return [var.name for var in self.variables]
    
    @property
    def required_variables(self) -> List[str]:
        """Get list of required variable names."""
        return [var.name for var in self.variables if var.required]
    
    def get_variable(self, name: str) -> Optional[TemplateVariable]:
        """Get variable by name."""
        for var in self.variables:
            if var.name == name:
                return var
        return None
    
    def validate_variables(self, provided_vars: Dict[str, str]) -> List[str]:
        """Validate provided variables and return missing required ones."""
        missing = []
        for var in self.variables:
            if var.required and var.name not in provided_vars:
                missing.append(var.name)
        return missing
    
    def render(self, variables: Dict[str, str]) -> str:
        """Render template with provided variables."""
        content = self.content_template
        
        # Replace variables in content
        for var_name, var_value in variables.items():
            placeholder = f"{{{var_name}}}"
            content = content.replace(placeholder, str(var_value))
        
        return content
    
    def increment_usage(self) -> None:
        """Increment usage count."""
        self.usage_count += 1
        self.updated_at = datetime.utcnow()
    
    def update_performance(self, engagement: float, success: bool) -> None:
        """Update performance metrics."""
        # Update success rate
        if success:
            self.success_rate = (self.success_rate * self.usage_count + 1) / (self.usage_count + 1)
        
        # Update average engagement
        self.average_engagement = (self.average_engagement * self.usage_count + engagement) / (self.usage_count + 1)
        
        self.updated_at = datetime.utcnow()
    
    def add_tag(self, tag: str) -> None:
        """Add a tag to the template."""
        if tag not in self.tags:
            self.tags.append(tag)
    
    def remove_tag(self, tag: str) -> None:
        """Remove a tag from the template."""
        if tag in self.tags:
            self.tags.remove(tag)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": str(self.id),
            "name": self.name,
            "description": self.description,
            "category": self.category.value,
            "template_type": self.template_type.value,
            "content_template": self.content_template,
            "variables": [var.to_dict() for var in self.variables],
            "tags": self.tags,
            "industry": self.industry,
            "target_audience": self.target_audience,
            "usage_count": self.usage_count,
            "success_rate": self.success_rate,
            "average_engagement": self.average_engagement,
            "is_active": self.is_active,
            "is_premium": self.is_premium,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "ai_score": self.ai_score,
            "optimization_suggestions": self.optimization_suggestions,
            "created_by": str(self.created_by) if self.created_by else None,
            "version": self.version
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Template':
        """Create from dictionary."""
        # Convert string ID to UUID
        if 'id' in data and isinstance(data['id'], str):
            data['id'] = UUID(data['id'])
        if 'created_by' in data and data['created_by'] and isinstance(data['created_by'], str):
            data['created_by'] = UUID(data['created_by'])
        
        # Convert string dates to datetime
        for date_field in ['created_at', 'updated_at']:
            if date_field in data and data[date_field]:
                if isinstance(data[date_field], str):
                    data[date_field] = datetime.fromisoformat(data[date_field])
        
        # Convert enums
        if 'category' in data and isinstance(data['category'], str):
            data['category'] = TemplateCategory(data['category'])
        if 'template_type' in data and isinstance(data['template_type'], str):
            data['template_type'] = TemplateType(data['template_type'])
        
        return cls(**data)
    
    def __str__(self) -> str:
        """String representation."""
        return f"Template(id={self.id}, name='{self.name}', category={self.category.value})"
    
    def __repr__(self) -> str:
        """Detailed representation."""
        return f"Template(id={self.id}, name='{self.name}', category={self.category.value}, usage={self.usage_count})" 