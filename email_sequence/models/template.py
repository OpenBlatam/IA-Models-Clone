from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from dataclasses import dataclass

# Constants
MAX_RETRIES = 100

from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from uuid import UUID, uuid4
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Email Template Models

This module contains models for email templates and template variables.
"""



class TemplateType(str, Enum):
    """Types of email templates"""
    WELCOME = "welcome"
    ONBOARDING = "onboarding"
    PROMOTIONAL = "promotional"
    EDUCATIONAL = "educational"
    TRANSACTIONAL = "transactional"
    NOTIFICATION = "notification"
    CUSTOM = "custom"


class VariableType(str, Enum):
    """Types of template variables"""
    TEXT = "text"
    NUMBER = "number"
    DATE = "date"
    BOOLEAN = "boolean"
    LIST = "list"
    OBJECT = "object"


class TemplateStatus(str, Enum):
    """Status of email templates"""
    DRAFT = "draft"
    ACTIVE = "active"
    ARCHIVED = "archived"
    DEPRECATED = "deprecated"


class TemplateVariable(BaseModel):
    """Model for template variables"""
    id: UUID = Field(default_factory=uuid4)
    name: str = Field(..., min_length=1, max_length=100)
    variable_type: VariableType
    description: Optional[str] = None
    default_value: Optional[Any] = None
    required: bool = False
    validation_rules: Optional[Dict[str, Any]] = None
    
    # For text variables
    max_length: Optional[int] = None
    min_length: Optional[int] = None
    pattern: Optional[str] = None
    
    # For number variables
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    
    # For list variables
    allowed_values: Optional[List[Any]] = None
    
    # For object variables
    schema: Optional[Dict[str, Any]] = None
    
    @validator('name')
    def validate_name(cls, v) -> bool:
        if not v.strip():
            raise ValueError("Variable name cannot be empty")
        # Ensure valid variable name format
        if not v.replace('_', '').isalnum():
            raise ValueError("Variable name must be alphanumeric with underscores only")
        return v.strip()
    
    @validator('max_length', 'min_length')
    def validate_length(cls, v) -> bool:
        if v is not None and v < 0:
            raise ValueError("Length must be non-negative")
        return v
    
    @validator('min_value', 'max_value')
    def validate_value(cls, v) -> bool:
        if v is not None and not isinstance(v, (int, float)):
            raise ValueError("Value must be a number")
        return v


class EmailTemplate(BaseModel):
    """Model for email templates"""
    id: UUID = Field(default_factory=uuid4)
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    template_type: TemplateType
    status: TemplateStatus = TemplateStatus.DRAFT
    
    # Content
    subject: str = Field(..., min_length=1, max_length=255)
    html_content: str = Field(..., min_length=1)
    text_content: Optional[str] = None
    preview_text: Optional[str] = Field(None, max_length=255)
    
    # Variables
    variables: List[TemplateVariable] = Field(default_factory=list)
    
    # Styling and branding
    css_styles: Optional[str] = None
    brand_colors: Optional[Dict[str, str]] = None
    logo_url: Optional[str] = None
    
    # Metadata
    tags: List[str] = Field(default_factory=list)
    category: Optional[str] = None
    version: str = "1.0.0"
    
    # Usage tracking
    usage_count: int = 0
    last_used_at: Optional[datetime] = None
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    published_at: Optional[datetime] = None
    
    # Performance metrics
    open_rate: Optional[float] = None
    click_rate: Optional[float] = None
    conversion_rate: Optional[float] = None
    
    @validator('name')
    def validate_name(cls, v) -> bool:
        if not v.strip():
            raise ValueError("Template name cannot be empty")
        return v.strip()
    
    @validator('subject')
    def validate_subject(cls, v) -> bool:
        if not v.strip():
            raise ValueError("Subject cannot be empty")
        return v.strip()
    
    @validator('html_content')
    def validate_html_content(cls, v) -> bool:
        if not v.strip():
            raise ValueError("HTML content cannot be empty")
        return v.strip()
    
    @validator('variables')
    def validate_variable_names(cls, v) -> bool:
        """Validate that variable names are unique"""
        names = [var.name for var in v]
        if len(names) != len(set(names)):
            raise ValueError("Variable names must be unique")
        return v
    
    def get_variable_by_name(self, name: str) -> Optional[TemplateVariable]:
        """Get variable by name"""
        for var in self.variables:
            if var.name == name:
                return var
        return None
    
    def add_variable(self, variable: TemplateVariable) -> None:
        """Add a new variable to the template"""
        # Check for duplicate names
        if self.get_variable_by_name(variable.name):
            raise ValueError(f"Variable '{variable.name}' already exists")
        
        self.variables.append(variable)
    
    def remove_variable(self, variable_name: str) -> bool:
        """Remove a variable from the template"""
        initial_length = len(self.variables)
        self.variables = [var for var in self.variables if var.name != variable_name]
        return len(self.variables) < initial_length
    
    def render(self, variables: Dict[str, Any]) -> Dict[str, str]:
        """Render template with provided variables"""
        rendered = {
            'subject': self.subject,
            'html_content': self.html_content,
            'text_content': self.text_content or self.html_content
        }
        
        # Replace variables in content
        for var in self.variables:
            value = variables.get(var.name, var.default_value)
            
            if var.required and value is None:
                raise ValueError(f"Required variable '{var.name}' is missing")
            
            if value is not None:
                placeholder = f"{{{{{var.name}}}}}"
                rendered['subject'] = rendered['subject'].replace(placeholder, str(value))
                rendered['html_content'] = rendered['html_content'].replace(placeholder, str(value))
                if rendered['text_content']:
                    rendered['text_content'] = rendered['text_content'].replace(placeholder, str(value))
        
        return rendered
    
    def validate_variables(self, variables: Dict[str, Any]) -> List[str]:
        """Validate provided variables against template requirements"""
        errors = []
        
        for var in self.variables:
            value = variables.get(var.name)
            
            # Check required variables
            if var.required and value is None:
                errors.append(f"Required variable '{var.name}' is missing")
                continue
            
            if value is None:
                continue
            
            # Type validation
            if var.variable_type == VariableType.TEXT:
                if not isinstance(value, str):
                    errors.append(f"Variable '{var.name}' must be a string")
                elif var.max_length and len(value) > var.max_length:
                    errors.append(f"Variable '{var.name}' exceeds maximum length of {var.max_length}")
                elif var.min_length and len(value) < var.min_length:
                    errors.append(f"Variable '{var.name}' is below minimum length of {var.min_length}")
            
            elif var.variable_type == VariableType.NUMBER:
                if not isinstance(value, (int, float)):
                    errors.append(f"Variable '{var.name}' must be a number")
                elif var.max_value is not None and value > var.max_value:
                    errors.append(f"Variable '{var.name}' exceeds maximum value of {var.max_value}")
                elif var.min_value is not None and value < var.min_value:
                    errors.append(f"Variable '{var.name}' is below minimum value of {var.min_value}")
            
            elif var.variable_type == VariableType.BOOLEAN:
                if not isinstance(value, bool):
                    errors.append(f"Variable '{var.name}' must be a boolean")
            
            elif var.variable_type == VariableType.LIST:
                if not isinstance(value, list):
                    errors.append(f"Variable '{var.name}' must be a list")
                elif var.allowed_values and not all(item in var.allowed_values for item in value):
                    errors.append(f"Variable '{var.name}' contains invalid values")
        
        return errors
    
    def publish(self) -> None:
        """Publish the template"""
        if self.status == TemplateStatus.DRAFT:
            self.status = TemplateStatus.ACTIVE
            self.published_at = datetime.utcnow()
    
    def archive(self) -> None:
        """Archive the template"""
        self.status = TemplateStatus.ARCHIVED
    
    def increment_usage(self) -> None:
        """Increment usage counter"""
        self.usage_count += 1
        self.last_used_at = datetime.utcnow()
    
    @dataclass
class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        } 