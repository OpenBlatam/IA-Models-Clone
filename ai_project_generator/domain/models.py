"""
Domain Models - Modelos de dominio
==================================

Modelos Pydantic que representan las entidades del dominio.
"""

from enum import Enum
from typing import Optional, Dict, Any, List
from datetime import datetime
from pydantic import BaseModel, Field


class ProjectStatus(str, Enum):
    """Estados de un proyecto"""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Project(BaseModel):
    """Modelo de proyecto"""
    project_id: str
    project_name: str
    description: str
    author: str
    status: ProjectStatus
    created_at: datetime
    updated_at: Optional[datetime] = None
    project_dir: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        use_enum_values = True


class ProjectRequest(BaseModel):
    """Request para crear un proyecto"""
    description: str = Field(..., min_length=10, max_length=2000)
    project_name: Optional[str] = Field(None, min_length=3, max_length=50)
    author: str = Field(default="Blatam Academy")
    version: str = Field(default="1.0.0")
    priority: int = Field(default=0, ge=-10, le=10)
    backend_framework: Optional[str] = Field(default="fastapi")
    frontend_framework: Optional[str] = Field(default="react")
    generate_tests: bool = Field(default=True)
    include_docker: bool = Field(default=True)
    include_docs: bool = Field(default=True)
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ProjectResponse(BaseModel):
    """Response de proyecto"""
    project_id: str
    status: str
    message: str
    project_info: Optional[Dict[str, Any]] = None


class GenerationTask(BaseModel):
    """Tarea de generación"""
    task_id: str
    project_request: ProjectRequest
    status: str
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class ValidationResult(BaseModel):
    """Resultado de validación"""
    valid: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    score: Optional[float] = None















