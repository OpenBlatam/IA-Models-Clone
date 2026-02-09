"""
Generator API - Backward Compatibility Wrapper
===============================================

This module provides backward compatibility for the legacy generator_api interface.
All new code should use the modular structure in api.app_factory and api.routes.

For new implementations, use:
    from api.app_factory import create_app
    
For domain models, use:
    from domain.models import ProjectRequest, ProjectResponse
"""

import logging
import re
from typing import Optional, List, Dict, Any
from fastapi import FastAPI
from pydantic import BaseModel, Field, validator

from ..domain.models import ProjectRequest as DomainProjectRequest, ProjectResponse as DomainProjectResponse

logger = logging.getLogger(__name__)


class ProjectRequest(DomainProjectRequest):
    """
    Request model for generating a project.
    
    DEPRECATED: Use domain.models.ProjectRequest instead.
    This class extends the domain model with additional fields for backward compatibility.
    """
    include_cicd: bool = Field(True, description="Include CI/CD pipelines")
    create_github_repo: bool = Field(False, description="Create GitHub repository automatically")
    github_token: Optional[str] = Field(None, description="GitHub token (if create_github_repo=True)")
    github_private: bool = Field(False, description="Private GitHub repository")
    
    @validator('description')
    def validate_description(cls, v):
        if not v or not v.strip():
            raise ValueError("La descripción no puede estar vacía")
        if len(set(v.split())) < 5:
            raise ValueError("La descripción debe tener al menos 5 palabras únicas")
        return v.strip()
    
    @validator('project_name')
    def validate_project_name(cls, v):
        if v:
            if not re.match(r'^[a-zA-Z0-9_-]+$', v):
                raise ValueError("El nombre del proyecto solo puede contener letras, números, guiones y guiones bajos")
        return v


class BatchProjectRequest(BaseModel):
    """Request for generating multiple projects in batch"""
    projects: List[ProjectRequest] = Field(..., min_items=1, max_items=50, description="List of projects to generate")
    parallel: bool = Field(True, description="Generate projects in parallel")
    stop_on_error: bool = Field(False, description="Stop on error")


class ProjectResponse(DomainProjectResponse):
    """
    Response model for project generation.
    
    DEPRECATED: Use domain.models.ProjectResponse instead.
    This class is kept for backward compatibility.
    """
    pass


def create_generator_app(
    base_output_dir: str = "generated_projects",
    enable_continuous: bool = True,
) -> FastAPI:
    """
    Creates the FastAPI application for the project generator.
    
    NOTE: This function maintains backward compatibility.
    For new modular structure, use api.app_factory.create_app()
    
    Args:
        base_output_dir: Base directory for generated projects
        enable_continuous: Enable continuous generation
    
    Returns:
        Configured FastAPI application
    """
    from .app_factory import create_app
    return create_app(
        base_output_dir=base_output_dir,
        enable_continuous=enable_continuous
    )
