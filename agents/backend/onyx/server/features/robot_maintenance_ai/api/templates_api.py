"""
Templates API for maintenance procedure templates and checklists.
Refactored to use BaseRouter for reduced duplication.
"""

from fastapi import Depends, Query
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime

from .base_router import BaseRouter
from .exceptions import NotFoundError
from ..utils.file_helpers import get_iso_timestamp, get_timestamp_id, create_resource, update_resource
from ..utils.data_helpers import filter_by_fields, filter_by_field_contains, ensure_resource_exists, sort_by_field

# Create base router instance
base = BaseRouter(
    prefix="/api/templates",
    tags=["Templates"],
    require_authentication=True,
    require_rate_limit=False
)

router = base.router


class TemplateRequest(BaseModel):
    """Request to create/update a template."""
    name: str = Field(..., min_length=3, max_length=200, description="Template name")
    robot_type: str = Field(..., description="Robot type this template applies to")
    maintenance_type: str = Field(..., description="Type of maintenance")
    difficulty: str = Field("intermedio", description="Difficulty level")
    steps: List[Dict[str, Any]] = Field(..., description="List of steps in the template")
    description: Optional[str] = Field(None, description="Template description")
    tags: Optional[List[str]] = Field(None, description="Tags for categorization")


# In-memory storage (would be database in production)
templates_store: Dict[str, Dict[str, Any]] = {}


@router.post("/create")
@base.timed_endpoint("create_template")
async def create_template(
    request: TemplateRequest,
    _: Dict = Depends(base.get_auth_dependency())
) -> Dict[str, Any]:
    """
    Create a new maintenance template.
    """
    base.log_request("create_template", template_name=request.name)
    
    template = create_resource(
        {
            "name": request.name,
            "robot_type": request.robot_type,
            "maintenance_type": request.maintenance_type,
            "difficulty": request.difficulty,
            "steps": request.steps,
            "description": request.description,
            "tags": request.tags or [],
            "usage_count": 0
        },
        id_prefix="template_"
    )
    
    templates_store[template["id"]] = template
    
    return base.success(template, message="Template created successfully")


@router.get("/list")
@base.timed_endpoint("list_templates")
async def list_templates(
    robot_type: Optional[str] = Query(None, description="Filter by robot type"),
    maintenance_type: Optional[str] = Query(None, description="Filter by maintenance type"),
    difficulty: Optional[str] = Query(None, description="Filter by difficulty"),
    tag: Optional[str] = Query(None, description="Filter by tag"),
    _: Dict = Depends(base.get_auth_dependency())
) -> Dict[str, Any]:
    """
    List templates with optional filters.
    """
    base.log_request("list_templates", robot_type=robot_type, maintenance_type=maintenance_type)
    
    templates = list(templates_store.values())
    
    # Apply standard filters using helper
    templates = filter_by_fields(
        templates,
        {
            "robot_type": robot_type,
            "maintenance_type": maintenance_type,
            "difficulty": difficulty
        }
    )
    
    # Apply tag filter (special case: value in list)
    if tag:
        templates = filter_by_field_contains(templates, "tags", tag)
    
    return base.success({
        "templates": templates,
        "total": len(templates),
        "filters": {
            "robot_type": robot_type,
            "maintenance_type": maintenance_type,
            "difficulty": difficulty,
            "tag": tag
        }
    })


@router.get("/{template_id}")
@base.timed_endpoint("get_template")
async def get_template(
    template_id: str,
    _: Dict = Depends(base.get_auth_dependency())
) -> Dict[str, Any]:
    """
    Get a specific template.
    """
    base.log_request("get_template", template_id=template_id)
    
    ensure_resource_exists(template_id, templates_store, "Template")
    template = templates_store[template_id]
    template["usage_count"] += 1
    
    return base.success(template)


@router.put("/{template_id}")
@base.timed_endpoint("update_template")
async def update_template(
    template_id: str,
    request: TemplateRequest,
    _: Dict = Depends(base.get_auth_dependency())
) -> Dict[str, Any]:
    """
    Update an existing template.
    """
    base.log_request("update_template", template_id=template_id)
    
    ensure_resource_exists(template_id, templates_store, "Template")
    template = templates_store[template_id]
    template = update_resource(
        template,
        {
            "name": request.name,
            "robot_type": request.robot_type,
            "maintenance_type": request.maintenance_type,
            "difficulty": request.difficulty,
            "steps": request.steps,
            "description": request.description,
            "tags": request.tags or []
        }
    )
    templates_store[template_id] = template
    
    return base.success(template, message="Template updated successfully")


@router.delete("/{template_id}")
@base.timed_endpoint("delete_template")
async def delete_template(
    template_id: str,
    _: Dict = Depends(base.get_auth_dependency())
) -> Dict[str, Any]:
    """
    Delete a template.
    """
    base.log_request("delete_template", template_id=template_id)
    
    ensure_resource_exists(template_id, templates_store, "Template")
    del templates_store[template_id]
    
    return base.success(None, message=f"Template {template_id} deleted successfully")


@router.post("/{template_id}/use")
@base.timed_endpoint("use_template")
async def use_template(
    template_id: str,
    customizations: Optional[Dict[str, Any]] = Field(None, description="Customizations to apply"),
    _: Dict = Depends(base.get_auth_dependency())
) -> Dict[str, Any]:
    """
    Use a template and optionally customize it.
    """
    base.log_request("use_template", template_id=template_id)
    
    ensure_resource_exists(template_id, templates_store, "Template")
    template = templates_store[template_id].copy()
    template["usage_count"] += 1
    
    # Apply customizations
    if customizations:
        if "steps" in customizations:
            template["steps"] = customizations["steps"]
        if "description" in customizations:
            template["description"] = customizations["description"]
    
    return base.success({
        "template": template,
        "customized": customizations is not None
    }, message="Template applied successfully")


@router.get("/popular")
@base.timed_endpoint("get_popular_templates")
async def get_popular_templates(
    limit: int = Query(10, ge=1, le=50, description="Number of templates"),
    _: Dict = Depends(base.get_auth_dependency())
) -> Dict[str, Any]:
    """
    Get most popular templates by usage count.
    """
    base.log_request("get_popular_templates", limit=limit)
    
    templates = list(templates_store.values())
    templates = sort_by_field(templates, "usage_count", reverse=True, default_value=0)
    
    return base.success({
        "templates": templates[:limit],
        "total": len(templates)
    })




