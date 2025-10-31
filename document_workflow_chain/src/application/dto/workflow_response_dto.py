"""
Workflow Response DTOs
======================

Data Transfer Objects for workflow responses.
"""

from __future__ import annotations
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime

from .node_dto import NodeResponseDTO


class WorkflowResponseDTO(BaseModel):
    """Workflow response DTO"""
    id: str = Field(..., description="Unique identifier of the workflow")
    name: str = Field(..., description="Name of the workflow")
    description: Optional[str] = Field(None, description="Description of the workflow")
    status: str = Field(..., description="Current status of the workflow")
    settings: Dict[str, Any] = Field(default_factory=dict, description="Workflow settings")
    created_at: Optional[str] = Field(None, description="Creation timestamp")
    updated_at: Optional[str] = Field(None, description="Last update timestamp")
    version: int = Field(..., description="Version number for optimistic locking")
    nodes: List[NodeResponseDTO] = Field(default_factory=list, description="List of workflow nodes")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class WorkflowListResponseDTO(BaseModel):
    """Workflow list response DTO"""
    workflows: List[WorkflowResponseDTO] = Field(..., description="List of workflows")
    total_count: int = Field(..., description="Total number of workflows")
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Number of items per page")
    has_next: bool = Field(..., description="Whether there are more pages")
    has_previous: bool = Field(..., description="Whether there are previous pages")


class WorkflowStatisticsResponseDTO(BaseModel):
    """Workflow statistics response DTO"""
    workflow_id: str = Field(..., description="Workflow identifier")
    statistics: Dict[str, Any] = Field(..., description="Workflow statistics")
    health_score: float = Field(..., description="Workflow health score (0-100)")
    complexity_score: float = Field(..., description="Workflow complexity score (0-100)")
    calculated_at: str = Field(..., description="When statistics were calculated")


class WorkflowHealthResponseDTO(BaseModel):
    """Workflow health response DTO"""
    workflow_id: str = Field(..., description="Workflow identifier")
    status: str = Field(..., description="Overall health status")
    health_score: float = Field(..., description="Health score (0-100)")
    issues: List[str] = Field(default_factory=list, description="List of health issues")
    recommendations: List[str] = Field(default_factory=list, description="Health improvement recommendations")
    checked_at: str = Field(..., description="When health was checked")


class WorkflowExportResponseDTO(BaseModel):
    """Workflow export response DTO"""
    workflow_id: str = Field(..., description="Workflow identifier")
    export_format: str = Field(..., description="Export format (json, yaml, xml)")
    content: str = Field(..., description="Exported content")
    exported_at: str = Field(..., description="When export was created")
    file_size: int = Field(..., description="Size of exported content in bytes")


class WorkflowImportResponseDTO(BaseModel):
    """Workflow import response DTO"""
    imported_workflow_id: str = Field(..., description="ID of the imported workflow")
    imported_nodes_count: int = Field(..., description="Number of imported nodes")
    warnings: List[str] = Field(default_factory=list, description="Import warnings")
    errors: List[str] = Field(default_factory=list, description="Import errors")
    imported_at: str = Field(..., description="When import was completed")


class WorkflowTemplateResponseDTO(BaseModel):
    """Workflow template response DTO"""
    template_id: str = Field(..., description="Template identifier")
    name: str = Field(..., description="Template name")
    description: Optional[str] = Field(None, description="Template description")
    category: str = Field(..., description="Template category")
    tags: List[str] = Field(default_factory=list, description="Template tags")
    workflow_structure: Dict[str, Any] = Field(..., description="Template workflow structure")
    created_at: str = Field(..., description="Template creation timestamp")
    usage_count: int = Field(..., description="Number of times template was used")
    rating: Optional[float] = Field(None, description="Template rating (1-5)")


class WorkflowCollaborationResponseDTO(BaseModel):
    """Workflow collaboration response DTO"""
    workflow_id: str = Field(..., description="Workflow identifier")
    collaborators: List[Dict[str, Any]] = Field(..., description="List of collaborators")
    permissions: Dict[str, List[str]] = Field(..., description="Permission mapping")
    shared_at: str = Field(..., description="When workflow was shared")
    last_activity: Optional[str] = Field(None, description="Last collaboration activity")


class WorkflowVersionResponseDTO(BaseModel):
    """Workflow version response DTO"""
    workflow_id: str = Field(..., description="Workflow identifier")
    version: int = Field(..., description="Version number")
    changes: List[str] = Field(..., description="List of changes in this version")
    created_by: str = Field(..., description="User who created this version")
    created_at: str = Field(..., description="Version creation timestamp")
    is_current: bool = Field(..., description="Whether this is the current version")


class WorkflowBackupResponseDTO(BaseModel):
    """Workflow backup response DTO"""
    backup_id: str = Field(..., description="Backup identifier")
    workflow_id: str = Field(..., description="Workflow identifier")
    backup_type: str = Field(..., description="Type of backup (manual, scheduled)")
    size_bytes: int = Field(..., description="Backup size in bytes")
    created_at: str = Field(..., description="Backup creation timestamp")
    expires_at: Optional[str] = Field(None, description="Backup expiration timestamp")
    status: str = Field(..., description="Backup status")


class WorkflowAuditResponseDTO(BaseModel):
    """Workflow audit response DTO"""
    audit_id: str = Field(..., description="Audit entry identifier")
    workflow_id: str = Field(..., description="Workflow identifier")
    action: str = Field(..., description="Action performed")
    user_id: Optional[str] = Field(None, description="User who performed the action")
    timestamp: str = Field(..., description="When action was performed")
    details: Dict[str, Any] = Field(default_factory=dict, description="Action details")
    ip_address: Optional[str] = Field(None, description="IP address of the user")
    user_agent: Optional[str] = Field(None, description="User agent string")


class WorkflowMetricsResponseDTO(BaseModel):
    """Workflow metrics response DTO"""
    workflow_id: str = Field(..., description="Workflow identifier")
    metrics: Dict[str, Any] = Field(..., description="Workflow metrics")
    time_range: str = Field(..., description="Time range for metrics")
    generated_at: str = Field(..., description="When metrics were generated")


class WorkflowSearchResponseDTO(BaseModel):
    """Workflow search response DTO"""
    query: str = Field(..., description="Search query")
    results: List[WorkflowResponseDTO] = Field(..., description="Search results")
    total_results: int = Field(..., description="Total number of results")
    search_time_ms: int = Field(..., description="Search execution time in milliseconds")
    facets: Dict[str, Any] = Field(default_factory=dict, description="Search facets")
    suggestions: List[str] = Field(default_factory=list, description="Search suggestions")




