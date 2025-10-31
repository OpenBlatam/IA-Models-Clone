"""
Node Response DTOs
==================

Data Transfer Objects for node responses.
"""

from __future__ import annotations
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class NodeResponseDTO(BaseModel):
    """Node response DTO"""
    id: str = Field(..., description="Unique identifier of the node")
    workflow_id: str = Field(..., description="ID of the parent workflow")
    title: str = Field(..., description="Title of the node")
    content: str = Field(..., description="Content of the node")
    prompt: str = Field(..., description="Prompt used to generate the node")
    parent_id: Optional[str] = Field(None, description="ID of the parent node")
    priority: int = Field(..., description="Priority level (1-5)")
    status: str = Field(..., description="Current status of the node")
    tags: List[str] = Field(default_factory=list, description="Node tags")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Node metadata")
    created_at: Optional[str] = Field(None, description="Creation timestamp")
    updated_at: Optional[str] = Field(None, description="Last update timestamp")
    version: int = Field(..., description="Version number for optimistic locking")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class NodeListResponseDTO(BaseModel):
    """Node list response DTO"""
    nodes: List[NodeResponseDTO] = Field(..., description="List of nodes")
    total_count: int = Field(..., description="Total number of nodes")
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Number of items per page")
    has_next: bool = Field(..., description="Whether there are more pages")
    has_previous: bool = Field(..., description="Whether there are previous pages")


class NodeContentAnalysisResponseDTO(BaseModel):
    """Node content analysis response DTO"""
    node_id: str = Field(..., description="Node identifier")
    analysis_type: str = Field(..., description="Type of analysis performed")
    results: Dict[str, Any] = Field(..., description="Analysis results")
    confidence_score: float = Field(..., description="Confidence score (0-1)")
    analyzed_at: str = Field(..., description="When analysis was performed")
    processing_time_ms: int = Field(..., description="Analysis processing time")


class NodeQualityScoreResponseDTO(BaseModel):
    """Node quality score response DTO"""
    node_id: str = Field(..., description="Node identifier")
    overall_score: int = Field(..., description="Overall quality score (0-100)")
    readability_score: int = Field(..., description="Readability score (0-100)")
    sentiment_score: int = Field(..., description="Sentiment score (-100 to 100)")
    seo_score: int = Field(..., description="SEO score (0-100)")
    grammar_score: int = Field(..., description="Grammar score (0-100)")
    coherence_score: int = Field(..., description="Coherence score (0-100)")
    calculated_at: str = Field(..., description="When scores were calculated")
    recommendations: List[str] = Field(default_factory=list, description="Improvement recommendations")


class NodeMetricsResponseDTO(BaseModel):
    """Node metrics response DTO"""
    node_id: str = Field(..., description="Node identifier")
    word_count: int = Field(..., description="Number of words")
    character_count: int = Field(..., description="Number of characters")
    sentence_count: int = Field(..., description="Number of sentences")
    paragraph_count: int = Field(..., description="Number of paragraphs")
    reading_time_minutes: int = Field(..., description="Estimated reading time in minutes")
    calculated_at: str = Field(..., description="When metrics were calculated")


class NodeVersionResponseDTO(BaseModel):
    """Node version response DTO"""
    node_id: str = Field(..., description="Node identifier")
    version: int = Field(..., description="Version number")
    changes: List[str] = Field(..., description="List of changes in this version")
    created_by: str = Field(..., description="User who created this version")
    created_at: str = Field(..., description="Version creation timestamp")
    is_current: bool = Field(..., description="Whether this is the current version")


class NodeRelationshipResponseDTO(BaseModel):
    """Node relationship response DTO"""
    node_id: str = Field(..., description="Node identifier")
    relationships: Dict[str, List[str]] = Field(..., description="Node relationships")
    hierarchy_level: int = Field(..., description="Level in the hierarchy")
    parent_path: List[str] = Field(..., description="Path to root node")
    children_count: int = Field(..., description="Number of child nodes")
    siblings_count: int = Field(..., description="Number of sibling nodes")


class NodeSearchResponseDTO(BaseModel):
    """Node search response DTO"""
    query: str = Field(..., description="Search query")
    results: List[NodeResponseDTO] = Field(..., description="Search results")
    total_results: int = Field(..., description="Total number of results")
    search_time_ms: int = Field(..., description="Search execution time in milliseconds")
    facets: Dict[str, Any] = Field(default_factory=dict, description="Search facets")
    highlights: Dict[str, List[str]] = Field(default_factory=dict, description="Search highlights")


class NodeExportResponseDTO(BaseModel):
    """Node export response DTO"""
    node_id: str = Field(..., description="Node identifier")
    export_format: str = Field(..., description="Export format (json, yaml, xml, markdown)")
    content: str = Field(..., description="Exported content")
    exported_at: str = Field(..., description="When export was created")
    file_size: int = Field(..., description="Size of exported content in bytes")


class NodeTemplateResponseDTO(BaseModel):
    """Node template response DTO"""
    template_id: str = Field(..., description="Template identifier")
    name: str = Field(..., description="Template name")
    description: Optional[str] = Field(None, description="Template description")
    category: str = Field(..., description="Template category")
    tags: List[str] = Field(default_factory=list, description="Template tags")
    template_structure: Dict[str, Any] = Field(..., description="Template structure")
    created_at: str = Field(..., description="Template creation timestamp")
    usage_count: int = Field(..., description="Number of times template was used")
    rating: Optional[float] = Field(None, description="Template rating (1-5)")


class NodeCollaborationResponseDTO(BaseModel):
    """Node collaboration response DTO"""
    node_id: str = Field(..., description="Node identifier")
    collaborators: List[Dict[str, Any]] = Field(..., description="List of collaborators")
    permissions: Dict[str, List[str]] = Field(..., description="Permission mapping")
    shared_at: str = Field(..., description="When node was shared")
    last_activity: Optional[str] = Field(None, description="Last collaboration activity")


class NodeAuditResponseDTO(BaseModel):
    """Node audit response DTO"""
    audit_id: str = Field(..., description="Audit entry identifier")
    node_id: str = Field(..., description="Node identifier")
    action: str = Field(..., description="Action performed")
    user_id: Optional[str] = Field(None, description="User who performed the action")
    timestamp: str = Field(..., description="When action was performed")
    details: Dict[str, Any] = Field(default_factory=dict, description="Action details")
    ip_address: Optional[str] = Field(None, description="IP address of the user")
    user_agent: Optional[str] = Field(None, description="User agent string")


class NodeBackupResponseDTO(BaseModel):
    """Node backup response DTO"""
    backup_id: str = Field(..., description="Backup identifier")
    node_id: str = Field(..., description="Node identifier")
    backup_type: str = Field(..., description="Type of backup (manual, scheduled)")
    size_bytes: int = Field(..., description="Backup size in bytes")
    created_at: str = Field(..., description="Backup creation timestamp")
    expires_at: Optional[str] = Field(None, description="Backup expiration timestamp")
    status: str = Field(..., description="Backup status")


class NodeWorkflowResponseDTO(BaseModel):
    """Node workflow response DTO"""
    node_id: str = Field(..., description="Node identifier")
    workflow_id: str = Field(..., description="Workflow identifier")
    workflow_name: str = Field(..., description="Workflow name")
    workflow_status: str = Field(..., description="Workflow status")
    position_in_workflow: int = Field(..., description="Position in workflow")
    dependencies: List[str] = Field(default_factory=list, description="Node dependencies")
    dependents: List[str] = Field(default_factory=list, description="Nodes that depend on this node")




