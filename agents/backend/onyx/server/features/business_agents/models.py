"""
Business Agents Models
======================

Pydantic models for request/response validation and serialization.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum

# Enums
class BusinessArea(str, Enum):
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

class DocumentType(str, Enum):
    STRATEGY = "strategy"
    PROPOSAL = "proposal"
    REPORT = "report"
    MANUAL = "manual"
    POLICY = "policy"
    PROCEDURE = "procedure"
    TEMPLATE = "template"
    PRESENTATION = "presentation"
    CONTRACT = "contract"
    ANALYSIS = "analysis"
    PLAN = "plan"
    BRIEF = "brief"
    CUSTOM = "custom"

class DocumentFormat(str, Enum):
    MARKDOWN = "markdown"
    HTML = "html"
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    JSON = "json"
    XML = "xml"

class WorkflowStatus(str, Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"

class StepType(str, Enum):
    TASK = "task"
    DECISION = "decision"
    PARALLEL = "parallel"
    SEQUENTIAL = "sequential"
    CONDITIONAL = "conditional"
    LOOP = "loop"
    API_CALL = "api_call"
    DOCUMENT_GENERATION = "document_generation"
    NOTIFICATION = "notification"
    WAIT = "wait"

# Agent Models
class AgentCapability(BaseModel):
    name: str = Field(..., description="Capability name")
    description: str = Field(..., description="Capability description")
    input_types: List[str] = Field(..., description="Expected input types")
    output_types: List[str] = Field(..., description="Expected output types")
    estimated_duration: int = Field(..., description="Estimated duration in seconds")

class AgentResponse(BaseModel):
    id: str = Field(..., description="Agent ID")
    name: str = Field(..., description="Agent name")
    business_area: str = Field(..., description="Business area")
    description: str = Field(..., description="Agent description")
    capabilities: List[AgentCapability] = Field(..., description="Agent capabilities")
    is_active: bool = Field(..., description="Whether agent is active")
    created_at: str = Field(..., description="Creation timestamp")
    updated_at: str = Field(..., description="Last update timestamp")

class CapabilityResponse(BaseModel):
    name: str = Field(..., description="Capability name")
    description: str = Field(..., description="Capability description")
    input_types: List[str] = Field(..., description="Expected input types")
    output_types: List[str] = Field(..., description="Expected output types")
    estimated_duration: int = Field(..., description="Estimated duration in seconds")

class ExecutionRequest(BaseModel):
    inputs: Dict[str, Any] = Field(..., description="Input data for execution")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Additional parameters")

class ExecutionResponse(BaseModel):
    status: str = Field(..., description="Execution status")
    agent_id: str = Field(..., description="Agent ID")
    capability: str = Field(..., description="Capability name")
    result: Optional[Dict[str, Any]] = Field(None, description="Execution result")
    error: Optional[str] = Field(None, description="Error message if failed")
    execution_time: Optional[int] = Field(None, description="Execution time in seconds")

# Workflow Models
class WorkflowStepRequest(BaseModel):
    name: str = Field(..., description="Step name")
    step_type: StepType = Field(..., description="Type of step")
    description: str = Field(..., description="Step description")
    agent_type: Optional[str] = Field(None, description="Agent type for the step")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Step parameters")
    dependencies: List[str] = Field(default_factory=list, description="Step dependencies")

class WorkflowCreateRequest(BaseModel):
    name: str = Field(..., description="Workflow name")
    description: str = Field(..., description="Workflow description")
    business_area: BusinessArea = Field(..., description="Business area")
    steps: List[WorkflowStepRequest] = Field(..., description="Workflow steps")
    variables: Optional[Dict[str, Any]] = Field(None, description="Workflow variables")

class WorkflowResponse(BaseModel):
    id: str = Field(..., description="Workflow ID")
    name: str = Field(..., description="Workflow name")
    description: str = Field(..., description="Workflow description")
    business_area: str = Field(..., description="Business area")
    status: str = Field(..., description="Workflow status")
    created_by: str = Field(..., description="Created by user")
    created_at: str = Field(..., description="Creation timestamp")
    updated_at: str = Field(..., description="Last update timestamp")
    steps: List[Dict[str, Any]] = Field(..., description="Workflow steps")
    variables: Dict[str, Any] = Field(..., description="Workflow variables")
    execution_results: Optional[Dict[str, Any]] = Field(None, description="Execution results")

class WorkflowExecutionResponse(BaseModel):
    workflow_id: str = Field(..., description="Workflow ID")
    status: str = Field(..., description="Execution status")
    execution_id: str = Field(..., description="Execution ID")
    started_at: str = Field(..., description="Start timestamp")
    estimated_completion: Optional[str] = Field(None, description="Estimated completion time")
    current_step: Optional[str] = Field(None, description="Current executing step")
    progress_percentage: Optional[float] = Field(None, description="Progress percentage")
    results: Optional[Dict[str, Any]] = Field(None, description="Execution results")
    error: Optional[str] = Field(None, description="Error message if failed")

# Document Models
class DocumentGenerationRequest(BaseModel):
    document_type: DocumentType = Field(..., description="Type of document to generate")
    title: str = Field(..., description="Document title")
    description: str = Field(..., description="Document description")
    business_area: str = Field(..., description="Business area")
    variables: Optional[Dict[str, Any]] = Field(None, description="Document variables")
    format: DocumentFormat = Field(DocumentFormat.MARKDOWN, description="Output format")

class DocumentGenerationResponse(BaseModel):
    document_id: str = Field(..., description="Document ID")
    request_id: str = Field(..., description="Request ID")
    title: str = Field(..., description="Document title")
    file_path: str = Field(..., description="File path")
    format: str = Field(..., description="Document format")
    size_bytes: int = Field(..., description="File size in bytes")
    created_at: str = Field(..., description="Creation timestamp")
    status: str = Field("generated", description="Generation status")

class DocumentRequest(BaseModel):
    document_type: DocumentType = Field(..., description="Type of document")
    title: str = Field(..., description="Document title")
    description: str = Field(..., description="Document description")
    business_area: str = Field(..., description="Business area")
    created_by: str = Field(..., description="Created by user")
    variables: Optional[Dict[str, Any]] = Field(None, description="Document variables")
    format: DocumentFormat = Field(DocumentFormat.MARKDOWN, description="Output format")

# System Models
class SystemInfoResponse(BaseModel):
    name: str = Field(..., description="System name")
    version: str = Field(..., description="System version")
    description: str = Field(..., description="System description")
    features: List[str] = Field(..., description="System features")
    business_areas: List[str] = Field(..., description="Supported business areas")
    endpoints: Dict[str, str] = Field(..., description="Available endpoints")

class HealthResponse(BaseModel):
    status: str = Field(..., description="Health status")
    timestamp: str = Field(..., description="Check timestamp")
    version: str = Field(..., description="System version")
    components: Dict[str, str] = Field(..., description="Component health status")
    metrics: Dict[str, Any] = Field(..., description="System metrics")

class MetricsResponse(BaseModel):
    agents: Dict[str, Any] = Field(..., description="Agent metrics")
    workflows: Dict[str, Any] = Field(..., description="Workflow metrics")
    documents: Dict[str, Any] = Field(..., description="Document metrics")
    system: Dict[str, Any] = Field(..., description="System metrics")

# List Response Models
class AgentListResponse(BaseModel):
    agents: List[AgentResponse] = Field(..., description="List of agents")
    total: int = Field(..., description="Total number of agents")
    business_areas: List[str] = Field(..., description="Available business areas")

class WorkflowListResponse(BaseModel):
    workflows: List[WorkflowResponse] = Field(..., description="List of workflows")
    total: int = Field(..., description="Total number of workflows")

class DocumentListResponse(BaseModel):
    documents: List[DocumentGenerationResponse] = Field(..., description="List of documents")
    total: int = Field(..., description="Total number of documents")

# Template Models
class DocumentTemplate(BaseModel):
    name: str = Field(..., description="Template name")
    description: str = Field(..., description="Template description")
    document_type: DocumentType = Field(..., description="Document type")
    sections: List[str] = Field(..., description="Template sections")
    variables: Optional[Dict[str, Any]] = Field(None, description="Template variables")

class WorkflowTemplate(BaseModel):
    name: str = Field(..., description="Template name")
    description: str = Field(..., description="Template description")
    business_area: BusinessArea = Field(..., description="Business area")
    steps: List[WorkflowStepRequest] = Field(..., description="Template steps")
    estimated_duration: Optional[int] = Field(None, description="Estimated duration")

# Error Models
class ErrorResponse(BaseModel):
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Error detail")
    code: Optional[str] = Field(None, description="Error code")
    timestamp: str = Field(..., description="Error timestamp")

class ValidationError(BaseModel):
    field: str = Field(..., description="Field name")
    message: str = Field(..., description="Validation message")
    value: Any = Field(..., description="Invalid value")

class ValidationErrorResponse(BaseModel):
    errors: List[ValidationError] = Field(..., description="Validation errors")
    message: str = Field(..., description="General error message")