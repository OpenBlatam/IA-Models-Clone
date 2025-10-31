"""
GraphQL Types
=============

GraphQL type definitions for the Business Agents System.
"""

import graphene
from graphene import ObjectType, String, Int, Float, Boolean, List, Field, InputObjectType
from graphene.types.datetime import DateTime
from typing import Optional, Dict, Any

class AgentCapabilityType(ObjectType):
    """GraphQL type for agent capabilities."""
    name = String(required=True)
    description = String(required=True)
    input_types = List(String, required=True)
    output_types = List(String, required=True)
    estimated_duration = Int(required=True)
    required_permissions = List(String)
    tags = List(String)

class AgentType(ObjectType):
    """GraphQL type for business agents."""
    id = String(required=True)
    name = String(required=True)
    business_area = String(required=True)
    description = String(required=True)
    capabilities = List(AgentCapabilityType, required=True)
    is_active = Boolean(required=True)
    created_at = DateTime(required=True)
    updated_at = DateTime(required=True)
    metadata = graphene.JSONString()
    version = String()
    author = String()
    dependencies = List(String)

class WorkflowStepType(ObjectType):
    """GraphQL type for workflow steps."""
    id = String(required=True)
    name = String(required=True)
    step_type = String(required=True)
    description = String(required=True)
    agent_type = String(required=True)
    parameters = graphene.JSONString()
    conditions = graphene.JSONString()
    next_steps = List(String)
    parallel_steps = List(String)
    max_retries = Int(required=True)
    timeout = Int(required=True)
    status = String(required=True)
    created_at = DateTime()
    completed_at = DateTime()
    error_message = String()

class WorkflowType(ObjectType):
    """GraphQL type for workflows."""
    id = String(required=True)
    name = String(required=True)
    description = String(required=True)
    business_area = String(required=True)
    status = String(required=True)
    created_by = String(required=True)
    created_at = DateTime(required=True)
    updated_at = DateTime(required=True)
    variables = graphene.JSONString()
    metadata = graphene.JSONString()
    steps = List(WorkflowStepType, required=True)

class DocumentType(ObjectType):
    """GraphQL type for documents."""
    id = String(required=True)
    request_id = String(required=True)
    title = String(required=True)
    content = String()
    file_path = String()
    format = String(required=True)
    size_bytes = Int()
    created_at = DateTime(required=True)
    completed_at = DateTime()
    error_message = String()
    metadata = graphene.JSONString()

class SystemMetricsType(ObjectType):
    """GraphQL type for system metrics."""
    agents = graphene.JSONString(required=True)
    workflows = graphene.JSONString(required=True)
    business_areas = graphene.JSONString(required=True)
    timestamp = DateTime(required=True)

class SystemInfoType(ObjectType):
    """GraphQL type for system information."""
    system = graphene.JSONString(required=True)
    capabilities = graphene.JSONString(required=True)
    business_areas = graphene.JSONString(required=True)
    workflow_templates = graphene.JSONString(required=True)
    configuration = graphene.JSONString(required=True)

class HealthStatusType(ObjectType):
    """GraphQL type for health status."""
    status = String(required=True)
    timestamp = DateTime(required=True)
    version = String(required=True)
    components = graphene.JSONString(required=True)
    metrics = graphene.JSONString(required=True)

# Input Types
class AgentCapabilityInput(InputObjectType):
    """Input type for agent capabilities."""
    name = String(required=True)
    description = String(required=True)
    input_types = List(String, required=True)
    output_types = List(String, required=True)
    parameters = graphene.JSONString()
    estimated_duration = Int(default_value=300)
    required_permissions = List(String)
    tags = List(String)

class AgentInput(InputObjectType):
    """Input type for creating agents."""
    name = String(required=True)
    business_area = String(required=True)
    description = String(required=True)
    capabilities = List(AgentCapabilityInput, required=True)
    is_active = Boolean(default_value=True)
    metadata = graphene.JSONString()

class WorkflowStepInput(InputObjectType):
    """Input type for workflow steps."""
    name = String(required=True)
    step_type = String(required=True)
    description = String(required=True)
    agent_type = String(required=True)
    parameters = graphene.JSONString()
    conditions = graphene.JSONString()
    next_steps = List(String)
    parallel_steps = List(String)
    max_retries = Int(default_value=3)
    timeout = Int(default_value=300)

class WorkflowInput(InputObjectType):
    """Input type for creating workflows."""
    name = String(required=True)
    description = String(required=True)
    business_area = String(required=True)
    steps = List(WorkflowStepInput, required=True)
    variables = graphene.JSONString()
    metadata = graphene.JSONString()

class DocumentInput(InputObjectType):
    """Input type for document generation."""
    document_type = String(required=True)
    title = String(required=True)
    description = String(required=True)
    business_area = String(required=True)
    variables = graphene.JSONString()
    template_id = String()
    format = String(default_value="markdown")
    priority = String(default_value="normal")

class CapabilityExecutionInput(InputObjectType):
    """Input type for capability execution."""
    agent_id = String(required=True)
    capability_name = String(required=True)
    inputs = graphene.JSONString(required=True)
    parameters = graphene.JSONString()

# Pagination Types
class PageInfo(ObjectType):
    """GraphQL type for pagination info."""
    has_next_page = Boolean(required=True)
    has_previous_page = Boolean(required=True)
    start_cursor = String()
    end_cursor = String()

class AgentConnection(ObjectType):
    """GraphQL connection type for agents."""
    edges = List(lambda: AgentEdge)
    page_info = Field(PageInfo, required=True)
    total_count = Int(required=True)

class AgentEdge(ObjectType):
    """GraphQL edge type for agents."""
    node = Field(AgentType, required=True)
    cursor = String(required=True)

class WorkflowConnection(ObjectType):
    """GraphQL connection type for workflows."""
    edges = List(lambda: WorkflowEdge)
    page_info = Field(PageInfo, required=True)
    total_count = Int(required=True)

class WorkflowEdge(ObjectType):
    """GraphQL edge type for workflows."""
    node = Field(WorkflowType, required=True)
    cursor = String(required=True)

class DocumentConnection(ObjectType):
    """GraphQL connection type for documents."""
    edges = List(lambda: DocumentEdge)
    page_info = Field(PageInfo, required=True)
    total_count = Int(required=True)

class DocumentEdge(ObjectType):
    """GraphQL edge type for documents."""
    node = Field(DocumentType, required=True)
    cursor = String(required=True)

# Subscription Types
class AgentExecutionUpdate(ObjectType):
    """GraphQL type for agent execution updates."""
    agent_id = String(required=True)
    capability_name = String(required=True)
    status = String(required=True)
    progress = Float()
    result = graphene.JSONString()
    error = String()
    timestamp = DateTime(required=True)

class WorkflowExecutionUpdate(ObjectType):
    """GraphQL type for workflow execution updates."""
    workflow_id = String(required=True)
    status = String(required=True)
    current_step = String()
    progress = Float()
    result = graphene.JSONString()
    error = String()
    timestamp = DateTime(required=True)

class SystemAlert(ObjectType):
    """GraphQL type for system alerts."""
    id = String(required=True)
    type = String(required=True)
    severity = String(required=True)
    message = String(required=True)
    details = graphene.JSONString()
    timestamp = DateTime(required=True)
