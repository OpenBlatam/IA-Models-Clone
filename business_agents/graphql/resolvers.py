"""
GraphQL Resolvers
=================

GraphQL resolvers for the Business Agents System.
"""

import graphene
from typing import List, Optional, Dict, Any
from datetime import datetime

from .types import (
    AgentType, WorkflowType, DocumentType, SystemInfoType, 
    SystemMetricsType, HealthStatusType, AgentConnection, 
    WorkflowConnection, DocumentConnection, AgentExecutionUpdate,
    WorkflowExecutionUpdate, SystemAlert
)
from ..core.dependencies import (
    get_agent_service, get_workflow_service, get_document_service,
    get_health_service, get_system_info_service, get_metrics_service
)
from ..services.agent_service import AgentService
from ..services.workflow_service import WorkflowService
from ..services.document_service import DocumentService
from ..services.health_service import HealthService
from ..services.system_info_service import SystemInfoService
from ..services.metrics_service import MetricsService

class AgentResolver:
    """GraphQL resolver for agent operations."""
    
    def __init__(self, agent_service: AgentService):
        self.agent_service = agent_service
    
    async def resolve_agents(
        self, 
        info, 
        business_area: Optional[str] = None,
        is_active: Optional[bool] = None,
        first: Optional[int] = None,
        after: Optional[str] = None
    ) -> AgentConnection:
        """Resolve agents with pagination."""
        try:
            # Get agents from service
            agents_data = await self.agent_service.list_agents(
                business_area=business_area,
                is_active=is_active
            )
            
            # Convert to GraphQL types
            agent_edges = []
            for i, agent_data in enumerate(agents_data):
                agent = AgentType(
                    id=agent_data["id"],
                    name=agent_data["name"],
                    business_area=agent_data["business_area"],
                    description=agent_data["description"],
                    capabilities=agent_data["capabilities"],
                    is_active=agent_data["is_active"],
                    created_at=agent_data["created_at"],
                    updated_at=agent_data["updated_at"],
                    metadata=agent_data.get("metadata"),
                    version=agent_data.get("version"),
                    author=agent_data.get("author"),
                    dependencies=agent_data.get("dependencies", [])
                )
                
                edge = {
                    "node": agent,
                    "cursor": f"agent_{i}"
                }
                agent_edges.append(edge)
            
            # Implement pagination
            if first and after:
                # Simple pagination implementation
                start_index = int(after.split("_")[1]) if after else 0
                end_index = start_index + first
                agent_edges = agent_edges[start_index:end_index]
            
            has_next_page = len(agents_data) > (start_index + first) if first else False
            has_previous_page = start_index > 0 if after else False
            
            page_info = {
                "has_next_page": has_next_page,
                "has_previous_page": has_previous_page,
                "start_cursor": agent_edges[0]["cursor"] if agent_edges else None,
                "end_cursor": agent_edges[-1]["cursor"] if agent_edges else None
            }
            
            return AgentConnection(
                edges=agent_edges,
                page_info=page_info,
                total_count=len(agents_data)
            )
            
        except Exception as e:
            raise Exception(f"Failed to resolve agents: {str(e)}")
    
    async def resolve_agent(self, info, id: str) -> Optional[AgentType]:
        """Resolve a single agent by ID."""
        try:
            agent_data = await self.agent_service.get_agent(id)
            if not agent_data:
                return None
            
            return AgentType(
                id=agent_data["id"],
                name=agent_data["name"],
                business_area=agent_data["business_area"],
                description=agent_data["description"],
                capabilities=agent_data["capabilities"],
                is_active=agent_data["is_active"],
                created_at=agent_data["created_at"],
                updated_at=agent_data["updated_at"],
                metadata=agent_data.get("metadata"),
                version=agent_data.get("version"),
                author=agent_data.get("author"),
                dependencies=agent_data.get("dependencies", [])
            )
            
        except Exception as e:
            raise Exception(f"Failed to resolve agent {id}: {str(e)}")
    
    async def resolve_execute_capability(
        self, 
        info, 
        agent_id: str, 
        capability_name: str, 
        inputs: Dict[str, Any],
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Resolve capability execution."""
        try:
            result = await self.agent_service.execute_agent_capability(
                agent_id=agent_id,
                capability_name=capability_name,
                inputs=inputs,
                parameters=parameters or {}
            )
            
            return result
            
        except Exception as e:
            raise Exception(f"Failed to execute capability: {str(e)}")

class WorkflowResolver:
    """GraphQL resolver for workflow operations."""
    
    def __init__(self, workflow_service: WorkflowService):
        self.workflow_service = workflow_service
    
    async def resolve_workflows(
        self, 
        info, 
        business_area: Optional[str] = None,
        status: Optional[str] = None,
        first: Optional[int] = None,
        after: Optional[str] = None
    ) -> WorkflowConnection:
        """Resolve workflows with pagination."""
        try:
            workflows_data = await self.workflow_service.list_workflows(
                business_area=business_area,
                status=status
            )
            
            # Convert to GraphQL types
            workflow_edges = []
            for i, workflow_data in enumerate(workflows_data):
                workflow = WorkflowType(
                    id=workflow_data["id"],
                    name=workflow_data["name"],
                    description=workflow_data["description"],
                    business_area=workflow_data["business_area"],
                    status=workflow_data["status"],
                    created_by=workflow_data["created_by"],
                    created_at=workflow_data["created_at"],
                    updated_at=workflow_data["updated_at"],
                    variables=workflow_data.get("variables"),
                    metadata=workflow_data.get("metadata"),
                    steps=workflow_data.get("steps", [])
                )
                
                edge = {
                    "node": workflow,
                    "cursor": f"workflow_{i}"
                }
                workflow_edges.append(edge)
            
            # Implement pagination
            if first and after:
                start_index = int(after.split("_")[1]) if after else 0
                end_index = start_index + first
                workflow_edges = workflow_edges[start_index:end_index]
            
            has_next_page = len(workflows_data) > (start_index + first) if first else False
            has_previous_page = start_index > 0 if after else False
            
            page_info = {
                "has_next_page": has_next_page,
                "has_previous_page": has_previous_page,
                "start_cursor": workflow_edges[0]["cursor"] if workflow_edges else None,
                "end_cursor": workflow_edges[-1]["cursor"] if workflow_edges else None
            }
            
            return WorkflowConnection(
                edges=workflow_edges,
                page_info=page_info,
                total_count=len(workflows_data)
            )
            
        except Exception as e:
            raise Exception(f"Failed to resolve workflows: {str(e)}")
    
    async def resolve_workflow(self, info, id: str) -> Optional[WorkflowType]:
        """Resolve a single workflow by ID."""
        try:
            workflow_data = await self.workflow_service.get_workflow(id)
            if not workflow_data:
                return None
            
            return WorkflowType(
                id=workflow_data["id"],
                name=workflow_data["name"],
                description=workflow_data["description"],
                business_area=workflow_data["business_area"],
                status=workflow_data["status"],
                created_by=workflow_data["created_by"],
                created_at=workflow_data["created_at"],
                updated_at=workflow_data["updated_at"],
                variables=workflow_data.get("variables"),
                metadata=workflow_data.get("metadata"),
                steps=workflow_data.get("steps", [])
            )
            
        except Exception as e:
            raise Exception(f"Failed to resolve workflow {id}: {str(e)}")
    
    async def resolve_execute_workflow(
        self, 
        info, 
        workflow_id: str,
        input_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Resolve workflow execution."""
        try:
            result = await self.workflow_service.execute_workflow(workflow_id)
            return result
            
        except Exception as e:
            raise Exception(f"Failed to execute workflow: {str(e)}")

class DocumentResolver:
    """GraphQL resolver for document operations."""
    
    def __init__(self, document_service: DocumentService):
        self.document_service = document_service
    
    async def resolve_documents(
        self, 
        info, 
        business_area: Optional[str] = None,
        document_type: Optional[str] = None,
        first: Optional[int] = None,
        after: Optional[str] = None
    ) -> DocumentConnection:
        """Resolve documents with pagination."""
        try:
            documents_data = await self.document_service.list_documents(
                business_area=business_area,
                document_type=document_type
            )
            
            # Convert to GraphQL types
            document_edges = []
            for i, document_data in enumerate(documents_data):
                document = DocumentType(
                    id=document_data["id"],
                    request_id=document_data["request_id"],
                    title=document_data["title"],
                    content=document_data.get("content"),
                    file_path=document_data.get("file_path"),
                    format=document_data["format"],
                    size_bytes=document_data.get("size_bytes"),
                    created_at=document_data["created_at"],
                    completed_at=document_data.get("completed_at"),
                    error_message=document_data.get("error_message"),
                    metadata=document_data.get("metadata")
                )
                
                edge = {
                    "node": document,
                    "cursor": f"document_{i}"
                }
                document_edges.append(edge)
            
            # Implement pagination
            if first and after:
                start_index = int(after.split("_")[1]) if after else 0
                end_index = start_index + first
                document_edges = document_edges[start_index:end_index]
            
            has_next_page = len(documents_data) > (start_index + first) if first else False
            has_previous_page = start_index > 0 if after else False
            
            page_info = {
                "has_next_page": has_next_page,
                "has_previous_page": has_previous_page,
                "start_cursor": document_edges[0]["cursor"] if document_edges else None,
                "end_cursor": document_edges[-1]["cursor"] if document_edges else None
            }
            
            return DocumentConnection(
                edges=document_edges,
                page_info=page_info,
                total_count=len(documents_data)
            )
            
        except Exception as e:
            raise Exception(f"Failed to resolve documents: {str(e)}")
    
    async def resolve_document(self, info, id: str) -> Optional[DocumentType]:
        """Resolve a single document by ID."""
        try:
            document_data = await self.document_service.get_document(id)
            if not document_data:
                return None
            
            return DocumentType(
                id=document_data["id"],
                request_id=document_data["request_id"],
                title=document_data["title"],
                content=document_data.get("content"),
                file_path=document_data.get("file_path"),
                format=document_data["format"],
                size_bytes=document_data.get("size_bytes"),
                created_at=document_data["created_at"],
                completed_at=document_data.get("completed_at"),
                error_message=document_data.get("error_message"),
                metadata=document_data.get("metadata")
            )
            
        except Exception as e:
            raise Exception(f"Failed to resolve document {id}: {str(e)}")
    
    async def resolve_generate_document(
        self, 
        info, 
        document_type: str,
        title: str,
        description: str,
        business_area: str,
        variables: Optional[Dict[str, Any]] = None,
        template_id: Optional[str] = None,
        format: str = "markdown"
    ) -> Dict[str, Any]:
        """Resolve document generation."""
        try:
            result = await self.document_service.generate_document(
                document_type=document_type,
                title=title,
                description=description,
                business_area=business_area,
                created_by="graphql_user",  # This should come from context
                variables=variables or {},
                format=format
            )
            
            return result
            
        except Exception as e:
            raise Exception(f"Failed to generate document: {str(e)}")

class SystemResolver:
    """GraphQL resolver for system operations."""
    
    def __init__(
        self, 
        health_service: HealthService,
        system_info_service: SystemInfoService,
        metrics_service: MetricsService
    ):
        self.health_service = health_service
        self.system_info_service = system_info_service
        self.metrics_service = metrics_service
    
    async def resolve_health(self, info) -> HealthStatusType:
        """Resolve system health status."""
        try:
            health_data = await self.health_service.get_health_status()
            
            return HealthStatusType(
                status=health_data["status"],
                timestamp=health_data["timestamp"],
                version=health_data.get("version", "1.0.0"),
                components=health_data.get("components", {}),
                metrics=health_data.get("metrics", {})
            )
            
        except Exception as e:
            raise Exception(f"Failed to resolve health status: {str(e)}")
    
    async def resolve_system_info(self, info) -> SystemInfoType:
        """Resolve system information."""
        try:
            system_info_data = await self.system_info_service.get_system_info()
            
            return SystemInfoType(
                system=system_info_data["system"],
                capabilities=system_info_data["capabilities"],
                business_areas=system_info_data["business_areas"],
                workflow_templates=system_info_data["workflow_templates"],
                configuration=system_info_data["configuration"]
            )
            
        except Exception as e:
            raise Exception(f"Failed to resolve system info: {str(e)}")
    
    async def resolve_metrics(self, info) -> SystemMetricsType:
        """Resolve system metrics."""
        try:
            metrics_data = await self.metrics_service.get_system_metrics()
            
            return SystemMetricsType(
                agents=metrics_data["agents"],
                workflows=metrics_data["workflows"],
                business_areas=metrics_data["business_areas"],
                timestamp=metrics_data["timestamp"]
            )
            
        except Exception as e:
            raise Exception(f"Failed to resolve metrics: {str(e)}")

class SubscriptionResolver:
    """GraphQL resolver for real-time subscriptions."""
    
    async def resolve_agent_execution_updates(self, info, agent_id: Optional[str] = None):
        """Resolve agent execution updates subscription."""
        # This would integrate with a real-time system like Redis pub/sub
        # For now, return a placeholder
        yield AgentExecutionUpdate(
            agent_id=agent_id or "example_agent",
            capability_name="example_capability",
            status="running",
            progress=50.0,
            result=None,
            error=None,
            timestamp=datetime.now()
        )
    
    async def resolve_workflow_execution_updates(self, info, workflow_id: Optional[str] = None):
        """Resolve workflow execution updates subscription."""
        # This would integrate with a real-time system like Redis pub/sub
        yield WorkflowExecutionUpdate(
            workflow_id=workflow_id or "example_workflow",
            status="running",
            current_step="step_1",
            progress=25.0,
            result=None,
            error=None,
            timestamp=datetime.now()
        )
    
    async def resolve_system_alerts(self, info, severity: Optional[str] = None):
        """Resolve system alerts subscription."""
        # This would integrate with a real-time alerting system
        yield SystemAlert(
            id="alert_001",
            type="performance",
            severity=severity or "warning",
            message="High CPU usage detected",
            details={"cpu_usage": 85.0, "threshold": 80.0},
            timestamp=datetime.now()
        )
