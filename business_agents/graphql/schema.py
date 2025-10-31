"""
GraphQL Schema
==============

Main GraphQL schema definition for the Business Agents System.
"""

import graphene
from graphene import ObjectType, Mutation, Subscription
from typing import Optional

from .types import (
    AgentType, WorkflowType, DocumentType, SystemInfoType,
    SystemMetricsType, HealthStatusType, AgentConnection,
    WorkflowConnection, DocumentConnection, AgentExecutionUpdate,
    WorkflowExecutionUpdate, SystemAlert, AgentInput, WorkflowInput,
    DocumentInput, CapabilityExecutionInput
)
from .resolvers import (
    AgentResolver, WorkflowResolver, DocumentResolver,
    SystemResolver, SubscriptionResolver
)

class Query(ObjectType):
    """GraphQL Query root."""
    
    # Agent queries
    agents = graphene.Field(
        AgentConnection,
        business_area=graphene.String(),
        is_active=graphene.Boolean(),
        first=graphene.Int(),
        after=graphene.String(),
        description="Get paginated list of business agents"
    )
    
    agent = graphene.Field(
        AgentType,
        id=graphene.String(required=True),
        description="Get a specific agent by ID"
    )
    
    # Workflow queries
    workflows = graphene.Field(
        WorkflowConnection,
        business_area=graphene.String(),
        status=graphene.String(),
        first=graphene.Int(),
        after=graphene.String(),
        description="Get paginated list of workflows"
    )
    
    workflow = graphene.Field(
        WorkflowType,
        id=graphene.String(required=True),
        description="Get a specific workflow by ID"
    )
    
    # Document queries
    documents = graphene.Field(
        DocumentConnection,
        business_area=graphene.String(),
        document_type=graphene.String(),
        first=graphene.Int(),
        after=graphene.String(),
        description="Get paginated list of documents"
    )
    
    document = graphene.Field(
        DocumentType,
        id=graphene.String(required=True),
        description="Get a specific document by ID"
    )
    
    # System queries
    health = graphene.Field(
        HealthStatusType,
        description="Get system health status"
    )
    
    system_info = graphene.Field(
        SystemInfoType,
        description="Get detailed system information"
    )
    
    metrics = graphene.Field(
        SystemMetricsType,
        description="Get system metrics and statistics"
    )
    
    # Resolvers
    async def resolve_agents(self, info, **kwargs):
        """Resolve agents query."""
        from ..core.dependencies import get_agent_service
        agent_service = await get_agent_service()
        resolver = AgentResolver(agent_service)
        return await resolver.resolve_agents(info, **kwargs)
    
    async def resolve_agent(self, info, id):
        """Resolve agent query."""
        from ..core.dependencies import get_agent_service
        agent_service = await get_agent_service()
        resolver = AgentResolver(agent_service)
        return await resolver.resolve_agent(info, id)
    
    async def resolve_workflows(self, info, **kwargs):
        """Resolve workflows query."""
        from ..core.dependencies import get_workflow_service
        workflow_service = await get_workflow_service()
        resolver = WorkflowResolver(workflow_service)
        return await resolver.resolve_workflows(info, **kwargs)
    
    async def resolve_workflow(self, info, id):
        """Resolve workflow query."""
        from ..core.dependencies import get_workflow_service
        workflow_service = await get_workflow_service()
        resolver = WorkflowResolver(workflow_service)
        return await resolver.resolve_workflow(info, id)
    
    async def resolve_documents(self, info, **kwargs):
        """Resolve documents query."""
        from ..core.dependencies import get_document_service
        document_service = await get_document_service()
        resolver = DocumentResolver(document_service)
        return await resolver.resolve_documents(info, **kwargs)
    
    async def resolve_document(self, info, id):
        """Resolve document query."""
        from ..core.dependencies import get_document_service
        document_service = await get_document_service()
        resolver = DocumentResolver(document_service)
        return await resolver.resolve_document(info, id)
    
    async def resolve_health(self, info):
        """Resolve health query."""
        from ..core.dependencies import get_health_service
        health_service = await get_health_service()
        resolver = SystemResolver(health_service, None, None)
        return await resolver.resolve_health(info)
    
    async def resolve_system_info(self, info):
        """Resolve system info query."""
        from ..core.dependencies import get_system_info_service
        system_info_service = await get_system_info_service()
        resolver = SystemResolver(None, system_info_service, None)
        return await resolver.resolve_system_info(info)
    
    async def resolve_metrics(self, info):
        """Resolve metrics query."""
        from ..core.dependencies import get_metrics_service
        metrics_service = await get_metrics_service()
        resolver = SystemResolver(None, None, metrics_service)
        return await resolver.resolve_metrics(info)

class AgentMutations(Mutation):
    """Agent-related mutations."""
    
    class Arguments:
        agent_input = AgentInput(required=True, description="Agent creation data")
    
    agent = graphene.Field(AgentType, description="Created agent")
    success = graphene.Boolean(description="Whether the operation was successful")
    message = graphene.String(description="Operation message")
    
    async def mutate(self, info, agent_input):
        """Create a new agent."""
        try:
            # This would integrate with the agent service
            # For now, return a placeholder
            return AgentMutations(
                agent=None,
                success=True,
                message="Agent creation not yet implemented"
            )
        except Exception as e:
            return AgentMutations(
                agent=None,
                success=False,
                message=f"Failed to create agent: {str(e)}"
            )

class WorkflowMutations(Mutation):
    """Workflow-related mutations."""
    
    class Arguments:
        workflow_input = WorkflowInput(required=True, description="Workflow creation data")
    
    workflow = graphene.Field(WorkflowType, description="Created workflow")
    success = graphene.Boolean(description="Whether the operation was successful")
    message = graphene.String(description="Operation message")
    
    async def mutate(self, info, workflow_input):
        """Create a new workflow."""
        try:
            # This would integrate with the workflow service
            return WorkflowMutations(
                workflow=None,
                success=True,
                message="Workflow creation not yet implemented"
            )
        except Exception as e:
            return WorkflowMutations(
                workflow=None,
                success=False,
                message=f"Failed to create workflow: {str(e)}"
            )

class DocumentMutations(Mutation):
    """Document-related mutations."""
    
    class Arguments:
        document_input = DocumentInput(required=True, description="Document generation data")
    
    document = graphene.Field(DocumentType, description="Generated document")
    success = graphene.Boolean(description="Whether the operation was successful")
    message = graphene.String(description="Operation message")
    
    async def mutate(self, info, document_input):
        """Generate a new document."""
        try:
            from ..core.dependencies import get_document_service
            document_service = await get_document_service()
            resolver = DocumentResolver(document_service)
            
            result = await resolver.resolve_generate_document(
                info,
                document_type=document_input["document_type"],
                title=document_input["title"],
                description=document_input["description"],
                business_area=document_input["business_area"],
                variables=document_input.get("variables"),
                template_id=document_input.get("template_id"),
                format=document_input.get("format", "markdown")
            )
            
            return DocumentMutations(
                document=None,  # Would create DocumentType from result
                success=result.get("success", True),
                message="Document generated successfully"
            )
        except Exception as e:
            return DocumentMutations(
                document=None,
                success=False,
                message=f"Failed to generate document: {str(e)}"
            )

class ExecutionMutations(Mutation):
    """Execution-related mutations."""
    
    class Arguments:
        capability_input = CapabilityExecutionInput(required=True, description="Capability execution data")
    
    result = graphene.JSONString(description="Execution result")
    success = graphene.Boolean(description="Whether the operation was successful")
    message = graphene.String(description="Operation message")
    
    async def mutate(self, info, capability_input):
        """Execute an agent capability."""
        try:
            from ..core.dependencies import get_agent_service
            agent_service = await get_agent_service()
            resolver = AgentResolver(agent_service)
            
            result = await resolver.resolve_execute_capability(
                info,
                agent_id=capability_input["agent_id"],
                capability_name=capability_input["capability_name"],
                inputs=capability_input["inputs"],
                parameters=capability_input.get("parameters")
            )
            
            return ExecutionMutations(
                result=result,
                success=result.get("status") == "completed",
                message="Capability executed successfully"
            )
        except Exception as e:
            return ExecutionMutations(
                result=None,
                success=False,
                message=f"Failed to execute capability: {str(e)}"
            )

class Mutation(ObjectType):
    """GraphQL Mutation root."""
    
    create_agent = AgentMutations.Field(description="Create a new business agent")
    create_workflow = WorkflowMutations.Field(description="Create a new workflow")
    generate_document = DocumentMutations.Field(description="Generate a new document")
    execute_capability = ExecutionMutations.Field(description="Execute an agent capability")

class Subscription(ObjectType):
    """GraphQL Subscription root."""
    
    agent_execution_updates = graphene.Field(
        AgentExecutionUpdate,
        agent_id=graphene.String(),
        description="Subscribe to agent execution updates"
    )
    
    workflow_execution_updates = graphene.Field(
        WorkflowExecutionUpdate,
        workflow_id=graphene.String(),
        description="Subscribe to workflow execution updates"
    )
    
    system_alerts = graphene.Field(
        SystemAlert,
        severity=graphene.String(),
        description="Subscribe to system alerts"
    )
    
    async def resolve_agent_execution_updates(self, info, agent_id=None):
        """Resolve agent execution updates subscription."""
        resolver = SubscriptionResolver()
        async for update in resolver.resolve_agent_execution_updates(info, agent_id):
            yield update
    
    async def resolve_workflow_execution_updates(self, info, workflow_id=None):
        """Resolve workflow execution updates subscription."""
        resolver = SubscriptionResolver()
        async for update in resolver.resolve_workflow_execution_updates(info, workflow_id):
            yield update
    
    async def resolve_system_alerts(self, info, severity=None):
        """Resolve system alerts subscription."""
        resolver = SubscriptionResolver()
        async for alert in resolver.resolve_system_alerts(info, severity):
            yield alert

def create_schema():
    """Create the GraphQL schema."""
    return graphene.Schema(
        query=Query,
        mutation=Mutation,
        subscription=Subscription,
        types=[
            AgentType, WorkflowType, DocumentType, SystemInfoType,
            SystemMetricsType, HealthStatusType, AgentConnection,
            WorkflowConnection, DocumentConnection, AgentExecutionUpdate,
            WorkflowExecutionUpdate, SystemAlert
        ]
    )

# Global schema instance
schema = create_schema()
