"""
Business Agents Manager
=======================

Centralized management system for all business area agents.
"""
import asyncio
import logging
from typing import Any, Dict, List, Optional

from .defaults import create_default_agents, get_default_workflow_templates
from .agent_models import AgentCapability, BusinessAgent, BusinessArea

# Optional imports with safe fallbacks
try:
    from .document_generator import DocumentFormat, DocumentGenerator, DocumentType
except Exception as e:
    logging.warning(f"Failed to import DocumentGenerator: {e}")
    DocumentGenerator = None
    DocumentType = None
    DocumentFormat = None

try:
    from .workflow_engine import StepType, Workflow, WorkflowEngine, WorkflowStep
except Exception as e:
    logging.warning(f"Failed to import WorkflowEngine: {e}")
    WorkflowEngine = None
    Workflow = None
    WorkflowStep = None
    StepType = None

logger = logging.getLogger(__name__)


class BusinessAgentManager:
    """Centralized manager for all business agents."""

    def __init__(self) -> None:
        self.agents: Dict[str, BusinessAgent] = {}
        
        # Initialize dependencies if available
        if WorkflowEngine:
            try:
                self.workflow_engine = WorkflowEngine()
            except Exception as e:
                logger.warning(f"Failed to initialize WorkflowEngine: {e}")
                self.workflow_engine = None
        else:
            self.workflow_engine = None

        if DocumentGenerator:
            try:
                self.document_generator = DocumentGenerator()
            except Exception as e:
                logger.warning(f"Failed to initialize DocumentGenerator: {e}")
                self.document_generator = None
        else:
            self.document_generator = None

        # Initialize default agents
        self._initialize_default_agents()

        # Register workflow handlers if engine is available
        if self.workflow_engine and StepType:
            self._register_workflow_handlers()

    def _initialize_default_agents(self) -> None:
        """Initialize default business agents."""
        default_agents = create_default_agents()
        for agent in default_agents:
            self.agents[agent.id] = agent

    def _register_workflow_handlers(self) -> None:
        """Register workflow step handlers."""
        if not self.workflow_engine or not StepType:
            return
            
        # Document generation handler
        self.workflow_engine.register_step_handler(
            StepType.DOCUMENT_GENERATION, self._handle_document_generation_step
        )
        # API call handler
        self.workflow_engine.register_step_handler(
            StepType.API_CALL, self._handle_api_call_step
        )
        # Notification handler
        self.workflow_engine.register_step_handler(
            StepType.NOTIFICATION, self._handle_notification_step
        )

    # ------------------------------------------------------------------
    # Step Handlers
    # ------------------------------------------------------------------

    async def _handle_document_generation_step(
        self, step: Any, workflow: Any, previous_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle document generation workflow step."""
        if not self.document_generator:
            return {"status": "failed", "error": "DocumentGenerator not available"}

        try:
            # Extract parameters with defaults
            params = step.parameters or {}
            doc_type_str = params.get("document_type", "custom")
            try:
                document_type = DocumentType(doc_type_str) if DocumentType else doc_type_str
            except (ValueError, TypeError):
                document_type = "custom"
            
            title = params.get("title", f"Document from {step.name}")
            description = params.get("description", "")
            business_area = params.get("business_area", "general")
            variables = params.get("variables", {})
            format_str = params.get("format", "markdown")
            try:
                 fmt = DocumentFormat(format_str) if DocumentFormat else "markdown"
            except (ValueError, TypeError):
                 fmt = "markdown"

            # Create document request
            request = await self.document_generator.create_document_request(
                document_type=document_type,
                title=title,
                description=description,
                business_area=business_area,
                created_by=workflow.created_by,
                variables=variables,
                format=fmt,
            )

            # Generate document
            document = await self.document_generator.generate_document(request.id)

            content_preview = document.content
            if len(content_preview) > 500:
                content_preview = content_preview[:500] + "..."

            return {
                "status": "completed",
                "document_id": document.id,
                "file_path": document.file_path,
                "content_preview": content_preview,
            }

        except Exception as e:
            logger.error(f"Document generation failed: {str(e)}")
            return {"status": "failed", "error": str(e)}

    async def _handle_api_call_step(
        self, step: Any, workflow: Any, previous_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle API call workflow step."""
        try:
            # Placeholder for actual API integration
            return {
                "status": "completed",
                "response": {"message": "API call completed successfully"},
                "status_code": 200,
            }
        except Exception as e:
            logger.error(f"API call failed: {str(e)}")
            return {"status": "failed", "error": str(e)}

    async def _handle_notification_step(
        self, step: Any, workflow: Any, previous_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle notification workflow step."""
        try:
            recipients = step.parameters.get("recipients", [])
            return {
                "status": "completed",
                "notification_sent": True,
                "recipients_count": len(recipients),
            }
        except Exception as e:
            logger.error(f"Notification failed: {str(e)}")
            return {"status": "failed", "error": str(e)}

    # ------------------------------------------------------------------
    # Public Agent API
    # ------------------------------------------------------------------

    def get_agent(self, agent_id: str) -> Optional[BusinessAgent]:
        """Get agent by ID."""
        return self.agents.get(agent_id)

    def list_agents(
        self, business_area: Optional[BusinessArea] = None, is_active: Optional[bool] = None
    ) -> List[BusinessAgent]:
        """List agents with optional filters."""
        agents = list(self.agents.values())
        if business_area:
            agents = [a for a in agents if a.business_area == business_area]
        if is_active is not None:
            agents = [a for a in agents if a.is_active == is_active]
        return agents

    def get_agent_capabilities(self, agent_id: str) -> List[AgentCapability]:
        """Get capabilities for a specific agent."""
        agent = self.get_agent(agent_id)
        return agent.capabilities if agent else []

    async def execute_agent_capability(
        self,
        agent_id: str,
        capability_name: str,
        inputs: Dict[str, Any],
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute a specific agent capability."""
        agent = self.get_agent(agent_id)
        if not agent:
            raise ValueError(f"Agent {agent_id} not found")
        if not agent.is_active:
            raise ValueError(f"Agent {agent_id} is not active")

        capability = next((c for c in agent.capabilities if c.name == capability_name), None)
        if not capability:
            raise ValueError(f"Capability {capability_name} not found for agent {agent_id}")

        try:
            result = await self._execute_capability_logic(
                agent, capability, inputs, parameters or {}
            )
            return {
                "status": "completed",
                "agent_id": agent_id,
                "capability": capability_name,
                "result": result,
                "execution_time": capability.estimated_duration,
            }
        except Exception as e:
            logger.error(f"Capability execution failed: {str(e)}")
            return {
                "status": "failed",
                "agent_id": agent_id,
                "capability": capability_name,
                "error": str(e),
            }

    async def _execute_capability_logic(
        self,
        agent: BusinessAgent,
        capability: AgentCapability,
        inputs: Dict[str, Any],
        parameters: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute agent capability logic (mock implementation)."""
        # This mirrors the logic from the original file but cleaner
        if capability.name == "campaign_planning":
            return {
                "campaign_plan": {
                    "name": f"Campaign for {inputs.get('target_audience', 'general audience')}",
                    "channels": ["social_media", "email", "content_marketing"],
                    "budget_allocation": {"social_media": 40, "email": 30, "content": 30},
                    "timeline": "4 weeks",
                    "expected_reach": 10000,
                }
            }
        elif capability.name == "lead_generation":
             return {
                "leads": [
                    {"name": "Lead 1", "email": "lead1@example.com", "score": 85},
                    {"name": "Lead 2", "email": "lead2@example.com", "score": 72},
                    {"name": "Lead 3", "email": "lead3@example.com", "score": 91}
                ],
                "total_leads": 3,
                "average_score": 82.7
            }
        # ... logic for other capabilities could be added here or delegated ...
        return {
            "message": f"Capability {capability.name} executed successfully",
            "inputs_processed": len(inputs),
        }

    # ------------------------------------------------------------------
    # Workflow API
    # ------------------------------------------------------------------

    async def create_business_workflow(
        self,
        name: str,
        description: str,
        business_area: BusinessArea,
        steps: List[Dict[str, Any]],
        created_by: str,
        variables: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Create a business workflow using the workflow engine."""
        if not self.workflow_engine:
             raise RuntimeError("WorkflowEngine not initialized")
        return await self.workflow_engine.create_workflow(
            name=name,
            description=description,
            business_area=str(business_area),
            steps=steps,
            created_by=created_by,
            variables=variables,
        )

    async def execute_business_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """Execute a business workflow."""
        if not self.workflow_engine:
             raise RuntimeError("WorkflowEngine not initialized")
        return await self.workflow_engine.execute_workflow(workflow_id)

    def get_workflow(self, workflow_id: str) -> Optional[Any]:
        """Get workflow by ID."""
        if not self.workflow_engine:
             return None
        return self.workflow_engine.get_workflow(workflow_id)

    def list_workflows(
        self, business_area: Optional[BusinessArea] = None, created_by: Optional[str] = None
    ) -> List[Any]:
        """List workflows with optional filters."""
        if not self.workflow_engine:
             return []
        return self.workflow_engine.list_workflows(
            business_area=str(business_area) if business_area else None,
            created_by=created_by,
        )

    def get_business_areas(self) -> List[BusinessArea]:
        """Get all available business areas."""
        return list(BusinessArea)

    def get_agents_by_business_area(self, business_area: BusinessArea) -> List[BusinessAgent]:
        """Get all agents for a specific business area."""
        return [
            agent for agent in self.agents.values() if agent.business_area == business_area
        ]

    def get_workflow_templates(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get predefined workflow templates for each business area."""
        return get_default_workflow_templates()

    # ------------------------------------------------------------------
    # Document API
    # ------------------------------------------------------------------

    async def generate_business_document(
        self,
        document_type: Any,
        title: str,
        description: str,
        business_area: str,
        created_by: str,
        variables: Optional[Dict[str, Any]] = None,
        format: Any = None,
    ) -> Dict[str, Any]:
        """Generate a business document directly."""
        if not self.document_generator:
             raise RuntimeError("DocumentGenerator not initialized")
             
        if format is None and DocumentFormat:
             format = DocumentFormat.MARKDOWN

        request = await self.document_generator.create_document_request(
            document_type=document_type,
            title=title,
            description=description,
            business_area=business_area,
            created_by=created_by,
            variables=variables,
            format=format,
        )

        document = await self.document_generator.generate_document(request.id)

        return {
            "document_id": document.id,
            "request_id": request.id,
            "title": document.title,
            "file_path": document.file_path,
            "format": document.format.value if document.format else "markdown",
            "size_bytes": document.size_bytes,
            "created_at": document.created_at.isoformat(),
        }
