"""
Enterprise Workflow Engine
=========================

Advanced workflow automation for enterprise content creation and approval processes.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
from uuid import UUID, uuid4

from ..schemas import CopywritingRequest, CopywritingVariant
from ..exceptions import WorkflowError, ValidationError
from ..utils import performance_tracker

logger = logging.getLogger(__name__)


class WorkflowStatus(str, Enum):
    """Workflow status options"""
    DRAFT = "draft"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
    IN_REVIEW = "in_review"
    PUBLISHED = "published"
    ARCHIVED = "archived"


class WorkflowStepType(str, Enum):
    """Workflow step types"""
    CONTENT_GENERATION = "content_generation"
    REVIEW = "review"
    APPROVAL = "approval"
    EDIT = "edit"
    PUBLISH = "publish"
    NOTIFICATION = "notification"
    CONDITIONAL = "conditional"


class ApprovalLevel(str, Enum):
    """Approval levels"""
    CREATOR = "creator"
    TEAM_LEAD = "team_lead"
    MANAGER = "manager"
    DIRECTOR = "director"
    EXECUTIVE = "executive"


@dataclass
class WorkflowStep:
    """Individual workflow step"""
    id: UUID
    name: str
    step_type: WorkflowStepType
    description: str
    assigned_to: Optional[str] = None
    approval_level: Optional[ApprovalLevel] = None
    is_required: bool = True
    estimated_duration_minutes: int = 30
    conditions: Optional[Dict[str, Any]] = None
    actions: Optional[List[str]] = None


@dataclass
class WorkflowTemplate:
    """Workflow template"""
    id: UUID
    name: str
    description: str
    steps: List[WorkflowStep]
    is_active: bool = True
    created_by: str = "system"
    created_at: datetime = None
    updated_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.updated_at is None:
            self.updated_at = datetime.utcnow()


@dataclass
class WorkflowInstance:
    """Workflow instance"""
    id: UUID
    template_id: UUID
    name: str
    status: WorkflowStatus
    current_step_id: Optional[UUID] = None
    created_by: str = "system"
    assigned_to: Optional[str] = None
    priority: int = 1  # 1-5, 5 being highest
    due_date: Optional[datetime] = None
    created_at: datetime = None
    updated_at: datetime = None
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.updated_at is None:
            self.updated_at = datetime.utcnow()
        if self.metadata is None:
            self.metadata = {}


@dataclass
class WorkflowExecution:
    """Workflow execution record"""
    id: UUID
    workflow_instance_id: UUID
    step_id: UUID
    status: WorkflowStatus
    executed_by: str
    executed_at: datetime
    comments: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    duration_minutes: Optional[int] = None


class WorkflowEngine:
    """Enterprise workflow engine for content creation and approval"""
    
    def __init__(self):
        self.templates: Dict[UUID, WorkflowTemplate] = {}
        self.instances: Dict[UUID, WorkflowInstance] = {}
        self.executions: Dict[UUID, List[WorkflowExecution]] = {}
        self._initialize_default_templates()
    
    def _initialize_default_templates(self):
        """Initialize default workflow templates"""
        
        # Standard Content Creation Workflow
        standard_steps = [
            WorkflowStep(
                id=uuid4(),
                name="Content Generation",
                step_type=WorkflowStepType.CONTENT_GENERATION,
                description="Generate initial content using AI",
                estimated_duration_minutes=15
            ),
            WorkflowStep(
                id=uuid4(),
                name="Initial Review",
                step_type=WorkflowStepType.REVIEW,
                description="Review generated content for quality",
                assigned_to="content_creator",
                estimated_duration_minutes=30
            ),
            WorkflowStep(
                id=uuid4(),
                name="Team Lead Approval",
                step_type=WorkflowStepType.APPROVAL,
                description="Team lead approval for content",
                assigned_to="team_lead",
                approval_level=ApprovalLevel.TEAM_LEAD,
                estimated_duration_minutes=60
            ),
            WorkflowStep(
                id=uuid4(),
                name="Publish Content",
                step_type=WorkflowStepType.PUBLISH,
                description="Publish approved content",
                assigned_to="content_creator",
                estimated_duration_minutes=10
            )
        ]
        
        standard_template = WorkflowTemplate(
            id=uuid4(),
            name="Standard Content Creation",
            description="Standard workflow for content creation and approval",
            steps=standard_steps
        )
        
        self.templates[standard_template.id] = standard_template
        
        # High-Priority Content Workflow
        high_priority_steps = [
            WorkflowStep(
                id=uuid4(),
                name="Content Generation",
                step_type=WorkflowStepType.CONTENT_GENERATION,
                description="Generate high-priority content",
                estimated_duration_minutes=10
            ),
            WorkflowStep(
                id=uuid4(),
                name="Manager Review",
                step_type=WorkflowStepType.REVIEW,
                description="Manager review for high-priority content",
                assigned_to="manager",
                approval_level=ApprovalLevel.MANAGER,
                estimated_duration_minutes=30
            ),
            WorkflowStep(
                id=uuid4(),
                name="Director Approval",
                step_type=WorkflowStepType.APPROVAL,
                description="Director approval required",
                assigned_to="director",
                approval_level=ApprovalLevel.DIRECTOR,
                estimated_duration_minutes=45
            ),
            WorkflowStep(
                id=uuid4(),
                name="Executive Sign-off",
                step_type=WorkflowStepType.APPROVAL,
                description="Executive sign-off for high-priority content",
                assigned_to="executive",
                approval_level=ApprovalLevel.EXECUTIVE,
                estimated_duration_minutes=60
            ),
            WorkflowStep(
                id=uuid4(),
                name="Publish Content",
                step_type=WorkflowStepType.PUBLISH,
                description="Publish approved high-priority content",
                assigned_to="content_creator",
                estimated_duration_minutes=5
            )
        ]
        
        high_priority_template = WorkflowTemplate(
            id=uuid4(),
            name="High-Priority Content",
            description="Workflow for high-priority content requiring executive approval",
            steps=high_priority_steps
        )
        
        self.templates[high_priority_template.id] = high_priority_template
        
        # Marketing Campaign Workflow
        campaign_steps = [
            WorkflowStep(
                id=uuid4(),
                name="Campaign Brief",
                step_type=WorkflowStepType.CONTENT_GENERATION,
                description="Create campaign brief and strategy",
                estimated_duration_minutes=60
            ),
            WorkflowStep(
                id=uuid4(),
                name="Content Generation",
                step_type=WorkflowStepType.CONTENT_GENERATION,
                description="Generate campaign content",
                estimated_duration_minutes=30
            ),
            WorkflowStep(
                id=uuid4(),
                name="Marketing Review",
                step_type=WorkflowStepType.REVIEW,
                description="Marketing team review",
                assigned_to="marketing_team",
                estimated_duration_minutes=45
            ),
            WorkflowStep(
                id=uuid4(),
                name="Legal Review",
                step_type=WorkflowStepType.REVIEW,
                description="Legal compliance review",
                assigned_to="legal_team",
                estimated_duration_minutes=60
            ),
            WorkflowStep(
                id=uuid4(),
                name="Manager Approval",
                step_type=WorkflowStepType.APPROVAL,
                description="Marketing manager approval",
                assigned_to="marketing_manager",
                approval_level=ApprovalLevel.MANAGER,
                estimated_duration_minutes=30
            ),
            WorkflowStep(
                id=uuid4(),
                name="Launch Campaign",
                step_type=WorkflowStepType.PUBLISH,
                description="Launch marketing campaign",
                assigned_to="marketing_team",
                estimated_duration_minutes=15
            )
        ]
        
        campaign_template = WorkflowTemplate(
            id=uuid4(),
            name="Marketing Campaign",
            description="Workflow for marketing campaign content creation and approval",
            steps=campaign_steps
        )
        
        self.templates[campaign_template.id] = campaign_template
    
    async def create_workflow_instance(
        self,
        template_id: UUID,
        name: str,
        created_by: str,
        priority: int = 1,
        due_date: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> WorkflowInstance:
        """Create a new workflow instance from template"""
        
        if template_id not in self.templates:
            raise WorkflowError(f"Template {template_id} not found")
        
        template = self.templates[template_id]
        
        instance = WorkflowInstance(
            id=uuid4(),
            template_id=template_id,
            name=name,
            status=WorkflowStatus.DRAFT,
            created_by=created_by,
            priority=priority,
            due_date=due_date,
            metadata=metadata or {}
        )
        
        self.instances[instance.id] = instance
        self.executions[instance.id] = []
        
        logger.info(f"Created workflow instance {instance.id} from template {template_id}")
        
        return instance
    
    async def start_workflow(self, instance_id: UUID, started_by: str) -> WorkflowInstance:
        """Start a workflow instance"""
        
        if instance_id not in self.instances:
            raise WorkflowError(f"Workflow instance {instance_id} not found")
        
        instance = self.instances[instance_id]
        template = self.templates[instance.template_id]
        
        if instance.status != WorkflowStatus.DRAFT:
            raise WorkflowError(f"Workflow {instance_id} is not in draft status")
        
        # Set current step to first step
        if template.steps:
            instance.current_step_id = template.steps[0].id
            instance.status = WorkflowStatus.PENDING_APPROVAL
            instance.assigned_to = template.steps[0].assigned_to
            instance.updated_at = datetime.utcnow()
        
        # Record execution
        execution = WorkflowExecution(
            id=uuid4(),
            workflow_instance_id=instance_id,
            step_id=template.steps[0].id if template.steps else None,
            status=WorkflowStatus.PENDING_APPROVAL,
            executed_by=started_by,
            executed_at=datetime.utcnow(),
            comments="Workflow started"
        )
        
        self.executions[instance_id].append(execution)
        
        logger.info(f"Started workflow {instance_id}")
        
        return instance
    
    async def execute_step(
        self,
        instance_id: UUID,
        step_id: UUID,
        executed_by: str,
        status: WorkflowStatus,
        comments: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None
    ) -> WorkflowInstance:
        """Execute a workflow step"""
        
        if instance_id not in self.instances:
            raise WorkflowError(f"Workflow instance {instance_id} not found")
        
        instance = self.instances[instance_id]
        template = self.templates[instance.template_id]
        
        # Find current step
        current_step = None
        for step in template.steps:
            if step.id == step_id:
                current_step = step
                break
        
        if not current_step:
            raise WorkflowError(f"Step {step_id} not found in workflow")
        
        if instance.current_step_id != step_id:
            raise WorkflowError(f"Step {step_id} is not the current step")
        
        # Record execution
        start_time = datetime.utcnow()
        execution = WorkflowExecution(
            id=uuid4(),
            workflow_instance_id=instance_id,
            step_id=step_id,
            status=status,
            executed_by=executed_by,
            executed_at=start_time,
            comments=comments,
            data=data
        )
        
        self.executions[instance_id].append(execution)
        
        # Update workflow status
        if status == WorkflowStatus.APPROVED:
            # Move to next step
            next_step = self._get_next_step(template.steps, step_id)
            if next_step:
                instance.current_step_id = next_step.id
                instance.assigned_to = next_step.assigned_to
                instance.status = WorkflowStatus.PENDING_APPROVAL
            else:
                # Workflow completed
                instance.current_step_id = None
                instance.status = WorkflowStatus.PUBLISHED
                instance.completed_at = datetime.utcnow()
        
        elif status == WorkflowStatus.REJECTED:
            instance.status = WorkflowStatus.REJECTED
            instance.completed_at = datetime.utcnow()
        
        elif status == WorkflowStatus.IN_REVIEW:
            instance.status = WorkflowStatus.IN_REVIEW
        
        instance.updated_at = datetime.utcnow()
        
        logger.info(f"Executed step {step_id} in workflow {instance_id} with status {status}")
        
        return instance
    
    async def get_workflow_status(self, instance_id: UUID) -> Dict[str, Any]:
        """Get comprehensive workflow status"""
        
        if instance_id not in self.instances:
            raise WorkflowError(f"Workflow instance {instance_id} not found")
        
        instance = self.instances[instance_id]
        template = self.templates[instance.template_id]
        executions = self.executions.get(instance_id, [])
        
        # Calculate progress
        total_steps = len(template.steps)
        completed_steps = len([e for e in executions if e.status == WorkflowStatus.APPROVED])
        progress_percentage = (completed_steps / total_steps * 100) if total_steps > 0 else 0
        
        # Calculate estimated completion
        remaining_steps = [step for step in template.steps if step.id != instance.current_step_id]
        estimated_remaining_minutes = sum(step.estimated_duration_minutes for step in remaining_steps)
        estimated_completion = datetime.utcnow() + timedelta(minutes=estimated_remaining_minutes)
        
        # Get current step details
        current_step = None
        if instance.current_step_id:
            for step in template.steps:
                if step.id == instance.current_step_id:
                    current_step = step
                    break
        
        return {
            "workflow_instance": asdict(instance),
            "template": {
                "id": template.id,
                "name": template.name,
                "description": template.description
            },
            "current_step": asdict(current_step) if current_step else None,
            "progress": {
                "completed_steps": completed_steps,
                "total_steps": total_steps,
                "progress_percentage": progress_percentage,
                "estimated_completion": estimated_completion.isoformat()
            },
            "executions": [asdict(execution) for execution in executions[-10:]],  # Last 10 executions
            "is_overdue": instance.due_date and datetime.utcnow() > instance.due_date
        }
    
    async def get_user_workflows(
        self,
        user_id: str,
        status_filter: Optional[WorkflowStatus] = None
    ) -> List[Dict[str, Any]]:
        """Get workflows assigned to a user"""
        
        user_workflows = []
        
        for instance in self.instances.values():
            if instance.assigned_to == user_id:
                if status_filter is None or instance.status == status_filter:
                    status_info = await self.get_workflow_status(instance.id)
                    user_workflows.append(status_info)
        
        # Sort by priority and due date
        user_workflows.sort(key=lambda x: (x["workflow_instance"]["priority"], x["workflow_instance"]["due_date"] or datetime.max))
        
        return user_workflows
    
    async def get_workflow_analytics(self) -> Dict[str, Any]:
        """Get workflow analytics and metrics"""
        
        total_workflows = len(self.instances)
        completed_workflows = len([i for i in self.instances.values() if i.status == WorkflowStatus.PUBLISHED])
        in_progress_workflows = len([i for i in self.instances.values() if i.status in [WorkflowStatus.PENDING_APPROVAL, WorkflowStatus.IN_REVIEW]])
        rejected_workflows = len([i for i in self.instances.values() if i.status == WorkflowStatus.REJECTED])
        
        # Calculate average completion time
        completed_instances = [i for i in self.instances.values() if i.completed_at]
        if completed_instances:
            total_duration = sum(
                (i.completed_at - i.created_at).total_seconds() / 3600  # Convert to hours
                for i in completed_instances
            )
            avg_completion_hours = total_duration / len(completed_instances)
        else:
            avg_completion_hours = 0
        
        # Get overdue workflows
        overdue_workflows = len([
            i for i in self.instances.values()
            if i.due_date and datetime.utcnow() > i.due_date and i.status not in [WorkflowStatus.PUBLISHED, WorkflowStatus.REJECTED]
        ])
        
        # Template usage statistics
        template_usage = {}
        for instance in self.instances.values():
            template_id = str(instance.template_id)
            template_usage[template_id] = template_usage.get(template_id, 0) + 1
        
        return {
            "summary": {
                "total_workflows": total_workflows,
                "completed_workflows": completed_workflows,
                "in_progress_workflows": in_progress_workflows,
                "rejected_workflows": rejected_workflows,
                "overdue_workflows": overdue_workflows,
                "completion_rate": (completed_workflows / total_workflows * 100) if total_workflows > 0 else 0,
                "average_completion_hours": avg_completion_hours
            },
            "template_usage": template_usage,
            "status_distribution": {
                status.value: len([i for i in self.instances.values() if i.status == status])
                for status in WorkflowStatus
            }
        }
    
    def _get_next_step(self, steps: List[WorkflowStep], current_step_id: UUID) -> Optional[WorkflowStep]:
        """Get the next step in the workflow"""
        current_index = None
        for i, step in enumerate(steps):
            if step.id == current_step_id:
                current_index = i
                break
        
        if current_index is not None and current_index + 1 < len(steps):
            return steps[current_index + 1]
        
        return None
    
    async def create_custom_template(
        self,
        name: str,
        description: str,
        steps: List[WorkflowStep],
        created_by: str
    ) -> WorkflowTemplate:
        """Create a custom workflow template"""
        
        template = WorkflowTemplate(
            id=uuid4(),
            name=name,
            description=description,
            steps=steps,
            created_by=created_by
        )
        
        self.templates[template.id] = template
        
        logger.info(f"Created custom workflow template {template.id}: {name}")
        
        return template
    
    async def update_workflow_priority(
        self,
        instance_id: UUID,
        new_priority: int,
        updated_by: str
    ) -> WorkflowInstance:
        """Update workflow priority"""
        
        if instance_id not in self.instances:
            raise WorkflowError(f"Workflow instance {instance_id} not found")
        
        instance = self.instances[instance_id]
        instance.priority = new_priority
        instance.updated_at = datetime.utcnow()
        
        # Record the change
        execution = WorkflowExecution(
            id=uuid4(),
            workflow_instance_id=instance_id,
            step_id=instance.current_step_id,
            status=instance.status,
            executed_by=updated_by,
            executed_at=datetime.utcnow(),
            comments=f"Priority updated to {new_priority}"
        )
        
        self.executions[instance_id].append(execution)
        
        logger.info(f"Updated priority for workflow {instance_id} to {new_priority}")
        
        return instance


# Global workflow engine instance
workflow_engine = WorkflowEngine()






























