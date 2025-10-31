"""
Workflow Automation Service
===========================

Advanced workflow automation for document approval, review, and collaboration.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from enum import Enum
from dataclasses import dataclass
from uuid import uuid4

from .models import ProfessionalDocument, DocumentType

logger = logging.getLogger(__name__)


class WorkflowStatus(str, Enum):
    """Workflow status types."""
    DRAFT = "draft"
    PENDING_REVIEW = "pending_review"
    IN_REVIEW = "in_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    PUBLISHED = "published"
    ARCHIVED = "archived"


class WorkflowAction(str, Enum):
    """Workflow action types."""
    SUBMIT = "submit"
    APPROVE = "approve"
    REJECT = "reject"
    REQUEST_CHANGES = "request_changes"
    PUBLISH = "publish"
    ARCHIVE = "archive"
    ASSIGN = "assign"
    COMMENT = "comment"


class UserRole(str, Enum):
    """User roles in workflow."""
    AUTHOR = "author"
    REVIEWER = "reviewer"
    APPROVER = "approver"
    PUBLISHER = "publisher"
    ADMIN = "admin"


@dataclass
class WorkflowStep:
    """Workflow step definition."""
    step_id: str
    name: str
    description: str
    required_role: UserRole
    is_required: bool
    auto_approve: bool
    timeout_hours: Optional[int] = None
    conditions: Optional[Dict[str, Any]] = None


@dataclass
class WorkflowInstance:
    """Workflow instance for a document."""
    instance_id: str
    document_id: str
    workflow_template_id: str
    current_step: str
    status: WorkflowStatus
    created_at: datetime
    updated_at: datetime
    assigned_users: Dict[str, List[str]]  # step_id -> user_ids
    step_history: List[Dict[str, Any]]
    metadata: Dict[str, Any]


@dataclass
class WorkflowAction:
    """Workflow action record."""
    action_id: str
    instance_id: str
    step_id: str
    user_id: str
    action_type: WorkflowAction
    timestamp: datetime
    comment: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class WorkflowAutomationService:
    """Advanced workflow automation service."""
    
    def __init__(self):
        self.workflow_templates = {}
        self.workflow_instances = {}
        self.workflow_actions = []
        self.notification_handlers = []
        self.auto_approval_rules = []
        
        # Load default workflow templates
        self._load_default_workflow_templates()
    
    def _load_default_workflow_templates(self):
        """Load default workflow templates."""
        
        # Standard document approval workflow
        standard_workflow = {
            "template_id": "standard_approval",
            "name": "Standard Document Approval",
            "description": "Standard workflow for document review and approval",
            "steps": [
                WorkflowStep(
                    step_id="draft",
                    name="Draft",
                    description="Document in draft state",
                    required_role=UserRole.AUTHOR,
                    is_required=True,
                    auto_approve=False
                ),
                WorkflowStep(
                    step_id="review",
                    name="Review",
                    description="Document under review",
                    required_role=UserRole.REVIEWER,
                    is_required=True,
                    auto_approve=False,
                    timeout_hours=72
                ),
                WorkflowStep(
                    step_id="approval",
                    name="Approval",
                    description="Document awaiting approval",
                    required_role=UserRole.APPROVER,
                    is_required=True,
                    auto_approve=False,
                    timeout_hours=48
                ),
                WorkflowStep(
                    step_id="published",
                    name="Published",
                    description="Document published",
                    required_role=UserRole.PUBLISHER,
                    is_required=True,
                    auto_approve=False
                )
            ]
        }
        
        # Fast track workflow for simple documents
        fast_track_workflow = {
            "template_id": "fast_track",
            "name": "Fast Track Approval",
            "description": "Fast track workflow for simple documents",
            "steps": [
                WorkflowStep(
                    step_id="draft",
                    name="Draft",
                    description="Document in draft state",
                    required_role=UserRole.AUTHOR,
                    is_required=True,
                    auto_approve=False
                ),
                WorkflowStep(
                    step_id="auto_approve",
                    name="Auto Approval",
                    description="Automatic approval for simple documents",
                    required_role=UserRole.AUTHOR,
                    is_required=True,
                    auto_approve=True,
                    conditions={"max_word_count": 1000, "document_types": ["newsletter", "brochure"]}
                ),
                WorkflowStep(
                    step_id="published",
                    name="Published",
                    description="Document published",
                    required_role=UserRole.PUBLISHER,
                    is_required=True,
                    auto_approve=False
                )
            ]
        }
        
        # Enterprise workflow with multiple reviewers
        enterprise_workflow = {
            "template_id": "enterprise",
            "name": "Enterprise Document Workflow",
            "description": "Enterprise workflow with multiple reviewers and approvers",
            "steps": [
                WorkflowStep(
                    step_id="draft",
                    name="Draft",
                    description="Document in draft state",
                    required_role=UserRole.AUTHOR,
                    is_required=True,
                    auto_approve=False
                ),
                WorkflowStep(
                    step_id="peer_review",
                    name="Peer Review",
                    description="Peer review by team members",
                    required_role=UserRole.REVIEWER,
                    is_required=True,
                    auto_approve=False,
                    timeout_hours=48
                ),
                WorkflowStep(
                    step_id="technical_review",
                    name="Technical Review",
                    description="Technical review by subject matter experts",
                    required_role=UserRole.REVIEWER,
                    is_required=True,
                    auto_approve=False,
                    timeout_hours=72
                ),
                WorkflowStep(
                    step_id="manager_approval",
                    name="Manager Approval",
                    description="Approval by department manager",
                    required_role=UserRole.APPROVER,
                    is_required=True,
                    auto_approve=False,
                    timeout_hours=24
                ),
                WorkflowStep(
                    step_id="executive_approval",
                    name="Executive Approval",
                    description="Final approval by executive",
                    required_role=UserRole.APPROVER,
                    is_required=True,
                    auto_approve=False,
                    timeout_hours=48
                ),
                WorkflowStep(
                    step_id="published",
                    name="Published",
                    description="Document published",
                    required_role=UserRole.PUBLISHER,
                    is_required=True,
                    auto_approve=False
                )
            ]
        }
        
        self.workflow_templates = {
            "standard_approval": standard_workflow,
            "fast_track": fast_track_workflow,
            "enterprise": enterprise_workflow
        }
    
    async def start_workflow(
        self,
        document: ProfessionalDocument,
        workflow_template_id: str,
        assigned_users: Optional[Dict[str, List[str]]] = None
    ) -> WorkflowInstance:
        """Start a workflow for a document."""
        
        try:
            # Get workflow template
            template = self.workflow_templates.get(workflow_template_id)
            if not template:
                raise ValueError(f"Workflow template {workflow_template_id} not found")
            
            # Create workflow instance
            instance = WorkflowInstance(
                instance_id=str(uuid4()),
                document_id=document.id,
                workflow_template_id=workflow_template_id,
                current_step="draft",
                status=WorkflowStatus.DRAFT,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                assigned_users=assigned_users or {},
                step_history=[],
                metadata={
                    "document_title": document.title,
                    "document_type": document.document_type.value,
                    "author": document.author
                }
            )
            
            # Store instance
            self.workflow_instances[instance.instance_id] = instance
            
            # Check for auto-approval rules
            await self._check_auto_approval_rules(instance, document)
            
            # Send notifications
            await self._send_workflow_notifications(instance, "workflow_started")
            
            logger.info(f"Started workflow {instance.instance_id} for document {document.id}")
            
            return instance
            
        except Exception as e:
            logger.error(f"Error starting workflow: {str(e)}")
            raise
    
    async def execute_workflow_action(
        self,
        instance_id: str,
        step_id: str,
        user_id: str,
        action_type: WorkflowAction,
        comment: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> WorkflowInstance:
        """Execute a workflow action."""
        
        try:
            # Get workflow instance
            instance = self.workflow_instances.get(instance_id)
            if not instance:
                raise ValueError(f"Workflow instance {instance_id} not found")
            
            # Validate action
            if not await self._validate_workflow_action(instance, step_id, user_id, action_type):
                raise ValueError("Invalid workflow action")
            
            # Create action record
            action = WorkflowAction(
                action_id=str(uuid4()),
                instance_id=instance_id,
                step_id=step_id,
                user_id=user_id,
                action_type=action_type,
                timestamp=datetime.now(),
                comment=comment,
                metadata=metadata
            )
            
            # Add to history
            instance.step_history.append({
                "action_id": action.action_id,
                "step_id": step_id,
                "user_id": user_id,
                "action_type": action_type.value,
                "timestamp": action.timestamp.isoformat(),
                "comment": comment,
                "metadata": metadata
            })
            
            # Update workflow state
            await self._update_workflow_state(instance, action)
            
            # Store action
            self.workflow_actions.append(action)
            
            # Send notifications
            await self._send_workflow_notifications(instance, f"action_{action_type.value}")
            
            logger.info(f"Executed workflow action {action_type.value} for instance {instance_id}")
            
            return instance
            
        except Exception as e:
            logger.error(f"Error executing workflow action: {str(e)}")
            raise
    
    async def _validate_workflow_action(
        self,
        instance: WorkflowInstance,
        step_id: str,
        user_id: str,
        action_type: WorkflowAction
    ) -> bool:
        """Validate workflow action."""
        
        # Get workflow template
        template = self.workflow_templates.get(instance.workflow_template_id)
        if not template:
            return False
        
        # Find current step
        current_step = None
        for step in template["steps"]:
            if step.step_id == instance.current_step:
                current_step = step
                break
        
        if not current_step:
            return False
        
        # Check if user has required role
        # This would typically check against user roles in the system
        # For now, we'll assume validation passes
        
        # Check if action is valid for current step
        valid_actions = self._get_valid_actions_for_step(current_step)
        if action_type not in valid_actions:
            return False
        
        return True
    
    def _get_valid_actions_for_step(self, step: WorkflowStep) -> List[WorkflowAction]:
        """Get valid actions for a workflow step."""
        
        if step.step_id == "draft":
            return [WorkflowAction.SUBMIT, WorkflowAction.ASSIGN, WorkflowAction.COMMENT]
        elif step.step_id in ["review", "approval"]:
            return [WorkflowAction.APPROVE, WorkflowAction.REJECT, WorkflowAction.REQUEST_CHANGES, WorkflowAction.COMMENT]
        elif step.step_id == "published":
            return [WorkflowAction.ARCHIVE, WorkflowAction.COMMENT]
        else:
            return [WorkflowAction.COMMENT]
    
    async def _update_workflow_state(self, instance: WorkflowInstance, action: WorkflowAction):
        """Update workflow state based on action."""
        
        # Get workflow template
        template = self.workflow_templates.get(instance.workflow_template_id)
        if not template:
            return
        
        # Update based on action type
        if action.action_type == WorkflowAction.SUBMIT:
            # Move to next step
            next_step = self._get_next_step(template, instance.current_step)
            if next_step:
                instance.current_step = next_step.step_id
                instance.status = WorkflowStatus.PENDING_REVIEW
        
        elif action.action_type == WorkflowAction.APPROVE:
            # Move to next step or complete
            next_step = self._get_next_step(template, instance.current_step)
            if next_step:
                instance.current_step = next_step.step_id
                if next_step.step_id == "published":
                    instance.status = WorkflowStatus.PUBLISHED
                else:
                    instance.status = WorkflowStatus.PENDING_REVIEW
            else:
                instance.status = WorkflowStatus.APPROVED
        
        elif action.action_type == WorkflowAction.REJECT:
            instance.status = WorkflowStatus.REJECTED
        
        elif action.action_type == WorkflowAction.REQUEST_CHANGES:
            instance.status = WorkflowStatus.DRAFT
            instance.current_step = "draft"
        
        elif action.action_type == WorkflowAction.PUBLISH:
            instance.status = WorkflowStatus.PUBLISHED
            instance.current_step = "published"
        
        elif action.action_type == WorkflowAction.ARCHIVE:
            instance.status = WorkflowStatus.ARCHIVED
        
        # Update timestamp
        instance.updated_at = datetime.now()
    
    def _get_next_step(self, template: Dict[str, Any], current_step_id: str) -> Optional[WorkflowStep]:
        """Get next step in workflow."""
        
        steps = template["steps"]
        current_index = None
        
        for i, step in enumerate(steps):
            if step.step_id == current_step_id:
                current_index = i
                break
        
        if current_index is None or current_index >= len(steps) - 1:
            return None
        
        return steps[current_index + 1]
    
    async def _check_auto_approval_rules(self, instance: WorkflowInstance, document: ProfessionalDocument):
        """Check auto-approval rules."""
        
        # Get workflow template
        template = self.workflow_templates.get(instance.workflow_template_id)
        if not template:
            return
        
        # Check each step for auto-approval conditions
        for step in template["steps"]:
            if step.auto_approve and step.conditions:
                if await self._evaluate_auto_approval_conditions(step.conditions, document):
                    # Auto-approve this step
                    await self.execute_workflow_action(
                        instance.instance_id,
                        step.step_id,
                        "system",
                        WorkflowAction.APPROVE,
                        "Auto-approved based on conditions"
                    )
    
    async def _evaluate_auto_approval_conditions(
        self,
        conditions: Dict[str, Any],
        document: ProfessionalDocument
    ) -> bool:
        """Evaluate auto-approval conditions."""
        
        # Check word count condition
        if "max_word_count" in conditions:
            if document.word_count > conditions["max_word_count"]:
                return False
        
        # Check document type condition
        if "document_types" in conditions:
            if document.document_type.value not in conditions["document_types"]:
                return False
        
        # Check quality score condition
        if "min_quality_score" in conditions:
            quality_score = document.metadata.get("quality_score", 0)
            if quality_score < conditions["min_quality_score"]:
                return False
        
        return True
    
    async def _send_workflow_notifications(self, instance: WorkflowInstance, event_type: str):
        """Send workflow notifications."""
        
        for handler in self.notification_handlers:
            try:
                await handler(instance, event_type)
            except Exception as e:
                logger.error(f"Error in notification handler: {str(e)}")
    
    def add_notification_handler(self, handler: Callable):
        """Add notification handler."""
        self.notification_handlers.append(handler)
    
    async def get_workflow_status(self, instance_id: str) -> Optional[Dict[str, Any]]:
        """Get workflow status."""
        
        instance = self.workflow_instances.get(instance_id)
        if not instance:
            return None
        
        # Get workflow template
        template = self.workflow_templates.get(instance.workflow_template_id)
        
        return {
            "instance_id": instance.instance_id,
            "document_id": instance.document_id,
            "workflow_template": template["name"] if template else "Unknown",
            "current_step": instance.current_step,
            "status": instance.status.value,
            "created_at": instance.created_at.isoformat(),
            "updated_at": instance.updated_at.isoformat(),
            "assigned_users": instance.assigned_users,
            "step_history": instance.step_history,
            "metadata": instance.metadata
        }
    
    async def get_user_workflows(self, user_id: str) -> List[Dict[str, Any]]:
        """Get workflows for a user."""
        
        user_workflows = []
        
        for instance in self.workflow_instances.values():
            # Check if user is assigned to any step
            user_assigned = False
            for step_users in instance.assigned_users.values():
                if user_id in step_users:
                    user_assigned = True
                    break
            
            if user_assigned:
                status = await self.get_workflow_status(instance.instance_id)
                if status:
                    user_workflows.append(status)
        
        return user_workflows
    
    async def get_workflow_analytics(self, time_range: str = "month") -> Dict[str, Any]:
        """Get workflow analytics."""
        
        # Calculate analytics based on workflow instances and actions
        total_workflows = len(self.workflow_instances)
        completed_workflows = len([i for i in self.workflow_instances.values() if i.status == WorkflowStatus.PUBLISHED])
        rejected_workflows = len([i for i in self.workflow_instances.values() if i.status == WorkflowStatus.REJECTED])
        
        # Calculate average completion time
        completion_times = []
        for instance in self.workflow_instances.values():
            if instance.status == WorkflowStatus.PUBLISHED:
                completion_time = (instance.updated_at - instance.created_at).total_seconds() / 3600  # hours
                completion_times.append(completion_time)
        
        avg_completion_time = sum(completion_times) / len(completion_times) if completion_times else 0
        
        # Calculate step distribution
        step_counts = {}
        for instance in self.workflow_instances.values():
            step = instance.current_step
            step_counts[step] = step_counts.get(step, 0) + 1
        
        return {
            "total_workflows": total_workflows,
            "completed_workflows": completed_workflows,
            "rejected_workflows": rejected_workflows,
            "completion_rate": (completed_workflows / total_workflows * 100) if total_workflows > 0 else 0,
            "rejection_rate": (rejected_workflows / total_workflows * 100) if total_workflows > 0 else 0,
            "average_completion_time_hours": round(avg_completion_time, 2),
            "current_step_distribution": step_counts,
            "total_actions": len(self.workflow_actions)
        }
    
    def create_custom_workflow_template(
        self,
        template_id: str,
        name: str,
        description: str,
        steps: List[WorkflowStep]
    ) -> Dict[str, Any]:
        """Create custom workflow template."""
        
        template = {
            "template_id": template_id,
            "name": name,
            "description": description,
            "steps": steps
        }
        
        self.workflow_templates[template_id] = template
        
        logger.info(f"Created custom workflow template: {template_id}")
        
        return template
    
    def get_available_workflow_templates(self) -> List[Dict[str, Any]]:
        """Get available workflow templates."""
        
        templates = []
        for template_id, template in self.workflow_templates.items():
            templates.append({
                "template_id": template_id,
                "name": template["name"],
                "description": template["description"],
                "steps_count": len(template["steps"]),
                "steps": [
                    {
                        "step_id": step.step_id,
                        "name": step.name,
                        "description": step.description,
                        "required_role": step.required_role.value,
                        "is_required": step.is_required,
                        "auto_approve": step.auto_approve,
                        "timeout_hours": step.timeout_hours
                    }
                    for step in template["steps"]
                ]
            })
        
        return templates



























