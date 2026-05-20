"""
Workflow Manager for Color Grading AI
======================================

Manages color grading workflows and pipelines.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime
import json
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)


class WorkflowStepType(Enum):
    """Workflow step types."""
    ANALYZE = "analyze"
    GRADE = "grade"
    COMPARE = "compare"
    EXPORT = "export"
    NOTIFY = "notify"


@dataclass
class WorkflowStep:
    """Workflow step definition."""
    step_id: str
    step_type: WorkflowStepType
    parameters: Dict[str, Any]
    depends_on: List[str] = field(default_factory=list)
    condition: Optional[str] = None  # Optional condition for execution


@dataclass
class Workflow:
    """Workflow definition."""
    id: str
    name: str
    description: str
    steps: List[WorkflowStep]
    created_at: datetime
    updated_at: datetime
    is_active: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data["created_at"] = self.created_at.isoformat()
        data["updated_at"] = self.updated_at.isoformat()
        data["steps"] = [
            {
                **step.__dict__,
                "step_type": step.step_type.value
            }
            for step in self.steps
        ]
        return data


class WorkflowManager:
    """
    Manages color grading workflows.
    
    Features:
    - Create and manage workflows
    - Execute workflows
    - Workflow templates
    - Conditional execution
    """
    
    def __init__(self, workflows_dir: str = "workflows"):
        """
        Initialize workflow manager.
        
        Args:
            workflows_dir: Directory for workflows storage
        """
        self.workflows_dir = Path(workflows_dir)
        self.workflows_dir.mkdir(parents=True, exist_ok=True)
        self._workflows: Dict[str, Workflow] = {}
        self._load_workflows()
    
    def _load_workflows(self):
        """Load workflows from disk."""
        workflows_file = self.workflows_dir / "workflows.json"
        if workflows_file.exists():
            try:
                with open(workflows_file, "r") as f:
                    data = json.load(f)
                
                for workflow_data in data.get("workflows", []):
                    workflow = self._workflow_from_dict(workflow_data)
                    self._workflows[workflow.id] = workflow
                
                logger.info(f"Loaded {len(self._workflows)} workflows")
            except Exception as e:
                logger.error(f"Error loading workflows: {e}")
    
    def _workflow_from_dict(self, data: Dict[str, Any]) -> Workflow:
        """Create workflow from dictionary."""
        steps = [
            WorkflowStep(
                step_id=step_data["step_id"],
                step_type=WorkflowStepType(step_data["step_type"]),
                parameters=step_data["parameters"],
                depends_on=step_data.get("depends_on", []),
                condition=step_data.get("condition")
            )
            for step_data in data["steps"]
        ]
        
        return Workflow(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            steps=steps,
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            is_active=data.get("is_active", True)
        )
    
    def create_workflow(
        self,
        name: str,
        description: str,
        steps: List[Dict[str, Any]]
    ) -> str:
        """
        Create a new workflow.
        
        Args:
            name: Workflow name
            description: Workflow description
            steps: List of step definitions
            
        Returns:
            Workflow ID
        """
        import uuid
        workflow_id = str(uuid.uuid4())
        now = datetime.now()
        
        workflow_steps = [
            WorkflowStep(
                step_id=step.get("step_id", str(uuid.uuid4())),
                step_type=WorkflowStepType(step["step_type"]),
                parameters=step["parameters"],
                depends_on=step.get("depends_on", []),
                condition=step.get("condition")
            )
            for step in steps
        ]
        
        workflow = Workflow(
            id=workflow_id,
            name=name,
            description=description,
            steps=workflow_steps,
            created_at=now,
            updated_at=now
        )
        
        self._workflows[workflow_id] = workflow
        self._save_workflows()
        
        logger.info(f"Created workflow: {name} ({workflow_id})")
        return workflow_id
    
    def get_workflow(self, workflow_id: str) -> Optional[Workflow]:
        """Get workflow by ID."""
        return self._workflows.get(workflow_id)
    
    def list_workflows(self, active_only: bool = False) -> List[Dict[str, Any]]:
        """
        List workflows.
        
        Args:
            active_only: Only return active workflows
            
        Returns:
            List of workflows
        """
        workflows = list(self._workflows.values())
        if active_only:
            workflows = [w for w in workflows if w.is_active]
        
        return [w.to_dict() for w in workflows]
    
    def _save_workflows(self):
        """Save workflows to disk."""
        workflows_file = self.workflows_dir / "workflows.json"
        data = {
            "workflows": [w.to_dict() for w in self._workflows.values()]
        }
        with open(workflows_file, "w") as f:
            json.dump(data, f, indent=2, default=str)
    
    async def execute_workflow(
        self,
        workflow_id: str,
        initial_data: Dict[str, Any],
        agent: Any
    ) -> Dict[str, Any]:
        """
        Execute a workflow.
        
        Args:
            workflow_id: Workflow ID
            initial_data: Initial data for workflow
            agent: Color grading agent
            
        Returns:
            Workflow execution result
        """
        workflow = self.get_workflow(workflow_id)
        if not workflow:
            raise ValueError(f"Workflow not found: {workflow_id}")
        
        results = {"initial_data": initial_data}
        executed_steps = set()
        
        # Execute steps in dependency order
        while len(executed_steps) < len(workflow.steps):
            # Find steps ready to execute
            ready_steps = [
                step for step in workflow.steps
                if step.step_id not in executed_steps
                and all(dep in executed_steps for dep in step.depends_on)
            ]
            
            if not ready_steps:
                # Circular dependency or missing dependency
                break
            
            # Execute ready steps
            for step in ready_steps:
                try:
                    step_result = await self._execute_step(step, results, agent)
                    results[step.step_id] = step_result
                    executed_steps.add(step.step_id)
                except Exception as e:
                    logger.error(f"Error executing step {step.step_id}: {e}")
                    results[step.step_id] = {"error": str(e)}
                    executed_steps.add(step.step_id)
        
        return results
    
    async def _execute_step(
        self,
        step: WorkflowStep,
        context: Dict[str, Any],
        agent: Any
    ) -> Dict[str, Any]:
        """Execute a workflow step."""
        if step.step_type == WorkflowStepType.ANALYZE:
            media_path = step.parameters.get("media_path") or context.get("media_path")
            return await agent.analyze_media(media_path)
        
        elif step.step_type == WorkflowStepType.GRADE:
            media_path = step.parameters.get("media_path") or context.get("media_path")
            media_type = step.parameters.get("media_type", "auto")
            
            if media_type == "video" or media_path.endswith((".mp4", ".mov", ".avi")):
                return await agent.grade_video(media_path, **step.parameters)
            else:
                return await agent.grade_image(media_path, **step.parameters)
        
        elif step.step_type == WorkflowStepType.COMPARE:
            before = step.parameters.get("before") or context.get("before_path")
            after = step.parameters.get("after") or context.get("after_path")
            output = step.parameters.get("output_path")
            return {"comparison_path": await agent.create_comparison(before, after, output)}
        
        elif step.step_type == WorkflowStepType.EXPORT:
            params = step.parameters.get("color_params") or context.get("color_params")
            output_path = step.parameters.get("output_path")
            format = step.parameters.get("format", "all")
            return agent.export_parameters(params, output_path, format)
        
        elif step.step_type == WorkflowStepType.NOTIFY:
            # Notification step
            return {"status": "notified"}
        
        return {"status": "completed"}




