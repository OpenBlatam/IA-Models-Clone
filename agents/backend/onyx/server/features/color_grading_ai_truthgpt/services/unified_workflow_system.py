"""
Unified Workflow System for Color Grading AI
=============================================

Consolidates workflow management services:
- WorkflowManager (workflow management)
- ServiceOrchestrator (service orchestration)
- DataPipeline (data pipelines)

Features:
- Unified workflow interface
- Service orchestration
- Data pipelines
- Conditional execution
- Dependency management
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from .workflow_manager import WorkflowManager, Workflow, WorkflowStep, WorkflowStepType
from .service_orchestrator import ServiceOrchestrator, ServiceTask, OrchestrationStrategy
from .data_pipeline import DataPipeline, PipelineStep, PipelineStage

logger = logging.getLogger(__name__)


class WorkflowMode(Enum):
    """Workflow execution modes."""
    WORKFLOW = "workflow"  # Use WorkflowManager
    ORCHESTRATION = "orchestration"  # Use ServiceOrchestrator
    PIPELINE = "pipeline"  # Use DataPipeline
    AUTO = "auto"  # Auto-select based on complexity


@dataclass
class UnifiedWorkflowResult:
    """Unified workflow result."""
    workflow_id: str
    mode: WorkflowMode
    success: bool
    results: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    execution_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


class UnifiedWorkflowSystem:
    """
    Unified workflow system.
    
    Consolidates:
    - WorkflowManager: Workflow management
    - ServiceOrchestrator: Service orchestration
    - DataPipeline: Data pipelines
    
    Features:
    - Unified interface for workflows
    - Auto-mode selection
    - Service orchestration
    - Data pipelines
    """
    
    def __init__(
        self,
        workflows_dir: str = "workflows",
        services: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize unified workflow system.
        
        Args:
            workflows_dir: Workflows directory
            services: Optional services dictionary for orchestration
        """
        self.workflow_manager = WorkflowManager(workflows_dir=workflows_dir)
        self.service_orchestrator = ServiceOrchestrator(services=services or {})
        self.data_pipeline = DataPipeline(name="default")
        
        logger.info("Initialized UnifiedWorkflowSystem")
    
    async def execute_workflow(
        self,
        workflow_id: str,
        input_data: Any,
        mode: WorkflowMode = WorkflowMode.AUTO
    ) -> UnifiedWorkflowResult:
        """
        Execute workflow.
        
        Args:
            workflow_id: Workflow ID
            input_data: Input data
            mode: Execution mode
            
        Returns:
            Unified workflow result
        """
        start_time = datetime.now()
        
        try:
            # Auto-select mode
            if mode == WorkflowMode.AUTO:
                mode = self._select_mode(workflow_id)
            
            if mode == WorkflowMode.WORKFLOW:
                # Use WorkflowManager
                workflow = self.workflow_manager.get_workflow(workflow_id)
                if not workflow:
                    raise ValueError(f"Workflow not found: {workflow_id}")
                
                result = await self.workflow_manager.execute_workflow(workflow_id, input_data)
                
                return UnifiedWorkflowResult(
                    workflow_id=workflow_id,
                    mode=mode,
                    success=result.get("success", False),
                    results=result,
                    execution_time=(datetime.now() - start_time).total_seconds()
                )
            
            elif mode == WorkflowMode.ORCHESTRATION:
                # Use ServiceOrchestrator
                workflow = self.workflow_manager.get_workflow(workflow_id)
                if not workflow:
                    raise ValueError(f"Workflow not found: {workflow_id}")
                
                # Convert workflow steps to service tasks
                tasks = self._workflow_to_tasks(workflow)
                result = await self.service_orchestrator.orchestrate(
                    tasks,
                    OrchestrationStrategy.SEQUENTIAL
                )
                
                return UnifiedWorkflowResult(
                    workflow_id=workflow_id,
                    mode=mode,
                    success=result.success,
                    results=result.results,
                    errors=list(result.errors.values()),
                    execution_time=(datetime.now() - start_time).total_seconds()
                )
            
            elif mode == WorkflowMode.PIPELINE:
                # Use DataPipeline
                workflow = self.workflow_manager.get_workflow(workflow_id)
                if not workflow:
                    raise ValueError(f"Workflow not found: {workflow_id}")
                
                # Convert workflow steps to pipeline steps
                self._workflow_to_pipeline(workflow)
                result = await self.data_pipeline.execute(input_data, parallel=False)
                
                return UnifiedWorkflowResult(
                    workflow_id=workflow_id,
                    mode=mode,
                    success=result.success,
                    results={"output": result.output, "intermediate": result.intermediate_results},
                    errors=result.errors,
                    execution_time=(datetime.now() - start_time).total_seconds()
                )
        
        except Exception as e:
            logger.error(f"Workflow execution error: {e}")
            return UnifiedWorkflowResult(
                workflow_id=workflow_id,
                mode=mode,
                success=False,
                errors=[str(e)],
                execution_time=(datetime.now() - start_time).total_seconds()
            )
    
    def _select_mode(self, workflow_id: str) -> WorkflowMode:
        """Auto-select execution mode based on workflow complexity."""
        workflow = self.workflow_manager.get_workflow(workflow_id)
        if not workflow:
            return WorkflowMode.WORKFLOW
        
        # Simple workflows use WorkflowManager
        if len(workflow.steps) <= 5:
            return WorkflowMode.WORKFLOW
        
        # Complex workflows with many dependencies use orchestration
        total_dependencies = sum(len(step.depends_on) for step in workflow.steps)
        if total_dependencies > 10:
            return WorkflowMode.ORCHESTRATION
        
        # Data processing workflows use pipeline
        return WorkflowMode.PIPELINE
    
    def _workflow_to_tasks(self, workflow: Workflow) -> List[ServiceTask]:
        """Convert workflow steps to service tasks."""
        tasks = []
        for step in workflow.steps:
            # Map workflow step types to services
            service_name = self._map_step_to_service(step.step_type)
            method_name = self._map_step_to_method(step.step_type)
            
            tasks.append(ServiceTask(
                service_name=service_name,
                method_name=method_name,
                parameters=step.parameters,
                dependencies=[s.step_id for s in workflow.steps if s.step_id in step.depends_on]
            ))
        
        return tasks
    
    def _workflow_to_pipeline(self, workflow: Workflow):
        """Convert workflow to data pipeline."""
        self.data_pipeline._steps.clear()
        
        for step in workflow.steps:
            stage = self._map_step_to_stage(step.step_type)
            processor = self._create_processor_for_step(step)
            
            self.data_pipeline.add_step(
                name=step.step_id,
                stage=stage,
                processor=processor,
                dependencies=step.depends_on
            )
    
    def _map_step_to_service(self, step_type: WorkflowStepType) -> str:
        """Map workflow step type to service name."""
        mapping = {
            WorkflowStepType.ANALYZE: "color_analyzer",
            WorkflowStepType.GRADE: "unified_processing_system",
            WorkflowStepType.COMPARE: "comparison_generator",
            WorkflowStepType.EXPORT: "parameter_exporter",
            WorkflowStepType.NOTIFY: "unified_communication_system",
        }
        return mapping.get(step_type, "workflow_manager")
    
    def _map_step_to_method(self, step_type: WorkflowStepType) -> str:
        """Map workflow step type to method name."""
        mapping = {
            WorkflowStepType.ANALYZE: "analyze_image",
            WorkflowStepType.GRADE: "process_media",
            WorkflowStepType.COMPARE: "generate_comparison",
            WorkflowStepType.EXPORT: "export_parameters",
            WorkflowStepType.NOTIFY: "send",
        }
        return mapping.get(step_type, "execute")
    
    def _map_step_to_stage(self, step_type: WorkflowStepType) -> PipelineStage:
        """Map workflow step type to pipeline stage."""
        mapping = {
            WorkflowStepType.ANALYZE: PipelineStage.PROCESS,
            WorkflowStepType.GRADE: PipelineStage.PROCESS,
            WorkflowStepType.COMPARE: PipelineStage.OUTPUT,
            WorkflowStepType.EXPORT: PipelineStage.OUTPUT,
            WorkflowStepType.NOTIFY: PipelineStage.OUTPUT,
        }
        return mapping.get(step_type, PipelineStage.PROCESS)
    
    def _create_processor_for_step(self, step: WorkflowStep) -> Callable:
        """Create processor function for workflow step."""
        # This would create a callable that executes the step
        # Implementation depends on how steps are executed
        async def processor(input_data: Any) -> Any:
            # Placeholder - actual implementation would call appropriate service
            return input_data
        return processor
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get workflow statistics."""
        return {
            "workflows_count": len(self.workflow_manager._workflows),
            "orchestrator_available": self.service_orchestrator is not None,
            "pipeline_available": self.data_pipeline is not None,
        }


