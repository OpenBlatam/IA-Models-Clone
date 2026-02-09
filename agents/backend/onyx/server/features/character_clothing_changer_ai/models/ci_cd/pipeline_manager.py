"""
CI/CD Pipeline Manager
======================

Continuous Integration and Deployment pipeline management.
"""

import time
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class PipelineStage(Enum):
    """Pipeline stage."""
    BUILD = "build"
    TEST = "test"
    LINT = "lint"
    SECURITY = "security"
    DEPLOY = "deploy"


class PipelineStatus(Enum):
    """Pipeline status."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class PipelineStep:
    """Pipeline step."""
    name: str
    stage: PipelineStage
    command: str
    timeout: float = 300.0
    retries: int = 0
    on_success: Optional[Callable] = None
    on_failure: Optional[Callable] = None


@dataclass
class PipelineResult:
    """Pipeline result."""
    pipeline_id: str
    status: PipelineStatus
    duration: float
    steps: List[Dict[str, Any]]
    errors: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []


class CICDPipelineManager:
    """CI/CD pipeline manager."""
    
    def __init__(self):
        """Initialize pipeline manager."""
        self.pipelines: Dict[str, List[PipelineStep]] = {}
        self.history: List[PipelineResult] = []
    
    def create_pipeline(
        self,
        name: str,
        steps: List[PipelineStep],
    ) -> None:
        """
        Create a pipeline.
        
        Args:
            name: Pipeline name
            steps: Pipeline steps
        """
        self.pipelines[name] = steps
        logger.info(f"Created pipeline: {name} with {len(steps)} steps")
    
    def run_pipeline(
        self,
        name: str,
        pipeline_id: Optional[str] = None,
    ) -> PipelineResult:
        """
        Run a pipeline.
        
        Args:
            name: Pipeline name
            pipeline_id: Optional pipeline ID
            
        Returns:
            Pipeline result
        """
        if name not in self.pipelines:
            raise ValueError(f"Pipeline not found: {name}")
        
        if pipeline_id is None:
            pipeline_id = f"{name}_{int(time.time())}"
        
        steps = self.pipelines[name]
        start_time = time.time()
        
        result = PipelineResult(
            pipeline_id=pipeline_id,
            status=PipelineStatus.RUNNING,
            duration=0.0,
            steps=[],
        )
        
        logger.info(f"Running pipeline: {name} ({pipeline_id})")
        
        try:
            for step in steps:
                step_result = self._run_step(step, pipeline_id)
                result.steps.append(step_result)
                
                if step_result["status"] == "failed":
                    result.status = PipelineStatus.FAILED
                    result.errors.append(f"Step {step.name} failed: {step_result.get('error')}")
                    break
            else:
                result.status = PipelineStatus.SUCCESS
            
        except Exception as e:
            result.status = PipelineStatus.FAILED
            result.errors.append(str(e))
            logger.error(f"Pipeline error: {e}")
        
        result.duration = time.time() - start_time
        
        self.history.append(result)
        
        # Keep only last 100 pipeline runs
        if len(self.history) > 100:
            self.history = self.history[-100:]
        
        logger.info(f"Pipeline completed: {name} - {result.status.value} ({result.duration:.2f}s)")
        
        return result
    
    def _run_step(
        self,
        step: PipelineStep,
        pipeline_id: str,
    ) -> Dict[str, Any]:
        """
        Run a pipeline step.
        
        Args:
            step: Pipeline step
            pipeline_id: Pipeline ID
            
        Returns:
            Step result
        """
        start_time = time.time()
        
        step_result = {
            "name": step.name,
            "stage": step.stage.value,
            "status": "running",
            "duration": 0.0,
        }
        
        logger.info(f"Running step: {step.name} ({step.stage.value})")
        
        try:
            # Simulate step execution
            # In real implementation, this would execute the command
            time.sleep(0.1)  # Simulate execution
            
            step_result["status"] = "success"
            step_result["duration"] = time.time() - start_time
            
            if step.on_success:
                step.on_success(step_result)
            
            logger.info(f"Step completed: {step.name} ({step_result['duration']:.2f}s)")
            
        except Exception as e:
            step_result["status"] = "failed"
            step_result["duration"] = time.time() - start_time
            step_result["error"] = str(e)
            
            if step.on_failure:
                step.on_failure(step_result)
            
            logger.error(f"Step failed: {step.name} - {e}")
        
        return step_result
    
    def get_pipeline_history(
        self,
        name: Optional[str] = None,
        limit: int = 10,
    ) -> List[PipelineResult]:
        """
        Get pipeline history.
        
        Args:
            name: Optional pipeline name filter
            limit: Maximum number of results
            
        Returns:
            Pipeline history
        """
        history = self.history
        
        if name:
            history = [r for r in history if r.pipeline_id.startswith(name)]
        
        return history[-limit:]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        if not self.history:
            return {
                "total_pipelines": 0,
                "defined_pipelines": len(self.pipelines),
            }
        
        successful = sum(1 for r in self.history if r.status == PipelineStatus.SUCCESS)
        failed = sum(1 for r in self.history if r.status == PipelineStatus.FAILED)
        avg_duration = sum(r.duration for r in self.history) / len(self.history)
        
        return {
            "total_pipelines": len(self.history),
            "defined_pipelines": len(self.pipelines),
            "successful": successful,
            "failed": failed,
            "success_rate": (successful / len(self.history) * 100) if self.history else 0.0,
            "average_duration": avg_duration,
        }

