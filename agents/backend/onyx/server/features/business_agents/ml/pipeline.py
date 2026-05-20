"""
ML Pipeline Management
======================

Machine learning pipeline orchestration and management.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import uuid
import json

from .types import (
    ModelType, TrainingStatus, ModelMetadata, TrainingConfig,
    ExperimentConfig, ExperimentResult, ModelDeployment
)

logger = logging.getLogger(__name__)

class MLPipeline:
    """Machine learning pipeline for training and inference."""
    
    def __init__(self, pipeline_id: str, name: str, description: str):
        self.pipeline_id = pipeline_id
        self.name = name
        self.description = description
        self.steps: List[Dict[str, Any]] = []
        self.status = "draft"
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        self.metadata: Dict[str, Any] = {}
    
    def add_step(self, step_type: str, config: Dict[str, Any], dependencies: List[str] = None):
        """Add a step to the pipeline."""
        step = {
            "id": str(uuid.uuid4()),
            "type": step_type,
            "config": config,
            "dependencies": dependencies or [],
            "status": "pending",
            "created_at": datetime.now()
        }
        self.steps.append(step)
        self.updated_at = datetime.now()
        logger.info(f"Added step {step['id']} to pipeline {self.pipeline_id}")
    
    def remove_step(self, step_id: str) -> bool:
        """Remove a step from the pipeline."""
        for i, step in enumerate(self.steps):
            if step["id"] == step_id:
                del self.steps[i]
                self.updated_at = datetime.now()
                logger.info(f"Removed step {step_id} from pipeline {self.pipeline_id}")
                return True
        return False
    
    def validate_pipeline(self) -> List[str]:
        """Validate the pipeline configuration."""
        errors = []
        
        # Check for circular dependencies
        step_ids = {step["id"] for step in self.steps}
        for step in self.steps:
            for dep in step["dependencies"]:
                if dep not in step_ids:
                    errors.append(f"Step {step['id']} depends on non-existent step {dep}")
        
        # Check for circular dependencies
        visited = set()
        rec_stack = set()
        
        def has_cycle(step_id: str) -> bool:
            if step_id in rec_stack:
                return True
            if step_id in visited:
                return False
            
            visited.add(step_id)
            rec_stack.add(step_id)
            
            step = next((s for s in self.steps if s["id"] == step_id), None)
            if step:
                for dep in step["dependencies"]:
                    if has_cycle(dep):
                        return True
            
            rec_stack.remove(step_id)
            return False
        
        for step in self.steps:
            if has_cycle(step["id"]):
                errors.append(f"Circular dependency detected involving step {step['id']}")
                break
        
        return errors
    
    def get_execution_order(self) -> List[str]:
        """Get the execution order of pipeline steps."""
        # Topological sort
        visited = set()
        temp_visited = set()
        order = []
        
        def visit(step_id: str):
            if step_id in temp_visited:
                raise ValueError(f"Circular dependency detected involving {step_id}")
            if step_id in visited:
                return
            
            temp_visited.add(step_id)
            
            step = next((s for s in self.steps if s["id"] == step_id), None)
            if step:
                for dep in step["dependencies"]:
                    visit(dep)
            
            temp_visited.remove(step_id)
            visited.add(step_id)
            order.append(step_id)
        
        for step in self.steps:
            if step["id"] not in visited:
                visit(step["id"])
        
        return order
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert pipeline to dictionary."""
        return {
            "pipeline_id": self.pipeline_id,
            "name": self.name,
            "description": self.description,
            "steps": self.steps,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MLPipeline':
        """Create pipeline from dictionary."""
        pipeline = cls(
            pipeline_id=data["pipeline_id"],
            name=data["name"],
            description=data["description"]
        )
        pipeline.steps = data.get("steps", [])
        pipeline.status = data.get("status", "draft")
        pipeline.created_at = datetime.fromisoformat(data["created_at"])
        pipeline.updated_at = datetime.fromisoformat(data["updated_at"])
        pipeline.metadata = data.get("metadata", {})
        return pipeline

class PipelineManager:
    """Manages ML pipelines."""
    
    def __init__(self):
        self.pipelines: Dict[str, MLPipeline] = {}
        self.executions: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()
    
    async def create_pipeline(
        self, 
        name: str, 
        description: str, 
        steps: List[Dict[str, Any]] = None
    ) -> str:
        """Create a new ML pipeline."""
        try:
            pipeline_id = str(uuid.uuid4())
            
            pipeline = MLPipeline(
                pipeline_id=pipeline_id,
                name=name,
                description=description
            )
            
            if steps:
                for step in steps:
                    pipeline.add_step(
                        step_type=step["type"],
                        config=step["config"],
                        dependencies=step.get("dependencies", [])
                    )
            
            async with self._lock:
                self.pipelines[pipeline_id] = pipeline
            
            logger.info(f"Created pipeline {pipeline_id}: {name}")
            return pipeline_id
            
        except Exception as e:
            logger.error(f"Failed to create pipeline: {str(e)}")
            raise
    
    async def get_pipeline(self, pipeline_id: str) -> Optional[MLPipeline]:
        """Get a pipeline by ID."""
        return self.pipelines.get(pipeline_id)
    
    async def update_pipeline(
        self, 
        pipeline_id: str, 
        updates: Dict[str, Any]
    ) -> bool:
        """Update a pipeline."""
        try:
            async with self._lock:
                if pipeline_id not in self.pipelines:
                    return False
                
                pipeline = self.pipelines[pipeline_id]
                
                if "name" in updates:
                    pipeline.name = updates["name"]
                if "description" in updates:
                    pipeline.description = updates["description"]
                if "metadata" in updates:
                    pipeline.metadata.update(updates["metadata"])
                
                pipeline.updated_at = datetime.now()
            
            logger.info(f"Updated pipeline {pipeline_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update pipeline {pipeline_id}: {str(e)}")
            return False
    
    async def delete_pipeline(self, pipeline_id: str) -> bool:
        """Delete a pipeline."""
        try:
            async with self._lock:
                if pipeline_id in self.pipelines:
                    del self.pipelines[pipeline_id]
                    logger.info(f"Deleted pipeline {pipeline_id}")
                    return True
                return False
                
        except Exception as e:
            logger.error(f"Failed to delete pipeline {pipeline_id}: {str(e)}")
            return False
    
    async def list_pipelines(self) -> List[Dict[str, Any]]:
        """List all pipelines."""
        return [pipeline.to_dict() for pipeline in self.pipelines.values()]
    
    async def execute_pipeline(
        self, 
        pipeline_id: str, 
        config: Dict[str, Any] = None
    ) -> str:
        """Execute a pipeline."""
        try:
            pipeline = await self.get_pipeline(pipeline_id)
            if not pipeline:
                raise ValueError(f"Pipeline {pipeline_id} not found")
            
            # Validate pipeline
            errors = pipeline.validate_pipeline()
            if errors:
                raise ValueError(f"Pipeline validation failed: {errors}")
            
            execution_id = str(uuid.uuid4())
            
            execution = {
                "execution_id": execution_id,
                "pipeline_id": pipeline_id,
                "status": "running",
                "started_at": datetime.now(),
                "config": config or {},
                "steps": {},
                "logs": []
            }
            
            async with self._lock:
                self.executions[execution_id] = execution
            
            # Execute pipeline asynchronously
            asyncio.create_task(self._execute_pipeline_async(execution_id))
            
            logger.info(f"Started pipeline execution {execution_id}")
            return execution_id
            
        except Exception as e:
            logger.error(f"Failed to execute pipeline {pipeline_id}: {str(e)}")
            raise
    
    async def _execute_pipeline_async(self, execution_id: str):
        """Execute pipeline steps asynchronously."""
        try:
            execution = self.executions[execution_id]
            pipeline = await self.get_pipeline(execution["pipeline_id"])
            
            if not pipeline:
                execution["status"] = "failed"
                execution["error"] = "Pipeline not found"
                return
            
            # Get execution order
            execution_order = pipeline.get_execution_order()
            
            # Execute steps in order
            for step_id in execution_order:
                step = next((s for s in pipeline.steps if s["id"] == step_id), None)
                if not step:
                    continue
                
                try:
                    execution["steps"][step_id] = {
                        "status": "running",
                        "started_at": datetime.now()
                    }
                    
                    # Execute step
                    result = await self._execute_step(step, execution["config"])
                    
                    execution["steps"][step_id].update({
                        "status": "completed",
                        "completed_at": datetime.now(),
                        "result": result
                    })
                    
                    logger.info(f"Completed step {step_id} in execution {execution_id}")
                    
                except Exception as e:
                    execution["steps"][step_id].update({
                        "status": "failed",
                        "completed_at": datetime.now(),
                        "error": str(e)
                    })
                    
                    logger.error(f"Failed step {step_id} in execution {execution_id}: {str(e)}")
                    execution["status"] = "failed"
                    execution["error"] = str(e)
                    return
            
            # All steps completed successfully
            execution["status"] = "completed"
            execution["completed_at"] = datetime.now()
            
            logger.info(f"Completed pipeline execution {execution_id}")
            
        except Exception as e:
            execution["status"] = "failed"
            execution["error"] = str(e)
            execution["completed_at"] = datetime.now()
            logger.error(f"Pipeline execution {execution_id} failed: {str(e)}")
    
    async def _execute_step(self, step: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single pipeline step."""
        step_type = step["type"]
        step_config = step["config"]
        
        # Simulate step execution based on type
        if step_type == "data_preprocessing":
            return await self._execute_data_preprocessing(step_config, config)
        elif step_type == "feature_engineering":
            return await self._execute_feature_engineering(step_config, config)
        elif step_type == "model_training":
            return await self._execute_model_training(step_config, config)
        elif step_type == "model_evaluation":
            return await self._execute_model_evaluation(step_config, config)
        elif step_type == "model_deployment":
            return await self._execute_model_deployment(step_config, config)
        else:
            raise ValueError(f"Unknown step type: {step_type}")
    
    async def _execute_data_preprocessing(self, step_config: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute data preprocessing step."""
        # Simulate data preprocessing
        await asyncio.sleep(1)  # Simulate processing time
        
        return {
            "processed_records": 1000,
            "features_created": 50,
            "data_quality_score": 0.95
        }
    
    async def _execute_feature_engineering(self, step_config: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute feature engineering step."""
        await asyncio.sleep(2)
        
        return {
            "features_engineered": 25,
            "feature_importance": {"feature_1": 0.8, "feature_2": 0.6},
            "feature_correlation": 0.3
        }
    
    async def _execute_model_training(self, step_config: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute model training step."""
        await asyncio.sleep(5)
        
        return {
            "model_id": str(uuid.uuid4()),
            "training_accuracy": 0.92,
            "validation_accuracy": 0.89,
            "training_time": 300
        }
    
    async def _execute_model_evaluation(self, step_config: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute model evaluation step."""
        await asyncio.sleep(1)
        
        return {
            "test_accuracy": 0.88,
            "precision": 0.87,
            "recall": 0.89,
            "f1_score": 0.88
        }
    
    async def _execute_model_deployment(self, step_config: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute model deployment step."""
        await asyncio.sleep(3)
        
        return {
            "deployment_id": str(uuid.uuid4()),
            "endpoint_url": "https://api.example.com/model/v1/predict",
            "status": "deployed"
        }
    
    async def get_execution(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get pipeline execution by ID."""
        return self.executions.get(execution_id)
    
    async def list_executions(self, pipeline_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List pipeline executions."""
        executions = list(self.executions.values())
        
        if pipeline_id:
            executions = [e for e in executions if e["pipeline_id"] == pipeline_id]
        
        return executions
    
    async def cancel_execution(self, execution_id: str) -> bool:
        """Cancel a pipeline execution."""
        try:
            async with self._lock:
                if execution_id in self.executions:
                    execution = self.executions[execution_id]
                    if execution["status"] == "running":
                        execution["status"] = "cancelled"
                        execution["cancelled_at"] = datetime.now()
                        logger.info(f"Cancelled pipeline execution {execution_id}")
                        return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to cancel execution {execution_id}: {str(e)}")
            return False

# Global pipeline manager instance
pipeline_manager = PipelineManager()
