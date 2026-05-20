"""
Data Pipeline Manager - Gestor de Pipelines de Datos
=====================================================

Sistema de gestión de pipelines de datos con transformaciones, validaciones y procesamiento paralelo.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from collections import defaultdict, deque
import uuid

logger = logging.getLogger(__name__)


class PipelineStatus(Enum):
    """Estado de pipeline."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class StageType(Enum):
    """Tipo de stage."""
    TRANSFORM = "transform"
    FILTER = "filter"
    VALIDATE = "validate"
    AGGREGATE = "aggregate"
    ENRICH = "enrich"
    EXPORT = "export"
    CUSTOM = "custom"


@dataclass
class PipelineStage:
    """Stage de pipeline."""
    stage_id: str
    stage_type: StageType
    name: str
    processor: Callable
    parallel: bool = False
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineExecution:
    """Ejecución de pipeline."""
    execution_id: str
    pipeline_id: str
    status: PipelineStatus = PipelineStatus.PENDING
    stages_completed: List[str] = field(default_factory=list)
    stages_failed: List[str] = field(default_factory=list)
    current_stage: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    input_data: Optional[Any] = None
    output_data: Optional[Any] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Pipeline:
    """Pipeline."""
    pipeline_id: str
    name: str
    description: str
    stages: List[PipelineStage] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class DataPipelineManager:
    """Gestor de pipelines de datos."""
    
    def __init__(self, max_parallel_executions: int = 10):
        self.pipelines: Dict[str, Pipeline] = {}
        self.executions: Dict[str, PipelineExecution] = {}
        self.execution_history: deque = deque(maxlen=100000)
        self.max_parallel_executions = max_parallel_executions
        self.active_executions: Dict[str, asyncio.Task] = {}
        self._lock = asyncio.Lock()
    
    def create_pipeline(
        self,
        pipeline_id: str,
        name: str,
        description: str = "",
        stages: Optional[List[PipelineStage]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Crear pipeline."""
        pipeline = Pipeline(
            pipeline_id=pipeline_id,
            name=name,
            description=description,
            stages=stages or [],
            metadata=metadata or {},
        )
        
        async def save_pipeline():
            async with self._lock:
                self.pipelines[pipeline_id] = pipeline
        
        asyncio.create_task(save_pipeline())
        
        logger.info(f"Created pipeline: {pipeline_id} - {name}")
        return pipeline_id
    
    def add_stage(
        self,
        pipeline_id: str,
        stage_id: str,
        stage_type: StageType,
        name: str,
        processor: Callable,
        parallel: bool = False,
        dependencies: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Agregar stage a pipeline."""
        stage = PipelineStage(
            stage_id=stage_id,
            stage_type=stage_type,
            name=name,
            processor=processor,
            parallel=parallel,
            dependencies=dependencies or [],
            metadata=metadata or {},
        )
        
        async def add_stage_async():
            async with self._lock:
                pipeline = self.pipelines.get(pipeline_id)
                if not pipeline:
                    raise ValueError(f"Pipeline {pipeline_id} not found")
                pipeline.stages.append(stage)
        
        asyncio.create_task(add_stage_async())
        
        logger.info(f"Added stage {stage_id} to pipeline {pipeline_id}")
        return stage_id
    
    async def execute_pipeline(
        self,
        pipeline_id: str,
        input_data: Any,
        execution_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Ejecutar pipeline."""
        pipeline = self.pipelines.get(pipeline_id)
        if not pipeline:
            raise ValueError(f"Pipeline {pipeline_id} not found")
        
        exec_id = execution_id or f"exec_{pipeline_id}_{uuid.uuid4().hex[:8]}"
        
        execution = PipelineExecution(
            execution_id=exec_id,
            pipeline_id=pipeline_id,
            input_data=input_data,
            metadata=metadata or {},
        )
        
        async with self._lock:
            self.executions[exec_id] = execution
        
        # Iniciar ejecución en background
        task = asyncio.create_task(self._run_pipeline(execution, pipeline))
        self.active_executions[exec_id] = task
        
        return exec_id
    
    async def _run_pipeline(self, execution: PipelineExecution, pipeline: Pipeline):
        """Ejecutar pipeline."""
        execution.status = PipelineStatus.RUNNING
        execution.started_at = datetime.now()
        
        try:
            current_data = execution.input_data
            
            # Ejecutar stages en orden
            for stage in pipeline.stages:
                # Verificar dependencias
                if stage.dependencies:
                    for dep_id in stage.dependencies:
                        if dep_id not in execution.stages_completed:
                            raise ValueError(f"Dependency {dep_id} not completed")
                
                execution.current_stage = stage.stage_id
                
                # Ejecutar stage
                try:
                    if asyncio.iscoroutinefunction(stage.processor):
                        result = await stage.processor(current_data)
                    else:
                        result = stage.processor(current_data)
                    
                    current_data = result
                    execution.stages_completed.append(stage.stage_id)
                    
                except Exception as e:
                    error_msg = f"Stage {stage.stage_id} failed: {str(e)}"
                    logger.error(error_msg)
                    execution.stages_failed.append(stage.stage_id)
                    execution.error = error_msg
                    execution.status = PipelineStatus.FAILED
                    execution.completed_at = datetime.now()
                    return
            
            execution.status = PipelineStatus.COMPLETED
            execution.output_data = current_data
            execution.completed_at = datetime.now()
            
        except Exception as e:
            error_msg = f"Pipeline execution failed: {str(e)}"
            logger.error(error_msg)
            execution.status = PipelineStatus.FAILED
            execution.error = error_msg
            execution.completed_at = datetime.now()
        
        finally:
            async with self._lock:
                self.execution_history.append(execution)
                if execution.execution_id in self.active_executions:
                    del self.active_executions[execution.execution_id]
    
    async def cancel_execution(self, execution_id: str) -> bool:
        """Cancelar ejecución."""
        execution = self.executions.get(execution_id)
        if not execution:
            return False
        
        if execution.status not in [PipelineStatus.PENDING, PipelineStatus.RUNNING]:
            return False
        
        execution.status = PipelineStatus.CANCELLED
        execution.completed_at = datetime.now()
        
        # Cancelar task si está activo
        if execution_id in self.active_executions:
            self.active_executions[execution_id].cancel()
            del self.active_executions[execution_id]
        
        logger.info(f"Cancelled execution: {execution_id}")
        return True
    
    def get_pipeline(self, pipeline_id: str) -> Optional[Dict[str, Any]]:
        """Obtener información de pipeline."""
        pipeline = self.pipelines.get(pipeline_id)
        if not pipeline:
            return None
        
        return {
            "pipeline_id": pipeline.pipeline_id,
            "name": pipeline.name,
            "description": pipeline.description,
            "stages": [
                {
                    "stage_id": s.stage_id,
                    "stage_type": s.stage_type.value,
                    "name": s.name,
                    "parallel": s.parallel,
                    "dependencies": s.dependencies,
                }
                for s in pipeline.stages
            ],
            "created_at": pipeline.created_at.isoformat(),
            "metadata": pipeline.metadata,
        }
    
    def get_execution(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Obtener información de ejecución."""
        execution = self.executions.get(execution_id)
        if not execution:
            return None
        
        return {
            "execution_id": execution.execution_id,
            "pipeline_id": execution.pipeline_id,
            "status": execution.status.value,
            "stages_completed": execution.stages_completed,
            "stages_failed": execution.stages_failed,
            "current_stage": execution.current_stage,
            "started_at": execution.started_at.isoformat() if execution.started_at else None,
            "completed_at": execution.completed_at.isoformat() if execution.completed_at else None,
            "error": execution.error,
            "has_output": execution.output_data is not None,
        }
    
    def get_pipeline_execution_history(self, pipeline_id: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Obtener historial de ejecuciones."""
        history = list(self.execution_history)
        
        if pipeline_id:
            history = [e for e in history if e.pipeline_id == pipeline_id]
        
        history.sort(key=lambda e: e.started_at or datetime.min, reverse=True)
        
        return [
            {
                "execution_id": e.execution_id,
                "pipeline_id": e.pipeline_id,
                "status": e.status.value,
                "stages_completed": len(e.stages_completed),
                "stages_failed": len(e.stages_failed),
                "started_at": e.started_at.isoformat() if e.started_at else None,
                "completed_at": e.completed_at.isoformat() if e.completed_at else None,
            }
            for e in history[:limit]
        ]
    
    def get_data_pipeline_manager_summary(self) -> Dict[str, Any]:
        """Obtener resumen del gestor."""
        by_status: Dict[str, int] = defaultdict(int)
        
        for execution in self.executions.values():
            by_status[execution.status.value] += 1
        
        return {
            "total_pipelines": len(self.pipelines),
            "total_executions": len(self.executions),
            "active_executions": len(self.active_executions),
            "executions_by_status": dict(by_status),
            "total_history": len(self.execution_history),
        }



