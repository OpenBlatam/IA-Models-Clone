"""
Data Pipeline - Pipeline de procesamiento de datos
===================================================
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Callable, Iterator
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class StageStatus(Enum):
    """Estados de una etapa"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class PipelineStage:
    """Etapa del pipeline"""
    id: str
    name: str
    processor: Callable
    dependencies: List[str] = field(default_factory=list)
    retry_count: int = 3
    timeout: Optional[float] = None
    batch_size: Optional[int] = None
    status: StageStatus = StageStatus.PENDING
    result: Any = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    records_processed: int = 0


@dataclass
class Pipeline:
    """Definición de un pipeline"""
    id: str
    name: str
    description: str
    stages: List[PipelineStage]
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class DataPipeline:
    """Pipeline de procesamiento de datos"""
    
    def __init__(self):
        self.pipelines: Dict[str, Pipeline] = {}
        self.executions: Dict[str, Dict[str, Any]] = {}
    
    def create_pipeline(
        self,
        pipeline_id: str,
        name: str,
        description: str,
        stages: List[PipelineStage]
    ) -> Pipeline:
        """Crea un nuevo pipeline"""
        pipeline = Pipeline(
            id=pipeline_id,
            name=name,
            description=description,
            stages=stages
        )
        self.pipelines[pipeline_id] = pipeline
        logger.info(f"Pipeline {pipeline_id} creado")
        return pipeline
    
    async def execute_pipeline(
        self,
        pipeline_id: str,
        input_data: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Ejecuta un pipeline"""
        if pipeline_id not in self.pipelines:
            raise ValueError(f"Pipeline {pipeline_id} no encontrado")
        
        pipeline = self.pipelines[pipeline_id]
        execution_id = f"{pipeline_id}_{datetime.now().timestamp()}"
        
        execution = {
            "execution_id": execution_id,
            "pipeline_id": pipeline_id,
            "status": "running",
            "started_at": datetime.now().isoformat(),
            "stages": {},
            "context": context or {}
        }
        self.executions[execution_id] = execution
        
        try:
            current_data = input_data
            stage_results = {}
            
            # Ejecutar stages en orden
            for stage in pipeline.stages:
                # Verificar dependencias
                if not all(dep in stage_results for dep in stage.dependencies):
                    raise ValueError(f"Stage {stage.id} tiene dependencias no satisfechas")
                
                # Ejecutar stage
                stage_result = await self._execute_stage(
                    stage,
                    current_data,
                    stage_results,
                    context
                )
                
                stage_results[stage.id] = stage_result
                current_data = stage_result
                execution["stages"][stage.id] = {
                    "status": stage.status.value,
                    "records_processed": stage.records_processed,
                    "error": stage.error
                }
            
            execution["status"] = "completed"
            execution["completed_at"] = datetime.now().isoformat()
            execution["result"] = current_data
            
            return execution
        except Exception as e:
            execution["status"] = "failed"
            execution["error"] = str(e)
            execution["completed_at"] = datetime.now().isoformat()
            logger.error(f"Error ejecutando pipeline {pipeline_id}: {e}")
            raise
    
    async def _execute_stage(
        self,
        stage: PipelineStage,
        input_data: Any,
        previous_results: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> Any:
        """Ejecuta una etapa del pipeline"""
        stage.status = StageStatus.RUNNING
        stage.started_at = datetime.now()
        
        try:
            # Si es un iterable y hay batch_size, procesar en lotes
            if stage.batch_size and hasattr(input_data, '__iter__'):
                results = []
                batch = []
                
                for item in input_data:
                    batch.append(item)
                    if len(batch) >= stage.batch_size:
                        batch_result = await self._process_batch(stage, batch, previous_results, context)
                        results.extend(batch_result)
                        batch = []
                        stage.records_processed += len(batch_result)
                
                # Procesar batch final
                if batch:
                    batch_result = await self._process_batch(stage, batch, previous_results, context)
                    results.extend(batch_result)
                    stage.records_processed += len(batch_result)
                
                stage.result = results
            else:
                # Procesar todo de una vez
                if stage.timeout:
                    result = await asyncio.wait_for(
                        stage.processor(input_data, previous_results, context),
                        timeout=stage.timeout
                    )
                else:
                    result = await stage.processor(input_data, previous_results, context)
                
                stage.result = result
                if isinstance(input_data, list):
                    stage.records_processed = len(input_data)
                else:
                    stage.records_processed = 1
            
            stage.status = StageStatus.COMPLETED
            stage.completed_at = datetime.now()
            return stage.result
        except Exception as e:
            stage.status = StageStatus.FAILED
            stage.error = str(e)
            stage.completed_at = datetime.now()
            logger.error(f"Error ejecutando stage {stage.id}: {e}")
            raise
    
    async def _process_batch(
        self,
        stage: PipelineStage,
        batch: List[Any],
        previous_results: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> List[Any]:
        """Procesa un lote de datos"""
        # Ejecutar con retry
        last_error = None
        for attempt in range(stage.retry_count):
            try:
                if stage.timeout:
                    result = await asyncio.wait_for(
                        stage.processor(batch, previous_results, context),
                        timeout=stage.timeout
                    )
                else:
                    result = await stage.processor(batch, previous_results, context)
                
                return result if isinstance(result, list) else [result]
            except Exception as e:
                last_error = e
                if attempt < stage.retry_count - 1:
                    await asyncio.sleep(2 ** attempt)
        
        raise last_error
    
    def get_pipeline_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Obtiene el estado de una ejecución de pipeline"""
        return self.executions.get(execution_id)
    
    def list_pipelines(self) -> List[Dict[str, Any]]:
        """Lista todos los pipelines"""
        return [
            {
                "id": p.id,
                "name": p.name,
                "description": p.description,
                "stages_count": len(p.stages)
            }
            for p in self.pipelines.values()
        ]
    
    def get_pipeline_executions(
        self,
        pipeline_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Obtiene ejecuciones de pipelines"""
        executions = list(self.executions.values())
        
        if pipeline_id:
            executions = [e for e in executions if e["pipeline_id"] == pipeline_id]
        
        return executions[-limit:]




