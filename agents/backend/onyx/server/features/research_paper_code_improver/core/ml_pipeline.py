"""
ML Pipeline - Pipeline de machine learning
===========================================
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Callable
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class MLStageType(Enum):
    """Tipos de etapas ML"""
    DATA_LOADING = "data_loading"
    PREPROCESSING = "preprocessing"
    FEATURE_ENGINEERING = "feature_engineering"
    TRAINING = "training"
    VALIDATION = "validation"
    EVALUATION = "evaluation"
    DEPLOYMENT = "deployment"


@dataclass
class MLStage:
    """Etapa del pipeline ML"""
    id: str
    name: str
    stage_type: MLStageType
    processor: Callable
    dependencies: List[str] = field(default_factory=list)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    status: str = "pending"
    result: Any = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


@dataclass
class MLPipeline:
    """Pipeline de machine learning"""
    id: str
    name: str
    description: str
    stages: List[MLStage]
    model_type: str
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class MLPipelineManager:
    """Gestor de pipelines de machine learning"""
    
    def __init__(self):
        self.pipelines: Dict[str, MLPipeline] = {}
        self.executions: Dict[str, Dict[str, Any]] = {}
        self.models: Dict[str, Any] = {}  # model_id -> trained model
    
    def create_pipeline(
        self,
        pipeline_id: str,
        name: str,
        description: str,
        stages: List[MLStage],
        model_type: str = "generic"
    ) -> MLPipeline:
        """Crea un nuevo pipeline ML"""
        pipeline = MLPipeline(
            id=pipeline_id,
            name=name,
            description=description,
            stages=stages,
            model_type=model_type
        )
        self.pipelines[pipeline_id] = pipeline
        logger.info(f"ML Pipeline {pipeline_id} creado")
        return pipeline
    
    async def execute_pipeline(
        self,
        pipeline_id: str,
        input_data: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Ejecuta un pipeline ML"""
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
            "metrics": {},
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
                stage_result = await self._execute_ml_stage(
                    stage,
                    current_data,
                    stage_results,
                    context
                )
                
                stage_results[stage.id] = stage_result
                current_data = stage_result
                
                execution["stages"][stage.id] = {
                    "status": stage.status,
                    "metrics": stage.metrics,
                    "error": stage.error
                }
                
                # Agregar métricas al nivel de ejecución
                execution["metrics"].update(stage.metrics)
            
            execution["status"] = "completed"
            execution["completed_at"] = datetime.now().isoformat()
            execution["result"] = current_data
            
            # Si hay un modelo entrenado, guardarlo
            if "model" in current_data:
                model_id = f"{pipeline_id}_{execution_id}"
                self.models[model_id] = current_data["model"]
                execution["model_id"] = model_id
            
            return execution
        except Exception as e:
            execution["status"] = "failed"
            execution["error"] = str(e)
            execution["completed_at"] = datetime.now().isoformat()
            logger.error(f"Error ejecutando ML pipeline {pipeline_id}: {e}")
            raise
    
    async def _execute_ml_stage(
        self,
        stage: MLStage,
        input_data: Any,
        previous_results: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> Any:
        """Ejecuta una etapa ML"""
        stage.status = "running"
        stage.started_at = datetime.now()
        
        try:
            # Ejecutar processor con hyperparameters
            result = await stage.processor(
                input_data,
                previous_results,
                context,
                stage.hyperparameters
            )
            
            stage.result = result
            stage.status = "completed"
            stage.completed_at = datetime.now()
            
            # Extraer métricas si están en el resultado
            if isinstance(result, dict) and "metrics" in result:
                stage.metrics.update(result["metrics"])
            
            return result
        except Exception as e:
            stage.status = "failed"
            stage.error = str(e)
            stage.completed_at = datetime.now()
            logger.error(f"Error ejecutando ML stage {stage.id}: {e}")
            raise
    
    def get_model(self, model_id: str) -> Optional[Any]:
        """Obtiene un modelo entrenado"""
        return self.models.get(model_id)
    
    def list_models(self) -> List[Dict[str, Any]]:
        """Lista todos los modelos entrenados"""
        return [
            {
                "model_id": model_id,
                "created_at": "N/A"  # En producción, guardar metadata
            }
            for model_id in self.models.keys()
        ]
    
    def get_pipeline_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Obtiene el estado de una ejecución de pipeline"""
        return self.executions.get(execution_id)
    
    def list_pipelines(self) -> List[Dict[str, Any]]:
        """Lista todos los pipelines ML"""
        return [
            {
                "id": p.id,
                "name": p.name,
                "description": p.description,
                "model_type": p.model_type,
                "stages_count": len(p.stages)
            }
            for p in self.pipelines.values()
        ]




