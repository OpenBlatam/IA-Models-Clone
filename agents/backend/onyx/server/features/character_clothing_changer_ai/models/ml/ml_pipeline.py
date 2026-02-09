"""
ML Pipeline System
==================
Sistema de pipeline de machine learning para entrenamiento y evaluación
"""

import time
import json
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum


class PipelineStage(Enum):
    """Etapas del pipeline"""
    DATA_LOADING = "data_loading"
    PREPROCESSING = "preprocessing"
    FEATURE_EXTRACTION = "feature_extraction"
    TRAINING = "training"
    VALIDATION = "validation"
    EVALUATION = "evaluation"
    DEPLOYMENT = "deployment"


class PipelineStatus(Enum):
    """Estados del pipeline"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class PipelineStageConfig:
    """Configuración de etapa"""
    stage: PipelineStage
    config: Dict[str, Any]
    timeout: float = 3600.0
    retry_count: int = 3


@dataclass
class PipelineExecution:
    """Ejecución de pipeline"""
    id: str
    pipeline_id: str
    started_at: float
    completed_at: Optional[float] = None
    status: PipelineStatus = PipelineStatus.PENDING
    current_stage: Optional[PipelineStage] = None
    results: Dict[str, Any] = None
    errors: List[str] = None
    
    def __post_init__(self):
        if self.results is None:
            self.results = {}
        if self.errors is None:
            self.errors = []


@dataclass
class MLPipeline:
    """Pipeline de ML"""
    id: str
    name: str
    description: str
    stages: List[PipelineStageConfig]
    created_at: float
    last_run: Optional[float] = None
    run_count: int = 0
    success_count: int = 0
    failure_count: int = 0


class MLPipelineSystem:
    """
    Sistema de pipeline de ML
    """
    
    def __init__(self):
        self.pipelines: Dict[str, MLPipeline] = {}
        self.executions: Dict[str, List[PipelineExecution]] = {}
        self.stage_handlers: Dict[PipelineStage, Callable] = {}
        self._init_default_handlers()
    
    def _init_default_handlers(self):
        """Inicializar handlers por defecto"""
        # Placeholder handlers
        self.stage_handlers[PipelineStage.DATA_LOADING] = self._default_data_loading
        self.stage_handlers[PipelineStage.PREPROCESSING] = self._default_preprocessing
        self.stage_handlers[PipelineStage.FEATURE_EXTRACTION] = self._default_feature_extraction
        self.stage_handlers[PipelineStage.TRAINING] = self._default_training
        self.stage_handlers[PipelineStage.VALIDATION] = self._default_validation
        self.stage_handlers[PipelineStage.EVALUATION] = self._default_evaluation
    
    def create_pipeline(
        self,
        name: str,
        description: str,
        stages: List[Dict[str, Any]]
    ) -> MLPipeline:
        """
        Crear pipeline
        
        Args:
            name: Nombre del pipeline
            description: Descripción
            stages: Lista de etapas
        """
        pipeline_id = f"pipeline_{int(time.time())}"
        
        pipeline_stages = [
            PipelineStageConfig(
                stage=PipelineStage(s['stage']),
                config=s.get('config', {}),
                timeout=s.get('timeout', 3600.0),
                retry_count=s.get('retry_count', 3)
            )
            for s in stages
        ]
        
        pipeline = MLPipeline(
            id=pipeline_id,
            name=name,
            description=description,
            stages=pipeline_stages,
            created_at=time.time()
        )
        
        self.pipelines[pipeline_id] = pipeline
        return pipeline
    
    def register_stage_handler(
        self,
        stage: PipelineStage,
        handler: Callable
    ):
        """Registrar handler para etapa"""
        self.stage_handlers[stage] = handler
    
    def execute_pipeline(
        self,
        pipeline_id: str,
        input_data: Optional[Dict[str, Any]] = None
    ) -> PipelineExecution:
        """
        Ejecutar pipeline
        
        Args:
            pipeline_id: ID del pipeline
            input_data: Datos de entrada
        """
        if pipeline_id not in self.pipelines:
            raise ValueError(f"Pipeline {pipeline_id} not found")
        
        pipeline = self.pipelines[pipeline_id]
        
        execution_id = f"exec_{int(time.time())}"
        execution = PipelineExecution(
            id=execution_id,
            pipeline_id=pipeline_id,
            started_at=time.time(),
            status=PipelineStatus.RUNNING
        )
        
        if pipeline_id not in self.executions:
            self.executions[pipeline_id] = []
        self.executions[pipeline_id].append(execution)
        
        try:
            current_data = input_data or {}
            
            # Ejecutar etapas en orden
            for stage_config in pipeline.stages:
                execution.current_stage = stage_config.stage
                
                handler = self.stage_handlers.get(stage_config.stage)
                if not handler:
                    raise ValueError(f"No handler for stage {stage_config.stage}")
                
                # Ejecutar con retry
                result = None
                last_error = None
                
                for attempt in range(stage_config.retry_count):
                    try:
                        result = handler(current_data, stage_config.config)
                        break
                    except Exception as e:
                        last_error = e
                        if attempt < stage_config.retry_count - 1:
                            time.sleep(2 ** attempt)  # Exponential backoff
                
                if result is None:
                    raise Exception(f"Stage {stage_config.stage} failed: {last_error}")
                
                # Guardar resultado
                execution.results[stage_config.stage.value] = result
                current_data = result  # Pasar resultado a siguiente etapa
            
            execution.status = PipelineStatus.COMPLETED
            execution.completed_at = time.time()
            pipeline.success_count += 1
            
        except Exception as e:
            execution.status = PipelineStatus.FAILED
            execution.completed_at = time.time()
            execution.errors.append(str(e))
            pipeline.failure_count += 1
        
        finally:
            pipeline.last_run = time.time()
            pipeline.run_count += 1
        
        return execution
    
    def _default_data_loading(self, data: Dict, config: Dict) -> Dict:
        """Handler por defecto para carga de datos"""
        # Placeholder
        return {'data': 'loaded', 'samples': 1000}
    
    def _default_preprocessing(self, data: Dict, config: Dict) -> Dict:
        """Handler por defecto para preprocesamiento"""
        # Placeholder
        return {'data': 'preprocessed', 'samples': data.get('samples', 1000)}
    
    def _default_feature_extraction(self, data: Dict, config: Dict) -> Dict:
        """Handler por defecto para extracción de features"""
        # Placeholder
        return {'features': 'extracted', 'feature_count': 128}
    
    def _default_training(self, data: Dict, config: Dict) -> Dict:
        """Handler por defecto para entrenamiento"""
        # Placeholder
        return {'model': 'trained', 'accuracy': 0.95}
    
    def _default_validation(self, data: Dict, config: Dict) -> Dict:
        """Handler por defecto para validación"""
        # Placeholder
        return {'validated': True, 'score': 0.92}
    
    def _default_evaluation(self, data: Dict, config: Dict) -> Dict:
        """Handler por defecto para evaluación"""
        # Placeholder
        return {'evaluated': True, 'metrics': {'precision': 0.94, 'recall': 0.93}}
    
    def get_pipeline_statistics(self, pipeline_id: str) -> Dict[str, Any]:
        """Obtener estadísticas de pipeline"""
        if pipeline_id not in self.pipelines:
            return {}
        
        pipeline = self.pipelines[pipeline_id]
        executions = self.executions.get(pipeline_id, [])
        
        successful = len([e for e in executions if e.status == PipelineStatus.COMPLETED])
        failed = len([e for e in executions if e.status == PipelineStatus.FAILED])
        
        avg_duration = 0
        if executions:
            durations = [
                (e.completed_at or time.time()) - e.started_at
                for e in executions if e.completed_at
            ]
            if durations:
                avg_duration = sum(durations) / len(durations)
        
        return {
            'pipeline_id': pipeline_id,
            'name': pipeline.name,
            'total_runs': pipeline.run_count,
            'successful_runs': successful,
            'failed_runs': failed,
            'success_rate': successful / len(executions) if executions else 0,
            'average_duration': avg_duration,
            'last_run': pipeline.last_run,
            'created_at': pipeline.created_at
        }


# Instancia global
ml_pipeline = MLPipelineSystem()

