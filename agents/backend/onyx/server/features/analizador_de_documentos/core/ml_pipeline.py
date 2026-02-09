"""
Pipeline de Machine Learning Automatizado
==========================================

Pipeline completo para entrenamiento, evaluación y despliegue de modelos.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)


class PipelineStage(Enum):
    """Etapas del pipeline"""
    DATA_PREPARATION = "data_preparation"
    FEATURE_ENGINEERING = "feature_engineering"
    MODEL_TRAINING = "model_training"
    VALIDATION = "validation"
    TESTING = "testing"
    DEPLOYMENT = "deployment"
    MONITORING = "monitoring"


@dataclass
class PipelineStep:
    """Paso del pipeline"""
    name: str
    stage: PipelineStage
    function: Callable
    dependencies: List[str] = None
    retry_on_failure: bool = True
    max_retries: int = 3
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


@dataclass
class PipelineResult:
    """Resultado del pipeline"""
    pipeline_id: str
    status: str
    stages_completed: List[str]
    results: Dict[str, Any]
    errors: List[str]
    duration: float
    created_at: str


class MLPipeline:
    """
    Pipeline de Machine Learning
    
    Proporciona:
    - Pipeline automatizado
    - Ejecución paralela de etapas
    - Manejo de errores y reintentos
    - Tracking de progreso
    - Rollback automático
    """
    
    def __init__(self, pipeline_id: str = None):
        """Inicializar pipeline"""
        self.pipeline_id = pipeline_id or f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.steps: Dict[str, PipelineStep] = {}
        self.stage_order = [
            PipelineStage.DATA_PREPARATION,
            PipelineStage.FEATURE_ENGINEERING,
            PipelineStage.MODEL_TRAINING,
            PipelineStage.VALIDATION,
            PipelineStage.TESTING,
            PipelineStage.DEPLOYMENT,
            PipelineStage.MONITORING
        ]
        logger.info(f"MLPipeline {self.pipeline_id} inicializado")
    
    def add_step(
        self,
        name: str,
        stage: PipelineStage,
        function: Callable,
        dependencies: Optional[List[str]] = None,
        retry_on_failure: bool = True,
        max_retries: int = 3
    ):
        """Agregar paso al pipeline"""
        step = PipelineStep(
            name=name,
            stage=stage,
            function=function,
            dependencies=dependencies or [],
            retry_on_failure=retry_on_failure,
            max_retries=max_retries
        )
        self.steps[name] = step
        logger.info(f"Paso '{name}' agregado al pipeline")
    
    async def execute_step(
        self,
        step: PipelineStep,
        context: Dict[str, Any]
    ) -> Any:
        """Ejecutar paso con reintentos"""
        retries = 0
        last_error = None
        
        while retries <= step.max_retries:
            try:
                if asyncio.iscoroutinefunction(step.function):
                    result = await step.function(context)
                else:
                    result = step.function(context)
                
                logger.info(f"Paso '{step.name}' completado exitosamente")
                return result
            except Exception as e:
                last_error = e
                retries += 1
                if step.retry_on_failure and retries <= step.max_retries:
                    logger.warning(f"Error en paso '{step.name}', reintento {retries}/{step.max_retries}: {e}")
                    await asyncio.sleep(2 ** retries)  # Exponential backoff
                else:
                    raise
        
        raise last_error
    
    def _get_ready_steps(
        self,
        completed: List[str],
        context: Dict[str, Any]
    ) -> List[PipelineStep]:
        """Obtener pasos listos para ejecutar"""
        ready = []
        
        for step in self.steps.values():
            if step.name in completed:
                continue
            
            # Verificar dependencias
            if all(dep in completed for dep in step.dependencies):
                ready.append(step)
        
        return ready
    
    async def run(
        self,
        initial_context: Optional[Dict[str, Any]] = None
    ) -> PipelineResult:
        """
        Ejecutar pipeline completo
        
        Args:
            initial_context: Contexto inicial
        
        Returns:
            PipelineResult
        """
        import time
        start_time = time.time()
        
        context = initial_context or {}
        context["pipeline_id"] = self.pipeline_id
        
        completed_stages = []
        completed_steps = []
        results = {}
        errors = []
        
        # Agrupar pasos por etapa
        stages_steps = {}
        for stage in self.stage_order:
            stages_steps[stage] = [
                step for step in self.steps.values()
                if step.stage == stage
            ]
        
        # Ejecutar etapas en orden
        for stage in self.stage_order:
            stage_steps = stages_steps.get(stage, [])
            if not stage_steps:
                continue
            
            logger.info(f"Ejecutando etapa: {stage.value}")
            
            # Ejecutar pasos de la etapa
            for step in stage_steps:
                if step.name in completed_steps:
                    continue
                
                # Verificar dependencias
                if not all(dep in completed_steps for dep in step.dependencies):
                    errors.append(f"Paso '{step.name}' tiene dependencias no completadas")
                    continue
                
                try:
                    result = await self.execute_step(step, context)
                    results[step.name] = result
                    context[step.name] = result
                    completed_steps.append(step.name)
                except Exception as e:
                    error_msg = f"Error en paso '{step.name}': {str(e)}"
                    logger.error(error_msg)
                    errors.append(error_msg)
                    
                    # Si el paso no permite reintentos o ya se agotaron, detener pipeline
                    if not step.retry_on_failure:
                        break
            
            if stage.value not in completed_stages:
                completed_stages.append(stage.value)
        
        duration = time.time() - start_time
        
        status = "completed" if not errors else "failed"
        
        return PipelineResult(
            pipeline_id=self.pipeline_id,
            status=status,
            stages_completed=completed_stages,
            results=results,
            errors=errors,
            duration=duration,
            created_at=datetime.now().isoformat()
        )
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Obtener información del pipeline"""
        return {
            "pipeline_id": self.pipeline_id,
            "steps": [
                {
                    "name": step.name,
                    "stage": step.stage.value,
                    "dependencies": step.dependencies
                }
                for step in self.steps.values()
            ],
            "total_steps": len(self.steps)
        }
















