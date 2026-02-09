"""
Pipeline A/B Testing
====================

Sistema de A/B testing para pipelines.
"""

import logging
import random
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum

from .pipeline import Pipeline
from .stages import PipelineStage

logger = logging.getLogger(__name__)


class Variant(Enum):
    """Variante de A/B test."""
    A = "A"
    B = "B"
    CONTROL = "control"


@dataclass
class ABTestResult:
    """
    Resultado de A/B test.
    """
    variant: Variant
    pipeline_name: str
    result: Any
    execution_time: float
    success: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


class ABTestManager:
    """
    Gestor de A/B tests para pipelines.
    """
    
    def __init__(self):
        """Inicializar gestor."""
        self.tests: Dict[str, Dict[str, Any]] = {}
        self.results: List[ABTestResult] = []
    
    def register_test(
        self,
        test_name: str,
        variant_a: Pipeline,
        variant_b: Pipeline,
        traffic_split: float = 0.5
    ) -> None:
        """
        Registrar A/B test.
        
        Args:
            test_name: Nombre del test
            variant_a: Pipeline variante A
            variant_b: Pipeline variante B
            traffic_split: División de tráfico (0.5 = 50/50)
        """
        self.tests[test_name] = {
            'variant_a': variant_a,
            'variant_b': variant_b,
            'traffic_split': traffic_split
        }
        logger.info(f"A/B test registrado: {test_name}")
    
    def select_variant(self, test_name: str) -> Variant:
        """
        Seleccionar variante para test.
        
        Args:
            test_name: Nombre del test
            
        Returns:
            Variante seleccionada
        """
        if test_name not in self.tests:
            raise ValueError(f"Test '{test_name}' no registrado")
        
        traffic_split = self.tests[test_name]['traffic_split']
        return Variant.A if random.random() < traffic_split else Variant.B
    
    def run_test(
        self,
        test_name: str,
        data: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> ABTestResult:
        """
        Ejecutar A/B test.
        
        Args:
            test_name: Nombre del test
            data: Datos a procesar
            context: Contexto
            
        Returns:
            Resultado del test
        """
        import time
        
        if test_name not in self.tests:
            raise ValueError(f"Test '{test_name}' no registrado")
        
        # Seleccionar variante
        variant = self.select_variant(test_name)
        
        # Ejecutar pipeline correspondiente
        pipeline = (
            self.tests[test_name]['variant_a'] if variant == Variant.A
            else self.tests[test_name]['variant_b']
        )
        
        start_time = time.time()
        success = False
        result = None
        
        try:
            result = pipeline.process(data, context)
            success = True
        except Exception as e:
            logger.error(f"Error en variante {variant.value}: {e}")
            result = None
        
        execution_time = time.time() - start_time
        
        # Guardar resultado
        test_result = ABTestResult(
            variant=variant,
            pipeline_name=pipeline.name,
            result=result,
            execution_time=execution_time,
            success=success,
            metadata={'test_name': test_name}
        )
        
        self.results.append(test_result)
        return test_result
    
    def get_test_statistics(self, test_name: str) -> Dict[str, Any]:
        """
        Obtener estadísticas de test.
        
        Args:
            test_name: Nombre del test
            
        Returns:
            Estadísticas
        """
        test_results = [
            r for r in self.results
            if r.metadata.get('test_name') == test_name
        ]
        
        if not test_results:
            return {}
        
        variant_a_results = [r for r in test_results if r.variant == Variant.A]
        variant_b_results = [r for r in test_results if r.variant == Variant.B]
        
        def calculate_stats(results: List[ABTestResult]) -> Dict[str, Any]:
            if not results:
                return {}
            
            success_count = sum(1 for r in results if r.success)
            avg_time = sum(r.execution_time for r in results) / len(results)
            
            return {
                'count': len(results),
                'success_count': success_count,
                'success_rate': success_count / len(results),
                'avg_execution_time': avg_time
            }
        
        return {
            'variant_a': calculate_stats(variant_a_results),
            'variant_b': calculate_stats(variant_b_results),
            'total_runs': len(test_results)
        }


class ABTestPipeline(Pipeline):
    """
    Pipeline con soporte para A/B testing.
    """
    
    def __init__(
        self,
        name: str = "pipeline",
        ab_test_manager: Optional[ABTestManager] = None,
        **kwargs
    ):
        """
        Inicializar pipeline con A/B testing.
        
        Args:
            name: Nombre del pipeline
            ab_test_manager: Gestor de A/B tests
            **kwargs: Argumentos adicionales
        """
        super().__init__(name, **kwargs)
        self.ab_test_manager = ab_test_manager or ABTestManager()
    
    def create_variant(
        self,
        variant_name: str,
        stages: Optional[List[PipelineStage]] = None
    ) -> Pipeline:
        """
        Crear variante del pipeline.
        
        Args:
            variant_name: Nombre de la variante
            stages: Etapas (None para usar las actuales)
            
        Returns:
            Pipeline variante
        """
        variant = Pipeline(f"{self.name}_{variant_name}")
        variant.stages = stages or list(self.stages)
        variant.middleware = list(self.middleware)
        variant.context = self.context.copy()
        return variant

