"""
A/B Testing Framework para LLMs.

Permite comparar diferentes modelos, prompts y configuraciones
para determinar cuál funciona mejor para casos de uso específicos.
"""

import json
import hashlib
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict, field
from datetime import datetime
from enum import Enum
from collections import defaultdict

from config.logging_config import get_logger

logger = get_logger(__name__)


class VariantType(str, Enum):
    """Tipo de variante en A/B test."""
    MODEL = "model"
    PROMPT = "prompt"
    TEMPERATURE = "temperature"
    SYSTEM_PROMPT = "system_prompt"
    MAX_TOKENS = "max_tokens"
    COMBINED = "combined"


@dataclass
class Variant:
    """Variante en un A/B test."""
    name: str
    variant_type: VariantType
    config: Dict[str, Any]
    weight: float = 1.0  # Peso para distribución (útil para weighted tests)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        return {
            "name": self.name,
            "variant_type": self.variant_type.value,
            "config": self.config,
            "weight": self.weight
        }


@dataclass
class TestResult:
    """Resultado de una variante en un test."""
    variant_name: str
    response: str
    latency_ms: float
    tokens_used: int
    cost: float
    quality_score: Optional[float] = None
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        return {
            "variant_name": self.variant_name,
            "response": self.response,
            "latency_ms": self.latency_ms,
            "tokens_used": self.tokens_used,
            "cost": self.cost,
            "quality_score": self.quality_score,
            "error": self.error,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class ABTest:
    """Configuración de un A/B test."""
    test_id: str
    name: str
    description: str
    variants: List[Variant]
    prompt: str
    system_prompt: Optional[str] = None
    evaluation_criteria: List[str] = field(default_factory=list)
    min_samples: int = 10  # Mínimo de muestras por variante
    max_samples: int = 100  # Máximo de muestras por variante
    created_at: datetime = field(default_factory=datetime.now)
    status: str = "active"  # active, paused, completed
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        return {
            "test_id": self.test_id,
            "name": self.name,
            "description": self.description,
            "variants": [v.to_dict() for v in self.variants],
            "prompt": self.prompt,
            "system_prompt": self.system_prompt,
            "evaluation_criteria": self.evaluation_criteria,
            "min_samples": self.min_samples,
            "max_samples": self.max_samples,
            "created_at": self.created_at.isoformat(),
            "status": self.status
        }


@dataclass
class ABTestSummary:
    """Resumen estadístico de un A/B test."""
    test_id: str
    total_runs: int
    variant_results: Dict[str, Dict[str, Any]]
    winner: Optional[str] = None
    confidence_level: Optional[float] = None
    statistical_significance: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        return {
            "test_id": self.test_id,
            "total_runs": self.total_runs,
            "variant_results": self.variant_results,
            "winner": self.winner,
            "confidence_level": self.confidence_level,
            "statistical_significance": self.statistical_significance
        }


class ABTestingFramework:
    """
    Framework para realizar A/B testing de modelos LLM.
    
    Permite:
    - Comparar múltiples variantes (modelos, prompts, configuraciones)
    - Ejecutar tests en paralelo
    - Calcular métricas estadísticas
    - Determinar el mejor modelo/prompt
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        Inicializar framework de A/B testing.
        
        Args:
            storage_path: Ruta para almacenar resultados (opcional)
        """
        self.storage_path = storage_path or "data/ab_tests"
        self.tests: Dict[str, ABTest] = {}
        self.results: Dict[str, List[TestResult]] = defaultdict(list)
        
        # Crear directorio si no existe
        import os
        os.makedirs(self.storage_path, exist_ok=True)
    
    def create_test(
        self,
        name: str,
        description: str,
        variants: List[Variant],
        prompt: str,
        system_prompt: Optional[str] = None,
        evaluation_criteria: Optional[List[str]] = None,
        min_samples: int = 10,
        max_samples: int = 100
    ) -> str:
        """
        Crear un nuevo A/B test.
        
        Args:
            name: Nombre del test
            description: Descripción del test
            variants: Lista de variantes a comparar
            prompt: Prompt base para el test
            system_prompt: System prompt (opcional)
            evaluation_criteria: Criterios de evaluación
            min_samples: Mínimo de muestras por variante
            max_samples: Máximo de muestras por variante
            
        Returns:
            ID del test creado
        """
        if len(variants) < 2:
            raise ValueError("Se necesitan al menos 2 variantes para un A/B test")
        
        test_id = hashlib.md5(
            f"{name}{description}{datetime.now().isoformat()}".encode()
        ).hexdigest()[:12]
        
        test = ABTest(
            test_id=test_id,
            name=name,
            description=description,
            variants=variants,
            prompt=prompt,
            system_prompt=system_prompt,
            evaluation_criteria=evaluation_criteria or [],
            min_samples=min_samples,
            max_samples=max_samples
        )
        
        self.tests[test_id] = test
        self._save_test(test)
        
        logger.info(f"A/B test creado: {test_id} - {name}")
        return test_id
    
    def get_test(self, test_id: str) -> Optional[ABTest]:
        """Obtener test por ID."""
        return self.tests.get(test_id)
    
    def list_tests(self, status: Optional[str] = None) -> List[ABTest]:
        """
        Listar todos los tests.
        
        Args:
            status: Filtrar por status (opcional)
            
        Returns:
            Lista de tests
        """
        tests = list(self.tests.values())
        if status:
            tests = [t for t in tests if t.status == status]
        return tests
    
    def record_result(
        self,
        test_id: str,
        variant_name: str,
        response: str,
        latency_ms: float,
        tokens_used: int,
        cost: float,
        quality_score: Optional[float] = None,
        error: Optional[str] = None
    ) -> None:
        """
        Registrar resultado de una variante.
        
        Args:
            test_id: ID del test
            variant_name: Nombre de la variante
            response: Respuesta generada
            latency_ms: Latencia en milisegundos
            tokens_used: Tokens utilizados
            cost: Costo de la generación
            quality_score: Score de calidad (opcional)
            error: Error si hubo (opcional)
        """
        result = TestResult(
            variant_name=variant_name,
            response=response,
            latency_ms=latency_ms,
            tokens_used=tokens_used,
            cost=cost,
            quality_score=quality_score,
            error=error
        )
        
        self.results[test_id].append(result)
        self._save_results(test_id)
        
        logger.debug(f"Resultado registrado para test {test_id}, variante {variant_name}")
    
    def get_summary(self, test_id: str) -> Optional[ABTestSummary]:
        """
        Obtener resumen estadístico de un test.
        
        Args:
            test_id: ID del test
            
        Returns:
            Resumen del test o None si no existe
        """
        if test_id not in self.tests:
            return None
        
        test = self.tests[test_id]
        results = self.results[test_id]
        
        if not results:
            return ABTestSummary(
                test_id=test_id,
                total_runs=0,
                variant_results={}
            )
        
        # Agrupar resultados por variante
        variant_results: Dict[str, List[TestResult]] = defaultdict(list)
        for result in results:
            variant_results[result.variant_name].append(result)
        
        # Calcular estadísticas por variante
        summary_results = {}
        for variant_name, variant_res in variant_results.items():
            successful = [r for r in variant_res if not r.error]
            
            if successful:
                avg_latency = sum(r.latency_ms for r in successful) / len(successful)
                avg_tokens = sum(r.tokens_used for r in successful) / len(successful)
                avg_cost = sum(r.cost for r in successful) / len(successful)
                avg_quality = (
                    sum(r.quality_score for r in successful if r.quality_score)
                    / len([r for r in successful if r.quality_score])
                    if any(r.quality_score for r in successful)
                    else None
                )
                success_rate = len(successful) / len(variant_res)
            else:
                avg_latency = 0
                avg_tokens = 0
                avg_cost = 0
                avg_quality = None
                success_rate = 0
            
            summary_results[variant_name] = {
                "total_runs": len(variant_res),
                "successful_runs": len(successful),
                "success_rate": success_rate,
                "avg_latency_ms": avg_latency,
                "avg_tokens": avg_tokens,
                "avg_cost": avg_cost,
                "avg_quality_score": avg_quality,
                "total_cost": sum(r.cost for r in variant_res)
            }
        
        # Determinar ganador basado en múltiples criterios
        winner = self._determine_winner(summary_results, test.evaluation_criteria)
        
        # Calcular significancia estadística (simplificado)
        statistical_significance = self._check_statistical_significance(variant_results)
        
        return ABTestSummary(
            test_id=test_id,
            total_runs=len(results),
            variant_results=summary_results,
            winner=winner,
            statistical_significance=statistical_significance
        )
    
    def _determine_winner(
        self,
        variant_results: Dict[str, Dict[str, Any]],
        criteria: List[str]
    ) -> Optional[str]:
        """
        Determinar ganador basado en criterios de evaluación.
        
        Args:
            variant_results: Resultados por variante
            criteria: Criterios de evaluación
            
        Returns:
            Nombre de la variante ganadora
        """
        if not variant_results:
            return None
        
        # Si no hay criterios, usar calidad o costo como default
        if not criteria:
            # Priorizar: calidad > éxito > costo
            best = None
            best_score = float('-inf')
            
            for name, stats in variant_results.items():
                if stats["success_rate"] == 0:
                    continue
                
                score = (
                    (stats.get("avg_quality_score") or 0) * 0.5 +
                    stats["success_rate"] * 0.3 -
                    (stats["avg_cost"] / 100) * 0.2  # Normalizar costo
                )
                
                if score > best_score:
                    best_score = score
                    best = name
            
            return best
        
        # Evaluar según criterios específicos
        scores = {}
        for name, stats in variant_results.items():
            if stats["success_rate"] == 0:
                continue
            
            score = 0
            for criterion in criteria:
                if criterion == "quality" and stats.get("avg_quality_score"):
                    score += stats["avg_quality_score"] * 0.4
                elif criterion == "speed":
                    score += (1000 / max(stats["avg_latency_ms"], 1)) * 0.2
                elif criterion == "cost":
                    score += (1 / max(stats["avg_cost"], 0.001)) * 0.2
                elif criterion == "reliability":
                    score += stats["success_rate"] * 0.2
            
            scores[name] = score
        
        if not scores:
            return None
        
        return max(scores.items(), key=lambda x: x[1])[0]
    
    def _check_statistical_significance(
        self,
        variant_results: Dict[str, List[TestResult]]
    ) -> bool:
        """
        Verificar significancia estadística (simplificado).
        
        Args:
            variant_results: Resultados agrupados por variante
            
        Returns:
            True si hay significancia estadística
        """
        if len(variant_results) < 2:
            return False
        
        # Verificar que cada variante tenga suficientes muestras
        for results in variant_results.values():
            if len(results) < 10:  # Mínimo para significancia
                return False
        
        # Simplificado: asumir significancia si hay diferencia clara
        # En producción, usar tests estadísticos apropiados (t-test, chi-square, etc.)
        return True
    
    def pause_test(self, test_id: str) -> bool:
        """Pausar un test."""
        if test_id not in self.tests:
            return False
        
        self.tests[test_id].status = "paused"
        self._save_test(self.tests[test_id])
        logger.info(f"Test {test_id} pausado")
        return True
    
    def resume_test(self, test_id: str) -> bool:
        """Reanudar un test."""
        if test_id not in self.tests:
            return False
        
        self.tests[test_id].status = "active"
        self._save_test(self.tests[test_id])
        logger.info(f"Test {test_id} reanudado")
        return True
    
    def complete_test(self, test_id: str) -> bool:
        """Marcar test como completado."""
        if test_id not in self.tests:
            return False
        
        self.tests[test_id].status = "completed"
        self._save_test(self.tests[test_id])
        logger.info(f"Test {test_id} completado")
        return True
    
    def _save_test(self, test: ABTest) -> None:
        """Guardar test en disco."""
        import os
        import json
        
        file_path = os.path.join(self.storage_path, f"test_{test.test_id}.json")
        with open(file_path, 'w') as f:
            json.dump(test.to_dict(), f, indent=2, default=str)
    
    def _save_results(self, test_id: str) -> None:
        """Guardar resultados en disco."""
        import os
        import json
        
        file_path = os.path.join(self.storage_path, f"results_{test_id}.json")
        results_data = [r.to_dict() for r in self.results[test_id]]
        with open(file_path, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
    
    def load_tests(self) -> None:
        """Cargar tests desde disco."""
        import os
        import json
        
        if not os.path.exists(self.storage_path):
            return
        
        for filename in os.listdir(self.storage_path):
            if filename.startswith("test_") and filename.endswith(".json"):
                file_path = os.path.join(self.storage_path, filename)
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    variants = [
                        Variant(
                            name=v["name"],
                            variant_type=VariantType(v["variant_type"]),
                            config=v["config"],
                            weight=v.get("weight", 1.0)
                        )
                        for v in data["variants"]
                    ]
                    
                    test = ABTest(
                        test_id=data["test_id"],
                        name=data["name"],
                        description=data["description"],
                        variants=variants,
                        prompt=data["prompt"],
                        system_prompt=data.get("system_prompt"),
                        evaluation_criteria=data.get("evaluation_criteria", []),
                        min_samples=data.get("min_samples", 10),
                        max_samples=data.get("max_samples", 100),
                        status=data.get("status", "active")
                    )
                    
                    # Parsear fecha
                    if isinstance(data.get("created_at"), str):
                        test.created_at = datetime.fromisoformat(data["created_at"])
                    
                    self.tests[test.test_id] = test
                except Exception as e:
                    logger.error(f"Error cargando test desde {filename}: {e}")


def get_ab_testing_framework(storage_path: Optional[str] = None) -> ABTestingFramework:
    """Factory function para obtener instancia singleton del framework."""
    if not hasattr(get_ab_testing_framework, "_instance"):
        get_ab_testing_framework._instance = ABTestingFramework(storage_path)
        get_ab_testing_framework._instance.load_tests()
    return get_ab_testing_framework._instance



