"""
Sistema de A/B Testing
======================

Sistema para probar diferentes modelos y configuraciones.
"""

import logging
import random
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class TestStatus(Enum):
    """Estado de test"""
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"


@dataclass
class TestVariant:
    """Variante de test"""
    variant_id: str
    model_config: Dict[str, Any]
    traffic_percentage: float
    metrics: Dict[str, float] = None
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {
                "requests": 0,
                "success": 0,
                "errors": 0,
                "avg_latency": 0.0,
                "accuracy": 0.0
            }


@dataclass
class ABTest:
    """Test A/B"""
    test_id: str
    name: str
    variants: List[TestVariant]
    status: TestStatus
    created_at: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    
    def __post_init__(self):
        if self.start_date is None:
            self.start_date = datetime.now().isoformat()


class ABTestingManager:
    """
    Gestor de A/B Testing
    
    Proporciona:
    - Creación y gestión de tests
    - Asignación de tráfico
    - Análisis de resultados
    - Determinación de ganador
    """
    
    def __init__(self):
        """Inicializar gestor"""
        self.tests: Dict[str, ABTest] = {}
        self.active_tests: Dict[str, ABTest] = {}
        logger.info("ABTestingManager inicializado")
    
    def create_test(
        self,
        test_id: str,
        name: str,
        variants: List[Dict[str, Any]]
    ) -> ABTest:
        """
        Crear nuevo test A/B
        
        Args:
            test_id: ID único del test
            name: Nombre del test
            variants: Lista de variantes con config y tráfico
        
        Returns:
            ABTest creado
        """
        test_variants = [
            TestVariant(
                variant_id=v["variant_id"],
                model_config=v["model_config"],
                traffic_percentage=v.get("traffic_percentage", 50.0)
            )
            for v in variants
        ]
        
        # Normalizar porcentajes de tráfico
        total_traffic = sum(v.traffic_percentage for v in test_variants)
        if total_traffic != 100.0:
            # Normalizar proporcionalmente
            for variant in test_variants:
                variant.traffic_percentage = (variant.traffic_percentage / total_traffic) * 100.0
        
        test = ABTest(
            test_id=test_id,
            name=name,
            variants=test_variants,
            status=TestStatus.ACTIVE,
            created_at=datetime.now().isoformat()
        )
        
        self.tests[test_id] = test
        self.active_tests[test_id] = test
        
        logger.info(f"Test A/B creado: {test_id}")
        return test
    
    def select_variant(self, test_id: str) -> Optional[str]:
        """
        Seleccionar variante para request
        
        Args:
            test_id: ID del test
        
        Returns:
            ID de variante seleccionada o None
        """
        if test_id not in self.active_tests:
            return None
        
        test = self.active_tests[test_id]
        if test.status != TestStatus.ACTIVE:
            return None
        
        # Selección aleatoria basada en porcentajes
        rand = random.random() * 100.0
        cumulative = 0.0
        
        for variant in test.variants:
            cumulative += variant.traffic_percentage
            if rand <= cumulative:
                return variant.variant_id
        
        # Fallback a primera variante
        return test.variants[0].variant_id if test.variants else None
    
    def record_result(
        self,
        test_id: str,
        variant_id: str,
        success: bool,
        latency: float,
        accuracy: Optional[float] = None
    ):
        """Registrar resultado de request"""
        if test_id not in self.tests:
            return
        
        test = self.tests[test_id]
        variant = next((v for v in test.variants if v.variant_id == variant_id), None)
        
        if not variant:
            return
        
        variant.metrics["requests"] += 1
        if success:
            variant.metrics["success"] += 1
        else:
            variant.metrics["errors"] += 1
        
        # Actualizar latencia promedio
        current_avg = variant.metrics["avg_latency"]
        n = variant.metrics["requests"]
        variant.metrics["avg_latency"] = (current_avg * (n - 1) + latency) / n
        
        if accuracy is not None:
            current_acc = variant.metrics["accuracy"]
            variant.metrics["accuracy"] = (current_acc * (n - 1) + accuracy) / n
    
    def get_test_results(self, test_id: str) -> Dict[str, Any]:
        """Obtener resultados del test"""
        if test_id not in self.tests:
            return {}
        
        test = self.tests[test_id]
        
        results = {
            "test_id": test_id,
            "name": test.name,
            "status": test.status.value,
            "variants": []
        }
        
        for variant in test.variants:
            metrics = variant.metrics
            success_rate = (metrics["success"] / metrics["requests"] * 100) if metrics["requests"] > 0 else 0
            
            results["variants"].append({
                "variant_id": variant.variant_id,
                "traffic_percentage": variant.traffic_percentage,
                "metrics": {
                    "requests": metrics["requests"],
                    "success_rate": success_rate,
                    "error_rate": (metrics["errors"] / metrics["requests"] * 100) if metrics["requests"] > 0 else 0,
                    "avg_latency": metrics["avg_latency"],
                    "accuracy": metrics["accuracy"]
                }
            })
        
        # Determinar ganador
        if len(test.variants) >= 2 and test.status == TestStatus.ACTIVE:
            sorted_variants = sorted(
                results["variants"],
                key=lambda x: x["metrics"]["accuracy"] * x["metrics"]["success_rate"] / 100,
                reverse=True
            )
            if len(sorted_variants) > 0:
                results["winner"] = sorted_variants[0]["variant_id"]
        
        return results
    
    def stop_test(self, test_id: str):
        """Detener test"""
        if test_id in self.active_tests:
            self.active_tests[test_id].status = TestStatus.COMPLETED
            del self.active_tests[test_id]
            logger.info(f"Test {test_id} detenido")


# Instancia global
_ab_testing_manager: Optional[ABTestingManager] = None


def get_ab_testing_manager() -> ABTestingManager:
    """Obtener instancia global del gestor"""
    global _ab_testing_manager
    if _ab_testing_manager is None:
        _ab_testing_manager = ABTestingManager()
    return _ab_testing_manager
















