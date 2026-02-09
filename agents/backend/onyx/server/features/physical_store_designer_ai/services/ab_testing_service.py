"""
A/B Testing Service - Sistema de A/B testing
"""

import logging
import random
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class TestStatus(str, Enum):
    """Estados de test"""
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"


class ABTestingService:
    """Servicio para A/B testing"""
    
    def __init__(self):
        self.tests: Dict[str, Dict[str, Any]] = {}
        self.results: Dict[str, List[Dict[str, Any]]] = {}
    
    def create_test(
        self,
        test_name: str,
        variants: List[Dict[str, Any]],
        traffic_split: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """Crear test A/B"""
        
        test_id = f"test_{len(self.tests) + 1}"
        
        # Validar que traffic_split suma 100%
        if traffic_split:
            total = sum(traffic_split.values())
            if abs(total - 100.0) > 0.01:
                raise ValueError("Traffic split debe sumar 100%")
        else:
            # Split equitativo por defecto
            split_per_variant = 100.0 / len(variants)
            traffic_split = {f"variant_{i+1}": split_per_variant for i in range(len(variants))}
        
        test = {
            "test_id": test_id,
            "test_name": test_name,
            "variants": variants,
            "traffic_split": traffic_split,
            "status": TestStatus.DRAFT.value,
            "created_at": datetime.now().isoformat(),
            "started_at": None,
            "completed_at": None,
            "participants": {},
            "results": {}
        }
        
        self.tests[test_id] = test
        
        return test
    
    def start_test(self, test_id: str) -> bool:
        """Iniciar test"""
        test = self.tests.get(test_id)
        
        if not test:
            return False
        
        test["status"] = TestStatus.RUNNING.value
        test["started_at"] = datetime.now().isoformat()
        
        return True
    
    def assign_variant(
        self,
        test_id: str,
        user_id: str
    ) -> Optional[str]:
        """Asignar variante a usuario"""
        test = self.tests.get(test_id)
        
        if not test or test["status"] != TestStatus.RUNNING.value:
            return None
        
        # Si ya tiene asignación, mantenerla
        if user_id in test["participants"]:
            return test["participants"][user_id]
        
        # Asignar según traffic split
        variant = self._select_variant(test["traffic_split"])
        test["participants"][user_id] = variant
        
        return variant
    
    def _select_variant(self, traffic_split: Dict[str, float]) -> str:
        """Seleccionar variante según traffic split"""
        rand = random.random() * 100
        cumulative = 0
        
        for variant, percentage in traffic_split.items():
            cumulative += percentage
            if rand <= cumulative:
                return variant
        
        # Fallback al último
        return list(traffic_split.keys())[-1]
    
    def record_conversion(
        self,
        test_id: str,
        user_id: str,
        conversion_type: str = "click"
    ) -> bool:
        """Registrar conversión"""
        test = self.tests.get(test_id)
        
        if not test:
            return False
        
        variant = test["participants"].get(user_id)
        if not variant:
            return False
        
        if test_id not in self.results:
            self.results[test_id] = []
        
        conversion = {
            "test_id": test_id,
            "user_id": user_id,
            "variant": variant,
            "conversion_type": conversion_type,
            "timestamp": datetime.now().isoformat()
        }
        
        self.results[test_id].append(conversion)
        
        return True
    
    def get_test_results(self, test_id: str) -> Dict[str, Any]:
        """Obtener resultados del test"""
        test = self.tests.get(test_id)
        
        if not test:
            return {"error": "Test no encontrado"}
        
        conversions = self.results.get(test_id, [])
        
        # Agrupar por variante
        variant_stats = {}
        for variant in test["variants"]:
            variant_id = variant.get("id", f"variant_{test['variants'].index(variant) + 1}")
            variant_conversions = [c for c in conversions if c["variant"] == variant_id]
            variant_participants = len([u for u, v in test["participants"].items() if v == variant_id])
            
            conversion_rate = (len(variant_conversions) / variant_participants * 100) if variant_participants > 0 else 0
            
            variant_stats[variant_id] = {
                "participants": variant_participants,
                "conversions": len(variant_conversions),
                "conversion_rate": round(conversion_rate, 2)
            }
        
        # Determinar ganador
        winner = max(variant_stats.items(), key=lambda x: x[1]["conversion_rate"])[0] if variant_stats else None
        
        return {
            "test_id": test_id,
            "test_name": test["test_name"],
            "status": test["status"],
            "total_participants": len(test["participants"]),
            "total_conversions": len(conversions),
            "variant_stats": variant_stats,
            "winner": winner,
            "confidence": self._calculate_confidence(variant_stats) if variant_stats else 0
        }
    
    def _calculate_confidence(self, variant_stats: Dict[str, Dict[str, Any]]) -> float:
        """Calcular confianza estadística"""
        # Simplificado - en producción usar test estadístico real
        if len(variant_stats) < 2:
            return 0.0
        
        rates = [v["conversion_rate"] for v in variant_stats.values()]
        max_rate = max(rates)
        min_rate = min(rates)
        
        if max_rate == 0:
            return 0.0
        
        difference = (max_rate - min_rate) / max_rate
        
        # Confianza basada en diferencia (simplificado)
        confidence = min(0.95, difference * 2)
        
        return round(confidence, 2)
    
    def complete_test(self, test_id: str) -> bool:
        """Completar test"""
        test = self.tests.get(test_id)
        
        if not test:
            return False
        
        test["status"] = TestStatus.COMPLETED.value
        test["completed_at"] = datetime.now().isoformat()
        
        return True




