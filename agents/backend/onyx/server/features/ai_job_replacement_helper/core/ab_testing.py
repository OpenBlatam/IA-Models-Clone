"""
A/B Testing Service - Sistema de A/B testing
============================================

Sistema para realizar pruebas A/B y optimización.
"""

import logging
import random
from typing import Dict, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ABTest:
    """Test A/B"""
    id: str
    name: str
    description: str
    variants: Dict[str, Any]  # variant_name -> config
    traffic_split: Dict[str, float]  # variant_name -> percentage
    active: bool = True
    start_date: datetime = field(default_factory=datetime.now)
    end_date: Optional[datetime] = None
    results: Dict[str, Dict[str, Any]] = field(default_factory=dict)


class ABTestingService:
    """Servicio de A/B testing"""
    
    def __init__(self):
        """Inicializar servicio"""
        self.tests: Dict[str, ABTest] = {}
        self.user_assignments: Dict[str, Dict[str, str]] = {}  # test_id -> {user_id -> variant}
        logger.info("ABTestingService initialized")
    
    def create_test(
        self,
        test_id: str,
        name: str,
        description: str,
        variants: Dict[str, Any],
        traffic_split: Optional[Dict[str, float]] = None
    ) -> ABTest:
        """Crear test A/B"""
        # Normalizar traffic split
        if not traffic_split:
            # Split 50/50 por defecto
            variant_names = list(variants.keys())
            traffic_split = {
                name: 100.0 / len(variant_names)
                for name in variant_names
            }
        
        test = ABTest(
            id=test_id,
            name=name,
            description=description,
            variants=variants,
            traffic_split=traffic_split,
        )
        
        self.tests[test_id] = test
        self.user_assignments[test_id] = {}
        
        logger.info(f"AB Test created: {test_id}")
        return test
    
    def assign_variant(self, test_id: str, user_id: str) -> str:
        """Asignar variante a usuario"""
        test = self.tests.get(test_id)
        if not test or not test.active:
            return "default"
        
        # Verificar si ya tiene asignación
        if test_id in self.user_assignments and user_id in self.user_assignments[test_id]:
            return self.user_assignments[test_id][user_id]
        
        # Asignar variante basado en traffic split
        rand = random.random() * 100
        cumulative = 0
        
        for variant_name, percentage in test.traffic_split.items():
            cumulative += percentage
            if rand <= cumulative:
                self.user_assignments[test_id][user_id] = variant_name
                logger.info(f"User {user_id} assigned to variant {variant_name} in test {test_id}")
                return variant_name
        
        # Fallback
        default_variant = list(test.variants.keys())[0]
        self.user_assignments[test_id][user_id] = default_variant
        return default_variant
    
    def track_conversion(
        self,
        test_id: str,
        user_id: str,
        conversion_type: str,
        value: Optional[float] = None
    ):
        """Trackear conversión"""
        test = self.tests.get(test_id)
        if not test:
            return
        
        variant = self.user_assignments.get(test_id, {}).get(user_id, "default")
        
        if variant not in test.results:
            test.results[variant] = {
                "conversions": {},
                "total_users": 0,
            }
        
        if conversion_type not in test.results[variant]["conversions"]:
            test.results[variant]["conversions"][conversion_type] = {
                "count": 0,
                "total_value": 0.0,
            }
        
        test.results[variant]["conversions"][conversion_type]["count"] += 1
        if value:
            test.results[variant]["conversions"][conversion_type]["total_value"] += value
    
    def get_test_results(self, test_id: str) -> Dict[str, Any]:
        """Obtener resultados del test"""
        test = self.tests.get(test_id)
        if not test:
            return {}
        
        # Calcular estadísticas
        stats = {}
        for variant_name, variant_results in test.results.items():
            total_conversions = sum(
                conv["count"]
                for conv in variant_results["conversions"].values()
            )
            
            stats[variant_name] = {
                "users": variant_results["total_users"],
                "conversions": total_conversions,
                "conversion_rate": (
                    total_conversions / variant_results["total_users"] * 100
                    if variant_results["total_users"] > 0 else 0
                ),
                "details": variant_results["conversions"],
            }
        
        return {
            "test_id": test_id,
            "name": test.name,
            "results": stats,
        }




