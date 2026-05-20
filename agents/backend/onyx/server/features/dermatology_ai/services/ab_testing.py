"""
Sistema de A/B Testing
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import random
import uuid
import statistics


class Variant(str, Enum):
    """Variantes de test"""
    A = "A"
    B = "B"
    C = "C"
    CONTROL = "control"


@dataclass
class ABTest:
    """Test A/B"""
    id: str
    name: str
    description: str
    variants: List[str]
    traffic_split: Dict[str, float]  # variant -> percentage
    active: bool = True
    created_at: str = None
    results: Dict[str, Dict] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        if self.results is None:
            self.results = {}
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "variants": self.variants,
            "traffic_split": self.traffic_split,
            "active": self.active,
            "created_at": self.created_at,
            "results": self.results
        }


class ABTestingSystem:
    """Sistema de A/B Testing"""
    
    def __init__(self):
        """Inicializa el sistema"""
        self.tests: Dict[str, ABTest] = {}
        self.user_assignments: Dict[str, Dict[str, str]] = {}  # user_id -> {test_id: variant}
        self.test_events: Dict[str, List[Dict]] = {}  # test_id -> [events]
    
    def create_test(self, name: str, description: str,
                   variants: List[str],
                   traffic_split: Optional[Dict[str, float]] = None) -> str:
        """
        Crea un test A/B
        
        Args:
            name: Nombre del test
            description: Descripción
            variants: Lista de variantes
            traffic_split: División de tráfico (opcional)
            
        Returns:
            ID del test
        """
        test_id = str(uuid.uuid4())
        
        # División de tráfico por defecto (50/50)
        if traffic_split is None:
            split = 100 / len(variants)
            traffic_split = {v: split for v in variants}
        
        test = ABTest(
            id=test_id,
            name=name,
            description=description,
            variants=variants,
            traffic_split=traffic_split
        )
        
        self.tests[test_id] = test
        return test_id
    
    def assign_variant(self, test_id: str, user_id: str) -> str:
        """
        Asigna variante a un usuario
        
        Args:
            test_id: ID del test
            user_id: ID del usuario
            
        Returns:
            Variante asignada
        """
        if test_id not in self.tests:
            return Variant.CONTROL.value
        
        test = self.tests[test_id]
        
        if not test.active:
            return Variant.CONTROL.value
        
        # Verificar si ya tiene asignación
        if user_id in self.user_assignments and test_id in self.user_assignments[user_id]:
            return self.user_assignments[user_id][test_id]
        
        # Asignar según división de tráfico
        rand = random.random() * 100
        cumulative = 0
        
        for variant, percentage in test.traffic_split.items():
            cumulative += percentage
            if rand <= cumulative:
                if user_id not in self.user_assignments:
                    self.user_assignments[user_id] = {}
                self.user_assignments[user_id][test_id] = variant
                return variant
        
        # Fallback
        return test.variants[0]
    
    def record_event(self, test_id: str, user_id: str, event_type: str, value: Any = None):
        """
        Registra evento de test
        
        Args:
            test_id: ID del test
            user_id: ID del usuario
            event_type: Tipo de evento
            value: Valor del evento
        """
        if test_id not in self.test_events:
            self.test_events[test_id] = []
        
        variant = self.assign_variant(test_id, user_id)
        
        event = {
            "user_id": user_id,
            "variant": variant,
            "event_type": event_type,
            "value": value,
            "timestamp": datetime.now().isoformat()
        }
        
        self.test_events[test_id].append(event)
    
    def get_test_results(self, test_id: str) -> Dict:
        """Obtiene resultados de un test"""
        if test_id not in self.tests:
            return {}
        
        test = self.tests[test_id]
        events = self.test_events.get(test_id, [])
        
        # Agrupar por variante
        variant_events = {}
        for variant in test.variants:
            variant_events[variant] = [e for e in events if e["variant"] == variant]
        
        # Calcular métricas
        results = {}
        for variant, variant_evts in variant_events.items():
            results[variant] = {
                "total_users": len(set(e["user_id"] for e in variant_evts)),
                "total_events": len(variant_evts),
                "conversion_rate": 0,  # Placeholder
                "average_value": statistics.mean([e["value"] for e in variant_evts if e["value"] is not None]) if variant_evts else 0
            }
        
        test.results = results
        return results
    
    def get_test(self, test_id: str) -> Optional[ABTest]:
        """Obtiene un test"""
        return self.tests.get(test_id)
    
    def list_tests(self) -> List[ABTest]:
        """Lista todos los tests"""
        return list(self.tests.values())






