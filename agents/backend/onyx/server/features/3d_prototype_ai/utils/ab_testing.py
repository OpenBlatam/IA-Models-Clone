"""
A/B Testing - Sistema de A/B testing
=====================================
"""

import logging
import random
from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class VariantType(str, Enum):
    """Tipos de variantes"""
    CONTROL = "control"
    VARIANT_A = "variant_a"
    VARIANT_B = "variant_b"
    VARIANT_C = "variant_c"


class ABTesting:
    """Sistema de A/B testing"""
    
    def __init__(self):
        self.experiments: Dict[str, Dict[str, Any]] = {}
        self.user_assignments: Dict[str, Dict[str, str]] = {}  # user_id -> experiment_id -> variant
        self.conversions: List[Dict[str, Any]] = []
    
    def create_experiment(self, experiment_id: str, name: str,
                         variants: List[str], traffic_split: Optional[Dict[str, float]] = None,
                         enabled: bool = True) -> Dict[str, Any]:
        """Crea un experimento A/B"""
        if not traffic_split:
            # Split equitativo
            split = 1.0 / len(variants)
            traffic_split = {variant: split for variant in variants}
        
        experiment = {
            "id": experiment_id,
            "name": name,
            "variants": variants,
            "traffic_split": traffic_split,
            "enabled": enabled,
            "created_at": datetime.now().isoformat(),
            "metrics": {
                "total_users": 0,
                "conversions": 0,
                "conversion_rate": 0.0
            }
        }
        
        self.experiments[experiment_id] = experiment
        
        logger.info(f"Experimento A/B creado: {experiment_id}")
        return experiment
    
    def assign_variant(self, experiment_id: str, user_id: str) -> str:
        """Asigna una variante a un usuario"""
        experiment = self.experiments.get(experiment_id)
        if not experiment or not experiment["enabled"]:
            return VariantType.CONTROL.value
        
        # Verificar si ya tiene asignación
        if user_id in self.user_assignments.get(experiment_id, {}):
            return self.user_assignments[experiment_id][user_id]
        
        # Asignar según traffic split
        rand = random.random()
        cumulative = 0.0
        
        for variant, percentage in experiment["traffic_split"].items():
            cumulative += percentage
            if rand <= cumulative:
                # Guardar asignación
                if experiment_id not in self.user_assignments:
                    self.user_assignments[experiment_id] = {}
                
                self.user_assignments[experiment_id][user_id] = variant
                experiment["metrics"]["total_users"] += 1
                
                logger.debug(f"Usuario {user_id} asignado a {variant} en experimento {experiment_id}")
                return variant
        
        # Fallback
        return VariantType.CONTROL.value
    
    def record_conversion(self, experiment_id: str, user_id: str,
                        variant: Optional[str] = None, value: float = 1.0):
        """Registra una conversión"""
        if not variant:
            variant = self.user_assignments.get(experiment_id, {}).get(user_id, VariantType.CONTROL.value)
        
        conversion = {
            "experiment_id": experiment_id,
            "user_id": user_id,
            "variant": variant,
            "value": value,
            "timestamp": datetime.now().isoformat()
        }
        
        self.conversions.append(conversion)
        
        # Actualizar métricas del experimento
        experiment = self.experiments.get(experiment_id)
        if experiment:
            experiment["metrics"]["conversions"] += 1
            total_users = experiment["metrics"]["total_users"]
            if total_users > 0:
                experiment["metrics"]["conversion_rate"] = experiment["metrics"]["conversions"] / total_users
        
        logger.info(f"Conversión registrada: {experiment_id} - {variant}")
    
    def get_experiment_results(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Obtiene resultados de un experimento"""
        experiment = self.experiments.get(experiment_id)
        if not experiment:
            return None
        
        # Calcular métricas por variante
        variant_metrics = {}
        for variant in experiment["variants"]:
            variant_conversions = [
                c for c in self.conversions
                if c["experiment_id"] == experiment_id and c["variant"] == variant
            ]
            
            variant_users = sum(
                1 for assignments in self.user_assignments.get(experiment_id, {}).values()
                if assignments == variant
            ) if experiment_id in self.user_assignments else 0
            
            conversion_rate = len(variant_conversions) / variant_users if variant_users > 0 else 0.0
            
            variant_metrics[variant] = {
                "users": variant_users,
                "conversions": len(variant_conversions),
                "conversion_rate": conversion_rate,
                "total_value": sum(c["value"] for c in variant_conversions)
            }
        
        return {
            "experiment": experiment,
            "variant_metrics": variant_metrics,
            "statistical_significance": self._calculate_significance(experiment_id)
        }
    
    def _calculate_significance(self, experiment_id: str) -> float:
        """Calcula significancia estadística (simplificado)"""
        # En producción, usaría test chi-cuadrado o similar
        return 0.85  # Simulado
    
    def list_experiments(self) -> List[Dict[str, Any]]:
        """Lista experimentos"""
        return [
            {
                "id": exp["id"],
                "name": exp["name"],
                "enabled": exp["enabled"],
                "variants": exp["variants"],
                "metrics": exp["metrics"]
            }
            for exp in self.experiments.values()
        ]




