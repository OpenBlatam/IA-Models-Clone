"""
A/B Testing - Sistema de pruebas A/B
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import uuid

logger = logging.getLogger(__name__)


class VariantStatus(Enum):
    """Estado de variante"""
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"


@dataclass
class Variant:
    """Variante de prueba A/B"""
    id: str
    name: str
    content: str
    status: VariantStatus
    impressions: int = 0
    conversions: int = 0
    created_at: datetime = None


@dataclass
class ABTest:
    """Prueba A/B"""
    id: str
    name: str
    description: str
    variants: List[Variant]
    start_date: datetime
    end_date: Optional[datetime] = None
    status: str = "active"


class ABTestingManager:
    """Gestor de pruebas A/B"""

    def __init__(self):
        """Inicializar gestor"""
        self.tests: Dict[str, ABTest] = {}
        self.results: Dict[str, List[Dict[str, Any]]] = {}

    def create_test(
        self,
        name: str,
        description: str,
        variants: List[Dict[str, str]],
        duration_days: Optional[int] = None
    ) -> str:
        """
        Crear prueba A/B.

        Args:
            name: Nombre de la prueba
            description: Descripción
            variants: Lista de variantes (cada una con 'name' y 'content')
            duration_days: Duración en días (opcional)

        Returns:
            ID de la prueba
        """
        test_id = str(uuid.uuid4())
        
        # Crear variantes
        variant_objects = []
        for variant_data in variants:
            variant = Variant(
                id=str(uuid.uuid4()),
                name=variant_data.get("name", f"Variant {len(variant_objects) + 1}"),
                content=variant_data.get("content", ""),
                status=VariantStatus.ACTIVE,
                created_at=datetime.utcnow()
            )
            variant_objects.append(variant)
        
        # Calcular fecha de fin
        end_date = None
        if duration_days:
            from datetime import timedelta
            end_date = datetime.utcnow() + timedelta(days=duration_days)
        
        test = ABTest(
            id=test_id,
            name=name,
            description=description,
            variants=variant_objects,
            start_date=datetime.utcnow(),
            end_date=end_date,
            status="active"
        )
        
        self.tests[test_id] = test
        self.results[test_id] = []
        
        logger.info(f"Prueba A/B creada: {test_id}")
        return test_id

    def record_impression(
        self,
        test_id: str,
        variant_id: str
    ):
        """
        Registrar impresión de variante.

        Args:
            test_id: ID de la prueba
            variant_id: ID de la variante
        """
        if test_id not in self.tests:
            logger.warning(f"Prueba A/B no encontrada: {test_id}")
            return
        
        test = self.tests[test_id]
        for variant in test.variants:
            if variant.id == variant_id:
                variant.impressions += 1
                logger.debug(f"Impresión registrada para variante {variant_id}")
                return

    def record_conversion(
        self,
        test_id: str,
        variant_id: str
    ):
        """
        Registrar conversión de variante.

        Args:
            test_id: ID de la prueba
            variant_id: ID de la variante
        """
        if test_id not in self.tests:
            logger.warning(f"Prueba A/B no encontrada: {test_id}")
            return
        
        test = self.tests[test_id]
        for variant in test.variants:
            if variant.id == variant_id:
                variant.conversions += 1
                
                # Registrar resultado
                conversion_rate = (
                    variant.conversions / variant.impressions
                    if variant.impressions > 0 else 0.0
                )
                
                self.results[test_id].append({
                    "variant_id": variant_id,
                    "variant_name": variant.name,
                    "conversion_rate": conversion_rate,
                    "timestamp": datetime.utcnow().isoformat()
                })
                
                logger.debug(f"Conversión registrada para variante {variant_id}")
                return

    def get_test_results(self, test_id: str) -> Dict[str, Any]:
        """
        Obtener resultados de prueba.

        Args:
            test_id: ID de la prueba

        Returns:
            Resultados de la prueba
        """
        if test_id not in self.tests:
            return {"error": "Prueba no encontrada"}
        
        test = self.tests[test_id]
        
        # Calcular estadísticas
        variant_stats = []
        for variant in test.variants:
            conversion_rate = (
                variant.conversions / variant.impressions
                if variant.impressions > 0 else 0.0
            )
            
            variant_stats.append({
                "variant_id": variant.id,
                "variant_name": variant.name,
                "impressions": variant.impressions,
                "conversions": variant.conversions,
                "conversion_rate": conversion_rate
            })
        
        # Determinar ganador
        if variant_stats:
            winner = max(variant_stats, key=lambda x: x["conversion_rate"])
        else:
            winner = None
        
        return {
            "test_id": test_id,
            "test_name": test.name,
            "status": test.status,
            "variants": variant_stats,
            "winner": winner,
            "total_impressions": sum(v.impressions for v in test.variants),
            "total_conversions": sum(v.conversions for v in test.variants)
        }

    def get_all_tests(self) -> List[Dict[str, Any]]:
        """
        Obtener todas las pruebas.

        Returns:
            Lista de pruebas
        """
        return [
            {
                "id": test.id,
                "name": test.name,
                "description": test.description,
                "status": test.status,
                "variant_count": len(test.variants),
                "start_date": test.start_date.isoformat(),
                "end_date": test.end_date.isoformat() if test.end_date else None
            }
            for test in self.tests.values()
        ]

    def end_test(self, test_id: str) -> Dict[str, Any]:
        """
        Finalizar prueba A/B.

        Args:
            test_id: ID de la prueba

        Returns:
            Resultados finales
        """
        if test_id not in self.tests:
            return {"error": "Prueba no encontrada"}
        
        test = self.tests[test_id]
        test.status = "completed"
        test.end_date = datetime.utcnow()
        
        results = self.get_test_results(test_id)
        logger.info(f"Prueba A/B finalizada: {test_id}")
        
        return results






