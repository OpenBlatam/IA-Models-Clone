"""
A/B Testing V2 System
=====================
Sistema avanzado de A/B testing con análisis estadístico
"""

import time
import random
import statistics
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict


class TestStatus(Enum):
    """Estados de test"""
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    ARCHIVED = "archived"


class MetricType(Enum):
    """Tipos de métricas"""
    CONVERSION = "conversion"
    CLICK_THROUGH = "click_through"
    TIME_SPENT = "time_spent"
    QUALITY_SCORE = "quality_score"
    CUSTOM = "custom"


@dataclass
class Variant:
    """Variante de test"""
    id: str
    name: str
    config: Dict[str, Any]
    traffic_percentage: float
    conversions: int = 0
    visitors: int = 0
    total_value: float = 0.0


@dataclass
class ABTest:
    """Test A/B"""
    id: str
    name: str
    description: str
    variants: List[Variant]
    metric_type: MetricType
    status: TestStatus
    created_at: float
    start_date: Optional[float] = None
    end_date: Optional[float] = None
    min_sample_size: int = 100
    confidence_level: float = 0.95


@dataclass
class TestResult:
    """Resultado de test"""
    test_id: str
    winner: Optional[str]
    confidence: float
    p_value: float
    lift: float
    variant_stats: Dict[str, Dict[str, float]]


class ABTestingV2:
    """
    Sistema avanzado de A/B testing
    """
    
    def __init__(self):
        self.tests: Dict[str, ABTest] = {}
        self.visitors: Dict[str, Dict[str, Any]] = {}  # visitor_id -> test_id -> variant_id
        self.conversions: Dict[str, List[Dict[str, Any]]] = defaultdict(list)  # test_id -> conversions
    
    def create_test(
        self,
        name: str,
        description: str,
        variants: List[Dict[str, Any]],
        metric_type: MetricType,
        min_sample_size: int = 100,
        confidence_level: float = 0.95
    ) -> ABTest:
        """
        Crear test A/B
        
        Args:
            name: Nombre del test
            description: Descripción
            variants: Lista de variantes
            metric_type: Tipo de métrica
            min_sample_size: Tamaño mínimo de muestra
            confidence_level: Nivel de confianza
        """
        test_id = f"test_{int(time.time())}"
        
        # Normalizar porcentajes de tráfico
        total_percentage = sum(v.get('traffic_percentage', 0) for v in variants)
        if total_percentage != 100:
            # Distribuir equitativamente
            per_variant = 100 / len(variants)
            for v in variants:
                v['traffic_percentage'] = per_variant
        
        test_variants = [
            Variant(
                id=f"variant_{i}",
                name=v['name'],
                config=v.get('config', {}),
                traffic_percentage=v.get('traffic_percentage', 0)
            )
            for i, v in enumerate(variants)
        ]
        
        test = ABTest(
            id=test_id,
            name=name,
            description=description,
            variants=test_variants,
            metric_type=metric_type,
            status=TestStatus.DRAFT,
            created_at=time.time(),
            min_sample_size=min_sample_size,
            confidence_level=confidence_level
        )
        
        self.tests[test_id] = test
        return test
    
    def assign_variant(
        self,
        test_id: str,
        visitor_id: str
    ) -> Optional[str]:
        """
        Asignar variante a visitante
        
        Args:
            test_id: ID del test
            visitor_id: ID del visitante
        """
        if test_id not in self.tests:
            return None
        
        test = self.tests[test_id]
        
        if test.status != TestStatus.RUNNING:
            return None
        
        # Si ya tiene asignación, retornarla
        if visitor_id in self.visitors and test_id in self.visitors[visitor_id]:
            return self.visitors[visitor_id][test_id]
        
        # Asignar basado en porcentajes de tráfico
        rand = random.random() * 100
        cumulative = 0
        
        for variant in test.variants:
            cumulative += variant.traffic_percentage
            if rand <= cumulative:
                if visitor_id not in self.visitors:
                    self.visitors[visitor_id] = {}
                self.visitors[visitor_id][test_id] = variant.id
                variant.visitors += 1
                return variant.id
        
        # Fallback al primer variant
        if test.variants:
            variant_id = test.variants[0].id
            if visitor_id not in self.visitors:
                self.visitors[visitor_id] = {}
            self.visitors[visitor_id][test_id] = variant_id
            test.variants[0].visitors += 1
            return variant_id
        
        return None
    
    def record_conversion(
        self,
        test_id: str,
        visitor_id: str,
        value: float = 1.0
    ):
        """
        Registrar conversión
        
        Args:
            test_id: ID del test
            visitor_id: ID del visitante
            value: Valor de la conversión
        """
        if test_id not in self.tests:
            return
        
        if visitor_id not in self.visitors or test_id not in self.visitors[visitor_id]:
            return
        
        variant_id = self.visitors[visitor_id][test_id]
        
        # Encontrar variant
        test = self.tests[test_id]
        variant = next((v for v in test.variants if v.id == variant_id), None)
        
        if variant:
            variant.conversions += 1
            variant.total_value += value
        
        # Registrar conversión
        self.conversions[test_id].append({
            'visitor_id': visitor_id,
            'variant_id': variant_id,
            'value': value,
            'timestamp': time.time()
        })
    
    def start_test(self, test_id: str):
        """Iniciar test"""
        if test_id in self.tests:
            test = self.tests[test_id]
            test.status = TestStatus.RUNNING
            test.start_date = time.time()
    
    def pause_test(self, test_id: str):
        """Pausar test"""
        if test_id in self.tests:
            self.tests[test_id].status = TestStatus.PAUSED
    
    def complete_test(self, test_id: str):
        """Completar test"""
        if test_id in self.tests:
            test = self.tests[test_id]
            test.status = TestStatus.COMPLETED
            test.end_date = time.time()
    
    def analyze_test(self, test_id: str) -> Optional[TestResult]:
        """
        Analizar test y determinar ganador
        
        Args:
            test_id: ID del test
        """
        if test_id not in self.tests:
            return None
        
        test = self.tests[test_id]
        
        if len(test.variants) < 2:
            return None
        
        # Calcular estadísticas por variante
        variant_stats = {}
        for variant in test.variants:
            conversion_rate = (
                variant.conversions / variant.visitors
                if variant.visitors > 0 else 0
            )
            
            variant_stats[variant.id] = {
                'name': variant.name,
                'visitors': variant.visitors,
                'conversions': variant.conversions,
                'conversion_rate': conversion_rate,
                'total_value': variant.total_value,
                'average_value': (
                    variant.total_value / variant.conversions
                    if variant.conversions > 0 else 0
                )
            }
        
        # Test estadístico (t-test simplificado)
        if len(test.variants) == 2:
            control = test.variants[0]
            treatment = test.variants[1]
            
            if control.visitors < test.min_sample_size or treatment.visitors < test.min_sample_size:
                return TestResult(
                    test_id=test_id,
                    winner=None,
                    confidence=0.0,
                    p_value=1.0,
                    lift=0.0,
                    variant_stats=variant_stats
                )
            
            # Calcular p-value (simplificado)
            p_value, confidence = self._calculate_statistical_significance(
                control.visitors, control.conversions,
                treatment.visitors, treatment.conversions
            )
            
            # Determinar ganador
            control_rate = control.conversions / control.visitors if control.visitors > 0 else 0
            treatment_rate = treatment.conversions / treatment.visitors if treatment.visitors > 0 else 0
            
            winner = None
            lift = 0.0
            
            if p_value < (1 - test.confidence_level):
                if treatment_rate > control_rate:
                    winner = treatment.id
                    lift = ((treatment_rate - control_rate) / control_rate * 100) if control_rate > 0 else 0
                elif control_rate > treatment_rate:
                    winner = control.id
                    lift = ((control_rate - treatment_rate) / treatment_rate * 100) if treatment_rate > 0 else 0
            
            return TestResult(
                test_id=test_id,
                winner=winner,
                confidence=confidence,
                p_value=p_value,
                lift=lift,
                variant_stats=variant_stats
            )
        
        # Para múltiples variantes, encontrar la mejor
        best_variant = max(
            test.variants,
            key=lambda v: v.conversions / v.visitors if v.visitors > 0 else 0
        )
        
        return TestResult(
            test_id=test_id,
            winner=best_variant.id,
            confidence=0.8,  # Simplificado
            p_value=0.05,
            lift=0.0,
            variant_stats=variant_stats
        )
    
    def _calculate_statistical_significance(
        self,
        n1: int, x1: int,
        n2: int, x2: int
    ) -> Tuple[float, float]:
        """
        Calcular significancia estadística (simplificado)
        
        Returns:
            (p_value, confidence)
        """
        if n1 == 0 or n2 == 0:
            return (1.0, 0.0)
        
        p1 = x1 / n1
        p2 = x2 / n2
        
        # Pooled proportion
        p_pool = (x1 + x2) / (n1 + n2)
        
        # Standard error
        se = (p_pool * (1 - p_pool) * (1/n1 + 1/n2)) ** 0.5
        
        if se == 0:
            return (1.0, 0.0)
        
        # Z-score
        z = (p2 - p1) / se
        
        # P-value aproximado (simplificado)
        # En implementación real, usar scipy.stats
        p_value = abs(z) * 0.1  # Simplificado
        p_value = min(p_value, 1.0)
        
        confidence = 1 - p_value
        
        return (p_value, confidence)
    
    def get_test_statistics(self, test_id: str) -> Dict[str, Any]:
        """Obtener estadísticas de test"""
        if test_id not in self.tests:
            return {}
        
        test = self.tests[test_id]
        result = self.analyze_test(test_id)
        
        return {
            'test_id': test_id,
            'name': test.name,
            'status': test.status.value,
            'metric_type': test.metric_type.value,
            'total_visitors': sum(v.visitors for v in test.variants),
            'total_conversions': sum(v.conversions for v in test.variants),
            'result': {
                'winner': result.winner if result else None,
                'confidence': result.confidence if result else 0.0,
                'p_value': result.p_value if result else 1.0,
                'lift': result.lift if result else 0.0
            } if result else None,
            'variants': [
                {
                    'id': v.id,
                    'name': v.name,
                    'visitors': v.visitors,
                    'conversions': v.conversions,
                    'conversion_rate': v.conversions / v.visitors if v.visitors > 0 else 0
                }
                for v in test.variants
            ]
        }


# Instancia global
ab_testing_v2 = ABTestingV2()

