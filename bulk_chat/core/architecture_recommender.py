"""
Architecture Recommender - Recomendador de Arquitectura
======================================================

Sistema que recomienda arquitecturas, patrones y mejores prácticas basado en requisitos y contexto.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from collections import defaultdict

logger = logging.getLogger(__name__)


class ArchitecturePattern(Enum):
    """Patrón arquitectónico."""
    MICROSERVICES = "microservices"
    MONOLITH = "monolith"
    SERVERLESS = "serverless"
    EVENT_DRIVEN = "event_driven"
    CQRS = "cqrs"
    LAYERED = "layered"
    HEXAGONAL = "hexagonal"


@dataclass
class ArchitectureRecommendation:
    """Recomendación de arquitectura."""
    recommendation_id: str
    pattern: ArchitecturePattern
    score: float
    reasoning: str
    pros: List[str] = field(default_factory=list)
    cons: List[str] = field(default_factory=list)
    requirements_met: List[str] = field(default_factory=list)
    requirements_missed: List[str] = field(default_factory=list)
    alternatives: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ArchitectureRequirement:
    """Requisito arquitectónico."""
    requirement_id: str
    category: str  # "scalability", "performance", "cost", "maintainability", etc.
    description: str
    priority: float = 1.0  # 0.0 to 1.0
    constraints: Dict[str, Any] = field(default_factory=dict)


class ArchitectureRecommender:
    """Recomendador de arquitectura."""
    
    def __init__(self):
        self.requirements: Dict[str, ArchitectureRequirement] = {}
        self.recommendations: List[ArchitectureRecommendation] = []
        self.pattern_scores: Dict[ArchitecturePattern, Dict[str, float]] = {
            ArchitecturePattern.MICROSERVICES: {
                "scalability": 0.9,
                "maintainability": 0.8,
                "performance": 0.6,
                "cost": 0.4,
                "complexity": 0.3,
            },
            ArchitecturePattern.MONOLITH: {
                "scalability": 0.4,
                "maintainability": 0.5,
                "performance": 0.8,
                "cost": 0.9,
                "complexity": 0.8,
            },
            ArchitecturePattern.SERVERLESS: {
                "scalability": 0.95,
                "maintainability": 0.7,
                "performance": 0.7,
                "cost": 0.6,
                "complexity": 0.6,
            },
            ArchitecturePattern.EVENT_DRIVEN: {
                "scalability": 0.85,
                "maintainability": 0.7,
                "performance": 0.7,
                "cost": 0.6,
                "complexity": 0.4,
            },
        }
        self._lock = asyncio.Lock()
    
    def add_requirement(
        self,
        requirement_id: str,
        category: str,
        description: str,
        priority: float = 1.0,
        constraints: Optional[Dict[str, Any]] = None,
    ):
        """Agregar requisito arquitectónico."""
        req = ArchitectureRequirement(
            requirement_id=requirement_id,
            category=category,
            description=description,
            priority=priority,
            constraints=constraints or {},
        )
        
        self.requirements[requirement_id] = req
        logger.info(f"Added requirement: {requirement_id} - {category}")
    
    async def recommend_architecture(
        self,
        requirements_ids: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[ArchitectureRecommendation]:
        """
        Recomendar arquitectura basada en requisitos.
        
        Args:
            requirements_ids: IDs de requisitos (todos si None)
            context: Contexto adicional
        
        Returns:
            Lista de recomendaciones ordenadas por score
        """
        if requirements_ids is None:
            requirements_ids = list(self.requirements.keys())
        
        reqs = [self.requirements[rid] for rid in requirements_ids if rid in self.requirements]
        
        if not reqs:
            return []
        
        recommendations = []
        
        for pattern in ArchitecturePattern:
            if pattern not in self.pattern_scores:
                continue
            
            # Calcular score
            total_score = 0.0
            total_weight = 0.0
            requirements_met = []
            requirements_missed = []
            
            for req in reqs:
                category = req.category
                pattern_score = self.pattern_scores[pattern].get(category, 0.5)
                weight = req.priority
                
                total_score += pattern_score * weight
                total_weight += weight
                
                if pattern_score >= 0.7:
                    requirements_met.append(req.requirement_id)
                else:
                    requirements_missed.append(req.requirement_id)
            
            if total_weight > 0:
                final_score = total_score / total_weight
            else:
                final_score = 0.5
            
            # Generar reasoning
            reasoning = self._generate_reasoning(pattern, reqs, final_score)
            
            # Pros y contras
            pros, cons = self._get_pros_cons(pattern)
            
            recommendation = ArchitectureRecommendation(
                recommendation_id=f"arch_rec_{pattern.value}_{datetime.now().timestamp()}",
                pattern=pattern,
                score=final_score,
                reasoning=reasoning,
                pros=pros,
                cons=cons,
                requirements_met=requirements_met,
                requirements_missed=requirements_missed,
                metadata={"context": context or {}},
            )
            
            recommendations.append(recommendation)
        
        # Ordenar por score
        recommendations.sort(key=lambda r: r.score, reverse=True)
        
        # Agregar alternativas
        for rec in recommendations:
            rec.alternatives = [
                r.pattern.value for r in recommendations
                if r.pattern != rec.pattern and r.score > rec.score * 0.8
            ]
        
        async with self._lock:
            self.recommendations.extend(recommendations)
        
        return recommendations
    
    def _generate_reasoning(
        self,
        pattern: ArchitecturePattern,
        requirements: List[ArchitectureRequirement],
        score: float,
    ) -> str:
        """Generar razonamiento para recomendación."""
        reasoning = f"El patrón {pattern.value} tiene un score de {score:.2f}. "
        
        if score >= 0.8:
            reasoning += "Excelente ajuste para los requisitos especificados. "
        elif score >= 0.6:
            reasoning += "Buen ajuste con algunas consideraciones. "
        else:
            reasoning += "Ajuste moderado, puede requerir adaptaciones. "
        
        # Agregar detalles específicos
        high_score_categories = [
            req.category for req in requirements
            if self.pattern_scores[pattern].get(req.category, 0) >= 0.8
        ]
        
        if high_score_categories:
            reasoning += f"Destaca en: {', '.join(high_score_categories)}. "
        
        return reasoning
    
    def _get_pros_cons(self, pattern: ArchitecturePattern) -> tuple:
        """Obtener pros y contras de un patrón."""
        pros_cons = {
            ArchitecturePattern.MICROSERVICES: (
                ["Alta escalabilidad", "Despliegue independiente", "Tecnología heterogénea"],
                ["Mayor complejidad", "Overhead de red", "Consistencia distribuida"],
            ),
            ArchitecturePattern.MONOLITH: (
                ["Simplicidad", "Bajo costo", "Alto rendimiento", "Desarrollo rápido"],
                ["Escalabilidad limitada", "Acoplamiento", "Deployment monolítico"],
            ),
            ArchitecturePattern.SERVERLESS: (
                ["Escalabilidad automática", "Sin gestión de servidores", "Pago por uso"],
                ["Cold starts", "Vendor lock-in", "Limitaciones de tiempo"],
            ),
            ArchitecturePattern.EVENT_DRIVEN: (
                ["Desacoplamiento", "Escalabilidad", "Resiliencia"],
                ["Complejidad de eventos", "Debugging difícil", "Consistencia eventual"],
            ),
        }
        
        return pros_cons.get(pattern, ([], []))
    
    def get_recommendations(
        self,
        pattern: Optional[ArchitecturePattern] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Obtener recomendaciones."""
        recs = self.recommendations
        
        if pattern:
            recs = [r for r in recs if r.pattern == pattern]
        
        return [
            {
                "recommendation_id": r.recommendation_id,
                "pattern": r.pattern.value,
                "score": r.score,
                "reasoning": r.reasoning,
                "pros": r.pros,
                "cons": r.cons,
                "requirements_met": r.requirements_met,
                "requirements_missed": r.requirements_missed,
                "alternatives": r.alternatives,
            }
            for r in recs[-limit:]
        ]
    
    def get_recommendation_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de recomendaciones."""
        by_pattern: Dict[str, int] = defaultdict(int)
        
        for rec in self.recommendations:
            by_pattern[rec.pattern.value] += 1
        
        return {
            "total_requirements": len(self.requirements),
            "total_recommendations": len(self.recommendations),
            "recommendations_by_pattern": dict(by_pattern),
        }
















