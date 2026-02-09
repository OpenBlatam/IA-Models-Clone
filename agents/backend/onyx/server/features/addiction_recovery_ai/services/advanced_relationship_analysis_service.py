"""
Servicio de Análisis de Relaciones Avanzado - Sistema completo de análisis de relaciones
"""

from typing import Dict, List, Optional
from datetime import datetime
from collections import defaultdict
import statistics


class AdvancedRelationshipAnalysisService:
    """Servicio de análisis de relaciones avanzado"""
    
    def __init__(self):
        """Inicializa el servicio de relaciones"""
        pass
    
    def analyze_relationships(
        self,
        user_id: str,
        relationships: List[Dict]
    ) -> Dict:
        """
        Analiza relaciones
        
        Args:
            user_id: ID del usuario
            relationships: Lista de relaciones
        
        Returns:
            Análisis de relaciones
        """
        if not relationships:
            return {
                "user_id": user_id,
                "analysis": "no_data"
            }
        
        return {
            "user_id": user_id,
            "analysis_id": f"relationships_{datetime.now().timestamp()}",
            "total_relationships": len(relationships),
            "relationship_types": self._categorize_relationships(relationships),
            "relationship_quality": self._assess_quality(relationships),
            "supportive_relationships": self._identify_supportive(relationships),
            "challenging_relationships": self._identify_challenging(relationships),
            "recommendations": self._generate_relationship_recommendations(relationships),
            "generated_at": datetime.now().isoformat()
        }
    
    def track_relationship_changes(
        self,
        user_id: str,
        relationship_id: str,
        changes: List[Dict]
    ) -> Dict:
        """
        Rastrea cambios en relación
        
        Args:
            user_id: ID del usuario
            relationship_id: ID de la relación
            changes: Lista de cambios
        
        Returns:
            Análisis de cambios
        """
        return {
            "user_id": user_id,
            "relationship_id": relationship_id,
            "total_changes": len(changes),
            "change_trend": self._analyze_change_trend(changes),
            "improvement_indicators": self._identify_improvements(changes),
            "concern_indicators": self._identify_concerns(changes),
            "generated_at": datetime.now().isoformat()
        }
    
    def assess_relationship_impact(
        self,
        user_id: str,
        relationships: List[Dict],
        recovery_data: List[Dict]
    ) -> Dict:
        """
        Evalúa impacto de relaciones en recuperación
        
        Args:
            user_id: ID del usuario
            relationships: Lista de relaciones
            recovery_data: Datos de recuperación
        
        Returns:
            Análisis de impacto
        """
        return {
            "user_id": user_id,
            "positive_impact_score": self._calculate_positive_impact(relationships, recovery_data),
            "negative_impact_score": self._calculate_negative_impact(relationships, recovery_data),
            "overall_impact": self._determine_overall_impact(relationships, recovery_data),
            "key_relationships": self._identify_key_relationships(relationships),
            "recommendations": self._generate_impact_recommendations(relationships, recovery_data),
            "generated_at": datetime.now().isoformat()
        }
    
    def _categorize_relationships(self, relationships: List[Dict]) -> Dict:
        """Categoriza relaciones"""
        categories = defaultdict(int)
        
        for relationship in relationships:
            rel_type = relationship.get("type", "unknown")
            categories[rel_type] += 1
        
        return dict(categories)
    
    def _assess_quality(self, relationships: List[Dict]) -> Dict:
        """Evalúa calidad de relaciones"""
        quality_scores = [r.get("quality_score", 5) for r in relationships]
        
        return {
            "average_quality": round(statistics.mean(quality_scores), 2) if quality_scores else 0,
            "high_quality_count": sum(1 for r in relationships if r.get("quality_score", 5) >= 7),
            "low_quality_count": sum(1 for r in relationships if r.get("quality_score", 5) < 4)
        }
    
    def _identify_supportive(self, relationships: List[Dict]) -> List[Dict]:
        """Identifica relaciones de apoyo"""
        supportive = [r for r in relationships if r.get("is_supportive", False)]
        return supportive[:5]
    
    def _identify_challenging(self, relationships: List[Dict]) -> List[Dict]:
        """Identifica relaciones desafiantes"""
        challenging = [r for r in relationships if r.get("is_challenging", False)]
        return challenging[:5]
    
    def _generate_relationship_recommendations(self, relationships: List[Dict]) -> List[str]:
        """Genera recomendaciones de relaciones"""
        recommendations = []
        
        supportive_count = len(self._identify_supportive(relationships))
        if supportive_count < 3:
            recommendations.append("Fortalecer relaciones de apoyo es importante para la recuperación")
        
        challenging_count = len(self._identify_challenging(relationships))
        if challenging_count > 0:
            recommendations.append("Considera establecer límites saludables en relaciones desafiantes")
        
        return recommendations
    
    def _analyze_change_trend(self, changes: List[Dict]) -> str:
        """Analiza tendencia de cambios"""
        if len(changes) < 2:
            return "stable"
        
        first_half = changes[:len(changes)//2]
        second_half = changes[len(changes)//2:]
        
        first_avg = statistics.mean([c.get("change_score", 5) for c in first_half]) if first_half else 0
        second_avg = statistics.mean([c.get("change_score", 5) for c in second_half]) if second_half else 0
        
        if second_avg > first_avg * 1.1:
            return "improving"
        elif second_avg < first_avg * 0.9:
            return "declining"
        return "stable"
    
    def _identify_improvements(self, changes: List[Dict]) -> List[str]:
        """Identifica mejoras"""
        improvements = []
        
        positive_changes = [c for c in changes if c.get("change_type") == "positive"]
        if positive_changes:
            improvements.append("Mejoras en comunicación")
        
        return improvements
    
    def _identify_concerns(self, changes: List[Dict]) -> List[str]:
        """Identifica preocupaciones"""
        concerns = []
        
        negative_changes = [c for c in changes if c.get("change_type") == "negative"]
        if negative_changes:
            concerns.append("Algunos cambios negativos detectados")
        
        return concerns
    
    def _calculate_positive_impact(self, relationships: List[Dict], recovery_data: List[Dict]) -> float:
        """Calcula impacto positivo"""
        supportive = self._identify_supportive(relationships)
        return round(len(supportive) / len(relationships) if relationships else 0, 2)
    
    def _calculate_negative_impact(self, relationships: List[Dict], recovery_data: List[Dict]) -> float:
        """Calcula impacto negativo"""
        challenging = self._identify_challenging(relationships)
        return round(len(challenging) / len(relationships) if relationships else 0, 2)
    
    def _determine_overall_impact(self, relationships: List[Dict], recovery_data: List[Dict]) -> str:
        """Determina impacto general"""
        positive = self._calculate_positive_impact(relationships, recovery_data)
        negative = self._calculate_negative_impact(relationships, recovery_data)
        
        if positive > negative * 1.5:
            return "positive"
        elif negative > positive * 1.5:
            return "negative"
        return "neutral"
    
    def _identify_key_relationships(self, relationships: List[Dict]) -> List[Dict]:
        """Identifica relaciones clave"""
        key_relationships = []
        
        for relationship in relationships:
            if relationship.get("is_key_relationship", False):
                key_relationships.append(relationship)
        
        return key_relationships[:5]
    
    def _generate_impact_recommendations(self, relationships: List[Dict], recovery_data: List[Dict]) -> List[str]:
        """Genera recomendaciones de impacto"""
        recommendations = []
        
        overall_impact = self._determine_overall_impact(relationships, recovery_data)
        if overall_impact == "negative":
            recommendations.append("Considera trabajar en mejorar la calidad de tus relaciones")
        
        return recommendations

