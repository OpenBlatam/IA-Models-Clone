"""
Servicio de Seguimiento de Relaciones Interpersonales - Sistema completo de relaciones
"""

from typing import Dict, List, Optional
from datetime import datetime
from enum import Enum


class RelationshipType(str, Enum):
    """Tipos de relaciones"""
    FAMILY = "family"
    FRIEND = "friend"
    SUPPORT_GROUP = "support_group"
    THERAPIST = "therapist"
    SPONSOR = "sponsor"
    PARTNER = "partner"


class RelationshipQuality(str, Enum):
    """Calidad de relación"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    TOXIC = "toxic"


class InterpersonalRelationshipsService:
    """Servicio de seguimiento de relaciones interpersonales"""
    
    def __init__(self):
        """Inicializa el servicio de relaciones"""
        pass
    
    def add_relationship(
        self,
        user_id: str,
        relationship_type: str,
        person_name: str,
        quality: str = RelationshipQuality.GOOD,
        notes: Optional[str] = None
    ) -> Dict:
        """
        Agrega una relación
        
        Args:
            user_id: ID del usuario
            relationship_type: Tipo de relación
            person_name: Nombre de la persona
            quality: Calidad de relación
            notes: Notas adicionales
        
        Returns:
            Relación agregada
        """
        relationship = {
            "id": f"relationship_{datetime.now().timestamp()}",
            "user_id": user_id,
            "relationship_type": relationship_type,
            "person_name": person_name,
            "quality": quality,
            "notes": notes,
            "created_at": datetime.now().isoformat(),
            "support_level": self._calculate_support_level(quality, relationship_type)
        }
        
        return relationship
    
    def record_interaction(
        self,
        relationship_id: str,
        user_id: str,
        interaction_type: str,
        interaction_data: Dict
    ) -> Dict:
        """
        Registra una interacción
        
        Args:
            relationship_id: ID de la relación
            user_id: ID del usuario
            interaction_type: Tipo de interacción
            interaction_data: Datos de la interacción
        
        Returns:
            Interacción registrada
        """
        interaction = {
            "id": f"interaction_{datetime.now().timestamp()}",
            "relationship_id": relationship_id,
            "user_id": user_id,
            "interaction_type": interaction_type,
            "interaction_data": interaction_data,
            "timestamp": datetime.now().isoformat(),
            "impact_score": self._calculate_impact_score(interaction_type, interaction_data)
        }
        
        return interaction
    
    def analyze_relationship_network(
        self,
        user_id: str,
        relationships: List[Dict]
    ) -> Dict:
        """
        Analiza red de relaciones
        
        Args:
            user_id: ID del usuario
            relationships: Lista de relaciones
        
        Returns:
            Análisis de red
        """
        return {
            "user_id": user_id,
            "total_relationships": len(relationships),
            "relationship_distribution": self._analyze_distribution(relationships),
            "support_network_strength": self._calculate_support_strength(relationships),
            "risk_relationships": self._identify_risk_relationships(relationships),
            "recommendations": self._generate_relationship_recommendations(relationships),
            "generated_at": datetime.now().isoformat()
        }
    
    def _calculate_support_level(self, quality: str, relationship_type: str) -> float:
        """Calcula nivel de apoyo"""
        quality_scores = {
            RelationshipQuality.EXCELLENT: 1.0,
            RelationshipQuality.GOOD: 0.75,
            RelationshipQuality.FAIR: 0.5,
            RelationshipQuality.POOR: 0.25,
            RelationshipQuality.TOXIC: 0.0
        }
        
        type_multipliers = {
            RelationshipType.THERAPIST: 1.2,
            RelationshipType.SPONSOR: 1.1,
            RelationshipType.SUPPORT_GROUP: 1.0,
            RelationshipType.FAMILY: 0.9,
            RelationshipType.FRIEND: 0.8
        }
        
        base_score = quality_scores.get(quality, 0.5)
        multiplier = type_multipliers.get(relationship_type, 1.0)
        
        return min(1.0, base_score * multiplier)
    
    def _calculate_impact_score(self, interaction_type: str, data: Dict) -> float:
        """Calcula puntuación de impacto"""
        return 0.7
    
    def _analyze_distribution(self, relationships: List[Dict]) -> Dict:
        """Analiza distribución de relaciones"""
        distribution = {}
        for rel in relationships:
            rel_type = rel.get("relationship_type")
            distribution[rel_type] = distribution.get(rel_type, 0) + 1
        return distribution
    
    def _calculate_support_strength(self, relationships: List[Dict]) -> float:
        """Calcula fuerza de red de apoyo"""
        if not relationships:
            return 0.0
        
        total_support = sum(rel.get("support_level", 0) for rel in relationships)
        return round(total_support / len(relationships), 2)
    
    def _identify_risk_relationships(self, relationships: List[Dict]) -> List[Dict]:
        """Identifica relaciones de riesgo"""
        risk_rels = []
        
        for rel in relationships:
            if rel.get("quality") == RelationshipQuality.TOXIC or \
               rel.get("quality") == RelationshipQuality.POOR:
                risk_rels.append({
                    "relationship_id": rel.get("id"),
                    "person_name": rel.get("person_name"),
                    "risk_reason": "low_quality_relationship"
                })
        
        return risk_rels
    
    def _generate_relationship_recommendations(self, relationships: List[Dict]) -> List[str]:
        """Genera recomendaciones de relaciones"""
        recommendations = []
        
        support_strength = self._calculate_support_strength(relationships)
        if support_strength < 0.5:
            recommendations.append("Fortalecer tu red de apoyo es importante para la recuperación")
        
        risk_rels = self._identify_risk_relationships(relationships)
        if risk_rels:
            recommendations.append("Considera limitar contacto con relaciones tóxicas")
        
        return recommendations

