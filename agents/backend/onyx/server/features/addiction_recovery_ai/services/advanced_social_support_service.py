"""
Servicio de Apoyo Social Avanzado - Sistema completo de apoyo social
"""

from typing import Dict, List, Optional
from datetime import datetime
from collections import defaultdict


class AdvancedSocialSupportService:
    """Servicio de apoyo social avanzado"""
    
    def __init__(self):
        """Inicializa el servicio de apoyo social"""
        pass
    
    def assess_social_support(
        self,
        user_id: str,
        support_data: Dict
    ) -> Dict:
        """
        Evalúa apoyo social
        
        Args:
            user_id: ID del usuario
            support_data: Datos de apoyo
        
        Returns:
            Evaluación de apoyo social
        """
        return {
            "user_id": user_id,
            "assessment_id": f"support_{datetime.now().timestamp()}",
            "support_score": self._calculate_support_score(support_data),
            "support_network": self._analyze_support_network(support_data),
            "support_types": self._categorize_support_types(support_data),
            "support_quality": self._assess_support_quality(support_data),
            "recommendations": self._generate_support_recommendations(support_data),
            "assessed_at": datetime.now().isoformat()
        }
    
    def connect_with_supporter(
        self,
        user_id: str,
        supporter_data: Dict
    ) -> Dict:
        """
        Conecta con persona de apoyo
        
        Args:
            user_id: ID del usuario
            supporter_data: Datos del apoyo
        
        Returns:
            Conexión establecida
        """
        connection = {
            "id": f"connection_{datetime.now().timestamp()}",
            "user_id": user_id,
            "supporter_data": supporter_data,
            "supporter_name": supporter_data.get("name", ""),
            "support_type": supporter_data.get("support_type", "general"),
            "connection_strength": supporter_data.get("connection_strength", 5),
            "connected_at": datetime.now().isoformat(),
            "status": "active"
        }
        
        return connection
    
    def track_support_interactions(
        self,
        user_id: str,
        interactions: List[Dict]
    ) -> Dict:
        """
        Rastrea interacciones de apoyo
        
        Args:
            user_id: ID del usuario
            interactions: Lista de interacciones
        
        Returns:
            Análisis de interacciones
        """
        return {
            "user_id": user_id,
            "total_interactions": len(interactions),
            "interaction_frequency": self._calculate_frequency(interactions),
            "support_effectiveness": self._calculate_effectiveness(interactions),
            "top_supporters": self._identify_top_supporters(interactions),
            "support_trends": self._analyze_support_trends(interactions),
            "generated_at": datetime.now().isoformat()
        }
    
    def recommend_support_resources(
        self,
        user_id: str,
        user_profile: Dict
    ) -> Dict:
        """
        Recomienda recursos de apoyo
        
        Args:
            user_id: ID del usuario
            user_profile: Perfil del usuario
        
        Returns:
            Recursos recomendados
        """
        return {
            "user_id": user_id,
            "recommended_resources": self._generate_resource_recommendations(user_profile),
            "support_groups": self._recommend_support_groups(user_profile),
            "professional_services": self._recommend_professional_services(user_profile),
            "online_resources": self._recommend_online_resources(user_profile),
            "generated_at": datetime.now().isoformat()
        }
    
    def _calculate_support_score(self, data: Dict) -> float:
        """Calcula puntuación de apoyo"""
        family_support = data.get("family_support", 5)
        friend_support = data.get("friend_support", 5)
        professional_support = data.get("professional_support", 5)
        community_support = data.get("community_support", 5)
        
        avg_support = (family_support + friend_support + professional_support + community_support) / 4
        
        return round(max(1, min(10, avg_support)), 2)
    
    def _analyze_support_network(self, data: Dict) -> Dict:
        """Analiza red de apoyo"""
        return {
            "family_members": data.get("family_members", []),
            "friends": data.get("friends", []),
            "professionals": data.get("professionals", []),
            "community_members": data.get("community_members", []),
            "total_connections": len(data.get("family_members", [])) + len(data.get("friends", [])) + len(data.get("professionals", [])) + len(data.get("community_members", []))
        }
    
    def _categorize_support_types(self, data: Dict) -> Dict:
        """Categoriza tipos de apoyo"""
        return {
            "emotional_support": data.get("emotional_support", 5),
            "practical_support": data.get("practical_support", 5),
            "informational_support": data.get("informational_support", 5),
            "companionship": data.get("companionship", 5)
        }
    
    def _assess_support_quality(self, data: Dict) -> float:
        """Evalúa calidad de apoyo"""
        support_score = self._calculate_support_score(data)
        return round(support_score / 10, 2)
    
    def _generate_support_recommendations(self, data: Dict) -> List[str]:
        """Genera recomendaciones de apoyo"""
        recommendations = []
        
        support_score = self._calculate_support_score(data)
        
        if support_score < 5:
            recommendations.append("Fortalecer tu red de apoyo social es crucial para la recuperación")
            recommendations.append("Considera unirte a grupos de apoyo")
        
        return recommendations
    
    def _calculate_frequency(self, interactions: List[Dict]) -> float:
        """Calcula frecuencia de interacciones"""
        if not interactions:
            return 0.0
        
        # Calcular interacciones por semana
        days = 7
        return round(len(interactions) / days, 2)
    
    def _calculate_effectiveness(self, interactions: List[Dict]) -> float:
        """Calcula efectividad de apoyo"""
        if not interactions:
            return 0.0
        
        effectiveness_scores = [i.get("effectiveness", 5) for i in interactions]
        return round(sum(effectiveness_scores) / len(effectiveness_scores), 2)
    
    def _identify_top_supporters(self, interactions: List[Dict]) -> List[Dict]:
        """Identifica principales apoyos"""
        supporter_counts = defaultdict(int)
        
        for interaction in interactions:
            supporter = interaction.get("supporter_name", "unknown")
            supporter_counts[supporter] += 1
        
        sorted_supporters = sorted(supporter_counts.items(), key=lambda x: x[1], reverse=True)
        
        return [
            {"name": name, "interaction_count": count}
            for name, count in sorted_supporters[:5]
        ]
    
    def _analyze_support_trends(self, interactions: List[Dict]) -> Dict:
        """Analiza tendencias de apoyo"""
        return {
            "trend": "stable",
            "engagement_level": "moderate"
        }
    
    def _generate_resource_recommendations(self, profile: Dict) -> List[Dict]:
        """Genera recomendaciones de recursos"""
        recommendations = []
        
        recommendations.append({
            "type": "support_group",
            "name": "Grupo de Apoyo Local",
            "description": "Grupo de apoyo para personas en recuperación"
        })
        
        return recommendations
    
    def _recommend_support_groups(self, profile: Dict) -> List[Dict]:
        """Recomienda grupos de apoyo"""
        return [
            {
                "name": "Grupo de Apoyo Semanal",
                "type": "peer_support",
                "frequency": "weekly"
            }
        ]
    
    def _recommend_professional_services(self, profile: Dict) -> List[Dict]:
        """Recomienda servicios profesionales"""
        return [
            {
                "type": "counseling",
                "description": "Servicios de consejería profesional"
            }
        ]
    
    def _recommend_online_resources(self, profile: Dict) -> List[Dict]:
        """Recomienda recursos en línea"""
        return [
            {
                "type": "online_forum",
                "description": "Foros en línea de apoyo"
            }
        ]

