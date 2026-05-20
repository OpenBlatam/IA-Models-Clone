"""
Sistema de personalización avanzada
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict
import statistics


@dataclass
class UserProfile:
    """Perfil de usuario"""
    user_id: str
    preferences: Dict
    skin_type: Optional[str] = None
    skin_concerns: List[str] = None
    favorite_products: List[str] = None
    routine_preferences: Dict = None
    created_at: str = None
    updated_at: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        if self.updated_at is None:
            self.updated_at = datetime.now().isoformat()
        if self.skin_concerns is None:
            self.skin_concerns = []
        if self.favorite_products is None:
            self.favorite_products = []
        if self.routine_preferences is None:
            self.routine_preferences = {}
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "user_id": self.user_id,
            "preferences": self.preferences,
            "skin_type": self.skin_type,
            "skin_concerns": self.skin_concerns,
            "favorite_products": self.favorite_products,
            "routine_preferences": self.routine_preferences,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }


class PersonalizationEngine:
    """Motor de personalización"""
    
    def __init__(self):
        """Inicializa el motor"""
        self.profiles: Dict[str, UserProfile] = {}
        self.user_behavior: Dict[str, List[Dict]] = {}  # user_id -> [actions]
    
    def get_or_create_profile(self, user_id: str) -> UserProfile:
        """Obtiene o crea perfil de usuario"""
        if user_id not in self.profiles:
            self.profiles[user_id] = UserProfile(
                user_id=user_id,
                preferences={}
            )
        return self.profiles[user_id]
    
    def update_profile(self, user_id: str, **updates):
        """
        Actualiza perfil de usuario
        
        Args:
            user_id: ID del usuario
            **updates: Campos a actualizar
        """
        profile = self.get_or_create_profile(user_id)
        
        for key, value in updates.items():
            if hasattr(profile, key):
                setattr(profile, key, value)
        
        profile.updated_at = datetime.now().isoformat()
    
    def record_behavior(self, user_id: str, action: str, data: Dict):
        """
        Registra comportamiento del usuario
        
        Args:
            user_id: ID del usuario
            action: Acción realizada
            data: Datos de la acción
        """
        if user_id not in self.user_behavior:
            self.user_behavior[user_id] = []
        
        behavior = {
            "action": action,
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
        
        self.user_behavior[user_id].append(behavior)
        
        # Mantener solo últimos 1000 eventos
        if len(self.user_behavior[user_id]) > 1000:
            self.user_behavior[user_id] = self.user_behavior[user_id][-1000:]
    
    def get_personalized_recommendations(self, user_id: str,
                                        analysis_result: Dict) -> Dict:
        """
        Obtiene recomendaciones personalizadas
        
        Args:
            user_id: ID del usuario
            analysis_result: Resultado del análisis
            
        Returns:
            Recomendaciones personalizadas
        """
        profile = self.get_or_create_profile(user_id)
        behavior = self.user_behavior.get(user_id, [])
        
        # Analizar comportamiento
        preferred_categories = self._analyze_preferences(behavior)
        time_preferences = self._analyze_time_preferences(behavior)
        
        recommendations = {
            "personalized": True,
            "based_on": {
                "profile": profile.to_dict(),
                "behavior_analysis": {
                    "preferred_categories": preferred_categories,
                    "time_preferences": time_preferences
                }
            },
            "recommendations": self._generate_personalized_recommendations(
                profile, analysis_result, preferred_categories
            )
        }
        
        return recommendations
    
    def _analyze_preferences(self, behavior: List[Dict]) -> List[str]:
        """Analiza preferencias del usuario"""
        category_counts = defaultdict(int)
        
        for event in behavior:
            if event["action"] == "view_product":
                category = event["data"].get("category", "")
                if category:
                    category_counts[category] += 1
        
        # Retornar top 3 categorías
        sorted_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
        return [cat for cat, count in sorted_categories[:3]]
    
    def _analyze_time_preferences(self, behavior: List[Dict]) -> Dict:
        """Analiza preferencias de tiempo"""
        hours = [datetime.fromisoformat(e["timestamp"]).hour for e in behavior]
        
        if not hours:
            return {"preferred_hour": None}
        
        # Calcular hora más común
        hour_counts = defaultdict(int)
        for hour in hours:
            hour_counts[hour] += 1
        
        preferred_hour = max(hour_counts.items(), key=lambda x: x[1])[0]
        
        return {
            "preferred_hour": preferred_hour,
            "most_active_period": self._get_period(preferred_hour)
        }
    
    def _get_period(self, hour: int) -> str:
        """Obtiene período del día"""
        if 6 <= hour < 12:
            return "morning"
        elif 12 <= hour < 18:
            return "afternoon"
        elif 18 <= hour < 22:
            return "evening"
        else:
            return "night"
    
    def _generate_personalized_recommendations(self, profile: UserProfile,
                                              analysis_result: Dict,
                                              preferred_categories: List[str]) -> List[Dict]:
        """Genera recomendaciones personalizadas"""
        recommendations = []
        
        # Basado en tipo de piel
        if profile.skin_type:
            recommendations.append({
                "type": "routine",
                "title": f"Rutina para {profile.skin_type}",
                "description": f"Rutina personalizada para tu tipo de piel: {profile.skin_type}",
                "priority": 1
            })
        
        # Basado en categorías preferidas
        for category in preferred_categories:
            recommendations.append({
                "type": "product",
                "title": f"Productos de {category}",
                "description": f"Basado en tus preferencias, te recomendamos productos de {category}",
                "priority": 2
            })
        
        return recommendations
    
    def get_user_insights(self, user_id: str) -> Dict:
        """Obtiene insights del usuario"""
        profile = self.get_or_create_profile(user_id)
        behavior = self.user_behavior.get(user_id, [])
        
        return {
            "profile": profile.to_dict(),
            "total_actions": len(behavior),
            "most_common_actions": self._get_most_common_actions(behavior),
            "engagement_score": self._calculate_engagement_score(behavior)
        }
    
    def _get_most_common_actions(self, behavior: List[Dict]) -> List[Dict]:
        """Obtiene acciones más comunes"""
        action_counts = defaultdict(int)
        for event in behavior:
            action_counts[event["action"]] += 1
        
        sorted_actions = sorted(action_counts.items(), key=lambda x: x[1], reverse=True)
        return [{"action": action, "count": count} for action, count in sorted_actions[:5]]
    
    def _calculate_engagement_score(self, behavior: List[Dict]) -> float:
        """Calcula score de engagement"""
        if not behavior:
            return 0.0
        
        # Score basado en frecuencia y diversidad de acciones
        unique_actions = len(set(e["action"] for e in behavior))
        total_actions = len(behavior)
        
        # Normalizar a 0-100
        score = min(100, (unique_actions * 10) + (total_actions / 10))
        return round(score, 2)






