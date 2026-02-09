"""
ML Recommendations Service - Recomendaciones ML avanzadas
"""

import logging
from typing import Dict, Any, List, Optional
from ..core.models import StoreType, DesignStyle
from ..services.llm_service import LLMService

logger = logging.getLogger(__name__)


class MLRecommendationsService:
    """Servicio para recomendaciones ML avanzadas"""
    
    def __init__(self, llm_service: Optional[LLMService] = None):
        self.llm_service = llm_service or LLMService()
        self.user_preferences: Dict[str, Dict[str, Any]] = {}
        self.design_similarities: Dict[str, List[str]] = {}
    
    async def generate_personalized_recommendations(
        self,
        user_id: str,
        design_history: List[Dict[str, Any]],
        current_design: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generar recomendaciones personalizadas usando ML"""
        
        # Analizar historial del usuario
        user_profile = self._build_user_profile(user_id, design_history)
        
        # Recomendaciones basadas en perfil
        style_recommendations = self._recommend_styles(user_profile)
        feature_recommendations = self._recommend_features(user_profile, current_design)
        optimization_recommendations = self._recommend_optimizations(user_profile, current_design)
        
        return {
            "user_profile": user_profile,
            "style_recommendations": style_recommendations,
            "feature_recommendations": feature_recommendations,
            "optimization_recommendations": optimization_recommendations,
            "similar_designs": self._find_similar_designs(current_design) if current_design else []
        }
    
    def _build_user_profile(
        self,
        user_id: str,
        design_history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Construir perfil de usuario"""
        
        if not design_history:
            return {
                "user_id": user_id,
                "preferred_store_types": [],
                "preferred_styles": [],
                "average_budget": 0,
                "design_count": 0
            }
        
        # Analizar preferencias
        store_types = {}
        styles = {}
        budgets = []
        
        for design in design_history:
            store_type = design.get("store_type")
            if store_type:
                store_types[store_type] = store_types.get(store_type, 0) + 1
            
            style = design.get("style")
            if style:
                styles[style] = styles.get(style, 0) + 1
            
            financial = design.get("financial_analysis", {})
            budget = financial.get("initial_investment", {}).get("total", 0)
            if budget > 0:
                budgets.append(budget)
        
        # Preferencias más comunes
        preferred_store_types = sorted(store_types.items(), key=lambda x: x[1], reverse=True)[:3]
        preferred_styles = sorted(styles.items(), key=lambda x: x[1], reverse=True)[:3]
        
        return {
            "user_id": user_id,
            "preferred_store_types": [t[0] for t in preferred_store_types],
            "preferred_styles": [s[0] for s in preferred_styles],
            "average_budget": sum(budgets) / len(budgets) if budgets else 0,
            "design_count": len(design_history),
            "store_type_distribution": store_types,
            "style_distribution": styles
        }
    
    def _recommend_styles(
        self,
        user_profile: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Recomendar estilos"""
        preferred_styles = user_profile.get("preferred_styles", [])
        
        # Si no hay preferencias, recomendar estilos populares
        if not preferred_styles:
            return [
                {"style": "modern", "reason": "Más popular y versátil", "confidence": 0.8},
                {"style": "minimalist", "reason": "Tendencia actual", "confidence": 0.7}
            ]
        
        # Recomendar estilos complementarios
        style_complements = {
            "modern": ["minimalist", "industrial"],
            "classic": ["luxury", "rustic"],
            "minimalist": ["modern", "eco_friendly"],
            "industrial": ["modern", "rustic"],
            "rustic": ["classic", "vintage"],
            "luxury": ["classic", "modern"],
            "eco_friendly": ["minimalist", "modern"],
            "vintage": ["rustic", "classic"]
        }
        
        recommendations = []
        for preferred in preferred_styles[:2]:
            complements = style_complements.get(preferred, [])
            for comp in complements:
                if comp not in preferred_styles:
                    recommendations.append({
                        "style": comp,
                        "reason": f"Complementa tu estilo preferido '{preferred}'",
                        "confidence": 0.7
                    })
        
        return recommendations[:5]
    
    def _recommend_features(
        self,
        user_profile: Dict[str, Any],
        current_design: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Recomendar funcionalidades"""
        recommendations = []
        
        # Basado en historial
        design_count = user_profile.get("design_count", 0)
        
        if design_count > 5:
            recommendations.append({
                "feature": "advanced_analytics",
                "reason": "Con tu experiencia, podrías beneficiarte de análisis avanzados",
                "priority": "high"
            })
        
        if current_design:
            # Recomendar features faltantes
            if not current_design.get("competitor_analysis"):
                recommendations.append({
                    "feature": "competitor_analysis",
                    "reason": "Análisis de competencia puede mejorar tu diseño",
                    "priority": "medium"
                })
            
            if not current_design.get("financial_analysis"):
                recommendations.append({
                    "feature": "financial_analysis",
                    "reason": "Análisis financiero esencial para viabilidad",
                    "priority": "high"
                })
        
        return recommendations
    
    def _recommend_optimizations(
        self,
        user_profile: Dict[str, Any],
        current_design: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Recomendar optimizaciones"""
        recommendations = []
        
        if not current_design:
            return recommendations
        
        # Optimizaciones basadas en perfil
        avg_budget = user_profile.get("average_budget", 0)
        current_budget = current_design.get("financial_analysis", {}).get("initial_investment", {}).get("total", 0)
        
        if current_budget > avg_budget * 1.2:
            recommendations.append({
                "type": "budget",
                "suggestion": "Presupuesto actual es 20% más alto que tu promedio",
                "action": "Considerar optimización de costos"
            })
        
        return recommendations
    
    def _find_similar_designs(
        self,
        current_design: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Encontrar diseños similares"""
        # En producción, usar embeddings o clustering
        return [
            {
                "store_id": "similar_1",
                "similarity_score": 0.85,
                "similarity_reasons": ["Mismo tipo de tienda", "Estilo similar"]
            }
        ]
    
    async def recommend_based_on_success(
        self,
        store_type: StoreType,
        location: Optional[str] = None
    ) -> Dict[str, Any]:
        """Recomendar basado en diseños exitosos similares"""
        
        if self.llm_service.client:
            try:
                return await self._recommend_with_llm(store_type, location)
            except Exception as e:
                logger.error(f"Error en recomendación ML: {e}")
                return self._recommend_basic(store_type)
        else:
            return self._recommend_basic(store_type)
    
    async def _recommend_with_llm(
        self,
        store_type: StoreType,
        location: Optional[str]
    ) -> Dict[str, Any]:
        """Recomendar usando LLM"""
        prompt = f"""Basado en datos de tiendas exitosas de tipo {store_type.value} en {location or "áreas similares"},
        proporciona recomendaciones específicas para:
        1. Elementos de diseño más efectivos
        2. Estrategias de marketing que funcionan
        3. Características que aumentan éxito
        4. Errores comunes a evitar
        
        Responde en formato JSON con estas claves:
        - design_elements: array de strings
        - marketing_strategies: array de strings
        - success_factors: array de strings
        - common_mistakes: array de strings"""
        
        result = await self.llm_service.generate_structured(
            prompt=prompt,
            system_prompt="Eres un experto en análisis de negocios exitosos y recomendaciones basadas en datos."
        )
        
        return result if result else self._recommend_basic(store_type)
    
    def _recommend_basic(self, store_type: StoreType) -> Dict[str, Any]:
        """Recomendaciones básicas"""
        return {
            "design_elements": [
                "Iluminación adecuada",
                "Flujo de tráfico optimizado",
                "Espacios funcionales"
            ],
            "marketing_strategies": [
                "Marketing local",
                "Redes sociales",
                "Programa de fidelización"
            ],
            "success_factors": [
                "Ubicación estratégica",
                "Calidad de productos/servicios",
                "Atención al cliente"
            ],
            "common_mistakes": [
                "Subestimar costos",
                "No investigar competencia",
                "Ignorar feedback de clientes"
            ]
        }




