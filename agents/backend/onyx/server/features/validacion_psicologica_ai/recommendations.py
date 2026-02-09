"""
Sistema de Recomendaciones Avanzado para Validación Psicológica AI
==================================================================
Recomendaciones personalizadas basadas en análisis profundo
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum
import structlog

from .models import PsychologicalProfile, ValidationReport

logger = structlog.get_logger()


class RecommendationCategory(str, Enum):
    """Categorías de recomendaciones"""
    MENTAL_HEALTH = "mental_health"
    SOCIAL_INTERACTION = "social_interaction"
    CONTENT_STRATEGY = "content_strategy"
    PRIVACY = "privacy"
    WORK_LIFE_BALANCE = "work_life_balance"
    EMOTIONAL_WELLBEING = "emotional_wellbeing"
    PERSONAL_GROWTH = "personal_growth"


class RecommendationPriority(str, Enum):
    """Prioridad de recomendación"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class Recommendation:
    """Representa una recomendación"""
    
    def __init__(
        self,
        title: str,
        description: str,
        category: RecommendationCategory,
        priority: RecommendationPriority,
        action_items: List[str],
        rationale: str,
        resources: Optional[List[Dict[str, str]]] = None
    ):
        self.title = title
        self.description = description
        self.category = category
        self.priority = priority
        self.action_items = action_items
        self.rationale = rationale
        self.resources = resources or []
        self.created_at = datetime.utcnow()
        self.id = f"{category.value}_{self.created_at.timestamp()}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario"""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "category": self.category.value,
            "priority": self.priority.value,
            "action_items": self.action_items,
            "rationale": self.rationale,
            "resources": self.resources,
            "created_at": self.created_at.isoformat()
        }


class RecommendationEngine:
    """Motor de recomendaciones avanzado"""
    
    def __init__(self):
        """Inicializar motor"""
        logger.info("RecommendationEngine initialized")
    
    def generate_recommendations(
        self,
        profile: PsychologicalProfile,
        report: Optional[ValidationReport] = None
    ) -> List[Recommendation]:
        """
        Generar recomendaciones basadas en perfil y reporte
        
        Args:
            profile: Perfil psicológico
            report: Reporte de validación (opcional)
            
        Returns:
            Lista de recomendaciones
        """
        recommendations = []
        
        # Analizar rasgos de personalidad
        recommendations.extend(self._analyze_personality_traits(profile))
        
        # Analizar estado emocional
        recommendations.extend(self._analyze_emotional_state(profile))
        
        # Analizar factores de riesgo
        recommendations.extend(self._analyze_risk_factors(profile))
        
        # Analizar patrones de comportamiento
        if report:
            recommendations.extend(self._analyze_behavioral_patterns(profile, report))
        
        # Analizar sentimientos
        if report and report.sentiment_analysis:
            recommendations.extend(self._analyze_sentiment(profile, report))
        
        # Ordenar por prioridad
        priority_order = {
            RecommendationPriority.URGENT: 4,
            RecommendationPriority.HIGH: 3,
            RecommendationPriority.MEDIUM: 2,
            RecommendationPriority.LOW: 1
        }
        recommendations.sort(key=lambda r: priority_order.get(r.priority, 0), reverse=True)
        
        logger.info(
            "Recommendations generated",
            count=len(recommendations),
            user_id=str(profile.user_id)
        )
        
        return recommendations
    
    def _analyze_personality_traits(
        self,
        profile: PsychologicalProfile
    ) -> List[Recommendation]:
        """Analizar rasgos de personalidad y generar recomendaciones"""
        recommendations = []
        
        traits = profile.personality_traits
        
        # Neuroticismo alto
        if traits.get("neuroticism", 0.5) > 0.7:
            recommendations.append(Recommendation(
                title="Gestión del Estrés y Ansiedad",
                description="Se detectó un nivel elevado de neuroticismo. Es importante desarrollar estrategias para manejar el estrés.",
                category=RecommendationCategory.MENTAL_HEALTH,
                priority=RecommendationPriority.HIGH,
                action_items=[
                    "Practicar técnicas de respiración profunda diariamente",
                    "Considerar actividades de mindfulness o meditación",
                    "Mantener un diario de emociones para identificar patrones",
                    "Establecer rutinas de sueño regulares"
                ],
                rationale=f"El neuroticismo ({traits.get('neuroticism', 0):.2f}) está por encima del rango saludable. Esto puede indicar mayor sensibilidad al estrés.",
                resources=[
                    {"title": "Técnicas de Mindfulness", "url": "https://example.com/mindfulness"},
                    {"title": "Gestión del Estrés", "url": "https://example.com/stress-management"}
                ]
            ))
        
        # Extraversión baja
        if traits.get("extraversion", 0.5) < 0.4:
            recommendations.append(Recommendation(
                title="Fortalecer Conexiones Sociales",
                description="Tu nivel de extraversión es bajo. Considera actividades que fomenten la interacción social.",
                category=RecommendationCategory.SOCIAL_INTERACTION,
                priority=RecommendationPriority.MEDIUM,
                action_items=[
                    "Participar en grupos o comunidades con intereses similares",
                    "Programar tiempo regular para interacciones sociales",
                    "Considerar actividades grupales que disfrutes",
                    "Mantener contacto regular con amigos y familia"
                ],
                rationale=f"La extraversión ({traits.get('extraversion', 0):.2f}) está por debajo del promedio. Las conexiones sociales son importantes para el bienestar.",
                resources=[
                    {"title": "Construir Conexiones Sociales", "url": "https://example.com/social-connections"}
                ]
            ))
        
        # Conscientiousness bajo
        if traits.get("conscientiousness", 0.5) < 0.4:
            recommendations.append(Recommendation(
                title="Desarrollar Organización y Planificación",
                description="Tu nivel de consciencia es bajo. Mejorar la organización puede aumentar la productividad y reducir el estrés.",
                category=RecommendationCategory.WORK_LIFE_BALANCE,
                priority=RecommendationPriority.MEDIUM,
                action_items=[
                    "Usar herramientas de planificación (calendarios, listas de tareas)",
                    "Establecer rutinas diarias y semanales",
                    "Dividir tareas grandes en pasos más pequeños",
                    "Establecer metas claras y alcanzables"
                ],
                rationale=f"La consciencia ({traits.get('conscientiousness', 0):.2f}) está por debajo del promedio. Mejorar la organización puede tener beneficios significativos.",
                resources=[
                    {"title": "Técnicas de Organización", "url": "https://example.com/organization"}
                ]
            ))
        
        return recommendations
    
    def _analyze_emotional_state(
        self,
        profile: PsychologicalProfile
    ) -> List[Recommendation]:
        """Analizar estado emocional y generar recomendaciones"""
        recommendations = []
        
        emotional_state = profile.emotional_state
        sentiment = emotional_state.get("overall_sentiment", "neutral")
        stress_level = emotional_state.get("stress_level", 0.0)
        
        # Sentimiento negativo
        if sentiment == "negative":
            recommendations.append(Recommendation(
                title="Apoyo Emocional y Bienestar",
                description="Se detectó un sentimiento predominantemente negativo. Es importante cuidar tu bienestar emocional.",
                category=RecommendationCategory.EMOTIONAL_WELLBEING,
                priority=RecommendationPriority.HIGH if stress_level > 0.7 else RecommendationPriority.MEDIUM,
                action_items=[
                    "Considerar hablar con un profesional de salud mental",
                    "Practicar actividades que disfrutes regularmente",
                    "Mantener contacto con personas de apoyo",
                    "Considerar técnicas de terapia cognitivo-conductual"
                ],
                rationale=f"El sentimiento general es negativo y el nivel de estrés es {stress_level:.1%}. El apoyo profesional puede ser beneficioso.",
                resources=[
                    {"title": "Recursos de Salud Mental", "url": "https://example.com/mental-health"},
                    {"title": "Líneas de Apoyo", "url": "https://example.com/support-lines"}
                ]
            ))
        
        # Estrés alto
        if stress_level > 0.7:
            recommendations.append(Recommendation(
                title="Reducción del Estrés",
                description="Tu nivel de estrés es alto. Es importante tomar medidas para reducirlo.",
                category=RecommendationCategory.MENTAL_HEALTH,
                priority=RecommendationPriority.HIGH,
                action_items=[
                    "Identificar y reducir fuentes de estrés",
                    "Practicar técnicas de relajación regularmente",
                    "Asegurar tiempo adecuado para descanso",
                    "Considerar ejercicio físico regular"
                ],
                rationale=f"El nivel de estrés ({stress_level:.1%}) está en un rango alto. La gestión proactiva del estrés es esencial.",
                resources=[
                    {"title": "Técnicas de Relajación", "url": "https://example.com/relaxation"}
                ]
            ))
        
        return recommendations
    
    def _analyze_risk_factors(
        self,
        profile: PsychologicalProfile
    ) -> List[Recommendation]:
        """Analizar factores de riesgo y generar recomendaciones"""
        recommendations = []
        
        if profile.risk_factors:
            for risk_factor in profile.risk_factors:
                if "depression" in risk_factor.lower() or "depresión" in risk_factor.lower():
                    recommendations.append(Recommendation(
                        title="Atención a Signos de Depresión",
                        description="Se detectaron indicadores que requieren atención profesional.",
                        category=RecommendationCategory.MENTAL_HEALTH,
                        priority=RecommendationPriority.URGENT,
                        action_items=[
                            "Consultar con un profesional de salud mental",
                            "Mantener contacto regular con personas de apoyo",
                            "Seguir rutinas diarias estructuradas",
                            "Evitar el aislamiento social"
                        ],
                        rationale=f"Factor de riesgo detectado: {risk_factor}",
                        resources=[
                            {"title": "Recursos de Depresión", "url": "https://example.com/depression-resources"},
                            {"title": "Ayuda Inmediata", "url": "https://example.com/immediate-help"}
                        ]
                    ))
        
        return recommendations
    
    def _analyze_behavioral_patterns(
        self,
        profile: PsychologicalProfile,
        report: ValidationReport
    ) -> List[Recommendation]:
        """Analizar patrones de comportamiento y generar recomendaciones"""
        recommendations = []
        
        # Análisis de engagement
        interaction_patterns = report.interaction_patterns
        engagement_level = interaction_patterns.get("engagement_level", "medium")
        
        if engagement_level == "low":
            recommendations.append(Recommendation(
                title="Mejorar Engagement en Redes Sociales",
                description="Tu nivel de engagement es bajo. Considera estrategias para aumentar la interacción.",
                category=RecommendationCategory.CONTENT_STRATEGY,
                priority=RecommendationPriority.LOW,
                action_items=[
                    "Publicar contenido más frecuentemente",
                    "Interactuar más con la comunidad",
                    "Responder a comentarios y mensajes",
                    "Crear contenido que genere conversación"
                ],
                rationale="El engagement bajo puede limitar las conexiones sociales y el impacto del contenido.",
                resources=[
                    {"title": "Estrategias de Engagement", "url": "https://example.com/engagement"}
                ]
            ))
        
        return recommendations
    
    def _analyze_sentiment(
        self,
        profile: PsychologicalProfile,
        report: ValidationReport
    ) -> List[Recommendation]:
        """Analizar sentimientos y generar recomendaciones"""
        recommendations = []
        
        sentiment_analysis = report.sentiment_analysis
        sentiment_dist = sentiment_analysis.get("sentiment_distribution", {})
        negative_ratio = sentiment_dist.get("negative", 0.0)
        
        if negative_ratio > 0.4:
            recommendations.append(Recommendation(
                title="Balancear Contenido en Redes Sociales",
                description="Una proporción significativa de tu contenido tiene sentimiento negativo. Considera balancear con contenido más positivo.",
                category=RecommendationCategory.CONTENT_STRATEGY,
                priority=RecommendationPriority.MEDIUM,
                action_items=[
                    "Incluir más contenido positivo en tus publicaciones",
                    "Compartir logros y momentos positivos",
                    "Considerar el impacto del contenido en tu bienestar",
                    "Equilibrar contenido personal y profesional"
                ],
                rationale=f"El {negative_ratio:.1%} del contenido tiene sentimiento negativo. Un balance puede mejorar el bienestar.",
                resources=[
                    {"title": "Contenido Positivo", "url": "https://example.com/positive-content"}
                ]
            ))
        
        return recommendations


# Instancia global del motor de recomendaciones
recommendation_engine = RecommendationEngine()




