"""
Prevención de recaídas - Detecta riesgos y proporciona estrategias de prevención
"""

from typing import Dict, List, Optional
from datetime import datetime
from utils.helpers import calculate_relapse_risk_score


class RelapsePrevention:
    """Sistema de prevención de recaídas"""
    
    def __init__(self):
        """Inicializa el sistema de prevención"""
        self.warning_signs = self._load_warning_signs()
        self.coping_strategies = self._load_coping_strategies()
    
    def check_relapse_risk(
        self,
        user_id: str,
        days_sober: int,
        current_state: Dict,
        history: Optional[List[Dict]] = None
    ) -> Dict:
        """
        Evalúa riesgo de recaída
        
        Args:
            user_id: ID del usuario
            days_sober: Días de sobriedad
            current_state: Estado actual (estrés, triggers, etc.)
            history: Historial de entradas (opcional)
        
        Returns:
            Análisis de riesgo de recaída
        """
        stress_level = current_state.get("stress_level", 5)
        support_level = current_state.get("support_level", 5)
        triggers_present = len(current_state.get("triggers", [])) > 0
        previous_relapses = current_state.get("previous_relapses", 0)
        
        risk_score = calculate_relapse_risk_score(
            days_sober,
            stress_level,
            support_level,
            triggers_present,
            previous_relapses
        )
        
        # Detectar señales de advertencia
        warning_signs = self._detect_warning_signs(current_state)
        
        # Determinar nivel de riesgo
        risk_level = self._determine_risk_level(risk_score)
        
        # Generar recomendaciones
        recommendations = self._generate_recommendations(
            risk_score,
            risk_level,
            warning_signs,
            current_state
        )
        
        # Plan de emergencia si es necesario
        emergency_plan = None
        if risk_level in ["alto", "crítico"]:
            emergency_plan = self._generate_emergency_plan(user_id, current_state)
        
        return {
            "user_id": user_id,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "warning_signs": warning_signs,
            "recommendations": recommendations,
            "emergency_plan": emergency_plan,
            "checked_at": datetime.now().isoformat()
        }
    
    def get_coping_strategies(
        self,
        situation: str,
        trigger_type: Optional[str] = None
    ) -> List[Dict]:
        """
        Obtiene estrategias de afrontamiento para una situación específica
        
        Args:
            situation: Situación actual
            trigger_type: Tipo de trigger (opcional)
        
        Returns:
            Lista de estrategias de afrontamiento
        """
        strategies = []
        
        # Estrategias generales
        strategies.extend(self.coping_strategies.get("general", []))
        
        # Estrategias específicas por situación
        if situation.lower() in ["craving", "ansiedad", "deseo"]:
            strategies.extend(self.coping_strategies.get("cravings", []))
        
        if situation.lower() in ["estrés", "stress", "presión"]:
            strategies.extend(self.coping_strategies.get("stress", []))
        
        if situation.lower() in ["social", "fiesta", "evento"]:
            strategies.extend(self.coping_strategies.get("social", []))
        
        # Estrategias específicas por trigger
        if trigger_type:
            trigger_strategies = self.coping_strategies.get(trigger_type.lower(), [])
            strategies.extend(trigger_strategies)
        
        return strategies[:5]  # Retornar top 5
    
    def generate_emergency_plan(
        self,
        user_id: str,
        current_situation: Dict
    ) -> Dict:
        """
        Genera plan de emergencia para situaciones de alto riesgo
        
        Args:
            user_id: ID del usuario
            current_situation: Situación actual
        
        Returns:
            Plan de emergencia
        """
        return self._generate_emergency_plan(user_id, current_situation)
    
    def _detect_warning_signs(self, current_state: Dict) -> List[str]:
        """Detecta señales de advertencia"""
        detected = []
        
        stress_level = current_state.get("stress_level", 5)
        if stress_level >= 8:
            detected.append("Alto nivel de estrés")
        
        if current_state.get("isolation", False):
            detected.append("Aislamiento social")
        
        if current_state.get("negative_thinking", False):
            detected.append("Pensamientos negativos")
        
        if current_state.get("romanticizing", False):
            detected.append("Romantizar el consumo")
        
        if current_state.get("skipping_support", False):
            detected.append("Evitar apoyo/sesiones")
        
        triggers = current_state.get("triggers", [])
        if len(triggers) >= 3:
            detected.append("Múltiples triggers presentes")
        
        return detected
    
    def _determine_risk_level(self, risk_score: float) -> str:
        """Determina nivel de riesgo basado en score"""
        if risk_score >= 75:
            return "crítico"
        elif risk_score >= 50:
            return "alto"
        elif risk_score >= 25:
            return "medio"
        else:
            return "bajo"
    
    def _generate_recommendations(
        self,
        risk_score: float,
        risk_level: str,
        warning_signs: List[str],
        current_state: Dict
    ) -> List[str]:
        """Genera recomendaciones basadas en el riesgo"""
        recommendations = []
        
        if risk_level == "crítico":
            recommendations.append("⚠️ RIESGO CRÍTICO: Contacta inmediatamente con tu apoyo o profesional")
            recommendations.append("Busca un lugar seguro y alejado de triggers")
            recommendations.append("Llama a una línea de crisis si es necesario")
        
        elif risk_level == "alto":
            recommendations.append("Riesgo alto detectado - Actúa ahora")
            recommendations.append("Contacta con tu sistema de apoyo")
            recommendations.append("Implementa estrategias de afrontamiento inmediatas")
        
        if "Alto nivel de estrés" in warning_signs:
            recommendations.append("Practica técnicas de relajación inmediatas")
            recommendations.append("Considera ejercicio físico para reducir estrés")
        
        if "Aislamiento social" in warning_signs:
            recommendations.append("Contacta con alguien de confianza ahora")
            recommendations.append("Considera asistir a una reunión de apoyo")
        
        if "Múltiples triggers presentes" in warning_signs:
            recommendations.append("Sal de la situación actual si es posible")
            recommendations.append("Usa técnicas de distracción")
        
        # Recomendaciones generales
        if not recommendations:
            recommendations.append("Continúa con tu plan de recuperación")
            recommendations.append("Mantén contacto regular con tu apoyo")
        
        return recommendations
    
    def _generate_emergency_plan(
        self,
        user_id: str,
        current_situation: Dict
    ) -> Dict:
        """Genera plan de emergencia detallado"""
        return {
            "user_id": user_id,
            "created_at": datetime.now().isoformat(),
            "steps": [
                {
                    "step": 1,
                    "action": "Detente y respira",
                    "description": "Toma 10 respiraciones profundas. No actúes impulsivamente."
                },
                {
                    "step": 2,
                    "action": "Contacta apoyo inmediato",
                    "description": "Llama a tu sponsor, consejero, o línea de crisis"
                },
                {
                    "step": 3,
                    "action": "Sal del entorno",
                    "description": "Si estás en un lugar con triggers, sal inmediatamente"
                },
                {
                    "step": 4,
                    "action": "Usa estrategias de afrontamiento",
                    "description": "Implementa técnicas de distracción o relajación"
                },
                {
                    "step": 5,
                    "action": "Recuerda tus motivaciones",
                    "description": "Lee tu lista de razones para estar sobrio"
                }
            ],
            "emergency_contacts": current_situation.get("emergency_contacts", []),
            "safe_places": current_situation.get("safe_places", []),
            "distraction_activities": current_situation.get("distraction_activities", [
                "Ejercicio",
                "Llamar a un amigo",
                "Leer",
                "Meditar",
                "Pasear"
            ])
        }
    
    def _load_warning_signs(self) -> Dict:
        """Carga señales de advertencia comunes"""
        return {
            "emotional": [
                "Aumento de estrés o ansiedad",
                "Sentimientos de depresión",
                "Irritabilidad extrema",
                "Pensamientos negativos constantes"
            ],
            "behavioral": [
                "Aislamiento social",
                "Evitar reuniones de apoyo",
                "Cambios en rutina",
                "Romantizar el consumo pasado"
            ],
            "environmental": [
                "Exposición a triggers",
                "Visitar lugares asociados con consumo",
                "Contacto con personas que consumen"
            ]
        }
    
    def _load_coping_strategies(self) -> Dict[str, List[Dict]]:
        """Carga estrategias de afrontamiento"""
        return {
            "general": [
                {
                    "name": "Técnica 4-7-8",
                    "description": "Inhala 4 segundos, mantén 7, exhala 8. Repite 4 veces.",
                    "duration": "2 minutos"
                },
                {
                    "name": "Llamar a apoyo",
                    "description": "Contacta con alguien de tu sistema de apoyo",
                    "duration": "5-10 minutos"
                },
                {
                    "name": "Recordar motivaciones",
                    "description": "Lee tu lista de razones para estar sobrio",
                    "duration": "3 minutos"
                }
            ],
            "cravings": [
                {
                    "name": "Delay (Retrasar)",
                    "description": "Espera 15 minutos antes de actuar. Los cravings suelen pasar.",
                    "duration": "15 minutos"
                },
                {
                    "name": "Distracción",
                    "description": "Haz algo que requiera concentración (ejercicio, leer, llamar)",
                    "duration": "20-30 minutos"
                },
                {
                    "name": "Agua fría",
                    "description": "Bebe agua fría o lávate la cara con agua fría",
                    "duration": "2 minutos"
                }
            ],
            "stress": [
                {
                    "name": "Ejercicio físico",
                    "description": "Caminar, correr, o cualquier actividad física",
                    "duration": "20-30 minutos"
                },
                {
                    "name": "Meditación",
                    "description": "Meditación guiada o mindfulness",
                    "duration": "10-15 minutos"
                },
                {
                    "name": "Técnicas de relajación",
                    "description": "Relajación muscular progresiva",
                    "duration": "15 minutos"
                }
            ],
            "social": [
                {
                    "name": "Tener un plan",
                    "description": "Antes de eventos sociales, ten un plan de salida",
                    "duration": "Preparación previa"
                },
                {
                    "name": "Llevar apoyo",
                    "description": "Asiste con alguien que sepa de tu recuperación",
                    "duration": "Durante el evento"
                },
                {
                    "name": "Tener excusa lista",
                    "description": "Prepara una excusa educada para salir temprano",
                    "duration": "Preparación previa"
                }
            ]
        }

