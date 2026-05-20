"""
Analizador de adicciones - Evalúa tipo, severidad y características de la adicción
"""

from typing import Dict, List, Optional
from datetime import datetime
from pydantic import BaseModel
import json


class AddictionAssessment(BaseModel):
    """Modelo para evaluación de adicción"""
    addiction_type: str
    severity: str  # leve, moderada, severa, crítica
    frequency: str  # diaria, semanal, ocasional
    duration_years: Optional[float] = None
    daily_cost: Optional[float] = None
    triggers: List[str] = []
    motivations: List[str] = []
    previous_attempts: int = 0
    support_system: bool = False
    medical_conditions: List[str] = []
    additional_info: Optional[str] = None


class AddictionAnalyzer:
    """Analizador de adicciones usando IA"""
    
    def __init__(self, openai_client=None):
        """
        Inicializa el analizador
        
        Args:
            openai_client: Cliente de OpenAI (opcional)
        """
        self.openai_client = openai_client
    
    def assess_addiction(self, assessment_data: Dict) -> Dict:
        """
        Evalúa una adicción basándose en los datos proporcionados
        
        Args:
            assessment_data: Diccionario con datos de evaluación
        
        Returns:
            Diccionario con análisis completo
        """
        try:
            assessment = AddictionAssessment(**assessment_data)
        except Exception as e:
            return {
                "error": f"Error validando datos: {str(e)}",
                "success": False
            }
        
        # Análisis básico
        analysis = {
            "addiction_type": assessment.addiction_type,
            "severity": assessment.severity,
            "risk_level": self._calculate_risk_level(assessment),
            "recommended_approach": self._recommend_approach(assessment),
            "key_insights": self._generate_insights(assessment),
            "immediate_actions": self._get_immediate_actions(assessment),
            "long_term_goals": self._get_long_term_goals(assessment),
            "success": True
        }
        
        # Si hay cliente de OpenAI, usar IA para análisis más profundo
        if self.openai_client:
            enhanced_analysis = self._enhance_with_ai(assessment, analysis)
            analysis.update(enhanced_analysis)
        
        return analysis
    
    def _calculate_risk_level(self, assessment: AddictionAssessment) -> str:
        """Calcula nivel de riesgo basado en severidad y otros factores"""
        risk_factors = 0
        
        if assessment.severity == "crítica":
            risk_factors += 3
        elif assessment.severity == "severa":
            risk_factors += 2
        elif assessment.severity == "moderada":
            risk_factors += 1
        
        if assessment.frequency == "diaria":
            risk_factors += 2
        elif assessment.frequency == "semanal":
            risk_factors += 1
        
        if assessment.duration_years and assessment.duration_years > 10:
            risk_factors += 1
        
        if not assessment.support_system:
            risk_factors += 1
        
        if risk_factors >= 5:
            return "alto"
        elif risk_factors >= 3:
            return "medio"
        else:
            return "bajo"
    
    def _recommend_approach(self, assessment: AddictionAssessment) -> str:
        """Recomienda enfoque de tratamiento"""
        if assessment.severity in ["severa", "crítica"]:
            return "Abstinencia total con supervisión médica recomendada"
        elif assessment.severity == "moderada":
            return "Abstinencia total o reducción gradual según preferencia"
        else:
            return "Reducción gradual o abstinencia total"
    
    def _generate_insights(self, assessment: AddictionAssessment) -> List[str]:
        """Genera insights clave sobre la adicción"""
        insights = []
        
        if assessment.triggers:
            insights.append(f"Triggers identificados: {', '.join(assessment.triggers[:3])}")
        
        if assessment.previous_attempts > 0:
            insights.append(f"Has intentado dejar {assessment.previous_attempts} vez(es) antes. Cada intento te acerca más al éxito.")
        
        if assessment.support_system:
            insights.append("Tienes un sistema de apoyo, lo cual es muy positivo para tu recuperación.")
        else:
            insights.append("Considera buscar apoyo de familiares, amigos o grupos de apoyo.")
        
        if assessment.motivations:
            insights.append(f"Tus motivaciones incluyen: {', '.join(assessment.motivations[:2])}")
        
        return insights
    
    def _get_immediate_actions(self, assessment: AddictionAssessment) -> List[str]:
        """Obtiene acciones inmediatas recomendadas"""
        actions = []
        
        if assessment.severity in ["severa", "crítica"]:
            actions.append("Consultar con un profesional de la salud")
            actions.append("Considerar programa de tratamiento supervisado")
        
        actions.append("Identificar y evitar triggers principales")
        actions.append("Crear un plan de emergencia para momentos difíciles")
        actions.append("Establecer fecha objetivo para comenzar")
        
        if not assessment.support_system:
            actions.append("Buscar grupos de apoyo o consejería")
        
        return actions
    
    def _get_long_term_goals(self, assessment: AddictionAssessment) -> List[str]:
        """Obtiene metas a largo plazo"""
        goals = [
            "Mantener sobriedad continua",
            "Desarrollar estrategias de afrontamiento saludables",
            "Mejorar calidad de vida general",
            "Construir una red de apoyo sólida"
        ]
        
        if assessment.daily_cost:
            goals.append(f"Ahorrar ${assessment.daily_cost * 365:.2f} al año")
        
        return goals
    
    def _enhance_with_ai(self, assessment: AddictionAssessment, base_analysis: Dict) -> Dict:
        """Mejora el análisis usando IA (OpenAI)"""
        if not self.openai_client:
            return {}
        
        try:
            prompt = f"""
            Analiza esta situación de adicción y proporciona insights adicionales:
            
            Tipo: {assessment.addiction_type}
            Severidad: {assessment.severity}
            Frecuencia: {assessment.frequency}
            Duración: {assessment.duration_years} años
            Triggers: {', '.join(assessment.triggers)}
            Motivaciones: {', '.join(assessment.motivations)}
            
            Proporciona:
            1. Análisis psicológico breve
            2. Estrategias personalizadas
            3. Mensaje motivacional
            
            Responde en formato JSON con las claves: psychological_analysis, personalized_strategies, motivational_message
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Eres un experto en adicciones y recuperación. Proporciona análisis compasivo y útil."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            
            ai_response = response.choices[0].message.content
            try:
                ai_data = json.loads(ai_response)
                return {
                    "ai_psychological_analysis": ai_data.get("psychological_analysis", ""),
                    "ai_personalized_strategies": ai_data.get("personalized_strategies", []),
                    "ai_motivational_message": ai_data.get("motivational_message", "")
                }
            except json.JSONDecodeError:
                return {
                    "ai_enhanced_insights": ai_response
                }
        except Exception as e:
            # Si falla la IA, retornar análisis base
            return {}

