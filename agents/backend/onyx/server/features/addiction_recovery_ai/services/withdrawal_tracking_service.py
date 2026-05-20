"""
Servicio de Seguimiento de Síntomas de Abstinencia - Monitoreo de síntomas durante abstinencia
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
from enum import Enum


class SymptomSeverity(str, Enum):
    """Severidad de síntomas"""
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"


class WithdrawalTrackingService:
    """Servicio de seguimiento de síntomas de abstinencia"""
    
    def __init__(self):
        """Inicializa el servicio de seguimiento de síntomas"""
        self.symptom_database = self._load_symptom_database()
    
    def record_symptom(
        self,
        user_id: str,
        symptom_name: str,
        severity: str,
        notes: Optional[str] = None,
        duration_minutes: Optional[int] = None
    ) -> Dict:
        """
        Registra un síntoma de abstinencia
        
        Args:
            user_id: ID del usuario
            symptom_name: Nombre del síntoma
            severity: Severidad
            notes: Notas adicionales
            duration_minutes: Duración en minutos (opcional)
        
        Returns:
            Síntoma registrado
        """
        symptom = {
            "id": f"symptom_{datetime.now().timestamp()}",
            "user_id": user_id,
            "symptom_name": symptom_name,
            "severity": severity,
            "notes": notes,
            "duration_minutes": duration_minutes,
            "recorded_at": datetime.now().isoformat()
        }
        
        return symptom
    
    def get_withdrawal_timeline(
        self,
        user_id: str,
        addiction_type: str,
        days_sober: int
    ) -> Dict:
        """
        Obtiene línea de tiempo esperada de síntomas de abstinencia
        
        Args:
            user_id: ID del usuario
            addiction_type: Tipo de adicción
            days_sober: Días de sobriedad
        
        Returns:
            Línea de tiempo de síntomas
        """
        timeline = self._get_expected_timeline(addiction_type, days_sober)
        
        return {
            "user_id": user_id,
            "addiction_type": addiction_type,
            "days_sober": days_sober,
            "current_phase": self._get_current_phase(addiction_type, days_sober),
            "expected_symptoms": timeline,
            "severity_peak": self._get_severity_peak(addiction_type),
            "recovery_estimate": self._get_recovery_estimate(addiction_type),
            "generated_at": datetime.now().isoformat()
        }
    
    def track_symptom_pattern(
        self,
        user_id: str,
        symptoms: List[Dict]
    ) -> Dict:
        """
        Analiza patrones de síntomas
        
        Args:
            user_id: ID del usuario
            symptoms: Lista de síntomas registrados
        
        Returns:
            Análisis de patrones
        """
        if not symptoms:
            return {
                "user_id": user_id,
                "pattern_analysis": "insufficient_data"
            }
        
        # Agrupar por tipo de síntoma
        symptom_counts = {}
        severity_distribution = {}
        
        for symptom in symptoms:
            name = symptom.get("symptom_name")
            severity = symptom.get("severity")
            
            symptom_counts[name] = symptom_counts.get(name, 0) + 1
            severity_distribution[severity] = severity_distribution.get(severity, 0) + 1
        
        # Identificar síntomas más comunes
        most_common = sorted(symptom_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            "user_id": user_id,
            "total_symptoms": len(symptoms),
            "most_common_symptoms": [{"name": s[0], "count": s[1]} for s in most_common],
            "severity_distribution": severity_distribution,
            "recommendations": self._generate_symptom_recommendations(most_common, severity_distribution),
            "generated_at": datetime.now().isoformat()
        }
    
    def get_symptom_management_tips(
        self,
        symptom_name: str
    ) -> List[Dict]:
        """
        Obtiene tips para manejar un síntoma específico
        
        Args:
            symptom_name: Nombre del síntoma
        
        Returns:
            Lista de tips
        """
        tips_database = {
            "ansiedad": [
                {
                    "tip": "Respiración profunda",
                    "description": "Practica respiración 4-7-8: inhala 4, mantén 7, exhala 8",
                    "duration": "5 minutos"
                },
                {
                    "tip": "Ejercicio ligero",
                    "description": "Caminar o hacer ejercicio suave puede reducir ansiedad",
                    "duration": "20-30 minutos"
                }
            ],
            "insomnio": [
                {
                    "tip": "Rutina de sueño",
                    "description": "Mantén horarios regulares de sueño",
                    "duration": "Consistente"
                },
                {
                    "tip": "Evitar pantallas",
                    "description": "Evita pantallas 1 hora antes de dormir",
                    "duration": "1 hora antes"
                }
            ],
            "irritabilidad": [
                {
                    "tip": "Técnicas de relajación",
                    "description": "Practica meditación o relajación muscular progresiva",
                    "duration": "10-15 minutos"
                }
            ]
        }
        
        return tips_database.get(symptom_name.lower(), [
            {
                "tip": "Contactar profesional",
                "description": "Si los síntomas son severos, contacta un profesional de la salud",
                "duration": "Inmediato"
            }
        ])
    
    def _get_expected_timeline(self, addiction_type: str, days_sober: int) -> Dict:
        """Obtiene línea de tiempo esperada de síntomas"""
        timelines = {
            "cigarrillos": {
                "peak": (1, 3),  # Días 1-3
                "acute": (4, 14),  # Días 4-14
                "subacute": (15, 30),  # Días 15-30
                "recovery": (31, 90)  # Días 31-90
            },
            "alcohol": {
                "peak": (1, 3),
                "acute": (4, 7),
                "subacute": (8, 14),
                "recovery": (15, 30)
            }
        }
        
        return timelines.get(addiction_type.lower(), {
            "peak": (1, 7),
            "acute": (8, 14),
            "subacute": (15, 30),
            "recovery": (31, 90)
        })
    
    def _get_current_phase(self, addiction_type: str, days_sober: int) -> str:
        """Obtiene fase actual de abstinencia"""
        timeline = self._get_expected_timeline(addiction_type, days_sober)
        
        if days_sober <= timeline["peak"][1]:
            return "peak"
        elif days_sober <= timeline["acute"][1]:
            return "acute"
        elif days_sober <= timeline["subacute"][1]:
            return "subacute"
        else:
            return "recovery"
    
    def _get_severity_peak(self, addiction_type: str) -> Dict:
        """Obtiene pico de severidad esperado"""
        return {
            "days": 2,
            "description": "Los síntomas suelen ser más intensos en los primeros días"
        }
    
    def _get_recovery_estimate(self, addiction_type: str) -> Dict:
        """Obtiene estimación de recuperación"""
        estimates = {
            "cigarrillos": {
                "physical": "2-4 semanas",
                "psychological": "3-6 meses"
            },
            "alcohol": {
                "physical": "1-2 semanas",
                "psychological": "3-6 meses"
            }
        }
        
        return estimates.get(addiction_type.lower(), {
            "physical": "1-4 semanas",
            "psychological": "3-6 meses"
        })
    
    def _generate_symptom_recommendations(
        self,
        most_common: List[tuple],
        severity_distribution: Dict
    ) -> List[str]:
        """Genera recomendaciones basadas en síntomas"""
        recommendations = []
        
        if severity_distribution.get(SymptomSeverity.SEVERE, 0) > 0:
            recommendations.append("⚠️ Se detectaron síntomas severos. Considera contactar un profesional de la salud.")
        
        if most_common:
            top_symptom = most_common[0][0]
            recommendations.append(f"Tu síntoma más común es '{top_symptom}'. Revisa los tips de manejo específicos.")
        
        return recommendations
    
    def _load_symptom_database(self) -> Dict:
        """Carga base de datos de síntomas"""
        return {
            "cigarrillos": [
                "ansiedad", "irritabilidad", "dificultad para concentrarse",
                "aumento del apetito", "insomnio", "deseos intensos"
            ],
            "alcohol": [
                "ansiedad", "temblores", "sudoración", "náuseas",
                "insomnio", "irritabilidad", "confusión"
            ],
            "drogas": [
                "ansiedad", "depresión", "fatiga", "insomnio",
                "dolores musculares", "deseos intensos"
            ]
        }

