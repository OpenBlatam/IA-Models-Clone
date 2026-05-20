"""
Servicio de Seguimiento de Salud Física - Monitoreo de salud durante recuperación
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
from utils.helpers import calculate_health_improvements


class HealthTrackingService:
    """Servicio de seguimiento de salud física"""
    
    def __init__(self):
        """Inicializa el servicio de seguimiento de salud"""
        self.health_metrics = self._load_health_metrics()
    
    def record_health_metric(
        self,
        user_id: str,
        metric_type: str,
        value: float,
        unit: str,
        notes: Optional[str] = None
    ) -> Dict:
        """
        Registra una métrica de salud
        
        Args:
            user_id: ID del usuario
            metric_type: Tipo de métrica (heart_rate, blood_pressure, sleep_hours, etc.)
            value: Valor de la métrica
            unit: Unidad de medida
            notes: Notas adicionales
        
        Returns:
            Métrica registrada
        """
        metric = {
            "user_id": user_id,
            "metric_type": metric_type,
            "value": value,
            "unit": unit,
            "notes": notes,
            "recorded_at": datetime.now().isoformat()
        }
        
        return metric
    
    def get_health_summary(
        self,
        user_id: str,
        days_sober: int,
        addiction_type: str,
        health_data: Optional[List[Dict]] = None
    ) -> Dict:
        """
        Obtiene resumen de salud del usuario
        
        Args:
            user_id: ID del usuario
            days_sober: Días de sobriedad
            addiction_type: Tipo de adicción
            health_data: Datos de salud históricos (opcional)
        
        Returns:
            Resumen de salud
        """
        # Calcular mejoras de salud basadas en días de sobriedad
        improvements = calculate_health_improvements(days_sober, addiction_type)
        
        # Análisis de tendencias si hay datos
        trends = {}
        if health_data:
            trends = self._analyze_health_trends(health_data)
        
        # Recomendaciones de salud
        recommendations = self._generate_health_recommendations(
            days_sober,
            addiction_type,
            improvements
        )
        
        return {
            "user_id": user_id,
            "days_sober": days_sober,
            "addiction_type": addiction_type,
            "health_improvements": improvements,
            "trends": trends,
            "recommendations": recommendations,
            "generated_at": datetime.now().isoformat()
        }
    
    def track_sleep(
        self,
        user_id: str,
        hours: float,
        quality: str,
        notes: Optional[str] = None
    ) -> Dict:
        """
        Registra datos de sueño
        
        Args:
            user_id: ID del usuario
            hours: Horas de sueño
            quality: Calidad (excelente, buena, regular, mala)
            notes: Notas adicionales
        
        Returns:
            Registro de sueño
        """
        return {
            "user_id": user_id,
            "metric_type": "sleep",
            "hours": hours,
            "quality": quality,
            "notes": notes,
            "recorded_at": datetime.now().isoformat()
        }
    
    def track_exercise(
        self,
        user_id: str,
        activity_type: str,
        duration_minutes: int,
        intensity: str,
        calories: Optional[int] = None
    ) -> Dict:
        """
        Registra actividad física
        
        Args:
            user_id: ID del usuario
            activity_type: Tipo de actividad
            duration_minutes: Duración en minutos
            intensity: Intensidad (baja, media, alta)
            calories: Calorías quemadas (opcional)
        
        Returns:
            Registro de ejercicio
        """
        return {
            "user_id": user_id,
            "metric_type": "exercise",
            "activity_type": activity_type,
            "duration_minutes": duration_minutes,
            "intensity": intensity,
            "calories": calories,
            "recorded_at": datetime.now().isoformat()
        }
    
    def get_health_goals(
        self,
        user_id: str,
        addiction_type: str
    ) -> List[Dict]:
        """
        Obtiene metas de salud recomendadas
        
        Args:
            user_id: ID del usuario
            addiction_type: Tipo de adicción
        
        Returns:
            Lista de metas de salud
        """
        goals = []
        
        if addiction_type.lower() in ["cigarrillos", "tabaco"]:
            goals.extend([
                {
                    "goal": "Mejorar capacidad pulmonar",
                    "target": "30 días",
                    "description": "La capacidad pulmonar mejora significativamente después de 30 días"
                },
                {
                    "goal": "Reducir frecuencia cardíaca",
                    "target": "14 días",
                    "description": "La frecuencia cardíaca comienza a normalizarse"
                }
            ])
        
        if addiction_type.lower() == "alcohol":
            goals.extend([
                {
                    "goal": "Mejorar función hepática",
                    "target": "30 días",
                    "description": "El hígado comienza a recuperarse después de 30 días"
                },
                {
                    "goal": "Mejorar calidad de sueño",
                    "target": "7 días",
                    "description": "La calidad del sueño mejora rápidamente"
                }
            ])
        
        # Metas generales
        goals.extend([
            {
                "goal": "Aumentar energía",
                "target": "14 días",
                "description": "Los niveles de energía aumentan significativamente"
            },
            {
                "goal": "Mejorar estado de ánimo",
                "target": "30 días",
                "description": "El estado de ánimo se estabiliza"
            }
        ])
        
        return goals
    
    def _analyze_health_trends(self, health_data: List[Dict]) -> Dict:
        """Analiza tendencias de salud"""
        if not health_data:
            return {}
        
        # Agrupar por tipo de métrica
        metrics_by_type = {}
        for entry in health_data:
            metric_type = entry.get("metric_type")
            if metric_type not in metrics_by_type:
                metrics_by_type[metric_type] = []
            metrics_by_type[metric_type].append(entry)
        
        trends = {}
        for metric_type, entries in metrics_by_type.items():
            if len(entries) >= 2:
                values = [e.get("value", 0) for e in entries]
                trends[metric_type] = {
                    "trend": "improving" if values[-1] > values[0] else "declining" if values[-1] < values[0] else "stable",
                    "current_value": values[-1],
                    "average": sum(values) / len(values)
                }
        
        return trends
    
    def _generate_health_recommendations(
        self,
        days_sober: int,
        addiction_type: str,
        improvements: Dict
    ) -> List[str]:
        """Genera recomendaciones de salud"""
        recommendations = []
        
        if days_sober < 7:
            recommendations.append("Los primeros días son críticos. Descansa y mantén hidratación adecuada.")
        
        if days_sober >= 7 and days_sober < 30:
            recommendations.append("Considera comenzar ejercicio ligero para mejorar tu salud física.")
            recommendations.append("Mantén una dieta balanceada para apoyar tu recuperación.")
        
        if days_sober >= 30:
            recommendations.append("Continúa con ejercicio regular y dieta saludable.")
            recommendations.append("Considera hacerte un chequeo médico para evaluar mejoras.")
        
        if addiction_type.lower() in ["cigarrillos", "tabaco"]:
            recommendations.append("Tu capacidad pulmonar está mejorando. Considera ejercicios de respiración.")
        
        return recommendations
    
    def _load_health_metrics(self) -> Dict:
        """Carga métricas de salud disponibles"""
        return {
            "heart_rate": {"unit": "bpm", "normal_range": (60, 100)},
            "blood_pressure": {"unit": "mmHg", "normal_range": (90, 140)},
            "sleep_hours": {"unit": "hours", "normal_range": (7, 9)},
            "weight": {"unit": "kg", "normal_range": None},
            "energy_level": {"unit": "1-10", "normal_range": (7, 10)}
        }

