"""
Servicio de análisis avanzado - Proporciona insights profundos
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
from collections import Counter
import statistics


class AnalyticsService:
    """Servicio de análisis avanzado"""
    
    def __init__(self):
        """Inicializa el servicio de análisis"""
        pass
    
    def generate_comprehensive_analytics(
        self,
        user_id: str,
        entries: List[Dict],
        start_date: Optional[datetime] = None
    ) -> Dict:
        """
        Genera análisis completo del usuario
        
        Args:
            user_id: ID del usuario
            entries: Lista de entradas diarias
            start_date: Fecha de inicio (opcional)
        
        Returns:
            Análisis completo
        """
        if not entries:
            return {
                "user_id": user_id,
                "message": "No hay datos suficientes para análisis",
                "status": "insufficient_data"
            }
        
        # Análisis de tendencias
        trends = self._analyze_trends(entries)
        
        # Análisis de patrones
        patterns = self._analyze_patterns(entries)
        
        # Análisis de éxito
        success_metrics = self._calculate_success_metrics(entries)
        
        # Predicciones
        predictions = self._generate_predictions(entries)
        
        # Recomendaciones basadas en datos
        recommendations = self._generate_data_driven_recommendations(entries, trends, patterns)
        
        analytics = {
            "user_id": user_id,
            "generated_at": datetime.now().isoformat(),
            "period": {
                "start": entries[0].get("date") if entries else None,
                "end": entries[-1].get("date") if entries else None,
                "total_days": len(entries)
            },
            "trends": trends,
            "patterns": patterns,
            "success_metrics": success_metrics,
            "predictions": predictions,
            "recommendations": recommendations,
            "status": "success"
        }
        
        return analytics
    
    def _analyze_trends(self, entries: List[Dict]) -> Dict:
        """Analiza tendencias en los datos"""
        if len(entries) < 7:
            return {"insufficient_data": True}
        
        # Dividir en períodos
        mid_point = len(entries) // 2
        first_half = entries[:mid_point]
        second_half = entries[mid_point:]
        
        # Calcular tasas de consumo
        first_rate = sum(1 for e in first_half if e.get("consumed", False)) / len(first_half)
        second_rate = sum(1 for e in second_half if e.get("consumed", False)) / len(second_half)
        
        # Calcular cravings promedio
        first_cravings = [e.get("cravings_level", 0) for e in first_half]
        second_cravings = [e.get("cravings_level", 0) for e in second_half]
        
        avg_first = statistics.mean(first_cravings) if first_cravings else 0
        avg_second = statistics.mean(second_cravings) if second_cravings else 0
        
        return {
            "consumption_trend": "mejorando" if second_rate < first_rate else "empeorando" if second_rate > first_rate else "estable",
            "consumption_rate_change": second_rate - first_rate,
            "cravings_trend": "disminuyendo" if avg_second < avg_first else "aumentando" if avg_second > avg_first else "estable",
            "cravings_change": avg_second - avg_first
        }
    
    def _analyze_patterns(self, entries: List[Dict]) -> Dict:
        """Analiza patrones en los datos"""
        # Patrones por día de la semana
        day_patterns = {}
        days = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
        
        for day in days:
            day_entries = [e for e in entries if self._get_day_of_week(e.get("date", "")) == day]
            if day_entries:
                day_patterns[day] = {
                    "consumption_rate": sum(1 for e in day_entries if e.get("consumed", False)) / len(day_entries),
                    "avg_cravings": statistics.mean([e.get("cravings_level", 0) for e in day_entries])
                }
        
        # Patrones de triggers
        all_triggers = []
        for entry in entries:
            all_triggers.extend(entry.get("triggers_encountered", []))
        
        trigger_frequency = Counter(all_triggers)
        most_common_triggers = trigger_frequency.most_common(5)
        
        # Patrones de estado de ánimo
        mood_frequency = Counter([e.get("mood", "neutral") for e in entries])
        
        return {
            "day_of_week_patterns": day_patterns,
            "most_common_triggers": [{"trigger": t[0], "count": t[1]} for t in most_common_triggers],
            "mood_distribution": dict(mood_frequency),
            "high_risk_days": [day for day, data in day_patterns.items() if data.get("consumption_rate", 0) > 0.3]
        }
    
    def _calculate_success_metrics(self, entries: List[Dict]) -> Dict:
        """Calcula métricas de éxito"""
        total_days = len(entries)
        sober_days = sum(1 for e in entries if not e.get("consumed", False))
        success_rate = (sober_days / total_days * 100) if total_days > 0 else 0
        
        # Racha más larga
        longest_streak = 0
        current_streak = 0
        for entry in entries:
            if not entry.get("consumed", False):
                current_streak += 1
                longest_streak = max(longest_streak, current_streak)
            else:
                current_streak = 0
        
        return {
            "success_rate": round(success_rate, 2),
            "sober_days": sober_days,
            "total_days": total_days,
            "longest_streak": longest_streak,
            "current_streak": current_streak
        }
    
    def _generate_predictions(self, entries: List[Dict]) -> Dict:
        """Genera predicciones basadas en datos"""
        if len(entries) < 14:
            return {"insufficient_data": True}
        
        # Predicción de éxito a 30 días
        recent_entries = entries[-7:] if len(entries) >= 7 else entries
        recent_success_rate = sum(1 for e in recent_entries if not e.get("consumed", False)) / len(recent_entries)
        
        predicted_30_day_success = recent_success_rate * 100
        
        return {
            "predicted_30_day_success_rate": round(predicted_30_day_success, 2),
            "confidence": "alta" if len(entries) >= 30 else "media" if len(entries) >= 14 else "baja",
            "based_on_recent_performance": True
        }
    
    def _generate_data_driven_recommendations(
        self,
        entries: List[Dict],
        trends: Dict,
        patterns: Dict
    ) -> List[str]:
        """Genera recomendaciones basadas en análisis de datos"""
        recommendations = []
        
        # Recomendaciones basadas en tendencias
        if trends.get("consumption_trend") == "empeorando":
            recommendations.append("⚠️ Tu tasa de consumo ha aumentado. Considera aumentar contacto con tu sistema de apoyo.")
        
        if trends.get("cravings_trend") == "aumentando":
            recommendations.append("Tus cravings están aumentando. Practica técnicas de afrontamiento más frecuentemente.")
        
        # Recomendaciones basadas en patrones
        high_risk_days = patterns.get("high_risk_days", [])
        if high_risk_days:
            recommendations.append(f"Los {', '.join(high_risk_days)} son días de mayor riesgo. Prepara estrategias específicas para estos días.")
        
        # Recomendaciones basadas en triggers
        most_common = patterns.get("most_common_triggers", [])
        if most_common:
            top_trigger = most_common[0].get("trigger", "")
            recommendations.append(f"Tu trigger más común es '{top_trigger}'. Desarrolla estrategias específicas para manejarlo.")
        
        if not recommendations:
            recommendations.append("Continúa con tu plan actual. Estás haciendo un buen progreso.")
        
        return recommendations
    
    def _get_day_of_week(self, date_str: str) -> str:
        """Obtiene día de la semana de una fecha"""
        try:
            date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            days = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
            return days[date.weekday()]
        except:
            return "desconocido"

