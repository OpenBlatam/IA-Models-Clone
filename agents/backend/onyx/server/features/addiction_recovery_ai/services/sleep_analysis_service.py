"""
Servicio de Análisis de Sueño - Análisis avanzado de patrones de sueño
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
import statistics


class SleepAnalysisService:
    """Servicio de análisis avanzado de sueño"""
    
    def __init__(self):
        """Inicializa el servicio de análisis de sueño"""
        pass
    
    def analyze_sleep_patterns(
        self,
        user_id: str,
        sleep_data: List[Dict]
    ) -> Dict:
        """
        Analiza patrones de sueño del usuario
        
        Args:
            user_id: ID del usuario
            sleep_data: Lista de registros de sueño
        
        Returns:
            Análisis de patrones de sueño
        """
        if not sleep_data or len(sleep_data) < 7:
            return {
                "user_id": user_id,
                "analysis": "insufficient_data",
                "message": "Se necesitan al menos 7 días de datos"
            }
        
        # Calcular métricas básicas
        hours_list = [entry.get("hours", 0) for entry in sleep_data]
        quality_list = [entry.get("quality", "regular") for entry in sleep_data]
        
        analysis = {
            "user_id": user_id,
            "period_days": len(sleep_data),
            "average_hours": round(statistics.mean(hours_list), 2) if hours_list else 0,
            "min_hours": min(hours_list) if hours_list else 0,
            "max_hours": max(hours_list) if hours_list else 0,
            "quality_distribution": self._analyze_quality_distribution(quality_list),
            "sleep_efficiency": self._calculate_sleep_efficiency(sleep_data),
            "consistency_score": self._calculate_consistency(sleep_data),
            "recommendations": self._generate_sleep_recommendations(sleep_data),
            "generated_at": datetime.now().isoformat()
        }
        
        return analysis
    
    def correlate_sleep_with_recovery(
        self,
        user_id: str,
        sleep_data: List[Dict],
        recovery_data: List[Dict]
    ) -> Dict:
        """
        Correlaciona datos de sueño con datos de recuperación
        
        Args:
            user_id: ID del usuario
            sleep_data: Datos de sueño
            recovery_data: Datos de recuperación
        
        Returns:
            Análisis de correlación
        """
        correlations = []
        
        # Correlación sueño vs cravings
        if sleep_data and recovery_data:
            avg_sleep = statistics.mean([s.get("hours", 0) for s in sleep_data])
            avg_cravings = statistics.mean([r.get("cravings_level", 0) for r in recovery_data])
            
            if avg_sleep >= 7 and avg_cravings < 3:
                correlations.append({
                    "metric_pair": "sleep_vs_cravings",
                    "correlation": "negative",
                    "insight": "Mejor sueño se correlaciona con menores cravings"
                })
        
        # Correlación calidad de sueño vs estado de ánimo
        quality_scores = {"excelente": 4, "buena": 3, "regular": 2, "mala": 1}
        avg_quality = statistics.mean([quality_scores.get(s.get("quality", "regular"), 2) for s in sleep_data])
        
        if avg_quality >= 3:
            correlations.append({
                "metric_pair": "sleep_quality_vs_mood",
                "correlation": "positive",
                "insight": "Mejor calidad de sueño mejora el estado de ánimo"
            })
        
        return {
            "user_id": user_id,
            "correlations": correlations,
            "recommendations": self._generate_correlation_recommendations(correlations),
            "generated_at": datetime.now().isoformat()
        }
    
    def predict_sleep_quality(
        self,
        user_id: str,
        current_factors: Dict
    ) -> Dict:
        """
        Predice calidad de sueño basada en factores actuales
        
        Args:
            user_id: ID del usuario
            current_factors: Factores actuales (ejercicio, estrés, etc.)
        
        Returns:
            Predicción de calidad de sueño
        """
        # Factores que afectan el sueño
        exercise_today = current_factors.get("exercise_today", False)
        stress_level = current_factors.get("stress_level", 5)
        caffeine_consumed = current_factors.get("caffeine_consumed", False)
        
        predicted_quality = "buena"
        confidence = 0.7
        
        if exercise_today:
            predicted_quality = "buena"
            confidence = 0.8
        elif stress_level >= 8:
            predicted_quality = "regular"
            confidence = 0.6
        elif caffeine_consumed:
            predicted_quality = "regular"
            confidence = 0.65
        
        return {
            "user_id": user_id,
            "predicted_quality": predicted_quality,
            "confidence": confidence,
            "factors_considered": list(current_factors.keys()),
            "recommendations": self._generate_sleep_optimization_tips(current_factors),
            "generated_at": datetime.now().isoformat()
        }
    
    def _analyze_quality_distribution(self, quality_list: List[str]) -> Dict:
        """Analiza distribución de calidad de sueño"""
        quality_count = {}
        for quality in quality_list:
            quality_count[quality] = quality_count.get(quality, 0) + 1
        
        return quality_count
    
    def _calculate_sleep_efficiency(self, sleep_data: List[Dict]) -> float:
        """Calcula eficiencia del sueño"""
        # Eficiencia = (horas de sueño / horas en cama) * 100
        # Simplificado: asumimos 8 horas en cama
        if not sleep_data:
            return 0.0
        
        total_hours = sum(entry.get("hours", 0) for entry in sleep_data)
        total_bed_hours = len(sleep_data) * 8  # Asumiendo 8 horas en cama
        
        efficiency = (total_hours / total_bed_hours * 100) if total_bed_hours > 0 else 0
        return round(efficiency, 2)
    
    def _calculate_consistency(self, sleep_data: List[Dict]) -> float:
        """Calcula consistencia de horarios de sueño"""
        if len(sleep_data) < 2:
            return 0.0
        
        # Calcular variabilidad en horas de sueño
        hours_list = [entry.get("hours", 0) for entry in sleep_data]
        if not hours_list:
            return 0.0
        
        std_dev = statistics.stdev(hours_list) if len(hours_list) > 1 else 0
        avg_hours = statistics.mean(hours_list)
        
        # Menor desviación = mayor consistencia
        consistency = max(0, 100 - (std_dev / avg_hours * 100)) if avg_hours > 0 else 0
        return round(consistency, 2)
    
    def _generate_sleep_recommendations(self, sleep_data: List[Dict]) -> List[str]:
        """Genera recomendaciones basadas en análisis de sueño"""
        recommendations = []
        
        hours_list = [entry.get("hours", 0) for entry in sleep_data]
        avg_hours = statistics.mean(hours_list) if hours_list else 0
        
        if avg_hours < 6:
            recommendations.append("⚠️ Duermes menos de 6 horas en promedio. Esto puede afectar tu recuperación.")
            recommendations.append("Intenta establecer una rutina de sueño más consistente.")
        elif avg_hours < 7:
            recommendations.append("Tu promedio de sueño está por debajo de lo recomendado (7-9 horas).")
        
        quality_list = [entry.get("quality", "regular") for entry in sleep_data]
        poor_quality_count = sum(1 for q in quality_list if q in ["mala", "regular"])
        
        if poor_quality_count > len(quality_list) * 0.5:
            recommendations.append("Se detectó baja calidad de sueño. Considera técnicas de higiene del sueño.")
        
        return recommendations
    
    def _generate_correlation_recommendations(self, correlations: List[Dict]) -> List[str]:
        """Genera recomendaciones basadas en correlaciones"""
        recommendations = []
        
        for corr in correlations:
            if corr.get("metric_pair") == "sleep_vs_cravings":
                recommendations.append("Prioriza 7-8 horas de sueño para reducir cravings")
        
        return recommendations
    
    def _generate_sleep_optimization_tips(self, factors: Dict) -> List[str]:
        """Genera tips para optimizar sueño"""
        tips = []
        
        if factors.get("stress_level", 5) >= 7:
            tips.append("Practica técnicas de relajación antes de dormir")
        
        if factors.get("caffeine_consumed", False):
            tips.append("Evita cafeína al menos 6 horas antes de dormir")
        
        if not factors.get("exercise_today", False):
            tips.append("El ejercicio regular mejora la calidad del sueño")
        
        if not tips:
            tips.append("Mantén una rutina de sueño consistente")
        
        return tips

