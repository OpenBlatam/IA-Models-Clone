"""
Servicio de Seguimiento de Sueño Avanzado con Wearables - Sistema completo de seguimiento de sueño
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
import statistics


class AdvancedSleepTrackingService:
    """Servicio de seguimiento de sueño avanzado"""
    
    def __init__(self):
        """Inicializa el servicio de seguimiento de sueño"""
        pass
    
    def record_sleep_data(
        self,
        user_id: str,
        sleep_start: str,
        sleep_end: str,
        sleep_stages: Optional[Dict] = None,
        wearable_data: Optional[Dict] = None
    ) -> Dict:
        """
        Registra datos de sueño
        
        Args:
            user_id: ID del usuario
            sleep_start: Hora de inicio de sueño
            sleep_end: Hora de fin de sueño
            sleep_stages: Etapas de sueño (opcional)
            wearable_data: Datos del wearable (opcional)
        
        Returns:
            Datos de sueño registrados
        """
        start = datetime.fromisoformat(sleep_start)
        end = datetime.fromisoformat(sleep_end)
        duration_hours = (end - start).total_seconds() / 3600
        
        sleep_record = {
            "id": f"sleep_{datetime.now().timestamp()}",
            "user_id": user_id,
            "sleep_start": sleep_start,
            "sleep_end": sleep_end,
            "duration_hours": round(duration_hours, 2),
            "sleep_stages": sleep_stages or {},
            "wearable_data": wearable_data,
            "quality_score": self._calculate_sleep_quality(duration_hours, sleep_stages),
            "recorded_at": datetime.now().isoformat()
        }
        
        return sleep_record
    
    def analyze_sleep_patterns_advanced(
        self,
        user_id: str,
        sleep_data: List[Dict],
        days: int = 30
    ) -> Dict:
        """
        Analiza patrones de sueño avanzado
        
        Args:
            user_id: ID del usuario
            sleep_data: Datos de sueño
            days: Número de días a analizar
        
        Returns:
            Análisis avanzado de patrones
        """
        if not sleep_data or len(sleep_data) < 7:
            return {
                "user_id": user_id,
                "analysis": "insufficient_data"
            }
        
        durations = [s.get("duration_hours", 0) for s in sleep_data]
        quality_scores = [s.get("quality_score", 5) for s in sleep_data]
        
        analysis = {
            "user_id": user_id,
            "period_days": len(sleep_data),
            "sleep_statistics": {
                "average_duration": round(statistics.mean(durations), 2),
                "min_duration": min(durations),
                "max_duration": max(durations),
                "consistency": self._calculate_sleep_consistency(durations),
                "average_quality": round(statistics.mean(quality_scores), 2)
            },
            "sleep_stages_analysis": self._analyze_sleep_stages(sleep_data),
            "correlations": self._find_sleep_correlations(sleep_data),
            "recommendations": self._generate_sleep_recommendations(sleep_data),
            "generated_at": datetime.now().isoformat()
        }
        
        return analysis
    
    def predict_sleep_quality(
        self,
        user_id: str,
        current_factors: Dict
    ) -> Dict:
        """
        Predice calidad de sueño
        
        Args:
            user_id: ID del usuario
            current_factors: Factores actuales
        
        Returns:
            Predicción de calidad de sueño
        """
        exercise_today = current_factors.get("exercise_today", False)
        stress_level = current_factors.get("stress_level", 5)
        caffeine_consumed = current_factors.get("caffeine_consumed", False)
        
        predicted_quality = 7.0
        
        if exercise_today:
            predicted_quality += 0.5
        if stress_level >= 8:
            predicted_quality -= 1.5
        elif stress_level >= 6:
            predicted_quality -= 0.5
        if caffeine_consumed:
            predicted_quality -= 0.5
        
        predicted_quality = max(1, min(10, predicted_quality))
        
        return {
            "user_id": user_id,
            "predicted_quality": round(predicted_quality, 2),
            "confidence": 0.75,
            "factors_considered": list(current_factors.keys()),
            "recommendations": self._generate_sleep_optimization_tips(current_factors),
            "predicted_at": datetime.now().isoformat()
        }
    
    def sync_wearable_sleep_data(
        self,
        user_id: str,
        device_id: str,
        sleep_data: Dict
    ) -> Dict:
        """
        Sincroniza datos de sueño desde wearable
        
        Args:
            user_id: ID del usuario
            device_id: ID del dispositivo
            sleep_data: Datos de sueño del wearable
        
        Returns:
            Resultado de sincronización
        """
        return {
            "user_id": user_id,
            "device_id": device_id,
            "sleep_data": sleep_data,
            "synced_at": datetime.now().isoformat(),
            "status": "success"
        }
    
    def _calculate_sleep_quality(self, duration: float, stages: Optional[Dict]) -> float:
        """Calcula calidad de sueño"""
        # Calidad basada en duración
        if 7 <= duration <= 9:
            quality = 8.0
        elif 6 <= duration < 7 or 9 < duration <= 10:
            quality = 6.5
        else:
            quality = 5.0
        
        # Ajustar por etapas de sueño si están disponibles
        if stages:
            deep_sleep_ratio = stages.get("deep_sleep_hours", 0) / duration if duration > 0 else 0
            if deep_sleep_ratio >= 0.2:
                quality += 1.0
        
        return round(max(1, min(10, quality)), 2)
    
    def _calculate_sleep_consistency(self, durations: List[float]) -> float:
        """Calcula consistencia de sueño"""
        if len(durations) < 2:
            return 0.0
        
        std_dev = statistics.stdev(durations) if len(durations) > 1 else 0
        avg = statistics.mean(durations)
        
        consistency = max(0, 100 - (std_dev / avg * 100)) if avg > 0 else 0
        return round(consistency, 2)
    
    def _analyze_sleep_stages(self, sleep_data: List[Dict]) -> Dict:
        """Analiza etapas de sueño"""
        return {
            "average_deep_sleep": 0.0,
            "average_rem_sleep": 0.0,
            "average_light_sleep": 0.0,
            "stage_distribution": {}
        }
    
    def _find_sleep_correlations(self, sleep_data: List[Dict]) -> List[Dict]:
        """Encuentra correlaciones de sueño"""
        return []
    
    def _generate_sleep_recommendations(self, sleep_data: List[Dict]) -> List[str]:
        """Genera recomendaciones de sueño"""
        recommendations = []
        
        durations = [s.get("duration_hours", 0) for s in sleep_data]
        avg_duration = statistics.mean(durations) if durations else 0
        
        if avg_duration < 6:
            recommendations.append("Tu promedio de sueño está por debajo de lo recomendado. Intenta dormir más.")
        
        return recommendations
    
    def _generate_sleep_optimization_tips(self, factors: Dict) -> List[str]:
        """Genera tips para optimizar sueño"""
        tips = []
        
        if factors.get("stress_level", 5) >= 7:
            tips.append("Practica técnicas de relajación antes de dormir")
        
        if factors.get("caffeine_consumed", False):
            tips.append("Evita cafeína al menos 6 horas antes de dormir")
        
        return tips

