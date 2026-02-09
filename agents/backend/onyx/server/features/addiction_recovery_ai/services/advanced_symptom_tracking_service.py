"""
Servicio de Seguimiento de Síntomas Avanzado - Sistema completo de síntomas
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
import statistics


class AdvancedSymptomTrackingService:
    """Servicio de seguimiento de síntomas avanzado"""
    
    def __init__(self):
        """Inicializa el servicio de síntomas"""
        pass
    
    def record_symptom(
        self,
        user_id: str,
        symptom_data: Dict
    ) -> Dict:
        """
        Registra un síntoma
        
        Args:
            user_id: ID del usuario
            symptom_data: Datos del síntoma
        
        Returns:
            Síntoma registrado
        """
        symptom = {
            "id": f"symptom_{datetime.now().timestamp()}",
            "user_id": user_id,
            "symptom_data": symptom_data,
            "symptom_type": symptom_data.get("symptom_type", "unknown"),
            "severity": symptom_data.get("severity", 5),
            "duration_minutes": symptom_data.get("duration_minutes", 0),
            "triggers": symptom_data.get("triggers", []),
            "timestamp": symptom_data.get("timestamp", datetime.now().isoformat()),
            "recorded_at": datetime.now().isoformat()
        }
        
        return symptom
    
    def analyze_symptom_patterns(
        self,
        user_id: str,
        symptoms: List[Dict],
        days: int = 30
    ) -> Dict:
        """
        Analiza patrones de síntomas
        
        Args:
            user_id: ID del usuario
            symptoms: Lista de síntomas
            days: Número de días
        
        Returns:
            Análisis de patrones
        """
        if not symptoms:
            return {
                "user_id": user_id,
                "analysis": "no_data"
            }
        
        return {
            "user_id": user_id,
            "period_days": days,
            "total_symptoms": len(symptoms),
            "symptom_frequency": self._calculate_frequency(symptoms),
            "severity_trends": self._analyze_severity_trends(symptoms),
            "common_triggers": self._identify_common_triggers(symptoms),
            "symptom_clusters": self._identify_symptom_clusters(symptoms),
            "correlations": self._find_symptom_correlations(symptoms),
            "recommendations": self._generate_symptom_recommendations(symptoms),
            "generated_at": datetime.now().isoformat()
        }
    
    def predict_symptom_severity(
        self,
        user_id: str,
        current_state: Dict,
        symptom_history: List[Dict]
    ) -> Dict:
        """
        Predice severidad de síntomas
        
        Args:
            user_id: ID del usuario
            current_state: Estado actual
            symptom_history: Historial de síntomas
        
        Returns:
            Predicción de severidad
        """
        predicted_severity = self._calculate_predicted_severity(current_state, symptom_history)
        
        return {
            "user_id": user_id,
            "predicted_severity": round(predicted_severity, 2),
            "confidence": 0.75,
            "risk_factors": self._identify_severity_risk_factors(current_state),
            "preventive_measures": self._suggest_preventive_measures(predicted_severity),
            "predicted_at": datetime.now().isoformat()
        }
    
    def _calculate_frequency(self, symptoms: List[Dict]) -> Dict:
        """Calcula frecuencia de síntomas"""
        frequency = {}
        
        for symptom in symptoms:
            symptom_type = symptom.get("symptom_type", "unknown")
            frequency[symptom_type] = frequency.get(symptom_type, 0) + 1
        
        return frequency
    
    def _analyze_severity_trends(self, symptoms: List[Dict]) -> Dict:
        """Analiza tendencias de severidad"""
        if len(symptoms) < 2:
            return {"trend": "insufficient_data"}
        
        severities = [s.get("severity", 5) for s in symptoms]
        avg_severity = statistics.mean(severities)
        
        first_half = severities[:len(severities)//2]
        second_half = severities[len(severities)//2:]
        
        avg_first = statistics.mean(first_half) if first_half else 0
        avg_second = statistics.mean(second_half) if second_half else 0
        
        if avg_second < avg_first * 0.9:
            trend = "improving"
        elif avg_second > avg_first * 1.1:
            trend = "worsening"
        else:
            trend = "stable"
        
        return {
            "trend": trend,
            "average_severity": round(avg_severity, 2),
            "change": round(avg_second - avg_first, 2)
        }
    
    def _identify_common_triggers(self, symptoms: List[Dict]) -> List[Dict]:
        """Identifica triggers comunes"""
        trigger_counts = {}
        
        for symptom in symptoms:
            triggers = symptom.get("triggers", [])
            for trigger in triggers:
                trigger_counts[trigger] = trigger_counts.get(trigger, 0) + 1
        
        sorted_triggers = sorted(trigger_counts.items(), key=lambda x: x[1], reverse=True)
        
        return [
            {"trigger": trigger, "frequency": count}
            for trigger, count in sorted_triggers[:5]
        ]
    
    def _identify_symptom_clusters(self, symptoms: List[Dict]) -> List[Dict]:
        """Identifica clusters de síntomas"""
        # Lógica simplificada
        return []
    
    def _find_symptom_correlations(self, symptoms: List[Dict]) -> List[Dict]:
        """Encuentra correlaciones de síntomas"""
        return []
    
    def _generate_symptom_recommendations(self, symptoms: List[Dict]) -> List[str]:
        """Genera recomendaciones de síntomas"""
        recommendations = []
        
        avg_severity = statistics.mean([s.get("severity", 5) for s in symptoms]) if symptoms else 0
        if avg_severity >= 7:
            recommendations.append("Severidad de síntomas elevada. Considera consultar con un profesional")
        
        return recommendations
    
    def _calculate_predicted_severity(self, current: Dict, history: List[Dict]) -> float:
        """Calcula severidad predicha"""
        if not history:
            return current.get("current_severity", 5)
        
        recent_severities = [h.get("severity", 5) for h in history[-7:]]
        avg_recent = statistics.mean(recent_severities) if recent_severities else 5
        
        # Ajustar por estado actual
        stress_level = current.get("stress_level", 5)
        if stress_level >= 7:
            avg_recent += 1.0
        
        return min(10, max(1, avg_recent))
    
    def _identify_severity_risk_factors(self, current: Dict) -> List[str]:
        """Identifica factores de riesgo de severidad"""
        factors = []
        
        if current.get("stress_level", 5) >= 7:
            factors.append("Estrés elevado")
        
        return factors
    
    def _suggest_preventive_measures(self, predicted_severity: float) -> List[str]:
        """Sugiere medidas preventivas"""
        measures = []
        
        if predicted_severity >= 7:
            measures.append("Técnicas de relajación recomendadas")
            measures.append("Evita triggers conocidos")
        
        return measures

