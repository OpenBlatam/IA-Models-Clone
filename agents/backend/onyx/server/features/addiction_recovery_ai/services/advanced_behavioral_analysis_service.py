"""
Servicio de Análisis de Comportamiento Avanzado - Sistema completo de análisis conductual
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import statistics


class AdvancedBehavioralAnalysisService:
    """Servicio de análisis de comportamiento avanzado"""
    
    def __init__(self):
        """Inicializa el servicio de análisis conductual"""
        pass
    
    def analyze_behavioral_patterns(
        self,
        user_id: str,
        behavioral_data: List[Dict],
        analysis_type: str = "comprehensive"
    ) -> Dict:
        """
        Analiza patrones de comportamiento
        
        Args:
            user_id: ID del usuario
            behavioral_data: Datos de comportamiento
            analysis_type: Tipo de análisis
        
        Returns:
            Análisis de patrones
        """
        if not behavioral_data:
            return {
                "user_id": user_id,
                "analysis": "no_data"
            }
        
        return {
            "user_id": user_id,
            "analysis_type": analysis_type,
            "total_records": len(behavioral_data),
            "behavioral_clusters": self._identify_behavioral_clusters(behavioral_data),
            "pattern_frequency": self._analyze_pattern_frequency(behavioral_data),
            "behavioral_trends": self._calculate_behavioral_trends(behavioral_data),
            "risk_behaviors": self._identify_risk_behaviors(behavioral_data),
            "positive_behaviors": self._identify_positive_behaviors(behavioral_data),
            "recommendations": self._generate_behavioral_recommendations(behavioral_data),
            "generated_at": datetime.now().isoformat()
        }
    
    def detect_behavioral_anomalies(
        self,
        user_id: str,
        current_behavior: Dict,
        historical_patterns: List[Dict]
    ) -> Dict:
        """
        Detecta anomalías de comportamiento
        
        Args:
            user_id: ID del usuario
            current_behavior: Comportamiento actual
            historical_patterns: Patrones históricos
        
        Returns:
            Anomalías detectadas
        """
        anomalies = []
        
        # Comparar con patrones históricos
        for pattern in historical_patterns:
            deviation = self._calculate_deviation(current_behavior, pattern)
            if deviation > 0.3:  # Umbral de anomalía
                anomalies.append({
                    "type": "behavioral_deviation",
                    "severity": "medium" if deviation > 0.5 else "low",
                    "description": f"Desviación detectada en {pattern.get('behavior_type')}",
                    "deviation_score": round(deviation, 3)
                })
        
        return {
            "user_id": user_id,
            "anomalies_detected": anomalies,
            "total_anomalies": len(anomalies),
            "risk_level": self._determine_anomaly_risk_level(anomalies),
            "recommendations": self._generate_anomaly_recommendations(anomalies),
            "detected_at": datetime.now().isoformat()
        }
    
    def predict_behavioral_outcome(
        self,
        user_id: str,
        current_behaviors: Dict,
        behavioral_history: List[Dict]
    ) -> Dict:
        """
        Predice resultado conductual
        
        Args:
            user_id: ID del usuario
            current_behaviors: Comportamientos actuales
            behavioral_history: Historial conductual
        
        Returns:
            Predicción de resultado
        """
        outcome_probability = self._calculate_outcome_probability(current_behaviors, behavioral_history)
        
        return {
            "user_id": user_id,
            "predicted_outcome": "positive" if outcome_probability > 0.6 else "neutral" if outcome_probability > 0.4 else "negative",
            "outcome_probability": round(outcome_probability, 3),
            "confidence": 0.75,
            "key_factors": self._identify_outcome_factors(current_behaviors),
            "predicted_at": datetime.now().isoformat()
        }
    
    def _identify_behavioral_clusters(self, data: List[Dict]) -> List[Dict]:
        """Identifica clusters de comportamiento"""
        clusters = defaultdict(list)
        
        for record in data:
            behavior_type = record.get("behavior_type", "unknown")
            clusters[behavior_type].append(record)
        
        return [
            {
                "cluster_type": cluster_type,
                "count": len(records),
                "average_frequency": len(records) / 30 if len(records) > 0 else 0
            }
            for cluster_type, records in clusters.items()
        ]
    
    def _analyze_pattern_frequency(self, data: List[Dict]) -> Dict:
        """Analiza frecuencia de patrones"""
        frequency = defaultdict(int)
        
        for record in data:
            pattern = record.get("pattern", "unknown")
            frequency[pattern] += 1
        
        return dict(frequency)
    
    def _calculate_behavioral_trends(self, data: List[Dict]) -> Dict:
        """Calcula tendencias conductuales"""
        if len(data) < 7:
            return {"trend": "insufficient_data"}
        
        # Dividir en dos mitades
        first_half = data[:len(data)//2]
        second_half = data[len(data)//2:]
        
        positive_count_1 = sum(1 for d in first_half if d.get("is_positive", False))
        positive_count_2 = sum(1 for d in second_half if d.get("is_positive", False))
        
        ratio_1 = positive_count_1 / len(first_half) if first_half else 0
        ratio_2 = positive_count_2 / len(second_half) if second_half else 0
        
        if ratio_2 > ratio_1 * 1.1:
            trend = "improving"
        elif ratio_2 < ratio_1 * 0.9:
            trend = "declining"
        else:
            trend = "stable"
        
        return {
            "trend": trend,
            "change_percentage": round(((ratio_2 - ratio_1) / ratio_1 * 100) if ratio_1 > 0 else 0, 2)
        }
    
    def _identify_risk_behaviors(self, data: List[Dict]) -> List[Dict]:
        """Identifica comportamientos de riesgo"""
        risk_behaviors = []
        
        for record in data:
            if record.get("is_risk_behavior", False):
                risk_behaviors.append({
                    "behavior": record.get("behavior_type"),
                    "severity": record.get("severity", "medium"),
                    "timestamp": record.get("timestamp")
                })
        
        return risk_behaviors
    
    def _identify_positive_behaviors(self, data: List[Dict]) -> List[Dict]:
        """Identifica comportamientos positivos"""
        positive_behaviors = []
        
        for record in data:
            if record.get("is_positive", False):
                positive_behaviors.append({
                    "behavior": record.get("behavior_type"),
                    "impact": record.get("impact", "medium"),
                    "timestamp": record.get("timestamp")
                })
        
        return positive_behaviors
    
    def _generate_behavioral_recommendations(self, data: List[Dict]) -> List[str]:
        """Genera recomendaciones conductuales"""
        recommendations = []
        
        risk_behaviors = self._identify_risk_behaviors(data)
        if len(risk_behaviors) > 5:
            recommendations.append("Se detectaron múltiples comportamientos de riesgo. Considera intervención")
        
        return recommendations
    
    def _calculate_deviation(self, current: Dict, pattern: Dict) -> float:
        """Calcula desviación de patrón"""
        # Lógica simplificada
        return 0.2
    
    def _determine_anomaly_risk_level(self, anomalies: List[Dict]) -> str:
        """Determina nivel de riesgo de anomalías"""
        if not anomalies:
            return "low"
        
        high_severity_count = sum(1 for a in anomalies if a.get("severity") == "high")
        
        if high_severity_count >= 2:
            return "high"
        elif high_severity_count >= 1 or len(anomalies) >= 3:
            return "medium"
        else:
            return "low"
    
    def _generate_anomaly_recommendations(self, anomalies: List[Dict]) -> List[str]:
        """Genera recomendaciones basadas en anomalías"""
        if anomalies:
            return [
                "Se detectaron anomalías en el comportamiento. Monitoreo cercano recomendado"
            ]
        return []
    
    def _calculate_outcome_probability(self, current: Dict, history: List[Dict]) -> float:
        """Calcula probabilidad de resultado"""
        base_probability = 0.5
        
        positive_behaviors = sum(1 for h in history if h.get("is_positive", False))
        if positive_behaviors > len(history) * 0.7:
            base_probability += 0.2
        
        return min(1.0, base_probability)
    
    def _identify_outcome_factors(self, behaviors: Dict) -> List[str]:
        """Identifica factores de resultado"""
        factors = []
        
        if behaviors.get("engagement_level", 5) >= 7:
            factors.append("Alto nivel de engagement")
        
        return factors

