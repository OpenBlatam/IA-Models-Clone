"""
Servicio de Seguimiento de Recaídas Avanzado - Sistema completo de seguimiento de recaídas
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import statistics


class AdvancedRelapseTrackingService:
    """Servicio de seguimiento de recaídas avanzado"""
    
    def __init__(self):
        """Inicializa el servicio de recaídas"""
        pass
    
    def record_relapse(
        self,
        user_id: str,
        relapse_data: Dict
    ) -> Dict:
        """
        Registra recaída
        
        Args:
            user_id: ID del usuario
            relapse_data: Datos de recaída
        
        Returns:
            Recaída registrada
        """
        relapse = {
            "id": f"relapse_{datetime.now().timestamp()}",
            "user_id": user_id,
            "relapse_data": relapse_data,
            "relapse_type": relapse_data.get("relapse_type", "full"),
            "severity": relapse_data.get("severity", "moderate"),
            "triggers": relapse_data.get("triggers", []),
            "context": relapse_data.get("context", {}),
            "days_sober_before": relapse_data.get("days_sober_before", 0),
            "recorded_at": datetime.now().isoformat()
        }
        
        return relapse
    
    def analyze_relapse_patterns(
        self,
        user_id: str,
        relapses: List[Dict]
    ) -> Dict:
        """
        Analiza patrones de recaída
        
        Args:
            user_id: ID del usuario
            relapses: Lista de recaídas
        
        Returns:
            Análisis de patrones
        """
        if not relapses:
            return {
                "user_id": user_id,
                "analysis": "no_relapses"
            }
        
        return {
            "user_id": user_id,
            "total_relapses": len(relapses),
            "relapse_frequency": self._calculate_frequency(relapses),
            "common_triggers": self._identify_common_triggers(relapses),
            "temporal_patterns": self._analyze_temporal_patterns(relapses),
            "severity_distribution": self._analyze_severity(relapses),
            "recovery_periods": self._analyze_recovery_periods(relapses),
            "recommendations": self._generate_relapse_recommendations(relapses),
            "generated_at": datetime.now().isoformat()
        }
    
    def predict_relapse_risk(
        self,
        user_id: str,
        current_state: Dict,
        relapse_history: List[Dict]
    ) -> Dict:
        """
        Predice riesgo de recaída
        
        Args:
            user_id: ID del usuario
            current_state: Estado actual
            relapse_history: Historial de recaídas
        
        Returns:
            Predicción de riesgo
        """
        risk_score = self._calculate_risk_score(current_state, relapse_history)
        
        return {
            "user_id": user_id,
            "risk_score": round(risk_score, 3),
            "risk_level": "high" if risk_score >= 0.7 else "medium" if risk_score >= 0.4 else "low",
            "risk_factors": self._identify_risk_factors(current_state, relapse_history),
            "prevention_strategies": self._generate_prevention_strategies(risk_score, current_state),
            "predicted_at": datetime.now().isoformat()
        }
    
    def _calculate_frequency(self, relapses: List[Dict]) -> float:
        """Calcula frecuencia de recaídas"""
        if not relapses:
            return 0.0
        
        # Calcular recaídas por mes
        if len(relapses) >= 2:
            first_date = datetime.fromisoformat(relapses[0].get("recorded_at", datetime.now().isoformat()))
            last_date = datetime.fromisoformat(relapses[-1].get("recorded_at", datetime.now().isoformat()))
            days_diff = (last_date - first_date).days
            months = days_diff / 30 if days_diff > 0 else 1
            return round(len(relapses) / months, 2)
        
        return len(relapses)
    
    def _identify_common_triggers(self, relapses: List[Dict]) -> List[Dict]:
        """Identifica triggers comunes"""
        trigger_counts = defaultdict(int)
        
        for relapse in relapses:
            triggers = relapse.get("triggers", [])
            for trigger in triggers:
                trigger_counts[trigger] += 1
        
        sorted_triggers = sorted(trigger_counts.items(), key=lambda x: x[1], reverse=True)
        
        return [
            {"trigger": trigger, "frequency": count}
            for trigger, count in sorted_triggers[:5]
        ]
    
    def _analyze_temporal_patterns(self, relapses: List[Dict]) -> Dict:
        """Analiza patrones temporales"""
        hourly_counts = defaultdict(int)
        day_counts = defaultdict(int)
        
        for relapse in relapses:
            timestamp = relapse.get("recorded_at")
            if timestamp:
                try:
                    dt = datetime.fromisoformat(timestamp)
                    hourly_counts[dt.hour] += 1
                    day_counts[dt.strftime("%A")] += 1
                except:
                    pass
        
        return {
            "peak_hours": sorted(hourly_counts.items(), key=lambda x: x[1], reverse=True)[:3],
            "peak_days": sorted(day_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        }
    
    def _analyze_severity(self, relapses: List[Dict]) -> Dict:
        """Analiza severidad"""
        severity_counts = defaultdict(int)
        
        for relapse in relapses:
            severity = relapse.get("severity", "moderate")
            severity_counts[severity] += 1
        
        return dict(severity_counts)
    
    def _analyze_recovery_periods(self, relapses: List[Dict]) -> Dict:
        """Analiza períodos de recuperación"""
        if len(relapses) < 2:
            return {"analysis": "insufficient_data"}
        
        recovery_periods = []
        
        for i in range(len(relapses) - 1):
            current = datetime.fromisoformat(relapses[i].get("recorded_at", datetime.now().isoformat()))
            next_relapse = datetime.fromisoformat(relapses[i+1].get("recorded_at", datetime.now().isoformat()))
            days = (next_relapse - current).days
            recovery_periods.append(days)
        
        if recovery_periods:
            return {
                "average_days": round(statistics.mean(recovery_periods), 2),
                "min_days": min(recovery_periods),
                "max_days": max(recovery_periods),
                "trend": "improving" if len(recovery_periods) >= 2 and recovery_periods[-1] > recovery_periods[0] else "stable"
            }
        
        return {}
    
    def _generate_relapse_recommendations(self, relapses: List[Dict]) -> List[str]:
        """Genera recomendaciones de recaída"""
        recommendations = []
        
        common_triggers = self._identify_common_triggers(relapses)
        if common_triggers:
            top_trigger = common_triggers[0].get("trigger", "")
            recommendations.append(f"Desarrolla estrategias para manejar el trigger: {top_trigger}")
        
        return recommendations
    
    def _calculate_risk_score(self, current: Dict, history: List[Dict]) -> float:
        """Calcula puntuación de riesgo"""
        base_risk = 0.3
        
        stress_level = current.get("stress_level", 5)
        if stress_level >= 7:
            base_risk += 0.2
        
        support_level = current.get("support_level", 5)
        if support_level < 4:
            base_risk += 0.2
        
        if history:
            recent_relapses = [r for r in history if self._is_recent(r)]
            if len(recent_relapses) >= 2:
                base_risk += 0.2
        
        return min(1.0, base_risk)
    
    def _is_recent(self, relapse: Dict) -> bool:
        """Verifica si recaída es reciente"""
        timestamp = relapse.get("recorded_at")
        if timestamp:
            try:
                dt = datetime.fromisoformat(timestamp)
                days_ago = (datetime.now() - dt).days
                return days_ago <= 30
            except:
                pass
        return False
    
    def _identify_risk_factors(self, current: Dict, history: List[Dict]) -> List[str]:
        """Identifica factores de riesgo"""
        factors = []
        
        if current.get("stress_level", 5) >= 7:
            factors.append("Estrés elevado")
        
        if current.get("support_level", 5) < 4:
            factors.append("Bajo apoyo social")
        
        return factors
    
    def _generate_prevention_strategies(self, risk_score: float, current: Dict) -> List[str]:
        """Genera estrategias de prevención"""
        strategies = []
        
        if risk_score >= 0.7:
            strategies.append("⚠️ Alto riesgo de recaída. Contacta tu sistema de apoyo inmediatamente")
            strategies.append("Evita situaciones de riesgo conocidas")
        elif risk_score >= 0.4:
            strategies.append("Aumenta tu sistema de apoyo")
            strategies.append("Practica técnicas de manejo de estrés")
        
        return strategies

