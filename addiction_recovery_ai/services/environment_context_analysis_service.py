"""
Servicio de Análisis de Entorno y Contexto - Sistema completo de análisis contextual
"""

from typing import Dict, List, Optional
from datetime import datetime
from enum import Enum


class EnvironmentType(str, Enum):
    """Tipos de entorno"""
    HOME = "home"
    WORK = "work"
    SOCIAL = "social"
    PUBLIC = "public"
    TREATMENT = "treatment"


class EnvironmentContextAnalysisService:
    """Servicio de análisis de entorno y contexto"""
    
    def __init__(self):
        """Inicializa el servicio de análisis contextual"""
        pass
    
    def record_environment_context(
        self,
        user_id: str,
        context_data: Dict
    ) -> Dict:
        """
        Registra contexto de entorno
        
        Args:
            user_id: ID del usuario
            context_data: Datos de contexto
        
        Returns:
            Contexto registrado
        """
        context = {
            "id": f"context_{datetime.now().timestamp()}",
            "user_id": user_id,
            "context_data": context_data,
            "environment_type": context_data.get("environment_type", "unknown"),
            "location": context_data.get("location", {}),
            "time_of_day": context_data.get("time_of_day", "unknown"),
            "people_present": context_data.get("people_present", []),
            "activities": context_data.get("activities", []),
            "timestamp": context_data.get("timestamp", datetime.now().isoformat()),
            "recorded_at": datetime.now().isoformat()
        }
        
        return context
    
    def analyze_environment_patterns(
        self,
        user_id: str,
        context_records: List[Dict]
    ) -> Dict:
        """
        Analiza patrones de entorno
        
        Args:
            user_id: ID del usuario
            context_records: Registros de contexto
        
        Returns:
            Análisis de patrones
        """
        if not context_records:
            return {
                "user_id": user_id,
                "analysis": "no_data"
            }
        
        return {
            "user_id": user_id,
            "total_records": len(context_records),
            "environment_distribution": self._analyze_environment_distribution(context_records),
            "time_patterns": self._analyze_time_patterns(context_records),
            "risk_environments": self._identify_risk_environments(context_records),
            "safe_environments": self._identify_safe_environments(context_records),
            "recommendations": self._generate_environment_recommendations(context_records),
            "generated_at": datetime.now().isoformat()
        }
    
    def predict_environment_risk(
        self,
        user_id: str,
        current_context: Dict
    ) -> Dict:
        """
        Predice riesgo de entorno
        
        Args:
            user_id: ID del usuario
            current_context: Contexto actual
        
        Returns:
            Predicción de riesgo
        """
        risk_score = self._calculate_environment_risk(current_context)
        
        return {
            "user_id": user_id,
            "current_context": current_context,
            "risk_score": risk_score,
            "risk_level": self._determine_risk_level(risk_score),
            "risk_factors": self._identify_risk_factors(current_context),
            "recommendations": self._generate_risk_recommendations(risk_score, current_context),
            "predicted_at": datetime.now().isoformat()
        }
    
    def _analyze_environment_distribution(self, records: List[Dict]) -> Dict:
        """Analiza distribución de entornos"""
        distribution = {}
        
        for record in records:
            env_type = record.get("environment_type", "unknown")
            distribution[env_type] = distribution.get(env_type, 0) + 1
        
        return distribution
    
    def _analyze_time_patterns(self, records: List[Dict]) -> Dict:
        """Analiza patrones temporales"""
        time_distribution = {}
        
        for record in records:
            time_of_day = record.get("time_of_day", "unknown")
            time_distribution[time_of_day] = time_distribution.get(time_of_day, 0) + 1
        
        return time_distribution
    
    def _identify_risk_environments(self, records: List[Dict]) -> List[Dict]:
        """Identifica entornos de riesgo"""
        risk_envs = []
        
        # Lógica simplificada
        for record in records:
            env_type = record.get("environment_type")
            if env_type == EnvironmentType.SOCIAL:
                risk_envs.append({
                    "environment": env_type,
                    "risk_reason": "social_environment"
                })
        
        return risk_envs
    
    def _identify_safe_environments(self, records: List[Dict]) -> List[Dict]:
        """Identifica entornos seguros"""
        safe_envs = []
        
        for record in records:
            env_type = record.get("environment_type")
            if env_type == EnvironmentType.HOME or env_type == EnvironmentType.TREATMENT:
                safe_envs.append({
                    "environment": env_type,
                    "safety_reason": "controlled_environment"
                })
        
        return safe_envs
    
    def _calculate_environment_risk(self, context: Dict) -> float:
        """Calcula puntuación de riesgo de entorno"""
        risk_score = 0.5  # Base
        
        env_type = context.get("environment_type")
        if env_type == EnvironmentType.SOCIAL:
            risk_score += 0.2
        elif env_type == EnvironmentType.PUBLIC:
            risk_score += 0.1
        
        # Ajustar por hora del día
        time_of_day = context.get("time_of_day", "")
        if time_of_day in ["evening", "night"]:
            risk_score += 0.1
        
        return min(1.0, risk_score)
    
    def _determine_risk_level(self, risk_score: float) -> str:
        """Determina nivel de riesgo"""
        if risk_score >= 0.7:
            return "high"
        elif risk_score >= 0.4:
            return "medium"
        else:
            return "low"
    
    def _identify_risk_factors(self, context: Dict) -> List[str]:
        """Identifica factores de riesgo"""
        factors = []
        
        env_type = context.get("environment_type")
        if env_type == EnvironmentType.SOCIAL:
            factors.append("Entorno social")
        
        return factors
    
    def _generate_environment_recommendations(self, records: List[Dict]) -> List[str]:
        """Genera recomendaciones de entorno"""
        recommendations = []
        
        risk_envs = self._identify_risk_environments(records)
        if risk_envs:
            recommendations.append("Considera limitar tiempo en entornos de riesgo")
        
        return recommendations
    
    def _generate_risk_recommendations(self, risk_score: float, context: Dict) -> List[str]:
        """Genera recomendaciones basadas en riesgo"""
        recommendations = []
        
        if risk_score >= 0.7:
            recommendations.append("⚠️ Alto riesgo detectado. Considera cambiar de entorno")
        elif risk_score >= 0.4:
            recommendations.append("Ten precaución en este entorno")
        
        return recommendations

