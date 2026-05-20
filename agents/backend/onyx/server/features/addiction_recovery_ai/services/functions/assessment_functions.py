"""
Pure functions for assessment logic
Functional programming approach - no classes
"""

from typing import Dict, Any
from datetime import datetime

try:
    from utils.cache import cache_result
except ImportError:
    from ...utils.cache import cache_result


@cache_result(ttl=300, key_prefix="assessment")
def calculate_severity_score(assessment_data: Dict[str, Any]) -> float:
    """Calculate severity score from assessment data"""
    base_score = 5.0
    
    severity_map = {
        "low": 2.0,
        "moderate": 5.0,
        "high": 7.5,
        "severe": 9.0
    }
    
    severity = assessment_data.get("severity", "moderate").lower()
    base_score = severity_map.get(severity, 5.0)
    
    # Adjust based on frequency
    frequency_map = {
        "daily": 1.5,
        "weekly": 1.0,
        "monthly": 0.5,
        "occasional": 0.3
    }
    
    frequency = assessment_data.get("frequency", "weekly").lower()
    frequency_multiplier = frequency_map.get(frequency, 1.0)
    
    # Adjust based on duration
    duration_years = assessment_data.get("duration_years", 0) or 0
    duration_adjustment = min(duration_years * 0.1, 1.0)
    
    final_score = base_score * frequency_multiplier + duration_adjustment
    return min(10.0, max(0.0, final_score))


def determine_risk_level(severity_score: float) -> str:
    """Determine risk level from severity score"""
    if severity_score >= 8.0:
        return "critical"
    if severity_score >= 6.0:
        return "high"
    if severity_score >= 4.0:
        return "moderate"
    return "low"


def generate_recommendations(
    addiction_type: str,
    severity_score: float,
    risk_level: str
) -> list[str]:
    """Generate recommendations based on assessment"""
    recommendations = []
    
    if risk_level in ["high", "critical"]:
        recommendations.append("Busca ayuda profesional inmediatamente")
        recommendations.append("Considera un programa de tratamiento residencial")
    
    if severity_score >= 7.0:
        recommendations.append("Consulta con un médico para evaluación médica")
    
    recommendations.append("Únete a un grupo de apoyo")
    recommendations.append("Crea un plan de recuperación personalizado")
    
    if addiction_type.lower() in ["alcohol", "drugs"]:
        recommendations.append("Considera desintoxicación médica supervisada")
    
    return recommendations


def generate_next_steps(risk_level: str) -> list[str]:
    """Generate next steps based on risk level"""
    steps = []
    
    if risk_level == "critical":
        steps.append("Contacta línea de crisis inmediatamente")
        steps.append("Busca atención médica de emergencia si es necesario")
    
    steps.append("Completa tu perfil de usuario")
    steps.append("Crea tu plan de recuperación")
    steps.append("Configura tu sistema de apoyo")
    
    return steps


def create_assessment_response(
    user_id: str | None,
    assessment_data: Dict[str, Any],
    severity_score: float,
    risk_level: str
) -> Dict[str, Any]:
    """Create assessment response data structure"""
    return {
        "user_id": user_id,
        "assessment_id": f"assess_{datetime.now().timestamp()}",
        "addiction_type": assessment_data.get("addiction_type", ""),
        "severity_score": severity_score,
        "risk_level": risk_level,
        "recommendations": generate_recommendations(
            assessment_data.get("addiction_type", ""),
            severity_score,
            risk_level
        ),
        "next_steps": generate_next_steps(risk_level),
        "assessed_at": datetime.now().isoformat()
    }
