"""
Pure functions for relapse prevention logic
Functional programming approach - no classes
"""

from typing import Dict, Any, List

try:
    from utils.cache import cache_result
except ImportError:
    from ...utils.cache import cache_result


def calculate_relapse_risk_score(
    days_sober: int,
    stress_level: int,
    support_level: int,
    risk_factors: List[str]
) -> float:
    """
    Calculate relapse risk score (0-100)
    
    Args:
        days_sober: Days since last use
        stress_level: Current stress level (0-10)
        support_level: Current support level (0-10)
        risk_factors: List of identified risk factors
    
    Returns:
        Risk score from 0 to 100
    """
    base_risk = 50.0
    
    # Adjust based on days sober (more days = lower risk)
    days_factor = max(0, 30 - days_sober) / 30.0
    base_risk += days_factor * 20.0
    
    # Adjust based on stress level
    stress_factor = (stress_level / 10.0) * 20.0
    base_risk += stress_factor
    
    # Adjust based on support level (more support = lower risk)
    support_factor = (10 - support_level) / 10.0 * 15.0
    base_risk += support_factor
    
    # Adjust based on risk factors
    risk_factor_count = len(risk_factors)
    base_risk += risk_factor_count * 5.0
    
    return min(100.0, max(0.0, base_risk))


def determine_risk_level(risk_score: float) -> str:
    """Determine risk level from risk score"""
    if risk_score >= 75:
        return "critical"
    if risk_score >= 50:
        return "high"
    if risk_score >= 25:
        return "moderate"
    return "low"


def identify_risk_factors(
    stress_level: int,
    support_level: int,
    isolation: bool,
    negative_thinking: bool,
    romanticizing: bool,
    skipping_support: bool
) -> List[str]:
    """Identify risk factors from current state"""
    factors = []
    
    if stress_level >= 7:
        factors.append("Estrés elevado")
    
    if support_level < 4:
        factors.append("Bajo apoyo social")
    
    if isolation:
        factors.append("Aislamiento social")
    
    if negative_thinking:
        factors.append("Pensamientos negativos")
    
    if romanticizing:
        factors.append("Idealización del consumo")
    
    if skipping_support:
        factors.append("Evitando apoyo")
    
    return factors


def identify_protective_factors(
    support_level: int,
    coping_skills: int,
    motivation: int,
    has_plan: bool
) -> List[str]:
    """Identify protective factors"""
    factors = []
    
    if support_level >= 7:
        factors.append("Alto apoyo social")
    
    if coping_skills >= 7:
        factors.append("Buenas habilidades de afrontamiento")
    
    if motivation >= 7:
        factors.append("Alta motivación")
    
    if has_plan:
        factors.append("Plan de recuperación activo")
    
    return factors


def generate_risk_recommendations(
    risk_level: str,
    risk_factors: List[str]
) -> List[str]:
    """Generate recommendations based on risk level and factors"""
    recommendations = []
    
    if risk_level == "critical":
        recommendations.append("Contacta línea de crisis inmediatamente")
        recommendations.append("Busca apoyo profesional urgente")
        recommendations.append("Activa tu plan de emergencia")
    
    if risk_level in ["high", "critical"]:
        recommendations.append("Contacta tu sistema de apoyo ahora")
        recommendations.append("Usa técnicas de afrontamiento inmediatas")
    
    if "Estrés elevado" in risk_factors:
        recommendations.append("Practica técnicas de relajación")
        recommendations.append("Considera ejercicio o meditación")
    
    if "Bajo apoyo social" in risk_factors:
        recommendations.append("Contacta a alguien de tu red de apoyo")
        recommendations.append("Considera unirse a un grupo de apoyo")
    
    if "Aislamiento social" in risk_factors:
        recommendations.append("Evita el aislamiento - contacta a alguien")
        recommendations.append("Participa en actividades sociales")
    
    recommendations.append("Revisa tu plan de recuperación")
    recommendations.append("Recuerda tus razones para estar sobrio")
    
    return recommendations


@cache_result(ttl=60, key_prefix="relapse_risk")
def calculate_relapse_risk(
    days_sober: int,
    stress_level: int,
    support_level: int,
    risk_factors: List[str]
) -> Dict[str, Any]:
    """
    Calculate comprehensive relapse risk assessment
    
    Returns:
        Dictionary with risk assessment data
    """
    risk_score = calculate_relapse_risk_score(
        days_sober, stress_level, support_level, risk_factors
    )
    risk_level = determine_risk_level(risk_score)
    
    return {
        "risk_score": risk_score,
        "risk_level": risk_level,
        "risk_factors": risk_factors,
        "recommendations": generate_risk_recommendations(risk_level, risk_factors)
    }

