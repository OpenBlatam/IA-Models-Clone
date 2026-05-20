"""
Funciones auxiliares para el sistema de recuperación
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional
import math


def calculate_days_sober(last_use_date: Optional[datetime], current_date: Optional[datetime] = None) -> int:
    """
    Calcula los días de sobriedad desde la última fecha de uso
    
    Args:
        last_use_date: Fecha de último uso
        current_date: Fecha actual (opcional, por defecto ahora)
    
    Returns:
        Número de días de sobriedad
    """
    if not last_use_date:
        return 0
    
    if current_date is None:
        current_date = datetime.now()
    
    delta = current_date - last_use_date
    return max(0, delta.days)


def calculate_money_saved(days_sober: int, daily_cost: float) -> float:
    """
    Calcula el dinero ahorrado al dejar la adicción
    
    Args:
        days_sober: Días de sobriedad
        daily_cost: Costo diario de la adicción
    
    Returns:
        Dinero ahorrado
    """
    return days_sober * daily_cost


def calculate_health_improvements(days_sober: int, addiction_type: str) -> Dict[str, any]:
    """
    Calcula mejoras de salud basadas en días de sobriedad
    
    Args:
        days_sober: Días de sobriedad
        addiction_type: Tipo de adicción
    
    Returns:
        Diccionario con mejoras de salud
    """
    improvements = {
        "circulation": 0,
        "lung_capacity": 0,
        "heart_rate": 0,
        "energy_level": 0,
        "sleep_quality": 0,
        "anxiety_reduction": 0
    }
    
    if addiction_type.lower() in ["cigarrillos", "tabaco"]:
        improvements["circulation"] = min(100, (days_sober / 20) * 100)
        improvements["lung_capacity"] = min(100, (days_sober / 90) * 100)
        improvements["heart_rate"] = min(100, (days_sober / 14) * 100)
    
    elif addiction_type.lower() == "alcohol":
        improvements["liver_function"] = min(100, (days_sober / 30) * 100)
        improvements["sleep_quality"] = min(100, (days_sober / 7) * 100)
        improvements["energy_level"] = min(100, (days_sober / 14) * 100)
    
    improvements["anxiety_reduction"] = min(100, (days_sober / 30) * 100)
    
    return improvements


def get_milestone_message(days_sober: int) -> Optional[str]:
    """
    Retorna mensaje de celebración para hitos importantes
    
    Args:
        days_sober: Días de sobriedad
    
    Returns:
        Mensaje de celebración o None
    """
    milestones = {
        1: "¡Felicitaciones! Has completado tu primer día. Cada paso cuenta.",
        7: "¡Una semana completa! Estás construyendo un hábito poderoso.",
        30: "¡Un mes de sobriedad! Has demostrado una fuerza increíble.",
        90: "¡Tres meses! Has alcanzado un hito significativo en tu recuperación.",
        180: "¡Seis meses! Tu compromiso es inspirador.",
        365: "¡Un año completo! Has transformado tu vida. ¡Eres increíble!",
    }
    
    return milestones.get(days_sober)


def calculate_relapse_risk_score(
    days_sober: int,
    stress_level: int,
    support_level: int,
    triggers_present: bool,
    previous_relapses: int
) -> float:
    """
    Calcula un score de riesgo de recaída (0-100, donde 100 es máximo riesgo)
    
    Args:
        days_sober: Días de sobriedad
        stress_level: Nivel de estrés (1-10)
        support_level: Nivel de apoyo (1-10)
        triggers_present: Si hay triggers presentes
        previous_relapses: Número de recaídas previas
    
    Returns:
        Score de riesgo (0-100)
    """
    risk = 0.0
    
    # Factor de días de sobriedad (más días = menos riesgo)
    if days_sober < 7:
        risk += 30
    elif days_sober < 30:
        risk += 20
    elif days_sober < 90:
        risk += 10
    
    # Factor de estrés
    risk += (stress_level / 10) * 25
    
    # Factor de apoyo (menos apoyo = más riesgo)
    risk += ((10 - support_level) / 10) * 20
    
    # Factor de triggers
    if triggers_present:
        risk += 15
    
    # Factor de recaídas previas
    risk += min(10, previous_relapses * 2)
    
    return min(100, max(0, risk))


def format_time_sober(days: int) -> str:
    """
    Formatea días de sobriedad en formato legible
    
    Args:
        days: Días de sobriedad
    
    Returns:
        String formateado (ej: "2 meses y 15 días")
    """
    if days < 1:
        return "Menos de un día"
    
    years = days // 365
    months = (days % 365) // 30
    remaining_days = days % 30
    
    parts = []
    if years > 0:
        parts.append(f"{years} {'año' if years == 1 else 'años'}")
    if months > 0:
        parts.append(f"{months} {'mes' if months == 1 else 'meses'}")
    if remaining_days > 0 and years == 0:  # Solo mostrar días si no hay años
        parts.append(f"{remaining_days} {'día' if remaining_days == 1 else 'días'}")
    
    if not parts:
        return f"{days} {'día' if days == 1 else 'días'}"
    
    if len(parts) == 1:
        return parts[0]
    elif len(parts) == 2:
        return f"{parts[0]} y {parts[1]}"
    else:
        return f"{parts[0]}, {parts[1]} y {parts[2]}"

