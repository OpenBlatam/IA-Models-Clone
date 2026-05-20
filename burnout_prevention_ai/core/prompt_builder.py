"""
Prompt Builder Utilities
========================
Helper functions for building prompts consistently.

Centralizes prompt construction logic to ensure consistency
and improve maintainability across all AI interactions.
"""

from typing import List, Optional
from .types import MessageList
from .constants import MAX_MESSAGE_LENGTH


def build_system_user_messages(
    system_prompt: str,
    user_prompt: str,
    conversation_history: Optional[MessageList] = None
) -> MessageList:
    """
    Build messages list for API calls.
    
    Args:
        system_prompt: System prompt content (validated to be non-empty)
        user_prompt: User prompt content (validated to be non-empty)
        conversation_history: Optional conversation history
        
    Returns:
        List of message dictionaries
        
    Raises:
        ValueError: If prompts are empty
    """
    if not system_prompt or not system_prompt.strip():
        raise ValueError("System prompt cannot be empty")
    
    if not user_prompt or not user_prompt.strip():
        raise ValueError("User prompt cannot be empty")
    
    # Validate and truncate if necessary
    system_content = system_prompt.strip()
    user_content = user_prompt.strip()
    
    # Truncate if too long (with warning)
    if len(user_content) > MAX_MESSAGE_LENGTH:
        from .logging_helpers import log_warning
        log_warning(
            "User prompt truncated",
            context={"original_length": len(user_content), "max_length": MAX_MESSAGE_LENGTH}
        )
        user_content = user_content[:MAX_MESSAGE_LENGTH]
    
    messages = [{"role": "system", "content": system_content}]
    
    if conversation_history:
        messages.extend(conversation_history)
    
    messages.append({"role": "user", "content": user_content})
    return messages


def format_list_items(items: Optional[List[str]], default: str = "Ninguno") -> str:
    """
    Format list of items as comma-separated string.
    
    Args:
        items: List of items to format (can be None)
        default: Default value if list is empty or None
        
    Returns:
        Formatted string
    """
    if not items:
        return default
    # Filter out empty strings for cleaner output
    filtered = [item for item in items if item and item.strip()]
    if not filtered:
        return default
    return ', '.join(filtered)


def format_optional_field(value: Optional[str], default: str = "No especificado") -> str:
    """
    Format optional field with default.
    
    Args:
        value: Optional value
        default: Default if value is None or empty
        
    Returns:
        Formatted string
    """
    return value if value and value.strip() else default


def build_assessment_prompt(
    work_hours: int,
    stress_level: int,
    sleep_hours: float,
    work_satisfaction: int,
    physical_symptoms: List[str],
    emotional_symptoms: List[str],
    work_environment: Optional[str] = None,
    additional_context: Optional[str] = None
) -> str:
    """
    Build assessment prompt efficiently.
    
    Args:
        work_hours: Work hours per week
        stress_level: Stress level (1-10)
        sleep_hours: Sleep hours per night
        work_satisfaction: Work satisfaction (1-10)
        physical_symptoms: List of physical symptoms
        emotional_symptoms: List of emotional symptoms
        work_environment: Optional work environment description
        additional_context: Optional additional context
        
    Returns:
        Formatted prompt string
    """
    symptoms_phys = format_list_items(physical_symptoms)
    symptoms_emo = format_list_items(emotional_symptoms)
    work_env = format_optional_field(work_environment)
    add_context = format_optional_field(additional_context, "Ninguno")
    
    return f"""Evalúa el riesgo de burnout basándote en:

Horas trabajadas por semana: {work_hours}
Nivel de estrés (1-10): {stress_level}
Horas de sueño por noche: {sleep_hours}
Satisfacción laboral (1-10): {work_satisfaction}
Síntomas físicos: {symptoms_phys}
Síntomas emocionales: {symptoms_emo}
Ambiente laboral: {work_env}
Contexto adicional: {add_context}

Proporciona una evaluación completa en formato JSON con:
- burnout_risk_level: "low", "medium", "high", o "critical"
- burnout_score: número del 0-100
- risk_factors: lista de factores de riesgo identificados
- recommendations: lista de recomendaciones personalizadas (mínimo 5)
- immediate_actions: lista de acciones inmediatas (mínimo 3)
- long_term_strategies: lista de estrategias a largo plazo (mínimo 3)

Sé específico y práctico en tus recomendaciones."""


def build_wellness_check_prompt(
    current_mood: str,
    energy_level: int,
    recent_challenges: Optional[str] = None,
    support_system: Optional[str] = None
) -> str:
    """
    Build wellness check prompt.
    
    Args:
        current_mood: Current mood description
        energy_level: Energy level (1-10)
        recent_challenges: Optional recent challenges
        support_system: Optional support system description
        
    Returns:
        Formatted prompt string
    """
    challenges = format_optional_field(recent_challenges, 'Ninguno')
    support = format_optional_field(support_system)
    
    return f"""Realiza un chequeo de bienestar basándote en:

Estado de ánimo actual: {current_mood}
Nivel de energía (1-10): {energy_level}
Desafíos recientes: {challenges}
Sistema de apoyo: {support}

Proporciona un análisis en formato JSON con:
- wellness_score: número del 0-100
- mood_analysis: análisis del estado de ánimo
- support_recommendations: lista de recomendaciones de apoyo (mínimo 3)
- self_care_suggestions: lista de sugerencias de autocuidado (mínimo 4)

Sé empático y constructivo."""


def build_coping_strategy_prompt(
    stressor_type: str,
    current_coping_methods: Optional[List[str]] = None,
    available_time: Optional[str] = None,
    preferences: Optional[List[str]] = None
) -> str:
    """
    Build coping strategy prompt.
    
    Args:
        stressor_type: Type of stressor
        current_coping_methods: Optional current coping methods
        available_time: Optional available time
        preferences: Optional preferences
        
    Returns:
        Formatted prompt string
    """
    methods = format_list_items(current_coping_methods, 'Ninguno')
    time_avail = format_optional_field(available_time, 'No especificado')
    prefs = format_list_items(preferences, 'Ninguna')
    
    return f"""Proporciona estrategias de afrontamiento para:

Tipo de estresor: {stressor_type}
Métodos actuales: {methods}
Tiempo disponible: {time_avail}
Preferencias: {prefs}

Proporciona en formato JSON:
- strategies: lista de objetos con {{
    "name": "nombre de la estrategia",
    "description": "descripción",
    "effectiveness": "alta/media/baja",
    "time_required": "tiempo necesario",
    "difficulty": "fácil/media/difícil"
  }}
- implementation_plan: pasos para implementar (mínimo 5)
- resources: recursos adicionales (libros, apps, etc.)

Sé práctico y específico."""


def build_progress_tracking_prompt(
    assessment_count: int,
    goals: Optional[List[str]] = None,
    current_status: Optional[str] = None
) -> str:
    """
    Build progress tracking prompt.
    
    Args:
        assessment_count: Number of assessments in history
        goals: Optional list of goals
        current_status: Optional current status
        
    Returns:
        Formatted prompt string
    """
    goals_str = format_list_items(goals, 'Ninguna')
    status_str = format_optional_field(current_status, 'No especificado')
    
    return f"""Analiza el progreso en la prevención de burnout basándote en:

Historial de evaluaciones: {assessment_count} evaluaciones
Metas establecidas: {goals_str}
Estado actual: {status_str}

Proporciona un análisis en formato JSON con:
- progress_score: número del 0-100 (progreso general)
- trend: "improving", "stable", o "declining"
- milestones_achieved: lista de logros alcanzados
- next_steps: lista de próximos pasos recomendados (mínimo 3)
- insights: análisis detallado del progreso (2-3 párrafos)

Sé específico y alentador."""


def build_trend_analysis_prompt(
    assessment_count: int,
    time_period_days: int
) -> str:
    """
    Build trend analysis prompt.
    
    Args:
        assessment_count: Number of assessments
        time_period_days: Time period in days
        
    Returns:
        Formatted prompt string
    """
    return f"""Analiza las tendencias en {assessment_count} evaluaciones de burnout 
durante los últimos {time_period_days} días.

Proporciona un análisis en formato JSON con:
- overall_trend: "improving", "stable", "declining", o "fluctuating"
- key_metrics: objeto con métricas clave (burnout_score_avg, stress_level_trend, etc.)
- patterns: lista de patrones identificados (mínimo 3)
- predictions: objeto con predicciones futuras (next_week, next_month)
- recommendations: lista de recomendaciones basadas en tendencias (mínimo 4)

Sé analítico y específico."""


def build_resource_prompt(
    topic: str,
    level: str,
    format_preference: Optional[str] = None
) -> str:
    """
    Build resource request prompt.
    
    Args:
        topic: Topic of interest
        level: Level (beginner/intermediate/advanced)
        format_preference: Optional format preference
        
    Returns:
        Formatted prompt string
    """
    format_pref = format_optional_field(format_preference, 'Cualquiera')
    
    return f"""Proporciona recursos educativos sobre burnout para:

Tema: {topic}
Nivel: {level}
Formato preferido: {format_pref}

Proporciona en formato JSON:
- resources: lista de objetos con {{
    "title": "título",
    "type": "article/video/podcast/book",
    "description": "descripción",
    "url": "enlace si está disponible",
    "duration": "duración si aplica"
  }} (mínimo 5 recursos)
- learning_path: ruta de aprendizaje sugerida (pasos ordenados, mínimo 4)
- key_concepts: conceptos clave a entender (mínimo 5)
- action_items: elementos de acción práctica (mínimo 3)

Sé específico y útil."""


def build_personalized_plan_prompt(
    goals: List[str],
    current_situation: Dict[str, Any],
    constraints: Optional[List[str]] = None,
    preferences: Optional[List[str]] = None
) -> str:
    """
    Build personalized plan prompt.
    
    Args:
        goals: List of goals
        current_situation: Current situation dictionary
        constraints: Optional constraints
        preferences: Optional preferences
        
    Returns:
        Formatted prompt string
    """
    goals_str = format_list_items(goals, 'Ninguna')
    constraints_str = format_list_items(constraints, 'Ninguna')
    prefs_str = format_list_items(preferences, 'Ninguna')
    
    # Format current situation as readable text
    situation_str = ', '.join(f"{k}: {v}" for k, v in current_situation.items()) if current_situation else "No especificado"
    
    return f"""Crea un plan personalizado de prevención de burnout con:

Metas: {goals_str}
Situación actual: {situation_str}
Restricciones: {constraints_str}
Preferencias: {prefs_str}

Proporciona en formato JSON:
- plan_structure: estructura del plan (objetivos, fases, timeline)
- weekly_actions: acciones semanales específicas (mínimo 5)
- monthly_milestones: hitos mensuales (mínimo 3)
- success_metrics: métricas de éxito (mínimo 4)
- adjustments: recomendaciones de ajuste según progreso

Sé realista, específico y accionable."""

