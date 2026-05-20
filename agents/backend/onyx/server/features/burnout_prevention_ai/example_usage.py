"""
Example Usage of Burnout Prevention AI
======================================
Ejemplos de cómo usar la API de prevención de burnout.
"""

import asyncio
import os
from infrastructure.openrouter import OpenRouterClient
from services import BurnoutPreventionService
from schemas import (
    BurnoutAssessmentRequest,
    WellnessCheckRequest,
    CopingStrategyRequest,
    ChatRequest
)


async def example_assess_burnout():
    """Ejemplo de evaluación de burnout."""
    client = OpenRouterClient()
    service = BurnoutPreventionService(client)
    
    request = BurnoutAssessmentRequest(
        work_hours_per_week=55,
        stress_level=8,
        sleep_hours_per_night=5.5,
        work_satisfaction=4,
        physical_symptoms=["fatiga", "dolores de cabeza", "tensión muscular"],
        emotional_symptoms=["ansiedad", "irritabilidad", "desmotivación"],
        work_environment="Alta presión, múltiples proyectos simultáneos, plazos ajustados",
        additional_context="Trabajo remoto, dificultad para desconectar"
    )
    
    try:
        result = await service.assess_burnout(request)
        print("=== Evaluación de Burnout ===")
        print(f"Nivel de riesgo: {result.burnout_risk_level}")
        print(f"Puntuación: {result.burnout_score}/100")
        print(f"\nFactores de riesgo:")
        for factor in result.risk_factors:
            print(f"  - {factor}")
        print(f"\nRecomendaciones:")
        for rec in result.recommendations:
            print(f"  - {rec}")
        print(f"\nAcciones inmediatas:")
        for action in result.immediate_actions:
            print(f"  - {action}")
    finally:
        await client.close()


async def example_wellness_check():
    """Ejemplo de chequeo de bienestar."""
    client = OpenRouterClient()
    service = BurnoutPreventionService(client)
    
    request = WellnessCheckRequest(
        current_mood="ansioso y agotado",
        energy_level=3,
        recent_challenges="Proyecto importante con plazo muy ajustado, conflictos en el equipo",
        support_system="Familia cercana y algunos colegas de confianza"
    )
    
    try:
        result = await service.wellness_check(request)
        print("\n=== Chequeo de Bienestar ===")
        print(f"Puntuación de bienestar: {result.wellness_score}/100")
        print(f"\nAnálisis del estado de ánimo:")
        print(f"  {result.mood_analysis}")
        print(f"\nRecomendaciones de apoyo:")
        for rec in result.support_recommendations:
            print(f"  - {rec}")
        print(f"\nSugerencias de autocuidado:")
        for suggestion in result.self_care_suggestions:
            print(f"  - {suggestion}")
    finally:
        await client.close()


async def example_coping_strategies():
    """Ejemplo de estrategias de afrontamiento."""
    client = OpenRouterClient()
    service = BurnoutPreventionService(client)
    
    request = CopingStrategyRequest(
        stressor_type="Sobrecarga de trabajo y falta de límites",
        current_coping_methods=["trabajar más horas", "ignorar el problema"],
        available_time="30 minutos diarios",
        preferences=["ejercicio", "meditación", "lectura"]
    )
    
    try:
        result = await service.get_coping_strategies(request)
        print("\n=== Estrategias de Afrontamiento ===")
        print(f"\nEstrategias recomendadas:")
        for strategy in result.strategies:
            if isinstance(strategy, dict):
                print(f"  - {strategy.get('name', 'Estrategia')}: {strategy.get('description', '')}")
            else:
                print(f"  - {strategy}")
        print(f"\nPlan de implementación:")
        for step in result.implementation_plan:
            print(f"  - {step}")
        print(f"\nRecursos adicionales:")
        for resource in result.resources:
            print(f"  - {resource}")
    finally:
        await client.close()


async def example_chat():
    """Ejemplo de chat conversacional."""
    client = OpenRouterClient()
    service = BurnoutPreventionService(client)
    
    request = ChatRequest(
        message="Me siento muy agotado últimamente y tengo dificultad para concentrarme. ¿Qué puedo hacer?",
        conversation_history=[]
    )
    
    try:
        result = await service.chat(request)
        print("\n=== Chat con Asistente ===")
        print(f"\nRespuesta:")
        print(f"  {result.response}")
        if result.suggestions:
            print(f"\nSugerencias de seguimiento:")
            for suggestion in result.suggestions:
                print(f"  - {suggestion}")
        if result.resources:
            print(f"\nRecursos:")
            for resource in result.resources:
                print(f"  - {resource}")
    finally:
        await client.close()


async def main():
    """Ejecutar todos los ejemplos."""
    print("Burnout Prevention AI - Ejemplos de Uso\n")
    print("=" * 50)
    
    # Verificar API key
    if not os.getenv("OPENROUTER_API_KEY"):
        print("⚠️  ADVERTENCIA: OPENROUTER_API_KEY no está configurada")
        print("   Configura la variable de entorno antes de ejecutar los ejemplos\n")
        return
    
    try:
        await example_assess_burnout()
        await example_wellness_check()
        await example_coping_strategies()
        await example_chat()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())

