"""
Basic usage examples for Robot Maintenance AI.
"""

import asyncio
import os
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from robot_maintenance_ai import RobotMaintenanceTutor, MaintenanceConfig


async def example_ask_question():
    """Example: Asking a maintenance question."""
    config = MaintenanceConfig()
    tutor = RobotMaintenanceTutor(config)
    
    print("=== Ejemplo: Pregunta de Mantenimiento ===\n")
    
    result = await tutor.ask_maintenance_question(
        question="¿Cómo cambio el aceite de un robot industrial?",
        robot_type="robots_industriales",
        maintenance_type="lubricacion"
    )
    
    print("Respuesta:")
    print(result['answer'])
    print(f"\nAnálisis NLP: {result.get('nlp_analysis', {})}")
    print(f"Predicción ML: {result.get('ml_prediction', {})}")
    
    await tutor.close()


async def example_diagnose_problem():
    """Example: Diagnosing a problem."""
    config = MaintenanceConfig()
    tutor = RobotMaintenanceTutor(config)
    
    print("\n=== Ejemplo: Diagnosticar Problema ===\n")
    
    result = await tutor.diagnose_problem(
        symptoms="El robot hace ruidos extraños y se mueve de forma errática. La temperatura está elevada.",
        robot_type="robots_industriales",
        sensor_data={
            "temperature": 85.0,
            "vibration": 6.5,
            "current": 12.3
        }
    )
    
    print("Diagnóstico:")
    print(result['answer'])
    print(f"\nAnálisis NLP: {result.get('nlp_analysis', {})}")
    print(f"Predicción ML: {result.get('ml_prediction', {})}")
    
    await tutor.close()


async def example_explain_procedure():
    """Example: Explaining a maintenance procedure."""
    config = MaintenanceConfig()
    tutor = RobotMaintenanceTutor(config)
    
    print("\n=== Ejemplo: Explicar Procedimiento ===\n")
    
    result = await tutor.explain_maintenance_procedure(
        procedure="calibración de encoders",
        robot_type="robots_industriales",
        difficulty="avanzado"
    )
    
    print("Explicación:")
    print(result['answer'])
    
    await tutor.close()


async def example_predict_maintenance():
    """Example: Predicting maintenance schedule."""
    config = MaintenanceConfig()
    tutor = RobotMaintenanceTutor(config)
    
    print("\n=== Ejemplo: Predecir Programa de Mantenimiento ===\n")
    
    sensor_data = {
        "temperature": 28.5,
        "vibration": 0.15,
        "pressure": 4.2,
        "runtime_hours": 8500,
        "battery_level": 75.0
    }
    
    result = await tutor.predict_maintenance_schedule(
        robot_type="robots_industriales",
        sensor_data=sensor_data,
        historical_data=None
    )
    
    print("Predicción:")
    print(result['answer'])
    print(f"\nPredicción ML: {result.get('ml_prediction', {})}")
    
    await tutor.close()


async def example_generate_checklist():
    """Example: Generating a maintenance checklist."""
    config = MaintenanceConfig()
    tutor = RobotMaintenanceTutor(config)
    
    print("\n=== Ejemplo: Generar Checklist de Mantenimiento ===\n")
    
    result = await tutor.generate_maintenance_checklist(
        robot_type="robots_industriales",
        maintenance_type="preventivo"
    )
    
    print("Checklist:")
    print(result['answer'])
    
    await tutor.close()


async def main():
    """Run all examples."""
    if not os.getenv("OPENROUTER_API_KEY"):
        print("ERROR: OPENROUTER_API_KEY environment variable not set!")
        print("Please set it with: export OPENROUTER_API_KEY='your-key-here'")
        return
    
    await example_ask_question()
    await example_diagnose_problem()
    await example_explain_procedure()
    await example_predict_maintenance()
    await example_generate_checklist()


if __name__ == "__main__":
    asyncio.run(main())
