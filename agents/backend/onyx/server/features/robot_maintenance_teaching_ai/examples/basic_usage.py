"""
Ejemplo básico de uso del Robot Maintenance Teaching AI.
"""

import asyncio
import os
from robot_maintenance_teaching_ai import RobotMaintenanceTutor, MaintenanceConfig

async def main():
    if not os.getenv("OPENROUTER_API_KEY"):
        print("⚠️  ADVERTENCIA: OPENROUTER_API_KEY no está configurada")
        print("Por favor, configura tu API key:")
        print("export OPENROUTER_API_KEY='tu-api-key-aqui'")
        return
    
    config = MaintenanceConfig()
    tutor = RobotMaintenanceTutor(config)
    
    print("🤖 Robot Maintenance Teaching AI - Ejemplo de Uso\n")
    print("=" * 60)
    
    print("\n1. Enseñar procedimiento de mantenimiento:")
    print("-" * 60)
    result = await tutor.teach_maintenance_procedure(
        robot_type="industrial_robot",
        maintenance_type="preventive",
        difficulty="intermediate"
    )
    print(result["content"][:500] + "...")
    
    print("\n2. Diagnosticar problema:")
    print("-" * 60)
    diagnosis = await tutor.diagnose_problem(
        symptoms="El robot hace ruidos extraños y se mueve de forma errática",
        robot_type="industrial_robot"
    )
    print(diagnosis["content"][:500] + "...")
    
    print("\n3. Explicar componente:")
    print("-" * 60)
    explanation = await tutor.explain_component(
        component_name="reductor de velocidad",
        robot_type="industrial_robot",
        difficulty="intermediate"
    )
    print(explanation["content"][:500] + "...")
    
    print("\n4. Generar programa de mantenimiento:")
    print("-" * 60)
    schedule = await tutor.generate_maintenance_schedule(
        robot_type="industrial_robot",
        usage_hours=8,
        environment="industrial"
    )
    print(schedule["content"][:500] + "...")
    
    print("\n5. Responder pregunta:")
    print("-" * 60)
    answer = await tutor.answer_question(
        question="¿Con qué frecuencia debo lubricar las juntas del robot?",
        robot_type="industrial_robot"
    )
    print(answer["content"][:500] + "...")
    
    await tutor.close()
    print("\n✅ Ejemplo completado!")

if __name__ == "__main__":
    asyncio.run(main())






