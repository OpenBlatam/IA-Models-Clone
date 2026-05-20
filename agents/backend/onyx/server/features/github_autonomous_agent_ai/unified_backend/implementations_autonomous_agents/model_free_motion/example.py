"""
Ejemplo de uso: Model-free Motion Planning of Autonomous Vehicles
==================================================================
"""

from model_free_motion import ModelFreeMotionPlanner, MotionType
from datetime import datetime

# Crear planificador de movimiento sin modelo
planner = ModelFreeMotionPlanner(
    name="motion_planner_01",
    config={
        "lookahead_distance": 50.0,
        "safety_margin": 5.0,
        "replan_threshold": 0.3
    }
)

# Ejemplo 1: Planificar movimiento básico
print("=== Planificación de Movimiento ===")
context = {
    "start_position": (0, 0),
    "goal_position": (100, 100),
    "obstacles": [
        {"id": "obs1", "position": (50, 50), "radius": 5.0, "velocity": (0, 0), "type": "static"},
        {"id": "obs2", "position": (80, 60), "radius": 3.0, "velocity": (1, 0), "type": "dynamic"}
    ]
}

result = planner.run("Navigate from start to goal avoiding obstacles")
print(f"Plan generado: {result['plan']['plan_id']}")
print(f"Tipo de movimiento: {result['plan']['motion_type']}")
print(f"Duración estimada: {result['plan']['duration']:.2f} segundos")
print(f"Puntuación de seguridad: {result['plan']['safety_score']:.2f}")
print(f"Número de waypoints: {len(result['plan']['waypoints'])}")

# Ejemplo 2: Obtener plan actual
print("\n=== Plan Actual ===")
current_plan = planner.get_current_plan()
if current_plan:
    print(f"ID del plan: {current_plan.plan_id}")
    print(f"Tipo: {current_plan.motion_type.value}")
    print(f"Waypoints: {len(current_plan.waypoints)}")
    print(f"Primeros 3 waypoints: {current_plan.waypoints[:3]}")

# Ejemplo 3: Observar y replanificar
print("\n=== Observación y Replanificación ===")
observation = {
    "position": (30, 30),
    "obstacles": [
        {"id": "obs3", "position": (40, 40), "radius": 4.0, "velocity": (0, 0), "type": "static"}
    ],
    "goal": (100, 100)
}

obs_result = planner.observe(observation)
print(f"Replanificación necesaria: {obs_result['replanned']}")

# Ejemplo 4: Planificar con diferentes tipos de movimiento
print("\n=== Diferentes Tipos de Movimiento ===")
scenarios = [
    {"start": (0, 0), "goal": (100, 0), "name": "Recto"},
    {"start": (0, 0), "goal": (100, 50), "name": "Curva"},
    {"start": (0, 0), "goal": (-50, 50), "name": "Vuelta"}
]

for scenario in scenarios:
    context = {
        "start_position": scenario["start"],
        "goal_position": scenario["goal"],
        "obstacles": []
    }
    thinking = planner.think(f"Planificar {scenario['name']}", context)
    if thinking.get("plan"):
        print(f"{scenario['name']}: {thinking['plan']['motion_type']}")

# Ejemplo 5: Estado del planificador
print("\n=== Estado del Planificador ===")
status = planner.get_status()
print(f"Nombre: {status['name']}")
print(f"Estado: {status['status']}")
print(f"Plan actual: {status['current_plan']}")
print(f"Planes generados: {status['plans_generated']}")
print(f"Obstáculos rastreados: {status['obstacles_tracked']}")



