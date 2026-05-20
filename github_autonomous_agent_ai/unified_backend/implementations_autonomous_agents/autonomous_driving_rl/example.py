"""
Ejemplo de uso: Deep Reinforcement Learning framework for Autonomous Driving
============================================================================
"""

from autonomous_driving_rl import AutonomousDrivingRL, DrivingState, DrivingAction
from datetime import datetime

# Crear agente de conducción autónoma con RL
agent = AutonomousDrivingRL(
    name="driving_agent_01",
    config={
        "learning_rate": 0.001,
        "gamma": 0.99,
        "epsilon": 0.1
    }
)

# Ejemplo 1: Entrenar el agente
print("=== Entrenamiento del Agente ===")
training_stats = agent.train(num_episodes=10)
print(f"Episodios entrenados: {training_stats['episodes']}")
print(f"Recompensa promedio: {training_stats['average_reward']:.2f}")
print(f"Longitud promedio: {training_stats['average_length']:.2f}")

# Ejemplo 2: Usar el agente para una tarea de conducción
print("\n=== Ejecución de Tarea ===")
initial_state = DrivingState(
    position=(0.0, 0.0),
    velocity=0.0,
    heading=0.0,
    lane_id=0,
    distance_to_obstacle=100.0,
    traffic_light_state="green",
    nearby_vehicles=[]
)

result = agent.run("Navigate to destination")
print(f"Tarea: {result['task']}")
print(f"Acción seleccionada: {result['thinking']['selected_action']}")

# Ejemplo 3: Pensar sobre un escenario de conducción
print("\n=== Análisis de Escenario ===")
context = {
    "state": {
        "position": (50.0, 30.0),
        "velocity": 25.0,
        "heading": 0.5,
        "lane_id": 1,
        "distance_to_obstacle": 15.0,
        "traffic_light_state": "yellow",
        "nearby_vehicles": [
            {"id": "v1", "position": (60.0, 30.0), "velocity": 20.0}
        ]
    }
}

thinking = agent.think("Approaching intersection with yellow light", context)
print(f"Estado actual: Velocidad={thinking['current_state']['velocity']}")
print(f"Acción recomendada: {thinking['selected_action']}")

# Ejemplo 4: Entrenar un episodio específico
print("\n=== Entrenamiento de Episodio ===")
episode_stats = agent.train_episode(initial_state, max_steps=50)
print(f"Recompensa del episodio: {episode_stats['episode_reward']:.2f}")
print(f"Pasos ejecutados: {episode_stats['steps']}")

# Ejemplo 5: Obtener estado del agente
print("\n=== Estado del Agente ===")
status = agent.get_status()
print(f"Nombre: {status['name']}")
print(f"Estado: {status['status']}")
print(f"Episodios entrenados: {status['episodes_trained']}")
print(f"Recompensa promedio: {status['average_reward']:.2f}")
print(f"Epsilon actual: {status['epsilon']:.3f}")



