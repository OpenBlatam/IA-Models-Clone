"""
Advanced Systems Examples - Humanoid Devin Robot
=================================================

Ejemplos de uso de sistemas avanzados:
- Adaptive Learning System
- Error Recovery System
- Energy Optimizer
- Telemetry System
- Predictive Planner
"""

import asyncio
import logging
from pathlib import Path

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def example_adaptive_learning():
    """Ejemplo 1: Sistema de Aprendizaje Adaptativo"""
    print("\n=== Ejemplo 1: Sistema de Aprendizaje Adaptativo ===\n")
    
    from humanoid_devin import AdaptiveLearningSystem
    
    # Crear sistema de aprendizaje
    learning_system = AdaptiveLearningSystem(
        memory_size=500,
        learning_rate=0.01,
        discount_factor=0.95
    )
    
    # Registrar experiencias
    learning_system.record_experience(
        action_type="walk",
        parameters={"direction": "forward", "distance": 2.0, "speed": 0.5},
        result={"success": True, "final_position": [2.0, 0.0, 0.0]},
        success=True,
        execution_time=4.5,
        energy_consumed=225.0
    )
    
    learning_system.record_experience(
        action_type="walk",
        parameters={"direction": "forward", "distance": 2.0, "speed": 0.8},
        result={"success": False, "error": "collision"},
        success=False,
        execution_time=2.1,
        energy_consumed=168.0
    )
    
    # Obtener parámetros óptimos
    optimal_params = learning_system.get_optimal_parameters(
        action_type="walk",
        default_parameters={"direction": "forward", "distance": 2.0, "speed": 0.5}
    )
    print(f"Parámetros óptimos: {optimal_params}")
    
    # Obtener probabilidad de éxito
    success_prob = learning_system.get_success_probability("walk")
    print(f"Probabilidad de éxito: {success_prob:.1%}")
    
    # Obtener sugerencias de optimización
    suggestions = learning_system.get_optimization_suggestions("walk")
    print(f"Sugerencias: {suggestions}")
    
    # Guardar datos de aprendizaje
    learning_system.save_learning_data("learning_data.json")
    print("Datos de aprendizaje guardados")
    
    # Obtener estadísticas
    stats = learning_system.get_statistics()
    print(f"Estadísticas: {stats}")


async def example_error_recovery():
    """Ejemplo 2: Sistema de Recuperación de Errores"""
    print("\n=== Ejemplo 2: Sistema de Recuperación de Errores ===\n")
    
    from humanoid_devin import ErrorRecoverySystem, RecoveryStrategy
    
    # Crear sistema de recuperación
    recovery_system = ErrorRecoverySystem(
        max_retries=3,
        retry_delay=1.0,
        enable_auto_recovery=True
    )
    
    # Simular error de conexión
    class ConnectionError(Exception):
        pass
    
    error = ConnectionError("Connection timeout")
    context = {
        "action_type": "connect",
        "action": lambda: asyncio.sleep(0.1)  # Simulación
    }
    
    # Intentar recuperación
    success = await recovery_system.recover_from_error(error, context)
    print(f"Recuperación exitosa: {success}")
    
    # Obtener estadísticas
    stats = recovery_system.get_recovery_statistics()
    print(f"Estadísticas de recuperación: {stats}")
    
    # Registrar estrategia personalizada
    async def custom_recovery_strategy(error, context, robot_driver):
        print("Aplicando estrategia personalizada...")
        return True
    
    recovery_system.register_recovery_strategy("CustomError", custom_recovery_strategy)
    print("Estrategia personalizada registrada")


async def example_energy_optimizer():
    """Ejemplo 3: Optimizador de Energía"""
    print("\n=== Ejemplo 3: Optimizador de Energía ===\n")
    
    from humanoid_devin import EnergyOptimizer
    
    # Crear optimizador de energía
    energy_optimizer = EnergyOptimizer(
        target_power_budget=100.0,  # 100 Watts
        enable_power_limiting=True
    )
    
    # Registrar consumo de energía
    energy_optimizer.record_power_consumption(
        component="left_arm",
        power=25.0,
        duration=5.0
    )
    
    energy_optimizer.record_power_consumption(
        component="right_arm",
        power=30.0,
        duration=5.0
    )
    
    energy_optimizer.record_power_consumption(
        component="locomotion",
        power=50.0,
        duration=10.0
    )
    
    # Estimar consumo para una acción
    estimated_power = energy_optimizer.estimate_power_consumption(
        action_type="walk",
        parameters={"speed": 0.5, "distance": 2.0}
    )
    print(f"Potencia estimada para caminar: {estimated_power:.2f}W")
    
    # Verificar presupuesto
    within_budget, warning = energy_optimizer.check_power_budget(estimated_power)
    print(f"Dentro del presupuesto: {within_budget}")
    if warning:
        print(f"Advertencia: {warning}")
    
    # Optimizar parámetros
    original_params = {"speed": 0.8, "distance": 3.0, "acceleration": 1.0}
    optimized_params = energy_optimizer.optimize_movement_parameters(
        action_type="walk",
        parameters=original_params,
        target_energy=500.0  # 500 Joules
    )
    print(f"Parámetros originales: {original_params}")
    print(f"Parámetros optimizados: {optimized_params}")
    
    # Obtener estadísticas
    stats = energy_optimizer.get_energy_statistics()
    print(f"Estadísticas de energía: {stats}")
    
    # Obtener recomendaciones
    recommendations = energy_optimizer.get_power_recommendations()
    print(f"Recomendaciones: {recommendations}")


async def example_integrated_systems():
    """Ejemplo 4: Sistemas Integrados"""
    print("\n=== Ejemplo 4: Sistemas Integrados ===\n")
    
    from humanoid_devin import (
        AdaptiveLearningSystem,
        ErrorRecoverySystem,
        EnergyOptimizer
    )
    
    # Crear todos los sistemas
    learning = AdaptiveLearningSystem()
    recovery = ErrorRecoverySystem()
    energy = EnergyOptimizer(target_power_budget=100.0)
    
    # Simular ciclo de trabajo integrado
    async def execute_action_with_all_systems(action_type, parameters):
        """Ejecutar acción con todos los sistemas integrados"""
        
        # 1. Obtener parámetros óptimos del sistema de aprendizaje
        optimal_params = learning.get_optimal_parameters(action_type, parameters)
        print(f"Parámetros óptimos: {optimal_params}")
        
        # 2. Estimar y verificar consumo de energía
        estimated_power = energy.estimate_power_consumption(action_type, optimal_params)
        within_budget, warning = energy.check_power_budget(estimated_power)
        
        if not within_budget:
            # Optimizar para reducir consumo
            optimal_params = energy.optimize_movement_parameters(
                action_type, optimal_params
            )
            print(f"Parámetros optimizados por energía: {optimal_params}")
        
        # 3. Intentar ejecutar con recuperación de errores
        try:
            # Simular ejecución
            success = True
            execution_time = 5.0
            energy_consumed = estimated_power * execution_time
            
            # Registrar experiencia
            learning.record_experience(
                action_type=action_type,
                parameters=optimal_params,
                result={"success": success},
                success=success,
                execution_time=execution_time,
                energy_consumed=energy_consumed
            )
            
            # Registrar consumo
            energy.record_power_consumption(
                component=action_type,
                power=estimated_power,
                duration=execution_time
            )
            
            print(f"Acción ejecutada exitosamente")
            
        except Exception as e:
            # Intentar recuperación
            context = {
                "action_type": action_type,
                "parameters": optimal_params
            }
            recovery_success = await recovery.recover_from_error(e, context)
            
            if recovery_success:
                print("Recuperación exitosa")
            else:
                print("Recuperación fallida")
    
    # Ejecutar ejemplo
    await execute_action_with_all_systems(
        "walk",
        {"direction": "forward", "distance": 2.0, "speed": 0.5}
    )
    
    # Mostrar estadísticas de todos los sistemas
    print("\n=== Estadísticas Integradas ===")
    print(f"Aprendizaje: {learning.get_statistics()}")
    print(f"Recuperación: {recovery.get_recovery_statistics()}")
    print(f"Energía: {energy.get_energy_statistics()}")


async def main():
    """Ejecutar todos los ejemplos"""
    print("=" * 60)
    print("Ejemplos de Sistemas Avanzados - Humanoid Devin Robot")
    print("=" * 60)
    
    await example_adaptive_learning()
    await example_error_recovery()
    await example_energy_optimizer()
    await example_integrated_systems()
    
    await example_telemetry()
    await example_predictive_planner()
    
    print("\n" + "=" * 60)
    print("Todos los ejemplos completados")
    print("=" * 60)


async def example_telemetry():
    """Ejemplo 5: Sistema de Telemetría"""
    print("\n=== Ejemplo 5: Sistema de Telemetría ===\n")
    
    from humanoid_devin import TelemetrySystem
    import numpy as np
    
    # Crear sistema de telemetría
    telemetry = TelemetrySystem(
        buffer_size=5000,
        enable_persistence=True,
        sampling_rate=10.0
    )
    
    # Registrar estados de articulaciones
    joint_positions = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    joint_velocities = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
    
    telemetry.record_joint_states(
        joint_positions=joint_positions,
        joint_velocities=joint_velocities
    )
    print("Estados de articulaciones registrados")
    
    # Registrar pose
    position = np.array([1.0, 2.0, 0.5])
    orientation = np.array([0.0, 0.0, 0.0, 1.0])  # Quaternion
    telemetry.record_pose(position, orientation, frame_id="base_link")
    print("Pose registrada")
    
    # Registrar consumo de potencia
    telemetry.record_power("left_arm", power=25.0, voltage=12.0, current=2.08)
    telemetry.record_power("right_arm", power=30.0)
    print("Consumo de potencia registrado")
    
    # Registrar temperatura
    telemetry.record_temperature("motor_1", temperature=45.0)
    telemetry.record_temperature("motor_2", temperature=75.0)  # Generará alerta
    print("Temperaturas registradas")
    
    # Agregar evento
    telemetry.add_event(
        "movement_completed",
        {"action": "walk", "distance": 2.0},
        severity="info"
    )
    print("Evento agregado")
    
    # Registrar callback
    def on_alert(alert_data):
        print(f"Alerta recibida: {alert_data['type']}")
    
    telemetry.register_callback("alert", on_alert)
    print("Callback registrado")
    
    # Obtener estadísticas
    stats = telemetry.get_statistics()
    print(f"Estadísticas: {stats}")
    
    # Exportar datos
    telemetry.export_data("telemetry_data.json", ["joint_states", "power_history"])
    print("Datos exportados")


async def example_predictive_planner():
    """Ejemplo 6: Planificador Predictivo"""
    print("\n=== Ejemplo 6: Planificador Predictivo ===\n")
    
    from humanoid_devin import PredictivePlanner
    import numpy as np
    
    # Crear planificador predictivo
    planner = PredictivePlanner(
        prediction_horizon=5.0,
        planning_frequency=10.0
    )
    
    # Estado actual
    current_state = {
        "joint_positions": [0.0, 0.1, 0.2, 0.3, 0.4],
        "joint_velocities": [0.01, 0.02, 0.03, 0.04, 0.05]
    }
    
    # Estado objetivo
    target_state = {
        "joint_positions": [0.5, 0.6, 0.7, 0.8, 0.9]
    }
    
    # Predecir trayectoria
    trajectory = planner.predict_trajectory(
        current_state=current_state,
        target_state=target_state
    )
    print(f"Trayectoria predicha: {len(trajectory['positions'])} pasos")
    print(f"Horizonte: {trajectory['prediction_horizon']}s")
    
    # Predecir colisiones
    obstacles = [
        {"position": [0.3, 0.0, 0.5], "radius": 0.1}
    ]
    collision_info = planner.predict_collision(trajectory, obstacles)
    print(f"Colisión detectada: {collision_info['collision_detected']}")
    
    # Optimizar trayectoria
    optimized = planner.optimize_trajectory(
        trajectory,
        objectives=["smoothness", "energy"],
        weights=[0.7, 0.3]
    )
    print("Trayectoria optimizada")
    
    # Crear plan
    plan = planner.create_plan(
        plan_id="plan_001",
        goal={"target_state": target_state},
        constraints={"max_joint_limits": [1.0] * 5, "min_joint_limits": [-1.0] * 5}
    )
    print(f"Plan creado: {plan['plan_id']}")
    
    # Actualizar plan
    updated_plan = planner.update_plan("plan_001", current_state)
    print("Plan actualizado")
    
    # Obtener estadísticas
    stats = planner.get_statistics()
    print(f"Estadísticas: {stats}")


if __name__ == "__main__":
    asyncio.run(main())

