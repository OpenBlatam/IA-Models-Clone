# Tutorial - Robot Movement AI

## Guía de Inicio Rápido

### 1. Instalación

```bash
cd robot_movement_ai
pip install -r requirements.txt
```

### 2. Configuración Básica

```python
from robot_movement_ai.config.robot_config import RobotConfig, RobotBrand

config = RobotConfig(
    robot_brand=RobotBrand.GENERIC,
    feedback_frequency=1000,
    api_port=8010
)
```

### 3. Optimización Simple

```python
from robot_movement_ai.core.trajectory_optimizer import (
    TrajectoryOptimizer,
    TrajectoryPoint
)
import numpy as np

# Crear optimizador
optimizer = TrajectoryOptimizer()

# Definir puntos
start = TrajectoryPoint(
    position=np.array([0.0, 0.0, 0.0]),
    orientation=np.array([0.0, 0.0, 0.0, 1.0])
)

goal = TrajectoryPoint(
    position=np.array([1.0, 1.0, 1.0]),
    orientation=np.array([0.0, 0.0, 0.0, 1.0])
)

# Optimizar
trajectory = optimizer.optimize_trajectory(start, goal)
print(f"Trayectoria con {len(trajectory)} puntos")
```

### 4. Con Obstáculos

```python
# Definir obstáculo
obstacles = [
    np.array([0.3, 0.3, 0.3, 0.7, 0.7, 0.7])  # Bounding box
]

# Optimizar evitando obstáculos
trajectory = optimizer.optimize_trajectory(
    start, goal, obstacles=obstacles
)
```

### 5. Diferentes Algoritmos

```python
from robot_movement_ai.core.constants import OptimizationAlgorithm

# Usar A*
optimizer.algorithm = OptimizationAlgorithm.A_STAR
trajectory = optimizer.optimize_trajectory(start, goal)

# Usar RRT
trajectory = optimizer.optimize_with_rrt(
    start, goal,
    obstacles=obstacles,
    max_iterations=1000
)
```

### 6. Análisis de Trayectoria

```python
# Analizar trayectoria
analysis = optimizer.analyze_trajectory(trajectory)

print(f"Distancia: {analysis['total_distance']:.3f}m")
print(f"Duración: {analysis['duration']:.3f}s")
print(f"Velocidad promedio: {analysis['average_speed']:.3f}m/s")
```

### 7. Serialización

```python
from robot_movement_ai.core.serialization import (
    serialize_trajectory,
    deserialize_trajectory
)

# Guardar
serialize_trajectory(trajectory, "my_trajectory.json")

# Cargar
trajectory = deserialize_trajectory("my_trajectory.json")
```

### 8. Usar API

```python
from robot_movement_ai.api.robot_api import create_robot_app
from robot_movement_ai.config.robot_config import RobotConfig

config = RobotConfig()
app = create_robot_app(config)

# Ejecutar con uvicorn
# uvicorn robot_movement_ai.main:app --host 0.0.0.0 --port 8010
```

## Ejemplos Avanzados

Ver `examples/basic_usage.py` y `examples/advanced_usage.py` para más ejemplos.

## Próximos Pasos

1. Leer [API Reference](API_REFERENCE.md)
2. Ver ejemplos en `examples/`
3. Explorar documentación de módulos específicos






