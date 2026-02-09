# API Reference - Robot Movement AI

## Core Modules

### Trajectory Optimizer

#### `TrajectoryOptimizer`

Optimizador principal de trayectorias.

**Methods:**

- `optimize_trajectory(start, goal, obstacles=None, constraints=None) -> List[TrajectoryPoint]`
  - Optimizar trayectoria desde punto inicial a objetivo
  
- `optimize_with_astar(start, goal, obstacles=None, grid_resolution=None) -> List[TrajectoryPoint]`
  - Optimizar usando algoritmo A*
  
- `optimize_with_rrt(start, goal, obstacles=None, max_iterations=None, step_size=None) -> List[TrajectoryPoint]`
  - Optimizar usando algoritmo RRT
  
- `optimize_multi_objective(start, goal, objectives=None) -> List[TrajectoryPoint]`
  - Optimización multi-objetivo
  
- `analyze_trajectory(trajectory) -> Dict[str, Any]`
  - Analizar trayectoria y retornar métricas
  
- `export_trajectory(trajectory, filepath, format="json") -> bool`
  - Exportar trayectoria a archivo
  
- `import_trajectory(filepath) -> Optional[List[TrajectoryPoint]]`
  - Importar trayectoria desde archivo
  
- `get_statistics() -> Dict[str, Any]`
  - Obtener estadísticas del optimizador

### Movement Engine

#### `RobotMovementEngine`

Motor principal de movimiento robótico.

**Methods:**

- `async initialize() -> None`
  - Inicializar y conectar con robot
  
- `async move_to_pose(pose) -> Dict[str, Any]`
  - Mover robot a pose específica
  
- `async move_along_path(waypoints) -> bool`
  - Mover robot a lo largo de ruta con múltiples waypoints
  
- `async stop_movement() -> None`
  - Detener movimiento actual
  
- `update_obstacles(obstacles) -> None`
  - Actualizar lista de obstáculos conocidos
  
- `get_status() -> Dict[str, Any]`
  - Obtener estado del robot
  
- `get_statistics() -> Dict[str, Any]`
  - Obtener estadísticas del motor

### Chat Controller

#### `ChatRobotController`

Controlador de chat para comandos naturales.

**Methods:**

- `async process_chat_message(message, context=None) -> Dict[str, Any]`
  - Procesar mensaje de chat
  
- `async process_command(command) -> Dict[str, Any]`
  - Procesar comando directo
  
- `get_statistics() -> Dict[str, Any]`
  - Obtener estadísticas del controlador

## Utilities

### Metrics

```python
from robot_movement_ai.core.metrics import (
    record_value,
    increment_counter,
    record_timing,
    get_metrics_collector
)

# Registrar métrica
record_value("metric_name", 42.0)

# Incrementar contador
increment_counter("counter_name")

# Registrar tiempo
record_timing("operation.duration", 0.123)
```

### Serialization

```python
from robot_movement_ai.core.serialization import (
    serialize_trajectory,
    deserialize_trajectory
)

# Serializar
serialize_trajectory(trajectory, "path.json")

# Deserializar
trajectory = deserialize_trajectory("path.json")
```

### Helpers

```python
from robot_movement_ai.core.helpers import (
    clamp,
    lerp,
    euclidean_distance,
    format_duration
)

# Limitar valor
value = clamp(5.0, 0.0, 10.0)

# Interpolación
result = lerp(0.0, 10.0, 0.5)

# Distancia
dist = euclidean_distance(point1, point2)

# Formatear
duration_str = format_duration(3661.5)  # "1h 1m 1.50s"
```

## API Endpoints

### REST API

- `POST /api/v1/move/to` - Mover robot a posición
- `POST /api/v1/chat` - Procesar mensaje de chat
- `POST /api/v1/move/path` - Mover a lo largo de ruta
- `GET /api/v1/status` - Estado del robot
- `GET /api/v1/statistics` - Estadísticas del sistema
- `GET /api/v1/metrics/` - Todas las métricas
- `GET /api/v1/metrics/summary` - Resumen de métricas

### WebSocket

- `ws://host:port/ws/chat` - Chat en tiempo real

## Configuration

### RobotConfig

```python
from robot_movement_ai.config.robot_config import RobotConfig, RobotBrand

config = RobotConfig(
    robot_brand=RobotBrand.KUKA,
    feedback_frequency=1000,
    max_velocity=1.0,
    max_acceleration=2.0
)
```

## Constants

```python
from robot_movement_ai.core.constants import (
    OptimizationAlgorithm,
    DEFAULT_LEARNING_RATE,
    MIN_OBSTACLE_DISTANCE
)

# Algoritmos disponibles
algorithm = OptimizationAlgorithm.PPO
algorithm = OptimizationAlgorithm.DQN
algorithm = OptimizationAlgorithm.A_STAR
algorithm = OptimizationAlgorithm.RRT
```






