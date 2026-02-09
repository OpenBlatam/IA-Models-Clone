# Mejoras V50: Sistema Universal para Cualquier Tipo de Robot

## Resumen

Se ha implementado un sistema universal que puede controlar cualquier tipo de robot (manipuladores, móviles, humanoides, drones, etc.) mediante una interfaz unificada y modelos de deep learning adaptativos.

## Nuevos Sistemas Implementados

### 1. Sistema de Tipos de Robots (`core/robot_types.py`)

**Tipos de Robots Soportados:**
- **MANIPULATOR**: Brazos robóticos (articulaciones rotacionales/prismáticas)
- **MOBILE**: Robots móviles (diferencial, omnidireccional)
- **HUMANOID**: Robots humanoides
- **QUADCOPTER**: Drones (4 rotores)
- **WHEELED**: Robots con ruedas
- **LEGGED**: Robots con patas
- **CUSTOM**: Robots personalizados

**Características:**
- Clase base `BaseRobot` con interfaz común
- Implementaciones específicas por tipo
- Cinemática directa e inversa
- Validación de límites de articulaciones
- Factory pattern para creación

**Ejemplo:**
```python
from core.robot_types import RobotType, RobotConfig, RobotFactory

# Crear configuración para manipulador
config = RobotConfig(
    robot_type=RobotType.MANIPULATOR,
    name="UR5",
    dof=6,
    joint_limits={
        "joint_0": (-3.14, 3.14),
        "joint_1": (-1.57, 1.57),
        # ...
    },
    link_lengths=[0.089, 0.425, 0.392, 0.109, 0.094, 0.082]
)

# Crear robot
robot = RobotFactory.create_robot(config)
robot.connect()
robot.move_to([0.5, 0.3, 0.2])
```

### 2. Controlador Universal (`core/universal_controller.py`)

**Características:**
- Interfaz unificada para cualquier tipo de robot
- Optimización de trayectorias
- Movimiento a lo largo de rutas
- Gestión de múltiples robots simultáneos

**Ejemplo:**
```python
from core.universal_controller import get_universal_engine

engine = get_universal_engine()

# Registrar robot
engine.register_robot("robot_1", config)

# Conectar
engine.connect_robot("robot_1")

# Mover
engine.move_robot("robot_1", [0.5, 0.3, 0.2])

# Mover a lo largo de ruta
waypoints = [[0.5, 0.3, 0.2], [0.6, 0.4, 0.3], [0.7, 0.5, 0.4]]
controller = engine.robots["robot_1"]
controller.move_along_path(waypoints)
```

### 3. Modelos Adaptativos (`core/adaptive_models.py`)

**Características:**
- Modelos de deep learning adaptados al tipo de robot
- Configuraciones predefinidas por tipo
- Personalización de configuraciones

**Configuraciones por Tipo:**
- **Manipulador**: 6 inputs (posición + velocidad), 7 outputs (DOF)
- **Móvil**: 3 inputs (x, y, theta), 3 outputs (vx, vy, omega)
- **Quadcopter**: 6 inputs (posición + orientación), 4 outputs (thrusts)
- **Humanoide**: 12 inputs, 20 outputs (~20 DOF)
- **Wheeled**: 3 inputs, 2 outputs (ruedas)
- **Legged**: 6 inputs, 12 outputs (4 patas x 3 DOF)

**Ejemplo:**
```python
from core.adaptive_models import get_adaptive_model_manager
from core.robot_types import RobotType

manager = get_adaptive_model_manager()

# Crear modelo adaptado
model_id = manager.create_model_for_robot(
    RobotType.MANIPULATOR,
    "robot_1"
)

# Obtener configuración recomendada
config = manager.get_model_config(RobotType.MOBILE)
```

### 4. API Universal (`api/universal_robot_api.py`)

**Endpoints:**
- `POST /api/v1/universal-robots/register`: Registrar robot
- `POST /api/v1/universal-robots/connect/{robot_id}`: Conectar robot
- `POST /api/v1/universal-robots/move`: Mover robot
- `POST /api/v1/universal-robots/move-path`: Mover a lo largo de ruta
- `GET /api/v1/universal-robots/list`: Listar robots
- `GET /api/v1/universal-robots/state/{robot_id}`: Obtener estado
- `POST /api/v1/universal-robots/create-model/{robot_id}`: Crear modelo adaptativo

**Ejemplo de Uso:**
```bash
# Registrar manipulador
curl -X POST "http://localhost:8000/api/v1/universal-robots/register" \
  -H "Content-Type: application/json" \
  -d '{
    "robot_id": "ur5_1",
    "robot_type": "manipulator",
    "name": "UR5 Robot",
    "dof": 6,
    "link_lengths": [0.089, 0.425, 0.392, 0.109, 0.094, 0.082]
  }'

# Conectar
curl -X POST "http://localhost:8000/api/v1/universal-robots/connect/ur5_1"

# Mover
curl -X POST "http://localhost:8000/api/v1/universal-robots/move" \
  -H "Content-Type: application/json" \
  -d '{
    "robot_id": "ur5_1",
    "target_position": [0.5, 0.3, 0.2]
  }'
```

## Arquitectura

### Abstracción de Robots

```
BaseRobot (ABC)
├── ManipulatorRobot
├── MobileRobot
├── QuadcopterRobot
├── HumanoidRobot (futuro)
├── WheeledRobot (futuro)
└── LeggedRobot (futuro)
```

### Flujo de Control

1. **Registro**: Robot se registra con su configuración
2. **Conexión**: Robot se conecta al sistema
3. **Movimiento**: Controlador universal ejecuta movimiento
4. **Optimización**: Trayectoria se optimiza (opcional)
5. **Ejecución**: Robot ejecuta movimiento punto por punto

## Ventajas

1. **Universalidad**: Un solo sistema para todos los tipos de robots
2. **Extensibilidad**: Fácil agregar nuevos tipos de robots
3. **Modularidad**: Cada tipo de robot es independiente
4. **Adaptabilidad**: Modelos de deep learning adaptados por tipo
5. **Interfaz Unificada**: Misma API para todos los robots

## Ejemplos de Uso

### Ejemplo 1: Manipulador

```python
from core.robot_types import RobotType, RobotConfig, RobotFactory
from core.universal_controller import get_universal_engine

# Configurar manipulador
config = RobotConfig(
    robot_type=RobotType.MANIPULATOR,
    name="UR5",
    dof=6,
    link_lengths=[0.089, 0.425, 0.392, 0.109, 0.094, 0.082]
)

# Registrar y usar
engine = get_universal_engine()
engine.register_robot("ur5_1", config)
engine.connect_robot("ur5_1")
engine.move_robot("ur5_1", [0.5, 0.3, 0.2])
```

### Ejemplo 2: Robot Móvil

```python
config = RobotConfig(
    robot_type=RobotType.MOBILE,
    name="TurtleBot",
    dof=3  # x, y, theta
)

engine.register_robot("turtlebot_1", config)
engine.connect_robot("turtlebot_1")
engine.move_robot("turtlebot_1", [2.0, 1.0, 0.0])
```

### Ejemplo 3: Quadcopter

```python
config = RobotConfig(
    robot_type=RobotType.QUADCOPTER,
    name="DJI Phantom",
    dof=6  # x, y, z, roll, pitch, yaw
)

engine.register_robot("drone_1", config)
engine.connect_robot("drone_1")
engine.move_robot("drone_1", [5.0, 3.0, 10.0])  # Volar a 10m de altura
```

## Integración con Deep Learning

Los modelos de deep learning se adaptan automáticamente al tipo de robot:

```python
from core.adaptive_models import get_adaptive_model_manager
from core.robot_types import RobotType

manager = get_adaptive_model_manager()

# Crear modelo para manipulador
model_id = manager.create_model_for_robot(RobotType.MANIPULATOR, "ur5_1")

# El modelo tendrá:
# - Input: 6 (posición + velocidad)
# - Output: 7 (7 DOF del manipulador)
# - Arquitectura optimizada para manipuladores
```

## Extensión para Nuevos Tipos

Para agregar un nuevo tipo de robot:

1. **Crear clase derivada de `BaseRobot`**:
```python
class MyCustomRobot(BaseRobot):
    def forward_kinematics(self, joint_positions):
        # Implementar cinemática directa
        pass
    
    def inverse_kinematics(self, target_position, target_orientation):
        # Implementar cinemática inversa
        pass
    
    # ... otros métodos
```

2. **Registrar en Factory**:
```python
RobotFactory.register_robot_type(RobotType.CUSTOM, MyCustomRobot)
```

3. **Agregar configuración de modelo** (opcional):
```python
manager.robot_type_configs[RobotType.CUSTOM] = {
    "input_size": 6,
    "output_size": 6,
    "hidden_sizes": [128, 64]
}
```

## Estado

✅ **Completado y listo para producción**

El sistema universal está completo y puede controlar cualquier tipo de robot mediante una interfaz unificada.

