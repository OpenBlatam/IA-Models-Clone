# Guía de Uso - Humanoid Devin Robot

## Tabla de Contenidos

1. [Instalación](#instalación)
2. [Uso Básico](#uso-básico)
3. [Control de Articulaciones](#control-de-articulaciones)
4. [Control de Pose](#control-de-pose)
5. [Navegación y Caminar](#navegación-y-caminar)
6. [Deep Learning](#deep-learning)
7. [Integración con Visión](#integración-con-visión)
8. [Manejo de Errores](#manejo-de-errores)
9. [Utilidades](#utilidades)
10. [Ejemplos](#ejemplos)

## Instalación

```python
# Instalar dependencias
pip install -r requirements.txt

# Verificar instalación
from humanoid_devin import HumanoidDevinDriver, DRIVER_AVAILABLE
print(f"Driver disponible: {DRIVER_AVAILABLE}")
```

## Uso Básico

### Conexión Simple

```python
import asyncio
from humanoid_devin import HumanoidDevinDriver, RobotType

async def main():
    # Crear driver
    robot = HumanoidDevinDriver(
        robot_ip="192.168.1.100",
        robot_port=30001,
        dof=32,
        robot_type=RobotType.GENERIC
    )
    
    # Conectar
    await robot.connect()
    
    # Obtener estado
    status = await robot.get_status()
    print(status)
    
    # Desconectar
    await robot.disconnect()

asyncio.run(main())
```

### Configuración Avanzada

```python
robot = HumanoidDevinDriver(
    robot_ip="192.168.1.100",
    dof=32,
    use_ml=True,              # Habilitar modelos de ML
    use_diffusion=True,       # Habilitar modelos de difusión
    use_ros2=True,            # Habilitar ROS 2
    use_moveit2=True,         # Habilitar MoveIt 2
    use_opencv=True,          # Habilitar OpenCV
    use_nav2=True,            # Habilitar Nav2
    robot_type=RobotType.GENERIC
)
```

## Control de Articulaciones

### Establecer Posiciones

```python
# Obtener posiciones actuales
current = await robot.get_joint_positions()

# Crear posiciones objetivo
target = current.copy()
target[0] = 0.5   # Mover primera articulación
target[1] = -0.3  # Mover segunda articulación

# Establecer posiciones
await robot.set_joint_positions(target)
```

### Interpolación Suave

```python
from humanoid_devin.utils import interpolate_joint_positions, smooth_trajectory

# Interpolar entre posiciones
trajectory = interpolate_joint_positions(
    start=current,
    end=target,
    num_steps=20
)

# Suavizar trayectoria
smooth_traj = smooth_trajectory(trajectory, window_size=5)

# Ejecutar paso a paso
for positions in smooth_traj:
    await robot.set_joint_positions(positions)
    await asyncio.sleep(0.05)
```

## Control de Pose

### Movimiento a Pose Específica

```python
import numpy as np
from humanoid_devin.utils import validate_pose

# Definir pose objetivo
position = np.array([0.3, -0.2, 1.0])
orientation = np.array([0.0, 0.0, 0.0, 1.0])  # Quaternion [x, y, z, w]

# Validar pose
valid_pos, valid_ori = validate_pose(position, orientation)

# Mover a pose
await robot.move_to_pose(
    position=valid_pos,
    orientation=valid_ori,
    hand="right"
)
```

### Conversión de Quaterniones

```python
from humanoid_devin.utils import (
    quaternion_to_euler,
    euler_to_quaternion,
    normalize_quaternion
)

# Convertir quaternion a Euler
roll, pitch, yaw = quaternion_to_euler([0.0, 0.0, 0.0, 1.0])

# Convertir Euler a quaternion
quaternion = euler_to_quaternion(roll, pitch, yaw)

# Normalizar quaternion
normalized = normalize_quaternion(quaternion)
```

## Navegación y Caminar

### Caminar Simple

```python
# Caminar hacia adelante
await robot.walk(
    direction="forward",
    distance=1.0,
    speed=0.5
)

# Girar
await robot.walk(
    direction="turn_right",
    distance=0.0,
    speed=0.3
)
```

### Navegación con Nav2

```python
# Si Nav2 está disponible, se usa automáticamente
await robot.walk(
    direction="forward",
    distance=2.0,
    speed=0.6
)
```

## Deep Learning

### Generar Trayectorias Suaves

```python
import numpy as np

# Generar trayectoria usando modelo de difusión
trajectory = await robot.generate_smooth_trajectory(
    start_position=np.array([0.0, 0.0, 0.0]),
    end_position=np.array([0.5, 0.0, 0.0]),
    num_steps=50
)

if trajectory is not None:
    print(f"Trayectoria generada: {len(trajectory)} pasos")
```

### Predecir Movimiento

```python
# Predecir movimiento usando Transformer
current = np.array(await robot.get_joint_positions())
target = current + 0.1

predicted = await robot.predict_joint_motion(
    current_joints=current,
    target_joints=target
)

if predicted is not None:
    print(f"Movimiento predicho: {predicted.shape}")
```

## Integración con Visión

### Detección de Caras

```python
import cv2

# Capturar imagen (ejemplo)
image = cv2.imread("image.jpg")

# Detectar caras
if robot.vision and robot.vision.available:
    faces = robot.vision.detect_faces(
        image,
        scale_factor=1.1,
        min_neighbors=4
    )
    
    for face in faces:
        bbox = face['bbox']
        print(f"Cara detectada: {bbox}")
```

### Detección de Bordes

```python
# Detectar bordes
edges = robot.vision.detect_edges(
    image,
    low_threshold=50,
    high_threshold=150
)

# Obtener información de imagen
info = robot.vision.get_image_info(image)
print(f"Imagen: {info['width']}x{info['height']}")
```

## Manejo de Errores

### Captura de Excepciones

```python
from humanoid_devin.exceptions import (
    HumanoidRobotError,
    RobotConnectionError,
    RobotControlError,
    ValidationError
)

try:
    await robot.connect()
    await robot.set_joint_positions([0.0] * 32)
except RobotConnectionError as e:
    print(f"Error de conexión: {e}")
except ValidationError as e:
    print(f"Error de validación: {e}")
except HumanoidRobotError as e:
    print(f"Error del robot: {e}")
```

### Validación de Parámetros

```python
from humanoid_devin.utils import validate_joint_positions

try:
    # Validar posiciones
    valid_positions = validate_joint_positions(
        positions=[0.0] * 32,
        dof=32,
        joint_limits=[(-np.pi, np.pi)] * 32
    )
except ValidationError as e:
    print(f"Error: {e}")
```

## Utilidades

### Funciones de Conversión

```python
from humanoid_devin.utils import (
    clamp,
    normalize_angle,
    calculate_distance
)

# Limitar valor
value = clamp(5.0, 0.0, 10.0)  # Resultado: 5.0

# Normalizar ángulo
angle = normalize_angle(3 * np.pi)  # Resultado: -π

# Calcular distancia
distance = calculate_distance(
    [0.0, 0.0, 0.0],
    [1.0, 1.0, 1.0]
)  # Resultado: √3
```

### Velocidades de Articulaciones

```python
from humanoid_devin.utils import get_joint_velocity

# Calcular velocidad
current = await robot.get_joint_positions()
await asyncio.sleep(0.1)
new = await robot.get_joint_positions()

velocities = get_joint_velocity(
    current_positions=current,
    previous_positions=new,
    dt=0.1
)
```

## Ejemplos

Ver los archivos de ejemplos para más casos de uso:

- `examples/basic_usage.py`: Ejemplos básicos
- `examples/advanced_usage.py`: Ejemplos avanzados

### Ejecutar Ejemplos

```bash
# Ejemplos básicos
python -m humanoid_devin.examples.basic_usage

# Ejemplos avanzados
python -m humanoid_devin.examples.advanced_usage
```

## Mejores Prácticas

1. **Siempre validar parámetros**: Usa las funciones de validación antes de enviar comandos
2. **Manejar errores**: Captura excepciones específicas para mejor debugging
3. **Usar interpolación**: Para movimientos suaves, usa interpolación y suavizado
4. **Monitorear estado**: Verifica el estado del robot regularmente
5. **Cerrar conexiones**: Siempre desconecta el robot al finalizar

## Troubleshooting

### Problemas Comunes

1. **Error de conexión**: Verifica IP y puerto del robot
2. **Validación falla**: Revisa que los parámetros estén en rangos válidos
3. **Integración no disponible**: Verifica que las dependencias estén instaladas
4. **Modelos DL no cargan**: Verifica que use_ml=True y que los modelos estén disponibles

### Logging

```python
import logging

# Configurar logging
logging.basicConfig(level=logging.DEBUG)

# Los módulos usan logging automáticamente
```

