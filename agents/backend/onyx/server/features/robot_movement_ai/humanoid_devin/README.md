# Humanoid Devin Robot

## 🚀 Descripción

Robot humanoide con control mediante **chat natural** y **Deep Learning**. Sistema completo y profesional que permite controlar robots humanoides usando lenguaje natural con integración avanzada de modelos de IA, Transformers y difusión.

## ✨ Características Principales

### Control mediante Chat Natural

- **Comandos de Movimiento**: Caminar, correr, saltar, girar
- **Posturas**: Pararse, sentarse, agacharse
- **Manipulación Bimanual**: Agarrar y soltar objetos con ambas manos
- **Gestos**: Saludar, expresiones, movimientos coordinados
- **Control de Manos**: Movimiento independiente de cada mano

### Integraciones Avanzadas

#### ROS 2 (Robot Operating System)
- **Nodos ROS 2**: Publicación y suscripción a topics estándar
- **TF2**: Transformaciones de coordenadas
- **Topics**: `/cmd_vel`, `/joint_states`, `/odom`, `/joint_commands`
- **Documentación**: https://docs.ros.org/

#### MoveIt 2 (Motion Planning)
- **Planificación de Trayectorias**: Planificación avanzada de movimientos
- **Colisiones**: Detección y evitación de colisiones
- **IK Solver**: Resolución de cinemática inversa
- **Documentación**: https://moveit.ai/

#### OpenCV (Computer Vision)
- **Detección de Objetos**: Detección de caras, objetos, gestos
- **Procesamiento de Imágenes**: Filtros, transformaciones, análisis
- **Integración ROS 2**: Conversión entre formatos ROS y OpenCV
- **Documentación**: https://opencv.org/

#### TensorFlow y PyTorch (Deep Learning)
- **Modelos de IA**: Entrenamiento y uso de modelos de deep learning
- **Control Neuronal**: Control basado en redes neuronales
- **Aprendizaje**: Aprendizaje por refuerzo y supervisado
- **Documentación**: 
  - TensorFlow: https://www.tensorflow.org/
  - PyTorch: https://pytorch.org/

#### Gazebo / Ignition Gazebo (Simulation)
- **Simulación Física**: Simulación realista del robot humanoide
- **Entorno Virtual**: Pruebas en entornos simulados
- **Documentación**: https://gazebosim.org/

#### NVIDIA Isaac Sim
- **Simulación Avanzada**: Simulación de alta fidelidad
- **Omniverse**: Integración con NVIDIA Omniverse
- **Documentación**: https://developer.nvidia.com/isaac

#### PCL (Point Cloud Library)
- **Procesamiento 3D**: Análisis de nubes de puntos
- **Detección de Objetos**: Segmentación y filtrado
- **LiDAR**: Procesamiento de datos LiDAR
- **Documentación**: https://pointclouds.org/

#### Nav2 (ROS 2 Navigation Stack)
- **Navegación Autónoma**: Navegación y localización
- **Path Planning**: Planificación de rutas
- **SLAM**: Mapeo simultáneo y localización
- **Documentación**: https://nav2.org/

#### Poppy Humanoid
- **Robot Poppy**: Soporte para robots Poppy Ergo Jr y Poppy Humanoid
- **PyPot**: Control de motores Dynamixel
- **Documentación**: https://www.poppy-project.org/

#### iCub
- **Robot iCub**: Soporte para robot humanoide iCub
- **YARP**: Comunicación vía YARP (Yet Another Robot Platform)
- **Documentación**: https://icub.iit.it/

### Sistemas Avanzados

#### Sistema de Aprendizaje Adaptativo
- **Aprendizaje Continuo**: Aprende de experiencias previas para mejorar movimientos
- **Optimización Automática**: Ajusta parámetros basándose en tasa de éxito
- **Predicción de Éxito**: Estima probabilidad de éxito para acciones futuras
- **Persistencia**: Guarda y carga datos de aprendizaje

#### Sistema de Recuperación de Errores
- **Recuperación Automática**: Detecta y recupera errores automáticamente
- **Múltiples Estrategias**: Retry, Rollback, Alternative, Emergency Stop
- **Reintentos Inteligentes**: Reintenta con delays adaptativos
- **Estrategias Personalizables**: Registra estrategias de recuperación personalizadas

#### Optimizador de Energía
- **Monitoreo de Consumo**: Rastrea consumo de energía en tiempo real
- **Optimización de Eficiencia**: Optimiza parámetros para reducir consumo
- **Presupuesto de Energía**: Verifica y limita consumo según presupuesto
- **Recomendaciones**: Sugiere optimizaciones para reducir consumo

#### Sistema de Telemetría Avanzada
- **Monitoreo Completo**: Registra todos los estados del robot en tiempo real
- **Historial Completo**: Poses, velocidades, aceleraciones, torques
- **Monitoreo de Salud**: Potencia, temperatura, eventos y alertas
- **Callbacks**: Sistema de callbacks para reaccionar a eventos
- **Exportación**: Exporta datos para análisis posterior

#### Planificador Predictivo
- **Predicción de Trayectorias**: Predice movimientos futuros
- **Detección de Colisiones**: Predice colisiones antes de que ocurran
- **Optimización Multi-objetivo**: Optimiza según múltiples objetivos
- **Planes Activos**: Gestiona múltiples planes predictivos simultáneos

### Arquitectura

- **32 DOF (Degrees of Freedom)**: Control completo del cuerpo humanoide
- **Control en Tiempo Real**: Feedback y control a alta frecuencia
- **Integración LLM**: Soporte para OpenAI y Anthropic para interpretación de comandos complejos
- **API RESTful**: Endpoints completos para control programático
- **WebSocket**: Chat en tiempo real

## 📦 Instalación

### Prerrequisitos

- Python 3.8+
- pip
- (Opcional) OpenAI API key o Anthropic API key para LLM
- ROS 2 (Humble o más reciente) - https://docs.ros.org/
- (Opcional) Gazebo o Ignition Gazebo para simulación
- (Opcional) NVIDIA Isaac Sim para simulación avanzada
- (Opcional) CUDA para aceleración GPU con TensorFlow/PyTorch

### Instalación

#### 1. Instalar ROS 2

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install ros-humble-desktop-full

# Configurar entorno
source /opt/ros/humble/setup.bash
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
```

#### 2. Instalar MoveIt 2

```bash
sudo apt install ros-humble-moveit
```

#### 3. Instalar Nav2

```bash
sudo apt install ros-humble-navigation2 ros-humble-nav2-bringup
```

#### 4. Instalar dependencias Python

```bash
cd humanoid_devin
pip install -r requirements.txt
```

#### 5. Instalar OpenCV

```bash
pip install opencv-python opencv-contrib-python
```

#### 6. Instalar TensorFlow y PyTorch

```bash
# TensorFlow
pip install tensorflow

# PyTorch (verificar versión según tu sistema)
pip install torch torchvision torchaudio
```

#### 7. Instalar PCL

```bash
# Ubuntu/Debian
sudo apt install libpcl-dev python3-pcl

# Python bindings
pip install python-pcl
```

#### 8. Instalar Poppy (Opcional)

```bash
pip install pypot poppy-humanoid
```

#### 9. Instalar iCub/YARP (Opcional)

```bash
# Ver instrucciones en: https://icub.iit.it/
# Requiere compilación desde fuente
```

## 🚀 Uso

### Iniciar Servidor

```bash
python -m humanoid_devin.main
```

O con opciones personalizadas:

```bash
python -m humanoid_devin.main \
    --host 0.0.0.0 \
    --port 8020 \
    --robot-ip 192.168.1.100 \
    --dof 32 \
    --llm-provider openai \
    --llm-api-key your_key_here
```

### Control mediante Chat

#### API REST

```bash
# Caminar hacia adelante
curl -X POST http://localhost:8020/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "walk forward 2 meters"}'

# Pararse
curl -X POST http://localhost:8020/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "stand up"}'

# Agarrar objeto
curl -X POST http://localhost:8020/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "grasp with right hand"}'

# Saludar
curl -X POST http://localhost:8020/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "wave with left hand"}'

# Estado
curl http://localhost:8020/api/v1/status

# Detener
curl -X POST http://localhost:8020/api/v1/stop
```

#### WebSocket (Tiempo Real)

```python
import asyncio
import websockets
import json

async def chat():
    uri = "ws://localhost:8020/ws/chat"
    async with websockets.connect(uri) as websocket:
        # Enviar comando
        await websocket.send("walk forward 1 meter")
        
        # Recibir respuesta
        response = await websocket.recv()
        print(json.loads(response))

asyncio.run(chat())
```

### Ejemplos de Comandos de Chat

#### Movimiento

- `walk forward 2 meters` - Caminar hacia adelante
- `walk backward 1 meter` - Caminar hacia atrás
- `walk left 0.5 meters` - Caminar a la izquierda
- `turn left 90 degrees` - Girar a la izquierda
- `turn right` - Girar a la derecha

#### Posturas

- `stand up` - Pararse
- `sit down` - Sentarse
- `crouch` - Agacharse

#### Manipulación

- `grasp with right hand` - Agarrar con mano derecha
- `grasp with left hand` - Agarrar con mano izquierda
- `release with right hand` - Soltar con mano derecha
- `move right hand to (0.5, 0.3, 1.0)` - Mover mano a posición

#### Gestos

- `wave with right hand` - Saludar con mano derecha
- `wave with left hand` - Saludar con mano izquierda

#### Control

- `stop` - Detener movimiento
- `status` - Obtener estado del robot

## 🏗️ Arquitectura

```
humanoid_devin/
├── drivers/
│   └── humanoid_devin_driver.py    # Driver del robot humanoide
├── core/
│   ├── humanoid_chat_controller.py # Controlador de chat natural
│   └── humanoid_movement_engine.py # Motor de movimiento
├── api/
│   └── humanoid_api.py             # API RESTful
├── config/
│   └── humanoid_config.py          # Configuración
└── docs/
    └── TUTORIAL.md                  # Tutorial completo
```

## 📊 Grados de Libertad (DOF)

El robot humanoide estándar tiene **32 DOF**:

- **Cabeza**: 2 DOF (yaw, pitch)
- **Brazos (cada uno)**: 7 DOF (shoulder: pitch, roll, yaw; elbow: pitch, roll; wrist: yaw, pitch)
- **Torso**: 3 DOF (yaw, pitch, roll)
- **Piernas (cada una)**: 6 DOF (hip: yaw, roll, pitch; knee: pitch; ankle: pitch, roll)

## 🔧 Configuración Avanzada

### Integración con LLM

```python
from humanoid_devin import HumanoidAPI

api = HumanoidAPI(
    robot_ip="192.168.1.100",
    llm_provider="openai",
    llm_api_key="your_key_here",
    llm_model="gpt-4"
)

api.run(port=8020)
```

### Uso Directo del Driver

```python
from humanoid_devin import HumanoidDevinDriver

driver = HumanoidDevinDriver(robot_ip="192.168.1.100", dof=32)
await driver.connect()

# Caminar
await driver.walk(direction="forward", distance=2.0, speed=0.5)

# Agarrar objeto
await driver.grasp(hand="right")

# Pararse
await driver.stand()
```

## 📈 Aplicaciones

### Investigación

- Robótica humanoide
- Interacción humano-robot
- Manipulación bimanual
- Locomoción bípeda

### Industria

- Asistencia en almacenes
- Manipulación de objetos
- Inspección y mantenimiento
- Interacción con clientes

## 🔒 Seguridad

- Detección de colisiones en tiempo real
- Límites de velocidad y aceleración configurables
- Parada de emergencia automática
- Validación de comandos antes de ejecución
- Monitoreo continuo de estado del robot

## 📝 Licencia

Copyright (c) 2025 Blatam Academy

## 🤝 Contribuir

Las contribuciones son bienvenidas. Por favor, abre un issue o pull request.

## 🔬 Sistemas Avanzados

### Sistema de Aprendizaje Adaptativo

El sistema aprende de experiencias previas para mejorar el rendimiento:

```python
from humanoid_devin import AdaptiveLearningSystem

# Crear sistema de aprendizaje
learning = AdaptiveLearningSystem(memory_size=1000, learning_rate=0.01)

# Registrar experiencia
learning.record_experience(
    action_type="walk",
    parameters={"speed": 0.5, "distance": 2.0},
    result={"success": True},
    success=True,
    execution_time=4.5
)

# Obtener parámetros óptimos
optimal = learning.get_optimal_parameters("walk", {"speed": 0.5, "distance": 2.0})
print(f"Parámetros óptimos: {optimal}")

# Obtener probabilidad de éxito
prob = learning.get_success_probability("walk")
print(f"Probabilidad de éxito: {prob:.1%}")
```

### Sistema de Recuperación de Errores

Recuperación automática de errores:

```python
from humanoid_devin import ErrorRecoverySystem

# Crear sistema de recuperación
recovery = ErrorRecoverySystem(max_retries=3, retry_delay=1.0)

# Intentar recuperación automática
try:
    await robot.walk(direction="forward", distance=2.0)
except Exception as e:
    success = await recovery.recover_from_error(e, {"action_type": "walk"})
    if success:
        print("Recuperación exitosa")
```

### Optimizador de Energía

Optimización de consumo de energía:

```python
from humanoid_devin import EnergyOptimizer

# Crear optimizador
energy = EnergyOptimizer(target_power_budget=100.0)

# Registrar consumo
energy.record_power_consumption("left_arm", power=25.0, duration=5.0)

# Estimar consumo
estimated = energy.estimate_power_consumption("walk", {"speed": 0.5})

# Optimizar parámetros
optimized = energy.optimize_movement_parameters("walk", {"speed": 0.8})
print(f"Parámetros optimizados: {optimized}")
```

### Sistema de Telemetría Avanzada

Monitoreo completo del robot:

```python
from humanoid_devin import TelemetrySystem
import numpy as np

# Crear sistema de telemetría
telemetry = TelemetrySystem(buffer_size=5000, sampling_rate=10.0)

# Registrar estados
telemetry.record_joint_states(
    joint_positions=np.array([0.1, 0.2, 0.3]),
    joint_velocities=np.array([0.01, 0.02, 0.03])
)

# Registrar potencia y temperatura
telemetry.record_power("left_arm", power=25.0)
telemetry.record_temperature("motor_1", temperature=45.0)

# Exportar datos
telemetry.export_data("telemetry_data.json")
```

### Planificador Predictivo

Planificación predictiva de movimientos:

```python
from humanoid_devin import PredictivePlanner

# Crear planificador
planner = PredictivePlanner(prediction_horizon=5.0)

# Predecir trayectoria
trajectory = planner.predict_trajectory(
    current_state={"joint_positions": [0.0, 0.1, 0.2]},
    target_state={"joint_positions": [0.5, 0.6, 0.7]}
)

# Predecir colisiones
collision = planner.predict_collision(trajectory, obstacles)

# Optimizar trayectoria
optimized = planner.optimize_trajectory(
    trajectory,
    objectives=["smoothness", "energy"]
)
```

Ver `examples/advanced_systems.py` para más ejemplos.

## 📞 Soporte

Para soporte técnico, contacta a: support@blatam-academy.com

## 🎯 Características Principales

| Característica | Humanoid Devin |
|---------------|----------------|
| Tipo de Robot | Humanoide |
| DOF | Hasta 100 (configurable) |
| Control Chat | ✅ Chat natural |
| LLM Integration | ✅ Integración completa |
| WebSocket | ✅ Comunicación en tiempo real |
| Movimientos | Locomoción + Manipulación |
| Posturas | Completas (stand, sit, crouch) |
| Manipulación | Bimanual avanzada |
| Deep Learning | ✅ Transformers + Diffusion |
| Visión | ✅ OpenCV integrado |
| Navegación | ✅ Nav2 integrado |

## 🔧 Uso de Integraciones

### ROS 2

```python
from humanoid_devin import HumanoidDevinDriver, ROS2Integration

driver = HumanoidDevinDriver(
    robot_ip="192.168.1.100",
    use_ros2=True
)

# El driver automáticamente usa ROS 2 para comunicación
await driver.connect()
await driver.walk(direction="forward", distance=2.0)
```

### MoveIt 2

```python
from humanoid_devin import MoveIt2Integration

moveit = MoveIt2Integration(group_name="arm_group")
plan = moveit.plan_to_pose(x=0.5, y=0.3, z=1.0)
if plan["success"]:
    moveit.execute_plan(plan["plan"])
```

### OpenCV

```python
from humanoid_devin import VisionProcessor
import cv2

vision = VisionProcessor()
image = cv2.imread("image.jpg")
results = vision.process_image(image)
faces = results["faces"]
objects = results["objects"]
```

### TensorFlow/PyTorch

```python
from humanoid_devin import AIModelManager, TensorFlowModel, PyTorchModel

ai_manager = AIModelManager()

# Cargar modelo TensorFlow
ai_manager.load_tensorflow_model("motion_model", "path/to/model.h5")

# Cargar modelo PyTorch
pytorch_model = PyTorchModel(input_size=32, output_size=32)
ai_manager.load_pytorch_model("control_model", pytorch_model)

# Usar modelos
joint_positions = ai_manager.predict_with_tf("motion_model", input_data)
```

### PCL

```python
from humanoid_devin import PointCloudProcessor
import numpy as np

pcl_processor = PointCloudProcessor()
points = np.random.rand(1000, 3).astype(np.float32) * 10
results = pcl_processor.process_point_cloud(points)
```

### Nav2

```python
from humanoid_devin import HumanoidDevinDriver

driver = HumanoidDevinDriver(
    robot_ip="192.168.1.100",
    use_nav2=True
)

await driver.connect()
# Nav2 se usa automáticamente en walk()
await driver.walk(direction="forward", distance=5.0)
```

### Poppy

```python
from humanoid_devin import HumanoidDevinDriver

driver = HumanoidDevinDriver(
    robot_ip="192.168.1.100",
    robot_type="poppy"
)

await driver.connect()
await driver.walk(direction="forward", distance=1.0)
```

### iCub

```python
from humanoid_devin import HumanoidDevinDriver

driver = HumanoidDevinDriver(
    robot_ip="192.168.1.100",
    robot_type="icub"
)

await driver.connect()
await driver.walk(direction="forward", distance=1.0)
```

## 🗺️ Roadmap

- [x] Integración con ROS 2
- [x] Integración con MoveIt 2
- [x] Integración con OpenCV
- [x] Integración con TensorFlow/PyTorch
- [x] Integración con PCL
- [x] Integración con Nav2
- [x] Integración con Poppy
- [x] Integración con iCub
- [ ] Dashboard web en tiempo real
- [ ] Entrenamiento de modelos personalizados
- [ ] Reconocimiento de voz
- [ ] Integración con Gazebo/Isaac Sim
- [ ] Control de equilibrio dinámico
- [ ] SLAM integrado

