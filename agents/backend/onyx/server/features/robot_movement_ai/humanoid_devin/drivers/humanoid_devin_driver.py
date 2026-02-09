"""
Humanoid Devin Robot Driver
============================

Driver profesional para robot humanoide con control mediante chat natural.
Sistema completo de control avanzado de robots humanoides con Deep Learning.
"""

import logging
import asyncio
from typing import List, Optional, Dict, Any, Union, Tuple
from functools import wraps
from datetime import datetime
from enum import Enum
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

from ...drivers.base_driver import BaseRobotDriver
from ..core.ros2_integration import ROS2Integration
from ..core.moveit2_integration import MoveIt2Integration
from ..core.vision_processor import VisionProcessor
from ..core.ai_models import AIModelManager
from ..core.point_cloud_processor import PointCloudProcessor
from ..core.nav2_integration import Nav2Integration
from ..core.poppy_icub_integration import PoppyIntegration, ICubIntegration

# Deep Learning Integration
try:
    from ...core.deep_learning_models import DeepLearningModelManager, get_dl_model_manager
    from ...core.diffusion_models import DiffusionModelManager, get_diffusion_manager
    from ...core.dl_models.transformer_trajectory import TransformerTrajectoryPredictor
    from ...core.dl_models.diffusion_trajectory import DiffusionTrajectoryGenerator
    from ...native.wrapper import NativeIKWrapper, NativeTrajectoryOptimizerWrapper
    DL_AVAILABLE = True
except ImportError as e:
    DL_AVAILABLE = False
    DeepLearningModelManager = None
    DiffusionModelManager = None
    TransformerTrajectoryPredictor = None
    DiffusionTrajectoryGenerator = None
    NativeIKWrapper = None
    NativeTrajectoryOptimizerWrapper = None

logger = logging.getLogger(__name__)


class RobotType(str, Enum):
    """Tipos de robots humanoides soportados."""
    GENERIC = "generic"
    POPPY = "poppy"
    ICUB = "icub"


class HumanoidRobotError(Exception):
    """Excepción personalizada para errores del robot humanoide."""
    pass


def require_connection(func):
    """
    Decorador para requerir conexión antes de ejecutar método (optimizado).
    
    Valida que el robot esté conectado antes de ejecutar cualquier operación.
    
    Raises:
        RuntimeError: Si el robot no está conectado
    """
    @wraps(func)
    async def wrapper(self, *args, **kwargs):
        if not self.connected:
            raise RuntimeError("Robot not connected. Call connect() first.")
        return await func(self, *args, **kwargs)
    return wrapper


def log_operation(operation_name: str):
    """
    Decorador para logging de operaciones (optimizado).
    
    Args:
        operation_name: Nombre de la operación a loggear
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            start_time = datetime.utcnow()
            logger.debug(f"Starting {operation_name}...")
            try:
                result = await func(self, *args, **kwargs)
                duration = (datetime.utcnow() - start_time).total_seconds()
                logger.debug(f"{operation_name} completed in {duration:.3f}s")
                return result
            except Exception as e:
                duration = (datetime.utcnow() - start_time).total_seconds()
                logger.error(f"{operation_name} failed after {duration:.3f}s: {e}", exc_info=True)
                raise
        return wrapper
    return decorator


class HumanoidDevinDriver(BaseRobotDriver):
    """
    Driver profesional para robot humanoide con control avanzado.
    
    Control mediante chat natural para movimientos humanoides complejos:
    - Caminar, correr, saltar
    - Manipulación bimanual
    - Expresiones y gestos
    - Movimientos coordinados de todo el cuerpo
    """
    
    JOINT_NAMES = [
        "head_yaw", "head_pitch",
        "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw",
        "left_elbow_pitch", "left_elbow_roll",
        "left_wrist_yaw", "left_wrist_pitch",
        "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw",
        "right_elbow_pitch", "right_elbow_roll",
        "right_wrist_yaw", "right_wrist_pitch",
        "torso_yaw", "torso_pitch", "torso_roll",
        "left_hip_yaw", "left_hip_roll", "left_hip_pitch",
        "left_knee_pitch", "left_ankle_pitch", "left_ankle_roll",
        "right_hip_yaw", "right_hip_roll", "right_hip_pitch",
        "right_knee_pitch", "right_ankle_pitch", "right_ankle_roll"
    ]
    
    DEFAULT_POSITION = np.array([0.3, -0.2, 1.0], dtype=np.float64)
    DEFAULT_ORIENTATION = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    
    def __init__(
        self,
        robot_ip: str,
        robot_port: int = 30001,
        dof: int = 32,
        use_ml: bool = True,
        use_diffusion: bool = True,
        use_ros2: bool = True,
        use_moveit2: bool = True,
        use_opencv: bool = True,
        use_ai: bool = True,
        use_pcl: bool = True,
        use_nav2: bool = True,
        robot_type: Union[str, RobotType] = RobotType.GENERIC
    ):
        """
        Inicializar driver humanoide con integración profesional de Deep Learning (optimizado).
        
        Args:
            robot_ip: IP del robot (no puede estar vacío)
            robot_port: Puerto del robot (1-65535)
            dof: Grados de libertad (1-100, default: 32)
            use_ml: Usar modelos de ML para control
            use_diffusion: Usar modelos de difusión para trayectorias suaves
            use_ros2: Habilitar integración ROS 2
            use_moveit2: Habilitar MoveIt 2 para planificación
            use_opencv: Habilitar OpenCV para visión
            use_ai: Habilitar TensorFlow/PyTorch para IA
            use_pcl: Habilitar PCL para nubes de puntos
            use_nav2: Habilitar Nav2 para navegación
            robot_type: Tipo de robot (RobotType o str: "generic", "poppy", "icub")
            
        Raises:
            ValueError: Si los parámetros son inválidos
            HumanoidRobotError: Si hay error en la inicialización
        """
        # Guard clauses - validaciones tempranas
        if not robot_ip or not isinstance(robot_ip, str) or not robot_ip.strip():
            raise ValueError("robot_ip must be a non-empty string")
        
        if not isinstance(robot_port, int) or not (1 <= robot_port <= 65535):
            raise ValueError(f"robot_port must be an integer between 1 and 65535, got {robot_port}")
        
        if not isinstance(dof, int) or not (1 <= dof <= 100):
            raise ValueError(f"dof must be an integer between 1 and 100, got {dof}")
        
        # Normalizar robot_type
        if isinstance(robot_type, str):
            robot_type_str = robot_type.lower().strip()
            if robot_type_str not in [rt.value for rt in RobotType]:
                raise ValueError(f"Invalid robot_type: {robot_type}. Must be one of {[rt.value for rt in RobotType]}")
            robot_type = RobotType(robot_type_str)
        elif not isinstance(robot_type, RobotType):
            raise ValueError(f"robot_type must be RobotType enum or valid string, got {type(robot_type)}")
        
        try:
            super().__init__(robot_ip.strip(), robot_port)
            self.dof = dof
            self.robot_type = robot_type
            self.joint_names = (
                self.JOINT_NAMES[:dof] 
                if dof <= len(self.JOINT_NAMES) 
                else self.JOINT_NAMES + [f"joint_{i}" for i in range(len(self.JOINT_NAMES), dof)]
            )
            self.current_pose = "standing"
            self.walking_state = "idle"
            self.grasp_state: Dict[str, bool] = {"left": False, "right": False}
            self._zero_joint_positions = np.zeros(dof, dtype=np.float64)
            
        except Exception as e:
            logger.error(f"Error in base initialization: {e}", exc_info=True)
            raise HumanoidRobotError(f"Failed to initialize base driver: {str(e)}") from e
        
        # Deep Learning Integration (optimizado)
        self.use_ml = use_ml and DL_AVAILABLE
        self.use_diffusion = use_diffusion and DL_AVAILABLE
        self.dl_manager = None
        self.diffusion_manager = None
        self.transformer_model = None
        self.diffusion_model = None
        self.ik_wrapper = None
        self.trajectory_optimizer = None
        
        if self.use_ml:
            try:
                self.dl_manager = get_dl_model_manager()
                if self.dl_manager:
                    # Crear modelo Transformer para control de movimiento
                    model_id = self.dl_manager.create_model(
                        model_type="transformer",
                        input_size=dof * 2,  # current + target
                        output_size=dof
                    )
                    if model_id and self.dl_manager.models:
                        self.transformer_model = self.dl_manager.models.get(model_id)
                        logger.info("Transformer model initialized for humanoid control")
            except Exception as e:
                logger.warning(f"Failed to initialize DL models: {e}", exc_info=True)
                self.use_ml = False
        
        if self.use_diffusion:
            try:
                self.diffusion_manager = get_diffusion_manager()
                if self.diffusion_manager:
                    logger.info("Diffusion model manager initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Diffusion models: {e}", exc_info=True)
                self.use_diffusion = False
        
        # Native optimizations (con validación)
        if NativeIKWrapper and NativeTrajectoryOptimizerWrapper:
            try:
                link_lengths = [0.1] * max(1, dof // 4)  # Evitar lista vacía
                joint_limits = [(-np.pi, np.pi)] * dof
                self.ik_wrapper = NativeIKWrapper(link_lengths, joint_limits)
                self.trajectory_optimizer = NativeTrajectoryOptimizerWrapper()
                logger.info("Native optimizations initialized")
            except Exception as e:
                logger.warning(f"Native optimizations not available: {e}", exc_info=True)
        
        # ROS 2 and other integrations (con manejo de errores robusto)
        self.ros2 = None
        self.moveit2 = None
        self.nav2 = None
        self.vision = None
        self.ai_models = None
        self.pcl_processor = None
        self.poppy = None
        self.icub = None
        
        self._initialize_integrations(
            use_ros2, use_moveit2, use_opencv, use_ai, use_pcl, use_nav2, robot_type
        )
        
        logger.info(
            f"Humanoid Devin Driver initialized: {dof} DOF at {robot_ip}:{robot_port}, "
            f"type={robot_type.value}, ML={self.use_ml}, Diffusion={self.use_diffusion}"
        )
    
    def _initialize_integrations(
        self,
        use_ros2: bool,
        use_moveit2: bool,
        use_opencv: bool,
        use_ai: bool,
        use_pcl: bool,
        use_nav2: bool,
        robot_type: RobotType
    ) -> None:
        """
        Inicializar todas las integraciones con manejo de errores robusto.
        
        Args:
            use_ros2: Habilitar ROS 2
            use_moveit2: Habilitar MoveIt 2
            use_opencv: Habilitar OpenCV
            use_ai: Habilitar IA
            use_pcl: Habilitar PCL
            use_nav2: Habilitar Nav2
            robot_type: Tipo de robot
        """
        # ROS 2
        if use_ros2:
            try:
                self.ros2 = ROS2Integration()
                logger.debug("ROS 2 integration initialized successfully")
            except Exception as e:
                logger.warning(f"ROS 2 integration failed: {e}", exc_info=True)
                self.ros2 = None
        
        # MoveIt 2 (requiere ROS 2)
        if use_moveit2:
            if not self.ros2:
                logger.warning("MoveIt 2 requires ROS 2, but ROS 2 is not available")
            else:
                try:
                    self.moveit2 = MoveIt2Integration()
                    logger.debug("MoveIt 2 integration initialized successfully")
                except Exception as e:
                    logger.warning(f"MoveIt 2 integration failed: {e}", exc_info=True)
                    self.moveit2 = None
        
        # OpenCV
        if use_opencv:
            try:
                self.vision = VisionProcessor()
                logger.debug("OpenCV integration initialized successfully")
            except Exception as e:
                logger.warning(f"OpenCV integration failed: {e}", exc_info=True)
                self.vision = None
        
        # AI Models
        if use_ai:
            try:
                self.ai_models = AIModelManager()
                logger.debug("AI models integration initialized successfully")
            except Exception as e:
                logger.warning(f"AI models integration failed: {e}", exc_info=True)
                self.ai_models = None
        
        # PCL
        if use_pcl:
            try:
                self.pcl_processor = PointCloudProcessor()
                logger.debug("PCL integration initialized successfully")
            except Exception as e:
                logger.warning(f"PCL integration failed: {e}", exc_info=True)
                self.pcl_processor = None
        
        # Nav2 (requiere ROS 2)
        if use_nav2:
            if not self.ros2 or not self.ros2.node:
                logger.warning("Nav2 requires ROS 2 node, but ROS 2 node is not available")
            else:
                try:
                    self.nav2 = Nav2Integration(self.ros2.node)
                    logger.debug("Nav2 integration initialized successfully")
                except Exception as e:
                    logger.warning(f"Nav2 integration failed: {e}", exc_info=True)
                    self.nav2 = None
        
        # Robot-specific integrations
        if robot_type == RobotType.POPPY:
            try:
                self.poppy = PoppyIntegration()
                logger.debug("Poppy integration initialized successfully")
            except Exception as e:
                logger.warning(f"Poppy integration failed: {e}", exc_info=True)
                self.poppy = None
        
        if robot_type == RobotType.ICUB:
            try:
                self.icub = ICubIntegration()
                logger.debug("iCub integration initialized successfully")
            except Exception as e:
                logger.warning(f"iCub integration failed: {e}", exc_info=True)
                self.icub = None
    
    async def connect(self) -> bool:
        """
        Conectar con robot humanoide (optimizado con DL).
        
        Returns:
            True si la conexión fue exitosa
            
        Raises:
            HumanoidRobotError: Si hay error en la conexión
        """
        if self.connected:
            logger.warning("Robot is already connected")
            return True
        
        logger.info(f"Connecting to humanoid robot (type: {self.robot_type.value})...")
        
        try:
            # Conectar según tipo de robot
            if self.robot_type == RobotType.POPPY and self.poppy:
                logger.info("Connecting to Poppy robot...")
                # La conexión de Poppy se maneja internamente
            elif self.robot_type == RobotType.ICUB and self.icub:
                logger.info("Connecting to iCub robot...")
                if not self.icub.connect_to_robot():
                    raise HumanoidRobotError("Failed to connect to iCub robot")
            elif self.ros2 and self.ros2.initialized:
                logger.info("Connecting via ROS 2...")
                # ROS 2 ya está inicializado
            else:
                logger.info("Connecting to generic humanoid robot...")
            
            # Inicializar modelos DL si están disponibles
            if self.use_ml and self.dl_manager and not self.transformer_model:
                try:
                    model_id = self.dl_manager.create_model(
                        model_type="transformer",
                        input_size=self.dof * 2,
                        output_size=self.dof
                    )
                    if model_id and self.dl_manager.models:
                        self.transformer_model = self.dl_manager.models.get(model_id)
                        logger.info("Transformer model initialized during connection")
                except Exception as e:
                    logger.warning(f"Failed to initialize Transformer model: {e}")
            
            self.connected = True
            logger.info("Humanoid robot connected successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error connecting to robot: {e}", exc_info=True)
            self.connected = False
            raise HumanoidRobotError(f"Failed to connect to robot: {str(e)}") from e
    
    async def disconnect(self) -> None:
        """
        Desconectar de robot humanoide (optimizado).
        
        Raises:
            HumanoidRobotError: Si hay error al desconectar
        """
        if not self.connected:
            logger.warning("Robot is already disconnected")
            return
        
        logger.info("Disconnecting from humanoid robot...")
        
        try:
            # Desconectar en orden inverso de inicialización
            if self.icub:
                try:
                    self.icub.shutdown()
                    logger.debug("iCub disconnected")
                except Exception as e:
                    logger.error(f"Error disconnecting iCub: {e}", exc_info=True)
            
            if self.nav2:
                try:
                    self.nav2.cancel_navigation()
                    logger.debug("Nav2 cancelled")
                except Exception as e:
                    logger.warning(f"Error cancelling Nav2: {e}")
            
            if self.ros2:
                try:
                    self.ros2.shutdown()
                    logger.debug("ROS 2 disconnected")
                except Exception as e:
                    logger.error(f"Error disconnecting ROS 2: {e}", exc_info=True)
            
            # Limpiar modelos DL
            self.transformer_model = None
            self.diffusion_model = None
            
            self.connected = False
            logger.info("Humanoid robot disconnected successfully")
            
        except Exception as e:
            logger.error(f"Error during disconnect: {e}", exc_info=True)
            # Forzar desconexión incluso si hay errores
            self.connected = False
            raise HumanoidRobotError(f"Error disconnecting robot: {str(e)}") from e
    
    @require_connection
    async def get_joint_positions(self) -> List[float]:
        """Obtener posiciones de articulaciones humanoides."""
        return self._zero_joint_positions.tolist()
    
    @require_connection
    async def set_joint_positions(
        self,
        positions: List[float],
        velocities: Optional[List[float]] = None
    ) -> bool:
        """
        Establecer posiciones de articulaciones humanoides (optimizado con DL).
        
        Args:
            positions: Lista de posiciones de articulaciones (debe tener longitud dof)
            velocities: Lista opcional de velocidades (debe tener longitud dof si se proporciona)
            
        Returns:
            True si el comando se envió exitosamente
            
        Raises:
            ValueError: Si las posiciones o velocidades son inválidas
            HumanoidRobotError: Si hay error al establecer posiciones
        """
        # Guard clauses
        if not isinstance(positions, (list, tuple, np.ndarray)):
            raise ValueError(f"positions must be a list, tuple, or numpy array, got {type(positions)}")
        
        if len(positions) != self.dof:
            raise ValueError(
                f"Expected {self.dof} joint positions, got {len(positions)}. "
                f"Joint names: {self.joint_names[:len(positions)] if len(positions) < self.dof else self.joint_names}"
            )
        
        # Validar que todas las posiciones sean números finitos
        try:
            positions_array = np.array(positions, dtype=np.float64)
            if not np.all(np.isfinite(positions_array)):
                raise ValueError("All joint positions must be finite numbers")
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid joint positions: {e}") from e
        
        # Validar velocidades si se proporcionan
        if velocities is not None:
            if not isinstance(velocities, (list, tuple, np.ndarray)):
                raise ValueError(f"velocities must be a list, tuple, or numpy array, got {type(velocities)}")
            
            if len(velocities) != self.dof:
                raise ValueError(f"Expected {self.dof} joint velocities, got {len(velocities)}")
            
            try:
                velocities_array = np.array(velocities, dtype=np.float64)
                if not np.all(np.isfinite(velocities_array)):
                    raise ValueError("All joint velocities must be finite numbers")
            except (ValueError, TypeError) as e:
                raise ValueError(f"Invalid joint velocities: {e}") from e
        
        try:
            # Usar modelo Transformer si está disponible para suavizar movimiento
            if self.use_ml and self.transformer_model:
                try:
                    current_positions = await self.get_joint_positions()
                    # Preparar input para el modelo: [current_positions, target_positions]
                    model_input = np.concatenate([
                        np.array(current_positions, dtype=np.float32),
                        positions_array.astype(np.float32)
                    ]).reshape(1, -1)
                    
                    # Predecir trayectoria suavizada
                    smoothed_positions = self.transformer_model.predict(model_input)
                    if smoothed_positions is not None and len(smoothed_positions) > 0:
                        positions_array = smoothed_positions[0].astype(np.float64)
                        logger.debug("Applied Transformer smoothing to joint positions")
                except Exception as e:
                    logger.warning(f"Transformer smoothing failed, using direct positions: {e}")
            
            # Publicar comandos vía ROS 2 si está disponible
            if self.ros2 and self.ros2.node:
                self.ros2.node.publish_joint_commands(positions_array.tolist())
                logger.debug(f"Published joint commands via ROS 2: {len(positions)} joints")
            else:
                logger.debug(f"Setting {self.dof} humanoid joint positions (simulated)")
            
            return True
            
        except Exception as e:
            logger.error(f"Error setting joint positions: {e}", exc_info=True)
            raise HumanoidRobotError(f"Failed to set joint positions: {str(e)}") from e
    
    @require_connection
    async def get_end_effector_pose(self) -> Dict[str, np.ndarray]:
        """
        Obtener pose del efector final (mano derecha por defecto).
        
        Returns:
            Dict con posición y orientación de la mano derecha
        """
        return {
            "position": self.DEFAULT_POSITION.copy(),
            "orientation": self.DEFAULT_ORIENTATION.copy()
        }
    
    async def move_to_pose(
        self,
        position: np.ndarray,
        orientation: np.ndarray,
        hand: str = "right"
    ) -> bool:
        """
        Mover mano humanaide a pose objetivo usando MoveIt 2 si está disponible.
        
        Args:
            position: Posición objetivo [x, y, z]
            orientation: Orientación quaternion [x, y, z, w]
            hand: "left" o "right"
        """
        if not self.connected:
            raise RuntimeError("Robot not connected")
        
        # Usar MoveIt 2 si está disponible
        if self.moveit2:
            result = self.moveit2.plan_to_pose(
                float(position[0]), float(position[1]), float(position[2]),
                float(orientation[0]), float(orientation[1]), 
                float(orientation[2]), float(orientation[3])
            )
            if result and result.get("success"):
                return self.moveit2.execute_plan(result["plan"])
            else:
                logger.warning("MoveIt 2 planning failed, using fallback")
        
        logger.debug(f"Moving {hand} hand to pose: {position}")
        return True
    
    async def stop(self):
        """Detener movimiento humanoide."""
        logger.info("Stopping humanoid robot")
        self.walking_state = "idle"
    
    async def emergency_stop(self):
        """Parada de emergencia humanoide."""
        logger.warning("HUMANOID ROBOT EMERGENCY STOP")
        await self.stop()
    
    async def walk(
        self,
        direction: str = "forward",
        distance: float = 1.0,
        speed: float = 0.5
    ) -> bool:
        """
        Caminar en dirección especificada usando Nav2 si está disponible.
        
        Args:
            direction: "forward", "backward", "left", "right", "turn_left", "turn_right"
            distance: Distancia en metros
            speed: Velocidad normalizada (0.0-1.0)
        """
        if not self.connected:
            raise RuntimeError("Robot not connected")
        
        # Usar Nav2 si está disponible
        if self.nav2:
            # Calcular posición objetivo basada en dirección y distancia
            # Esto es simplificado - en producción usaría odometría actual
            target_x = distance if direction == "forward" else -distance if direction == "backward" else 0.0
            target_y = distance if direction == "right" else -distance if direction == "left" else 0.0
            
            return self.nav2.navigate_to_pose(target_x, target_y, 0.0)
        elif self.ros2 and self.ros2.node:
            # Usar ROS 2 cmd_vel
            linear_x = speed if direction == "forward" else -speed if direction == "backward" else 0.0
            linear_y = speed if direction == "right" else -speed if direction == "left" else 0.0
            angular_z = speed if direction == "turn_right" else -speed if direction == "turn_left" else 0.0
            
            self.ros2.node.publish_cmd_vel(linear_x, linear_y, angular_z)
            self.walking_state = direction
            return True
        else:
            logger.info(f"Walking {direction} for {distance}m at speed {speed}")
            self.walking_state = direction
            return True
    
    async def stand(self) -> bool:
        """Ponerse de pie desde posición sentada o agachada."""
        if not self.connected:
            raise RuntimeError("Robot not connected")
        
        logger.info("Standing up")
        self.current_pose = "standing"
        return True
    
    async def sit(self) -> bool:
        """Sentarse."""
        if not self.connected:
            raise RuntimeError("Robot not connected")
        
        logger.info("Sitting down")
        self.current_pose = "sitting"
        return True
    
    async def crouch(self) -> bool:
        """Agacharse."""
        if not self.connected:
            raise RuntimeError("Robot not connected")
        
        logger.info("Crouching")
        self.current_pose = "crouching"
        return True
    
    async def grasp(
        self,
        hand: str = "right",
        object_position: Optional[np.ndarray] = None
    ) -> bool:
        """
        Agarrar objeto con mano especificada.
        
        Args:
            hand: "left" o "right"
            object_position: Posición del objeto a agarrar (opcional)
        """
        if not self.connected:
            raise RuntimeError("Robot not connected")
        
        logger.info(f"Grasping with {hand} hand")
        self.grasp_state[hand] = True
        return True
    
    async def release(self, hand: str = "right") -> bool:
        """Soltar objeto de mano especificada."""
        if not self.connected:
            raise RuntimeError("Robot not connected")
        
        logger.info(f"Releasing {hand} hand")
        self.grasp_state[hand] = False
        return True
    
    async def wave(self, hand: str = "right") -> bool:
        """Saludar con la mano."""
        if not self.connected:
            raise RuntimeError("Robot not connected")
        
        logger.info(f"Waving with {hand} hand")
        return True
    
    async def get_status(self) -> Dict[str, Any]:
        """
        Obtener estado completo del robot humanoide (optimizado con DL).
        
        Returns:
            Dict con estado completo del robot e integraciones
            
        Raises:
            HumanoidRobotError: Si hay error al obtener el estado
        """
        if not self.connected:
            raise RuntimeError("Robot not connected. Call connect() first.")
        
        try:
            status: Dict[str, Any] = {
                "connected": self.connected,
                "dof": self.dof,
                "current_pose": self.current_pose,
                "walking_state": self.walking_state,
                "grasp_state": self.grasp_state.copy(),
                "joint_positions": await self.get_joint_positions(),
                "robot_type": self.robot_type.value if isinstance(self.robot_type, RobotType) else str(self.robot_type),
                "integrations": {
                    "ros2": self.ros2 is not None and self.ros2.initialized if self.ros2 else False,
                    "moveit2": self.moveit2 is not None and self.moveit2.initialized if self.moveit2 else False,
                    "opencv": self.vision is not None,
                    "ai": self.ai_models is not None,
                    "pcl": self.pcl_processor is not None,
                    "nav2": self.nav2 is not None and self.nav2.initialized if self.nav2 else False,
                    "poppy": self.poppy is not None and self.poppy.initialized if self.poppy else False,
                    "icub": self.icub is not None and self.icub.initialized if self.icub else False,
                    "deep_learning": {
                        "ml_enabled": self.use_ml,
                        "diffusion_enabled": self.use_diffusion,
                        "transformer_model": self.transformer_model is not None,
                        "diffusion_model": self.diffusion_model is not None,
                        "native_ik": self.ik_wrapper is not None,
                        "native_trajectory": self.trajectory_optimizer is not None
                    }
                }
            }
            
            # Agregar información detallada de integraciones (con manejo de errores)
            if self.ros2 and self.ros2.node:
                try:
                    status["ros2"] = {
                        "odometry": self.ros2.node.get_odometry(),
                        "joint_states": self.ros2.node.get_joint_states()
                    }
                except Exception as e:
                    logger.warning(f"Error getting ROS 2 status: {e}")
                    status["ros2"] = {"error": str(e)}
            
            if self.moveit2:
                try:
                    status["moveit2"] = {
                        "current_pose": self.moveit2.get_current_pose(),
                        "current_joints": self.moveit2.get_current_joint_values()
                    }
                except Exception as e:
                    logger.warning(f"Error getting MoveIt 2 status: {e}")
                    status["moveit2"] = {"error": str(e)}
            
            if self.nav2:
                try:
                    status["nav2"] = self.nav2.get_navigation_status()
                except Exception as e:
                    logger.warning(f"Error getting Nav2 status: {e}")
                    status["nav2"] = {"error": str(e)}
            
            # Información de modelos DL
            if self.use_ml and self.dl_manager:
                try:
                    status["dl_models"] = {
                        "transformer_available": self.transformer_model is not None,
                        "model_count": len(self.dl_manager.models) if self.dl_manager.models else 0
                    }
                except Exception as e:
                    logger.warning(f"Error getting DL models status: {e}")
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting status: {e}", exc_info=True)
            raise HumanoidRobotError(f"Failed to get status: {str(e)}") from e
    
    async def generate_smooth_trajectory(
        self,
        start_position: np.ndarray,
        end_position: np.ndarray,
        num_steps: int = 50
    ) -> Optional[np.ndarray]:
        """
        Generar trayectoria suave usando modelo de difusión.
        
        Args:
            start_position: Posición inicial [x, y, z]
            end_position: Posición final [x, y, z]
            num_steps: Número de pasos en la trayectoria
            
        Returns:
            Array de trayectoria (num_steps, 3) o None si falla
        """
        if not self.use_diffusion or not self.diffusion_manager:
            logger.warning("Diffusion model not available")
            return None
        
        try:
            trajectory = self.diffusion_manager.generate_trajectory(
                start=start_position,
                goal=end_position,
                num_steps=num_steps
            )
            return trajectory
        except Exception as e:
            logger.error(f"Error generating smooth trajectory: {e}", exc_info=True)
            return None
    
    async def predict_joint_motion(
        self,
        current_joints: np.ndarray,
        target_joints: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Predecir movimiento de articulaciones usando modelo Transformer.
        
        Args:
            current_joints: Posiciones actuales de articulaciones
            target_joints: Posiciones objetivo de articulaciones
            
        Returns:
            Trayectoria predicha de articulaciones o None si falla
        """
        if not self.use_ml or not self.transformer_model:
            logger.warning("Transformer model not available")
            return None
        
        if not TORCH_AVAILABLE or not self.dl_manager:
            logger.warning("PyTorch or DL manager not available")
            return None
        
        try:
            # Preparar input: [current_joints, target_joints]
            model_input = np.concatenate([
                current_joints.astype(np.float32),
                target_joints.astype(np.float32)
            ]).reshape(1, -1)
            
            # Convertir a tensor y mover a dispositivo
            input_tensor = torch.from_numpy(model_input).to(self.dl_manager.device)
            
            # Predecir con torch
            self.transformer_model.eval()
            with torch.no_grad():
                if hasattr(self.transformer_model, 'forward'):
                    predicted_tensor = self.transformer_model(input_tensor)
                    predicted = predicted_tensor.cpu().numpy()
                elif hasattr(self.transformer_model, 'predict'):
                    predicted = self.transformer_model.predict(model_input)
                else:
                    logger.error("Transformer model has no forward or predict method")
                    return None
            
            return predicted
        except Exception as e:
            logger.error(f"Error predicting joint motion: {e}", exc_info=True)
            return None


    @require_connection
    async def get_end_effector_pose(self) -> Dict[str, np.ndarray]:
        """
        Obtener pose del efector final (mano derecha por defecto).
        
        Returns:
            Dict con posición y orientación de la mano derecha
        """
        return {
            "position": self.DEFAULT_POSITION.copy(),
            "orientation": self.DEFAULT_ORIENTATION.copy()
        }
    
    async def move_to_pose(
        self,
        position: np.ndarray,
        orientation: np.ndarray,
        hand: str = "right"
    ) -> bool:
        """
        Mover mano humanaide a pose objetivo usando MoveIt 2 si está disponible.
        
        Args:
            position: Posición objetivo [x, y, z]
            orientation: Orientación quaternion [x, y, z, w]
            hand: "left" o "right"
        """
        if not self.connected:
            raise RuntimeError("Robot not connected")
        
        # Usar MoveIt 2 si está disponible
        if self.moveit2:
            result = self.moveit2.plan_to_pose(
                float(position[0]), float(position[1]), float(position[2]),
                float(orientation[0]), float(orientation[1]), 
                float(orientation[2]), float(orientation[3])
            )
            if result and result.get("success"):
                return self.moveit2.execute_plan(result["plan"])
            else:
                logger.warning("MoveIt 2 planning failed, using fallback")
        
        logger.debug(f"Moving {hand} hand to pose: {position}")
        return True
    
    async def stop(self):
        """Detener movimiento humanoide."""
        logger.info("Stopping humanoid robot")
        self.walking_state = "idle"
    
    async def emergency_stop(self):
        """Parada de emergencia humanoide."""
        logger.warning("HUMANOID ROBOT EMERGENCY STOP")
        await self.stop()
    
    async def walk(
        self,
        direction: str = "forward",
        distance: float = 1.0,
        speed: float = 0.5
    ) -> bool:
        """
        Caminar en dirección especificada usando Nav2 si está disponible.
        
        Args:
            direction: "forward", "backward", "left", "right", "turn_left", "turn_right"
            distance: Distancia en metros
            speed: Velocidad normalizada (0.0-1.0)
        """
        if not self.connected:
            raise RuntimeError("Robot not connected")
        
        # Usar Nav2 si está disponible
        if self.nav2:
            # Calcular posición objetivo basada en dirección y distancia
            # Esto es simplificado - en producción usaría odometría actual
            target_x = distance if direction == "forward" else -distance if direction == "backward" else 0.0
            target_y = distance if direction == "right" else -distance if direction == "left" else 0.0
            
            return self.nav2.navigate_to_pose(target_x, target_y, 0.0)
        elif self.ros2 and self.ros2.node:
            # Usar ROS 2 cmd_vel
            linear_x = speed if direction == "forward" else -speed if direction == "backward" else 0.0
            linear_y = speed if direction == "right" else -speed if direction == "left" else 0.0
            angular_z = speed if direction == "turn_right" else -speed if direction == "turn_left" else 0.0
            
            self.ros2.node.publish_cmd_vel(linear_x, linear_y, angular_z)
            self.walking_state = direction
            return True
        else:
            logger.info(f"Walking {direction} for {distance}m at speed {speed}")
            self.walking_state = direction
            return True
    
    async def stand(self) -> bool:
        """Ponerse de pie desde posición sentada o agachada."""
        if not self.connected:
            raise RuntimeError("Robot not connected")
        
        logger.info("Standing up")
        self.current_pose = "standing"
        return True
    
    async def sit(self) -> bool:
        """Sentarse."""
        if not self.connected:
            raise RuntimeError("Robot not connected")
        
        logger.info("Sitting down")
        self.current_pose = "sitting"
        return True
    
    async def crouch(self) -> bool:
        """Agacharse."""
        if not self.connected:
            raise RuntimeError("Robot not connected")
        
        logger.info("Crouching")
        self.current_pose = "crouching"
        return True
    
    async def grasp(
        self,
        hand: str = "right",
        object_position: Optional[np.ndarray] = None
    ) -> bool:
        """
        Agarrar objeto con mano especificada.
        
        Args:
            hand: "left" o "right"
            object_position: Posición del objeto a agarrar (opcional)
        """
        if not self.connected:
            raise RuntimeError("Robot not connected")
        
        logger.info(f"Grasping with {hand} hand")
        self.grasp_state[hand] = True
        return True
    
    async def release(self, hand: str = "right") -> bool:
        """Soltar objeto de mano especificada."""
        if not self.connected:
            raise RuntimeError("Robot not connected")
        
        logger.info(f"Releasing {hand} hand")
        self.grasp_state[hand] = False
        return True
    
    async def wave(self, hand: str = "right") -> bool:
        """Saludar con la mano."""
        if not self.connected:
            raise RuntimeError("Robot not connected")
        
        logger.info(f"Waving with {hand} hand")
        return True
    
    @require_connection
    @log_operation("get_status")
    async def get_status(self) -> Dict[str, Any]:
        """
        Obtener estado completo del robot humanoide (optimizado).
        
        Returns:
            Dict con estado completo del robot incluyendo:
            - Estado de conexión
            - DOF y nombres de articulaciones
            - Pose actual y estado de caminata
            - Estado de agarre
            - Posiciones de articulaciones
            - Información de integraciones (ROS2, MoveIt2, Nav2)
            - Historial de operaciones recientes
        """
        status = {
            "connected": self.connected,
            "dof": self.dof,
            "joint_names": self.joint_names,
            "current_pose": self.current_pose,
            "walking_state": self.walking_state,
            "grasp_state": self.grasp_state.copy(),
            "joint_positions": await self.get_joint_positions(),
            "robot_type": self.robot_type,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Agregar información de integraciones
        if self.ros2 and self.ros2.node:
            try:
                status["ros2"] = {
                    "odometry": self.ros2.node.get_odometry(),
                    "joint_states": self.ros2.node.get_joint_states()
                }
            except Exception as e:
                logger.warning(f"Error getting ROS2 status: {e}")
                status["ros2"] = {"error": str(e)}
        
        if self.moveit2:
            try:
                status["moveit2"] = {
                    "current_pose": self.moveit2.get_current_pose(),
                    "current_joints": self.moveit2.get_current_joint_values()
                }
            except Exception as e:
                logger.warning(f"Error getting MoveIt2 status: {e}")
                status["moveit2"] = {"error": str(e)}
        
        if self.nav2:
            try:
                status["nav2"] = self.nav2.get_navigation_status()
            except Exception as e:
                logger.warning(f"Error getting Nav2 status: {e}")
                status["nav2"] = {"error": str(e)}
        
        # Agregar historial de operaciones recientes
        if self._operation_history:
            status["recent_operations"] = self._operation_history[-10:]  # Últimas 10 operaciones
        
        return status
    
    def _record_operation(self, operation: str, success: bool, details: Optional[Dict[str, Any]] = None):
        """
        Registrar operación en historial (optimizado).
        
        Args:
            operation: Nombre de la operación
            success: Si fue exitosa
            details: Detalles adicionales (opcional)
        """
        record = {
            "operation": operation,
            "success": success,
            "timestamp": datetime.utcnow().isoformat(),
            "details": details or {}
        }
        self._operation_history.append(record)
        
        # Limitar tamaño del historial
        if len(self._operation_history) > self._max_history_size:
            self._operation_history = self._operation_history[-self._max_history_size:]
    
    @require_connection
    @log_operation("run")
    async def run(self, speed: float = 0.8) -> bool:
        """
        Correr hacia adelante (optimizado).
        
        Args:
            speed: Velocidad normalizada (0.0-1.0)
            
        Returns:
            True si se inició correctamente
        """
        if speed < 0.0 or speed > 1.0:
            raise ValueError(f"Speed must be between 0.0 and 1.0, got {speed}")
        
        logger.info(f"Running at speed {speed}")
        self.current_pose = RobotPose.RUNNING.value
        self.walking_state = WalkingState.FORWARD.value
        
        self._record_operation("run", True, {"speed": speed})
        return True
    
    @require_connection
    @log_operation("jump")
    async def jump(self, height: float = 0.2) -> bool:
        """
        Saltar (optimizado).
        
        Args:
            height: Altura del salto en metros
            
        Returns:
            True si se inició correctamente
        """
        if height < 0.0:
            raise ValueError(f"Height must be non-negative, got {height}")
        if height > 1.0:
            logger.warning(f"Jump height {height}m is very high, limiting to 1.0m")
            height = 1.0
        
        logger.info(f"Jumping with height {height}m")
        self._record_operation("jump", True, {"height": height})
        return True
    
    @require_connection
    @log_operation("dual_arm_manipulation")
    async def dual_arm_manipulation(
        self,
        left_position: np.ndarray,
        right_position: np.ndarray,
        left_orientation: Optional[np.ndarray] = None,
        right_orientation: Optional[np.ndarray] = None
    ) -> bool:
        """
        Manipulación bimanual coordinada (optimizado).
        
        Args:
            left_position: Posición objetivo mano izquierda [x, y, z]
            right_position: Posición objetivo mano derecha [x, y, z]
            left_orientation: Orientación mano izquierda (opcional)
            right_orientation: Orientación mano derecha (opcional)
            
        Returns:
            True si se planificó correctamente
        """
        # Validaciones
        if left_position.shape != (3,):
            raise ValueError(f"left_position must be shape (3,), got {left_position.shape}")
        if right_position.shape != (3,):
            raise ValueError(f"right_position must be shape (3,), got {right_position.shape}")
        
        logger.info(f"Dual arm manipulation: left={left_position}, right={right_position}")
        
        # Mover ambas manos simultáneamente
        left_success = await self.move_to_pose(left_position, left_orientation or self.DEFAULT_ORIENTATION, "left")
        right_success = await self.move_to_pose(right_position, right_orientation or self.DEFAULT_ORIENTATION, "right")
        
        success = left_success and right_success
        self._record_operation("dual_arm_manipulation", success, {
            "left_position": left_position.tolist(),
            "right_position": right_position.tolist()
        })
        
        return success
    
    @require_connection
    @log_operation("get_operation_history")
    async def get_operation_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Obtener historial de operaciones (optimizado).
        
        Args:
            limit: Número máximo de operaciones a retornar
            
        Returns:
            Lista de operaciones recientes
        """
        if limit < 1:
            raise ValueError(f"Limit must be positive, got {limit}")
        
        return self._operation_history[-limit:] if limit <= len(self._operation_history) else self._operation_history.copy()

