"""
ROS 2 Integration for Humanoid Robot (Optimizado)
==================================================

Integración completa con ROS 2 (Robot Operating System) para robot humanoide.
Incluye validaciones robustas, manejo de errores mejorado, y optimizaciones.
"""

import logging
from typing import Optional, Dict, Any, List, Union
import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import Twist, Pose, PoseStamped, Point, Quaternion
    from sensor_msgs.msg import JointState, Image, PointCloud2
    from std_msgs.msg import String, Bool, Float32MultiArray
    from nav_msgs.msg import Odometry, Path
    from tf2_ros import TransformBroadcaster, Buffer, TransformListener
    from geometry_msgs.msg import TransformStamped
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    logging.warning("ROS 2 not available. Install rclpy and related packages.")

logger = logging.getLogger(__name__)


class ROS2IntegrationError(Exception):
    """Excepción personalizada para errores de integración ROS 2."""
    pass


class HumanoidROS2Node(Node):
    """
    Nodo ROS 2 para robot humanoide.
    
    Publica y suscribe a topics estándar de ROS 2 para control humanoide.
    """
    
    def __init__(self, node_name: str = "humanoid_devin_robot"):
        """
        Inicializar nodo ROS 2 (optimizado).
        
        Args:
            node_name: Nombre del nodo ROS 2
            
        Raises:
            ImportError: Si ROS 2 no está disponible
            ROS2IntegrationError: Si hay error en la inicialización
        """
        if not ROS2_AVAILABLE:
            raise ImportError("ROS 2 is not available. Please install rclpy.")
        
        # Guard clauses
        if not node_name or not isinstance(node_name, str) or not node_name.strip():
            raise ValueError("node_name must be a non-empty string")
        
        try:
            super().__init__(node_name.strip())
        except Exception as e:
            logger.error(f"Error initializing ROS 2 node: {e}", exc_info=True)
            raise ROS2IntegrationError(f"Failed to initialize ROS 2 node: {str(e)}") from e
        
        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.joint_cmd_pub = self.create_publisher(Float32MultiArray, '/joint_commands', 10)
        self.pose_goal_pub = self.create_publisher(PoseStamped, '/move_base_simple/goal', 10)
        
        # Subscribers
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self._joint_state_callback,
            10
        )
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self._odom_callback,
            10
        )
        
        # Services (simplified - would need service definitions)
        # self.move_service = self.create_service(...)
        
        # TF Broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # State
        self.current_joint_states: Optional[JointState] = None
        self.current_odom: Optional[Odometry] = None
        
        logger.info(f"ROS 2 node '{node_name}' initialized")
    
    def _joint_state_callback(self, msg: JointState):
        """Callback para estados de articulaciones."""
        self.current_joint_states = msg
        logger.debug(f"Received joint states: {len(msg.name)} joints")
    
    def _odom_callback(self, msg: Odometry):
        """Callback para odometría."""
        self.current_odom = msg
        logger.debug("Received odometry update")
    
    def publish_cmd_vel(
        self, 
        linear_x: float = 0.0, 
        linear_y: float = 0.0, 
        angular_z: float = 0.0
    ) -> None:
        """
        Publicar comando de velocidad (para caminar) (optimizado).
        
        Args:
            linear_x: Velocidad lineal hacia adelante (m/s)
            linear_y: Velocidad lineal lateral (m/s)
            angular_z: Velocidad angular (rad/s)
            
        Raises:
            ROS2IntegrationError: Si hay error al publicar
        """
        # Validar que los valores sean números finitos
        try:
            linear_x = float(linear_x)
            linear_y = float(linear_y)
            angular_z = float(angular_z)
            
            if not all(np.isfinite([linear_x, linear_y, angular_z])):
                raise ValueError("All velocity values must be finite numbers")
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid velocity values: {e}") from e
        
        try:
            twist = Twist()
            twist.linear.x = linear_x
            twist.linear.y = linear_y
            twist.angular.z = angular_z
            
            self.cmd_vel_pub.publish(twist)
            logger.debug(f"Published cmd_vel: linear=({linear_x:.3f}, {linear_y:.3f}), angular={angular_z:.3f}")
        except Exception as e:
            logger.error(f"Error publishing cmd_vel: {e}", exc_info=True)
            raise ROS2IntegrationError(f"Failed to publish cmd_vel: {str(e)}") from e
    
    def publish_joint_commands(self, joint_positions: Union[List[float], np.ndarray]) -> None:
        """
        Publicar comandos de articulaciones (optimizado).
        
        Args:
            joint_positions: Lista o array de posiciones de articulaciones
            
        Raises:
            ValueError: Si las posiciones son inválidas
            ROS2IntegrationError: Si hay error al publicar
        """
        # Guard clauses
        if not isinstance(joint_positions, (list, tuple, np.ndarray)):
            raise ValueError(f"joint_positions must be a list, tuple, or numpy array, got {type(joint_positions)}")
        
        if len(joint_positions) == 0:
            raise ValueError("joint_positions cannot be empty")
        
        # Validar y convertir
        try:
            positions_array = np.array(joint_positions, dtype=np.float32)
            if not np.all(np.isfinite(positions_array)):
                raise ValueError("All joint positions must be finite numbers")
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid joint positions: {e}") from e
        
        try:
            msg = Float32MultiArray()
            msg.data = positions_array.tolist()
            self.joint_cmd_pub.publish(msg)
            logger.debug(f"Published joint commands: {len(joint_positions)} joints")
        except Exception as e:
            logger.error(f"Error publishing joint commands: {e}", exc_info=True)
            raise ROS2IntegrationError(f"Failed to publish joint commands: {str(e)}") from e
    
    def publish_pose_goal(
        self, 
        x: float, 
        y: float, 
        z: float,
        qx: float = 0.0, 
        qy: float = 0.0, 
        qz: float = 0.0, 
        qw: float = 1.0,
        frame_id: str = "map"
    ) -> None:
        """
        Publicar goal de pose para navegación (optimizado).
        
        Args:
            x, y, z: Posición (m)
            qx, qy, qz, qw: Orientación quaternion
            frame_id: Frame ID para la pose (default: "map")
            
        Raises:
            ValueError: Si los parámetros son inválidos
            ROS2IntegrationError: Si hay error al publicar
        """
        # Validar parámetros
        try:
            x, y, z = float(x), float(y), float(z)
            qx, qy, qz, qw = float(qx), float(qy), float(qz), float(qw)
            
            if not all(np.isfinite([x, y, z, qx, qy, qz, qw])):
                raise ValueError("All pose values must be finite numbers")
            
            # Normalizar quaternion
            quat_norm = np.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
            if quat_norm < 1e-6:
                raise ValueError("Quaternion norm is too small (near zero)")
            qx, qy, qz, qw = qx/quat_norm, qy/quat_norm, qz/quat_norm, qw/quat_norm
            
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid pose parameters: {e}") from e
        
        if not isinstance(frame_id, str) or not frame_id.strip():
            raise ValueError("frame_id must be a non-empty string")
        
        try:
            pose_stamped = PoseStamped()
            pose_stamped.header.frame_id = frame_id.strip()
            pose_stamped.header.stamp = self.get_clock().now().to_msg()
            
            pose_stamped.pose.position.x = x
            pose_stamped.pose.position.y = y
            pose_stamped.pose.position.z = z
            pose_stamped.pose.orientation.x = qx
            pose_stamped.pose.orientation.y = qy
            pose_stamped.pose.orientation.z = qz
            pose_stamped.pose.orientation.w = qw
            
            self.pose_goal_pub.publish(pose_stamped)
            logger.info(f"Published pose goal: ({x:.3f}, {y:.3f}, {z:.3f}) in frame '{frame_id}'")
        except Exception as e:
            logger.error(f"Error publishing pose goal: {e}", exc_info=True)
            raise ROS2IntegrationError(f"Failed to publish pose goal: {str(e)}") from e
    
    def get_joint_states(self) -> Optional[Dict[str, float]]:
        """Obtener estados actuales de articulaciones."""
        if self.current_joint_states is None:
            return None
        
        return {
            name: pos for name, pos in zip(
                self.current_joint_states.name,
                self.current_joint_states.position
            )
        }
    
    def get_odometry(self) -> Optional[Dict[str, Any]]:
        """Obtener odometría actual."""
        if self.current_odom is None:
            return None
        
        pos = self.current_odom.pose.pose.position
        orient = self.current_odom.pose.pose.orientation
        
        return {
            "position": {"x": pos.x, "y": pos.y, "z": pos.z},
            "orientation": {"x": orient.x, "y": orient.y, "z": orient.z, "w": orient.w},
            "linear_velocity": {
                "x": self.current_odom.twist.twist.linear.x,
                "y": self.current_odom.twist.twist.linear.y,
                "z": self.current_odom.twist.twist.linear.z
            },
            "angular_velocity": {
                "z": self.current_odom.twist.twist.angular.z
            }
        }


class ROS2Integration:
    """
    Wrapper para integración ROS 2 (optimizado).
    
    Gestiona el ciclo de vida completo de la integración ROS 2.
    """
    
    def __init__(self, node_name: str = "humanoid_devin_robot"):
        """
        Inicializar integración ROS 2 (optimizado).
        
        Args:
            node_name: Nombre del nodo ROS 2
            
        Raises:
            ROS2IntegrationError: Si hay error en la inicialización
        """
        if not ROS2_AVAILABLE:
            logger.warning("ROS 2 not available. ROS 2 features will be disabled.")
            self.node = None
            self.initialized = False
            return
        
        # Guard clauses
        if not node_name or not isinstance(node_name, str) or not node_name.strip():
            raise ValueError("node_name must be a non-empty string")
        
        try:
            if not rclpy.ok():
                rclpy.init()
            
            self.node = HumanoidROS2Node(node_name.strip())
            self.initialized = True
            logger.info(f"ROS 2 integration initialized with node: {node_name}")
        except Exception as e:
            logger.error(f"Error initializing ROS 2 integration: {e}", exc_info=True)
            self.node = None
            self.initialized = False
            raise ROS2IntegrationError(f"Failed to initialize ROS 2 integration: {str(e)}") from e
    
    def spin_once(self, timeout_sec: float = 0.1) -> None:
        """
        Ejecutar un ciclo de ROS 2 (optimizado).
        
        Args:
            timeout_sec: Tiempo de espera en segundos
        """
        if not self.initialized or not self.node:
            logger.warning("ROS 2 not initialized, cannot spin")
            return
        
        try:
            rclpy.spin_once(self.node, timeout_sec=timeout_sec)
        except Exception as e:
            logger.error(f"Error in ROS 2 spin_once: {e}", exc_info=True)
    
    def shutdown(self) -> None:
        """
        Apagar integración ROS 2 (optimizado).
        
        Raises:
            ROS2IntegrationError: Si hay error al apagar
        """
        if not self.initialized:
            logger.warning("ROS 2 not initialized, nothing to shutdown")
            return
        
        try:
            if self.node:
                self.node.destroy_node()
                self.node = None
            
            if rclpy.ok():
                rclpy.shutdown()
            
            self.initialized = False
            logger.info("ROS 2 integration shut down successfully")
        except Exception as e:
            logger.error(f"Error shutting down ROS 2: {e}", exc_info=True)
            self.initialized = False
            raise ROS2IntegrationError(f"Failed to shutdown ROS 2: {str(e)}") from e

