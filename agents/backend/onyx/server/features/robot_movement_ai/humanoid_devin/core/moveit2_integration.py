"""
MoveIt 2 Integration for Humanoid Robot (Optimizado)
=====================================================

Integración profesional con MoveIt 2 para planificación de movimiento humanoide.
Incluye validaciones robustas, manejo de errores mejorado, y optimizaciones.
"""

import logging
from typing import Optional, Dict, Any, List, Union, Tuple
import numpy as np

try:
    from moveit_commander import MoveGroupCommander, RobotCommander, PlanningSceneInterface
    from moveit_msgs.msg import MoveItErrorCodes
    from geometry_msgs.msg import Pose, PoseStamped, Quaternion
    MOVEIT_AVAILABLE = True
except ImportError:
    MOVEIT_AVAILABLE = False
    logging.warning("MoveIt 2 not available. Install moveit-commander.")

logger = logging.getLogger(__name__)


class MoveIt2Error(Exception):
    """Excepción personalizada para errores de MoveIt 2."""
    pass


class MoveIt2Integration:
    """
    Integración con MoveIt 2 para planificación de movimiento.
    
    Proporciona planificación de trayectorias para brazos y cuerpo humanoide.
    """
    
    def __init__(
        self, 
        group_name: str = "arm", 
        planning_time: float = 5.0,
        num_planning_attempts: int = 10,
        goal_tolerance: float = 0.01
    ):
        """
        Inicializar integración MoveIt 2 (optimizado).
        
        Args:
            group_name: Nombre del grupo de articulaciones (no puede estar vacío)
            planning_time: Tiempo máximo de planificación en segundos (0.1-60.0)
            num_planning_attempts: Número de intentos de planificación (1-100)
            goal_tolerance: Tolerancia del goal en metros (0.001-1.0)
            
        Raises:
            ImportError: Si MoveIt 2 no está disponible
            ValueError: Si los parámetros son inválidos
            MoveIt2Error: Si hay error en la inicialización
        """
        if not MOVEIT_AVAILABLE:
            raise ImportError("MoveIt 2 is not available. Please install moveit-commander.")
        
        # Guard clauses
        if not group_name or not isinstance(group_name, str) or not group_name.strip():
            raise ValueError("group_name must be a non-empty string")
        
        if not isinstance(planning_time, (int, float)) or not (0.1 <= planning_time <= 60.0):
            raise ValueError(f"planning_time must be between 0.1 and 60.0 seconds, got {planning_time}")
        
        if not isinstance(num_planning_attempts, int) or not (1 <= num_planning_attempts <= 100):
            raise ValueError(f"num_planning_attempts must be between 1 and 100, got {num_planning_attempts}")
        
        if not isinstance(goal_tolerance, (int, float)) or not (0.001 <= goal_tolerance <= 1.0):
            raise ValueError(f"goal_tolerance must be between 0.001 and 1.0, got {goal_tolerance}")
        
        try:
            self.robot = RobotCommander()
            self.move_group = MoveGroupCommander(group_name.strip())
            self.scene = PlanningSceneInterface()
            
            self.move_group.set_planning_time(float(planning_time))
            self.move_group.set_num_planning_attempts(num_planning_attempts)
            self.move_group.set_goal_tolerance(goal_tolerance)
            
            self.initialized = True
            logger.info(
                f"MoveIt 2 integration initialized for group: {group_name}, "
                f"planning_time={planning_time}s, attempts={num_planning_attempts}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize MoveIt 2: {e}", exc_info=True)
            self.move_group = None
            self.robot = None
            self.scene = None
            self.initialized = False
            raise MoveIt2Error(f"Failed to initialize MoveIt 2: {str(e)}") from e
    
    def plan_to_pose(
        self,
        x: float,
        y: float,
        z: float,
        qx: float = 0.0,
        qy: float = 0.0,
        qz: float = 0.0,
        qw: float = 1.0
    ) -> Optional[Dict[str, Any]]:
        """
        Planificar movimiento a pose objetivo (optimizado).
        
        Args:
            x, y, z: Posición objetivo (metros)
            qx, qy, qz, qw: Orientación quaternion
            
        Returns:
            Dict con plan y éxito, o None si falla
            
        Raises:
            MoveIt2Error: Si hay error en la planificación
        """
        if not self.initialized or self.move_group is None:
            logger.warning("MoveIt 2 not initialized")
            return None
        
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
        
        try:
            self.move_group.set_pose_target([x, y, z, qx, qy, qz, qw])
            plan = self.move_group.plan()
            
            if plan[0]:
                return {
                    "success": True,
                    "plan": plan[1],
                    "planning_time": plan[2] if len(plan) > 2 else None,
                    "trajectory_points": len(plan[1].joint_trajectory.points) if plan[1] else 0
                }
            else:
                logger.warning("MoveIt 2 planning failed")
                return {"success": False, "plan": None, "error": "Planning failed"}
        except Exception as e:
            logger.error(f"Error in MoveIt 2 planning: {e}", exc_info=True)
            raise MoveIt2Error(f"Failed to plan to pose: {str(e)}") from e
    
    def execute_plan(self, plan: Any, wait: bool = True) -> bool:
        """
        Ejecutar plan de movimiento (optimizado).
        
        Args:
            plan: Plan de movimiento de MoveIt
            wait: Esperar a que termine la ejecución
            
        Returns:
            True si se ejecutó exitosamente
            
        Raises:
            MoveIt2Error: Si hay error en la ejecución
        """
        if not self.initialized or self.move_group is None:
            logger.warning("MoveIt 2 not initialized")
            return False
        
        if plan is None:
            raise ValueError("Plan cannot be None")
        
        try:
            success = self.move_group.execute(plan, wait=wait)
            if success:
                logger.info("MoveIt 2 plan executed successfully")
            else:
                logger.warning("MoveIt 2 plan execution returned False")
            return success == MoveItErrorCodes.SUCCESS if hasattr(MoveItErrorCodes, 'SUCCESS') else bool(success)
        except Exception as e:
            logger.error(f"Error executing MoveIt 2 plan: {e}", exc_info=True)
            raise MoveIt2Error(f"Failed to execute plan: {str(e)}") from e
    
    def get_current_pose(self) -> Optional[Dict[str, float]]:
        """Obtener pose actual del efector final."""
        if not self.initialized or self.move_group is None:
            return None
        
        try:
            pose = self.move_group.get_current_pose().pose
            return {
                "position": {"x": pose.position.x, "y": pose.position.y, "z": pose.position.z},
                "orientation": {
                    "x": pose.orientation.x,
                    "y": pose.orientation.y,
                    "z": pose.orientation.z,
                    "w": pose.orientation.w
                }
            }
        except Exception as e:
            logger.error(f"Error getting current pose: {e}")
            return None
    
    def get_current_joint_values(self) -> Optional[List[float]]:
        """Obtener valores actuales de articulaciones."""
        if not self.initialized or self.move_group is None:
            return None
        
        try:
            return list(self.move_group.get_current_joint_values())
        except Exception as e:
            logger.error(f"Error getting joint values: {e}")
            return None
