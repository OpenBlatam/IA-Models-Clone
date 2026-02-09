"""
Nav2 Integration for Humanoid Robot (Optimizado)
=================================================

Integración profesional con Nav2 (ROS 2 Navigation Stack).
Incluye validaciones robustas, manejo de errores mejorado, y optimizaciones.
"""

import logging
from typing import Optional, Dict, Any, Union
import numpy as np

try:
    from nav2_simple_commander import BasicNavigator
    from geometry_msgs.msg import PoseStamped
    NAV2_AVAILABLE = True
except ImportError:
    NAV2_AVAILABLE = False
    logging.warning("Nav2 not available. Install nav2 packages.")

logger = logging.getLogger(__name__)


class Nav2Error(Exception):
    """Excepción personalizada para errores de Nav2."""
    pass


class Nav2Integration:
    """
    Integración con Nav2 para navegación humanoide.
    
    Proporciona navegación autónoma y planificación de rutas.
    """
    
    def __init__(self, ros2_node=None):
        """
        Inicializar integración Nav2 (optimizado).
        
        Args:
            ros2_node: Nodo ROS 2 opcional (requerido para algunas funcionalidades)
            
        Raises:
            Nav2Error: Si hay error en la inicialización
        """
        if not NAV2_AVAILABLE:
            logger.warning("Nav2 not available. Navigation features will be disabled.")
            self.navigator = None
            self.initialized = False
            self.ros2_node = ros2_node
            return
        
        try:
            self.navigator = BasicNavigator()
            self.ros2_node = ros2_node
            self.initialized = True
            logger.info("Nav2 integration initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Nav2: {e}", exc_info=True)
            self.navigator = None
            self.initialized = False
            self.ros2_node = ros2_node
            raise Nav2Error(f"Failed to initialize Nav2: {str(e)}") from e
    
    def navigate_to_pose(
        self, 
        x: float, 
        y: float, 
        yaw: float = 0.0,
        frame_id: str = "map"
    ) -> bool:
        """
        Navegar a pose objetivo (optimizado).
        
        Args:
            x, y: Posición objetivo (metros)
            yaw: Orientación en radianes (-π a π)
            frame_id: Frame de referencia (default: "map")
            
        Returns:
            True si se inició la navegación exitosamente
            
        Raises:
            ValueError: Si los parámetros son inválidos
            Nav2Error: Si hay error en la navegación
        """
        if not self.initialized or self.navigator is None:
            logger.warning("Nav2 not initialized")
            return False
        
        # Validar parámetros
        try:
            x, y, yaw = float(x), float(y), float(yaw)
            
            if not all(np.isfinite([x, y, yaw])):
                raise ValueError("All pose values must be finite numbers")
            
            # Normalizar yaw a [-π, π]
            yaw = ((yaw + np.pi) % (2 * np.pi)) - np.pi
            
            if not frame_id or not isinstance(frame_id, str) or not frame_id.strip():
                raise ValueError("frame_id must be a non-empty string")
            
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid navigation parameters: {e}") from e
        
        try:
            goal_pose = PoseStamped()
            goal_pose.header.frame_id = frame_id.strip()
            goal_pose.pose.position.x = x
            goal_pose.pose.position.y = y
            goal_pose.pose.orientation.z = np.sin(yaw / 2.0)
            goal_pose.pose.orientation.w = np.cos(yaw / 2.0)
            
            self.navigator.goToPose(goal_pose)
            logger.info(f"Navigating to pose: ({x:.3f}, {y:.3f}, {yaw:.3f}) in frame '{frame_id}'")
            return True
        except Exception as e:
            logger.error(f"Error navigating to pose: {e}", exc_info=True)
            raise Nav2Error(f"Failed to navigate to pose: {str(e)}") from e
    
    def cancel_navigation(self) -> bool:
        """
        Cancelar navegación actual (optimizado).
        
        Returns:
            True si se canceló exitosamente
        """
        if not self.initialized or self.navigator is None:
            return False
        
        try:
            self.navigator.cancelTask()
            logger.info("Navigation cancelled")
            return True
        except Exception as e:
            logger.error(f"Error cancelling navigation: {e}", exc_info=True)
            return False
    
    def get_navigation_status(self) -> Dict[str, Any]:
        """
        Obtener estado de navegación (optimizado).
        
        Returns:
            Dict con estado de navegación
        """
        if not self.initialized or self.navigator is None:
            return {"status": "unavailable", "initialized": False}
        
        try:
            is_complete = self.navigator.isTaskComplete()
            return {
                "status": "active" if not is_complete else "complete",
                "is_nav_complete": is_complete,
                "initialized": True
            }
        except Exception as e:
            logger.error(f"Error getting navigation status: {e}", exc_info=True)
            return {"status": "error", "error": str(e), "initialized": True}
