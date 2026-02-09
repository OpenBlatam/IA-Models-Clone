"""
Adaptive Models for Different Robot Types
==========================================

Modelos de deep learning adaptativos según el tipo de robot.
"""

import logging
from typing import Dict, Any, Optional

from .robot_types import RobotType
from .dl_models import ModelFactory, ModelType
from .dl_models.base_model import BaseRobotModel

logger = logging.getLogger(__name__)


class AdaptiveModelManager:
    """
    Gestor de modelos adaptativos.
    
    Crea modelos de deep learning adaptados al tipo de robot.
    """
    
    def __init__(self):
        """Inicializar gestor."""
        self.models: Dict[str, BaseRobotModel] = {}
        self.robot_type_configs = {
            RobotType.MANIPULATOR: {
                "input_size": 6,  # x, y, z, vx, vy, vz
                "output_size": 7,  # 7 DOF típico
                "hidden_sizes": [128, 64, 32]
            },
            RobotType.MOBILE: {
                "input_size": 3,  # x, y, theta
                "output_size": 3,  # vx, vy, omega
                "hidden_sizes": [64, 32]
            },
            RobotType.QUADCOPTER: {
                "input_size": 6,  # x, y, z, roll, pitch, yaw
                "output_size": 4,  # 4 thrusts
                "hidden_sizes": [128, 64]
            },
            RobotType.HUMANOID: {
                "input_size": 12,  # posición base + orientación + velocidades
                "output_size": 20,  # ~20 DOF típico
                "hidden_sizes": [256, 128, 64]
            },
            RobotType.WHEELED: {
                "input_size": 3,  # x, y, theta
                "output_size": 2,  # left_wheel, right_wheel
                "hidden_sizes": [64, 32]
            },
            RobotType.LEGGED: {
                "input_size": 6,  # posición base + orientación
                "output_size": 12,  # 12 DOF típico (4 patas x 3 DOF)
                "hidden_sizes": [128, 64, 32]
            }
        }
    
    def create_model_for_robot(
        self,
        robot_type: RobotType,
        robot_id: str,
        custom_config: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Crear modelo adaptado al tipo de robot.
        
        Args:
            robot_type: Tipo de robot
            robot_id: ID del robot
            custom_config: Configuración personalizada (opcional)
            
        Returns:
            ID del modelo
        """
        # Obtener configuración base
        if robot_type not in self.robot_type_configs:
            logger.warning(f"Unknown robot type: {robot_type}, using default config")
            config = {
                "input_size": 6,
                "output_size": 6,
                "hidden_sizes": [128, 64, 32]
            }
        else:
            config = self.robot_type_configs[robot_type].copy()
        
        # Aplicar configuración personalizada
        if custom_config:
            config.update(custom_config)
        
        # Crear modelo
        model = ModelFactory.create_model(
            ModelType.TRAJECTORY_PREDICTOR,
            config
        )
        
        model_id = f"{robot_id}_{robot_type.value}"
        self.models[model_id] = model
        
        logger.info(f"Created adaptive model for {robot_type.value}: {model_id}")
        
        return model_id
    
    def get_model(self, model_id: str) -> Optional[BaseRobotModel]:
        """Obtener modelo."""
        return self.models.get(model_id)
    
    def get_model_config(self, robot_type: RobotType) -> Dict[str, Any]:
        """Obtener configuración recomendada para tipo de robot."""
        return self.robot_type_configs.get(robot_type, {
            "input_size": 6,
            "output_size": 6,
            "hidden_sizes": [128, 64, 32]
        })


# Instancia global
_adaptive_model_manager: Optional[AdaptiveModelManager] = None


def get_adaptive_model_manager() -> AdaptiveModelManager:
    """Obtener instancia global del gestor de modelos adaptativos."""
    global _adaptive_model_manager
    if _adaptive_model_manager is None:
        _adaptive_model_manager = AdaptiveModelManager()
    return _adaptive_model_manager

