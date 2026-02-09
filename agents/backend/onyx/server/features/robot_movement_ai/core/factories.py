"""
Factory Patterns
================

Patrones de factory para crear instancias de componentes.
"""

from typing import Optional, Dict, Any, Type
import logging

from .robot import TrajectoryOptimizer, RobotMovementEngine
from .optimization.trajectory_optimizer import OptimizationParams
from .constants import OptimizationAlgorithm
from ..config.robot_config import RobotConfig, RobotBrand

logger = logging.getLogger(__name__)


class TrajectoryOptimizerFactory:
    """Factory para crear optimizadores de trayectoria."""
    
    @staticmethod
    def create(
        algorithm: OptimizationAlgorithm = OptimizationAlgorithm.PPO,
        optimization_params: Optional[OptimizationParams] = None,
        model_path: Optional[str] = None,
        **kwargs
    ) -> TrajectoryOptimizer:
        """
        Crear optimizador de trayectoria.
        
        Args:
            algorithm: Algoritmo a usar
            optimization_params: Parámetros de optimización
            model_path: Ruta al modelo RL
            **kwargs: Parámetros adicionales
            
        Returns:
            TrajectoryOptimizer configurado
        """
        optimizer = TrajectoryOptimizer(
            optimization_params=optimization_params,
            model_path=model_path
        )
        optimizer.algorithm = algorithm
        
        # Configurar parámetros adicionales
        if 'learning_rate' in kwargs:
            optimizer.learning_rate = kwargs['learning_rate']
        if 'epsilon' in kwargs:
            optimizer.epsilon = kwargs['epsilon']
        
        logger.info(f"Created TrajectoryOptimizer with algorithm: {algorithm.value}")
        return optimizer
    
    @staticmethod
    def create_fast() -> TrajectoryOptimizer:
        """Crear optimizador rápido (heurístico)."""
        params = OptimizationParams(
            max_iterations=50,
            convergence_threshold=1e-5
        )
        return TrajectoryOptimizerFactory.create(
            algorithm=OptimizationAlgorithm.HEURISTIC,
            optimization_params=params
        )
    
    @staticmethod
    def create_precise() -> TrajectoryOptimizer:
        """Crear optimizador preciso (más iteraciones)."""
        params = OptimizationParams(
            max_iterations=200,
            convergence_threshold=1e-7
        )
        return TrajectoryOptimizerFactory.create(
            algorithm=OptimizationAlgorithm.PPO,
            optimization_params=params
        )


class MovementEngineFactory:
    """Factory para crear motores de movimiento."""
    
    @staticmethod
    def create(
        robot_brand: RobotBrand = RobotBrand.GENERIC,
        **config_kwargs
    ) -> RobotMovementEngine:
        """
        Crear motor de movimiento.
        
        Args:
            robot_brand: Marca del robot
            **config_kwargs: Parámetros adicionales de configuración
            
        Returns:
            RobotMovementEngine configurado
        """
        config = RobotConfig(robot_brand=robot_brand, **config_kwargs)
        engine = RobotMovementEngine(config)
        logger.info(f"Created RobotMovementEngine for {robot_brand.value}")
        return engine
    
    @staticmethod
    def create_simulation() -> RobotMovementEngine:
        """Crear motor para simulación (sin ROS)."""
        return MovementEngineFactory.create(
            robot_brand=RobotBrand.GENERIC,
            ros_enabled=False,
            camera_enabled=False
        )
    
    @staticmethod
    def create_for_brand(robot_brand: RobotBrand) -> RobotMovementEngine:
        """Crear motor para marca específica."""
        return MovementEngineFactory.create(robot_brand=robot_brand)


class ComponentBuilder:
    """
    Builder pattern para construir componentes complejos.
    """
    
    def __init__(self):
        """Inicializar builder."""
        self.config: Optional[RobotConfig] = None
        self.optimizer: Optional[TrajectoryOptimizer] = None
        self.engine: Optional[RobotMovementEngine] = None
    
    def with_config(self, config: RobotConfig) -> 'ComponentBuilder':
        """Agregar configuración."""
        self.config = config
        return self
    
    def with_optimizer(
        self,
        algorithm: OptimizationAlgorithm = OptimizationAlgorithm.PPO,
        **kwargs
    ) -> 'ComponentBuilder':
        """Agregar optimizador."""
        self.optimizer = TrajectoryOptimizerFactory.create(
            algorithm=algorithm,
            **kwargs
        )
        return self
    
    def with_engine(
        self,
        robot_brand: RobotBrand = RobotBrand.GENERIC
    ) -> 'ComponentBuilder':
        """Agregar motor de movimiento."""
        if self.config:
            self.engine = RobotMovementEngine(self.config)
        else:
            self.engine = MovementEngineFactory.create(robot_brand=robot_brand)
        return self
    
    def build_optimizer(self) -> TrajectoryOptimizer:
        """Construir optimizador."""
        if self.optimizer is None:
            self.optimizer = TrajectoryOptimizerFactory.create()
        return self.optimizer
    
    def build_engine(self) -> RobotMovementEngine:
        """Construir motor de movimiento."""
        if self.engine is None:
            if self.config:
                self.engine = RobotMovementEngine(self.config)
            else:
                self.engine = MovementEngineFactory.create()
        return self
    
    def build_all(self) -> Dict[str, Any]:
        """Construir todos los componentes."""
        return {
            "optimizer": self.build_optimizer(),
            "engine": self.build_engine(),
            "config": self.config or RobotConfig()
        }






