"""
Adaptive Learning System for Humanoid Devin Robot (Optimizado)
===============================================================

Sistema de aprendizaje adaptativo para mejorar el rendimiento del robot
basado en experiencia previa.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import numpy as np
from collections import deque
import json
from pathlib import Path

logger = logging.getLogger(__name__)


def ErrorCode(description: str):
    """Decorador para anotar excepciones con códigos de error y descripciones."""
    def decorator(cls):
        cls._error_description = description
        return cls
    return decorator


@ErrorCode(description="Error in adaptive learning system")
class AdaptiveLearningError(Exception):
    """Excepción para errores de aprendizaje adaptativo."""
    
    def __init__(self):
        """Initialize exception with description from @ErrorCode decorator."""
        message = getattr(self.__class__, '_error_description', "Error in adaptive learning system")
        super().__init__(message)
        self.message = message


class AdaptiveLearningSystem:
    """
    Sistema de aprendizaje adaptativo para el robot humanoide.
    
    Aprende de experiencias previas para mejorar movimientos y decisiones.
    """
    
    def __init__(
        self,
        memory_size: int = 1000,
        learning_rate: float = 0.01,
        discount_factor: float = 0.95
    ):
        """
        Inicializar sistema de aprendizaje adaptativo.
        
        Args:
            memory_size: Tamaño de la memoria de experiencias
            learning_rate: Tasa de aprendizaje (0-1)
            discount_factor: Factor de descuento para recompensas futuras (0-1)
        """
        if not isinstance(memory_size, int) or memory_size <= 0:
            raise ValueError("memory_size must be a positive integer")
        if not 0 < learning_rate <= 1:
            raise ValueError("learning_rate must be between 0 and 1")
        if not 0 < discount_factor <= 1:
            raise ValueError("discount_factor must be between 0 and 1")
        
        self.memory_size = memory_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        
        # Memoria de experiencias
        self.experience_memory: deque = deque(maxlen=memory_size)
        
        # Modelos de aprendizaje
        self.movement_patterns: Dict[str, Dict[str, Any]] = {}
        self.success_rates: Dict[str, float] = {}
        self.optimization_suggestions: Dict[str, List[str]] = {}
        
        # Estadísticas
        self.total_experiences = 0
        self.total_improvements = 0
        
        logger.info(
            f"Adaptive learning system initialized: "
            f"memory_size={memory_size}, learning_rate={learning_rate}"
        )
    
    def record_experience(
        self,
        action_type: str,
        parameters: Dict[str, Any],
        result: Dict[str, Any],
        success: bool,
        execution_time: float,
        energy_consumed: Optional[float] = None
    ) -> None:
        """
        Registrar una experiencia para aprendizaje.
        
        Args:
            action_type: Tipo de acción (walk, grasp, gesture, etc.)
            parameters: Parámetros de la acción
            result: Resultado de la acción
            success: Si la acción fue exitosa
            execution_time: Tiempo de ejecución en segundos
            energy_consumed: Energía consumida (opcional)
        """
        if not action_type or not isinstance(action_type, str):
            raise ValueError("action_type must be a non-empty string")
        if not isinstance(parameters, dict):
            raise ValueError("parameters must be a dictionary")
        if not isinstance(result, dict):
            raise ValueError("result must be a dictionary")
        if not isinstance(success, bool):
            raise ValueError("success must be a boolean")
        if not np.isfinite(execution_time) or execution_time < 0:
            raise ValueError("execution_time must be a non-negative finite number")
        
        experience = {
            "timestamp": datetime.now().isoformat(),
            "action_type": action_type,
            "parameters": parameters,
            "result": result,
            "success": success,
            "execution_time": float(execution_time),
            "energy_consumed": float(energy_consumed) if energy_consumed is not None else None
        }
        
        self.experience_memory.append(experience)
        self.total_experiences += 1
        
        # Actualizar estadísticas
        self._update_statistics(action_type, success, execution_time)
        
        logger.debug(f"Experience recorded: {action_type}, success={success}")
    
    def _update_statistics(
        self,
        action_type: str,
        success: bool,
        execution_time: float
    ) -> None:
        """Actualizar estadísticas internas."""
        if action_type not in self.success_rates:
            self.success_rates[action_type] = 0.0
            self.movement_patterns[action_type] = {
                "total_attempts": 0,
                "successful_attempts": 0,
                "avg_execution_time": 0.0,
                "best_execution_time": float('inf'),
                "parameter_history": []
            }
        
        pattern = self.movement_patterns[action_type]
        pattern["total_attempts"] += 1
        
        if success:
            pattern["successful_attempts"] += 1
            pattern["best_execution_time"] = min(
                pattern["best_execution_time"],
                execution_time
            )
        
        # Actualizar tasa de éxito (media móvil exponencial)
        current_rate = self.success_rates[action_type]
        new_rate = (
            current_rate * (1 - self.learning_rate) +
            (1.0 if success else 0.0) * self.learning_rate
        )
        self.success_rates[action_type] = new_rate
        
        # Actualizar tiempo promedio de ejecución
        pattern["avg_execution_time"] = (
            pattern["avg_execution_time"] * (1 - self.learning_rate) +
            execution_time * self.learning_rate
        )
    
    def get_optimal_parameters(
        self,
        action_type: str,
        default_parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Obtener parámetros óptimos basados en experiencia previa.
        
        Args:
            action_type: Tipo de acción
            default_parameters: Parámetros por defecto
            
        Returns:
            Parámetros optimizados
        """
        if action_type not in self.movement_patterns:
            return default_parameters.copy()
        
        pattern = self.movement_patterns[action_type]
        
        # Buscar experiencias exitosas similares
        successful_experiences = [
            exp for exp in self.experience_memory
            if exp["action_type"] == action_type and exp["success"]
        ]
        
        if not successful_experiences:
            return default_parameters.copy()
        
        # Calcular parámetros promedio de experiencias exitosas
        optimal_params = default_parameters.copy()
        
        # Analizar parámetros comunes
        param_values = {}
        for exp in successful_experiences:
            for key, value in exp["parameters"].items():
                if key not in param_values:
                    param_values[key] = []
                if isinstance(value, (int, float)) and np.isfinite(value):
                    param_values[key].append(value)
        
        # Usar mediana para parámetros numéricos (más robusto que promedio)
        for key, values in param_values.items():
            if values and key in optimal_params:
                if isinstance(optimal_params[key], (int, float)):
                    optimal_params[key] = float(np.median(values))
        
        return optimal_params
    
    def get_success_probability(self, action_type: str) -> float:
        """
        Obtener probabilidad de éxito estimada para un tipo de acción.
        
        Args:
            action_type: Tipo de acción
            
        Returns:
            Probabilidad de éxito (0-1)
        """
        return self.success_rates.get(action_type, 0.5)
    
    def get_optimization_suggestions(self, action_type: str) -> List[str]:
        """
        Obtener sugerencias de optimización para un tipo de acción.
        
        Args:
            action_type: Tipo de acción
            
        Returns:
            Lista de sugerencias
        """
        if action_type not in self.movement_patterns:
            return []
        
        pattern = self.movement_patterns[action_type]
        suggestions = []
        
        # Analizar tasa de éxito
        success_rate = self.success_rates.get(action_type, 0.0)
        if success_rate < 0.7:
            suggestions.append(
                f"Low success rate ({success_rate:.1%}). "
                "Consider adjusting parameters or checking robot state."
            )
        
        # Analizar tiempo de ejecución
        avg_time = pattern.get("avg_execution_time", 0.0)
        best_time = pattern.get("best_execution_time", float('inf'))
        if avg_time > best_time * 1.5:
            suggestions.append(
                f"Execution time is {avg_time:.2f}s (best: {best_time:.2f}s). "
                "Consider optimizing movement trajectory."
            )
        
        return suggestions
    
    def learn_from_batch(self, experiences: List[Dict[str, Any]]) -> None:
        """
        Aprender de un lote de experiencias.
        
        Args:
            experiences: Lista de experiencias
        """
        if not isinstance(experiences, list):
            raise ValueError("experiences must be a list")
        
        for exp in experiences:
            try:
                self.record_experience(
                    action_type=exp.get("action_type", "unknown"),
                    parameters=exp.get("parameters", {}),
                    result=exp.get("result", {}),
                    success=exp.get("success", False),
                    execution_time=exp.get("execution_time", 0.0),
                    energy_consumed=exp.get("energy_consumed")
                )
            except Exception as e:
                logger.warning(f"Error processing experience: {e}", exc_info=True)
        
        logger.info(f"Learned from {len(experiences)} experiences")
    
    def save_learning_data(self, file_path: str) -> None:
        """
        Guardar datos de aprendizaje en archivo.
        
        Args:
            file_path: Ruta al archivo
        """
        data = {
            "movement_patterns": self.movement_patterns,
            "success_rates": self.success_rates,
            "total_experiences": self.total_experiences,
            "total_improvements": self.total_improvements,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            save_file = Path(file_path)
            save_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(save_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Learning data saved to {file_path}")
        except Exception as e:
            logger.error(f"Error saving learning data: {e}", exc_info=True)
            raise AdaptiveLearningError(f"Failed to save learning data: {str(e)}") from e
    
    def load_learning_data(self, file_path: str) -> None:
        """
        Cargar datos de aprendizaje desde archivo.
        
        Args:
            file_path: Ruta al archivo
        """
        load_file = Path(file_path)
        
        if not load_file.exists():
            raise AdaptiveLearningError(f"Learning data file not found: {file_path}")
        
        try:
            with open(load_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.movement_patterns = data.get("movement_patterns", {})
            self.success_rates = data.get("success_rates", {})
            self.total_experiences = data.get("total_experiences", 0)
            self.total_improvements = data.get("total_improvements", 0)
            
            logger.info(f"Learning data loaded from {file_path}")
        except Exception as e:
            logger.error(f"Error loading learning data: {e}", exc_info=True)
            raise AdaptiveLearningError(f"Failed to load learning data: {str(e)}") from e
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Obtener estadísticas del sistema de aprendizaje.
        
        Returns:
            Dict con estadísticas
        """
        return {
            "total_experiences": self.total_experiences,
            "total_improvements": self.total_improvements,
            "memory_usage": len(self.experience_memory),
            "memory_capacity": self.memory_size,
            "learned_actions": len(self.movement_patterns),
            "success_rates": self.success_rates.copy(),
            "learning_rate": self.learning_rate,
            "discount_factor": self.discount_factor
        }
    
    def reset_learning(self) -> None:
        """Resetear todo el aprendizaje."""
        self.experience_memory.clear()
        self.movement_patterns.clear()
        self.success_rates.clear()
        self.optimization_suggestions.clear()
        self.total_experiences = 0
        self.total_improvements = 0
        logger.info("Learning data reset")

