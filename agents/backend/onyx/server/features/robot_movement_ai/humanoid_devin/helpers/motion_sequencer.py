"""
Motion Sequencer for Humanoid Devin Robot (Optimizado)
=======================================================

Secuenciador de movimientos para ejecutar secuencias complejas.
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional, Callable, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class MotionType(str, Enum):
    """Tipos de movimientos."""
    JOINT_POSITIONS = "joint_positions"
    POSE = "pose"
    WALK = "walk"
    GRASP = "grasp"
    RELEASE = "release"
    WAVE = "wave"
    STAND = "stand"
    CUSTOM = "custom"


@dataclass
class MotionStep:
    """Paso de movimiento en una secuencia."""
    motion_type: MotionType
    parameters: Dict[str, Any]
    duration: float = 1.0
    wait: bool = True
    name: Optional[str] = None
    condition: Optional[Callable] = None


def ErrorCode(description: str):
    """Decorador para anotar excepciones con códigos de error y descripciones."""
    def decorator(cls):
        cls._error_description = description
        return cls
    return decorator


@ErrorCode(description="Error in motion sequencer system")
class MotionSequencerError(Exception):
    """Excepción para errores del secuenciador."""
    
    def __init__(self):
        """Initialize exception with description from @ErrorCode decorator."""
        message = getattr(self.__class__, '_error_description', "Error in motion sequencer system")
        super().__init__(message)
        self.message = message


class MotionSequencer:
    """
    Secuenciador de movimientos para el robot humanoide.
    
    Permite ejecutar secuencias complejas de movimientos de forma coordinada.
    """
    
    def __init__(self, robot_driver):
        """
        Inicializar secuenciador de movimientos.
        
        Args:
            robot_driver: Instancia de HumanoidDevinDriver
        """
        if robot_driver is None:
            raise ValueError("robot_driver cannot be None")
        
        self.robot = robot_driver
        self.sequence: List[MotionStep] = []
        self.current_step: int = 0
        self.is_running: bool = False
        
        logger.info("Motion sequencer initialized")
    
    def add_step(
        self,
        motion_type: Union[MotionType, str],
        parameters: Dict[str, Any],
        duration: float = 1.0,
        wait: bool = True,
        name: Optional[str] = None,
        condition: Optional[Callable] = None
    ) -> None:
        """
        Agregar paso a la secuencia.
        
        Args:
            motion_type: Tipo de movimiento
            parameters: Parámetros del movimiento
            duration: Duración del movimiento (segundos)
            wait: Esperar a que termine antes del siguiente paso
            name: Nombre del paso (opcional)
            condition: Función de condición para ejecutar el paso (opcional)
        """
        if isinstance(motion_type, str):
            try:
                motion_type = MotionType(motion_type)
            except ValueError:
                raise ValueError(f"Invalid motion_type: {motion_type}")
        
        step = MotionStep(
            motion_type=motion_type,
            parameters=parameters,
            duration=duration,
            wait=wait,
            name=name,
            condition=condition
        )
        
        self.sequence.append(step)
        logger.debug(f"Added motion step: {motion_type.value} ({name or 'unnamed'})")
    
    def clear_sequence(self) -> None:
        """Limpiar secuencia actual."""
        self.sequence.clear()
        self.current_step = 0
        logger.info("Sequence cleared")
    
    async def execute_step(self, step: MotionStep) -> bool:
        """
        Ejecutar un paso de movimiento.
        
        Args:
            step: Paso a ejecutar
            
        Returns:
            True si se ejecutó exitosamente
        """
        # Verificar condición si existe
        if step.condition is not None:
            try:
                if not step.condition():
                    logger.debug(f"Step condition not met, skipping: {step.name}")
                    return False
            except Exception as e:
                logger.error(f"Error evaluating condition: {e}")
                return False
        
        try:
            if step.motion_type == MotionType.JOINT_POSITIONS:
                positions = step.parameters.get("positions")
                if positions is None:
                    raise ValueError("positions parameter required for joint_positions")
                return await self.robot.set_joint_positions(positions)
            
            elif step.motion_type == MotionType.POSE:
                position = step.parameters.get("position")
                orientation = step.parameters.get("orientation")
                hand = step.parameters.get("hand", "right")
                
                if position is None or orientation is None:
                    raise ValueError("position and orientation required for pose")
                
                return await self.robot.move_to_pose(position, orientation, hand)
            
            elif step.motion_type == MotionType.WALK:
                direction = step.parameters.get("direction", "forward")
                distance = step.parameters.get("distance", 1.0)
                speed = step.parameters.get("speed", 0.5)
                
                return await self.robot.walk(direction, distance, speed)
            
            elif step.motion_type == MotionType.GRASP:
                hand = step.parameters.get("hand", "right")
                object_position = step.parameters.get("object_position")
                
                return await self.robot.grasp(hand, object_position)
            
            elif step.motion_type == MotionType.RELEASE:
                hand = step.parameters.get("hand", "right")
                return await self.robot.release(hand)
            
            elif step.motion_type == MotionType.WAVE:
                hand = step.parameters.get("hand", "right")
                return await self.robot.wave(hand)
            
            elif step.motion_type == MotionType.STAND:
                return await self.robot.stand()
            
            elif step.motion_type == MotionType.CUSTOM:
                func = step.parameters.get("function")
                if func is None or not callable(func):
                    raise ValueError("function parameter must be callable for custom motion")
                
                args = step.parameters.get("args", [])
                kwargs = step.parameters.get("kwargs", {})
                
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
            
            else:
                raise ValueError(f"Unknown motion type: {step.motion_type}")
        
        except Exception as e:
            logger.error(f"Error executing motion step: {e}", exc_info=True)
            raise MotionSequencerError(f"Failed to execute step: {str(e)}") from e
    
    async def execute_sequence(
        self,
        start_from: int = 0,
        stop_on_error: bool = True
    ) -> Dict[str, Any]:
        """
        Ejecutar secuencia completa.
        
        Args:
            start_from: Índice del paso desde el cual empezar
            stop_on_error: Detener secuencia si hay error
            
        Returns:
            Dict con resultados de la ejecución
        """
        if not self.robot.connected:
            raise MotionSequencerError("Robot not connected")
        
        if len(self.sequence) == 0:
            logger.warning("Empty sequence, nothing to execute")
            return {"success": False, "error": "Empty sequence"}
        
        self.is_running = True
        self.current_step = start_from
        
        results = {
            "success": True,
            "steps_executed": 0,
            "steps_total": len(self.sequence),
            "errors": []
        }
        
        logger.info(f"Executing sequence: {len(self.sequence)} steps")
        
        try:
            for i, step in enumerate(self.sequence[start_from:], start=start_from):
                self.current_step = i
                step_name = step.name or f"step_{i}"
                
                logger.info(f"Executing step {i+1}/{len(self.sequence)}: {step_name}")
                
                try:
                    success = await self.execute_step(step)
                    
                    if success:
                        results["steps_executed"] += 1
                        logger.debug(f"Step {i+1} completed successfully")
                    else:
                        error_msg = f"Step {i+1} ({step_name}) returned False"
                        results["errors"].append(error_msg)
                        logger.warning(error_msg)
                        
                        if stop_on_error:
                            results["success"] = False
                            break
                    
                    # Esperar si está configurado
                    if step.wait and step.duration > 0:
                        await asyncio.sleep(step.duration)
                
                except Exception as e:
                    error_msg = f"Step {i+1} ({step_name}) failed: {str(e)}"
                    results["errors"].append(error_msg)
                    logger.error(error_msg, exc_info=True)
                    
                    if stop_on_error:
                        results["success"] = False
                        break
            
            results["success"] = results["success"] and len(results["errors"]) == 0
            
        finally:
            self.is_running = False
        
        logger.info(
            f"Sequence execution completed: "
            f"{results['steps_executed']}/{results['steps_total']} steps, "
            f"success={results['success']}"
        )
        
        return results
    
    def get_sequence_info(self) -> Dict[str, Any]:
        """
        Obtener información de la secuencia.
        
        Returns:
            Dict con información de la secuencia
        """
        total_duration = sum(step.duration for step in self.sequence)
        
        return {
            "total_steps": len(self.sequence),
            "total_duration": total_duration,
            "current_step": self.current_step,
            "is_running": self.is_running,
            "steps": [
                {
                    "index": i,
                    "name": step.name or f"step_{i}",
                    "type": step.motion_type.value,
                    "duration": step.duration,
                    "wait": step.wait
                }
                for i, step in enumerate(self.sequence)
            ]
        }
    
    def pause(self) -> None:
        """Pausar ejecución de secuencia."""
        # Implementación simplificada - en producción usaría threading/async
        logger.info("Sequence paused")
    
    def resume(self) -> None:
        """Reanudar ejecución de secuencia."""
        logger.info("Sequence resumed")
    
    def stop(self) -> None:
        """Detener ejecución de secuencia."""
        self.is_running = False
        logger.info("Sequence stopped")

