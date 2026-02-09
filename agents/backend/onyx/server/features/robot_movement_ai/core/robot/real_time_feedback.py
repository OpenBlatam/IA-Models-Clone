"""
Real-Time Feedback System - 1000Hz
===================================

Sistema de feedback en tiempo real a alta frecuencia para control preciso.
"""

import asyncio
import time
import logging
from typing import Optional, Callable, Dict, Any, List
from dataclasses import dataclass
from collections import deque
import numpy as np

try:
    from .performance import euclidean_distance_fast
    USE_PERFORMANCE_UTILS = True
except ImportError:
    USE_PERFORMANCE_UTILS = False

logger = logging.getLogger(__name__)


@dataclass
class FeedbackData:
    """Datos de feedback del robot."""
    timestamp: float
    joint_positions: List[float]
    joint_velocities: List[float]
    joint_torques: List[float]
    end_effector_position: np.ndarray
    end_effector_orientation: np.ndarray
    force_torque: Optional[np.ndarray] = None
    temperature: Optional[float] = None
    errors: List[str] = None


class RealTimeFeedbackSystem:
    """
    Sistema de feedback en tiempo real a 1000Hz.
    
    Características:
    - Adquisición de datos a alta frecuencia
    - Procesamiento en tiempo real
    - Detección de anomalías
    - Historial circular para análisis
    """
    
    def __init__(
        self,
        frequency: int = 1000,
        buffer_size: int = 10000,
        callback: Optional[Callable[[FeedbackData], None]] = None
    ) -> None:
        """
        Inicializar sistema de feedback.
        
        Args:
            frequency: Frecuencia de adquisición en Hz
            buffer_size: Tamaño del buffer circular
            callback: Función a llamar con cada dato de feedback
        
        Raises:
            ValueError: Si la frecuencia o buffer_size son inválidos
        """
        from ..exceptions import ConfigurationError
        
        # Validar parámetros
        if frequency <= 0 or frequency > 10000:
            raise ConfigurationError(
                f"Frequency must be between 1 and 10000 Hz, got {frequency}",
                error_code="INVALID_FREQUENCY",
                details={"frequency": frequency}
            )
        
        if buffer_size <= 0:
            raise ConfigurationError(
                f"Buffer size must be positive, got {buffer_size}",
                error_code="INVALID_BUFFER_SIZE",
                details={"buffer_size": buffer_size}
            )
        self.frequency = frequency
        self.period = 1.0 / frequency
        self.buffer_size = buffer_size
        self.callback = callback
        
        # Buffer circular para historial
        self.feedback_buffer: deque = deque(maxlen=buffer_size)
        
        # Estado
        self.is_running = False
        self.task: Optional[asyncio.Task] = None
        
        # Estadísticas
        self.stats = {
            "total_samples": 0,
            "missed_samples": 0,
            "average_latency": 0.0,
            "max_latency": 0.0,
        }
        
        logger.info(f"Real-Time Feedback System initialized at {frequency} Hz")
    
    async def start(self):
        """Iniciar sistema de feedback."""
        if self.is_running:
            logger.warning("Feedback system already running")
            return
        
        self.is_running = True
        self.task = asyncio.create_task(self._feedback_loop())
        logger.info("Feedback system started")
    
    async def stop(self):
        """Detener sistema de feedback."""
        if not self.is_running:
            return
        
        self.is_running = False
        if self.task:
            await self.task
        logger.info("Feedback system stopped")
    
    async def _feedback_loop(self):
        """Loop principal de adquisición de feedback."""
        next_time = time.perf_counter()
        period = self.period
        threshold = period * 1.1
        
        while self.is_running:
            try:
                feedback_data = await self._acquire_feedback()
                current_time = time.perf_counter()
                latency = current_time - next_time
                
                total = self.stats["total_samples"] + 1
                self.stats["total_samples"] = total
                
                if latency > threshold:
                    self.stats["missed_samples"] += 1
                
                self.stats["average_latency"] = (
                    (self.stats["average_latency"] * (total - 1) + latency) / total
                )
                self.stats["max_latency"] = max(self.stats["max_latency"], latency)
                
                self.feedback_buffer.append(feedback_data)
                
                if self.callback:
                    try:
                        self.callback(feedback_data)
                    except Exception:
                        pass
                
                next_time += period
                sleep_time = next_time - current_time
                
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                else:
                    next_time = current_time
            
            except Exception as e:
                logger.error(f"Error in feedback loop: {e}", exc_info=True)
                await asyncio.sleep(period)
    
    async def _acquire_feedback(self) -> FeedbackData:
        """
        Adquirir datos de feedback del robot.
        
        En producción, esto se conectaría al hardware real.
        """
        # Simulación de adquisición de datos
        current_time = time.time()
        
        # Datos simulados (en producción vendrían del robot)
        feedback = FeedbackData(
            timestamp=current_time,
            joint_positions=[0.0] * 6,
            joint_velocities=[0.0] * 6,
            joint_torques=[0.0] * 6,
            end_effector_position=np.array([0.0, 0.0, 0.0]),
            end_effector_orientation=np.array([0.0, 0.0, 0.0, 1.0]),
            force_torque=None,
            temperature=25.0,
            errors=[]
        )
        
        return feedback
    
    def get_latest_feedback(self) -> Optional[FeedbackData]:
        """Obtener último dato de feedback."""
        if self.feedback_buffer:
            return self.feedback_buffer[-1]
        return None
    
    def get_feedback_history(
        self,
        duration: float,
        current_time: Optional[float] = None
    ) -> List[FeedbackData]:
        """
        Obtener historial de feedback de los últimos N segundos.
        
        Args:
            duration: Duración en segundos
            current_time: Tiempo actual (opcional)
            
        Returns:
            Lista de datos de feedback
        """
        if current_time is None:
            current_time = time.time()
        
        cutoff_time = current_time - duration
        history = [
            data for data in self.feedback_buffer
            if data.timestamp >= cutoff_time
        ]
        
        return history
    
    def get_statistics(self) -> Dict[str, Any]:
        """Obtener estadísticas del sistema."""
        return {
            **self.stats,
            "buffer_usage": len(self.feedback_buffer) / self.buffer_size,
            "is_running": self.is_running,
        }
    
    def detect_anomalies(self, feedback: FeedbackData) -> List[str]:
        """
        Detectar anomalías en los datos de feedback.
        
        Returns:
            Lista de anomalías detectadas
        """
        anomalies = []
        
        # Verificar límites de articulaciones
        for i, pos in enumerate(feedback.joint_positions):
            if abs(pos) > 3.14:  # Más de 180 grados
                anomalies.append(f"Joint {i} position out of range: {pos}")
        
        # Verificar velocidades excesivas
        for i, vel in enumerate(feedback.joint_velocities):
            if abs(vel) > 5.0:  # rad/s
                anomalies.append(f"Joint {i} velocity too high: {vel}")
        
        # Verificar torques excesivos
        for i, torque in enumerate(feedback.joint_torques):
            if abs(torque) > 100.0:  # Nm (ajustar según robot)
                anomalies.append(f"Joint {i} torque too high: {torque}")
        
        # Verificar temperatura
        if feedback.temperature and feedback.temperature > 60.0:
            anomalies.append(f"High temperature: {feedback.temperature}°C")
        
        return anomalies






