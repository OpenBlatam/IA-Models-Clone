"""
Advanced Telemetry System for Humanoid Devin Robot (Optimizado)
================================================================

Sistema de telemetría avanzada para monitoreo completo del robot.
"""

import logging
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from collections import deque
import json
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)


def ErrorCode(description: str):
    """
    Decorador para anotar excepciones con códigos de error y descripciones.
    
    Args:
        description: Descripción del error que se usará en el constructor.
    
    Usage:
        @ErrorCode(description="Invalid input provided")
        class MyException(Exception):
            def __init__(self):
                super().__init__(description)
    """
    def decorator(cls):
        # Almacenar la descripción en la clase
        cls._error_description = description
        return cls
    return decorator


@ErrorCode(description="Error in telemetry system")
class TelemetryError(Exception):
    """Excepción para errores de telemetría."""
    
    def __init__(self):
        """Initialize exception with description from @ErrorCode decorator."""
        message = getattr(self.__class__, '_error_description', "Error in telemetry system")
        super().__init__(message)
        self.message = message


class TelemetrySystem:
    """
    Sistema de telemetría avanzada para el robot humanoide.
    
    Recopila, almacena y analiza datos de telemetría en tiempo real.
    """
    
    def __init__(
        self,
        buffer_size: int = 10000,
        enable_persistence: bool = True,
        sampling_rate: float = 10.0
    ):
        """
        Inicializar sistema de telemetría.
        
        Args:
            buffer_size: Tamaño del buffer de telemetría
            enable_persistence: Habilitar persistencia de datos
            sampling_rate: Tasa de muestreo en Hz
        """
        if not isinstance(buffer_size, int) or buffer_size <= 0:
            raise ValueError("buffer_size must be a positive integer")
        if not isinstance(sampling_rate, (int, float)) or sampling_rate <= 0:
            raise ValueError("sampling_rate must be a positive number")
        
        self.buffer_size = buffer_size
        self.enable_persistence = enable_persistence
        self.sampling_rate = sampling_rate
        
        # Buffers de datos
        self.joint_states: deque = deque(maxlen=buffer_size)
        self.pose_history: deque = deque(maxlen=buffer_size)
        self.velocity_history: deque = deque(maxlen=buffer_size)
        self.acceleration_history: deque = deque(maxlen=buffer_size)
        self.torque_history: deque = deque(maxlen=buffer_size)
        self.power_history: deque = deque(maxlen=buffer_size)
        self.temperature_history: deque = deque(maxlen=buffer_size)
        
        # Eventos y alertas
        self.events: deque = deque(maxlen=1000)
        self.alerts: deque = deque(maxlen=500)
        
        # Callbacks
        self.callbacks: Dict[str, List[Callable]] = {}
        
        # Estadísticas
        self.total_samples = 0
        self.start_time = datetime.now()
        
        logger.info(
            f"Telemetry system initialized: "
            f"buffer_size={buffer_size}, sampling_rate={sampling_rate}Hz"
        )
    
    def record_joint_states(
        self,
        joint_positions: np.ndarray,
        joint_velocities: Optional[np.ndarray] = None,
        joint_torques: Optional[np.ndarray] = None
    ) -> None:
        """
        Registrar estados de articulaciones.
        
        Args:
            joint_positions: Posiciones de articulaciones
            joint_velocities: Velocidades de articulaciones (opcional)
            joint_torques: Torques de articulaciones (opcional)
        """
        if not isinstance(joint_positions, np.ndarray):
            joint_positions = np.array(joint_positions)
        
        if not np.all(np.isfinite(joint_positions)):
            raise ValueError("joint_positions must contain finite values")
        
        timestamp = datetime.now()
        
        record = {
            "timestamp": timestamp.isoformat(),
            "positions": joint_positions.tolist(),
            "velocities": joint_velocities.tolist() if joint_velocities is not None else None,
            "torques": joint_torques.tolist() if joint_torques is not None else None
        }
        
        self.joint_states.append(record)
        self.total_samples += 1
        
        # Calcular velocidades si no se proporcionan
        if joint_velocities is None and len(self.joint_states) > 1:
            prev_record = self.joint_states[-2]
            dt = (timestamp - datetime.fromisoformat(prev_record["timestamp"])).total_seconds()
            if dt > 0:
                velocities = (joint_positions - np.array(prev_record["positions"])) / dt
                self.velocity_history.append({
                    "timestamp": timestamp.isoformat(),
                    "velocities": velocities.tolist()
                })
        
        # Calcular aceleraciones
        if len(self.velocity_history) > 1:
            prev_vel = self.velocity_history[-2]
            dt = (timestamp - datetime.fromisoformat(prev_vel["timestamp"])).total_seconds()
            if dt > 0:
                current_vel = np.array(self.velocity_history[-1]["velocities"])
                prev_vel_array = np.array(prev_vel["velocities"])
                accelerations = (current_vel - prev_vel_array) / dt
                self.acceleration_history.append({
                    "timestamp": timestamp.isoformat(),
                    "accelerations": accelerations.tolist()
                })
        
        # Trigger callbacks
        self._trigger_callbacks("joint_states", record)
    
    def record_pose(
        self,
        position: np.ndarray,
        orientation: np.ndarray,
        frame_id: str = "base_link"
    ) -> None:
        """
        Registrar pose del robot.
        
        Args:
            position: Posición [x, y, z]
            orientation: Orientación [x, y, z, w] (quaternion)
            frame_id: ID del frame de referencia
        """
        if not isinstance(position, np.ndarray):
            position = np.array(position)
        if not isinstance(orientation, np.ndarray):
            orientation = np.array(orientation)
        
        if position.shape != (3,):
            raise ValueError("position must have shape (3,)")
        if orientation.shape != (4,):
            raise ValueError("orientation must have shape (4,)")
        
        record = {
            "timestamp": datetime.now().isoformat(),
            "position": position.tolist(),
            "orientation": orientation.tolist(),
            "frame_id": frame_id
        }
        
        self.pose_history.append(record)
        self._trigger_callbacks("pose", record)
    
    def record_power(
        self,
        component: str,
        power: float,
        voltage: Optional[float] = None,
        current: Optional[float] = None
    ) -> None:
        """
        Registrar consumo de potencia.
        
        Args:
            component: Componente
            power: Potencia en Watts
            voltage: Voltaje en Volts (opcional)
            current: Corriente en Amperes (opcional)
        """
        if not np.isfinite(power) or power < 0:
            raise ValueError("power must be a non-negative finite number")
        
        record = {
            "timestamp": datetime.now().isoformat(),
            "component": component,
            "power": float(power),
            "voltage": float(voltage) if voltage is not None else None,
            "current": float(current) if current is not None else None
        }
        
        self.power_history.append(record)
        self._trigger_callbacks("power", record)
    
    def record_temperature(
        self,
        component: str,
        temperature: float
    ) -> None:
        """
        Registrar temperatura.
        
        Args:
            component: Componente
            temperature: Temperatura en Celsius
        """
        if not np.isfinite(temperature):
            raise ValueError("temperature must be a finite number")
        
        record = {
            "timestamp": datetime.now().isoformat(),
            "component": component,
            "temperature": float(temperature)
        }
        
        self.temperature_history.append(record)
        
        # Alerta si temperatura es alta
        if temperature > 70.0:  # 70°C threshold
            self.add_alert("high_temperature", {
                "component": component,
                "temperature": temperature,
                "severity": "warning" if temperature < 85.0 else "critical"
            })
        
        self._trigger_callbacks("temperature", record)
    
    def add_event(
        self,
        event_type: str,
        data: Dict[str, Any],
        severity: str = "info"
    ) -> None:
        """
        Agregar evento.
        
        Args:
            event_type: Tipo de evento
            data: Datos del evento
            severity: Severidad (info, warning, error, critical)
        """
        if severity not in ["info", "warning", "error", "critical"]:
            raise ValueError(f"Invalid severity: {severity}")
        
        event = {
            "timestamp": datetime.now().isoformat(),
            "type": event_type,
            "data": data,
            "severity": severity
        }
        
        self.events.append(event)
        self._trigger_callbacks("event", event)
    
    def add_alert(
        self,
        alert_type: str,
        data: Dict[str, Any]
    ) -> None:
        """
        Agregar alerta.
        
        Args:
            alert_type: Tipo de alerta
            data: Datos de la alerta
        """
        alert = {
            "timestamp": datetime.now().isoformat(),
            "type": alert_type,
            "data": data
        }
        
        self.alerts.append(alert)
        self._trigger_callbacks("alert", alert)
        logger.warning(f"Alert: {alert_type} - {data}")
    
    def register_callback(
        self,
        event_type: str,
        callback: Callable
    ) -> None:
        """
        Registrar callback para eventos.
        
        Args:
            event_type: Tipo de evento
            callback: Función callback
        """
        if not callable(callback):
            raise ValueError("callback must be callable")
        
        if event_type not in self.callbacks:
            self.callbacks[event_type] = []
        
        self.callbacks[event_type].append(callback)
        logger.debug(f"Callback registered for {event_type}")
    
    def _trigger_callbacks(
        self,
        event_type: str,
        data: Dict[str, Any]
    ) -> None:
        """Trigger callbacks para un tipo de evento."""
        if event_type in self.callbacks:
            for callback in self.callbacks[event_type]:
                try:
                    callback(data)
                except Exception as e:
                    logger.error(f"Error in callback: {e}", exc_info=True)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Obtener estadísticas de telemetría.
        
        Returns:
            Dict con estadísticas
        """
        uptime = (datetime.now() - self.start_time).total_seconds()
        
        stats = {
            "uptime_seconds": uptime,
            "total_samples": self.total_samples,
            "sampling_rate_actual": self.total_samples / uptime if uptime > 0 else 0.0,
            "buffer_usage": {
                "joint_states": len(self.joint_states),
                "pose_history": len(self.pose_history),
                "velocity_history": len(self.velocity_history),
                "acceleration_history": len(self.acceleration_history),
                "power_history": len(self.power_history),
                "temperature_history": len(self.temperature_history),
                "events": len(self.events),
                "alerts": len(self.alerts)
            }
        }
        
        # Estadísticas de articulaciones
        if self.joint_states:
            recent_positions = [
                np.array(record["positions"])
                for record in list(self.joint_states)[-100:]
            ]
            if recent_positions:
                positions_array = np.array(recent_positions)
                stats["joint_statistics"] = {
                    "mean_positions": np.mean(positions_array, axis=0).tolist(),
                    "std_positions": np.std(positions_array, axis=0).tolist(),
                    "min_positions": np.min(positions_array, axis=0).tolist(),
                    "max_positions": np.max(positions_array, axis=0).tolist()
                }
        
        # Estadísticas de potencia
        if self.power_history:
            recent_power = [r["power"] for r in list(self.power_history)[-100:]]
            stats["power_statistics"] = {
                "mean": float(np.mean(recent_power)),
                "std": float(np.std(recent_power)),
                "min": float(np.min(recent_power)),
                "max": float(np.max(recent_power))
            }
        
        return stats
    
    def export_data(
        self,
        file_path: str,
        data_types: Optional[List[str]] = None
    ) -> None:
        """
        Exportar datos de telemetría.
        
        Args:
            file_path: Ruta al archivo
            data_types: Tipos de datos a exportar (None = todos)
        """
        if data_types is None:
            data_types = [
                "joint_states", "pose_history", "velocity_history",
                "acceleration_history", "power_history", "temperature_history",
                "events", "alerts"
            ]
        
        export_data = {}
        
        if "joint_states" in data_types:
            export_data["joint_states"] = list(self.joint_states)
        if "pose_history" in data_types:
            export_data["pose_history"] = list(self.pose_history)
        if "velocity_history" in data_types:
            export_data["velocity_history"] = list(self.velocity_history)
        if "acceleration_history" in data_types:
            export_data["acceleration_history"] = list(self.acceleration_history)
        if "power_history" in data_types:
            export_data["power_history"] = list(self.power_history)
        if "temperature_history" in data_types:
            export_data["temperature_history"] = list(self.temperature_history)
        if "events" in data_types:
            export_data["events"] = list(self.events)
        if "alerts" in data_types:
            export_data["alerts"] = list(self.alerts)
        
        export_data["metadata"] = {
            "export_timestamp": datetime.now().isoformat(),
            "total_samples": self.total_samples,
            "uptime_seconds": (datetime.now() - self.start_time).total_seconds()
        }
        
        try:
            export_file = Path(file_path)
            export_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(export_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Telemetry data exported to {file_path}")
        except Exception as e:
            logger.error(f"Error exporting telemetry data: {e}", exc_info=True)
            raise TelemetryError(f"Failed to export data: {str(e)}") from e
    
    def clear_data(self) -> None:
        """Limpiar todos los datos de telemetría."""
        self.joint_states.clear()
        self.pose_history.clear()
        self.velocity_history.clear()
        self.acceleration_history.clear()
        self.torque_history.clear()
        self.power_history.clear()
        self.temperature_history.clear()
        self.events.clear()
        self.alerts.clear()
        self.total_samples = 0
        logger.info("Telemetry data cleared")

