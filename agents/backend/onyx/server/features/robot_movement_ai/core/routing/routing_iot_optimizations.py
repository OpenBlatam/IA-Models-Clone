"""
Optimizaciones de IoT para Routing.

Este módulo implementa optimizaciones específicas para dispositivos IoT,
incluyendo bajo consumo, comunicación eficiente y procesamiento en el edge.
"""

import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json

logger = logging.getLogger(__name__)


class IoTProtocol(Enum):
    """Protocolos de comunicación IoT."""
    MQTT = "mqtt"
    COAP = "coap"
    HTTP = "http"
    LORAWAN = "lorawan"
    ZIGBEE = "zigbee"


class PowerMode(Enum):
    """Modos de consumo de energía."""
    HIGH_PERFORMANCE = "high_performance"
    BALANCED = "balanced"
    LOW_POWER = "low_power"
    ULTRA_LOW_POWER = "ultra_low_power"


@dataclass
class IoTDevice:
    """Dispositivo IoT."""
    device_id: str
    device_type: str
    battery_level: float = 100.0
    power_mode: PowerMode = PowerMode.BALANCED
    location: Dict[str, float] = field(default_factory=dict)
    capabilities: List[str] = field(default_factory=list)
    last_update: float = field(default_factory=time.time)


@dataclass
class IoTMessage:
    """Mensaje IoT."""
    device_id: str
    message_type: str
    payload: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    priority: int = 0


class PowerManager:
    """Gestor de energía para dispositivos IoT."""
    
    def __init__(self):
        self.power_profiles: Dict[PowerMode, Dict[str, float]] = {
            PowerMode.HIGH_PERFORMANCE: {
                "cpu_freq": 1.0,
                "radio_power": 1.0,
                "sensor_rate": 1.0
            },
            PowerMode.BALANCED: {
                "cpu_freq": 0.7,
                "radio_power": 0.7,
                "sensor_rate": 0.7
            },
            PowerMode.LOW_POWER: {
                "cpu_freq": 0.4,
                "radio_power": 0.4,
                "sensor_rate": 0.4
            },
            PowerMode.ULTRA_LOW_POWER: {
                "cpu_freq": 0.1,
                "radio_power": 0.1,
                "sensor_rate": 0.1
            }
        }
        self.current_mode = PowerMode.BALANCED
        self.energy_consumption = 0.0
    
    def set_power_mode(self, mode: PowerMode):
        """Establecer modo de energía."""
        self.current_mode = mode
        profile = self.power_profiles[mode]
        logger.info(f"Power mode set to {mode.value}: {profile}")
    
    def calculate_energy_consumption(self, operation: str, duration: float) -> float:
        """Calcular consumo de energía."""
        profile = self.power_profiles[self.current_mode]
        base_consumption = {
            "compute": 10.0,
            "communicate": 15.0,
            "sense": 5.0
        }
        
        consumption = base_consumption.get(operation, 10.0) * profile["cpu_freq"] * duration
        self.energy_consumption += consumption
        return consumption
    
    def estimate_battery_life(self, battery_level: float) -> float:
        """Estimar vida de batería en horas."""
        if self.energy_consumption == 0:
            return float('inf')
        
        hours_per_percent = 1.0 / (self.energy_consumption / 100.0)
        return battery_level * hours_per_percent


class MessageQueue:
    """Cola de mensajes optimizada para IoT."""
    
    def __init__(self, max_size: int = 100):
        self.queue: List[IoTMessage] = []
        self.max_size = max_size
        self.total_messages = 0
        self.dropped_messages = 0
    
    def enqueue(self, message: IoTMessage):
        """Agregar mensaje a la cola."""
        if len(self.queue) >= self.max_size:
            # Eliminar mensaje de menor prioridad
            if message.priority > min(m.priority for m in self.queue):
                self.queue.sort(key=lambda m: m.priority, reverse=True)
                self.queue.pop()
                self.queue.append(message)
            else:
                self.dropped_messages += 1
                return
        else:
            self.queue.append(message)
        
        self.total_messages += 1
    
    def dequeue(self) -> Optional[IoTMessage]:
        """Obtener siguiente mensaje."""
        if not self.queue:
            return None
        
        # Prioridad más alta primero
        self.queue.sort(key=lambda m: m.priority, reverse=True)
        return self.queue.pop(0)
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas."""
        return {
            "queue_size": len(self.queue),
            "max_size": self.max_size,
            "total_messages": self.total_messages,
            "dropped_messages": self.dropped_messages,
            "drop_rate": self.dropped_messages / max(self.total_messages, 1)
        }


class EdgeProcessor:
    """Procesador edge para IoT."""
    
    def __init__(self, max_processing_time: float = 0.1):
        self.max_processing_time = max_processing_time
        self.processed_tasks = 0
        self.failed_tasks = 0
        self.avg_processing_time = 0.0
    
    def process_route(self, route_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Procesar ruta en el edge."""
        start_time = time.time()
        
        try:
            # Procesamiento simplificado
            result = {
                "route_id": route_data.get("route_id"),
                "processed": True,
                "timestamp": time.time()
            }
            
            processing_time = time.time() - start_time
            
            if processing_time > self.max_processing_time:
                self.failed_tasks += 1
                return None
            
            self.processed_tasks += 1
            self.avg_processing_time = (
                (self.avg_processing_time * (self.processed_tasks - 1) + processing_time) 
                / self.processed_tasks
            )
            
            return result
        except Exception as e:
            logger.error(f"Edge processing failed: {e}")
            self.failed_tasks += 1
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas."""
        return {
            "processed_tasks": self.processed_tasks,
            "failed_tasks": self.failed_tasks,
            "success_rate": self.processed_tasks / max(self.processed_tasks + self.failed_tasks, 1),
            "avg_processing_time": self.avg_processing_time,
            "max_processing_time": self.max_processing_time
        }


class IoTProtocolHandler:
    """Manejador de protocolos IoT."""
    
    def __init__(self, protocol: IoTProtocol = IoTProtocol.MQTT):
        self.protocol = protocol
        self.message_size_limit = {
            IoTProtocol.MQTT: 256 * 1024,  # 256 KB
            IoTProtocol.COAP: 1024,  # 1 KB
            IoTProtocol.HTTP: 10 * 1024 * 1024,  # 10 MB
            IoTProtocol.LORAWAN: 242,  # 242 bytes
            IoTProtocol.ZIGBEE: 127  # 127 bytes
        }
        self.total_messages = 0
        self.total_bytes = 0
    
    def encode_message(self, data: Dict[str, Any]) -> bytes:
        """Codificar mensaje según protocolo."""
        json_data = json.dumps(data).encode('utf-8')
        size_limit = self.message_size_limit[self.protocol]
        
        if len(json_data) > size_limit:
            # Comprimir o truncar
            json_data = json_data[:size_limit]
        
        self.total_messages += 1
        self.total_bytes += len(json_data)
        
        return json_data
    
    def decode_message(self, data: bytes) -> Dict[str, Any]:
        """Decodificar mensaje."""
        try:
            return json.loads(data.decode('utf-8'))
        except Exception as e:
            logger.error(f"Failed to decode message: {e}")
            return {}
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas."""
        return {
            "protocol": self.protocol.value,
            "total_messages": self.total_messages,
            "total_bytes": self.total_bytes,
            "avg_message_size": self.total_bytes / max(self.total_messages, 1),
            "size_limit": self.message_size_limit[self.protocol]
        }


class IoTOptimizer:
    """Optimizador principal para IoT."""
    
    def __init__(self, enable_iot: bool = True,
                 protocol: IoTProtocol = IoTProtocol.MQTT,
                 power_mode: PowerMode = PowerMode.BALANCED):
        self.enable_iot = enable_iot
        self.power_manager = PowerManager() if enable_iot else None
        self.message_queue = MessageQueue() if enable_iot else None
        self.edge_processor = EdgeProcessor() if enable_iot else None
        self.protocol_handler = IoTProtocolHandler(protocol=protocol) if enable_iot else None
        self.devices: Dict[str, IoTDevice] = {}
        
        if self.power_manager:
            self.power_manager.set_power_mode(power_mode)
    
    def register_device(self, device_id: str, device_type: str,
                       location: Dict[str, float]) -> bool:
        """Registrar dispositivo IoT."""
        if not self.enable_iot:
            return False
        
        device = IoTDevice(
            device_id=device_id,
            device_type=device_type,
            location=location
        )
        self.devices[device_id] = device
        return True
    
    def send_message(self, device_id: str, message_type: str,
                    payload: Dict[str, Any], priority: int = 0) -> bool:
        """Enviar mensaje a dispositivo."""
        if not self.enable_iot or not self.message_queue:
            return False
        
        if device_id not in self.devices:
            return False
        
        message = IoTMessage(
            device_id=device_id,
            message_type=message_type,
            payload=payload,
            priority=priority
        )
        
        # Codificar según protocolo
        if self.protocol_handler:
            encoded = self.protocol_handler.encode_message(payload)
            # Enviar a cola
            self.message_queue.enqueue(message)
        
        return True
    
    def process_edge_route(self, route_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Procesar ruta en el edge."""
        if not self.enable_iot or not self.edge_processor:
            return None
        
        return self.edge_processor.process_route(route_data)
    
    def optimize_for_power(self, battery_level: float):
        """Optimizar para consumo de energía."""
        if not self.power_manager:
            return
        
        if battery_level < 20:
            self.power_manager.set_power_mode(PowerMode.ULTRA_LOW_POWER)
        elif battery_level < 50:
            self.power_manager.set_power_mode(PowerMode.LOW_POWER)
        elif battery_level < 80:
            self.power_manager.set_power_mode(PowerMode.BALANCED)
        else:
            self.power_manager.set_power_mode(PowerMode.HIGH_PERFORMANCE)
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas."""
        if not self.enable_iot:
            return {
                "iot_enabled": False
            }
        
        stats = {
            "iot_enabled": True,
            "registered_devices": len(self.devices),
            "power_mode": self.power_manager.current_mode.value if self.power_manager else None,
            "energy_consumption": self.power_manager.energy_consumption if self.power_manager else 0.0
        }
        
        if self.message_queue:
            stats["message_queue"] = self.message_queue.get_stats()
        
        if self.edge_processor:
            stats["edge_processing"] = self.edge_processor.get_stats()
        
        if self.protocol_handler:
            stats["protocol"] = self.protocol_handler.get_stats()
        
        return stats


