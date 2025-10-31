"""
IoT Integration Support for Internet of Things Connectivity
Sistema de Integración IoT para conectividad de Internet de las Cosas ultra-optimizado
"""

import asyncio
import logging
import time
import json
import socket
import threading
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.routing import APIRouter
from collections import defaultdict, deque
import uuid
import hashlib

logger = logging.getLogger(__name__)


class IoTDeviceType(Enum):
    """Tipos de dispositivos IoT"""
    SENSOR = "sensor"
    ACTUATOR = "actuator"
    GATEWAY = "gateway"
    CAMERA = "camera"
    SMART_DEVICE = "smart_device"
    WEARABLE = "wearable"
    VEHICLE = "vehicle"
    INDUSTRIAL = "industrial"
    HOME_AUTOMATION = "home_automation"
    HEALTHCARE = "healthcare"


class IoTProtocol(Enum):
    """Protocolos IoT"""
    MQTT = "mqtt"
    COAP = "coap"
    HTTP = "http"
    WEBSOCKET = "websocket"
    LORA = "lora"
    ZIGBEE = "zigbee"
    BLUETOOTH = "bluetooth"
    WIFI = "wifi"
    CELLULAR = "cellular"
    NB_IOT = "nb_iot"


class IoTDataFormat(Enum):
    """Formatos de datos IoT"""
    JSON = "json"
    XML = "xml"
    BINARY = "binary"
    CSV = "csv"
    PROTOBUF = "protobuf"
    MESSAGE_PACK = "message_pack"
    AVRO = "avro"
    PARQUET = "parquet"


class IoTDeviceStatus(Enum):
    """Estados de dispositivos IoT"""
    ONLINE = "online"
    OFFLINE = "offline"
    SLEEPING = "sleeping"
    MAINTENANCE = "maintenance"
    ERROR = "error"
    CONFIGURING = "configuring"


@dataclass
class IoTDevice:
    """Dispositivo IoT"""
    id: str
    name: str
    type: IoTDeviceType
    protocol: IoTProtocol
    status: IoTDeviceStatus
    ip_address: Optional[str]
    mac_address: Optional[str]
    location: Dict[str, float]  # lat, lon, alt
    capabilities: List[str]
    sensors: List[str]
    actuators: List[str]
    firmware_version: str
    hardware_version: str
    battery_level: Optional[float]
    signal_strength: Optional[float]
    last_seen: float
    created_at: float
    metadata: Dict[str, Any]


@dataclass
class IoTSensor:
    """Sensor IoT"""
    id: str
    device_id: str
    name: str
    type: str
    unit: str
    min_value: float
    max_value: float
    accuracy: float
    resolution: float
    sampling_rate: float
    is_active: bool
    calibration_data: Dict[str, Any]


@dataclass
class IoTActuator:
    """Actuador IoT"""
    id: str
    device_id: str
    name: str
    type: str
    control_type: str  # digital, analog, pwm
    min_value: float
    max_value: float
    current_value: float
    is_active: bool
    control_commands: List[str]


@dataclass
class IoTDataPoint:
    """Punto de datos IoT"""
    id: str
    device_id: str
    sensor_id: str
    timestamp: float
    value: Union[float, int, str, bool]
    unit: str
    quality: float  # 0-1
    metadata: Dict[str, Any]


@dataclass
class IoTCommand:
    """Comando IoT"""
    id: str
    device_id: str
    actuator_id: str
    command: str
    parameters: Dict[str, Any]
    timestamp: float
    status: str  # pending, sent, executed, failed
    response: Optional[Dict[str, Any]]


@dataclass
class IoTAlert:
    """Alerta IoT"""
    id: str
    device_id: str
    type: str
    severity: str  # low, medium, high, critical
    message: str
    timestamp: float
    is_acknowledged: bool
    metadata: Dict[str, Any]


class IoTDeviceManager:
    """Manager de dispositivos IoT"""
    
    def __init__(self):
        self.devices: Dict[str, IoTDevice] = {}
        self.sensors: Dict[str, IoTSensor] = {}
        self.actuators: Dict[str, IoTActuator] = {}
        self.data_points: Dict[str, List[IoTDataPoint]] = defaultdict(list)
        self.commands: Dict[str, IoTCommand] = {}
        self.alerts: Dict[str, IoTAlert] = {}
        self._lock = threading.Lock()
        self._heartbeat_interval = 60.0
        self._data_retention_days = 30
    
    async def register_device(self, device_info: Dict[str, Any]) -> str:
        """Registrar dispositivo IoT"""
        device_id = device_info.get("id", f"iot_device_{uuid.uuid4().hex[:8]}")
        
        device = IoTDevice(
            id=device_id,
            name=device_info["name"],
            type=IoTDeviceType(device_info["type"]),
            protocol=IoTProtocol(device_info["protocol"]),
            status=IoTDeviceStatus.ONLINE,
            ip_address=device_info.get("ip_address"),
            mac_address=device_info.get("mac_address"),
            location=device_info.get("location", {"lat": 0.0, "lon": 0.0, "alt": 0.0}),
            capabilities=device_info.get("capabilities", []),
            sensors=device_info.get("sensors", []),
            actuators=device_info.get("actuators", []),
            firmware_version=device_info.get("firmware_version", "1.0.0"),
            hardware_version=device_info.get("hardware_version", "1.0.0"),
            battery_level=device_info.get("battery_level"),
            signal_strength=device_info.get("signal_strength"),
            last_seen=time.time(),
            created_at=time.time(),
            metadata=device_info.get("metadata", {})
        )
        
        async with self._lock:
            self.devices[device_id] = device
            
            # Registrar sensores
            for sensor_info in device_info.get("sensors", []):
                await self._register_sensor(device_id, sensor_info)
            
            # Registrar actuadores
            for actuator_info in device_info.get("actuators", []):
                await self._register_actuator(device_id, actuator_info)
        
        logger.info(f"IoT device registered: {device_id} ({device.name})")
        return device_id
    
    async def _register_sensor(self, device_id: str, sensor_info: Dict[str, Any]):
        """Registrar sensor"""
        sensor_id = f"{device_id}_sensor_{sensor_info['name']}"
        
        sensor = IoTSensor(
            id=sensor_id,
            device_id=device_id,
            name=sensor_info["name"],
            type=sensor_info["type"],
            unit=sensor_info.get("unit", ""),
            min_value=sensor_info.get("min_value", 0.0),
            max_value=sensor_info.get("max_value", 100.0),
            accuracy=sensor_info.get("accuracy", 0.95),
            resolution=sensor_info.get("resolution", 0.01),
            sampling_rate=sensor_info.get("sampling_rate", 1.0),
            is_active=True,
            calibration_data=sensor_info.get("calibration_data", {})
        )
        
        self.sensors[sensor_id] = sensor
    
    async def _register_actuator(self, device_id: str, actuator_info: Dict[str, Any]):
        """Registrar actuador"""
        actuator_id = f"{device_id}_actuator_{actuator_info['name']}"
        
        actuator = IoTActuator(
            id=actuator_id,
            device_id=device_id,
            name=actuator_info["name"],
            type=actuator_info["type"],
            control_type=actuator_info.get("control_type", "digital"),
            min_value=actuator_info.get("min_value", 0.0),
            max_value=actuator_info.get("max_value", 100.0),
            current_value=actuator_info.get("current_value", 0.0),
            is_active=True,
            control_commands=actuator_info.get("control_commands", [])
        )
        
        self.actuators[actuator_id] = actuator
    
    async def update_device_status(self, device_id: str, status: IoTDeviceStatus, 
                                 battery_level: Optional[float] = None,
                                 signal_strength: Optional[float] = None):
        """Actualizar estado del dispositivo"""
        async with self._lock:
            if device_id in self.devices:
                device = self.devices[device_id]
                device.status = status
                device.last_seen = time.time()
                
                if battery_level is not None:
                    device.battery_level = battery_level
                
                if signal_strength is not None:
                    device.signal_strength = signal_strength
    
    async def send_data(self, device_id: str, sensor_id: str, value: Union[float, int, str, bool],
                       unit: str = "", quality: float = 1.0, metadata: Dict[str, Any] = None):
        """Enviar datos del sensor"""
        data_point = IoTDataPoint(
            id=f"data_{uuid.uuid4().hex[:8]}",
            device_id=device_id,
            sensor_id=sensor_id,
            timestamp=time.time(),
            value=value,
            unit=unit,
            quality=quality,
            metadata=metadata or {}
        )
        
        async with self._lock:
            self.data_points[device_id].append(data_point)
            
            # Mantener solo los últimos datos (por retención)
            max_points = self._data_retention_days * 24 * 60 * 60  # Asumiendo 1 punto por segundo
            if len(self.data_points[device_id]) > max_points:
                self.data_points[device_id] = self.data_points[device_id][-max_points:]
        
        # Verificar alertas
        await self._check_alerts(device_id, sensor_id, value)
    
    async def _check_alerts(self, device_id: str, sensor_id: str, value: Union[float, int, str, bool]):
        """Verificar alertas"""
        if device_id not in self.devices:
            return
        
        device = self.devices[device_id]
        sensor = self.sensors.get(sensor_id)
        
        if not sensor:
            return
        
        # Verificar límites del sensor
        if isinstance(value, (int, float)):
            if value < sensor.min_value or value > sensor.max_value:
                await self._create_alert(
                    device_id, "sensor_out_of_range",
                    "high", f"Sensor {sensor.name} value {value} is out of range [{sensor.min_value}, {sensor.max_value}]"
                )
        
        # Verificar batería baja
        if device.battery_level is not None and device.battery_level < 0.2:
            await self._create_alert(
                device_id, "low_battery",
                "medium", f"Device {device.name} battery level is low: {device.battery_level:.1%}"
            )
        
        # Verificar señal débil
        if device.signal_strength is not None and device.signal_strength < 0.3:
            await self._create_alert(
                device_id, "weak_signal",
                "medium", f"Device {device.name} signal strength is weak: {device.signal_strength:.1%}"
            )
    
    async def _create_alert(self, device_id: str, alert_type: str, severity: str, message: str):
        """Crear alerta"""
        alert = IoTAlert(
            id=f"alert_{uuid.uuid4().hex[:8]}",
            device_id=device_id,
            type=alert_type,
            severity=severity,
            message=message,
            timestamp=time.time(),
            is_acknowledged=False,
            metadata={}
        )
        
        async with self._lock:
            self.alerts[alert.id] = alert
    
    async def send_command(self, device_id: str, actuator_id: str, command: str,
                          parameters: Dict[str, Any] = None) -> str:
        """Enviar comando al actuador"""
        command_id = f"cmd_{uuid.uuid4().hex[:8]}"
        
        cmd = IoTCommand(
            id=command_id,
            device_id=device_id,
            actuator_id=actuator_id,
            command=command,
            parameters=parameters or {},
            timestamp=time.time(),
            status="pending",
            response=None
        )
        
        async with self._lock:
            self.commands[command_id] = cmd
        
        # Simular envío del comando
        await self._execute_command(cmd)
        
        return command_id
    
    async def _execute_command(self, command: IoTCommand):
        """Ejecutar comando"""
        try:
            # Simular ejecución del comando
            await asyncio.sleep(0.1)  # Simular latencia de red
            
            command.status = "sent"
            
            # Simular respuesta del dispositivo
            await asyncio.sleep(0.2)
            
            command.status = "executed"
            command.response = {
                "success": True,
                "execution_time": 0.3,
                "result": f"Command {command.command} executed successfully"
            }
            
        except Exception as e:
            command.status = "failed"
            command.response = {
                "success": False,
                "error": str(e)
            }
    
    async def get_device_data(self, device_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Obtener datos del dispositivo"""
        async with self._lock:
            if device_id not in self.data_points:
                return []
            
            data_points = self.data_points[device_id][-limit:]
            return [
                {
                    "id": dp.id,
                    "sensor_id": dp.sensor_id,
                    "timestamp": dp.timestamp,
                    "value": dp.value,
                    "unit": dp.unit,
                    "quality": dp.quality,
                    "metadata": dp.metadata
                }
                for dp in data_points
            ]
    
    async def get_device_stats(self, device_id: str) -> Dict[str, Any]:
        """Obtener estadísticas del dispositivo"""
        async with self._lock:
            if device_id not in self.devices:
                return {}
            
            device = self.devices[device_id]
            data_points = self.data_points.get(device_id, [])
            
            # Calcular estadísticas de datos
            if data_points:
                values = [dp.value for dp in data_points if isinstance(dp.value, (int, float))]
                if values:
                    stats = {
                        "count": len(values),
                        "min": min(values),
                        "max": max(values),
                        "avg": sum(values) / len(values),
                        "latest": values[-1] if values else None
                    }
                else:
                    stats = {"count": len(data_points)}
            else:
                stats = {"count": 0}
            
            return {
                "device_id": device_id,
                "name": device.name,
                "type": device.type.value,
                "status": device.status.value,
                "battery_level": device.battery_level,
                "signal_strength": device.signal_strength,
                "last_seen": device.last_seen,
                "sensors_count": len(device.sensors),
                "actuators_count": len(device.actuators),
                "data_points": stats,
                "alerts_count": len([a for a in self.alerts.values() if a.device_id == device_id])
            }


class IoTProtocolHandler:
    """Handler de protocolos IoT"""
    
    def __init__(self):
        self.handlers: Dict[IoTProtocol, Callable] = {
            IoTProtocol.MQTT: self._handle_mqtt,
            IoTProtocol.COAP: self._handle_coap,
            IoTProtocol.HTTP: self._handle_http,
            IoTProtocol.WEBSOCKET: self._handle_websocket,
            IoTProtocol.LORA: self._handle_lora,
            IoTProtocol.ZIGBEE: self._handle_zigbee,
            IoTProtocol.BLUETOOTH: self._handle_bluetooth,
            IoTProtocol.WIFI: self._handle_wifi,
            IoTProtocol.CELLULAR: self._handle_cellular,
            IoTProtocol.NB_IOT: self._handle_nb_iot
        }
    
    async def handle_data(self, protocol: IoTProtocol, data: Dict[str, Any]) -> Dict[str, Any]:
        """Manejar datos según protocolo"""
        handler = self.handlers.get(protocol)
        if handler:
            return await handler(data)
        else:
            raise ValueError(f"Unsupported protocol: {protocol}")
    
    async def _handle_mqtt(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Manejar datos MQTT"""
        return {
            "protocol": "mqtt",
            "topic": data.get("topic", ""),
            "payload": data.get("payload", ""),
            "qos": data.get("qos", 0),
            "retain": data.get("retain", False),
            "processed_at": time.time()
        }
    
    async def _handle_coap(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Manejar datos CoAP"""
        return {
            "protocol": "coap",
            "method": data.get("method", "GET"),
            "uri": data.get("uri", ""),
            "payload": data.get("payload", ""),
            "code": data.get("code", "2.05"),
            "processed_at": time.time()
        }
    
    async def _handle_http(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Manejar datos HTTP"""
        return {
            "protocol": "http",
            "method": data.get("method", "GET"),
            "url": data.get("url", ""),
            "headers": data.get("headers", {}),
            "body": data.get("body", ""),
            "processed_at": time.time()
        }
    
    async def _handle_websocket(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Manejar datos WebSocket"""
        return {
            "protocol": "websocket",
            "message_type": data.get("message_type", "text"),
            "data": data.get("data", ""),
            "processed_at": time.time()
        }
    
    async def _handle_lora(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Manejar datos LoRa"""
        return {
            "protocol": "lora",
            "frequency": data.get("frequency", 868.1),
            "spreading_factor": data.get("spreading_factor", 7),
            "bandwidth": data.get("bandwidth", 125),
            "payload": data.get("payload", ""),
            "processed_at": time.time()
        }
    
    async def _handle_zigbee(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Manejar datos Zigbee"""
        return {
            "protocol": "zigbee",
            "cluster_id": data.get("cluster_id", ""),
            "endpoint": data.get("endpoint", 1),
            "payload": data.get("payload", ""),
            "processed_at": time.time()
        }
    
    async def _handle_bluetooth(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Manejar datos Bluetooth"""
        return {
            "protocol": "bluetooth",
            "service_uuid": data.get("service_uuid", ""),
            "characteristic_uuid": data.get("characteristic_uuid", ""),
            "data": data.get("data", ""),
            "processed_at": time.time()
        }
    
    async def _handle_wifi(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Manejar datos WiFi"""
        return {
            "protocol": "wifi",
            "ssid": data.get("ssid", ""),
            "rssi": data.get("rssi", 0),
            "data": data.get("data", ""),
            "processed_at": time.time()
        }
    
    async def _handle_cellular(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Manejar datos Cellular"""
        return {
            "protocol": "cellular",
            "operator": data.get("operator", ""),
            "signal_strength": data.get("signal_strength", 0),
            "data": data.get("data", ""),
            "processed_at": time.time()
        }
    
    async def _handle_nb_iot(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Manejar datos NB-IoT"""
        return {
            "protocol": "nb_iot",
            "operator": data.get("operator", ""),
            "signal_strength": data.get("signal_strength", 0),
            "data": data.get("data", ""),
            "processed_at": time.time()
        }


class IoTDataProcessor:
    """Procesador de datos IoT"""
    
    def __init__(self):
        self.processors: Dict[str, Callable] = {
            "filter": self._filter_data,
            "aggregate": self._aggregate_data,
            "transform": self._transform_data,
            "validate": self._validate_data,
            "enrich": self._enrich_data
        }
    
    async def process_data(self, data: List[IoTDataPoint], operations: List[Dict[str, Any]]) -> List[IoTDataPoint]:
        """Procesar datos IoT"""
        result = data.copy()
        
        for operation in operations:
            op_type = operation.get("type")
            params = operation.get("parameters", {})
            
            if op_type in self.processors:
                result = await self.processors[op_type](result, params)
        
        return result
    
    async def _filter_data(self, data: List[IoTDataPoint], params: Dict[str, Any]) -> List[IoTDataPoint]:
        """Filtrar datos"""
        filtered = []
        
        for dp in data:
            # Filtro por calidad
            if "min_quality" in params and dp.quality < params["min_quality"]:
                continue
            
            # Filtro por rango de tiempo
            if "start_time" in params and dp.timestamp < params["start_time"]:
                continue
            if "end_time" in params and dp.timestamp > params["end_time"]:
                continue
            
            # Filtro por valor
            if "min_value" in params and isinstance(dp.value, (int, float)) and dp.value < params["min_value"]:
                continue
            if "max_value" in params and isinstance(dp.value, (int, float)) and dp.value > params["max_value"]:
                continue
            
            filtered.append(dp)
        
        return filtered
    
    async def _aggregate_data(self, data: List[IoTDataPoint], params: Dict[str, Any]) -> List[IoTDataPoint]:
        """Agregar datos"""
        if not data:
            return []
        
        aggregation_type = params.get("type", "average")
        window_size = params.get("window_size", 60)  # segundos
        
        # Agrupar por ventana de tiempo
        groups = defaultdict(list)
        for dp in data:
            window_start = int(dp.timestamp // window_size) * window_size
            groups[window_start].append(dp)
        
        aggregated = []
        for window_start, group_data in groups.items():
            if not group_data:
                continue
            
            # Calcular agregación
            if aggregation_type == "average":
                if all(isinstance(dp.value, (int, float)) for dp in group_data):
                    avg_value = sum(dp.value for dp in group_data) / len(group_data)
                    aggregated.append(IoTDataPoint(
                        id=f"agg_{uuid.uuid4().hex[:8]}",
                        device_id=group_data[0].device_id,
                        sensor_id=group_data[0].sensor_id,
                        timestamp=window_start,
                        value=avg_value,
                        unit=group_data[0].unit,
                        quality=sum(dp.quality for dp in group_data) / len(group_data),
                        metadata={"aggregation_type": "average", "count": len(group_data)}
                    ))
            elif aggregation_type == "sum":
                if all(isinstance(dp.value, (int, float)) for dp in group_data):
                    sum_value = sum(dp.value for dp in group_data)
                    aggregated.append(IoTDataPoint(
                        id=f"agg_{uuid.uuid4().hex[:8]}",
                        device_id=group_data[0].device_id,
                        sensor_id=group_data[0].sensor_id,
                        timestamp=window_start,
                        value=sum_value,
                        unit=group_data[0].unit,
                        quality=sum(dp.quality for dp in group_data) / len(group_data),
                        metadata={"aggregation_type": "sum", "count": len(group_data)}
                    ))
        
        return aggregated
    
    async def _transform_data(self, data: List[IoTDataPoint], params: Dict[str, Any]) -> List[IoTDataPoint]:
        """Transformar datos"""
        transform_type = params.get("type", "scale")
        
        transformed = []
        for dp in data:
            new_dp = IoTDataPoint(
                id=dp.id,
                device_id=dp.device_id,
                sensor_id=dp.sensor_id,
                timestamp=dp.timestamp,
                value=dp.value,
                unit=dp.unit,
                quality=dp.quality,
                metadata=dp.metadata.copy()
            )
            
            if transform_type == "scale" and isinstance(dp.value, (int, float)):
                factor = params.get("factor", 1.0)
                new_dp.value = dp.value * factor
            elif transform_type == "offset" and isinstance(dp.value, (int, float)):
                offset = params.get("offset", 0.0)
                new_dp.value = dp.value + offset
            elif transform_type == "unit_conversion":
                # Conversión de unidades simple
                from_unit = params.get("from_unit", "")
                to_unit = params.get("to_unit", "")
                if from_unit == "celsius" and to_unit == "fahrenheit" and isinstance(dp.value, (int, float)):
                    new_dp.value = (dp.value * 9/5) + 32
                    new_dp.unit = to_unit
            
            transformed.append(new_dp)
        
        return transformed
    
    async def _validate_data(self, data: List[IoTDataPoint], params: Dict[str, Any]) -> List[IoTDataPoint]:
        """Validar datos"""
        validated = []
        
        for dp in data:
            is_valid = True
            
            # Validar calidad
            if "min_quality" in params and dp.quality < params["min_quality"]:
                is_valid = False
            
            # Validar rango de valores
            if "min_value" in params and isinstance(dp.value, (int, float)) and dp.value < params["min_value"]:
                is_valid = False
            if "max_value" in params and isinstance(dp.value, (int, float)) and dp.value > params["max_value"]:
                is_valid = False
            
            # Validar timestamp
            if "max_age" in params and (time.time() - dp.timestamp) > params["max_age"]:
                is_valid = False
            
            if is_valid:
                validated.append(dp)
        
        return validated
    
    async def _enrich_data(self, data: List[IoTDataPoint], params: Dict[str, Any]) -> List[IoTDataPoint]:
        """Enriquecer datos"""
        enriched = []
        
        for dp in data:
            new_dp = IoTDataPoint(
                id=dp.id,
                device_id=dp.device_id,
                sensor_id=dp.sensor_id,
                timestamp=dp.timestamp,
                value=dp.value,
                unit=dp.unit,
                quality=dp.quality,
                metadata=dp.metadata.copy()
            )
            
            # Agregar información adicional
            new_dp.metadata.update({
                "processed_at": time.time(),
                "enrichment": params.get("enrichment_data", {})
            })
            
            enriched.append(new_dp)
        
        return enriched


class IoTIntegrationManager:
    """Manager principal de integración IoT"""
    
    def __init__(self):
        self.device_manager = IoTDeviceManager()
        self.protocol_handler = IoTProtocolHandler()
        self.data_processor = IoTDataProcessor()
        self.is_running = False
        self._cleanup_task = None
    
    async def start(self):
        """Iniciar IoT integration manager"""
        try:
            self.is_running = True
            
            # Iniciar tareas de limpieza
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            
            logger.info("IoT integration manager started")
            
        except Exception as e:
            logger.error(f"Error starting IoT integration manager: {e}")
            raise
    
    async def stop(self):
        """Detener IoT integration manager"""
        try:
            self.is_running = False
            
            if self._cleanup_task:
                self._cleanup_task.cancel()
                try:
                    await self._cleanup_task
                except asyncio.CancelledError:
                    pass
            
            logger.info("IoT integration manager stopped")
            
        except Exception as e:
            logger.error(f"Error stopping IoT integration manager: {e}")
    
    async def _cleanup_loop(self):
        """Loop de limpieza"""
        while self.is_running:
            try:
                # Limpiar datos antiguos
                await self._cleanup_old_data()
                
                # Limpiar dispositivos inactivos
                await self._cleanup_inactive_devices()
                
                await asyncio.sleep(3600)  # Limpiar cada hora
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(3600)
    
    async def _cleanup_old_data(self):
        """Limpiar datos antiguos"""
        cutoff_time = time.time() - (self.device_manager._data_retention_days * 24 * 60 * 60)
        
        async with self.device_manager._lock:
            for device_id in list(self.device_manager.data_points.keys()):
                self.device_manager.data_points[device_id] = [
                    dp for dp in self.device_manager.data_points[device_id]
                    if dp.timestamp > cutoff_time
                ]
    
    async def _cleanup_inactive_devices(self):
        """Limpiar dispositivos inactivos"""
        cutoff_time = time.time() - (24 * 60 * 60)  # 24 horas
        
        async with self.device_manager._lock:
            inactive_devices = []
            for device_id, device in self.device_manager.devices.items():
                if device.last_seen < cutoff_time:
                    inactive_devices.append(device_id)
            
            for device_id in inactive_devices:
                device = self.device_manager.devices[device_id]
                device.status = IoTDeviceStatus.OFFLINE
                logger.info(f"Device {device_id} marked as offline due to inactivity")
    
    async def get_system_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del sistema"""
        return {
            "is_running": self.is_running,
            "devices": {
                "total": len(self.device_manager.devices),
                "online": sum(1 for d in self.device_manager.devices.values() if d.status == IoTDeviceStatus.ONLINE),
                "offline": sum(1 for d in self.device_manager.devices.values() if d.status == IoTDeviceStatus.OFFLINE),
                "by_type": {
                    device_type.value: sum(1 for d in self.device_manager.devices.values() if d.type == device_type)
                    for device_type in IoTDeviceType
                }
            },
            "sensors": len(self.device_manager.sensors),
            "actuators": len(self.device_manager.actuators),
            "data_points": sum(len(points) for points in self.device_manager.data_points.values()),
            "commands": len(self.device_manager.commands),
            "alerts": len(self.device_manager.alerts)
        }


# Instancia global del manager de integración IoT
iot_integration_manager = IoTIntegrationManager()


# Router para endpoints de integración IoT
iot_integration_router = APIRouter()


@iot_integration_router.post("/iot/devices/register")
async def register_iot_device_endpoint(device_data: dict):
    """Registrar dispositivo IoT"""
    try:
        device_id = await iot_integration_manager.device_manager.register_device(device_data)
        
        return {
            "message": "IoT device registered successfully",
            "device_id": device_id,
            "name": device_data["name"],
            "type": device_data["type"],
            "protocol": device_data["protocol"]
        }
        
    except Exception as e:
        logger.error(f"Error registering IoT device: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to register IoT device: {str(e)}")


@iot_integration_router.get("/iot/devices")
async def get_iot_devices_endpoint():
    """Obtener dispositivos IoT"""
    try:
        devices = iot_integration_manager.device_manager.devices
        return {
            "devices": [
                {
                    "id": device.id,
                    "name": device.name,
                    "type": device.type.value,
                    "protocol": device.protocol.value,
                    "status": device.status.value,
                    "ip_address": device.ip_address,
                    "location": device.location,
                    "battery_level": device.battery_level,
                    "signal_strength": device.signal_strength,
                    "last_seen": device.last_seen,
                    "created_at": device.created_at
                }
                for device in devices.values()
            ]
        }
    except Exception as e:
        logger.error(f"Error getting IoT devices: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get IoT devices: {str(e)}")


@iot_integration_router.get("/iot/devices/{device_id}")
async def get_iot_device_endpoint(device_id: str):
    """Obtener dispositivo IoT específico"""
    try:
        if device_id not in iot_integration_manager.device_manager.devices:
            raise HTTPException(status_code=404, detail="IoT device not found")
        
        device = iot_integration_manager.device_manager.devices[device_id]
        stats = await iot_integration_manager.device_manager.get_device_stats(device_id)
        
        return {
            "device": {
                "id": device.id,
                "name": device.name,
                "type": device.type.value,
                "protocol": device.protocol.value,
                "status": device.status.value,
                "ip_address": device.ip_address,
                "mac_address": device.mac_address,
                "location": device.location,
                "capabilities": device.capabilities,
                "sensors": device.sensors,
                "actuators": device.actuators,
                "firmware_version": device.firmware_version,
                "hardware_version": device.hardware_version,
                "battery_level": device.battery_level,
                "signal_strength": device.signal_strength,
                "last_seen": device.last_seen,
                "created_at": device.created_at,
                "metadata": device.metadata
            },
            "stats": stats
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting IoT device: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get IoT device: {str(e)}")


@iot_integration_router.post("/iot/devices/{device_id}/data")
async def send_iot_data_endpoint(device_id: str, data: dict):
    """Enviar datos IoT"""
    try:
        sensor_id = data["sensor_id"]
        value = data["value"]
        unit = data.get("unit", "")
        quality = data.get("quality", 1.0)
        metadata = data.get("metadata", {})
        
        await iot_integration_manager.device_manager.send_data(
            device_id, sensor_id, value, unit, quality, metadata
        )
        
        return {
            "message": "IoT data sent successfully",
            "device_id": device_id,
            "sensor_id": sensor_id,
            "value": value
        }
        
    except Exception as e:
        logger.error(f"Error sending IoT data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to send IoT data: {str(e)}")


@iot_integration_router.get("/iot/devices/{device_id}/data")
async def get_iot_device_data_endpoint(device_id: str, limit: int = 100):
    """Obtener datos del dispositivo IoT"""
    try:
        data = await iot_integration_manager.device_manager.get_device_data(device_id, limit)
        return {
            "device_id": device_id,
            "data_points": data,
            "count": len(data)
        }
    except Exception as e:
        logger.error(f"Error getting IoT device data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get IoT device data: {str(e)}")


@iot_integration_router.post("/iot/devices/{device_id}/commands")
async def send_iot_command_endpoint(device_id: str, command_data: dict):
    """Enviar comando IoT"""
    try:
        actuator_id = command_data["actuator_id"]
        command = command_data["command"]
        parameters = command_data.get("parameters", {})
        
        command_id = await iot_integration_manager.device_manager.send_command(
            device_id, actuator_id, command, parameters
        )
        
        return {
            "message": "IoT command sent successfully",
            "command_id": command_id,
            "device_id": device_id,
            "actuator_id": actuator_id,
            "command": command
        }
        
    except Exception as e:
        logger.error(f"Error sending IoT command: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to send IoT command: {str(e)}")


@iot_integration_router.get("/iot/commands/{command_id}")
async def get_iot_command_endpoint(command_id: str):
    """Obtener comando IoT"""
    try:
        if command_id not in iot_integration_manager.device_manager.commands:
            raise HTTPException(status_code=404, detail="IoT command not found")
        
        command = iot_integration_manager.device_manager.commands[command_id]
        
        return {
            "id": command.id,
            "device_id": command.device_id,
            "actuator_id": command.actuator_id,
            "command": command.command,
            "parameters": command.parameters,
            "timestamp": command.timestamp,
            "status": command.status,
            "response": command.response
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting IoT command: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get IoT command: {str(e)}")


@iot_integration_router.get("/iot/alerts")
async def get_iot_alerts_endpoint(severity: Optional[str] = None, acknowledged: Optional[bool] = None):
    """Obtener alertas IoT"""
    try:
        alerts = iot_integration_manager.device_manager.alerts
        
        filtered_alerts = []
        for alert in alerts.values():
            if severity and alert.severity != severity:
                continue
            if acknowledged is not None and alert.is_acknowledged != acknowledged:
                continue
            
            filtered_alerts.append({
                "id": alert.id,
                "device_id": alert.device_id,
                "type": alert.type,
                "severity": alert.severity,
                "message": alert.message,
                "timestamp": alert.timestamp,
                "is_acknowledged": alert.is_acknowledged,
                "metadata": alert.metadata
            })
        
        return {
            "alerts": filtered_alerts,
            "count": len(filtered_alerts)
        }
    except Exception as e:
        logger.error(f"Error getting IoT alerts: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get IoT alerts: {str(e)}")


@iot_integration_router.post("/iot/alerts/{alert_id}/acknowledge")
async def acknowledge_iot_alert_endpoint(alert_id: str):
    """Reconocer alerta IoT"""
    try:
        if alert_id not in iot_integration_manager.device_manager.alerts:
            raise HTTPException(status_code=404, detail="IoT alert not found")
        
        alert = iot_integration_manager.device_manager.alerts[alert_id]
        alert.is_acknowledged = True
        
        return {
            "message": "IoT alert acknowledged successfully",
            "alert_id": alert_id
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error acknowledging IoT alert: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to acknowledge IoT alert: {str(e)}")


@iot_integration_router.post("/iot/data/process")
async def process_iot_data_endpoint(processing_data: dict):
    """Procesar datos IoT"""
    try:
        device_id = processing_data["device_id"]
        operations = processing_data["operations"]
        
        # Obtener datos del dispositivo
        raw_data = await iot_integration_manager.device_manager.get_device_data(device_id, 1000)
        
        # Convertir a IoTDataPoint objects
        data_points = []
        for dp_data in raw_data:
            dp = IoTDataPoint(
                id=dp_data["id"],
                device_id=dp_data["device_id"],
                sensor_id=dp_data["sensor_id"],
                timestamp=dp_data["timestamp"],
                value=dp_data["value"],
                unit=dp_data["unit"],
                quality=dp_data["quality"],
                metadata=dp_data["metadata"]
            )
            data_points.append(dp)
        
        # Procesar datos
        processed_data = await iot_integration_manager.data_processor.process_data(data_points, operations)
        
        return {
            "message": "IoT data processed successfully",
            "device_id": device_id,
            "original_count": len(data_points),
            "processed_count": len(processed_data),
            "processed_data": [
                {
                    "id": dp.id,
                    "sensor_id": dp.sensor_id,
                    "timestamp": dp.timestamp,
                    "value": dp.value,
                    "unit": dp.unit,
                    "quality": dp.quality,
                    "metadata": dp.metadata
                }
                for dp in processed_data
            ]
        }
        
    except Exception as e:
        logger.error(f"Error processing IoT data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process IoT data: {str(e)}")


@iot_integration_router.get("/iot/stats")
async def get_iot_integration_stats_endpoint():
    """Obtener estadísticas de integración IoT"""
    try:
        stats = await iot_integration_manager.get_system_stats()
        return stats
    except Exception as e:
        logger.error(f"Error getting IoT integration stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get IoT integration stats: {str(e)}")


# Funciones de utilidad para integración
async def start_iot_integration():
    """Iniciar integración IoT"""
    await iot_integration_manager.start()


async def stop_iot_integration():
    """Detener integración IoT"""
    await iot_integration_manager.stop()


async def register_iot_device(device_info: Dict[str, Any]) -> str:
    """Registrar dispositivo IoT"""
    return await iot_integration_manager.device_manager.register_device(device_info)


async def send_iot_data(device_id: str, sensor_id: str, value: Union[float, int, str, bool],
                       unit: str = "", quality: float = 1.0, metadata: Dict[str, Any] = None):
    """Enviar datos IoT"""
    await iot_integration_manager.device_manager.send_data(device_id, sensor_id, value, unit, quality, metadata)


async def send_iot_command(device_id: str, actuator_id: str, command: str,
                          parameters: Dict[str, Any] = None) -> str:
    """Enviar comando IoT"""
    return await iot_integration_manager.device_manager.send_command(device_id, actuator_id, command, parameters)


async def get_iot_integration_stats() -> Dict[str, Any]:
    """Obtener estadísticas de integración IoT"""
    return await iot_integration_manager.get_system_stats()


logger.info("IoT integration support module loaded successfully")

