"""
IoT Engine - Motor de Internet de las Cosas
"""

import asyncio
import logging
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import uuid
from pathlib import Path
import aiohttp
import hashlib
import struct
import socket
from decimal import Decimal
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class DeviceType(Enum):
    """Tipos de dispositivos IoT."""
    SENSOR = "sensor"
    ACTUATOR = "actuator"
    GATEWAY = "gateway"
    CAMERA = "camera"
    SMART_DEVICE = "smart_device"
    WEARABLE = "wearable"
    VEHICLE = "vehicle"
    INDUSTRIAL = "industrial"


class ProtocolType(Enum):
    """Tipos de protocolos IoT."""
    MQTT = "mqtt"
    HTTP = "http"
    COAP = "coap"
    WEBSOCKET = "websocket"
    MODBUS = "modbus"
    OPCUA = "opcua"
    ZIGBEE = "zigbee"
    LORAWAN = "lorawan"


class DataType(Enum):
    """Tipos de datos IoT."""
    TEMPERATURE = "temperature"
    HUMIDITY = "humidity"
    PRESSURE = "pressure"
    LIGHT = "light"
    MOTION = "motion"
    SOUND = "sound"
    VIBRATION = "vibration"
    LOCATION = "location"
    BATTERY = "battery"
    CUSTOM = "custom"


@dataclass
class IoTDevice:
    """Dispositivo IoT."""
    device_id: str
    name: str
    device_type: DeviceType
    protocol: ProtocolType
    ip_address: str
    port: int
    status: str = "offline"
    last_seen: datetime = field(default_factory=datetime.now)
    battery_level: Optional[int] = None
    location: Optional[Dict[str, float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class IoTData:
    """Datos de dispositivo IoT."""
    data_id: str
    device_id: str
    data_type: DataType
    value: Any
    unit: str
    timestamp: datetime = field(default_factory=datetime.now)
    quality: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IoTAlert:
    """Alerta IoT."""
    alert_id: str
    device_id: str
    alert_type: str
    severity: str
    message: str
    threshold: Optional[float] = None
    current_value: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    acknowledged: bool = False
    resolved: bool = False


class IoTEngine:
    """
    Motor de Internet de las Cosas.
    """
    
    def __init__(self, config_directory: str = "iot_config"):
        """Inicializar motor IoT."""
        self.config_directory = Path(config_directory)
        self.config_directory.mkdir(exist_ok=True)
        
        # Configuración de protocolos
        self.protocols = {
            ProtocolType.MQTT: {
                "port": 1883,
                "secure_port": 8883,
                "keep_alive": 60,
                "qos": 1
            },
            ProtocolType.HTTP: {
                "port": 80,
                "secure_port": 443,
                "timeout": 30
            },
            ProtocolType.COAP: {
                "port": 5683,
                "secure_port": 5684,
                "timeout": 10
            },
            ProtocolType.WEBSOCKET: {
                "port": 8080,
                "secure_port": 8443,
                "ping_interval": 20
            }
        }
        
        # Dispositivos y datos
        self.devices: Dict[str, IoTDevice] = {}
        self.data_streams: Dict[str, List[IoTData]] = {}
        self.alerts: Dict[str, IoTAlert] = {}
        
        # Configuración
        self.data_retention_days = 30
        self.alert_thresholds = {
            DataType.TEMPERATURE: {"min": -10, "max": 50},
            DataType.HUMIDITY: {"min": 0, "max": 100},
            DataType.PRESSURE: {"min": 800, "max": 1200},
            DataType.BATTERY: {"min": 10, "max": 100}
        }
        
        # Estadísticas
        self.stats = {
            "total_devices": 0,
            "online_devices": 0,
            "total_data_points": 0,
            "active_alerts": 0,
            "data_throughput": 0,
            "start_time": datetime.now()
        }
        
        # Inicializar conexiones
        self._initialize_connections()
        
        logger.info("IoTEngine inicializado")
    
    async def initialize(self):
        """Inicializar el motor IoT."""
        try:
            # Cargar dispositivos existentes
            self._load_devices()
            
            # Iniciar monitoreo de dispositivos
            asyncio.create_task(self._monitor_devices())
            
            # Iniciar procesamiento de datos
            asyncio.create_task(self._process_data_streams())
            
            logger.info("IoTEngine inicializado exitosamente")
            
        except Exception as e:
            logger.error(f"Error al inicializar IoTEngine: {e}")
            raise
    
    async def shutdown(self):
        """Cerrar el motor IoT."""
        try:
            # Guardar dispositivos
            await self._save_devices()
            
            # Cerrar conexiones
            await self._close_connections()
            
            logger.info("IoTEngine cerrado")
            
        except Exception as e:
            logger.error(f"Error al cerrar IoTEngine: {e}")
    
    def _initialize_connections(self):
        """Inicializar conexiones IoT."""
        try:
            # Inicializar servidores para diferentes protocolos
            self.servers = {}
            self.connections = {}
            
            logger.info("Conexiones IoT inicializadas")
            
        except Exception as e:
            logger.error(f"Error al inicializar conexiones IoT: {e}")
    
    async def _monitor_devices(self):
        """Monitorear dispositivos IoT."""
        while True:
            try:
                for device_id, device in self.devices.items():
                    # Verificar estado del dispositivo
                    is_online = await self._check_device_status(device)
                    
                    if is_online and device.status == "offline":
                        device.status = "online"
                        device.last_seen = datetime.now()
                        logger.info(f"Dispositivo {device.name} conectado")
                        
                    elif not is_online and device.status == "online":
                        device.status = "offline"
                        logger.warning(f"Dispositivo {device.name} desconectado")
                        
                        # Crear alerta de desconexión
                        await self._create_alert(
                            device_id=device_id,
                            alert_type="device_offline",
                            severity="warning",
                            message=f"Dispositivo {device.name} se desconectó"
                        )
                
                # Actualizar estadísticas
                self.stats["online_devices"] = sum(
                    1 for device in self.devices.values() 
                    if device.status == "online"
                )
                
                await asyncio.sleep(30)  # Verificar cada 30 segundos
                
            except Exception as e:
                logger.error(f"Error en monitoreo de dispositivos: {e}")
                await asyncio.sleep(60)
    
    async def _process_data_streams(self):
        """Procesar flujos de datos IoT."""
        while True:
            try:
                for device_id, data_list in self.data_streams.items():
                    if not data_list:
                        continue
                    
                    # Procesar datos recientes
                    recent_data = [
                        data for data in data_list 
                        if (datetime.now() - data.timestamp).total_seconds() < 300
                    ]
                    
                    for data in recent_data:
                        # Verificar umbrales y crear alertas
                        await self._check_data_thresholds(data)
                        
                        # Actualizar estadísticas
                        self.stats["total_data_points"] += 1
                        self.stats["data_throughput"] += 1
                    
                    # Limpiar datos antiguos
                    cutoff_time = datetime.now() - timedelta(days=self.data_retention_days)
                    self.data_streams[device_id] = [
                        data for data in data_list 
                        if data.timestamp > cutoff_time
                    ]
                
                await asyncio.sleep(10)  # Procesar cada 10 segundos
                
            except Exception as e:
                logger.error(f"Error en procesamiento de datos: {e}")
                await asyncio.sleep(30)
    
    async def _check_device_status(self, device: IoTDevice) -> bool:
        """Verificar estado de dispositivo."""
        try:
            if device.protocol == ProtocolType.HTTP:
                async with aiohttp.ClientSession() as session:
                    url = f"http://{device.ip_address}:{device.port}/health"
                    async with session.get(url, timeout=5) as response:
                        return response.status == 200
                        
            elif device.protocol == ProtocolType.MQTT:
                # Simular verificación MQTT
                return True
                
            elif device.protocol == ProtocolType.WEBSOCKET:
                # Simular verificación WebSocket
                return True
                
            else:
                # Verificación básica de conectividad
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5)
                result = sock.connect_ex((device.ip_address, device.port))
                sock.close()
                return result == 0
                
        except Exception as e:
            logger.debug(f"Error al verificar dispositivo {device.name}: {e}")
            return False
    
    async def _check_data_thresholds(self, data: IoTData):
        """Verificar umbrales de datos."""
        try:
            if data.data_type not in self.alert_thresholds:
                return
            
            thresholds = self.alert_thresholds[data.data_type]
            value = float(data.value)
            
            # Verificar umbrales
            if value < thresholds["min"]:
                await self._create_alert(
                    device_id=data.device_id,
                    alert_type="threshold_low",
                    severity="warning",
                    message=f"Valor {data.data_type.value} por debajo del umbral: {value} < {thresholds['min']}",
                    threshold=thresholds["min"],
                    current_value=value
                )
                
            elif value > thresholds["max"]:
                await self._create_alert(
                    device_id=data.device_id,
                    alert_type="threshold_high",
                    severity="critical",
                    message=f"Valor {data.data_type.value} por encima del umbral: {value} > {thresholds['max']}",
                    threshold=thresholds["max"],
                    current_value=value
                )
                
        except Exception as e:
            logger.error(f"Error al verificar umbrales: {e}")
    
    async def _create_alert(self, device_id: str, alert_type: str, severity: str, 
                          message: str, threshold: Optional[float] = None, 
                          current_value: Optional[float] = None):
        """Crear alerta IoT."""
        try:
            alert_id = str(uuid.uuid4())
            
            alert = IoTAlert(
                alert_id=alert_id,
                device_id=device_id,
                alert_type=alert_type,
                severity=severity,
                message=message,
                threshold=threshold,
                current_value=current_value
            )
            
            self.alerts[alert_id] = alert
            self.stats["active_alerts"] += 1
            
            logger.warning(f"Alerta IoT creada: {message}")
            
        except Exception as e:
            logger.error(f"Error al crear alerta: {e}")
    
    def _load_devices(self):
        """Cargar dispositivos existentes."""
        try:
            devices_file = self.config_directory / "devices.json"
            if devices_file.exists():
                with open(devices_file, 'r') as f:
                    devices_data = json.load(f)
                
                for device_id, data in devices_data.items():
                    data['device_type'] = DeviceType(data['device_type'])
                    data['protocol'] = ProtocolType(data['protocol'])
                    data['created_at'] = datetime.fromisoformat(data['created_at'])
                    data['last_seen'] = datetime.fromisoformat(data['last_seen'])
                    
                    self.devices[device_id] = IoTDevice(**data)
                
                logger.info(f"Cargados {len(self.devices)} dispositivos IoT")
                
        except Exception as e:
            logger.error(f"Error al cargar dispositivos: {e}")
    
    async def _save_devices(self):
        """Guardar dispositivos."""
        try:
            devices_file = self.config_directory / "devices.json"
            
            devices_data = {}
            for device_id, device in self.devices.items():
                data = device.__dict__.copy()
                data['device_type'] = data['device_type'].value
                data['protocol'] = data['protocol'].value
                data['created_at'] = data['created_at'].isoformat()
                data['last_seen'] = data['last_seen'].isoformat()
                devices_data[device_id] = data
            
            with open(devices_file, 'w') as f:
                json.dump(devices_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error al guardar dispositivos: {e}")
    
    async def _close_connections(self):
        """Cerrar conexiones IoT."""
        try:
            # Cerrar servidores y conexiones
            for server in self.servers.values():
                if hasattr(server, 'close'):
                    await server.close()
            
            for connection in self.connections.values():
                if hasattr(connection, 'close'):
                    await connection.close()
                    
        except Exception as e:
            logger.error(f"Error al cerrar conexiones: {e}")
    
    async def register_device(
        self,
        name: str,
        device_type: DeviceType,
        protocol: ProtocolType,
        ip_address: str,
        port: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Registrar nuevo dispositivo IoT."""
        try:
            device_id = str(uuid.uuid4())
            
            device = IoTDevice(
                device_id=device_id,
                name=name,
                device_type=device_type,
                protocol=protocol,
                ip_address=ip_address,
                port=port,
                metadata=metadata or {}
            )
            
            # Verificar conectividad inicial
            device.status = "online" if await self._check_device_status(device) else "offline"
            
            self.devices[device_id] = device
            self.data_streams[device_id] = []
            self.stats["total_devices"] += 1
            
            logger.info(f"Dispositivo IoT registrado: {name} ({device_type.value})")
            return device_id
            
        except Exception as e:
            logger.error(f"Error al registrar dispositivo IoT: {e}")
            raise
    
    async def send_data(
        self,
        device_id: str,
        data_type: DataType,
        value: Any,
        unit: str = "",
        quality: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Enviar datos de dispositivo IoT."""
        try:
            if device_id not in self.devices:
                raise ValueError(f"Dispositivo {device_id} no encontrado")
            
            data_id = str(uuid.uuid4())
            
            data = IoTData(
                data_id=data_id,
                device_id=device_id,
                data_type=data_type,
                value=value,
                unit=unit,
                quality=quality,
                metadata=metadata or {}
            )
            
            # Agregar a flujo de datos
            if device_id not in self.data_streams:
                self.data_streams[device_id] = []
            
            self.data_streams[device_id].append(data)
            
            # Actualizar último visto del dispositivo
            self.devices[device_id].last_seen = datetime.now()
            
            logger.debug(f"Datos IoT recibidos: {device_id} - {data_type.value}: {value}")
            return data_id
            
        except Exception as e:
            logger.error(f"Error al enviar datos IoT: {e}")
            raise
    
    async def get_device_data(
        self,
        device_id: str,
        data_type: Optional[DataType] = None,
        limit: int = 100,
        hours: int = 24
    ) -> List[Dict[str, Any]]:
        """Obtener datos de dispositivo IoT."""
        try:
            if device_id not in self.data_streams:
                return []
            
            data_list = self.data_streams[device_id]
            
            # Filtrar por tipo de datos
            if data_type:
                data_list = [data for data in data_list if data.data_type == data_type]
            
            # Filtrar por tiempo
            cutoff_time = datetime.now() - timedelta(hours=hours)
            data_list = [data for data in data_list if data.timestamp > cutoff_time]
            
            # Ordenar por timestamp descendente
            data_list.sort(key=lambda x: x.timestamp, reverse=True)
            
            # Limitar resultados
            data_list = data_list[:limit]
            
            return [
                {
                    "data_id": data.data_id,
                    "device_id": data.device_id,
                    "data_type": data.data_type.value,
                    "value": data.value,
                    "unit": data.unit,
                    "quality": data.quality,
                    "timestamp": data.timestamp.isoformat(),
                    "metadata": data.metadata
                }
                for data in data_list
            ]
            
        except Exception as e:
            logger.error(f"Error al obtener datos de dispositivo: {e}")
            raise
    
    async def get_device_status(self, device_id: str) -> Dict[str, Any]:
        """Obtener estado de dispositivo IoT."""
        try:
            if device_id not in self.devices:
                raise ValueError(f"Dispositivo {device_id} no encontrado")
            
            device = self.devices[device_id]
            
            # Obtener datos recientes
            recent_data = await self.get_device_data(device_id, limit=10, hours=1)
            
            # Obtener alertas activas
            active_alerts = [
                alert for alert in self.alerts.values()
                if alert.device_id == device_id and not alert.resolved
            ]
            
            return {
                "device_id": device_id,
                "name": device.name,
                "device_type": device.device_type.value,
                "protocol": device.protocol.value,
                "ip_address": device.ip_address,
                "port": device.port,
                "status": device.status,
                "last_seen": device.last_seen.isoformat(),
                "battery_level": device.battery_level,
                "location": device.location,
                "metadata": device.metadata,
                "created_at": device.created_at.isoformat(),
                "recent_data": recent_data,
                "active_alerts": len(active_alerts),
                "data_points_count": len(self.data_streams.get(device_id, []))
            }
            
        except Exception as e:
            logger.error(f"Error al obtener estado de dispositivo: {e}")
            raise
    
    async def get_iot_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de IoT."""
        return {
            **self.stats,
            "uptime_seconds": (datetime.now() - self.stats["start_time"]).total_seconds(),
            "devices_count": len(self.devices),
            "online_devices_count": self.stats["online_devices"],
            "offline_devices_count": len(self.devices) - self.stats["online_devices"],
            "total_alerts": len(self.alerts),
            "active_alerts": self.stats["active_alerts"],
            "data_streams_count": len(self.data_streams),
            "protocols_supported": [protocol.value for protocol in ProtocolType],
            "device_types_supported": [device_type.value for device_type in DeviceType],
            "timestamp": datetime.now().isoformat()
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Verificar salud del motor IoT."""
        try:
            # Verificar dispositivos
            device_status = {}
            for device_id, device in self.devices.items():
                device_status[device_id] = {
                    "name": device.name,
                    "status": device.status,
                    "last_seen": device.last_seen.isoformat(),
                    "protocol": device.protocol.value
                }
            
            return {
                "status": "healthy",
                "devices_count": len(self.devices),
                "online_devices": self.stats["online_devices"],
                "data_streams": len(self.data_streams),
                "active_alerts": self.stats["active_alerts"],
                "device_status": device_status,
                "stats": self.stats,
                "uptime_seconds": (datetime.now() - self.stats["start_time"]).total_seconds(),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error en health check de IoT: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }




