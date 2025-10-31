"""
Advanced IoT API endpoints
"""

from typing import List, Dict, Any, Optional, Union
from fastapi import APIRouter, Depends, HTTPException, status, Query
from pydantic import BaseModel, Field
from datetime import datetime

from ....services.advanced_iot_service import AdvancedIoTService, IoTDeviceType, IoTProtocol, IoTDataType, IoTAlertLevel, IoTCommandType
from ....api.dependencies import CurrentUserDep, DatabaseSessionDep
from ....core.exceptions import DatabaseError, ValidationError

router = APIRouter()


class RegisterIoTDeviceRequest(BaseModel):
    """Request model for registering an IoT device."""
    name: str = Field(..., description="Device name")
    device_type: str = Field(..., description="Device type")
    protocol: str = Field(..., description="Communication protocol")
    location: Optional[str] = Field(default=None, description="Device location")
    zone: Optional[str] = Field(default=None, description="Device zone")
    mac_address: Optional[str] = Field(default=None, description="MAC address")
    ip_address: Optional[str] = Field(default=None, description="IP address")
    configuration: Optional[Dict[str, Any]] = Field(default=None, description="Device configuration")


class AddIoTSensorRequest(BaseModel):
    """Request model for adding an IoT sensor."""
    device_id: str = Field(..., description="Device ID")
    name: str = Field(..., description="Sensor name")
    data_type: str = Field(..., description="Data type")
    unit: str = Field(..., description="Measurement unit")
    min_value: Optional[float] = Field(default=None, description="Minimum value")
    max_value: Optional[float] = Field(default=None, description="Maximum value")
    accuracy: Optional[float] = Field(default=None, description="Sensor accuracy")
    resolution: Optional[float] = Field(default=None, description="Sensor resolution")
    sampling_rate: Optional[int] = Field(default=None, description="Sampling rate")


class AddIoTActuatorRequest(BaseModel):
    """Request model for adding an IoT actuator."""
    device_id: str = Field(..., description="Device ID")
    name: str = Field(..., description="Actuator name")
    actuator_type: str = Field(..., description="Actuator type")
    control_type: str = Field(..., description="Control type")
    min_value: Optional[float] = Field(default=None, description="Minimum value")
    max_value: Optional[float] = Field(default=None, description="Maximum value")
    default_value: Optional[float] = Field(default=None, description="Default value")
    unit: Optional[str] = Field(default=None, description="Control unit")


class SendIoTDataRequest(BaseModel):
    """Request model for sending IoT data."""
    device_id: str = Field(..., description="Device ID")
    sensor_id: str = Field(..., description="Sensor ID")
    data_type: str = Field(..., description="Data type")
    value: Union[str, int, float, bool] = Field(..., description="Data value")
    unit: str = Field(..., description="Data unit")
    quality: float = Field(default=1.0, ge=0.0, le=1.0, description="Data quality")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Data metadata")


class SendIoTCommandRequest(BaseModel):
    """Request model for sending IoT command."""
    device_id: str = Field(..., description="Device ID")
    actuator_id: str = Field(..., description="Actuator ID")
    command_type: str = Field(..., description="Command type")
    command_data: Dict[str, Any] = Field(..., description="Command data")


class CreateIoTAlertRequest(BaseModel):
    """Request model for creating an IoT alert."""
    device_id: str = Field(..., description="Device ID")
    alert_type: str = Field(..., description="Alert type")
    level: str = Field(..., description="Alert level")
    message: str = Field(..., description="Alert message")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Alert metadata")


async def get_iot_service(session: DatabaseSessionDep) -> AdvancedIoTService:
    """Get IoT service instance."""
    return AdvancedIoTService(session)


@router.post("/devices", response_model=Dict[str, Any])
async def register_iot_device(
    request: RegisterIoTDeviceRequest = Depends(),
    iot_service: AdvancedIoTService = Depends(get_iot_service),
    current_user: CurrentUserDep = Depends()
):
    """Register a new IoT device."""
    try:
        # Convert enums
        try:
            device_type_enum = IoTDeviceType(request.device_type.lower())
            protocol_enum = IoTProtocol(request.protocol.lower())
        except ValueError as e:
            raise ValidationError(f"Invalid enum value: {e}")
        
        result = await iot_service.register_iot_device(
            name=request.name,
            device_type=device_type_enum,
            protocol=protocol_enum,
            user_id=str(current_user.id),
            location=request.location,
            zone=request.zone,
            mac_address=request.mac_address,
            ip_address=request.ip_address,
            configuration=request.configuration
        )
        
        return {
            "success": True,
            "data": result,
            "message": "IoT device registered successfully"
        }
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to register IoT device"
        )


@router.post("/sensors", response_model=Dict[str, Any])
async def add_iot_sensor(
    request: AddIoTSensorRequest = Depends(),
    iot_service: AdvancedIoTService = Depends(get_iot_service),
    current_user: CurrentUserDep = Depends()
):
    """Add a sensor to an IoT device."""
    try:
        # Convert data type to enum
        try:
            data_type_enum = IoTDataType(request.data_type.lower())
        except ValueError:
            raise ValidationError(f"Invalid data type: {request.data_type}")
        
        result = await iot_service.add_iot_sensor(
            device_id=request.device_id,
            name=request.name,
            data_type=data_type_enum,
            unit=request.unit,
            min_value=request.min_value,
            max_value=request.max_value,
            accuracy=request.accuracy,
            resolution=request.resolution,
            sampling_rate=request.sampling_rate
        )
        
        return {
            "success": True,
            "data": result,
            "message": "IoT sensor added successfully"
        }
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to add IoT sensor"
        )


@router.post("/actuators", response_model=Dict[str, Any])
async def add_iot_actuator(
    request: AddIoTActuatorRequest = Depends(),
    iot_service: AdvancedIoTService = Depends(get_iot_service),
    current_user: CurrentUserDep = Depends()
):
    """Add an actuator to an IoT device."""
    try:
        result = await iot_service.add_iot_actuator(
            device_id=request.device_id,
            name=request.name,
            actuator_type=request.actuator_type,
            control_type=request.control_type,
            min_value=request.min_value,
            max_value=request.max_value,
            default_value=request.default_value,
            unit=request.unit
        )
        
        return {
            "success": True,
            "data": result,
            "message": "IoT actuator added successfully"
        }
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to add IoT actuator"
        )


@router.post("/data", response_model=Dict[str, Any])
async def send_iot_data(
    request: SendIoTDataRequest = Depends(),
    iot_service: AdvancedIoTService = Depends(get_iot_service),
    current_user: CurrentUserDep = Depends()
):
    """Send IoT sensor data."""
    try:
        result = await iot_service.send_iot_data(
            device_id=request.device_id,
            sensor_id=request.sensor_id,
            data_type=request.data_type,
            value=request.value,
            unit=request.unit,
            quality=request.quality,
            metadata=request.metadata
        )
        
        return {
            "success": True,
            "data": result,
            "message": "IoT data sent successfully"
        }
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to send IoT data"
        )


@router.post("/commands", response_model=Dict[str, Any])
async def send_iot_command(
    request: SendIoTCommandRequest = Depends(),
    iot_service: AdvancedIoTService = Depends(get_iot_service),
    current_user: CurrentUserDep = Depends()
):
    """Send command to IoT actuator."""
    try:
        # Convert command type to enum
        try:
            command_type_enum = IoTCommandType(request.command_type.lower())
        except ValueError:
            raise ValidationError(f"Invalid command type: {request.command_type}")
        
        result = await iot_service.send_iot_command(
            device_id=request.device_id,
            actuator_id=request.actuator_id,
            command_type=command_type_enum,
            command_data=request.command_data,
            user_id=str(current_user.id)
        )
        
        return {
            "success": True,
            "data": result,
            "message": "IoT command sent successfully"
        }
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to send IoT command"
        )


@router.post("/alerts", response_model=Dict[str, Any])
async def create_iot_alert(
    request: CreateIoTAlertRequest = Depends(),
    iot_service: AdvancedIoTService = Depends(get_iot_service),
    current_user: CurrentUserDep = Depends()
):
    """Create an IoT alert."""
    try:
        # Convert alert level to enum
        try:
            alert_level_enum = IoTAlertLevel(request.level.lower())
        except ValueError:
            raise ValidationError(f"Invalid alert level: {request.level}")
        
        result = await iot_service.create_iot_alert(
            device_id=request.device_id,
            alert_type=request.alert_type,
            level=alert_level_enum,
            message=request.message,
            metadata=request.metadata
        )
        
        return {
            "success": True,
            "data": result,
            "message": "IoT alert created successfully"
        }
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create IoT alert"
        )


@router.get("/analytics", response_model=Dict[str, Any])
async def get_iot_analytics(
    device_id: Optional[str] = Query(default=None, description="Device ID"),
    sensor_id: Optional[str] = Query(default=None, description="Sensor ID"),
    data_type: Optional[str] = Query(default=None, description="Data type"),
    time_period: str = Query(default="24_hours", description="Time period"),
    iot_service: AdvancedIoTService = Depends(get_iot_service),
    current_user: CurrentUserDep = Depends()
):
    """Get IoT analytics."""
    try:
        result = await iot_service.get_iot_analytics(
            device_id=device_id,
            sensor_id=sensor_id,
            data_type=data_type,
            time_period=time_period
        )
        
        return {
            "success": True,
            "data": result["data"],
            "message": "IoT analytics retrieved successfully"
        }
        
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get IoT analytics"
        )


@router.get("/stats", response_model=Dict[str, Any])
async def get_iot_stats(
    iot_service: AdvancedIoTService = Depends(get_iot_service),
    current_user: CurrentUserDep = Depends()
):
    """Get IoT system statistics."""
    try:
        result = await iot_service.get_iot_stats()
        
        return {
            "success": True,
            "data": result["data"],
            "message": "IoT statistics retrieved successfully"
        }
        
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get IoT statistics"
        )


@router.get("/device-types", response_model=Dict[str, Any])
async def get_iot_device_types():
    """Get available IoT device types."""
    device_types = {
        "sensor": {
            "name": "Sensor",
            "description": "Device that measures physical quantities",
            "icon": "📊",
            "capabilities": ["measure", "monitor", "detect"]
        },
        "actuator": {
            "name": "Actuator",
            "description": "Device that performs physical actions",
            "icon": "⚙️",
            "capabilities": ["control", "actuate", "operate"]
        },
        "gateway": {
            "name": "Gateway",
            "description": "Device that connects different networks",
            "icon": "🌐",
            "capabilities": ["bridge", "translate", "route"]
        },
        "controller": {
            "name": "Controller",
            "description": "Device that controls other devices",
            "icon": "🎮",
            "capabilities": ["control", "automate", "schedule"]
        },
        "camera": {
            "name": "Camera",
            "description": "Device that captures images and video",
            "icon": "📷",
            "capabilities": ["capture", "record", "stream"]
        },
        "speaker": {
            "name": "Speaker",
            "description": "Device that produces audio output",
            "icon": "🔊",
            "capabilities": ["play", "announce", "alert"]
        },
        "display": {
            "name": "Display",
            "description": "Device that shows visual information",
            "icon": "📺",
            "capabilities": ["display", "show", "present"]
        },
        "switch": {
            "name": "Switch",
            "description": "Device that controls electrical circuits",
            "icon": "🔌",
            "capabilities": ["on_off", "toggle", "control"]
        },
        "light": {
            "name": "Light",
            "description": "Device that provides illumination",
            "icon": "💡",
            "capabilities": ["illuminate", "dim", "color"]
        },
        "thermostat": {
            "name": "Thermostat",
            "description": "Device that controls temperature",
            "icon": "🌡️",
            "capabilities": ["heat", "cool", "maintain"]
        },
        "lock": {
            "name": "Lock",
            "description": "Device that secures doors and containers",
            "icon": "🔒",
            "capabilities": ["lock", "unlock", "secure"]
        },
        "motion_detector": {
            "name": "Motion Detector",
            "description": "Device that detects movement",
            "icon": "👁️",
            "capabilities": ["detect", "monitor", "alert"]
        },
        "smoke_detector": {
            "name": "Smoke Detector",
            "description": "Device that detects smoke and fire",
            "icon": "🔥",
            "capabilities": ["detect", "alert", "safety"]
        },
        "water_leak_detector": {
            "name": "Water Leak Detector",
            "description": "Device that detects water leaks",
            "icon": "💧",
            "capabilities": ["detect", "monitor", "prevent"]
        },
        "doorbell": {
            "name": "Doorbell",
            "description": "Device that notifies of visitors",
            "icon": "🔔",
            "capabilities": ["notify", "announce", "monitor"]
        },
        "security_camera": {
            "name": "Security Camera",
            "description": "Device for security monitoring",
            "icon": "📹",
            "capabilities": ["monitor", "record", "alert"]
        },
        "smart_plug": {
            "name": "Smart Plug",
            "description": "Device that controls power outlets",
            "icon": "🔌",
            "capabilities": ["control", "monitor", "schedule"]
        },
        "smart_bulb": {
            "name": "Smart Bulb",
            "description": "Device that provides smart lighting",
            "icon": "💡",
            "capabilities": ["illuminate", "color", "schedule"]
        },
        "smart_thermostat": {
            "name": "Smart Thermostat",
            "description": "Device that provides smart temperature control",
            "icon": "🌡️",
            "capabilities": ["control", "schedule", "learn"]
        },
        "smart_lock": {
            "name": "Smart Lock",
            "description": "Device that provides smart access control",
            "icon": "🔐",
            "capabilities": ["control", "monitor", "schedule"]
        }
    }
    
    return {
        "success": True,
        "data": {
            "device_types": device_types,
            "total_types": len(device_types)
        },
        "message": "IoT device types retrieved successfully"
    }


@router.get("/protocols", response_model=Dict[str, Any])
async def get_iot_protocols():
    """Get available IoT protocols."""
    protocols = {
        "mqtt": {
            "name": "MQTT",
            "description": "Message Queuing Telemetry Transport",
            "port": 1883,
            "secure_port": 8883,
            "features": ["publish_subscribe", "qos", "retain", "will"],
            "use_cases": ["IoT messaging", "telemetry", "real-time data"]
        },
        "http": {
            "name": "HTTP",
            "description": "Hypertext Transfer Protocol",
            "port": 80,
            "secure_port": 443,
            "features": ["rest", "json", "xml", "authentication"],
            "use_cases": ["web services", "REST APIs", "data exchange"]
        },
        "websocket": {
            "name": "WebSocket",
            "description": "WebSocket Protocol",
            "port": 80,
            "secure_port": 443,
            "features": ["real_time", "bidirectional", "low_latency"],
            "use_cases": ["real-time communication", "live updates", "interactive apps"]
        },
        "coap": {
            "name": "CoAP",
            "description": "Constrained Application Protocol",
            "port": 5683,
            "secure_port": 5684,
            "features": ["lightweight", "udp", "restful", "observe"],
            "use_cases": ["constrained devices", "IoT sensors", "low power"]
        },
        "modbus": {
            "name": "Modbus",
            "description": "Modbus Protocol",
            "port": 502,
            "features": ["industrial", "tcp", "rtu", "ascii"],
            "use_cases": ["industrial automation", "SCADA", "PLC communication"]
        },
        "zigbee": {
            "name": "Zigbee",
            "description": "Zigbee Protocol",
            "frequency": "2.4GHz",
            "features": ["mesh", "low_power", "home_automation"],
            "use_cases": ["smart home", "building automation", "sensor networks"]
        },
        "z_wave": {
            "name": "Z-Wave",
            "description": "Z-Wave Protocol",
            "frequency": "908.42MHz",
            "features": ["mesh", "low_power", "home_automation"],
            "use_cases": ["smart home", "security systems", "energy management"]
        },
        "bluetooth": {
            "name": "Bluetooth",
            "description": "Bluetooth Protocol",
            "frequency": "2.4GHz",
            "features": ["short_range", "low_power", "pairing"],
            "use_cases": ["wearables", "mobile devices", "short range IoT"]
        },
        "wifi": {
            "name": "WiFi",
            "description": "WiFi Protocol",
            "frequency": "2.4GHz/5GHz",
            "features": ["high_speed", "long_range", "internet"],
            "use_cases": ["high bandwidth IoT", "video streaming", "cloud connectivity"]
        },
        "lora": {
            "name": "LoRa",
            "description": "Long Range Protocol",
            "frequency": "433MHz/868MHz/915MHz",
            "features": ["long_range", "low_power", "wide_area"],
            "use_cases": ["smart cities", "agriculture", "environmental monitoring"]
        },
        "nb_iot": {
            "name": "NB-IoT",
            "description": "Narrowband Internet of Things",
            "frequency": "LTE bands",
            "features": ["cellular", "low_power", "wide_coverage"],
            "use_cases": ["smart meters", "asset tracking", "industrial IoT"]
        },
        "lte_m": {
            "name": "LTE-M",
            "description": "LTE for Machines",
            "frequency": "LTE bands",
            "features": ["cellular", "low_power", "mobility"],
            "use_cases": ["fleet management", "wearables", "mobile IoT"]
        },
        "thread": {
            "name": "Thread",
            "description": "Thread Protocol",
            "frequency": "2.4GHz",
            "features": ["mesh", "ipv6", "low_power"],
            "use_cases": ["smart home", "building automation", "connected devices"]
        },
        "matter": {
            "name": "Matter",
            "description": "Matter Protocol",
            "frequency": "2.4GHz",
            "features": ["interoperability", "security", "simplicity"],
            "use_cases": ["smart home", "device compatibility", "unified ecosystem"]
        }
    }
    
    return {
        "success": True,
        "data": {
            "protocols": protocols,
            "total_protocols": len(protocols)
        },
        "message": "IoT protocols retrieved successfully"
    }


@router.get("/data-types", response_model=Dict[str, Any])
async def get_iot_data_types():
    """Get available IoT data types."""
    data_types = {
        "temperature": {
            "name": "Temperature",
            "description": "Temperature measurement data",
            "icon": "🌡️",
            "unit": "°C",
            "range": "-40 to 125"
        },
        "humidity": {
            "name": "Humidity",
            "description": "Humidity measurement data",
            "icon": "💧",
            "unit": "%",
            "range": "0 to 100"
        },
        "pressure": {
            "name": "Pressure",
            "description": "Atmospheric pressure data",
            "icon": "📊",
            "unit": "Pa",
            "range": "300 to 1100"
        },
        "light": {
            "name": "Light",
            "description": "Light intensity measurement",
            "icon": "☀️",
            "unit": "lux",
            "range": "0 to 100000"
        },
        "motion": {
            "name": "Motion",
            "description": "Motion detection data",
            "icon": "👁️",
            "unit": "boolean",
            "range": "true/false"
        },
        "sound": {
            "name": "Sound",
            "description": "Sound level measurement",
            "icon": "🔊",
            "unit": "dB",
            "range": "0 to 120"
        },
        "vibration": {
            "name": "Vibration",
            "description": "Vibration intensity data",
            "icon": "📳",
            "unit": "g",
            "range": "0 to 16"
        },
        "proximity": {
            "name": "Proximity",
            "description": "Proximity detection data",
            "icon": "📏",
            "unit": "cm",
            "range": "0 to 200"
        },
        "acceleration": {
            "name": "Acceleration",
            "description": "Acceleration measurement data",
            "icon": "⚡",
            "unit": "m/s²",
            "range": "-16 to 16"
        },
        "gyroscope": {
            "name": "Gyroscope",
            "description": "Angular velocity data",
            "icon": "🌀",
            "unit": "°/s",
            "range": "-2000 to 2000"
        },
        "magnetometer": {
            "name": "Magnetometer",
            "description": "Magnetic field data",
            "icon": "🧲",
            "unit": "μT",
            "range": "-4900 to 4900"
        },
        "gps": {
            "name": "GPS",
            "description": "Global positioning data",
            "icon": "📍",
            "unit": "degrees",
            "range": "lat/lng"
        },
        "battery": {
            "name": "Battery",
            "description": "Battery level data",
            "icon": "🔋",
            "unit": "%",
            "range": "0 to 100"
        },
        "voltage": {
            "name": "Voltage",
            "description": "Electrical voltage data",
            "icon": "⚡",
            "unit": "V",
            "range": "0 to 500"
        },
        "current": {
            "name": "Current",
            "description": "Electrical current data",
            "icon": "🔌",
            "unit": "A",
            "range": "0 to 100"
        },
        "power": {
            "name": "Power",
            "description": "Electrical power data",
            "icon": "⚡",
            "unit": "W",
            "range": "0 to 10000"
        },
        "energy": {
            "name": "Energy",
            "description": "Energy consumption data",
            "icon": "🔋",
            "unit": "kWh",
            "range": "0 to 1000000"
        },
        "flow": {
            "name": "Flow",
            "description": "Fluid flow rate data",
            "icon": "🌊",
            "unit": "L/min",
            "range": "0 to 1000"
        },
        "level": {
            "name": "Level",
            "description": "Liquid level data",
            "icon": "📊",
            "unit": "%",
            "range": "0 to 100"
        },
        "ph": {
            "name": "pH",
            "description": "pH level data",
            "icon": "🧪",
            "unit": "pH",
            "range": "0 to 14"
        },
        "conductivity": {
            "name": "Conductivity",
            "description": "Electrical conductivity data",
            "icon": "⚡",
            "unit": "μS/cm",
            "range": "0 to 20000"
        },
        "turbidity": {
            "name": "Turbidity",
            "description": "Water turbidity data",
            "icon": "🌊",
            "unit": "NTU",
            "range": "0 to 1000"
        },
        "dissolved_oxygen": {
            "name": "Dissolved Oxygen",
            "description": "Dissolved oxygen data",
            "icon": "💨",
            "unit": "mg/L",
            "range": "0 to 20"
        },
        "custom": {
            "name": "Custom",
            "description": "Custom data type",
            "icon": "🔧",
            "unit": "varies",
            "range": "varies"
        }
    }
    
    return {
        "success": True,
        "data": {
            "data_types": data_types,
            "total_types": len(data_types)
        },
        "message": "IoT data types retrieved successfully"
    }


@router.get("/alert-levels", response_model=Dict[str, Any])
async def get_iot_alert_levels():
    """Get available IoT alert levels."""
    alert_levels = {
        "info": {
            "name": "Info",
            "description": "Informational alert",
            "icon": "ℹ️",
            "color": "blue",
            "priority": 1
        },
        "warning": {
            "name": "Warning",
            "description": "Warning alert",
            "icon": "⚠️",
            "color": "yellow",
            "priority": 2
        },
        "error": {
            "name": "Error",
            "description": "Error alert",
            "icon": "❌",
            "color": "red",
            "priority": 3
        },
        "critical": {
            "name": "Critical",
            "description": "Critical alert",
            "icon": "🚨",
            "color": "red",
            "priority": 4
        },
        "emergency": {
            "name": "Emergency",
            "description": "Emergency alert",
            "icon": "🆘",
            "color": "red",
            "priority": 5
        }
    }
    
    return {
        "success": True,
        "data": {
            "alert_levels": alert_levels,
            "total_levels": len(alert_levels)
        },
        "message": "IoT alert levels retrieved successfully"
    }


@router.get("/command-types", response_model=Dict[str, Any])
async def get_iot_command_types():
    """Get available IoT command types."""
    command_types = {
        "read": {
            "name": "Read",
            "description": "Read data from device",
            "icon": "📖",
            "category": "data"
        },
        "write": {
            "name": "Write",
            "description": "Write data to device",
            "icon": "✏️",
            "category": "data"
        },
        "execute": {
            "name": "Execute",
            "description": "Execute action on device",
            "icon": "▶️",
            "category": "action"
        },
        "configure": {
            "name": "Configure",
            "description": "Configure device settings",
            "icon": "⚙️",
            "category": "config"
        },
        "update": {
            "name": "Update",
            "description": "Update device firmware",
            "icon": "🔄",
            "category": "maintenance"
        },
        "restart": {
            "name": "Restart",
            "description": "Restart device",
            "icon": "🔄",
            "category": "maintenance"
        },
        "reset": {
            "name": "Reset",
            "description": "Reset device to defaults",
            "icon": "🔄",
            "category": "maintenance"
        },
        "calibrate": {
            "name": "Calibrate",
            "description": "Calibrate device sensors",
            "icon": "🎯",
            "category": "maintenance"
        },
        "diagnostic": {
            "name": "Diagnostic",
            "description": "Run device diagnostics",
            "icon": "🔍",
            "category": "maintenance"
        },
        "custom": {
            "name": "Custom",
            "description": "Custom command",
            "icon": "🔧",
            "category": "custom"
        }
    }
    
    return {
        "success": True,
        "data": {
            "command_types": command_types,
            "total_types": len(command_types)
        },
        "message": "IoT command types retrieved successfully"
    }


@router.get("/health", response_model=Dict[str, Any])
async def get_iot_health(
    iot_service: AdvancedIoTService = Depends(get_iot_service),
    current_user: CurrentUserDep = Depends()
):
    """Get IoT system health status."""
    try:
        # Get IoT stats
        stats = await iot_service.get_iot_stats()
        
        # Calculate health metrics
        total_devices = stats["data"].get("total_devices", 0)
        total_sensors = stats["data"].get("total_sensors", 0)
        total_actuators = stats["data"].get("total_actuators", 0)
        total_data_points = stats["data"].get("total_data_points", 0)
        total_alerts = stats["data"].get("total_alerts", 0)
        online_devices = stats["data"].get("online_devices", 0)
        offline_devices = stats["data"].get("offline_devices", 0)
        devices_by_type = stats["data"].get("devices_by_type", {})
        devices_by_protocol = stats["data"].get("devices_by_protocol", {})
        
        # Calculate health score
        health_score = 100
        
        # Check device connectivity
        if total_devices > 0:
            online_ratio = online_devices / total_devices
            if online_ratio < 0.5:
                health_score -= 30
            elif online_ratio < 0.8:
                health_score -= 15
        
        # Check device diversity
        if len(devices_by_type) < 2:
            health_score -= 20
        elif len(devices_by_type) > 10:
            health_score -= 5
        
        # Check protocol diversity
        if len(devices_by_protocol) < 2:
            health_score -= 15
        elif len(devices_by_protocol) > 8:
            health_score -= 5
        
        # Check data activity
        if total_devices > 0:
            data_per_device = total_data_points / total_devices
            if data_per_device < 10:
                health_score -= 25
            elif data_per_device > 10000:
                health_score -= 10
        
        # Check alert ratio
        if total_data_points > 0:
            alert_ratio = total_alerts / total_data_points
            if alert_ratio > 0.1:
                health_score -= 20
            elif alert_ratio > 0.05:
                health_score -= 10
        
        # Check sensor/actuator balance
        if total_sensors > 0 and total_actuators > 0:
            sensor_actuator_ratio = total_sensors / total_actuators
            if sensor_actuator_ratio < 0.5 or sensor_actuator_ratio > 5:
                health_score -= 10
        
        health_status = "excellent" if health_score >= 90 else "good" if health_score >= 70 else "fair" if health_score >= 50 else "poor"
        
        return {
            "success": True,
            "data": {
                "health_status": health_status,
                "health_score": health_score,
                "total_devices": total_devices,
                "total_sensors": total_sensors,
                "total_actuators": total_actuators,
                "total_data_points": total_data_points,
                "total_alerts": total_alerts,
                "online_devices": online_devices,
                "offline_devices": offline_devices,
                "online_ratio": online_ratio if total_devices > 0 else 0,
                "device_diversity": len(devices_by_type),
                "protocol_diversity": len(devices_by_protocol),
                "data_per_device": data_per_device if total_devices > 0 else 0,
                "alert_ratio": alert_ratio if total_data_points > 0 else 0,
                "sensor_actuator_ratio": sensor_actuator_ratio if total_sensors > 0 and total_actuators > 0 else 0,
                "devices_by_type": devices_by_type,
                "devices_by_protocol": devices_by_protocol,
                "timestamp": datetime.utcnow().isoformat()
            },
            "message": "IoT health status retrieved successfully"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get IoT health status"
        )
























