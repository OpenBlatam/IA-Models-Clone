"""
IoT API - Advanced Implementation
===============================

Advanced IoT API with device management, sensor data processing, and real-time monitoring.
"""

from __future__ import annotations
import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel
from datetime import datetime

from ..services import iot_service, DeviceType, SensorType, DataType

# Create router
router = APIRouter()

logger = logging.getLogger(__name__)


# Request/Response models
class DeviceRegisterRequest(BaseModel):
    """Device register request model"""
    device_id: str
    device_type: str
    device_name: str
    location: Dict[str, float]
    capabilities: List[str]
    protocol: str = "mqtt"
    firmware_version: str = "1.0.0"


class SensorAddRequest(BaseModel):
    """Sensor add request model"""
    device_id: str
    sensor_id: str
    sensor_type: str
    sensor_name: str
    data_type: str
    sampling_rate: float = 1.0
    unit: str = ""
    calibration_data: Optional[Dict[str, Any]] = None


class SensorDataRequest(BaseModel):
    """Sensor data request model"""
    sensor_id: str
    value: Any
    timestamp: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class DeviceGroupCreateRequest(BaseModel):
    """Device group create request model"""
    group_name: str
    device_ids: List[str]
    group_type: str = "logical"
    description: str = ""


class IoTNetworkCreateRequest(BaseModel):
    """IoT network create request model"""
    network_name: str
    network_type: str
    devices: List[str]
    network_config: Dict[str, Any]


class AlertRuleCreateRequest(BaseModel):
    """Alert rule create request model"""
    rule_name: str
    sensor_id: str
    condition: str
    threshold: Any
    alert_type: str = "notification"
    actions: Optional[List[Dict[str, Any]]] = None


class DeviceCommandRequest(BaseModel):
    """Device command request model"""
    device_id: str
    command: str
    parameters: Optional[Dict[str, Any]] = None


class DeviceResponse(BaseModel):
    """Device response model"""
    device_id: str
    type: str
    name: str
    location: Dict[str, float]
    capabilities: List[str]
    protocol: str
    firmware_version: str
    status: str
    message: str


class SensorResponse(BaseModel):
    """Sensor response model"""
    sensor_id: str
    device_id: str
    type: str
    name: str
    data_type: str
    sampling_rate: float
    unit: str
    status: str
    message: str


class SensorDataResponse(BaseModel):
    """Sensor data response model"""
    data_id: str
    sensor_id: str
    device_id: str
    value: Any
    timestamp: str
    message: str


class DeviceGroupResponse(BaseModel):
    """Device group response model"""
    group_id: str
    name: str
    device_ids: List[str]
    group_type: str
    description: str
    status: str
    message: str


class IoTNetworkResponse(BaseModel):
    """IoT network response model"""
    network_id: str
    name: str
    type: str
    devices: List[str]
    status: str
    message: str


class AlertRuleResponse(BaseModel):
    """Alert rule response model"""
    rule_id: str
    name: str
    sensor_id: str
    condition: str
    threshold: Any
    alert_type: str
    status: str
    message: str


class DeviceCommandResponse(BaseModel):
    """Device command response model"""
    command_id: str
    device_id: str
    command: str
    status: str
    message: str


class DeviceStatusResponse(BaseModel):
    """Device status response model"""
    id: str
    type: str
    name: str
    status: str
    location: Dict[str, float]
    capabilities: List[str]
    protocol: str
    firmware_version: str
    battery_level: float
    signal_strength: float
    data_sent: int
    data_received: int
    uptime: int
    sensors_count: int
    last_seen: str
    registered_at: str


class SensorStatusResponse(BaseModel):
    """Sensor status response model"""
    id: str
    device_id: str
    type: str
    name: str
    data_type: str
    sampling_rate: float
    unit: str
    status: str
    last_reading: Optional[str]
    readings_count: int
    min_value: Optional[float]
    max_value: Optional[float]
    avg_value: Optional[float]
    created_at: str


class IoTStatsResponse(BaseModel):
    """IoT statistics response model"""
    total_devices: int
    active_devices: int
    total_sensors: int
    active_sensors: int
    total_data_points: int
    total_alerts: int
    devices_by_type: Dict[str, int]
    sensors_by_type: Dict[str, int]
    data_by_type: Dict[str, int]
    total_groups: int
    total_networks: int
    total_alert_rules: int


# Device management endpoints
@router.post("/devices", response_model=DeviceResponse)
async def register_device(request: DeviceRegisterRequest):
    """Register a new IoT device"""
    try:
        # Validate device type
        try:
            device_type = DeviceType(request.device_type)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid device type: {request.device_type}"
            )
        
        device_id = await iot_service.register_device(
            device_id=request.device_id,
            device_type=device_type,
            device_name=request.device_name,
            location=request.location,
            capabilities=request.capabilities,
            protocol=request.protocol,
            firmware_version=request.firmware_version
        )
        
        return DeviceResponse(
            device_id=device_id,
            type=request.device_type,
            name=request.device_name,
            location=request.location,
            capabilities=request.capabilities,
            protocol=request.protocol,
            firmware_version=request.firmware_version,
            status="active",
            message="Device registered successfully"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to register device: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to register device: {str(e)}"
        )


@router.post("/sensors", response_model=SensorResponse)
async def add_sensor(request: SensorAddRequest):
    """Add a sensor to a device"""
    try:
        # Validate sensor type
        try:
            sensor_type = SensorType(request.sensor_type)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid sensor type: {request.sensor_type}"
            )
        
        # Validate data type
        try:
            data_type = DataType(request.data_type)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid data type: {request.data_type}"
            )
        
        sensor_id = await iot_service.add_sensor(
            device_id=request.device_id,
            sensor_id=request.sensor_id,
            sensor_type=sensor_type,
            sensor_name=request.sensor_name,
            data_type=data_type,
            sampling_rate=request.sampling_rate,
            unit=request.unit,
            calibration_data=request.calibration_data
        )
        
        return SensorResponse(
            sensor_id=sensor_id,
            device_id=request.device_id,
            type=request.sensor_type,
            name=request.sensor_name,
            data_type=request.data_type,
            sampling_rate=request.sampling_rate,
            unit=request.unit,
            status="active",
            message="Sensor added successfully"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to add sensor: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to add sensor: {str(e)}"
        )


@router.post("/sensor-data", response_model=SensorDataResponse)
async def send_sensor_data(request: SensorDataRequest):
    """Send sensor data"""
    try:
        timestamp = None
        if request.timestamp:
            timestamp = datetime.fromisoformat(request.timestamp)
        
        data_id = await iot_service.send_sensor_data(
            sensor_id=request.sensor_id,
            value=request.value,
            timestamp=timestamp,
            metadata=request.metadata
        )
        
        # Get sensor and device info
        sensor = iot_service.sensors.get(request.sensor_id)
        device_id = sensor["device_id"] if sensor else "unknown"
        
        return SensorDataResponse(
            data_id=data_id,
            sensor_id=request.sensor_id,
            device_id=device_id,
            value=request.value,
            timestamp=timestamp.isoformat() if timestamp else datetime.utcnow().isoformat(),
            message="Sensor data sent successfully"
        )
    
    except Exception as e:
        logger.error(f"Failed to send sensor data: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to send sensor data: {str(e)}"
        )


@router.post("/device-groups", response_model=DeviceGroupResponse)
async def create_device_group(request: DeviceGroupCreateRequest):
    """Create a device group"""
    try:
        group_id = await iot_service.create_device_group(
            group_name=request.group_name,
            device_ids=request.device_ids,
            group_type=request.group_type,
            description=request.description
        )
        
        return DeviceGroupResponse(
            group_id=group_id,
            name=request.group_name,
            device_ids=request.device_ids,
            group_type=request.group_type,
            description=request.description,
            status="active",
            message="Device group created successfully"
        )
    
    except Exception as e:
        logger.error(f"Failed to create device group: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create device group: {str(e)}"
        )


@router.post("/networks", response_model=IoTNetworkResponse)
async def create_iot_network(request: IoTNetworkCreateRequest):
    """Create an IoT network"""
    try:
        network_id = await iot_service.create_iot_network(
            network_name=request.network_name,
            network_type=request.network_type,
            devices=request.devices,
            network_config=request.network_config
        )
        
        return IoTNetworkResponse(
            network_id=network_id,
            name=request.network_name,
            type=request.network_type,
            devices=request.devices,
            status="active",
            message="IoT network created successfully"
        )
    
    except Exception as e:
        logger.error(f"Failed to create IoT network: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create IoT network: {str(e)}"
        )


@router.post("/alert-rules", response_model=AlertRuleResponse)
async def create_alert_rule(request: AlertRuleCreateRequest):
    """Create an alert rule"""
    try:
        rule_id = await iot_service.create_alert_rule(
            rule_name=request.rule_name,
            sensor_id=request.sensor_id,
            condition=request.condition,
            threshold=request.threshold,
            alert_type=request.alert_type,
            actions=request.actions
        )
        
        return AlertRuleResponse(
            rule_id=rule_id,
            name=request.rule_name,
            sensor_id=request.sensor_id,
            condition=request.condition,
            threshold=request.threshold,
            alert_type=request.alert_type,
            status="active",
            message="Alert rule created successfully"
        )
    
    except Exception as e:
        logger.error(f"Failed to create alert rule: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create alert rule: {str(e)}"
        )


@router.post("/commands", response_model=DeviceCommandResponse)
async def send_device_command(request: DeviceCommandRequest):
    """Send command to device"""
    try:
        command_id = await iot_service.send_device_command(
            device_id=request.device_id,
            command=request.command,
            parameters=request.parameters
        )
        
        return DeviceCommandResponse(
            command_id=command_id,
            device_id=request.device_id,
            command=request.command,
            status="sent",
            message="Device command sent successfully"
        )
    
    except Exception as e:
        logger.error(f"Failed to send device command: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to send device command: {str(e)}"
        )


# Query endpoints
@router.get("/devices/{device_id}/status", response_model=DeviceStatusResponse)
async def get_device_status(device_id: str):
    """Get device status and metrics"""
    try:
        status = await iot_service.get_device_status(device_id)
        if not status:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Device not found"
            )
        
        return DeviceStatusResponse(**status)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get device status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get device status: {str(e)}"
        )


@router.get("/sensors/{sensor_id}/status", response_model=SensorStatusResponse)
async def get_sensor_status(sensor_id: str):
    """Get sensor status and metrics"""
    try:
        status = await iot_service.get_sensor_status(sensor_id)
        if not status:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Sensor not found"
            )
        
        return SensorStatusResponse(**status)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get sensor status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get sensor status: {str(e)}"
        )


@router.get("/sensors/{sensor_id}/data")
async def get_sensor_data(
    sensor_id: str,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    limit: int = 100
):
    """Get sensor data within time range"""
    try:
        start_dt = None
        if start_time:
            start_dt = datetime.fromisoformat(start_time)
        
        end_dt = None
        if end_time:
            end_dt = datetime.fromisoformat(end_time)
        
        data = await iot_service.get_sensor_data(
            sensor_id=sensor_id,
            start_time=start_dt,
            end_time=end_dt,
            limit=limit
        )
        
        return {
            "sensor_id": sensor_id,
            "data_points": data,
            "count": len(data),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Failed to get sensor data: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get sensor data: {str(e)}"
        )


# Statistics endpoint
@router.get("/stats", response_model=IoTStatsResponse)
async def get_iot_stats():
    """Get IoT service statistics"""
    try:
        stats = await iot_service.get_iot_stats()
        return IoTStatsResponse(**stats)
    
    except Exception as e:
        logger.error(f"Failed to get IoT stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get IoT stats: {str(e)}"
        )


# Health check endpoint
@router.get("/health")
async def iot_health():
    """IoT service health check"""
    try:
        stats = await iot_service.get_iot_stats()
        
        return {
            "service": "iot_service",
            "status": "healthy",
            "total_devices": stats["total_devices"],
            "active_devices": stats["active_devices"],
            "total_sensors": stats["total_sensors"],
            "active_sensors": stats["active_sensors"],
            "total_data_points": stats["total_data_points"],
            "total_alerts": stats["total_alerts"],
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        logger.error(f"IoT service health check failed: {e}")
        return {
            "service": "iot_service",
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

