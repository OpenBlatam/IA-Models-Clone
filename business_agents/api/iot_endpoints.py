"""
IoT API Endpoints
=================

REST API endpoints for IoT device management, data collection,
and automation control.
"""

from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks, WebSocket, WebSocketDisconnect
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime

from ..services.iot_service import (
    IoTService, DeviceType, DeviceStatus, DataType, Protocol,
    IoTDevice, IoTData, IoTCommand, IoTAlert
)
from ..middleware.auth_middleware import get_current_user, require_permission, User

# Create API router
router = APIRouter(prefix="/iot", tags=["IoT"])

# Pydantic models
class DeviceRegistrationRequest(BaseModel):
    name: str = Field(..., description="Device name")
    device_type: str = Field(..., description="Device type")
    protocol: str = Field(..., description="Communication protocol")
    location: Dict[str, float] = Field(..., description="Device location")
    capabilities: List[str] = Field(default_factory=list, description="Device capabilities")
    configuration: Dict[str, Any] = Field(default_factory=dict, description="Device configuration")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Device metadata")

class DeviceUpdateRequest(BaseModel):
    name: Optional[str] = Field(None, description="Device name")
    status: Optional[str] = Field(None, description="Device status")
    location: Optional[Dict[str, float]] = Field(None, description="Device location")
    configuration: Optional[Dict[str, Any]] = Field(None, description="Device configuration")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Device metadata")

class CommandRequest(BaseModel):
    device_id: str = Field(..., description="Target device ID")
    command_type: str = Field(..., description="Command type")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Command parameters")

class DataQueryRequest(BaseModel):
    device_id: str = Field(..., description="Device ID")
    data_type: Optional[str] = Field(None, description="Data type filter")
    start_time: Optional[datetime] = Field(None, description="Start time filter")
    end_time: Optional[datetime] = Field(None, description="End time filter")
    limit: int = Field(100, description="Maximum number of records")

class AlertAcknowledgeRequest(BaseModel):
    alert_id: str = Field(..., description="Alert ID to acknowledge")

class AlertResolveRequest(BaseModel):
    alert_id: str = Field(..., description="Alert ID to resolve")

# Global IoT service instance
iot_service = None

def get_iot_service() -> IoTService:
    """Get global IoT service instance."""
    global iot_service
    if iot_service is None:
        iot_service = IoTService({
            "mqtt": {
                "broker": "localhost",
                "port": 1883,
                "username": None,
                "password": None,
                "keepalive": 60
            },
            "websocket": {
                "host": "localhost",
                "port": 8080,
                "path": "/ws"
            }
        })
    return iot_service

# API Endpoints

@router.post("/initialize", response_model=Dict[str, str])
async def initialize_iot_service(
    current_user: User = Depends(require_permission("iot:manage"))
):
    """Initialize the IoT service."""
    
    iot_service = get_iot_service()
    
    try:
        await iot_service.initialize()
        return {"message": "IoT Service initialized successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize IoT service: {str(e)}")

@router.get("/status", response_model=Dict[str, Any])
async def get_iot_status(
    current_user: User = Depends(require_permission("iot:view"))
):
    """Get IoT service status."""
    
    iot_service = get_iot_service()
    
    try:
        status = await iot_service.get_service_status()
        return status
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get IoT status: {str(e)}")

@router.post("/devices/register", response_model=Dict[str, str])
async def register_device(
    request: DeviceRegistrationRequest,
    current_user: User = Depends(require_permission("iot:manage"))
):
    """Register a new IoT device."""
    
    iot_service = get_iot_service()
    
    try:
        # Convert string to enum
        device_type = DeviceType(request.device_type)
        protocol = Protocol(request.protocol)
        
        # Create IoT device
        device = IoTDevice(
            device_id="",  # Will be generated
            name=request.name,
            device_type=device_type,
            protocol=protocol,
            status=DeviceStatus.ONLINE,
            location=request.location,
            capabilities=request.capabilities,
            configuration=request.configuration,
            last_seen=datetime.utcnow(),
            metadata=request.metadata
        )
        
        # Register device
        device_id = await iot_service.register_device(device)
        
        return {
            "message": "Device registered successfully",
            "device_id": device_id
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to register device: {str(e)}")

@router.delete("/devices/{device_id}", response_model=Dict[str, str])
async def unregister_device(
    device_id: str,
    current_user: User = Depends(require_permission("iot:manage"))
):
    """Unregister an IoT device."""
    
    iot_service = get_iot_service()
    
    try:
        success = await iot_service.unregister_device(device_id)
        
        if success:
            return {"message": f"Device {device_id} unregistered successfully"}
        else:
            raise HTTPException(status_code=404, detail="Device not found")
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to unregister device: {str(e)}")

@router.get("/devices", response_model=List[Dict[str, Any]])
async def get_devices(
    device_type: Optional[str] = Query(None, description="Filter by device type"),
    status: Optional[str] = Query(None, description="Filter by device status"),
    current_user: User = Depends(require_permission("iot:view"))
):
    """Get IoT devices."""
    
    iot_service = get_iot_service()
    
    try:
        # Convert string to enum if provided
        device_type_enum = DeviceType(device_type) if device_type else None
        status_enum = DeviceStatus(status) if status else None
        
        # Get devices
        devices = await iot_service.get_devices(device_type_enum)
        
        # Filter by status if provided
        if status_enum:
            devices = [d for d in devices if d.status == status_enum]
        
        result = []
        for device in devices:
            device_dict = {
                "device_id": device.device_id,
                "name": device.name,
                "device_type": device.device_type.value,
                "protocol": device.protocol.value,
                "status": device.status.value,
                "location": device.location,
                "capabilities": device.capabilities,
                "configuration": device.configuration,
                "last_seen": device.last_seen.isoformat(),
                "metadata": device.metadata
            }
            result.append(device_dict)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get devices: {str(e)}")

@router.get("/devices/{device_id}", response_model=Dict[str, Any])
async def get_device(
    device_id: str,
    current_user: User = Depends(require_permission("iot:view"))
):
    """Get specific IoT device."""
    
    iot_service = get_iot_service()
    
    try:
        device = await iot_service.get_device(device_id)
        
        if not device:
            raise HTTPException(status_code=404, detail="Device not found")
        
        return {
            "device_id": device.device_id,
            "name": device.name,
            "device_type": device.device_type.value,
            "protocol": device.protocol.value,
            "status": device.status.value,
            "location": device.location,
            "capabilities": device.capabilities,
            "configuration": device.configuration,
            "last_seen": device.last_seen.isoformat(),
            "metadata": device.metadata
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get device: {str(e)}")

@router.put("/devices/{device_id}", response_model=Dict[str, Any])
async def update_device(
    device_id: str,
    request: DeviceUpdateRequest,
    current_user: User = Depends(require_permission("iot:manage"))
):
    """Update IoT device."""
    
    iot_service = get_iot_service()
    
    try:
        device = await iot_service.get_device(device_id)
        
        if not device:
            raise HTTPException(status_code=404, detail="Device not found")
        
        # Update device properties
        if request.name is not None:
            device.name = request.name
        if request.status is not None:
            device.status = DeviceStatus(request.status)
        if request.location is not None:
            device.location = request.location
        if request.configuration is not None:
            device.configuration = request.configuration
        if request.metadata is not None:
            device.metadata = request.metadata
        
        return {
            "message": "Device updated successfully",
            "device_id": device_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update device: {str(e)}")

@router.get("/devices/{device_id}/data", response_model=List[Dict[str, Any]])
async def get_device_data(
    device_id: str,
    data_type: Optional[str] = Query(None, description="Filter by data type"),
    limit: int = Query(100, description="Maximum number of records"),
    current_user: User = Depends(require_permission("iot:view"))
):
    """Get device data."""
    
    iot_service = get_iot_service()
    
    try:
        # Convert string to enum if provided
        data_type_enum = DataType(data_type) if data_type else None
        
        # Get device data
        data = await iot_service.get_device_data(device_id, data_type_enum, limit)
        
        result = []
        for data_point in data:
            data_dict = {
                "data_id": data_point.data_id,
                "device_id": data_point.device_id,
                "data_type": data_point.data_type.value,
                "value": data_point.value,
                "unit": data_point.unit,
                "timestamp": data_point.timestamp.isoformat(),
                "quality": data_point.quality,
                "metadata": data_point.metadata
            }
            result.append(data_dict)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get device data: {str(e)}")

@router.post("/devices/{device_id}/commands", response_model=Dict[str, Any])
async def send_command(
    device_id: str,
    request: CommandRequest,
    current_user: User = Depends(require_permission("iot:execute"))
):
    """Send command to IoT device."""
    
    iot_service = get_iot_service()
    
    try:
        # Create IoT command
        command = IoTCommand(
            command_id=f"cmd_{device_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            device_id=device_id,
            command_type=request.command_type,
            parameters=request.parameters,
            timestamp=datetime.utcnow(),
            status="pending"
        )
        
        # Send command
        success = await iot_service.send_command(device_id, command)
        
        if success:
            return {
                "message": "Command sent successfully",
                "command_id": command.command_id,
                "device_id": device_id,
                "command_type": request.command_type,
                "status": command.status
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to send command")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to send command: {str(e)}")

@router.get("/commands", response_model=List[Dict[str, Any]])
async def get_commands(
    device_id: Optional[str] = Query(None, description="Filter by device ID"),
    status: Optional[str] = Query(None, description="Filter by command status"),
    limit: int = Query(100, description="Maximum number of records"),
    current_user: User = Depends(require_permission("iot:view"))
):
    """Get IoT commands."""
    
    iot_service = get_iot_service()
    
    try:
        commands = list(iot_service.commands.values())
        
        # Filter by device ID if provided
        if device_id:
            commands = [c for c in commands if c.device_id == device_id]
        
        # Filter by status if provided
        if status:
            commands = [c for c in commands if c.status == status]
        
        # Limit results
        commands = commands[-limit:] if limit else commands
        
        result = []
        for command in commands:
            command_dict = {
                "command_id": command.command_id,
                "device_id": command.device_id,
                "command_type": command.command_type,
                "parameters": command.parameters,
                "timestamp": command.timestamp.isoformat(),
                "status": command.status,
                "response": command.response
            }
            result.append(command_dict)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get commands: {str(e)}")

@router.get("/alerts", response_model=List[Dict[str, Any]])
async def get_alerts(
    device_id: Optional[str] = Query(None, description="Filter by device ID"),
    severity: Optional[str] = Query(None, description="Filter by severity"),
    limit: int = Query(100, description="Maximum number of records"),
    current_user: User = Depends(require_permission("iot:view"))
):
    """Get IoT alerts."""
    
    iot_service = get_iot_service()
    
    try:
        alerts = await iot_service.get_alerts(device_id, severity, limit)
        
        result = []
        for alert in alerts:
            alert_dict = {
                "alert_id": alert.alert_id,
                "device_id": alert.device_id,
                "alert_type": alert.alert_type,
                "severity": alert.severity,
                "message": alert.message,
                "timestamp": alert.timestamp.isoformat(),
                "acknowledged": alert.acknowledged,
                "resolved": alert.resolved,
                "metadata": alert.metadata
            }
            result.append(alert_dict)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get alerts: {str(e)}")

@router.post("/alerts/acknowledge", response_model=Dict[str, str])
async def acknowledge_alert(
    request: AlertAcknowledgeRequest,
    current_user: User = Depends(require_permission("iot:manage"))
):
    """Acknowledge an alert."""
    
    iot_service = get_iot_service()
    
    try:
        success = await iot_service.acknowledge_alert(request.alert_id)
        
        if success:
            return {"message": f"Alert {request.alert_id} acknowledged successfully"}
        else:
            raise HTTPException(status_code=404, detail="Alert not found")
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to acknowledge alert: {str(e)}")

@router.post("/alerts/resolve", response_model=Dict[str, str])
async def resolve_alert(
    request: AlertResolveRequest,
    current_user: User = Depends(require_permission("iot:manage"))
):
    """Resolve an alert."""
    
    iot_service = get_iot_service()
    
    try:
        success = await iot_service.resolve_alert(request.alert_id)
        
        if success:
            return {"message": f"Alert {request.alert_id} resolved successfully"}
        else:
            raise HTTPException(status_code=404, detail="Alert not found")
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to resolve alert: {str(e)}")

@router.get("/analytics", response_model=Dict[str, Any])
async def get_iot_analytics(
    current_user: User = Depends(require_permission("iot:view"))
):
    """Get IoT analytics."""
    
    iot_service = get_iot_service()
    
    try:
        # Get service status
        status = await iot_service.get_service_status()
        
        # Get devices
        devices = await iot_service.get_devices()
        
        # Get alerts
        alerts = await iot_service.get_alerts()
        
        # Calculate analytics
        analytics = {
            "total_devices": len(devices),
            "online_devices": len([d for d in devices if d.status == DeviceStatus.ONLINE]),
            "offline_devices": len([d for d in devices if d.status == DeviceStatus.OFFLINE]),
            "device_types": {
                device_type.value: len([d for d in devices if d.device_type == device_type])
                for device_type in DeviceType
            },
            "total_alerts": len(alerts),
            "active_alerts": len([a for a in alerts if not a.resolved]),
            "acknowledged_alerts": len([a for a in alerts if a.acknowledged]),
            "resolved_alerts": len([a for a in alerts if a.resolved]),
            "alert_severities": {
                severity: len([a for a in alerts if a.severity == severity])
                for severity in ["low", "medium", "high", "critical"]
            },
            "total_data_points": sum(len(stream) for stream in iot_service.data_streams.values()),
            "mqtt_connected": status.get("mqtt_connected", False),
            "websocket_connections": status.get("websocket_connections", 0),
            "generated_at": datetime.utcnow().isoformat()
        }
        
        return analytics
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get IoT analytics: {str(e)}")

@router.websocket("/ws/{device_id}")
async def websocket_endpoint(websocket: WebSocket, device_id: str):
    """WebSocket endpoint for real-time device communication."""
    
    iot_service = get_iot_service()
    
    try:
        await websocket.accept()
        
        # Store WebSocket connection
        iot_service.websocket_connections[device_id] = websocket
        
        # Send welcome message
        await websocket.send_json({
            "type": "connection",
            "message": f"Connected to device {device_id}",
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Keep connection alive and handle messages
        while True:
            try:
                # Receive message from client
                data = await websocket.receive_json()
                
                # Handle different message types
                if data.get("type") == "command":
                    # Process command
                    command = IoTCommand(
                        command_id=f"ws_cmd_{device_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                        device_id=device_id,
                        command_type=data.get("command_type", "unknown"),
                        parameters=data.get("parameters", {}),
                        timestamp=datetime.utcnow(),
                        status="pending"
                    )
                    
                    # Send command
                    success = await iot_service.send_command(device_id, command)
                    
                    # Send response
                    await websocket.send_json({
                        "type": "command_response",
                        "command_id": command.command_id,
                        "success": success,
                        "timestamp": datetime.utcnow().isoformat()
                    })
                    
                elif data.get("type") == "ping":
                    # Respond to ping
                    await websocket.send_json({
                        "type": "pong",
                        "timestamp": datetime.utcnow().isoformat()
                    })
                    
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"WebSocket error: {str(e)}")
                await websocket.send_json({
                    "type": "error",
                    "message": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                })
                
    except Exception as e:
        logger.error(f"WebSocket connection error: {str(e)}")
    finally:
        # Clean up connection
        if device_id in iot_service.websocket_connections:
            del iot_service.websocket_connections[device_id]

@router.get("/health", response_model=Dict[str, Any])
async def iot_health_check():
    """IoT service health check."""
    
    iot_service = get_iot_service()
    
    try:
        # Check if service is initialized
        initialized = hasattr(iot_service, 'devices') and len(iot_service.devices) > 0
        
        # Get service status
        status = await iot_service.get_service_status()
        
        return {
            "status": "healthy" if initialized else "initializing",
            "initialized": initialized,
            "total_devices": status.get("total_devices", 0),
            "online_devices": status.get("online_devices", 0),
            "mqtt_connected": status.get("mqtt_connected", False),
            "websocket_connections": status.get("websocket_connections", 0),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }




























