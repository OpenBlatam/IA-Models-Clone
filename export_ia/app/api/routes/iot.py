"""
IoT API Routes - Rutas API para sistema IoT
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime
from decimal import Decimal
import logging

from ..iot.iot_engine import IoTEngine, DeviceType, ProtocolType, DataType

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/iot", tags=["IoT"])

# Instancia global del motor IoT
iot_engine = IoTEngine()


# Modelos Pydantic
class RegisterDeviceRequest(BaseModel):
    name: str
    device_type: str
    protocol: str
    ip_address: str
    port: int
    metadata: Optional[Dict[str, Any]] = None


class SendDataRequest(BaseModel):
    device_id: str
    data_type: str
    value: Any
    unit: str = ""
    quality: float = 1.0
    metadata: Optional[Dict[str, Any]] = None


class DeviceDataQuery(BaseModel):
    device_id: str
    data_type: Optional[str] = None
    limit: int = 100
    hours: int = 24


# Rutas de Dispositivos IoT
@router.post("/devices")
async def register_iot_device(request: RegisterDeviceRequest):
    """Registrar nuevo dispositivo IoT."""
    try:
        device_type = DeviceType(request.device_type)
        protocol = ProtocolType(request.protocol)
        
        device_id = await iot_engine.register_device(
            name=request.name,
            device_type=device_type,
            protocol=protocol,
            ip_address=request.ip_address,
            port=request.port,
            metadata=request.metadata
        )
        
        device = iot_engine.devices[device_id]
        
        return {
            "device_id": device_id,
            "name": device.name,
            "device_type": device.device_type.value,
            "protocol": device.protocol.value,
            "ip_address": device.ip_address,
            "port": device.port,
            "status": device.status,
            "created_at": device.created_at.isoformat(),
            "success": True,
            "message": "Dispositivo IoT registrado exitosamente",
            "timestamp": datetime.now().isoformat()
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error al registrar dispositivo IoT: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/devices")
async def get_iot_devices():
    """Obtener todos los dispositivos IoT."""
    try:
        devices = []
        for device_id, device in iot_engine.devices.items():
            devices.append({
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
                "created_at": device.created_at.isoformat()
            })
        
        return {
            "devices": devices,
            "count": len(devices),
            "online_count": sum(1 for d in devices if d["status"] == "online"),
            "offline_count": sum(1 for d in devices if d["status"] == "offline"),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error al obtener dispositivos IoT: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/devices/{device_id}")
async def get_iot_device(device_id: str):
    """Obtener dispositivo IoT específico."""
    try:
        device_status = await iot_engine.get_device_status(device_id)
        
        return {
            "device": device_status,
            "timestamp": datetime.now().isoformat()
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error al obtener dispositivo IoT: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/devices/{device_id}")
async def delete_iot_device(device_id: str):
    """Eliminar dispositivo IoT."""
    try:
        if device_id not in iot_engine.devices:
            raise HTTPException(status_code=404, detail="Dispositivo no encontrado")
        
        device = iot_engine.devices[device_id]
        del iot_engine.devices[device_id]
        
        # Limpiar datos del dispositivo
        if device_id in iot_engine.data_streams:
            del iot_engine.data_streams[device_id]
        
        # Limpiar alertas del dispositivo
        device_alerts = [
            alert_id for alert_id, alert in iot_engine.alerts.items()
            if alert.device_id == device_id
        ]
        for alert_id in device_alerts:
            del iot_engine.alerts[alert_id]
        
        iot_engine.stats["total_devices"] -= 1
        
        return {
            "device_id": device_id,
            "name": device.name,
            "success": True,
            "message": "Dispositivo IoT eliminado exitosamente",
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error al eliminar dispositivo IoT: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Rutas de Datos IoT
@router.post("/data")
async def send_iot_data(request: SendDataRequest):
    """Enviar datos de dispositivo IoT."""
    try:
        data_type = DataType(request.data_type)
        
        data_id = await iot_engine.send_data(
            device_id=request.device_id,
            data_type=data_type,
            value=request.value,
            unit=request.unit,
            quality=request.quality,
            metadata=request.metadata
        )
        
        return {
            "data_id": data_id,
            "device_id": request.device_id,
            "data_type": request.data_type,
            "value": request.value,
            "unit": request.unit,
            "quality": request.quality,
            "success": True,
            "message": "Datos IoT enviados exitosamente",
            "timestamp": datetime.now().isoformat()
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error al enviar datos IoT: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/devices/{device_id}/data")
async def get_device_data(
    device_id: str,
    data_type: Optional[str] = Query(None, description="Tipo de datos"),
    limit: int = Query(100, description="Límite de datos"),
    hours: int = Query(24, description="Horas de datos")
):
    """Obtener datos de dispositivo IoT."""
    try:
        data_type_enum = DataType(data_type) if data_type else None
        
        data = await iot_engine.get_device_data(
            device_id=device_id,
            data_type=data_type_enum,
            limit=limit,
            hours=hours
        )
        
        return {
            "device_id": device_id,
            "data": data,
            "count": len(data),
            "limit": limit,
            "hours": hours,
            "data_type_filter": data_type,
            "timestamp": datetime.now().isoformat()
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error al obtener datos de dispositivo: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/data")
async def get_all_iot_data(
    limit: int = Query(100, description="Límite de datos"),
    hours: int = Query(24, description="Horas de datos"),
    data_type: Optional[str] = Query(None, description="Tipo de datos")
):
    """Obtener todos los datos IoT."""
    try:
        all_data = []
        
        for device_id in iot_engine.data_streams:
            data_type_enum = DataType(data_type) if data_type else None
            
            device_data = await iot_engine.get_device_data(
                device_id=device_id,
                data_type=data_type_enum,
                limit=limit,
                hours=hours
            )
            
            all_data.extend(device_data)
        
        # Ordenar por timestamp descendente
        all_data.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return {
            "data": all_data[:limit],
            "count": len(all_data[:limit]),
            "total_devices": len(iot_engine.data_streams),
            "limit": limit,
            "hours": hours,
            "data_type_filter": data_type,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error al obtener todos los datos IoT: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Rutas de Alertas IoT
@router.get("/alerts")
async def get_iot_alerts(
    device_id: Optional[str] = Query(None, description="ID del dispositivo"),
    severity: Optional[str] = Query(None, description="Severidad de alerta"),
    resolved: Optional[bool] = Query(None, description="Estado de resolución")
):
    """Obtener alertas IoT."""
    try:
        alerts = []
        
        for alert_id, alert in iot_engine.alerts.items():
            # Filtrar por dispositivo
            if device_id and alert.device_id != device_id:
                continue
            
            # Filtrar por severidad
            if severity and alert.severity != severity:
                continue
            
            # Filtrar por estado de resolución
            if resolved is not None and alert.resolved != resolved:
                continue
            
            alerts.append({
                "alert_id": alert_id,
                "device_id": alert.device_id,
                "alert_type": alert.alert_type,
                "severity": alert.severity,
                "message": alert.message,
                "threshold": alert.threshold,
                "current_value": alert.current_value,
                "timestamp": alert.timestamp.isoformat(),
                "acknowledged": alert.acknowledged,
                "resolved": alert.resolved
            })
        
        # Ordenar por timestamp descendente
        alerts.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return {
            "alerts": alerts,
            "count": len(alerts),
            "filters": {
                "device_id": device_id,
                "severity": severity,
                "resolved": resolved
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error al obtener alertas IoT: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: str):
    """Reconocer alerta IoT."""
    try:
        if alert_id not in iot_engine.alerts:
            raise HTTPException(status_code=404, detail="Alerta no encontrada")
        
        alert = iot_engine.alerts[alert_id]
        alert.acknowledged = True
        
        return {
            "alert_id": alert_id,
            "acknowledged": True,
            "message": "Alerta reconocida exitosamente",
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error al reconocer alerta: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/alerts/{alert_id}/resolve")
async def resolve_alert(alert_id: str):
    """Resolver alerta IoT."""
    try:
        if alert_id not in iot_engine.alerts:
            raise HTTPException(status_code=404, detail="Alerta no encontrada")
        
        alert = iot_engine.alerts[alert_id]
        alert.resolved = True
        iot_engine.stats["active_alerts"] -= 1
        
        return {
            "alert_id": alert_id,
            "resolved": True,
            "message": "Alerta resuelta exitosamente",
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error al resolver alerta: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Rutas de Estadísticas
@router.get("/stats")
async def get_iot_stats():
    """Obtener estadísticas de IoT."""
    try:
        stats = await iot_engine.get_iot_stats()
        
        return {
            "stats": stats,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error al obtener estadísticas IoT: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def iot_health_check():
    """Verificar salud del sistema IoT."""
    try:
        health = await iot_engine.health_check()
        
        return {
            "health": health,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error en health check de IoT: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Rutas de utilidad
@router.get("/device-types")
async def get_device_types():
    """Obtener tipos de dispositivos disponibles."""
    return {
        "device_types": [
            {
                "value": device_type.value,
                "name": device_type.name,
                "description": f"Tipo de dispositivo {device_type.value}"
            }
            for device_type in DeviceType
        ],
        "timestamp": datetime.now().isoformat()
    }


@router.get("/protocols")
async def get_protocols():
    """Obtener protocolos disponibles."""
    return {
        "protocols": [
            {
                "value": protocol.value,
                "name": protocol.name,
                "description": f"Protocolo {protocol.value}",
                "port": iot_engine.protocols.get(protocol, {}).get("port", "N/A")
            }
            for protocol in ProtocolType
        ],
        "timestamp": datetime.now().isoformat()
    }


@router.get("/data-types")
async def get_data_types():
    """Obtener tipos de datos disponibles."""
    return {
        "data_types": [
            {
                "value": data_type.value,
                "name": data_type.name,
                "description": f"Tipo de dato {data_type.value}"
            }
            for data_type in DataType
        ],
        "timestamp": datetime.now().isoformat()
    }


# Rutas de ejemplo
@router.post("/examples/simulate-sensor")
async def simulate_sensor_data():
    """Ejemplo: Simular datos de sensor IoT."""
    try:
        # Registrar dispositivo sensor
        device_id = await iot_engine.register_device(
            name="Sensor de Temperatura Simulado",
            device_type=DeviceType.SENSOR,
            protocol=ProtocolType.HTTP,
            ip_address="192.168.1.100",
            port=8080,
            metadata={"location": "Oficina Principal", "room": "A101"}
        )
        
        # Simular datos de temperatura
        import random
        for i in range(10):
            temperature = round(random.uniform(18.0, 25.0), 1)
            await iot_engine.send_data(
                device_id=device_id,
                data_type=DataType.TEMPERATURE,
                value=temperature,
                unit="°C",
                quality=0.95,
                metadata={"sensor_id": f"sensor_{i}", "calibration": "recent"}
            )
        
        # Simular datos de humedad
        for i in range(5):
            humidity = round(random.uniform(40.0, 60.0), 1)
            await iot_engine.send_data(
                device_id=device_id,
                data_type=DataType.HUMIDITY,
                value=humidity,
                unit="%",
                quality=0.98,
                metadata={"sensor_id": f"humidity_{i}"}
            )
        
        device = iot_engine.devices[device_id]
        
        return {
            "device": {
                "device_id": device_id,
                "name": device.name,
                "device_type": device.device_type.value,
                "status": device.status,
                "created_at": device.created_at.isoformat()
            },
            "simulated_data": {
                "temperature_points": 10,
                "humidity_points": 5,
                "total_points": 15
            },
            "success": True,
            "message": "Datos de sensor simulados exitosamente",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error en simulación de sensor: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/examples/dashboard-data")
async def get_dashboard_data():
    """Ejemplo: Obtener datos para dashboard IoT."""
    try:
        # Obtener estadísticas generales
        stats = await iot_engine.get_iot_stats()
        
        # Obtener dispositivos recientes
        recent_devices = []
        for device_id, device in iot_engine.devices.items():
            recent_devices.append({
                "device_id": device_id,
                "name": device.name,
                "status": device.status,
                "last_seen": device.last_seen.isoformat(),
                "device_type": device.device_type.value
            })
        
        # Obtener alertas recientes
        recent_alerts = []
        for alert_id, alert in iot_engine.alerts.items():
            if not alert.resolved:
                recent_alerts.append({
                    "alert_id": alert_id,
                    "device_id": alert.device_id,
                    "severity": alert.severity,
                    "message": alert.message,
                    "timestamp": alert.timestamp.isoformat()
                })
        
        # Ordenar por timestamp
        recent_alerts.sort(key=lambda x: x["timestamp"], reverse=True)
        recent_alerts = recent_alerts[:5]  # Últimas 5 alertas
        
        return {
            "dashboard": {
                "stats": stats,
                "recent_devices": recent_devices[:10],  # Últimos 10 dispositivos
                "recent_alerts": recent_alerts,
                "device_types_distribution": {
                    device_type.value: sum(
                        1 for device in iot_engine.devices.values()
                        if device.device_type == device_type
                    )
                    for device_type in DeviceType
                },
                "protocols_distribution": {
                    protocol.value: sum(
                        1 for device in iot_engine.devices.values()
                        if device.protocol == protocol
                    )
                    for protocol in ProtocolType
                }
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error al obtener datos de dashboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))




