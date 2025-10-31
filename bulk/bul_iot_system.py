"""
BUL - Business Universal Language (IoT System)
==============================================

Advanced IoT system with device management, data collection, and automation.
"""

import asyncio
import logging
import json
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import sqlite3
from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
from sqlalchemy import create_engine, Column, String, DateTime, Text, Integer, Boolean, Float, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import redis
from prometheus_client import Counter, Histogram, Gauge
import paho.mqtt.client as mqtt
import requests
import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database setup
Base = declarative_base()
engine = create_engine('sqlite:///bul_iot.db', echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Prometheus Metrics
IOT_DEVICES = Gauge('bul_iot_devices_total', 'Total IoT devices', ['status'])
IOT_DATA_POINTS = Counter('bul_iot_data_points_total', 'Total IoT data points', ['device_type', 'data_type'])
IOT_ALERTS = Counter('bul_iot_alerts_total', 'Total IoT alerts', ['severity', 'type'])
IOT_AUTOMATIONS = Counter('bul_iot_automations_total', 'Total IoT automations executed', ['automation_type'])

class DeviceType(str, Enum):
    """IoT device type enumeration."""
    SENSOR = "sensor"
    ACTUATOR = "actuator"
    GATEWAY = "gateway"
    CAMERA = "camera"
    SMART_PLUG = "smart_plug"
    THERMOSTAT = "thermostat"
    LIGHT = "light"
    LOCK = "lock"
    MOTION_DETECTOR = "motion_detector"
    HUMIDITY_SENSOR = "humidity_sensor"
    PRESSURE_SENSOR = "pressure_sensor"
    AIR_QUALITY_SENSOR = "air_quality_sensor"

class DataType(str, Enum):
    """IoT data type enumeration."""
    TEMPERATURE = "temperature"
    HUMIDITY = "humidity"
    PRESSURE = "pressure"
    MOTION = "motion"
    LIGHT = "light"
    SOUND = "sound"
    VOLTAGE = "voltage"
    CURRENT = "current"
    POWER = "power"
    AIR_QUALITY = "air_quality"
    GPS = "gps"
    IMAGE = "image"
    VIDEO = "video"
    STATUS = "status"
    ERROR = "error"

class DeviceStatus(str, Enum):
    """Device status enumeration."""
    ONLINE = "online"
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"
    ERROR = "error"
    UNKNOWN = "unknown"

class AlertSeverity(str, Enum):
    """Alert severity enumeration."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AutomationType(str, Enum):
    """Automation type enumeration."""
    SCHEDULED = "scheduled"
    TRIGGERED = "triggered"
    CONDITIONAL = "conditional"
    MANUAL = "manual"

# Database Models
class IoTDevice(Base):
    __tablename__ = "iot_devices"
    
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    device_type = Column(String, nullable=False)
    mac_address = Column(String, unique=True, nullable=False)
    ip_address = Column(String)
    location = Column(String)
    description = Column(Text)
    manufacturer = Column(String)
    model = Column(String)
    firmware_version = Column(String)
    status = Column(String, default=DeviceStatus.UNKNOWN)
    is_active = Column(Boolean, default=True)
    last_seen = Column(DateTime)
    battery_level = Column(Float)
    signal_strength = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)
    configuration = Column(Text, default="{}")
    capabilities = Column(Text, default="[]")

class IoTDataPoint(Base):
    __tablename__ = "iot_data_points"
    
    id = Column(String, primary_key=True)
    device_id = Column(String, ForeignKey("iot_devices.id"))
    data_type = Column(String, nullable=False)
    value = Column(Float, nullable=False)
    unit = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)
    metadata = Column(Text, default="{}")
    quality_score = Column(Float, default=1.0)
    
    # Relationships
    device = relationship("IoTDevice")

class IoTAlert(Base):
    __tablename__ = "iot_alerts"
    
    id = Column(String, primary_key=True)
    device_id = Column(String, ForeignKey("iot_devices.id"))
    alert_type = Column(String, nullable=False)
    severity = Column(String, default=AlertSeverity.MEDIUM)
    message = Column(Text, nullable=False)
    threshold_value = Column(Float)
    actual_value = Column(Float)
    is_resolved = Column(Boolean, default=False)
    resolved_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    metadata = Column(Text, default="{}")
    
    # Relationships
    device = relationship("IoTDevice")

class IoTAutomation(Base):
    __tablename__ = "iot_automations"
    
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    automation_type = Column(String, nullable=False)
    trigger_device_id = Column(String, ForeignKey("iot_devices.id"))
    trigger_condition = Column(Text, nullable=False)
    action_device_id = Column(String, ForeignKey("iot_devices.id"))
    action_command = Column(Text, nullable=False)
    is_active = Column(Boolean, default=True)
    last_executed = Column(DateTime)
    execution_count = Column(Integer, default=0)
    success_count = Column(Integer, default=0)
    failure_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    trigger_device = relationship("IoTDevice", foreign_keys=[trigger_device_id])
    action_device = relationship("IoTDevice", foreign_keys=[action_device_id])

class IoTGateway(Base):
    __tablename__ = "iot_gateways"
    
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    ip_address = Column(String, nullable=False)
    port = Column(Integer, default=1883)
    protocol = Column(String, default="mqtt")
    is_active = Column(Boolean, default=True)
    connected_devices = Column(Integer, default=0)
    last_heartbeat = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)

# Create tables
Base.metadata.create_all(bind=engine)

# IoT Configuration
IOT_CONFIG = {
    "mqtt_broker": "localhost",
    "mqtt_port": 1883,
    "mqtt_username": "iot_user",
    "mqtt_password": "iot_password",
    "data_retention_days": 30,
    "alert_thresholds": {
        "temperature": {"min": -10, "max": 50},
        "humidity": {"min": 0, "max": 100},
        "pressure": {"min": 800, "max": 1200},
        "battery_level": {"min": 10, "max": 100},
        "signal_strength": {"min": -100, "max": 0}
    },
    "automation_check_interval": 60,  # seconds
    "device_heartbeat_interval": 300,  # seconds
    "data_aggregation_interval": 300,  # seconds
    "max_data_points_per_device": 10000,
    "enable_real_time_processing": True,
    "enable_machine_learning": True,
    "enable_predictive_maintenance": True
}

class AdvancedIoTSystem:
    """Advanced IoT system with comprehensive features."""
    
    def __init__(self):
        self.app = FastAPI(
            title="BUL IoT System",
            description="Advanced IoT system with device management, data collection, and automation",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # Database session
        self.db = SessionLocal()
        
        # IoT components
        self.mqtt_client = None
        self.websocket_connections: List[WebSocket] = []
        self.device_registry: Dict[str, IoTDevice] = {}
        self.automation_engine = None
        
        # Setup components
        self.setup_middleware()
        self.setup_routes()
        self.setup_default_data()
        self.setup_mqtt_client()
        self.start_background_tasks()
        
        logger.info("Advanced IoT System initialized")
    
    def setup_middleware(self):
        """Setup IoT middleware."""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def setup_routes(self):
        """Setup IoT API routes."""
        
        @self.app.get("/", tags=["System"])
        async def root():
            """Root endpoint with IoT system information."""
            return {
                "message": "BUL IoT System",
                "version": "1.0.0",
                "status": "operational",
                "features": [
                    "Device Management",
                    "Real-time Data Collection",
                    "Automation Engine",
                    "Alert System",
                    "Data Analytics",
                    "Predictive Maintenance",
                    "Gateway Management",
                    "WebSocket Streaming"
                ],
                "device_types": [device_type.value for device_type in DeviceType],
                "data_types": [data_type.value for data_type in DataType],
                "timestamp": datetime.now().isoformat()
            }
        
        @self.app.post("/devices/register", tags=["Devices"])
        async def register_device(device_request: dict):
            """Register new IoT device."""
            try:
                # Validate request
                required_fields = ["name", "device_type", "mac_address"]
                if not all(field in device_request for field in required_fields):
                    raise HTTPException(status_code=400, detail="Missing required fields")
                
                name = device_request["name"]
                device_type = device_request["device_type"]
                mac_address = device_request["mac_address"]
                
                # Check if device already exists
                existing_device = self.db.query(IoTDevice).filter(
                    IoTDevice.mac_address == mac_address
                ).first()
                
                if existing_device:
                    raise HTTPException(status_code=400, detail="Device already registered")
                
                # Create device
                device = IoTDevice(
                    id=f"device_{int(time.time())}",
                    name=name,
                    device_type=device_type,
                    mac_address=mac_address,
                    ip_address=device_request.get("ip_address"),
                    location=device_request.get("location"),
                    description=device_request.get("description"),
                    manufacturer=device_request.get("manufacturer"),
                    model=device_request.get("model"),
                    firmware_version=device_request.get("firmware_version"),
                    status=DeviceStatus.ONLINE,
                    last_seen=datetime.utcnow(),
                    configuration=json.dumps(device_request.get("configuration", {})),
                    capabilities=json.dumps(device_request.get("capabilities", []))
                )
                
                self.db.add(device)
                self.db.commit()
                
                # Add to device registry
                self.device_registry[device.id] = device
                
                IOT_DEVICES.labels(status=device.status).inc()
                
                return {
                    "message": "Device registered successfully",
                    "device_id": device.id,
                    "name": device.name,
                    "device_type": device.device_type,
                    "status": device.status
                }
                
            except HTTPException:
                raise
            except Exception as e:
                self.db.rollback()
                logger.error(f"Error registering device: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/devices", tags=["Devices"])
        async def get_devices():
            """Get all IoT devices."""
            try:
                devices = self.db.query(IoTDevice).filter(IoTDevice.is_active == True).all()
                
                return {
                    "devices": [
                        {
                            "id": device.id,
                            "name": device.name,
                            "device_type": device.device_type,
                            "mac_address": device.mac_address,
                            "ip_address": device.ip_address,
                            "location": device.location,
                            "status": device.status,
                            "last_seen": device.last_seen.isoformat() if device.last_seen else None,
                            "battery_level": device.battery_level,
                            "signal_strength": device.signal_strength,
                            "configuration": json.loads(device.configuration),
                            "capabilities": json.loads(device.capabilities),
                            "created_at": device.created_at.isoformat()
                        }
                        for device in devices
                    ],
                    "total": len(devices)
                }
                
            except Exception as e:
                logger.error(f"Error getting devices: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/devices/{device_id}/data", tags=["Data"])
        async def send_device_data(device_id: str, data_request: dict):
            """Send data from IoT device."""
            try:
                # Validate request
                required_fields = ["data_type", "value"]
                if not all(field in data_request for field in required_fields):
                    raise HTTPException(status_code=400, detail="Missing required fields")
                
                # Get device
                device = self.db.query(IoTDevice).filter(IoTDevice.id == device_id).first()
                if not device:
                    raise HTTPException(status_code=404, detail="Device not found")
                
                data_type = data_request["data_type"]
                value = float(data_request["value"])
                unit = data_request.get("unit")
                metadata = data_request.get("metadata", {})
                
                # Create data point
                data_point = IoTDataPoint(
                    id=f"data_{int(time.time())}",
                    device_id=device_id,
                    data_type=data_type,
                    value=value,
                    unit=unit,
                    metadata=json.dumps(metadata),
                    quality_score=self.calculate_data_quality(device, data_type, value)
                )
                
                self.db.add(data_point)
                
                # Update device status
                device.last_seen = datetime.utcnow()
                device.status = DeviceStatus.ONLINE
                
                self.db.commit()
                
                # Check for alerts
                await self.check_alerts(device, data_type, value)
                
                # Process automations
                await self.process_automations(device, data_type, value)
                
                # Broadcast to WebSocket connections
                await self.broadcast_data_update(device, data_point)
                
                IOT_DATA_POINTS.labels(device_type=device.device_type, data_type=data_type).inc()
                
                return {
                    "message": "Data received successfully",
                    "data_point_id": data_point.id,
                    "device_id": device_id,
                    "data_type": data_type,
                    "value": value,
                    "quality_score": data_point.quality_score
                }
                
            except HTTPException:
                raise
            except Exception as e:
                self.db.rollback()
                logger.error(f"Error processing device data: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/devices/{device_id}/data", tags=["Data"])
        async def get_device_data(device_id: str, limit: int = 100, data_type: str = None):
            """Get data from specific device."""
            try:
                # Get device
                device = self.db.query(IoTDevice).filter(IoTDevice.id == device_id).first()
                if not device:
                    raise HTTPException(status_code=404, detail="Device not found")
                
                # Build query
                query = self.db.query(IoTDataPoint).filter(IoTDataPoint.device_id == device_id)
                
                if data_type:
                    query = query.filter(IoTDataPoint.data_type == data_type)
                
                data_points = query.order_by(IoTDataPoint.timestamp.desc()).limit(limit).all()
                
                return {
                    "device_id": device_id,
                    "device_name": device.name,
                    "data_points": [
                        {
                            "id": dp.id,
                            "data_type": dp.data_type,
                            "value": dp.value,
                            "unit": dp.unit,
                            "timestamp": dp.timestamp.isoformat(),
                            "metadata": json.loads(dp.metadata),
                            "quality_score": dp.quality_score
                        }
                        for dp in data_points
                    ],
                    "total": len(data_points)
                }
                
            except Exception as e:
                logger.error(f"Error getting device data: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/automations", tags=["Automations"])
        async def create_automation(automation_request: dict):
            """Create IoT automation."""
            try:
                # Validate request
                required_fields = ["name", "automation_type", "trigger_device_id", "trigger_condition", "action_device_id", "action_command"]
                if not all(field in automation_request for field in required_fields):
                    raise HTTPException(status_code=400, detail="Missing required fields")
                
                # Create automation
                automation = IoTAutomation(
                    id=f"automation_{int(time.time())}",
                    name=automation_request["name"],
                    automation_type=automation_request["automation_type"],
                    trigger_device_id=automation_request["trigger_device_id"],
                    trigger_condition=automation_request["trigger_condition"],
                    action_device_id=automation_request["action_device_id"],
                    action_command=automation_request["action_command"],
                    is_active=automation_request.get("is_active", True)
                )
                
                self.db.add(automation)
                self.db.commit()
                
                return {
                    "message": "Automation created successfully",
                    "automation_id": automation.id,
                    "name": automation.name,
                    "automation_type": automation.automation_type,
                    "is_active": automation.is_active
                }
                
            except Exception as e:
                self.db.rollback()
                logger.error(f"Error creating automation: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/automations", tags=["Automations"])
        async def get_automations():
            """Get all automations."""
            try:
                automations = self.db.query(IoTAutomation).filter(IoTAutomation.is_active == True).all()
                
                return {
                    "automations": [
                        {
                            "id": automation.id,
                            "name": automation.name,
                            "automation_type": automation.automation_type,
                            "trigger_device_id": automation.trigger_device_id,
                            "trigger_condition": automation.trigger_condition,
                            "action_device_id": automation.action_device_id,
                            "action_command": automation.action_command,
                            "last_executed": automation.last_executed.isoformat() if automation.last_executed else None,
                            "execution_count": automation.execution_count,
                            "success_count": automation.success_count,
                            "failure_count": automation.failure_count,
                            "created_at": automation.created_at.isoformat()
                        }
                        for automation in automations
                    ],
                    "total": len(automations)
                }
                
            except Exception as e:
                logger.error(f"Error getting automations: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/alerts", tags=["Alerts"])
        async def get_alerts(limit: int = 100, severity: str = None, resolved: bool = None):
            """Get IoT alerts."""
            try:
                query = self.db.query(IoTAlert)
                
                if severity:
                    query = query.filter(IoTAlert.severity == severity)
                
                if resolved is not None:
                    query = query.filter(IoTAlert.is_resolved == resolved)
                
                alerts = query.order_by(IoTAlert.created_at.desc()).limit(limit).all()
                
                return {
                    "alerts": [
                        {
                            "id": alert.id,
                            "device_id": alert.device_id,
                            "device_name": alert.device.name if alert.device else None,
                            "alert_type": alert.alert_type,
                            "severity": alert.severity,
                            "message": alert.message,
                            "threshold_value": alert.threshold_value,
                            "actual_value": alert.actual_value,
                            "is_resolved": alert.is_resolved,
                            "resolved_at": alert.resolved_at.isoformat() if alert.resolved_at else None,
                            "metadata": json.loads(alert.metadata),
                            "created_at": alert.created_at.isoformat()
                        }
                        for alert in alerts
                    ],
                    "total": len(alerts)
                }
                
            except Exception as e:
                logger.error(f"Error getting alerts: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time IoT data."""
            await websocket.accept()
            self.websocket_connections.append(websocket)
            
            try:
                while True:
                    # Keep connection alive
                    await websocket.receive_text()
            except WebSocketDisconnect:
                self.websocket_connections.remove(websocket)
        
        @self.app.get("/dashboard", tags=["Dashboard"])
        async def get_iot_dashboard():
            """Get IoT system dashboard."""
            try:
                # Get statistics
                total_devices = self.db.query(IoTDevice).count()
                online_devices = self.db.query(IoTDevice).filter(IoTDevice.status == DeviceStatus.ONLINE).count()
                offline_devices = self.db.query(IoTDevice).filter(IoTDevice.status == DeviceStatus.OFFLINE).count()
                total_data_points = self.db.query(IoTDataPoint).count()
                active_alerts = self.db.query(IoTAlert).filter(IoTAlert.is_resolved == False).count()
                total_automations = self.db.query(IoTAutomation).count()
                active_automations = self.db.query(IoTAutomation).filter(IoTAutomation.is_active == True).count()
                
                # Get device type distribution
                device_types = {}
                for device_type in DeviceType:
                    count = self.db.query(IoTDevice).filter(IoTDevice.device_type == device_type.value).count()
                    device_types[device_type.value] = count
                
                # Get recent data points
                recent_data = self.db.query(IoTDataPoint).order_by(
                    IoTDataPoint.timestamp.desc()
                ).limit(20).all()
                
                # Get recent alerts
                recent_alerts = self.db.query(IoTAlert).order_by(
                    IoTAlert.created_at.desc()
                ).limit(10).all()
                
                return {
                    "summary": {
                        "total_devices": total_devices,
                        "online_devices": online_devices,
                        "offline_devices": offline_devices,
                        "total_data_points": total_data_points,
                        "active_alerts": active_alerts,
                        "total_automations": total_automations,
                        "active_automations": active_automations
                    },
                    "device_type_distribution": device_types,
                    "recent_data": [
                        {
                            "device_id": dp.device_id,
                            "device_name": dp.device.name if dp.device else None,
                            "data_type": dp.data_type,
                            "value": dp.value,
                            "unit": dp.unit,
                            "timestamp": dp.timestamp.isoformat(),
                            "quality_score": dp.quality_score
                        }
                        for dp in recent_data
                    ],
                    "recent_alerts": [
                        {
                            "id": alert.id,
                            "device_name": alert.device.name if alert.device else None,
                            "alert_type": alert.alert_type,
                            "severity": alert.severity,
                            "message": alert.message,
                            "created_at": alert.created_at.isoformat()
                        }
                        for alert in recent_alerts
                    ],
                    "timestamp": datetime.now().isoformat()
                }
                
            except Exception as e:
                logger.error(f"Error getting dashboard data: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    
    def setup_default_data(self):
        """Setup default IoT data."""
        try:
            # Create sample devices
            sample_devices = [
                {
                    "name": "Living Room Temperature Sensor",
                    "device_type": DeviceType.SENSOR,
                    "mac_address": "AA:BB:CC:DD:EE:01",
                    "location": "Living Room",
                    "description": "Temperature and humidity sensor",
                    "manufacturer": "BUL IoT",
                    "model": "TEMP-001",
                    "capabilities": ["temperature", "humidity"]
                },
                {
                    "name": "Smart Light Bulb",
                    "device_type": DeviceType.LIGHT,
                    "mac_address": "AA:BB:CC:DD:EE:02",
                    "location": "Bedroom",
                    "description": "Smart LED light bulb",
                    "manufacturer": "BUL IoT",
                    "model": "LIGHT-001",
                    "capabilities": ["brightness", "color", "on_off"]
                },
                {
                    "name": "Motion Detector",
                    "device_type": DeviceType.MOTION_DETECTOR,
                    "mac_address": "AA:BB:CC:DD:EE:03",
                    "location": "Hallway",
                    "description": "PIR motion detector",
                    "manufacturer": "BUL IoT",
                    "model": "MOTION-001",
                    "capabilities": ["motion_detection", "light_level"]
                }
            ]
            
            for device_data in sample_devices:
                device = IoTDevice(
                    id=f"device_{device_data['name'].lower().replace(' ', '_')}",
                    name=device_data["name"],
                    device_type=device_data["device_type"],
                    mac_address=device_data["mac_address"],
                    location=device_data["location"],
                    description=device_data["description"],
                    manufacturer=device_data["manufacturer"],
                    model=device_data["model"],
                    status=DeviceStatus.ONLINE,
                    last_seen=datetime.utcnow(),
                    capabilities=json.dumps(device_data["capabilities"]),
                    is_active=True
                )
                
                self.db.add(device)
                self.device_registry[device.id] = device
            
            self.db.commit()
            logger.info("Default IoT data created")
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error creating default IoT data: {e}")
    
    def setup_mqtt_client(self):
        """Setup MQTT client for IoT communication."""
        try:
            self.mqtt_client = mqtt.Client()
            self.mqtt_client.username_pw_set(
                IOT_CONFIG["mqtt_username"], 
                IOT_CONFIG["mqtt_password"]
            )
            
            # Set callbacks
            self.mqtt_client.on_connect = self.on_mqtt_connect
            self.mqtt_client.on_message = self.on_mqtt_message
            self.mqtt_client.on_disconnect = self.on_mqtt_disconnect
            
            # Connect to broker
            self.mqtt_client.connect(
                IOT_CONFIG["mqtt_broker"], 
                IOT_CONFIG["mqtt_port"], 
                60
            )
            
            # Start loop
            self.mqtt_client.loop_start()
            
            logger.info("MQTT client connected")
            
        except Exception as e:
            logger.error(f"Error setting up MQTT client: {e}")
    
    def on_mqtt_connect(self, client, userdata, flags, rc):
        """MQTT connection callback."""
        if rc == 0:
            logger.info("Connected to MQTT broker")
            # Subscribe to device topics
            client.subscribe("iot/devices/+/data")
            client.subscribe("iot/devices/+/status")
        else:
            logger.error(f"Failed to connect to MQTT broker: {rc}")
    
    def on_mqtt_message(self, client, userdata, msg):
        """MQTT message callback."""
        try:
            topic = msg.topic
            payload = json.loads(msg.payload.decode())
            
            # Extract device ID from topic
            topic_parts = topic.split('/')
            if len(topic_parts) >= 3:
                device_id = topic_parts[2]
                
                if topic.endswith('/data'):
                    # Process device data
                    asyncio.create_task(self.process_mqtt_data(device_id, payload))
                elif topic.endswith('/status'):
                    # Process device status
                    asyncio.create_task(self.process_mqtt_status(device_id, payload))
            
        except Exception as e:
            logger.error(f"Error processing MQTT message: {e}")
    
    def on_mqtt_disconnect(self, client, userdata, rc):
        """MQTT disconnection callback."""
        logger.warning(f"Disconnected from MQTT broker: {rc}")
    
    async def process_mqtt_data(self, device_id: str, payload: dict):
        """Process MQTT data message."""
        try:
            # Get device
            device = self.db.query(IoTDevice).filter(IoTDevice.id == device_id).first()
            if not device:
                return
            
            # Process data
            data_type = payload.get("data_type")
            value = payload.get("value")
            unit = payload.get("unit")
            metadata = payload.get("metadata", {})
            
            if data_type and value is not None:
                # Create data point
                data_point = IoTDataPoint(
                    id=f"data_{int(time.time())}",
                    device_id=device_id,
                    data_type=data_type,
                    value=float(value),
                    unit=unit,
                    metadata=json.dumps(metadata),
                    quality_score=self.calculate_data_quality(device, data_type, float(value))
                )
                
                self.db.add(data_point)
                
                # Update device status
                device.last_seen = datetime.utcnow()
                device.status = DeviceStatus.ONLINE
                
                self.db.commit()
                
                # Check for alerts
                await self.check_alerts(device, data_type, float(value))
                
                # Process automations
                await self.process_automations(device, data_type, float(value))
                
                # Broadcast to WebSocket connections
                await self.broadcast_data_update(device, data_point)
                
                IOT_DATA_POINTS.labels(device_type=device.device_type, data_type=data_type).inc()
                
        except Exception as e:
            logger.error(f"Error processing MQTT data: {e}")
    
    async def process_mqtt_status(self, device_id: str, payload: dict):
        """Process MQTT status message."""
        try:
            # Get device
            device = self.db.query(IoTDevice).filter(IoTDevice.id == device_id).first()
            if not device:
                return
            
            # Update device status
            device.last_seen = datetime.utcnow()
            device.status = payload.get("status", DeviceStatus.ONLINE)
            device.battery_level = payload.get("battery_level")
            device.signal_strength = payload.get("signal_strength")
            
            self.db.commit()
            
        except Exception as e:
            logger.error(f"Error processing MQTT status: {e}")
    
    def calculate_data_quality(self, device: IoTDevice, data_type: str, value: float) -> float:
        """Calculate data quality score."""
        try:
            # Basic quality calculation based on thresholds
            thresholds = IOT_CONFIG["alert_thresholds"].get(data_type)
            if not thresholds:
                return 1.0
            
            min_val = thresholds.get("min")
            max_val = thresholds.get("max")
            
            if min_val is not None and value < min_val:
                return 0.3
            elif max_val is not None and value > max_val:
                return 0.3
            else:
                return 1.0
                
        except Exception:
            return 1.0
    
    async def check_alerts(self, device: IoTDevice, data_type: str, value: float):
        """Check for alerts based on data values."""
        try:
            thresholds = IOT_CONFIG["alert_thresholds"].get(data_type)
            if not thresholds:
                return
            
            min_val = thresholds.get("min")
            max_val = thresholds.get("max")
            
            alert_triggered = False
            severity = AlertSeverity.MEDIUM
            message = ""
            
            if min_val is not None and value < min_val:
                alert_triggered = True
                severity = AlertSeverity.HIGH
                message = f"{data_type} value {value} is below minimum threshold {min_val}"
            elif max_val is not None and value > max_val:
                alert_triggered = True
                severity = AlertSeverity.HIGH
                message = f"{data_type} value {value} is above maximum threshold {max_val}"
            
            if alert_triggered:
                # Create alert
                alert = IoTAlert(
                    id=f"alert_{int(time.time())}",
                    device_id=device.id,
                    alert_type=f"{data_type}_threshold",
                    severity=severity,
                    message=message,
                    threshold_value=min_val or max_val,
                    actual_value=value,
                    metadata=json.dumps({"data_type": data_type})
                )
                
                self.db.add(alert)
                self.db.commit()
                
                IOT_ALERTS.labels(severity=severity, type=f"{data_type}_threshold").inc()
                
                # Broadcast alert to WebSocket connections
                await self.broadcast_alert(alert)
                
        except Exception as e:
            logger.error(f"Error checking alerts: {e}")
    
    async def process_automations(self, device: IoTDevice, data_type: str, value: float):
        """Process automations based on device data."""
        try:
            # Get active automations for this device
            automations = self.db.query(IoTAutomation).filter(
                IoTAutomation.trigger_device_id == device.id,
                IoTAutomation.is_active == True
            ).all()
            
            for automation in automations:
                try:
                    # Parse trigger condition
                    condition = json.loads(automation.trigger_condition)
                    
                    # Check if condition is met
                    if self.evaluate_condition(condition, data_type, value):
                        # Execute automation
                        await self.execute_automation(automation)
                        
                except Exception as e:
                    logger.error(f"Error processing automation {automation.id}: {e}")
                    
        except Exception as e:
            logger.error(f"Error processing automations: {e}")
    
    def evaluate_condition(self, condition: dict, data_type: str, value: float) -> bool:
        """Evaluate automation condition."""
        try:
            if condition.get("data_type") != data_type:
                return False
            
            operator = condition.get("operator")
            threshold = condition.get("threshold")
            
            if operator == "greater_than":
                return value > threshold
            elif operator == "less_than":
                return value < threshold
            elif operator == "equals":
                return value == threshold
            elif operator == "greater_than_or_equal":
                return value >= threshold
            elif operator == "less_than_or_equal":
                return value <= threshold
            
            return False
            
        except Exception:
            return False
    
    async def execute_automation(self, automation: IoTAutomation):
        """Execute automation action."""
        try:
            # Get action device
            action_device = self.db.query(IoTDevice).filter(
                IoTDevice.id == automation.action_device_id
            ).first()
            
            if not action_device:
                automation.failure_count += 1
                self.db.commit()
                return
            
            # Parse action command
            command = json.loads(automation.action_command)
            
            # Send command to device via MQTT
            topic = f"iot/devices/{action_device.id}/command"
            payload = json.dumps(command)
            
            self.mqtt_client.publish(topic, payload)
            
            # Update automation stats
            automation.last_executed = datetime.utcnow()
            automation.execution_count += 1
            automation.success_count += 1
            
            self.db.commit()
            
            IOT_AUTOMATIONS.labels(automation_type=automation.automation_type).inc()
            
            logger.info(f"Automation {automation.name} executed successfully")
            
        except Exception as e:
            logger.error(f"Error executing automation {automation.id}: {e}")
            
            # Update failure count
            automation.failure_count += 1
            self.db.commit()
    
    async def broadcast_data_update(self, device: IoTDevice, data_point: IoTDataPoint):
        """Broadcast data update to WebSocket connections."""
        try:
            message = {
                "type": "data_update",
                "device_id": device.id,
                "device_name": device.name,
                "data_type": data_point.data_type,
                "value": data_point.value,
                "unit": data_point.unit,
                "timestamp": data_point.timestamp.isoformat(),
                "quality_score": data_point.quality_score
            }
            
            for connection in self.websocket_connections:
                try:
                    await connection.send_text(json.dumps(message))
                except:
                    # Remove disconnected connections
                    self.websocket_connections.remove(connection)
                    
        except Exception as e:
            logger.error(f"Error broadcasting data update: {e}")
    
    async def broadcast_alert(self, alert: IoTAlert):
        """Broadcast alert to WebSocket connections."""
        try:
            message = {
                "type": "alert",
                "alert_id": alert.id,
                "device_id": alert.device_id,
                "device_name": alert.device.name if alert.device else None,
                "alert_type": alert.alert_type,
                "severity": alert.severity,
                "message": alert.message,
                "timestamp": alert.created_at.isoformat()
            }
            
            for connection in self.websocket_connections:
                try:
                    await connection.send_text(json.dumps(message))
                except:
                    # Remove disconnected connections
                    self.websocket_connections.remove(connection)
                    
        except Exception as e:
            logger.error(f"Error broadcasting alert: {e}")
    
    def start_background_tasks(self):
        """Start background tasks."""
        def background_loop():
            while True:
                try:
                    # Check device heartbeats
                    asyncio.create_task(self.check_device_heartbeats())
                    
                    # Clean old data
                    asyncio.create_task(self.clean_old_data())
                    
                    time.sleep(IOT_CONFIG["automation_check_interval"])
                except Exception as e:
                    logger.error(f"Background task error: {e}")
                    time.sleep(60)
        
        import threading
        background_thread = threading.Thread(target=background_loop, daemon=True)
        background_thread.start()
        
        logger.info("Background tasks started")
    
    async def check_device_heartbeats(self):
        """Check device heartbeats and mark offline devices."""
        try:
            cutoff_time = datetime.utcnow() - timedelta(seconds=IOT_CONFIG["device_heartbeat_interval"])
            
            offline_devices = self.db.query(IoTDevice).filter(
                IoTDevice.last_seen < cutoff_time,
                IoTDevice.status == DeviceStatus.ONLINE
            ).all()
            
            for device in offline_devices:
                device.status = DeviceStatus.OFFLINE
                logger.warning(f"Device {device.name} marked as offline")
            
            if offline_devices:
                self.db.commit()
                
        except Exception as e:
            logger.error(f"Error checking device heartbeats: {e}")
    
    async def clean_old_data(self):
        """Clean old data points."""
        try:
            cutoff_time = datetime.utcnow() - timedelta(days=IOT_CONFIG["data_retention_days"])
            
            old_data = self.db.query(IoTDataPoint).filter(
                IoTDataPoint.timestamp < cutoff_time
            ).all()
            
            for data_point in old_data:
                self.db.delete(data_point)
            
            if old_data:
                self.db.commit()
                logger.info(f"Cleaned {len(old_data)} old data points")
                
        except Exception as e:
            logger.error(f"Error cleaning old data: {e}")
    
    def run(self, host: str = "0.0.0.0", port: int = 8009, debug: bool = False):
        """Run the IoT system."""
        logger.info(f"Starting IoT System on {host}:{port}")
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            log_level="debug" if debug else "info"
        )

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="BUL IoT System")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8009, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    # Create and run IoT system
    system = AdvancedIoTSystem()
    system.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == "__main__":
    main()
