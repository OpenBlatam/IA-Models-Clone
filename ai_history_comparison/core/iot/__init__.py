"""
IoT Integration Module

This module provides comprehensive Internet of Things capabilities for the AI History Comparison System.
"""

from .iot_integration import (
    AdvancedIoTManager,
    DeviceType,
    ProtocolType,
    DataType,
    DeviceStatus,
    IoTDevice,
    SensorData,
    DeviceCommand,
    IoTGateway,
    BaseIoTProtocol,
    MQTTProtocol,
    CoAPProtocol,
    HTTPProtocol,
    DeviceManager,
    DataCollector,
    CommandManager,
    EdgeProcessor,
    get_iot_manager,
    initialize_iot,
    shutdown_iot,
    register_iot_device,
    send_iot_command
)

__all__ = [
    "AdvancedIoTManager",
    "DeviceType",
    "ProtocolType",
    "DataType",
    "DeviceStatus",
    "IoTDevice",
    "SensorData",
    "DeviceCommand",
    "IoTGateway",
    "BaseIoTProtocol",
    "MQTTProtocol",
    "CoAPProtocol",
    "HTTPProtocol",
    "DeviceManager",
    "DataCollector",
    "CommandManager",
    "EdgeProcessor",
    "get_iot_manager",
    "initialize_iot",
    "shutdown_iot",
    "register_iot_device",
    "send_iot_command"
]





















