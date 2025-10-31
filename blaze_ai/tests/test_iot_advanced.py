"""
Tests for Blaze AI Advanced IoT Module

This module provides comprehensive testing for all IoT functionality
including device management, data collection, protocols, and security.
"""

import asyncio
import json
import pytest
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

from blaze_ai.modules.iot_advanced import (
    IoTAdvancedModule, IoTAdvancedConfig, DeviceType, ProtocolType,
    DeviceStatus, DataType, SecurityLevel, DeviceInfo, IoTData,
    DeviceCommand, IoTMetrics, DeviceManager, DataManager,
    ProtocolManager, SecurityManager
)

# Test Configuration
@pytest.fixture
def iot_config():
    """Create a test IoT configuration."""
    return IoTAdvancedConfig(
        network_name="test-iot-network",
        max_devices=100,
        supported_protocols=[ProtocolType.MQTT, ProtocolType.HTTP],
        security_level=SecurityLevel.STANDARD,
        health_check_interval=1.0,  # Fast for testing
        data_sync_interval=2.0
    )

@pytest.fixture
def iot_module(iot_config):
    """Create an IoT Advanced module for testing."""
    return IoTAdvancedModule(iot_config)

@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

# Test Device Manager
class TestDeviceManager:
    """Test device manager functionality."""
    
    def test_init(self, iot_config):
        """Test device manager initialization."""
        manager = DeviceManager(iot_config)
        assert manager.config == iot_config
        assert manager.devices == {}
        assert manager.device_commands == {}
        assert manager.device_data == {}
        assert manager.device_handlers == {}
    
    @pytest.mark.asyncio
    async def test_register_device(self, iot_config):
        """Test device registration."""
        manager = DeviceManager(iot_config)
        
        device_info = DeviceInfo(
            device_id="test_device_001",
            device_name="Test Device",
            device_type=DeviceType.SENSOR,
            protocol=ProtocolType.MQTT,
            status=DeviceStatus.ONLINE,
            ip_address="192.168.1.100",
            mac_address="AA:BB:CC:DD:EE:01",
            firmware_version="1.0.0",
            hardware_version="1.0.0",
            capabilities=["test_capability"],
            data_types=[DataType.TEMPERATURE],
            location={"lat": 40.0, "lon": -74.0, "alt": 10.0},
            created_at=datetime.now(),
            last_seen=datetime.now()
        )
        
        result = await manager.register_device(device_info)
        assert result is True
        assert device_info.device_id in manager.devices
        assert device_info.device_id in manager.device_data
    
    @pytest.mark.asyncio
    async def test_register_duplicate_device(self, iot_config):
        """Test registering duplicate device."""
        manager = DeviceManager(iot_config)
        
        device_info = DeviceInfo(
            device_id="test_device_002",
            device_name="Test Device",
            device_type=DeviceType.SENSOR,
            protocol=ProtocolType.MQTT,
            status=DeviceStatus.ONLINE,
            ip_address="192.168.1.101",
            mac_address="AA:BB:CC:DD:EE:02",
            firmware_version="1.0.0",
            hardware_version="1.0.0",
            capabilities=["test_capability"],
            data_types=[DataType.TEMPERATURE],
            location={"lat": 40.0, "lon": -74.0, "alt": 10.0},
            created_at=datetime.now(),
            last_seen=datetime.now()
        )
        
        # Register first time
        result1 = await manager.register_device(device_info)
        assert result1 is True
        
        # Try to register again
        result2 = await manager.register_device(device_info)
        assert result2 is False
    
    @pytest.mark.asyncio
    async def test_unregister_device(self, iot_config):
        """Test device unregistration."""
        manager = DeviceManager(iot_config)
        
        device_info = DeviceInfo(
            device_id="test_device_003",
            device_name="Test Device",
            device_type=DeviceType.SENSOR,
            protocol=ProtocolType.MQTT,
            status=DeviceStatus.ONLINE,
            ip_address="192.168.1.102",
            mac_address="AA:BB:CC:DD:EE:03",
            firmware_version="1.0.0",
            hardware_version="1.0.0",
            capabilities=["test_capability"],
            data_types=[DataType.TEMPERATURE],
            location={"lat": 40.0, "lon": -74.0, "alt": 10.0},
            created_at=datetime.now(),
            last_seen=datetime.now()
        )
        
        # Register device
        await manager.register_device(device_info)
        assert device_info.device_id in manager.devices
        
        # Unregister device
        result = await manager.unregister_device(device_info.device_id)
        assert result is True
        assert device_info.device_id not in manager.devices
        assert device_info.device_id not in manager.device_data
    
    @pytest.mark.asyncio
    async def test_update_device_status(self, iot_config):
        """Test device status update."""
        manager = DeviceManager(iot_config)
        
        device_info = DeviceInfo(
            device_id="test_device_004",
            device_name="Test Device",
            device_type=DeviceType.SENSOR,
            protocol=ProtocolType.MQTT,
            status=DeviceStatus.ONLINE,
            ip_address="192.168.1.103",
            mac_address="AA:BB:CC:DD:EE:04",
            firmware_version="1.0.0",
            hardware_version="1.0.0",
            capabilities=["test_capability"],
            data_types=[DataType.TEMPERATURE],
            location={"lat": 40.0, "lon": -74.0, "alt": 10.0},
            created_at=datetime.now(),
            last_seen=datetime.now()
        )
        
        # Register device
        await manager.register_device(device_info)
        
        # Update status
        result = await manager.update_device_status(device_info.device_id, DeviceStatus.OFFLINE)
        assert result is True
        
        # Check status was updated
        device = manager.devices[device_info.device_id]
        assert device.status == DeviceStatus.OFFLINE
    
    @pytest.mark.asyncio
    async def test_get_devices_by_type(self, iot_config):
        """Test getting devices by type."""
        manager = DeviceManager(iot_config)
        
        # Register multiple devices of different types
        sensor_device = DeviceInfo(
            device_id="sensor_001",
            device_name="Temperature Sensor",
            device_type=DeviceType.SENSOR,
            protocol=ProtocolType.MQTT,
            status=DeviceStatus.ONLINE,
            ip_address="192.168.1.200",
            mac_address="AA:BB:CC:DD:EE:10",
            firmware_version="1.0.0",
            hardware_version="1.0.0",
            capabilities=["temperature_reading"],
            data_types=[DataType.TEMPERATURE],
            location={"lat": 40.0, "lon": -74.0, "alt": 10.0},
            created_at=datetime.now(),
            last_seen=datetime.now()
        )
        
        actuator_device = DeviceInfo(
            device_id="actuator_001",
            device_name="Smart Light",
            device_type=DeviceType.ACTUATOR,
            protocol=ProtocolType.HTTP,
            status=DeviceStatus.ONLINE,
            ip_address="192.168.1.201",
            mac_address="AA:BB:CC:DD:EE:11",
            firmware_version="1.0.0",
            hardware_version="1.0.0",
            capabilities=["light_control"],
            data_types=[DataType.BINARY],
            location={"lat": 40.0, "lon": -74.0, "alt": 10.0},
            created_at=datetime.now(),
            last_seen=datetime.now()
        )
        
        await manager.register_device(sensor_device)
        await manager.register_device(actuator_device)
        
        # Get devices by type
        sensors = await manager.get_devices_by_type(DeviceType.SENSOR)
        actuators = await manager.get_devices_by_type(DeviceType.ACTUATOR)
        
        assert len(sensors) == 1
        assert len(actuators) == 1
        assert sensors[0].device_type == DeviceType.SENSOR
        assert actuators[0].device_type == DeviceType.ACTUATOR

# Test Data Manager
class TestDataManager:
    """Test data manager functionality."""
    
    def test_init(self, iot_config):
        """Test data manager initialization."""
        manager = DataManager(iot_config)
        assert manager.config == iot_config
        assert manager.data_storage == {}
        assert manager.data_processors == {}
        assert manager.data_filters == {}
    
    @pytest.mark.asyncio
    async def test_store_data(self, iot_config):
        """Test data storage."""
        manager = DataManager(iot_config)
        
        data = IoTData(
            data_id="test_data_001",
            device_id="test_device",
            data_type=DataType.TEMPERATURE,
            value=25.5,
            timestamp=datetime.now(),
            quality=0.95,
            unit="celsius"
        )
        
        result = await manager.store_data(data)
        assert result is True
        assert data.device_id in manager.data_storage
        assert len(manager.data_storage[data.device_id]) == 1
    
    @pytest.mark.asyncio
    async def test_get_device_data(self, iot_config):
        """Test retrieving device data."""
        manager = DataManager(iot_config)
        
        # Store multiple data points
        for i in range(5):
            data = IoTData(
                data_id=f"test_data_{i:03d}",
                device_id="test_device",
                data_type=DataType.TEMPERATURE,
                value=20.0 + i,
                timestamp=datetime.now() - timedelta(minutes=i),
                quality=0.9,
                unit="celsius"
            )
            await manager.store_data(data)
        
        # Get all data
        all_data = await manager.get_device_data("test_device")
        assert len(all_data) == 5
        
        # Get data with time filter
        recent_data = await manager.get_device_data(
            "test_device",
            start_time=datetime.now() - timedelta(minutes=3)
        )
        assert len(recent_data) == 4
        
        # Get data with type filter
        temp_data = await manager.get_device_data(
            "test_device",
            data_type=DataType.TEMPERATURE
        )
        assert len(temp_data) == 5
    
    @pytest.mark.asyncio
    async def test_get_aggregated_data(self, iot_config):
        """Test data aggregation."""
        manager = DataManager(iot_config)
        
        # Store data points
        for i in range(10):
            data = IoTData(
                data_id=f"test_data_{i:03d}",
                device_id="test_device",
                data_type=DataType.TEMPERATURE,
                value=20.0 + i,
                timestamp=datetime.now() - timedelta(minutes=i),
                quality=0.9,
                unit="celsius"
            )
            await manager.store_data(data)
        
        # Test average aggregation
        avg_result = await manager.get_aggregated_data("test_device", "average", timedelta(hours=1))
        assert "error" not in avg_result
        assert avg_result["aggregation"] == "average"
        assert avg_result["data_points"] == 10
        
        # Test sum aggregation
        sum_result = await manager.get_aggregated_data("test_device", "sum", timedelta(hours=1))
        assert "error" not in sum_result
        assert sum_result["aggregation"] == "sum"
        
        # Test unknown aggregation
        unknown_result = await manager.get_aggregated_data("test_device", "unknown", timedelta(hours=1))
        assert "error" in unknown_result

# Test Protocol Manager
class TestProtocolManager:
    """Test protocol manager functionality."""
    
    def test_init(self, iot_config):
        """Test protocol manager initialization."""
        manager = ProtocolManager(iot_config)
        assert manager.config == iot_config
        assert manager.mqtt_client is None
        assert manager.http_server is None
        assert manager.websocket_server is None
        assert manager.active_connections == {}
    
    @pytest.mark.asyncio
    async def test_initialize(self, iot_config):
        """Test protocol manager initialization."""
        manager = ProtocolManager(iot_config)
        
        # Mock MQTT initialization
        with patch.object(manager, '_initialize_mqtt') as mock_mqtt:
            mock_mqtt.return_value = None
            
            result = await manager.initialize()
            assert result is True
    
    @pytest.mark.asyncio
    async def test_send_command(self, iot_config):
        """Test sending commands."""
        manager = ProtocolManager(iot_config)
        
        command = DeviceCommand(
            command_id="test_command_001",
            device_id="test_device",
            command_type="test",
            parameters={},
            priority=1,
            timeout=30.0
        )
        
        # Mock device info
        with patch.object(manager, '_get_device_info') as mock_get_info:
            mock_device = Mock()
            mock_device.protocol = ProtocolType.MQTT
            mock_get_info.return_value = mock_device
            
            # Mock MQTT command sending
            with patch.object(manager, '_send_mqtt_command') as mock_mqtt:
                mock_mqtt.return_value = True
                
                result = await manager.send_command("test_device", command)
                assert result is True

# Test Security Manager
class TestSecurityManager:
    """Test security manager functionality."""
    
    def test_init(self, iot_config):
        """Test security manager initialization."""
        manager = SecurityManager(iot_config)
        assert manager.config == iot_config
        assert manager.device_certificates == {}
        assert manager.access_tokens == {}
        assert manager.security_policies == {}
    
    @pytest.mark.asyncio
    async def test_authenticate_device_none_security(self, iot_config):
        """Test authentication with no security."""
        config = IoTAdvancedConfig(security_level=SecurityLevel.NONE)
        manager = SecurityManager(config)
        
        result = await manager.authenticate_device("test_device", {})
        assert result is True
    
    @pytest.mark.asyncio
    async def test_authenticate_device_basic_security(self, iot_config):
        """Test authentication with basic security."""
        config = IoTAdvancedConfig(security_level=SecurityLevel.BASIC)
        manager = SecurityManager(config)
        
        # Test with valid device
        credentials = {"valid_devices": ["test_device"]}
        result = await manager.authenticate_device("test_device", credentials)
        assert result is True
        
        # Test with invalid device
        result = await manager.authenticate_device("invalid_device", credentials)
        assert result is False
    
    @pytest.mark.asyncio
    async def test_authenticate_device_standard_security(self, iot_config):
        """Test authentication with standard security."""
        config = IoTAdvancedConfig(security_level=SecurityLevel.STANDARD)
        manager = SecurityManager(config)
        
        # Test with valid API key
        credentials = {"api_key": "blaze_ai_test_key"}
        result = await manager.authenticate_device("test_device", credentials)
        assert result is True
        
        # Test with invalid API key
        credentials = {"api_key": "invalid_key"}
        result = await manager.authenticate_device("test_device", credentials)
        assert result is False
    
    @pytest.mark.asyncio
    async def test_authorize_device(self, iot_config):
        """Test device authorization."""
        manager = SecurityManager(iot_config)
        
        # Set up security policy
        manager.security_policies["test_device"] = {
            "allowed_actions": ["read", "write"],
            "allowed_resources": ["data", "config"]
        }
        
        # Test authorized action
        result = await manager.authorize_device("test_device", "read", "data")
        assert result is True
        
        # Test unauthorized action
        result = await manager.authorize_device("test_device", "delete", "data")
        assert result is False
        
        # Test unauthorized resource
        result = await manager.authorize_device("test_device", "read", "system")
        assert result is False

# Test IoT Advanced Module
class TestIoTAdvancedModule:
    """Test main IoT Advanced module functionality."""
    
    @pytest.mark.asyncio
    async def test_init(self, iot_config):
        """Test IoT Advanced module initialization."""
        module = IoTAdvancedModule(iot_config)
        
        assert module.config == iot_config
        assert module.device_manager is not None
        assert module.data_manager is not None
        assert module.protocol_manager is not None
        assert module.security_manager is not None
        assert module.health_check_task is None
        assert module.data_sync_task is None
    
    @pytest.mark.asyncio
    async def test_initialize(self, iot_module):
        """Test module initialization."""
        # Mock protocol manager initialization
        with patch.object(iot_module.protocol_manager, 'initialize') as mock_init:
            mock_init.return_value = True
            
            result = await iot_module.initialize()
            assert result is True
            assert iot_module.status.value == "running"
            assert iot_module.health_check_task is not None
            assert iot_module.data_sync_task is not None
    
    @pytest.mark.asyncio
    async def test_shutdown(self, iot_module):
        """Test module shutdown."""
        # Initialize first
        with patch.object(iot_module.protocol_manager, 'initialize') as mock_init:
            mock_init.return_value = True
            await iot_module.initialize()
        
        # Shutdown
        with patch.object(iot_module.protocol_manager, 'shutdown') as mock_shutdown:
            mock_shutdown.return_value = None
            
            result = await iot_module.shutdown()
            assert result is True
            assert iot_module.status.value == "stopped"
    
    @pytest.mark.asyncio
    async def test_register_device(self, iot_module):
        """Test device registration."""
        # Mock device manager registration
        with patch.object(iot_module.device_manager, 'register_device') as mock_register:
            mock_register.return_value = True
            
            device_data = {
                "name": "Test Device",
                "type": "sensor",
                "protocol": "mqtt",
                "ip_address": "192.168.1.100",
                "mac_address": "AA:BB:CC:DD:EE:01"
            }
            
            device_id = await iot_module.register_device(device_data)
            assert device_id is not None
            assert iot_module.metrics.total_devices == 1
            assert iot_module.metrics.online_devices == 1
    
    @pytest.mark.asyncio
    async def test_store_device_data(self, iot_module):
        """Test storing device data."""
        # Mock data manager storage
        with patch.object(iot_module.data_manager, 'store_data') as mock_store:
            mock_store.return_value = True
            
            data = {
                "device_id": "test_device",
                "type": "temperature",
                "value": 25.5,
                "timestamp": datetime.now().isoformat(),
                "quality": 0.95,
                "unit": "celsius"
            }
            
            result = await iot_module.store_device_data(data)
            assert result is True
            assert iot_module.metrics.total_data_points == 1
    
    @pytest.mark.asyncio
    async def test_get_device_data(self, iot_module):
        """Test getting device data."""
        # Mock data manager retrieval
        with patch.object(iot_module.data_manager, 'get_device_data') as mock_get:
            mock_data = [Mock(), Mock(), Mock()]
            mock_get.return_value = mock_data
            
            data = await iot_module.get_device_data("test_device")
            assert len(data) == 3
    
    @pytest.mark.asyncio
    async def test_get_metrics(self, iot_module):
        """Test getting IoT metrics."""
        # Mock device manager devices
        iot_module.device_manager.devices = {
            "device1": Mock(status=DeviceStatus.ONLINE),
            "device2": Mock(status=DeviceStatus.OFFLINE),
            "device3": Mock(status=DeviceStatus.ERROR)
        }
        
        metrics = await iot_module.get_metrics()
        assert metrics.total_devices == 3
        assert metrics.online_devices == 1
        assert metrics.offline_devices == 1
        assert metrics.error_devices == 1
    
    @pytest.mark.asyncio
    async def test_health_check(self, iot_module):
        """Test health check functionality."""
        # Mock device manager devices
        iot_module.device_manager.devices = {
            "device1": Mock(status=DeviceStatus.ONLINE)
        }
        
        health = await iot_module.health_check()
        
        assert "status" in health
        assert "total_devices" in health
        assert "online_devices" in health
        assert "offline_devices" in health
        assert "error_devices" in health
        assert "total_data_points" in health
        assert "health_check_active" in health
        assert "data_sync_active" in health
        assert "supported_protocols" in health
        assert "security_level" in health

# Test Factory Functions
class TestFactoryFunctions:
    """Test factory functions for creating IoT Advanced modules."""
    
    def test_create_iot_advanced_module_default(self):
        """Test creating IoT Advanced module with default config."""
        module = create_iot_advanced_module()
        assert isinstance(module, IoTAdvancedModule)
        assert module.config.network_name == "blaze-ai-iot-network"
        assert module.config.security_level == SecurityLevel.STANDARD
    
    def test_create_iot_advanced_module_custom(self):
        """Test creating IoT Advanced module with custom config."""
        module = create_iot_advanced_module(
            IoTAdvancedConfig(
                network_name="custom-iot-network",
                security_level=SecurityLevel.HIGH,
                max_devices=500
            )
        )
        assert isinstance(module, IoTAdvancedModule)
        assert module.config.network_name == "custom-iot-network"
        assert module.config.security_level == SecurityLevel.HIGH
        assert module.config.max_devices == 500
    
    def test_create_iot_advanced_module_with_defaults(self):
        """Test creating IoT Advanced module with default overrides."""
        module = create_iot_advanced_module_with_defaults(
            network_name="override-iot-network",
            security_level=SecurityLevel.BASIC
        )
        assert isinstance(module, IoTAdvancedModule)
        assert module.config.network_name == "override-iot-network"
        assert module.config.security_level == SecurityLevel.BASIC
        assert module.config.max_devices == 10000  # Default

# Integration Tests
class TestIntegration:
    """Integration tests for the complete IoT system."""
    
    @pytest.mark.asyncio
    async def test_full_iot_workflow(self, temp_dir):
        """Test complete IoT workflow."""
        config = IoTAdvancedConfig(
            network_name="integration-test",
            iot_data_path=temp_dir,
            health_check_interval=0.5,  # Fast for testing
            data_sync_interval=1.0
        )
        
        iot = IoTAdvancedModule(config)
        
        try:
            # Initialize
            with patch.object(iot.protocol_manager, 'initialize') as mock_init:
                mock_init.return_value = True
                success = await iot.initialize()
                assert success is True
            
            # Register device
            device_id = await iot.register_device({
                "name": "Integration_Device",
                "type": "sensor",
                "protocol": "mqtt",
                "ip_address": "192.168.1.100",
                "mac_address": "AA:BB:CC:DD:EE:99"
            })
            assert device_id is not None
            
            # Store data
            data_success = await iot.store_device_data({
                "device_id": device_id,
                "type": "temperature",
                "value": 23.5,
                "timestamp": datetime.now().isoformat(),
                "quality": 0.95,
                "unit": "celsius"
            })
            assert data_success is True
            
            # Get data
            data = await iot.get_device_data(device_id)
            assert len(data) == 1
            
            # Check metrics
            metrics = await iot.get_metrics()
            assert metrics.total_devices == 1
            assert metrics.total_data_points == 1
            
        finally:
            await iot.shutdown()

# Performance Tests
class TestPerformance:
    """Performance tests for the IoT system."""
    
    @pytest.mark.asyncio
    async def test_device_registration_performance(self, temp_dir):
        """Test device registration performance."""
        config = IoTAdvancedConfig(
            network_name="performance-test",
            iot_data_path=temp_dir
        )
        
        iot = IoTAdvancedModule(config)
        
        try:
            with patch.object(iot.protocol_manager, 'initialize') as mock_init:
                mock_init.return_value = True
                await iot.initialize()
            
            # Register many devices quickly
            start_time = asyncio.get_event_loop().time()
            
            for i in range(100):
                device_id = await iot.register_device({
                    "name": f"Perf_Device_{i:03d}",
                    "type": "sensor",
                    "protocol": "mqtt",
                    "ip_address": f"192.168.1.{100+i}",
                    "mac_address": f"AA:BB:CC:DD:EE:{i:02d}"
                })
                assert device_id is not None
            
            end_time = asyncio.get_event_loop().time()
            registration_time = end_time - start_time
            
            # Check results
            metrics = await iot.get_metrics()
            assert metrics.total_devices == 100
            assert registration_time < 10.0  # Should register quickly
            
        finally:
            await iot.shutdown()

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

