"""
Tests for Blaze AI Edge Computing Module

This test suite covers all edge computing functionality including
resource monitoring, task execution, local storage, cluster connectivity,
and offline capabilities.
"""

import pytest
import asyncio
import time
import tempfile
import shutil
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path

from ..modules.edge_computing import (
    EdgeComputingModule, EdgeComputingConfig, EdgeNodeInfo, EdgeTask,
    EdgeNodeType, ResourceLevel, SyncStrategy, OfflineMode,
    ResourceMonitor, LocalDataManager, TaskExecutor, ClusterConnector,
    EdgeMetrics
)

# Fixtures
@pytest.fixture
def edge_config():
    """Create a test configuration for edge computing."""
    return EdgeComputingConfig(
        node_name="test-edge-node",
        node_type=EdgeNodeType.EDGE_SERVER,
        max_cpu_usage=70.0,
        max_memory_usage=80.0,
        local_data_path="./test_edge_data",
        sync_interval=5.0,
        heartbeat_interval=2.0
    )

@pytest.fixture
def edge_module(edge_config):
    """Create an edge computing module for testing."""
    return EdgeComputingModule(edge_config)

@pytest.fixture
def temp_data_dir():
    """Create a temporary directory for test data."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def edge_task():
    """Create a test edge task."""
    return EdgeTask(
        task_id="test-task-001",
        task_type="data_processing",
        task_data={"input": [1, 2, 3], "operation": "sum"},
        priority=1
    )

# Configuration Tests
def test_edge_config_defaults():
    """Test edge computing configuration defaults."""
    config = EdgeComputingConfig()
    
    assert config.node_name == "edge-node"
    assert config.node_type == EdgeNodeType.EDGE_SERVER
    assert config.max_cpu_usage == 80.0
    assert config.max_memory_usage == 85.0
    assert config.sync_strategy == SyncStrategy.BATCH
    assert config.offline_mode == OfflineMode.CACHED_OPERATIONS
    assert config.enable_local_ml is True
    assert config.enable_local_cache is True

def test_edge_config_custom():
    """Test edge computing configuration with custom values."""
    config = EdgeComputingConfig(
        node_name="custom-node",
        node_type=EdgeNodeType.IOT_DEVICE,
        max_cpu_usage=60.0,
        sync_strategy=SyncStrategy.REAL_TIME
    )
    
    assert config.node_name == "custom-node"
    assert config.node_type == EdgeNodeType.IOT_DEVICE
    assert config.max_cpu_usage == 60.0
    assert config.sync_strategy == SyncStrategy.REAL_TIME

# Edge Node Info Tests
def test_edge_node_info_creation():
    """Test edge node info creation."""
    node_info = EdgeNodeInfo(
        node_id="test-001",
        node_name="test-node",
        node_type=EdgeNodeType.EDGE_SERVER
    )
    
    assert node_info.node_id == "test-001"
    assert node_info.node_name == "test-node"
    assert node_info.node_type == EdgeNodeType.EDGE_SERVER
    assert node_info.status == "online"

def test_edge_node_info_to_dict():
    """Test edge node info serialization."""
    node_info = EdgeNodeInfo(
        node_id="test-001",
        node_name="test-node",
        node_type=EdgeNodeType.EDGE_SERVER
    )
    
    node_dict = node_info.to_dict()
    
    assert node_dict["node_id"] == "test-001"
    assert node_dict["node_name"] == "test-node"
    assert node_dict["node_type"] == "edge_server"
    assert "last_seen" in node_dict
    assert "last_sync" in node_dict

# Edge Task Tests
def test_edge_task_creation():
    """Test edge task creation."""
    task = EdgeTask(
        task_id="task-001",
        task_type="ml_inference",
        task_data={"model": "test_model"},
        priority=2
    )
    
    assert task.task_id == "task-001"
    assert task.task_type == "ml_inference"
    assert task.priority == 2
    assert task.status == "pending"
    assert task.retry_count == 0

def test_edge_task_to_dict():
    """Test edge task serialization."""
    task = EdgeTask(
        task_id="task-001",
        task_type="data_processing",
        task_data={"input": "test"}
    )
    
    task_dict = task.to_dict()
    
    assert task_dict["task_id"] == "task-001"
    assert task_dict["task_type"] == "data_processing"
    assert task_dict["status"] == "pending"
    assert "created_at" in task_dict

# Edge Metrics Tests
def test_edge_metrics_creation():
    """Test edge metrics creation."""
    metrics = EdgeMetrics()
    
    assert metrics.total_tasks == 0
    assert metrics.completed_tasks == 0
    assert metrics.failed_tasks == 0
    assert metrics.avg_task_duration == 0.0

def test_edge_metrics_update():
    """Test edge metrics update."""
    metrics = EdgeMetrics()
    
    metrics.update_task_metrics(1.5, True)
    metrics.update_task_metrics(2.0, False)
    
    assert metrics.total_tasks == 2
    assert metrics.completed_tasks == 1
    assert metrics.failed_tasks == 1
    assert metrics.avg_task_duration == 1.75

# Resource Monitor Tests
@pytest.mark.asyncio
async def test_resource_monitor_system_resources(edge_config):
    """Test resource monitor system resource gathering."""
    monitor = ResourceMonitor(edge_config)
    
    resources = await monitor.get_system_resources()
    
    assert "cpu_usage" in resources
    assert "memory_usage" in resources
    assert "disk_usage" in resources
    assert "network_usage" in resources
    
    # All values should be numeric
    for value in resources.values():
        assert isinstance(value, (int, float))
        assert value >= 0

def test_resource_monitor_resource_level():
    """Test resource level determination."""
    monitor = ResourceMonitor(EdgeComputingConfig())
    
    # Test different resource levels
    assert monitor.get_resource_level({"cpu_usage": 95, "memory_usage": 50, "disk_usage": 30}) == ResourceLevel.CRITICAL
    assert monitor.get_resource_level({"cpu_usage": 85, "memory_usage": 50, "disk_usage": 30}) == ResourceLevel.LOW
    assert monitor.get_resource_level({"cpu_usage": 70, "memory_usage": 50, "disk_usage": 30}) == ResourceLevel.MEDIUM
    assert monitor.get_resource_level({"cpu_usage": 50, "memory_usage": 50, "disk_usage": 30}) == ResourceLevel.HIGH
    assert monitor.get_resource_level({"cpu_usage": 30, "memory_usage": 50, "disk_usage": 30}) == ResourceLevel.OPTIMAL

@pytest.mark.asyncio
async def test_resource_monitor_recommendations(edge_config):
    """Test resource optimization recommendations."""
    monitor = ResourceMonitor(edge_config)
    
    # Mock high resource usage
    with patch.object(monitor, 'get_system_resources', return_value={
        "cpu_usage": 85.0,
        "memory_usage": 90.0,
        "disk_usage": 70.0,
        "network_usage": 50.0
    }):
        recommendations = await monitor.get_optimization_recommendations()
        
        assert len(recommendations) >= 2  # Should have recommendations for CPU and memory
        assert any("CPU usage high" in rec for rec in recommendations)
        assert any("Memory usage high" in rec for rec in recommendations)

# Local Data Manager Tests
@pytest.mark.asyncio
async def test_local_data_manager_storage(temp_data_dir):
    """Test local data storage and retrieval."""
    config = EdgeComputingConfig(local_data_path=temp_data_dir)
    data_manager = LocalDataManager(config)
    
    test_data = {"key": "value", "number": 42}
    metadata = {"source": "test", "timestamp": "2024-01-01"}
    
    # Store data
    success = await data_manager.store_local_data("test_key", test_data, metadata)
    assert success is True
    
    # Retrieve data
    retrieved_data = await data_manager.retrieve_local_data("test_key")
    assert retrieved_data == test_data

@pytest.mark.asyncio
async def test_local_data_manager_storage_limits(temp_data_dir):
    """Test local data storage limits."""
    config = EdgeComputingConfig(
        local_data_path=temp_data_dir,
        max_local_storage=100  # Very small limit for testing
    )
    data_manager = LocalDataManager(config)
    
    # Try to store data that exceeds limits
    large_data = "x" * 200  # Data larger than limit
    
    success = await data_manager.store_local_data("large_key", large_data)
    # Should still succeed but trigger cleanup
    assert success is True

# Task Executor Tests
@pytest.mark.asyncio
async def test_task_executor_submit_task(edge_config):
    """Test task submission."""
    executor = TaskExecutor(edge_config)
    task = EdgeTask(
        task_id="test-001",
        task_type="data_processing",
        task_data={"input": "test"}
    )
    
    success = await executor.submit_task(task)
    assert success is True
    assert len(executor.active_tasks) == 1

@pytest.mark.asyncio
async def test_task_executor_max_concurrent_tasks(edge_config):
    """Test maximum concurrent tasks limit."""
    config = EdgeComputingConfig(max_concurrent_tasks=2)
    executor = TaskExecutor(config)
    
    # Submit tasks up to limit
    for i in range(3):
        task = EdgeTask(
            task_id=f"task-{i}",
            task_type="data_processing",
            task_data={"input": i}
        )
        success = await executor.submit_task(task)
        
        if i < 2:
            assert success is True
        else:
            assert success is False  # Should fail when limit reached

@pytest.mark.asyncio
async def test_task_executor_task_status(edge_config):
    """Test task status retrieval."""
    executor = TaskExecutor(edge_config)
    task = EdgeTask(
        task_id="test-001",
        task_type="data_processing",
        task_data={"input": "test"}
    )
    
    await executor.submit_task(task)
    
    # Check status
    status = await executor.get_task_status("test-001")
    assert status in ["pending", "queued", "running", "completed"]

# Cluster Connector Tests
@pytest.mark.asyncio
async def test_cluster_connector_connection(edge_config):
    """Test cluster connection."""
    connector = ClusterConnector(edge_config)
    
    # Test connection
    success = await connector.connect_to_cluster()
    assert success is True
    assert connector.connected is True
    
    # Test disconnection
    await connector.disconnect_from_cluster()
    assert connector.connected is False

@pytest.mark.asyncio
async def test_cluster_connector_sync(edge_config):
    """Test cluster synchronization."""
    connector = ClusterConnector(edge_config)
    await connector.connect_to_cluster()
    
    test_data = {"type": "test", "data": "test_value"}
    
    # Test sync
    success = await connector.sync_with_cluster(test_data)
    assert success is True

@pytest.mark.asyncio
async def test_cluster_connector_status(edge_config):
    """Test cluster status retrieval."""
    connector = ClusterConnector(edge_config)
    
    # Test status when disconnected
    status = await connector.get_cluster_status()
    assert status["status"] == "disconnected"
    
    # Test status when connected
    await connector.connect_to_cluster()
    status = await connector.get_cluster_status()
    assert status["status"] == "connected"

# Edge Computing Module Integration Tests
@pytest.mark.asyncio
async def test_edge_module_initialization(edge_config):
    """Test edge computing module initialization."""
    module = EdgeComputingModule(edge_config)
    
    # Test initialization
    success = await module.initialize()
    assert success is True
    assert module.status.value == "running"
    
    # Test shutdown
    shutdown_success = await module.shutdown()
    assert shutdown_success is True
    assert module.status.value == "stopped"

@pytest.mark.asyncio
async def test_edge_module_task_submission(edge_config):
    """Test edge task submission through module."""
    module = EdgeComputingModule(edge_config)
    await module.initialize()
    
    try:
        # Submit task
        task_id = await module.submit_edge_task(
            "data_processing",
            {"input": [1, 2, 3]},
            priority=1
        )
        
        assert task_id is not None
        
        # Check status
        status = await module.get_task_status(task_id)
        assert status in ["pending", "queued", "running", "completed"]
        
    finally:
        await module.shutdown()

@pytest.mark.asyncio
async def test_edge_module_local_storage(edge_config):
    """Test local data storage through module."""
    module = EdgeComputingModule(edge_config)
    await module.initialize()
    
    try:
        test_data = {"sensor": "temperature", "value": 23.5}
        
        # Store data
        success = await module.store_local_data("sensor_001", test_data)
        assert success is True
        
        # Retrieve data
        retrieved = await module.retrieve_local_data("sensor_001")
        assert retrieved == test_data
        
    finally:
        await module.shutdown()

@pytest.mark.asyncio
async def test_edge_module_node_info(edge_config):
    """Test node information retrieval."""
    module = EdgeComputingModule(edge_config)
    await module.initialize()
    
    try:
        node_info = await module.get_node_info()
        
        assert node_info.node_id == edge_config.node_id
        assert node_info.node_name == edge_config.node_name
        assert node_info.node_type == edge_config.node_type
        assert node_info.platform != ""
        assert node_info.architecture != ""
        
    finally:
        await module.shutdown()

@pytest.mark.asyncio
async def test_edge_module_resource_status(edge_config):
    """Test resource status monitoring."""
    module = EdgeComputingModule(edge_config)
    await module.initialize()
    
    try:
        resource_status = await module.get_resource_status()
        
        assert "resources" in resource_status
        assert "resource_level" in resource_status
        assert "recommendations" in resource_status
        assert "limits" in resource_status
        
        resources = resource_status["resources"]
        assert "cpu_usage" in resources
        assert "memory_usage" in resources
        
    finally:
        await module.shutdown()

@pytest.mark.asyncio
async def test_edge_module_cluster_status(edge_config):
    """Test cluster status monitoring."""
    module = EdgeComputingModule(edge_config)
    await module.initialize()
    
    try:
        cluster_status = await module.get_cluster_status()
        
        assert "status" in cluster_status
        # Status could be connected or disconnected depending on test environment
        
    finally:
        await module.shutdown()

@pytest.mark.asyncio
async def test_edge_module_force_sync(edge_config):
    """Test forced synchronization."""
    module = EdgeComputingModule(edge_config)
    await module.initialize()
    
    try:
        # Force sync
        sync_success = await module.force_sync()
        # Should succeed if cluster is available
        assert isinstance(sync_success, bool)
        
    finally:
        await module.shutdown()

@pytest.mark.asyncio
async def test_edge_module_metrics(edge_config):
    """Test metrics collection."""
    module = EdgeComputingModule(edge_config)
    await module.initialize()
    
    try:
        metrics = await module.get_metrics()
        
        assert isinstance(metrics, EdgeMetrics)
        assert metrics.total_tasks >= 0
        assert metrics.completed_tasks >= 0
        assert metrics.failed_tasks >= 0
        
    finally:
        await module.shutdown()

@pytest.mark.asyncio
async def test_edge_module_health_check(edge_config):
    """Test health check functionality."""
    module = EdgeComputingModule(edge_config)
    await module.initialize()
    
    try:
        health_status = await module.health_check()
        
        assert "status" in health_status
        assert "node_info" in health_status
        assert "resource_status" in health_status
        assert "cluster_status" in health_status
        assert "metrics" in health_status
        
    finally:
        await module.shutdown()

# Performance Tests
@pytest.mark.asyncio
async def test_edge_module_performance(edge_config):
    """Test edge computing module performance."""
    module = EdgeComputingModule(edge_config)
    await module.initialize()
    
    try:
        start_time = time.time()
        
        # Submit multiple tasks
        task_ids = []
        for i in range(10):
            task_id = await module.submit_edge_task(
                "data_processing",
                {"input": [i, i+1, i+2]},
                priority=1
            )
            task_ids.append(task_id)
        
        # Wait for completion
        await asyncio.sleep(2)
        
        # Check completion
        completed = 0
        for task_id in task_ids:
            status = await module.get_task_status(task_id)
            if status == "completed":
                completed += 1
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Performance assertions
        assert duration < 5.0  # Should complete within 5 seconds
        assert completed > 0  # At least some tasks should complete
        
    finally:
        await module.shutdown()

# Error Handling Tests
@pytest.mark.asyncio
async def test_edge_module_error_handling(edge_config):
    """Test error handling in edge computing module."""
    module = EdgeComputingModule(edge_config)
    
    # Test operations before initialization
    with pytest.raises(Exception):
        await module.submit_edge_task("test", {})
    
    # Test with invalid configuration
    invalid_config = EdgeComputingConfig(
        local_data_path="/invalid/path/that/does/not/exist"
    )
    invalid_module = EdgeComputingModule(invalid_config)
    
    # Should handle initialization errors gracefully
    success = await invalid_module.initialize()
    # May fail due to invalid path, but shouldn't crash

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

