"""
Tests for Blaze AI Distributed Processing Module

This test suite covers all distributed processing functionality including
node management, task scheduling, load balancing, fault tolerance, and
cluster management.
"""

import pytest
import asyncio
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

from ..modules.distributed_processing import (
    DistributedProcessingModule, DistributedProcessingConfig, NodeInfo, DistributedTask,
    TaskStatus, TaskPriority, NodeStatus, LoadBalancingStrategy, FaultToleranceStrategy,
    NetworkProtocol, HTTPProtocol, NodeDiscovery, LoadBalancer, TaskScheduler,
    FaultToleranceManager, ClusterMetrics
)

# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def distributed_config():
    """Create a test distributed processing configuration."""
    return DistributedProcessingConfig(
        node_id="test_node_001",
        node_name="Test Node",
        node_capacity=100,
        node_weight=1.0,
        discovery_port=8888,
        communication_port=8889,
        heartbeat_interval=1.0,
        node_timeout=10.0,
        max_task_retries=3,
        task_timeout=60.0,
        batch_size=10,
        enable_checkpointing=True,
        checkpoint_interval=30.0,
        load_balancing_strategy=LoadBalancingStrategy.ADAPTIVE,
        enable_auto_scaling=True,
        min_nodes=1,
        max_nodes=10,
        scaling_threshold=0.8,
        fault_tolerance_strategy=FaultToleranceStrategy.CIRCUIT_BREAKER,
        replication_factor=2,
        circuit_breaker_threshold=3,
        circuit_breaker_timeout=30.0,
        enable_persistence=True,
        backup_interval=1800.0
    )

@pytest.fixture
def distributed_module(distributed_config):
    """Create a test distributed processing module."""
    return DistributedProcessingModule(distributed_config)

@pytest.fixture
def test_node():
    """Create a test node."""
    return NodeInfo(
        node_id="test_node_002",
        node_name="Test Node 2",
        node_address="127.0.0.1",
        node_port=8890,
        node_capacity=150,
        node_weight=1.5,
        node_status=NodeStatus.ONLINE
    )

@pytest.fixture
def test_task():
    """Create a test distributed task."""
    return DistributedTask(
        task_id="test_task_001",
        task_type="test_task",
        task_data={"test": "data"},
        priority=TaskPriority.NORMAL,
        status=TaskStatus.PENDING,
        source_node="test_node_001"
    )

# ============================================================================
# CONFIGURATION TESTS
# ============================================================================

def test_distributed_config_defaults():
    """Test distributed processing configuration default values."""
    config = DistributedProcessingConfig()
    
    assert config.node_id.startswith("node_")
    assert config.node_name == "Blaze AI Node"
    assert config.node_capacity == 100
    assert config.node_weight == 1.0
    assert config.discovery_port == 8888
    assert config.communication_port == 8889
    assert config.heartbeat_interval == 5.0
    assert config.node_timeout == 30.0
    assert config.max_task_retries == 3
    assert config.task_timeout == 300.0
    assert config.load_balancing_strategy == LoadBalancingStrategy.ADAPTIVE
    assert config.fault_tolerance_strategy == FaultToleranceStrategy.REPLICATION

def test_distributed_config_custom():
    """Test distributed processing configuration with custom values."""
    config = DistributedProcessingConfig(
        node_id="custom_node",
        node_capacity=500,
        node_weight=2.0,
        load_balancing_strategy=LoadBalancingStrategy.ROUND_ROBIN,
        fault_tolerance_strategy=FaultToleranceStrategy.CIRCUIT_BREAKER
    )
    
    assert config.node_id == "custom_node"
    assert config.node_capacity == 500
    assert config.node_weight == 2.0
    assert config.load_balancing_strategy == LoadBalancingStrategy.ROUND_ROBIN
    assert config.fault_tolerance_strategy == FaultToleranceStrategy.CIRCUIT_BREAKER

# ============================================================================
# NODE INFO TESTS
# ============================================================================

def test_node_info_creation():
    """Test node info creation."""
    node = NodeInfo(
        node_id="test_001",
        node_name="Test Node",
        node_address="192.168.1.100",
        node_port=8888,
        node_capacity=200,
        node_weight=1.0,
        node_status=NodeStatus.ONLINE
    )
    
    assert node.node_id == "test_001"
    assert node.node_name == "Test Node"
    assert node.node_address == "192.168.1.100"
    assert node.node_port == 8888
    assert node.node_capacity == 200
    assert node.node_weight == 1.0
    assert node.node_status == NodeStatus.ONLINE
    assert node.current_load == 0
    assert node.cpu_usage == 0.0
    assert node.memory_usage == 0.0

def test_node_info_with_metrics():
    """Test node info with performance metrics."""
    node = NodeInfo(
        node_id="test_002",
        node_name="Test Node 2",
        node_address="192.168.1.101",
        node_port=8889,
        node_capacity=300,
        node_weight=1.5,
        node_status=NodeStatus.BUSY,
        current_load=150,
        cpu_usage=75.5,
        memory_usage=60.2
    )
    
    assert node.current_load == 150
    assert node.cpu_usage == 75.5
    assert node.memory_usage == 60.2
    assert node.node_status == NodeStatus.BUSY

# ============================================================================
# DISTRIBUTED TASK TESTS
# ============================================================================

def test_distributed_task_creation():
    """Test distributed task creation."""
    task = DistributedTask(
        task_id="task_001",
        task_type="data_processing",
        task_data={"dataset": "large.csv", "operation": "aggregation"},
        priority=TaskPriority.HIGH,
        status=TaskStatus.PENDING,
        source_node="node_001"
    )
    
    assert task.task_id == "task_001"
    assert task.task_type == "data_processing"
    assert task.task_data["dataset"] == "large.csv"
    assert task.task_data["operation"] == "aggregation"
    assert task.priority == TaskPriority.HIGH
    assert task.status == TaskStatus.PENDING
    assert task.source_node == "node_001"
    assert task.target_node is None
    assert task.retry_count == 0
    assert task.max_retries == 3
    assert task.timeout == 300.0

def test_distributed_task_with_dependencies():
    """Test distributed task with dependencies."""
    task = DistributedTask(
        task_id="task_002",
        task_type="model_training",
        task_data={"model": "transformer", "epochs": 100},
        priority=TaskPriority.CRITICAL,
        status=TaskStatus.PENDING,
        source_node="node_001",
        dependencies=["task_001", "task_003"],
        timeout=600.0
    )
    
    assert len(task.dependencies) == 2
    assert "task_001" in task.dependencies
    assert "task_003" in task.dependencies
    assert task.timeout == 600.0

# ============================================================================
# CLUSTER METRICS TESTS
# ============================================================================

def test_cluster_metrics_creation():
    """Test cluster metrics creation."""
    metrics = ClusterMetrics()
    
    assert metrics.total_nodes == 0
    assert metrics.active_nodes == 0
    assert metrics.total_tasks == 0
    assert metrics.completed_tasks == 0
    assert metrics.failed_tasks == 0
    assert metrics.average_response_time == 0.0
    assert metrics.cluster_load == 0.0
    assert metrics.network_latency == 0.0

def test_cluster_metrics_update():
    """Test cluster metrics update."""
    metrics = ClusterMetrics()
    
    metrics.total_nodes = 5
    metrics.active_nodes = 4
    metrics.total_tasks = 100
    metrics.completed_tasks = 80
    metrics.failed_tasks = 5
    metrics.average_response_time = 150.5
    metrics.cluster_load = 0.75
    metrics.network_latency = 25.3
    
    assert metrics.total_nodes == 5
    assert metrics.active_nodes == 4
    assert metrics.total_tasks == 100
    assert metrics.completed_tasks == 80
    assert metrics.failed_tasks == 5
    assert metrics.average_response_time == 150.5
    assert metrics.cluster_load == 0.75
    assert metrics.network_latency == 25.3

# ============================================================================
# LOAD BALANCER TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_load_balancer_round_robin():
    """Test round-robin load balancing strategy."""
    config = DistributedProcessingConfig(load_balancing_strategy=LoadBalancingStrategy.ROUND_ROBIN)
    balancer = LoadBalancer(config)
    
    nodes = [
        NodeInfo("node1", "Node 1", "127.0.0.1", 8888, 100, 1.0, NodeStatus.ONLINE),
        NodeInfo("node2", "Node 2", "127.0.0.2", 8889, 100, 1.0, NodeStatus.ONLINE),
        NodeInfo("node3", "Node 3", "127.0.0.3", 8890, 100, 1.0, NodeStatus.ONLINE)
    ]
    
    task = DistributedTask("task1", "test", {}, TaskPriority.NORMAL, TaskStatus.PENDING, "source")
    
    # Test round-robin selection
    selected1 = await balancer.select_node(task, nodes)
    selected2 = await balancer.select_node(task, nodes)
    selected3 = await balancer.select_node(task, nodes)
    selected4 = await balancer.select_node(task, nodes)
    
    # Should cycle through nodes
    assert selected1 != selected2
    assert selected2 != selected3
    assert selected3 != selected4
    assert selected4 == selected1  # Back to first

@pytest.mark.asyncio
async def test_load_balancer_least_connections():
    """Test least connections load balancing strategy."""
    config = DistributedProcessingConfig(load_balancing_strategy=LoadBalancingStrategy.LEAST_CONNECTIONS)
    balancer = LoadBalancer(config)
    
    nodes = [
        NodeInfo("node1", "Node 1", "127.0.0.1", 8888, 100, 1.0, NodeStatus.ONLINE, current_load=50),
        NodeInfo("node2", "Node 2", "127.0.0.2", 8889, 100, 1.0, NodeStatus.ONLINE, current_load=20),
        NodeInfo("node3", "Node 3", "127.0.0.3", 8890, 100, 1.0, NodeStatus.ONLINE, current_load=80)
    ]
    
    task = DistributedTask("task1", "test", {}, TaskPriority.NORMAL, TaskStatus.PENDING, "source")
    
    # Should select node with least load
    selected = await balancer.select_node(task, nodes)
    assert selected == "node2"  # Has lowest load (20)

@pytest.mark.asyncio
async def test_load_balancer_adaptive():
    """Test adaptive load balancing strategy."""
    config = DistributedProcessingConfig(load_balancing_strategy=LoadBalancingStrategy.ADAPTIVE)
    balancer = LoadBalancer(config)
    
    nodes = [
        NodeInfo("node1", "Node 1", "127.0.0.1", 8888, 100, 1.0, NodeStatus.ONLINE, current_load=30),
        NodeInfo("node2", "Node 2", "127.0.0.2", 8889, 100, 1.5, NodeStatus.ONLINE, current_load=20),
        NodeInfo("node3", "Node 3", "127.0.0.3", 8890, 100, 1.0, NodeStatus.ONLINE, current_load=80)
    ]
    
    task = DistributedTask("task1", "test", {}, TaskPriority.NORMAL, TaskStatus.PENDING, "source")
    
    # Should select best node based on multiple factors
    selected = await balancer.select_node(task, nodes)
    assert selected is not None
    assert selected in ["node1", "node2", "node3"]

# ============================================================================
# TASK SCHEDULER TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_task_scheduler_submit_task():
    """Test task scheduler task submission."""
    config = DistributedProcessingConfig()
    scheduler = TaskScheduler(config)
    
    task = DistributedTask("task1", "test", {}, TaskPriority.NORMAL, TaskStatus.PENDING, "source")
    
    success = await scheduler.submit_task(task)
    assert success is True
    assert len(scheduler.pending_tasks) == 1
    assert scheduler.pending_tasks[0].task_id == "task1"

@pytest.mark.asyncio
async def test_task_scheduler_get_next_task():
    """Test task scheduler getting next task."""
    config = DistributedProcessingConfig()
    scheduler = TaskScheduler(config)
    
    # Submit multiple tasks with different priorities
    task1 = DistributedTask("task1", "test", {}, TaskPriority.LOW, TaskStatus.PENDING, "source")
    task2 = DistributedTask("task2", "test", {}, TaskPriority.HIGH, TaskStatus.PENDING, "source")
    task3 = DistributedTask("task3", "test", {}, TaskPriority.NORMAL, TaskStatus.PENDING, "source")
    
    await scheduler.submit_task(task1)
    await scheduler.submit_task(task2)
    await scheduler.submit_task(task3)
    
    # Should get highest priority task first
    next_task = await scheduler.get_next_task()
    assert next_task.task_id == "task2"  # HIGH priority
    assert next_task not in scheduler.pending_tasks

@pytest.mark.asyncio
async def test_task_scheduler_task_lifecycle():
    """Test task scheduler complete task lifecycle."""
    config = DistributedProcessingConfig()
    scheduler = TaskScheduler(config)
    
    task = DistributedTask("task1", "test", {}, TaskPriority.NORMAL, TaskStatus.PENDING, "source")
    await scheduler.submit_task(task)
    
    # Start task
    success = await scheduler.start_task(task)
    assert success is True
    assert task.status == TaskStatus.RUNNING
    assert task.task_id in scheduler.running_tasks
    
    # Complete task
    success = await scheduler.complete_task(task.task_id, "result")
    assert success is True
    assert task.status == TaskStatus.COMPLETED
    assert task.result == "result"
    assert task.task_id not in scheduler.running_tasks
    assert task.task_id in scheduler.completed_tasks

@pytest.mark.asyncio
async def test_task_scheduler_task_failure():
    """Test task scheduler handling task failure."""
    config = DistributedProcessingConfig()
    scheduler = TaskScheduler(config)
    
    task = DistributedTask("task1", "test", {}, TaskPriority.NORMAL, TaskStatus.PENDING, "source")
    await scheduler.submit_task(task)
    
    # Start task
    await scheduler.start_task(task)
    
    # Fail task
    success = await scheduler.fail_task(task.task_id, "Test error")
    assert success is True
    assert task.status == TaskStatus.FAILED
    assert task.error == "Test error"
    assert task.task_id not in scheduler.running_tasks
    
    # Should be retried if under max retries
    if task.retry_count < task.max_retries:
        assert task.task_id in scheduler.pending_tasks
        assert task.status == TaskStatus.PENDING

# ============================================================================
# FAULT TOLERANCE TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_fault_tolerance_circuit_breaker():
    """Test circuit breaker fault tolerance strategy."""
    config = DistributedProcessingConfig(
        fault_tolerance_strategy=FaultToleranceStrategy.CIRCUIT_BREAKER,
        circuit_breaker_threshold=3,
        circuit_breaker_timeout=30.0
    )
    fault_tolerance = FaultToleranceManager(config)
    
    # Initially node should be available
    assert await fault_tolerance.is_node_available("node1") is True
    
    # Simulate failures
    for i in range(3):
        await fault_tolerance.handle_node_failure("node1")
    
    # After threshold, node should be unavailable
    assert await fault_tolerance.is_node_available("node1") is False
    
    # Record success should reset circuit breaker
    await fault_tolerance.record_success("node1")
    assert await fault_tolerance.is_node_available("node1") is True

@pytest.mark.asyncio
async def test_fault_tolerance_different_strategies():
    """Test different fault tolerance strategies."""
    strategies = [
        FaultToleranceStrategy.REPLICATION,
        FaultToleranceStrategy.CHECKPOINTING,
        FaultToleranceStrategy.RETRY
    ]
    
    for strategy in strategies:
        config = DistributedProcessingConfig(fault_tolerance_strategy=strategy)
        fault_tolerance = FaultToleranceManager(config)
        
        # Should handle failure without exception
        result = await fault_tolerance.handle_node_failure("node1")
        assert result is True

# ============================================================================
# NODE DISCOVERY TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_node_discovery_register_node():
    """Test node discovery registration."""
    config = DistributedProcessingConfig()
    discovery = NodeDiscovery(config)
    
    node = NodeInfo("node1", "Test Node", "127.0.0.1", 8888, 100, 1.0, NodeStatus.ONLINE)
    
    success = await discovery.register_node(node)
    assert success is True
    assert "node1" in discovery.known_nodes
    assert discovery.known_nodes["node1"].node_name == "Test Node"

@pytest.mark.asyncio
async def test_node_discovery_unregister_node():
    """Test node discovery unregistration."""
    config = DistributedProcessingConfig()
    discovery = NodeDiscovery(config)
    
    node = NodeInfo("node1", "Test Node", "127.0.0.1", 8888, 100, 1.0, NodeStatus.ONLINE)
    await discovery.register_node(node)
    
    success = await discovery.unregister_node("node1")
    assert success is True
    assert "node1" not in discovery.known_nodes

@pytest.mark.asyncio
async def test_node_discovery_update_status():
    """Test node discovery status update."""
    config = DistributedProcessingConfig()
    discovery = NodeDiscovery(config)
    
    node = NodeInfo("node1", "Test Node", "127.0.0.1", 8888, 100, 1.0, NodeStatus.ONLINE)
    await discovery.register_node(node)
    
    success = await discovery.update_node_status("node1", NodeStatus.BUSY, current_load=50)
    assert success is True
    assert discovery.known_nodes["node1"].node_status == NodeStatus.BUSY
    assert discovery.known_nodes["node1"].current_load == 50

@pytest.mark.asyncio
async def test_node_discovery_get_available_nodes():
    """Test node discovery getting available nodes."""
    config = DistributedProcessingConfig()
    discovery = NodeDiscovery(config)
    
    # Register multiple nodes with different statuses
    node1 = NodeInfo("node1", "Node 1", "127.0.0.1", 8888, 100, 1.0, NodeStatus.ONLINE)
    node2 = NodeInfo("node2", "Node 2", "127.0.0.2", 8889, 100, 1.0, NodeStatus.OFFLINE)
    node3 = NodeInfo("node3", "Node 3", "127.0.0.3", 8890, 100, 1.0, NodeStatus.ONLINE)
    
    await discovery.register_node(node1)
    await discovery.register_node(node2)
    await discovery.register_node(node3)
    
    available_nodes = await discovery.get_available_nodes()
    assert len(available_nodes) == 2  # Only ONLINE nodes
    assert all(node.node_status == NodeStatus.ONLINE for node in available_nodes)

# ============================================================================
# DISTRIBUTED PROCESSING MODULE INTEGRATION TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_distributed_module_initialization():
    """Test distributed processing module initialization."""
    config = DistributedProcessingConfig()
    module = DistributedProcessingModule(config)
    
    # Initialize module
    success = await module.initialize()
    assert success is True
    assert module.status.value == "ACTIVE"
    
    # Check that components were initialized
    assert module.local_node is not None
    assert module.local_node.node_id == config.node_id
    
    # Cleanup
    await module.shutdown()

@pytest.mark.asyncio
async def test_distributed_module_task_submission():
    """Test distributed processing module task submission."""
    config = DistributedProcessingConfig()
    module = DistributedProcessingModule(config)
    await module.initialize()
    
    # Submit task
    task_id = await module.submit_distributed_task(
        task_type="test_task",
        task_data={"test": "data"},
        priority=TaskPriority.HIGH
    )
    
    assert task_id is not None
    assert task_id.startswith("task_")
    
    # Check task status
    status = await module.get_task_status(task_id)
    assert status == TaskStatus.PENDING
    
    # Cleanup
    await module.shutdown()

@pytest.mark.asyncio
async def test_distributed_module_cluster_status():
    """Test distributed processing module cluster status."""
    config = DistributedProcessingConfig()
    module = DistributedProcessingModule(config)
    await module.initialize()
    
    # Get cluster status
    cluster_status = await module.get_cluster_status()
    
    assert 'metrics' in cluster_status
    assert 'nodes' in cluster_status
    assert 'tasks' in cluster_status
    
    metrics = cluster_status['metrics']
    assert metrics.total_nodes >= 1  # At least local node
    assert metrics.active_nodes >= 1
    
    # Cleanup
    await module.shutdown()

@pytest.mark.asyncio
async def test_distributed_module_health_check():
    """Test distributed processing module health check."""
    config = DistributedProcessingConfig()
    module = DistributedProcessingModule(config)
    await module.initialize()
    
    # Check health
    health = await module.health_check()
    
    assert 'status' in health
    assert 'total_nodes' in health
    assert 'active_nodes' in health
    assert 'total_tasks' in health
    assert 'cluster_load' in health
    
    # Cleanup
    await module.shutdown()

# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_distributed_module_performance():
    """Test distributed processing module performance under load."""
    config = DistributedProcessingConfig()
    module = DistributedProcessingModule(config)
    await module.initialize()
    
    # Submit multiple tasks
    start_time = time.time()
    
    task_ids = []
    for i in range(100):
        task_id = await module.submit_distributed_task(
            task_type="performance_test",
            task_data={"test_id": i},
            priority=TaskPriority.NORMAL
        )
        if task_id:
            task_ids.append(task_id)
    
    submission_time = time.time() - start_time
    
    # Performance should be reasonable
    assert submission_time < 2.0  # Less than 2 seconds for 100 tasks
    assert len(task_ids) == 100
    
    # Cleanup
    await module.shutdown()

# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_distributed_module_error_handling():
    """Test distributed processing module error handling."""
    config = DistributedProcessingConfig()
    module = DistributedProcessingModule(config)
    await module.initialize()
    
    # Test invalid task submission
    task_id = await module.submit_distributed_task("", {}, TaskPriority.NORMAL)
    assert task_id is None
    
    # Test getting status of non-existent task
    status = await module.get_task_status("non_existent_task")
    assert status is None
    
    # Test getting result of non-existent task
    result = await module.get_task_result("non_existent_task")
    assert result is None
    
    # Test cancelling non-existent task
    success = await module.cancel_task("non_existent_task")
    assert success is False
    
    # Cleanup
    await module.shutdown()

# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

