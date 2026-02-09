"""
Tests for Cluster Manager
==========================
"""

import pytest
import asyncio
from ..core.clustering import ClusterManager, Node


@pytest.fixture
def cluster_manager():
    """Create cluster manager for testing."""
    return ClusterManager(node_id="test_node", node_address="127.0.0.1:8000")


@pytest.mark.asyncio
async def test_register_node(cluster_manager):
    """Test registering a node."""
    node_id = cluster_manager.register_node(
        node_id="node_1",
        address="127.0.0.1:8001",
        metadata={"region": "us-east"}
    )
    
    assert node_id == "node_1"
    assert "node_1" in cluster_manager.nodes


@pytest.mark.asyncio
async def test_get_node(cluster_manager):
    """Test getting a node."""
    cluster_manager.register_node("node_1", "127.0.0.1:8001")
    
    node = cluster_manager.get_node("node_1")
    
    assert node is not None
    assert node.node_id == "node_1"
    assert node.address == "127.0.0.1:8001"


@pytest.mark.asyncio
async def test_remove_node(cluster_manager):
    """Test removing a node."""
    cluster_manager.register_node("node_1", "127.0.0.1:8001")
    
    assert "node_1" in cluster_manager.nodes
    
    cluster_manager.remove_node("node_1")
    
    assert "node_1" not in cluster_manager.nodes


@pytest.mark.asyncio
async def test_get_cluster_status(cluster_manager):
    """Test getting cluster status."""
    cluster_manager.register_node("node_1", "127.0.0.1:8001")
    cluster_manager.register_node("node_2", "127.0.0.1:8002")
    
    status = cluster_manager.get_cluster_status()
    
    assert status["total_nodes"] >= 3  # node_1, node_2, test_node
    assert status["node_id"] == "test_node"


@pytest.mark.asyncio
async def test_get_cluster_summary(cluster_manager):
    """Test getting cluster summary."""
    cluster_manager.register_node("node_1", "127.0.0.1:8001")
    
    summary = cluster_manager.get_cluster_summary()
    
    assert summary["node_id"] == "test_node"
    assert summary["total_nodes"] >= 2


