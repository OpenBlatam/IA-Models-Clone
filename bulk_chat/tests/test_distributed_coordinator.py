"""
Tests for Distributed Coordinator
==================================
"""

import pytest
from ..core.distributed_coordinator import DistributedCoordinator, ConsensusAlgorithm, NodeRole


@pytest.fixture
def coordinator():
    """Create distributed coordinator for testing."""
    return DistributedCoordinator(node_id="test_node", algorithm=ConsensusAlgorithm.SIMPLE_MAJORITY)


@pytest.mark.asyncio
async def test_register_node(coordinator):
    """Test registering a node."""
    node_id = coordinator.register_node(
        node_id="node_1",
        address="127.0.0.1:8000",
        metadata={"region": "us-east"}
    )
    
    assert node_id == "node_1"
    assert "node_1" in coordinator.nodes


@pytest.mark.asyncio
async def test_propose_value(coordinator):
    """Test proposing a value for consensus."""
    # First register nodes
    coordinator.register_node("node_1", "127.0.0.1:8000")
    coordinator.register_node("node_2", "127.0.0.1:8001")
    coordinator.register_node("test_node", "127.0.0.1:8002")
    
    # Set as leader
    coordinator.leader_id = "test_node"
    coordinator.current_term = 1
    
    proposal_id = await coordinator.propose_value(
        value={"action": "test"},
        metadata={"test": True}
    )
    
    assert proposal_id is not None
    assert proposal_id in coordinator.proposals


@pytest.mark.asyncio
async def test_get_leader(coordinator):
    """Test getting leader information."""
    coordinator.register_node("leader_node", "127.0.0.1:8000")
    coordinator.leader_id = "leader_node"
    
    leader = coordinator.get_leader()
    
    assert leader is not None
    assert leader["node_id"] == "leader_node"


@pytest.mark.asyncio
async def test_get_coordination_status(coordinator):
    """Test getting coordination status."""
    coordinator.register_node("node_1", "127.0.0.1:8000")
    coordinator.register_node("node_2", "127.0.0.1:8001")
    
    status = coordinator.get_coordination_status()
    
    assert status["total_nodes"] == 3  # node_1, node_2, test_node
    assert status["node_id"] == "test_node"


@pytest.mark.asyncio
async def test_get_coordinator_summary(coordinator):
    """Test getting coordinator summary."""
    coordinator.register_node("node_1", "127.0.0.1:8000")
    
    summary = coordinator.get_distributed_coordinator_summary()
    
    assert summary["node_id"] == "test_node"
    assert summary["total_nodes"] >= 2


