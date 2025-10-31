"""
Tests for Blaze AI Federated Learning Advanced Module

This test suite covers all components of the federated learning module
including secure aggregation, privacy management, and client management.
"""

import pytest
import asyncio
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock

from blaze_ai.modules.federated_learning import (
    FederatedLearningModule,
    FederatedLearningConfig,
    AggregationMethod,
    PrivacyLevel,
    CommunicationProtocol,
    ModelStatus,
    ClientInfo,
    ModelUpdate,
    TrainingRound,
    FederatedMetrics,
    SecureAggregator,
    PrivacyManager,
    AggregationEngine,
    ClientManager,
    create_federated_learning_module,
    create_federated_learning_module_with_defaults
)

# Test fixtures
@pytest.fixture
def config():
    """Create a test configuration."""
    return FederatedLearningConfig(
        name="test_federated_learning",
        max_clients=10,
        min_clients_per_round=2,
        max_clients_per_round=3,
        aggregation_method=AggregationMethod.FEDAVG,
        privacy_level=PrivacyLevel.STANDARD,
        communication_protocol=CommunicationProtocol.HTTP
    )

@pytest.fixture
async def module(config):
    """Create a test module."""
    module = FederatedLearningModule(config)
    await module.initialize()
    yield module
    await module.shutdown()

@pytest.fixture
def sample_model_weights():
    """Create sample model weights for testing."""
    return {
        "layer1.weight": np.random.randn(64, 32) * 0.1,
        "layer1.bias": np.random.randn(64) * 0.1,
        "layer2.weight": np.random.randn(32, 64) * 0.1,
        "layer2.bias": np.random.randn(32) * 0.1
    }

# Test SecureAggregator
class TestSecureAggregator:
    """Test the SecureAggregator class."""
    
    def test_init(self, config):
        """Test SecureAggregator initialization."""
        aggregator = SecureAggregator(config)
        assert aggregator.config == config
        assert aggregator.private_key is not None
        assert aggregator.public_key is not None
    
    @pytest.mark.asyncio
    async def test_generate_shares(self, config):
        """Test share generation."""
        aggregator = SecureAggregator(config)
        value = np.array([[1.0, 2.0], [3.0, 4.0]])
        num_shares = 3
        
        shares = await aggregator.generate_shares(value, num_shares)
        
        assert len(shares) == num_shares
        assert all(share.shape == value.shape for share in shares)
        
        # Verify shares sum to original value
        total = sum(shares)
        np.testing.assert_array_almost_equal(total, value)
    
    @pytest.mark.asyncio
    async def test_aggregate_shares(self, config):
        """Test share aggregation."""
        aggregator = SecureAggregator(config)
        shares = [
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            np.array([[5.0, 6.0], [7.0, 8.0]]),
            np.array([[9.0, 10.0], [11.0, 12.0]])
        ]
        
        result = await aggregator.aggregate_shares(shares)
        expected = np.array([[15.0, 18.0], [21.0, 24.0]])
        
        np.testing.assert_array_equal(result, expected)
    
    @pytest.mark.asyncio
    async def test_add_differential_privacy(self, config):
        """Test differential privacy noise addition."""
        aggregator = SecureAggregator(config)
        value = np.array([[1.0, 2.0], [3.0, 4.0]])
        sensitivity = 1.0
        epsilon = 1.0
        delta = 1e-5
        
        result = await aggregator.add_differential_privacy(value, sensitivity, epsilon, delta)
        
        assert result.shape == value.shape
        # Result should be different due to noise
        assert not np.array_equal(result, value)

# Test PrivacyManager
class TestPrivacyManager:
    """Test the PrivacyManager class."""
    
    def test_init(self, config):
        """Test PrivacyManager initialization."""
        manager = PrivacyManager(config)
        assert manager.config == config
        assert manager.privacy_budget == {}
    
    @pytest.mark.asyncio
    async def test_check_privacy_budget(self, config):
        """Test privacy budget checking."""
        manager = PrivacyManager(config)
        client_id = "test_client"
        
        # Initially no budget consumed
        assert await manager.check_privacy_budget(client_id, 0.5)
        
        # Consume some budget
        await manager.consume_privacy_budget(client_id, 0.3)
        
        # Check remaining budget
        assert await manager.check_privacy_budget(client_id, 0.2)
        assert not await manager.check_privacy_budget(client_id, 0.8)
    
    @pytest.mark.asyncio
    async def test_consume_privacy_budget(self, config):
        """Test privacy budget consumption."""
        manager = PrivacyManager(config)
        client_id = "test_client"
        
        await manager.consume_privacy_budget(client_id, 0.5)
        assert manager.privacy_budget[client_id] == 0.5
        
        await manager.consume_privacy_budget(client_id, 0.3)
        assert manager.privacy_budget[client_id] == 0.8
    
    @pytest.mark.asyncio
    async def test_reset_privacy_budget(self, config):
        """Test privacy budget reset."""
        manager = PrivacyManager(config)
        client_id = "test_client"
        
        await manager.consume_privacy_budget(client_id, 0.5)
        assert manager.privacy_budget[client_id] == 0.5
        
        await manager.reset_privacy_budget(client_id)
        assert manager.privacy_budget[client_id] == 0.0
    
    @pytest.mark.asyncio
    async def test_apply_clipping(self, config):
        """Test gradient clipping."""
        manager = PrivacyManager(config)
        gradients = {
            "layer1.weight": np.array([[1.0, 2.0], [3.0, 4.0]]),
            "layer1.bias": np.array([5.0, 6.0])
        }
        clipping_norm = 2.0
        
        clipped = await manager.apply_clipping(gradients, clipping_norm)
        
        # Check that clipping was applied
        total_norm = np.sqrt(np.sum([np.sum(grad ** 2) for grad in clipped.values()]))
        assert total_norm <= clipping_norm

# Test AggregationEngine
class TestAggregationEngine:
    """Test the AggregationEngine class."""
    
    def test_init(self, config):
        """Test AggregationEngine initialization."""
        engine = AggregationEngine(config)
        assert engine.config == config
        assert engine.secure_aggregator is not None
        assert engine.privacy_manager is not None
    
    @pytest.mark.asyncio
    async def test_aggregate_updates_fedavg(self, config, sample_model_weights):
        """Test FedAvg aggregation."""
        engine = AggregationEngine(config)
        
        updates = [
            ModelUpdate(
                client_id=f"client_{i}",
                round_id="round_1",
                model_weights=sample_model_weights,
                metadata={},
                timestamp=datetime.now()
            )
            for i in range(3)
        ]
        
        result = await engine.aggregate_updates(updates, AggregationMethod.FEDAVG)
        
        assert isinstance(result, dict)
        assert all(key in result for key in sample_model_weights.keys())
        
        # Check that weights are averaged
        for key in result:
            expected = sample_model_weights[key]  # All weights are the same in this test
            np.testing.assert_array_almost_equal(result[key], expected)
    
    @pytest.mark.asyncio
    async def test_aggregate_updates_secure(self, config, sample_model_weights):
        """Test secure aggregation."""
        engine = AggregationEngine(config)
        
        updates = [
            ModelUpdate(
                client_id=f"client_{i}",
                round_id="round_1",
                model_weights=sample_model_weights,
                metadata={},
                timestamp=datetime.now()
            )
            for i in range(2)
        ]
        
        result = await engine.aggregate_updates(updates, AggregationMethod.SECURE_AGGREGATION)
        
        assert isinstance(result, dict)
        assert all(key in result for key in sample_model_weights.keys())
    
    @pytest.mark.asyncio
    async def test_aggregate_updates_differential_privacy(self, config, sample_model_weights):
        """Test differential privacy aggregation."""
        engine = AggregationEngine(config)
        
        updates = [
            ModelUpdate(
                client_id=f"client_{i}",
                round_id="round_1",
                model_weights=sample_model_weights,
                metadata={},
                timestamp=datetime.now()
            )
            for i in range(2)
        ]
        
        result = await engine.aggregate_updates(updates, AggregationMethod.DIFFERENTIAL_PRIVACY)
        
        assert isinstance(result, dict)
        assert all(key in result for key in sample_model_weights.keys())

# Test ClientManager
class TestClientManager:
    """Test the ClientManager class."""
    
    def test_init(self, config):
        """Test ClientManager initialization."""
        manager = ClientManager(config)
        assert manager.config == config
        assert manager.clients == {}
        assert manager.client_rounds == {}
    
    @pytest.mark.asyncio
    async def test_register_client(self, config):
        """Test client registration."""
        manager = ClientManager(config)
        
        client_info = ClientInfo(
            client_id="test_client",
            name="Test Client",
            capabilities=["ml_training"],
            data_size=1000,
            compute_power=1.0,
            network_speed=100,
            last_seen=datetime.now()
        )
        
        client_id = await manager.register_client(client_info)
        assert client_id == "test_client"
        assert "test_client" in manager.clients
    
    @pytest.mark.asyncio
    async def test_unregister_client(self, config):
        """Test client unregistration."""
        manager = ClientManager(config)
        
        client_info = ClientInfo(
            client_id="test_client",
            name="Test Client",
            capabilities=["ml_training"],
            data_size=1000,
            compute_power=1.0,
            network_speed=100,
            last_seen=datetime.now()
        )
        
        await manager.register_client(client_info)
        assert "test_client" in manager.clients
        
        await manager.unregister_client("test_client")
        assert "test_client" not in manager.clients
    
    @pytest.mark.asyncio
    async def test_get_active_clients(self, config):
        """Test getting active clients."""
        manager = ClientManager(config)
        
        # Register active and inactive clients
        active_client = ClientInfo(
            client_id="active_client",
            name="Active Client",
            capabilities=["ml_training"],
            data_size=1000,
            compute_power=1.0,
            network_speed=100,
            last_seen=datetime.now(),
            status="active"
        )
        
        inactive_client = ClientInfo(
            client_id="inactive_client",
            name="Inactive Client",
            capabilities=["ml_training"],
            data_size=1000,
            compute_power=1.0,
            network_speed=100,
            last_seen=datetime.now(),
            status="inactive"
        )
        
        await manager.register_client(active_client)
        await manager.register_client(inactive_client)
        
        active_clients = await manager.get_active_clients()
        assert len(active_clients) == 1
        assert active_clients[0].client_id == "active_client"
    
    @pytest.mark.asyncio
    async def test_select_clients_for_round(self, config):
        """Test client selection for training rounds."""
        manager = ClientManager(config)
        
        # Register multiple clients
        for i in range(5):
            client_info = ClientInfo(
                client_id=f"client_{i}",
                name=f"Client {i}",
                capabilities=["ml_training"],
                data_size=1000,
                compute_power=1.0,
                network_speed=100,
                last_seen=datetime.now()
            )
            await manager.register_client(client_info)
        
        selected = await manager.select_clients_for_round("round_1", 3)
        assert len(selected) == 3
        assert all(client_id in manager.clients for client_id in selected)
        
        # Check that participation is recorded
        for client_id in selected:
            assert "round_1" in manager.client_rounds[client_id]

# Test FederatedLearningModule
class TestFederatedLearningModule:
    """Test the main FederatedLearningModule class."""
    
    @pytest.mark.asyncio
    async def test_initialization(self, config):
        """Test module initialization."""
        module = FederatedLearningModule(config)
        assert module.status == "uninitialized"
        
        await module.initialize()
        assert module.status == "active"
        
        await module.shutdown()
        assert module.status == "shutdown"
    
    @pytest.mark.asyncio
    async def test_register_client(self, module):
        """Test client registration."""
        client_info = {
            "name": "Test Client",
            "capabilities": ["ml_training"],
            "data_size": 1000,
            "compute_power": 1.0,
            "network_speed": 100
        }
        
        client_id = await module.register_client(client_info)
        assert client_id is not None
        assert module.metrics.total_clients == 1
        assert module.metrics.active_clients == 1
    
    @pytest.mark.asyncio
    async def test_start_training_round(self, module):
        """Test starting a training round."""
        # Register some clients first
        for i in range(3):
            client_info = {
                "name": f"Client {i}",
                "capabilities": ["ml_training"],
                "data_size": 1000,
                "compute_power": 1.0,
                "network_speed": 100
            }
            await module.register_client(client_info)
        
        round_config = {"num_clients": 2}
        round_id = await module.start_training_round(round_config)
        
        assert round_id is not None
        assert round_id in module.training_rounds
        assert module.metrics.total_rounds == 1
    
    @pytest.mark.asyncio
    async def test_submit_model_update(self, module, sample_model_weights):
        """Test model update submission."""
        # Register clients and start a round
        for i in range(2):
            client_info = {
                "name": f"Client {i}",
                "capabilities": ["ml_training"],
                "data_size": 1000,
                "compute_power": 1.0,
                "network_speed": 100
            }
            await module.register_client(client_info)
        
        round_config = {"num_clients": 2}
        round_id = await module.start_training_round(round_config)
        
        # Submit updates
        for i in range(2):
            update_data = {
                "client_id": f"Client {i}",
                "model_weights": sample_model_weights,
                "metadata": {"client": i}
            }
            
            client_id = await module.submit_model_update(round_id, update_data)
            assert client_id == f"Client {i}"
        
        # Check that round was completed
        round_status = await module.get_round_status(round_id)
        assert round_status["status"] == "completed"
    
    @pytest.mark.asyncio
    async def test_get_round_status(self, module):
        """Test getting round status."""
        # Register clients and start a round
        for i in range(2):
            client_info = {
                "name": f"Client {i}",
                "capabilities": ["ml_training"],
                "data_size": 1000,
                "compute_power": 1.0,
                "network_speed": 100
            }
            await module.register_client(client_info)
        
        round_config = {"num_clients": 2}
        round_id = await module.start_training_round(round_config)
        
        status = await module.get_round_status(round_id)
        assert status["round_id"] == round_id
        assert status["status"] == "pending"
        assert len(status["clients"]) == 2
    
    @pytest.mark.asyncio
    async def test_get_client_info(self, module):
        """Test getting client information."""
        client_info = {
            "name": "Test Client",
            "capabilities": ["ml_training"],
            "data_size": 1000,
            "compute_power": 1.0,
            "network_speed": 100
        }
        
        client_id = await module.register_client(client_info)
        retrieved_info = await module.get_client_info(client_id)
        
        assert retrieved_info is not None
        assert retrieved_info["name"] == "Test Client"
        assert retrieved_info["capabilities"] == ["ml_training"]
    
    @pytest.mark.asyncio
    async def test_get_metrics(self, module):
        """Test getting system metrics."""
        metrics = await module.get_metrics()
        
        assert isinstance(metrics, FederatedMetrics)
        assert metrics.total_rounds >= 0
        assert metrics.completed_rounds >= 0
        assert metrics.failed_rounds >= 0
    
    @pytest.mark.asyncio
    async def test_health_check(self, module):
        """Test health check."""
        health = await module.health_check()
        
        assert "status" in health
        assert "active_clients" in health
        assert "total_clients" in health
        assert "active_rounds" in health

# Test Factory Functions
class TestFactoryFunctions:
    """Test the factory functions."""
    
    @pytest.mark.asyncio
    async def test_create_federated_learning_module(self, config):
        """Test creating module with config."""
        module = await create_federated_learning_module(config)
        
        assert isinstance(module, FederatedLearningModule)
        assert module.status == "active"
        
        await module.shutdown()
    
    @pytest.mark.asyncio
    async def test_create_federated_learning_module_with_defaults(self):
        """Test creating module with default overrides."""
        module = await create_federated_learning_module_with_defaults(
            max_clients=50,
            privacy_level=PrivacyLevel.HIGH
        )
        
        assert isinstance(module, FederatedLearningModule)
        assert module.config.max_clients == 50
        assert module.config.privacy_level == PrivacyLevel.HIGH
        assert module.status == "active"
        
        await module.shutdown()

# Test Integration
class TestIntegration:
    """Test integration between components."""
    
    @pytest.mark.asyncio
    async def test_full_training_cycle(self, config):
        """Test a complete federated learning training cycle."""
        module = await create_federated_learning_module(config)
        
        try:
            # Register multiple clients
            clients = []
            for i in range(4):
                client_info = {
                    "name": f"Client {i}",
                    "capabilities": ["ml_training"],
                    "data_size=1000 + i * 1000,
                    "compute_power": 1.0 + i * 0.1,
                    "network_speed": 100 + i * 10
                }
                client_id = await module.register_client(client_info)
                clients.append(client_id)
            
            # Start training round
            round_config = {"num_clients": 3}
            round_id = await module.start_training_round(round_config)
            
            # Submit model updates
            for i, client_id in enumerate(clients[:3]):
                model_weights = {
                    "layer1.weight": np.random.randn(32, 16) * 0.1,
                    "layer1.bias": np.random.randn(32) * 0.1
                }
                
                update_data = {
                    "client_id": client_id,
                    "model_weights": model_weights,
                    "metadata": {"client": i}
                }
                
                await module.submit_model_update(round_id, update_data)
            
            # Wait for aggregation
            await asyncio.sleep(1)
            
            # Check results
            round_status = await module.get_round_status(round_id)
            assert round_status["status"] == "completed"
            
            metrics = await module.get_metrics()
            assert metrics.completed_rounds == 1
            
        finally:
            await module.shutdown()

# Test Performance
class TestPerformance:
    """Test performance characteristics."""
    
    @pytest.mark.asyncio
    async def test_large_number_of_clients(self, config):
        """Test performance with many clients."""
        config.max_clients = 1000
        config.min_clients_per_round = 10
        config.max_clients_per_round = 20
        
        module = await create_federated_learning_module(config)
        
        try:
            # Register many clients
            start_time = asyncio.get_event_loop().time()
            
            for i in range(100):
                client_info = {
                    "name": f"Client {i}",
                    "capabilities": ["ml_training"],
                    "data_size": 1000,
                    "compute_power": 1.0,
                    "network_speed": 100
                }
                await module.register_client(client_info)
            
            registration_time = asyncio.get_event_loop().time() - start_time
            assert registration_time < 5.0  # Should complete in under 5 seconds
            
            # Test round creation
            start_time = asyncio.get_event_loop().time()
            round_config = {"num_clients": 15}
            round_id = await module.start_training_round(round_config)
            round_creation_time = asyncio.get_event_loop().time() - start_time
            
            assert round_creation_time < 1.0  # Should complete in under 1 second
            
        finally:
            await module.shutdown()

if __name__ == "__main__":
    pytest.main([__file__])

