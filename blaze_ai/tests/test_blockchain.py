"""
Tests for Blaze AI Blockchain Module

This module provides comprehensive testing for all blockchain functionality
including consensus, smart contracts, transactions, and storage.
"""

import asyncio
import json
import pytest
import tempfile
import shutil
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

from blaze_ai.modules.blockchain import (
    BlockchainModule, BlockchainConfig, ConsensusAlgorithm,
    TransactionType, SmartContractStatus, Block, Transaction,
    SmartContract, BlockchainMetrics, ConsensusEngine,
    TransactionPool, SmartContractEngine, BlockchainStorage
)

# Test Configuration
@pytest.fixture
def blockchain_config():
    """Create a test blockchain configuration."""
    return BlockchainConfig(
        network_name="test-network",
        consensus_algorithm=ConsensusAlgorithm.PROOF_OF_STAKE,
        block_time=1.0,  # Fast for testing
        min_validators=2,
        blockchain_data_path="./test_blockchain_data"
    )

@pytest.fixture
def blockchain_module(blockchain_config):
    """Create a blockchain module for testing."""
    return BlockchainModule(blockchain_config)

@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

# Test Consensus Engine
class TestConsensusEngine:
    """Test consensus engine functionality."""
    
    def test_init(self, blockchain_config):
        """Test consensus engine initialization."""
        engine = ConsensusEngine(blockchain_config)
        assert engine.config == blockchain_config
        assert engine.validators == {}
        assert engine.consensus_rounds == 0
        assert engine.current_validator is None
    
    @pytest.mark.asyncio
    async def test_select_pos_validator(self, blockchain_config):
        """Test Proof of Stake validator selection."""
        engine = ConsensusEngine(blockchain_config)
        engine.validators = {
            "validator1": 100.0,
            "validator2": 200.0,
            "validator3": 300.0
        }
        
        validator = await engine.select_pos_validator()
        assert validator in engine.validators
    
    @pytest.mark.asyncio
    async def test_select_pow_validator(self, blockchain_config):
        """Test Proof of Work validator selection."""
        config = BlockchainConfig(
            consensus_algorithm=ConsensusAlgorithm.PROOF_OF_WORK
        )
        engine = ConsensusEngine(config)
        
        validator = await engine.select_validator()
        assert validator == "miner_node"
    
    @pytest.mark.asyncio
    async def test_validate_block(self, blockchain_config):
        """Test block validation through consensus."""
        engine = ConsensusEngine(blockchain_config)
        validators = ["validator1", "validator2", "validator3"]
        
        # Mock block
        block = Mock()
        
        result = await engine.validate_block(block, validators)
        assert result is True

# Test Transaction Pool
class TestTransactionPool:
    """Test transaction pool functionality."""
    
    def test_init(self, blockchain_config):
        """Test transaction pool initialization."""
        pool = TransactionPool(blockchain_config)
        assert pool.config == blockchain_config
        assert pool.pending_transactions == {}
        assert pool.transaction_history == []
    
    @pytest.mark.asyncio
    async def test_add_valid_transaction(self, blockchain_config):
        """Test adding a valid transaction."""
        pool = TransactionPool(blockchain_config)
        
        transaction = Transaction(
            transaction_id="test_tx_001",
            transaction_type=TransactionType.TOKEN_TRANSFER,
            sender_address="sender_001",
            recipient_address="recipient_001",
            amount=100.0,
            gas_price=0.00000001,
            gas_limit=100000,
            data={"message": "test"},
            timestamp=datetime.now(),
            signature="test_signature"
        )
        
        result = await pool.add_transaction(transaction)
        assert result is True
        assert transaction.transaction_id in pool.pending_transactions
    
    @pytest.mark.asyncio
    async def test_add_invalid_transaction(self, blockchain_config):
        """Test adding an invalid transaction."""
        pool = TransactionPool(blockchain_config)
        
        # Transaction with negative amount
        transaction = Transaction(
            transaction_id="test_tx_002",
            transaction_type=TransactionType.TOKEN_TRANSFER,
            sender_address="sender_001",
            recipient_address="recipient_001",
            amount=-100.0,  # Invalid
            gas_price=0.00000001,
            gas_limit=100000,
            data={"message": "test"},
            timestamp=datetime.now(),
            signature="test_signature"
        )
        
        result = await pool.add_transaction(transaction)
        assert result is False
    
    @pytest.mark.asyncio
    async def test_get_transactions_for_block(self, blockchain_config):
        """Test getting transactions for block creation."""
        pool = TransactionPool(blockchain_config)
        
        # Add multiple transactions
        for i in range(5):
            transaction = Transaction(
                transaction_id=f"test_tx_{i:03d}",
                transaction_type=TransactionType.TOKEN_TRANSFER,
                sender_address=f"sender_{i:03d}",
                recipient_address=f"recipient_{i:03d}",
                amount=10.0 + i,
                gas_price=0.00000001 + (i * 0.00000001),
                gas_limit=100000,
                data={"message": f"test_{i}"},
                timestamp=datetime.now(),
                signature=f"signature_{i}"
            )
            await pool.add_transaction(transaction)
        
        # Get transactions for block
        transactions = await pool.get_transactions_for_block(1000)
        assert len(transactions) > 0
        
        # Should be sorted by gas price (descending)
        gas_prices = [tx.gas_price for tx in transactions]
        assert gas_prices == sorted(gas_prices, reverse=True)

# Test Smart Contract Engine
class TestSmartContractEngine:
    """Test smart contract engine functionality."""
    
    def test_init(self, blockchain_config):
        """Test smart contract engine initialization."""
        engine = SmartContractEngine(blockchain_config)
        assert engine.config == blockchain_config
        assert engine.contracts == {}
        assert engine.execution_history == []
    
    @pytest.mark.asyncio
    async def test_deploy_valid_contract(self, blockchain_config):
        """Test deploying a valid smart contract."""
        engine = SmartContractEngine(blockchain_config)
        
        contract = SmartContract(
            contract_id="test_contract_001",
            contract_name="Test Contract",
            contract_code="def test_function(): return 'hello'",
            owner_address="owner_001",
            gas_limit=1000000,
            gas_price=0.00000001,
            status=SmartContractStatus.PENDING,
            created_at=datetime.now()
        )
        
        result = await engine.deploy_contract(contract)
        assert result is True
        assert contract.contract_id in engine.contracts
    
    @pytest.mark.asyncio
    async def test_deploy_invalid_contract(self, blockchain_config):
        """Test deploying an invalid smart contract."""
        engine = SmartContractEngine(blockchain_config)
        
        contract = SmartContract(
            contract_id="test_contract_002",
            contract_name="Invalid Contract",
            contract_code="",  # Empty code
            owner_address="owner_001",
            gas_limit=1000000,
            gas_price=0.00000001,
            status=SmartContractStatus.PENDING,
            created_at=datetime.now()
        )
        
        result = await engine.deploy_contract(contract)
        assert result is False
    
    @pytest.mark.asyncio
    async def test_execute_contract(self, blockchain_config):
        """Test executing a smart contract."""
        engine = SmartContractEngine(blockchain_config)
        
        # Deploy contract first
        contract = SmartContract(
            contract_id="test_contract_003",
            contract_name="Math Contract",
            contract_code="def add(a, b): return a + b",
            owner_address="owner_001",
            gas_limit=1000000,
            gas_price=0.00000001,
            status=SmartContractStatus.PENDING,
            created_at=datetime.now()
        )
        
        await engine.deploy_contract(contract)
        
        # Execute contract
        result = await engine.execute_contract("test_contract_003", {
            "operation": "add",
            "a": 5,
            "b": 3
        })
        
        assert "error" not in result
        assert result["result"] == 8
    
    @pytest.mark.asyncio
    async def test_execute_nonexistent_contract(self, blockchain_config):
        """Test executing a non-existent contract."""
        engine = SmartContractEngine(blockchain_config)
        
        result = await engine.execute_contract("nonexistent_contract", {})
        assert "error" in result

# Test Blockchain Storage
class TestBlockchainStorage:
    """Test blockchain storage functionality."""
    
    @pytest.mark.asyncio
    async def test_init(self, temp_dir):
        """Test blockchain storage initialization."""
        config = BlockchainConfig(blockchain_data_path=temp_dir)
        storage = BlockchainStorage(config)
        
        assert storage.data_path == Path(temp_dir)
        assert storage.blocks == {}
        assert storage.transactions == {}
        assert storage.chain_tip is None
        assert storage.block_height == 0
    
    @pytest.mark.asyncio
    async def test_add_block(self, temp_dir):
        """Test adding a block to storage."""
        config = BlockchainConfig(blockchain_data_path=temp_dir)
        storage = BlockchainStorage(config)
        
        block = Block(
            block_hash="test_block_001",
            previous_hash="0000000000000000000000000000000000000000000000000000000000000000",
            timestamp=datetime.now(),
            nonce=0,
            difficulty=1,
            merkle_root="test_merkle_root",
            transactions=[],
            validator_signature="test_signature",
            block_number=1
        )
        
        result = await storage.add_block(block)
        assert result is True
        assert block.block_hash in storage.blocks
        assert storage.chain_tip == block.block_hash
        assert storage.block_height == 1
    
    @pytest.mark.asyncio
    async def test_get_block(self, temp_dir):
        """Test retrieving a block from storage."""
        config = BlockchainConfig(blockchain_data_path=temp_dir)
        storage = BlockchainStorage(config)
        
        block = Block(
            block_hash="test_block_002",
            previous_hash="0000000000000000000000000000000000000000000000000000000000000000",
            timestamp=datetime.now(),
            nonce=0,
            difficulty=1,
            merkle_root="test_merkle_root",
            transactions=[],
            validator_signature="test_signature",
            block_number=2
        )
        
        await storage.add_block(block)
        
        retrieved_block = await storage.get_block("test_block_002")
        assert retrieved_block is not None
        assert retrieved_block.block_hash == block.block_hash
    
    @pytest.mark.asyncio
    async def test_get_latest_block(self, temp_dir):
        """Test getting the latest block."""
        config = BlockchainConfig(blockchain_data_path=temp_dir)
        storage = BlockchainStorage(config)
        
        # Add multiple blocks
        for i in range(3):
            block = Block(
                block_hash=f"test_block_{i:03d}",
                previous_hash="0000000000000000000000000000000000000000000000000000000000000000",
                timestamp=datetime.now(),
                nonce=0,
                difficulty=1,
                merkle_root="test_merkle_root",
                transactions=[],
                validator_signature="test_signature",
                block_number=i
            )
            await storage.add_block(block)
        
        latest_block = await storage.get_latest_block()
        assert latest_block is not None
        assert latest_block.block_number == 2

# Test Blockchain Module
class TestBlockchainModule:
    """Test main blockchain module functionality."""
    
    @pytest.mark.asyncio
    async def test_init(self, blockchain_config):
        """Test blockchain module initialization."""
        module = BlockchainModule(blockchain_config)
        
        assert module.config == blockchain_config
        assert module.consensus_engine is not None
        assert module.transaction_pool is not None
        assert module.smart_contract_engine is not None
        assert module.blockchain_storage is not None
        assert module.genesis_block is None
        assert module.current_block is None
        assert module.mining_task is None
        assert module.consensus_task is None
    
    @pytest.mark.asyncio
    async def test_initialize(self, blockchain_module):
        """Test module initialization."""
        result = await blockchain_module.initialize()
        assert result is True
        assert blockchain_module.status.value == "running"
        assert blockchain_module.genesis_block is not None
        assert blockchain_module.mining_task is not None
        assert blockchain_module.consensus_task is not None
    
    @pytest.mark.asyncio
    async def test_shutdown(self, blockchain_module):
        """Test module shutdown."""
        await blockchain_module.initialize()
        result = await blockchain_module.shutdown()
        assert result is True
        assert blockchain_module.status.value == "stopped"
    
    @pytest.mark.asyncio
    async def test_submit_transaction(self, blockchain_module):
        """Test transaction submission."""
        await blockchain_module.initialize()
        
        tx_id = await blockchain_module.submit_transaction({
            "type": "token_transfer",
            "sender": "test_sender",
            "recipient": "test_recipient",
            "amount": 100.0,
            "gas_price": 0.00000001,
            "gas_limit": 100000,
            "data": {"message": "test"}
        })
        
        assert tx_id is not None
        assert blockchain_module.metrics.total_transactions == 1
    
    @pytest.mark.asyncio
    async def test_deploy_smart_contract(self, blockchain_module):
        """Test smart contract deployment."""
        await blockchain_module.initialize()
        
        contract_id = await blockchain_module.deploy_smart_contract({
            "name": "Test Contract",
            "code": "def test(): return 'hello'",
            "owner": "test_owner",
            "gas_limit": 1000000,
            "gas_price": 0.00000001
        })
        
        assert contract_id is not None
        assert blockchain_module.metrics.total_contracts == 1
    
    @pytest.mark.asyncio
    async def test_execute_smart_contract(self, blockchain_module):
        """Test smart contract execution."""
        await blockchain_module.initialize()
        
        # Deploy contract first
        contract_id = await blockchain_module.deploy_smart_contract({
            "name": "Math Contract",
            "code": "def add(a, b): return a + b",
            "owner": "test_owner",
            "gas_limit": 1000000,
            "gas_price": 0.00000001
        })
        
        # Execute contract
        result = await blockchain_module.execute_smart_contract(contract_id, {
            "operation": "add",
            "a": 10,
            "b": 20
        })
        
        assert "error" not in result
        assert result["result"] == 30
    
    @pytest.mark.asyncio
    async def test_get_blockchain_status(self, blockchain_module):
        """Test getting blockchain status."""
        await blockchain_module.initialize()
        
        status = await blockchain_module.get_blockchain_status()
        
        assert "block_height" in status
        assert "latest_block_hash" in status
        assert "pending_transactions" in status
        assert "total_contracts" in status
        assert "consensus_algorithm" in status
        assert "active_validators" in status
        assert "network_id" in status
    
    @pytest.mark.asyncio
    async def test_get_metrics(self, blockchain_module):
        """Test getting blockchain metrics."""
        await blockchain_module.initialize()
        
        metrics = await blockchain_module.get_metrics()
        
        assert isinstance(metrics, BlockchainMetrics)
        assert metrics.total_blocks >= 0
        assert metrics.total_transactions >= 0
        assert metrics.total_contracts >= 0
        assert metrics.consensus_rounds >= 0
    
    @pytest.mark.asyncio
    async def test_health_check(self, blockchain_module):
        """Test health check functionality."""
        await blockchain_module.initialize()
        
        health = await blockchain_module.health_check()
        
        assert "status" in health
        assert "blockchain_height" in health
        assert "latest_block_hash" in health
        assert "pending_transactions" in health
        assert "total_contracts" in health
        assert "consensus_algorithm" in health
        assert "active_validators" in health
        assert "mining_active" in health
        assert "consensus_active" in health

# Test Factory Functions
class TestFactoryFunctions:
    """Test factory functions for creating blockchain modules."""
    
    def test_create_blockchain_module_default(self):
        """Test creating blockchain module with default config."""
        module = create_blockchain_module()
        assert isinstance(module, BlockchainModule)
        assert module.config.network_name == "blaze-ai-network"
        assert module.config.consensus_algorithm == ConsensusAlgorithm.PROOF_OF_STAKE
    
    def test_create_blockchain_module_custom(self):
        """Test creating blockchain module with custom config."""
        module = create_blockchain_module(
            BlockchainConfig(
                network_name="custom-network",
                consensus_algorithm=ConsensusAlgorithm.PROOF_OF_WORK,
                block_time=30.0
            )
        )
        assert isinstance(module, BlockchainModule)
        assert module.config.network_name == "custom-network"
        assert module.config.consensus_algorithm == ConsensusAlgorithm.PROOF_OF_WORK
        assert module.config.block_time == 30.0
    
    def test_create_blockchain_module_with_defaults(self):
        """Test creating blockchain module with default overrides."""
        module = create_blockchain_module_with_defaults(
            network_name="override-network",
            block_time=45.0
        )
        assert isinstance(module, BlockchainModule)
        assert module.config.network_name == "override-network"
        assert module.config.block_time == 45.0
        assert module.config.consensus_algorithm == ConsensusAlgorithm.PROOF_OF_STAKE  # Default

# Integration Tests
class TestIntegration:
    """Integration tests for the complete blockchain system."""
    
    @pytest.mark.asyncio
    async def test_full_blockchain_workflow(self, temp_dir):
        """Test complete blockchain workflow."""
        config = BlockchainConfig(
            network_name="integration-test",
            blockchain_data_path=temp_dir,
            block_time=0.5  # Fast for testing
        )
        
        blockchain = BlockchainModule(config)
        
        try:
            # Initialize
            success = await blockchain.initialize()
            assert success is True
            
            # Wait for genesis block
            await asyncio.sleep(1)
            
            # Submit transactions
            tx_ids = []
            for i in range(3):
                tx_id = await blockchain.submit_transaction({
                    "type": "token_transfer",
                    "sender": f"sender_{i}",
                    "recipient": f"recipient_{i}",
                    "amount": 10.0 + i,
                    "gas_price": 0.00000001,
                    "gas_limit": 100000,
                    "data": {"message": f"test_{i}"}
                })
                tx_ids.append(tx_id)
            
            # Wait for block creation
            await asyncio.sleep(2)
            
            # Check status
            status = await blockchain.get_blockchain_status()
            assert status["block_height"] > 0
            assert status["pending_transactions"] == 0  # All processed
            
            # Check metrics
            metrics = await blockchain.get_metrics()
            assert metrics.total_blocks > 0
            assert metrics.total_transactions >= 3
            
        finally:
            await blockchain.shutdown()

# Performance Tests
class TestPerformance:
    """Performance tests for the blockchain system."""
    
    @pytest.mark.asyncio
    async def test_transaction_throughput(self, temp_dir):
        """Test transaction processing throughput."""
        config = BlockchainConfig(
            network_name="performance-test",
            blockchain_data_path=temp_dir,
            block_time=0.1  # Very fast
        )
        
        blockchain = BlockchainModule(config)
        
        try:
            await blockchain.initialize()
            await asyncio.sleep(0.5)
            
            # Submit many transactions quickly
            start_time = asyncio.get_event_loop().time()
            
            for i in range(100):
                await blockchain.submit_transaction({
                    "type": "token_transfer",
                    "sender": f"sender_{i}",
                    "recipient": f"recipient_{i}",
                    "amount": 1.0,
                    "gas_price": 0.00000001,
                    "gas_limit": 10000,
                    "data": {"index": i}
                })
            
            # Wait for processing
            await asyncio.sleep(2)
            
            end_time = asyncio.get_event_loop().time()
            processing_time = end_time - start_time
            
            # Check results
            status = await blockchain.get_blockchain_status()
            metrics = await blockchain.get_metrics()
            
            assert metrics.total_transactions >= 100
            assert processing_time < 5.0  # Should process quickly
            
        finally:
            await blockchain.shutdown()

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

