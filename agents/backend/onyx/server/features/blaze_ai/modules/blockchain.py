"""
Blaze AI Blockchain Module v7.7.0

Advanced blockchain system for decentralization, smart contracts,
distributed consensus, and secure AI operations.
"""

import asyncio
import hashlib
import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Callable, Set, Tuple
from datetime import datetime, timedelta
import asyncio_mqtt
import websockets
from pathlib import Path

from .base import BaseModule, ModuleConfig, ModuleStatus

logger = logging.getLogger(__name__)

# Enums
class ConsensusAlgorithm(Enum):
    """Blockchain consensus algorithms."""
    PROOF_OF_WORK = "proof_of_work"
    PROOF_OF_STAKE = "proof_of_stake"
    PROOF_OF_AUTHORITY = "proof_of_authority"
    BYZANTINE_FAULT_TOLERANCE = "byzantine_fault_tolerance"
    PRACTICAL_BYZANTINE_FAULT_TOLERANCE = "practical_byzantine_fault_tolerance"

class BlockStatus(Enum):
    """Block validation status."""
    PENDING = "pending"
    VALIDATED = "validated"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"

class TransactionType(Enum):
    """Types of blockchain transactions."""
    AI_MODEL_TRAINING = "ai_model_training"
    DATA_SHARING = "data_sharing"
    COMPUTATION_RENTAL = "computation_rental"
    SMART_CONTRACT_EXECUTION = "smart_contract_execution"
    TOKEN_TRANSFER = "token_transfer"
    GOVERNANCE_VOTE = "governance_vote"

class SmartContractStatus(Enum):
    """Smart contract execution status."""
    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"

# Configuration and Data Classes
@dataclass
class BlockchainConfig(ModuleConfig):
    """Configuration for Blockchain module."""
    
    # Network settings
    network_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    network_name: str = "blaze-ai-network"
    consensus_algorithm: ConsensusAlgorithm = ConsensusAlgorithm.PROOF_OF_STAKE
    
    # Block settings
    block_time: float = 15.0  # seconds
    max_block_size: int = 1024 * 1024  # 1MB
    difficulty_adjustment_interval: int = 2016
    
    # Consensus settings
    min_validators: int = 3
    validator_stake_requirement: float = 1000.0
    consensus_threshold: float = 0.67  # 67%
    
    # Smart contract settings
    max_contract_execution_time: float = 300.0  # 5 minutes
    gas_limit: int = 1000000
    gas_price: float = 0.00000001
    
    # Security settings
    enable_encryption: bool = True
    enable_signature_verification: bool = True
    max_transaction_age: int = 3600  # 1 hour
    
    # Storage settings
    blockchain_data_path: str = "./blockchain_data"
    max_blockchain_size: int = 10 * 1024 * 1024 * 1024  # 10GB

@dataclass
class Block:
    """Blockchain block structure."""
    
    block_hash: str
    previous_hash: str
    timestamp: datetime
    nonce: int
    difficulty: int
    merkle_root: str
    transactions: List[str]  # Transaction IDs
    validator_signature: str
    block_number: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "block_hash": self.block_hash,
            "previous_hash": self.previous_hash,
            "timestamp": self.timestamp.isoformat(),
            "nonce": self.nonce,
            "difficulty": self.difficulty,
            "merkle_root": self.merkle_root,
            "transactions": self.transactions,
            "validator_signature": self.validator_signature,
            "block_number": self.block_number
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Block':
        """Create block from dictionary."""
        return cls(
            block_hash=data["block_hash"],
            previous_hash=data["previous_hash"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            nonce=data["nonce"],
            difficulty=data["difficulty"],
            merkle_root=data["merkle_root"],
            transactions=data["transactions"],
            validator_signature=data["validator_signature"],
            block_number=data["block_number"]
        )

@dataclass
class Transaction:
    """Blockchain transaction structure."""
    
    transaction_id: str
    transaction_type: TransactionType
    sender_address: str
    recipient_address: str
    amount: float
    gas_price: float
    gas_limit: int
    data: Dict[str, Any]
    timestamp: datetime
    signature: str
    status: str = "pending"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "transaction_id": self.transaction_id,
            "transaction_type": self.transaction_type.value,
            "sender_address": self.sender_address,
            "recipient_address": self.recipient_address,
            "amount": self.amount,
            "gas_price": self.gas_price,
            "gas_limit": self.gas_limit,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
            "signature": self.signature,
            "status": self.status
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Transaction':
        """Create transaction from dictionary."""
        return cls(
            transaction_id=data["transaction_id"],
            transaction_type=TransactionType(data["transaction_type"]),
            sender_address=data["sender_address"],
            recipient_address=data["recipient_address"],
            amount=data["amount"],
            gas_price=data["gas_price"],
            gas_limit=data["gas_limit"],
            data=data["data"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            signature=data["signature"],
            status=data["status"]
        )

@dataclass
class SmartContract:
    """Smart contract structure."""
    
    contract_id: str
    contract_name: str
    contract_code: str
    owner_address: str
    gas_limit: int
    gas_price: float
    status: SmartContractStatus
    created_at: datetime
    executed_at: Optional[datetime] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "contract_id": self.contract_id,
            "contract_name": self.contract_name,
            "contract_code": self.contract_code,
            "owner_address": self.owner_address,
            "gas_limit": self.gas_limit,
            "gas_price": self.gas_price,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "executed_at": self.executed_at.isoformat() if self.executed_at else None,
            "result": self.result,
            "error": self.error
        }

@dataclass
class BlockchainMetrics:
    """Blockchain performance metrics."""
    
    # Block metrics
    total_blocks: int = 0
    blocks_per_second: float = 0.0
    average_block_time: float = 0.0
    
    # Transaction metrics
    total_transactions: int = 0
    transactions_per_second: float = 0.0
    pending_transactions: int = 0
    
    # Smart contract metrics
    total_contracts: int = 0
    contracts_executed: int = 0
    contracts_failed: int = 0
    
    # Network metrics
    active_validators: int = 0
    network_hashrate: float = 0.0
    consensus_rounds: int = 0
    
    # Timestamps
    last_updated: datetime = field(default_factory=datetime.now)

# Core Components
class ConsensusEngine:
    """Handles blockchain consensus mechanisms."""
    
    def __init__(self, config: BlockchainConfig):
        self.config = config
        self.validators: Dict[str, float] = {}  # address -> stake
        self.consensus_rounds: int = 0
        self.current_validator: Optional[str] = None
    
    async def select_validator(self) -> Optional[str]:
        """Select next validator based on consensus algorithm."""
        if self.config.consensus_algorithm == ConsensusAlgorithm.PROOF_OF_STAKE:
            return await self._select_pos_validator()
        elif self.config.consensus_algorithm == ConsensusAlgorithm.PROOF_OF_WORK:
            return await self._select_pow_validator()
        else:
            return await self._select_default_validator()
    
    async def _select_pos_validator(self) -> Optional[str]:
        """Select validator using Proof of Stake."""
        if not self.validators:
            return None
        
        # Weighted random selection based on stake
        total_stake = sum(self.validators.values())
        if total_stake == 0:
            return None
        
        # Simple weighted selection (in production, use more sophisticated methods)
        import random
        random.seed(time.time())
        rand_val = random.uniform(0, total_stake)
        
        current_sum = 0
        for address, stake in self.validators.items():
            current_sum += stake
            if rand_val <= current_sum:
                return address
        
        return list(self.validators.keys())[-1]
    
    async def _select_pow_validator(self) -> Optional[str]:
        """Select validator using Proof of Work."""
        # In PoW, any node can mine
        return "miner_node"
    
    async def _select_default_validator(self) -> Optional[str]:
        """Select validator using default method."""
        if self.validators:
            return list(self.validators.keys())[0]
        return None
    
    async def validate_block(self, block: Block, validators: List[str]) -> bool:
        """Validate block through consensus."""
        if len(validators) < self.config.min_validators:
            return False
        
        # Simulate consensus validation
        await asyncio.sleep(0.1)
        
        # Count positive validations
        positive_votes = len(validators)  # Simulate all validators approve
        consensus_ratio = positive_votes / len(validators)
        
        return consensus_ratio >= self.config.consensus_threshold

class TransactionPool:
    """Manages pending transactions."""
    
    def __init__(self, config: BlockchainConfig):
        self.config = config
        self.pending_transactions: Dict[str, Transaction] = {}
        self.transaction_history: List[Transaction] = []
    
    async def add_transaction(self, transaction: Transaction) -> bool:
        """Add transaction to pool."""
        try:
            # Validate transaction
            if not await self._validate_transaction(transaction):
                return False
            
            # Check if transaction is too old
            if (datetime.now() - transaction.timestamp).total_seconds() > self.config.max_transaction_age:
                logger.warning(f"Transaction {transaction.transaction_id} is too old")
                return False
            
            self.pending_transactions[transaction.transaction_id] = transaction
            return True
            
        except Exception as e:
            logger.error(f"Error adding transaction: {e}")
            return False
    
    async def get_transactions_for_block(self, max_size: int) -> List[Transaction]:
        """Get transactions for new block."""
        # Sort by gas price (higher priority) and timestamp
        sorted_transactions = sorted(
            self.pending_transactions.values(),
            key=lambda t: (t.gas_price, t.timestamp),
            reverse=True
        )
        
        selected_transactions = []
        current_size = 0
        
        for transaction in sorted_transactions:
            if current_size + len(str(transaction.data)) <= max_size:
                selected_transactions.append(transaction)
                current_size += len(str(transaction.data))
            else:
                break
        
        return selected_transactions
    
    async def remove_transactions(self, transaction_ids: List[str]):
        """Remove transactions from pool after block creation."""
        for tx_id in transaction_ids:
            if tx_id in self.pending_transactions:
                transaction = self.pending_transactions.pop(tx_id)
                self.transaction_history.append(transaction)
    
    async def _validate_transaction(self, transaction: Transaction) -> bool:
        """Validate transaction structure and signature."""
        # Basic validation
        if not transaction.sender_address or not transaction.recipient_address:
            return False
        
        if transaction.amount < 0:
            return False
        
        if transaction.gas_price < 0 or transaction.gas_limit <= 0:
            return False
        
        # Signature validation (simplified)
        if self.config.enable_signature_verification:
            # In production, implement proper signature verification
            if not transaction.signature:
                return False
        
        return True

class SmartContractEngine:
    """Executes smart contracts."""
    
    def __init__(self, config: BlockchainConfig):
        self.config = config
        self.contracts: Dict[str, SmartContract] = {}
        self.execution_history: List[Dict[str, Any]] = []
    
    async def deploy_contract(self, contract: SmartContract) -> bool:
        """Deploy a new smart contract."""
        try:
            # Validate contract code
            if not await self._validate_contract_code(contract.contract_code):
                return False
            
            # Store contract
            self.contracts[contract.contract_id] = contract
            logger.info(f"Contract {contract.contract_name} deployed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error deploying contract: {e}")
            return False
    
    async def execute_contract(self, contract_id: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a smart contract."""
        try:
            if contract_id not in self.contracts:
                raise ValueError(f"Contract {contract_id} not found")
            
            contract = self.contracts[contract_id]
            contract.status = SmartContractStatus.EXECUTING
            contract.executed_at = datetime.now()
            
            # Execute contract (simplified)
            result = await self._execute_contract_code(contract, input_data)
            
            contract.status = SmartContractStatus.COMPLETED
            contract.result = result
            
            # Record execution
            execution_record = {
                "contract_id": contract_id,
                "input_data": input_data,
                "result": result,
                "execution_time": datetime.now().isoformat(),
                "gas_used": len(str(input_data))  # Simplified gas calculation
            }
            self.execution_history.append(execution_record)
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing contract {contract_id}: {e}")
            if contract_id in self.contracts:
                contract = self.contracts[contract_id]
                contract.status = SmartContractStatus.FAILED
                contract.error = str(e)
            
            return {"error": str(e)}
    
    async def _validate_contract_code(self, code: str) -> bool:
        """Validate smart contract code."""
        # Basic validation (in production, implement proper validation)
        if not code or len(code.strip()) == 0:
            return False
        
        # Check for basic syntax (simplified)
        if "def" not in code and "class" not in code:
            return False
        
        return True
    
    async def _execute_contract_code(self, contract: SmartContract, input_data: Dict[str, Any]) -> Any:
        """Execute smart contract code."""
        try:
            # In production, use a secure sandbox environment
            # This is a simplified execution for demonstration
            
            # Simulate execution time
            await asyncio.sleep(0.1)
            
            # Simple contract logic based on input
            if "operation" in input_data:
                operation = input_data["operation"]
                if operation == "add":
                    return {"result": input_data.get("a", 0) + input_data.get("b", 0)}
                elif operation == "multiply":
                    return {"result": input_data.get("a", 0) * input_data.get("b", 0)}
                elif operation == "ai_inference":
                    return {"result": "AI inference completed", "confidence": 0.95}
            
            return {"result": "Contract executed successfully", "input": input_data}
            
        except Exception as e:
            logger.error(f"Contract execution error: {e}")
            raise

class BlockchainStorage:
    """Manages blockchain data storage."""
    
    def __init__(self, config: BlockchainConfig):
        self.config = config
        self.data_path = Path(config.blockchain_data_path)
        self.data_path.mkdir(parents=True, exist_ok=True)
        
        # In-memory storage for performance
        self.blocks: Dict[str, Block] = {}
        self.transactions: Dict[str, Transaction] = {}
        self.chain_tip: Optional[str] = None
        self.block_height: int = 0
    
    async def add_block(self, block: Block) -> bool:
        """Add block to blockchain."""
        try:
            # Store in memory
            self.blocks[block.block_hash] = block
            self.chain_tip = block.block_hash
            self.block_height = block.block_number
            
            # Persist to disk
            await self._persist_block(block)
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding block: {e}")
            return False
    
    async def get_block(self, block_hash: str) -> Optional[Block]:
        """Get block by hash."""
        return self.blocks.get(block_hash)
    
    async def get_latest_block(self) -> Optional[Block]:
        """Get the latest block in the chain."""
        if self.chain_tip:
            return self.blocks.get(self.chain_tip)
        return None
    
    async def get_block_by_height(self, height: int) -> Optional[Block]:
        """Get block by height."""
        for block in self.blocks.values():
            if block.block_number == height:
                return block
        return None
    
    async def _persist_block(self, block: Block):
        """Persist block to disk."""
        try:
            file_path = self.data_path / f"block_{block.block_number}_{block.block_hash[:8]}.json"
            
            with open(file_path, 'w') as f:
                json.dump(block.to_dict(), f, indent=2)
                
        except Exception as e:
            logger.error(f"Error persisting block: {e}")

# Main Module
class BlockchainModule(BaseModule):
    """Advanced blockchain module for Blaze AI system."""
    
    def __init__(self, config: BlockchainConfig):
        super().__init__(config)
        self.config = config
        
        # Core components
        self.consensus_engine = ConsensusEngine(config)
        self.transaction_pool = TransactionPool(config)
        self.smart_contract_engine = SmartContractEngine(config)
        self.blockchain_storage = BlockchainStorage(config)
        
        # Blockchain state
        self.genesis_block: Optional[Block] = None
        self.current_block: Optional[Block] = None
        self.mining_task: Optional[asyncio.Task] = None
        self.consensus_task: Optional[asyncio.Task] = None
        
        # Metrics
        self.metrics = BlockchainMetrics()
    
    async def initialize(self) -> bool:
        """Initialize the blockchain module."""
        try:
            logger.info("Initializing Blockchain Module")
            
            # Create genesis block if not exists
            if not self.genesis_block:
                await self._create_genesis_block()
            
            # Start background tasks
            self.mining_task = asyncio.create_task(self._mining_loop())
            self.consensus_task = asyncio.create_task(self._consensus_loop())
            
            self.status = ModuleStatus.RUNNING
            logger.info("Blockchain Module initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Blockchain Module: {e}")
            self.status = ModuleStatus.ERROR
            return False
    
    async def shutdown(self) -> bool:
        """Shutdown the blockchain module."""
        try:
            logger.info("Shutting down Blockchain Module")
            
            # Cancel background tasks
            if self.mining_task:
                self.mining_task.cancel()
            if self.consensus_task:
                self.consensus_task.cancel()
            
            self.status = ModuleStatus.STOPPED
            logger.info("Blockchain Module shut down successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            return False
    
    async def submit_transaction(self, transaction_data: Dict[str, Any]) -> Optional[str]:
        """Submit a new transaction to the blockchain."""
        try:
            # Create transaction
            transaction = Transaction(
                transaction_id=str(uuid.uuid4()),
                transaction_type=TransactionType(transaction_data.get("type", "token_transfer")),
                sender_address=transaction_data.get("sender", "unknown"),
                recipient_address=transaction_data.get("recipient", "unknown"),
                amount=transaction_data.get("amount", 0.0),
                gas_price=transaction_data.get("gas_price", self.config.gas_price),
                gas_limit=transaction_data.get("gas_limit", self.config.gas_limit),
                data=transaction_data.get("data", {}),
                timestamp=datetime.now(),
                signature=transaction_data.get("signature", "dummy_signature")
            )
            
            # Add to transaction pool
            success = await self.transaction_pool.add_transaction(transaction)
            if success:
                self.metrics.total_transactions += 1
                return transaction.transaction_id
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error submitting transaction: {e}")
            return None
    
    async def deploy_smart_contract(self, contract_data: Dict[str, Any]) -> Optional[str]:
        """Deploy a new smart contract."""
        try:
            contract = SmartContract(
                contract_id=str(uuid.uuid4()),
                contract_name=contract_data.get("name", "Unnamed Contract"),
                contract_code=contract_data.get("code", ""),
                owner_address=contract_data.get("owner", "unknown"),
                gas_limit=contract_data.get("gas_limit", self.config.gas_limit),
                gas_price=contract_data.get("gas_price", self.config.gas_price),
                status=SmartContractStatus.PENDING,
                created_at=datetime.now()
            )
            
            success = await self.smart_contract_engine.deploy_contract(contract)
            if success:
                self.metrics.total_contracts += 1
                return contract.contract_id
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error deploying smart contract: {e}")
            return None
    
    async def execute_smart_contract(self, contract_id: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a smart contract."""
        try:
            result = await self.smart_contract_engine.execute_contract(contract_id, input_data)
            
            if "error" not in result:
                self.metrics.contracts_executed += 1
            else:
                self.metrics.contracts_failed += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing smart contract: {e}")
            return {"error": str(e)}
    
    async def get_blockchain_status(self) -> Dict[str, Any]:
        """Get current blockchain status."""
        latest_block = await self.blockchain_storage.get_latest_block()
        
        return {
            "block_height": self.blockchain_storage.block_height,
            "latest_block_hash": latest_block.block_hash if latest_block else None,
            "pending_transactions": len(self.transaction_pool.pending_transactions),
            "total_contracts": self.metrics.total_contracts,
            "consensus_algorithm": self.config.consensus_algorithm.value,
            "active_validators": len(self.consensus_engine.validators),
            "network_id": self.config.network_id
        }
    
    async def get_block_info(self, block_hash: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific block."""
        block = await self.blockchain_storage.get_block(block_hash)
        if block:
            return block.to_dict()
        return None
    
    async def get_transaction_info(self, transaction_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific transaction."""
        # Check pending transactions
        if transaction_id in self.transaction_pool.pending_transactions:
            return self.transaction_pool.pending_transactions[transaction_id].to_dict()
        
        # Check transaction history
        for transaction in self.transaction_pool.transaction_history:
            if transaction.transaction_id == transaction_id:
                return transaction.to_dict()
        
        return None
    
    async def _create_genesis_block(self):
        """Create the genesis block."""
        genesis_block = Block(
            block_hash="genesis_hash",
            previous_hash="0000000000000000000000000000000000000000000000000000000000000000",
            timestamp=datetime.now(),
            nonce=0,
            difficulty=1,
            merkle_root="genesis_merkle_root",
            transactions=[],
            validator_signature="genesis_signature",
            block_number=0
        )
        
        self.genesis_block = genesis_block
        await self.blockchain_storage.add_block(genesis_block)
        self.current_block = genesis_block
    
    async def _mining_loop(self):
        """Main mining loop for creating new blocks."""
        while True:
            try:
                # Wait for block time
                await asyncio.sleep(self.config.block_time)
                
                # Create new block
                await self._create_new_block()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Mining loop error: {e}")
                await asyncio.sleep(5)
    
    async def _create_new_block(self):
        """Create a new block."""
        try:
            # Get transactions for new block
            transactions = await self.transaction_pool.get_transactions_for_block(self.config.max_block_size)
            
            if not transactions and self.current_block:
                # No transactions, skip block creation
                return
            
            # Get previous block
            previous_block = self.current_block or self.genesis_block
            if not previous_block:
                return
            
            # Create new block
            new_block = Block(
                block_hash=f"block_{time.time()}",
                previous_hash=previous_block.block_hash,
                timestamp=datetime.now(),
                nonce=0,
                difficulty=previous_block.difficulty,
                merkle_root="merkle_root_placeholder",
                transactions=[tx.transaction_id for tx in transactions],
                validator_signature="validator_signature_placeholder",
                block_number=previous_block.block_number + 1
            )
            
            # Add block to blockchain
            await self.blockchain_storage.add_block(new_block)
            self.current_block = new_block
            
            # Remove transactions from pool
            await self.transaction_pool.remove_transactions([tx.transaction_id for tx in transactions])
            
            # Update metrics
            self.metrics.total_blocks += 1
            self.metrics.blocks_per_second = 1.0 / self.config.block_time
            
            logger.info(f"New block created: {new_block.block_hash[:8]} with {len(transactions)} transactions")
            
        except Exception as e:
            logger.error(f"Error creating new block: {e}")
    
    async def _consensus_loop(self):
        """Consensus mechanism loop."""
        while True:
            try:
                # Wait for consensus round
                await asyncio.sleep(self.config.block_time / 2)
                
                # Run consensus
                await self._run_consensus_round()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Consensus loop error: {e}")
                await asyncio.sleep(5)
    
    async def _run_consensus_round(self):
        """Run a consensus round."""
        try:
            # Select validator
            validator = await self.consensus_engine.select_validator()
            if validator:
                self.consensus_engine.current_validator = validator
                self.consensus_engine.consensus_rounds += 1
                
                # Update metrics
                self.metrics.consensus_rounds += 1
                self.metrics.active_validators = len(self.consensus_engine.validators)
                
        except Exception as e:
            logger.error(f"Error in consensus round: {e}")
    
    async def get_metrics(self) -> BlockchainMetrics:
        """Get current blockchain metrics."""
        # Update pending transactions count
        self.metrics.pending_transactions = len(self.transaction_pool.pending_transactions)
        
        # Update last updated timestamp
        self.metrics.last_updated = datetime.now()
        
        return self.metrics
    
    async def health_check(self) -> Dict[str, Any]:
        """Get module health status."""
        try:
            latest_block = await self.blockchain_storage.get_latest_block()
            
            return {
                "status": self.status.value,
                "blockchain_height": self.blockchain_storage.block_height,
                "latest_block_hash": latest_block.block_hash if latest_block else None,
                "pending_transactions": len(self.transaction_pool.pending_transactions),
                "total_contracts": self.metrics.total_contracts,
                "consensus_algorithm": self.config.consensus_algorithm.value,
                "active_validators": len(self.consensus_engine.validators),
                "mining_active": self.mining_task is not None and not self.mining_task.done(),
                "consensus_active": self.consensus_task is not None and not self.consensus_task.done()
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

# Factory Functions
def create_blockchain_module(config: Optional[BlockchainConfig] = None) -> BlockchainModule:
    """Create a Blockchain module with the given configuration."""
    if config is None:
        config = BlockchainConfig()
    return BlockchainModule(config)

def create_blockchain_module_with_defaults(**kwargs) -> BlockchainModule:
    """Create a Blockchain module with default configuration and custom overrides."""
    config = BlockchainConfig(**kwargs)
    return BlockchainModule(config)

__all__ = [
    # Enums
    "ConsensusAlgorithm", "BlockStatus", "TransactionType", "SmartContractStatus",
    
    # Configuration and Data Classes
    "BlockchainConfig", "Block", "Transaction", "SmartContract", "BlockchainMetrics",
    
    # Core Components
    "ConsensusEngine", "TransactionPool", "SmartContractEngine", "BlockchainStorage",
    
    # Main Module
    "BlockchainModule",
    
    # Factory Functions
    "create_blockchain_module", "create_blockchain_module_with_defaults"
]

