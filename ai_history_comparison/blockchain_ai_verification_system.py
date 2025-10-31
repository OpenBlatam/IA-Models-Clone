"""
Blockchain AI Verification System
================================

Advanced blockchain-based AI verification system for AI model analysis with
immutable records, smart contracts, and decentralized verification.
"""

import asyncio
import logging
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
import hashlib
import uuid
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing
import queue
import time
import psutil
import gc
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class BlockchainType(str, Enum):
    """Blockchain types"""
    ETHEREUM = "ethereum"
    BITCOIN = "bitcoin"
    HYPERLEDGER = "hyperledger"
    CORDRA = "cordra"
    PRIVATE = "private"
    CONSORTIUM = "consortium"
    PUBLIC = "public"
    HYBRID = "hybrid"


class SmartContractType(str, Enum):
    """Smart contract types"""
    MODEL_VERIFICATION = "model_verification"
    PERFORMANCE_TRACKING = "performance_tracking"
    AUDIT_TRAIL = "audit_trail"
    CONSENSUS_MECHANISM = "consensus_mechanism"
    REWARD_DISTRIBUTION = "reward_distribution"
    GOVERNANCE = "governance"
    COMPLIANCE = "compliance"
    DATA_INTEGRITY = "data_integrity"


class VerificationStatus(str, Enum):
    """Verification status"""
    PENDING = "pending"
    VERIFIED = "verified"
    REJECTED = "rejected"
    DISPUTED = "disputed"
    EXPIRED = "expired"
    INVALID = "invalid"
    CONFIRMED = "confirmed"
    FAILED = "failed"


class ConsensusAlgorithm(str, Enum):
    """Consensus algorithms"""
    PROOF_OF_WORK = "proof_of_work"
    PROOF_OF_STAKE = "proof_of_stake"
    PROOF_OF_AUTHORITY = "proof_of_authority"
    PROOF_OF_ELAPSED_TIME = "proof_of_elapsed_time"
    RAFT = "raft"
    PBFT = "pbft"
    DELEGATED_PROOF_OF_STAKE = "delegated_proof_of_stake"
    PROOF_OF_HISTORY = "proof_of_history"


class TransactionType(str, Enum):
    """Transaction types"""
    MODEL_REGISTRATION = "model_registration"
    PERFORMANCE_UPDATE = "performance_update"
    VERIFICATION_REQUEST = "verification_request"
    AUDIT_LOG = "audit_log"
    CONSENSUS_VOTE = "consensus_vote"
    REWARD_CLAIM = "reward_claim"
    GOVERNANCE_PROPOSAL = "governance_proposal"
    COMPLIANCE_CHECK = "compliance_check"


@dataclass
class BlockchainNode:
    """Blockchain node"""
    node_id: str
    node_type: str
    network_address: str
    public_key: str
    stake_amount: float
    reputation_score: float
    is_validator: bool
    last_activity: datetime
    performance_metrics: Dict[str, float]
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class SmartContract:
    """Smart contract"""
    contract_id: str
    contract_type: SmartContractType
    contract_address: str
    bytecode: str
    abi: Dict[str, Any]
    creator: str
    version: str
    gas_limit: int
    execution_count: int
    last_execution: datetime
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class BlockchainTransaction:
    """Blockchain transaction"""
    transaction_id: str
    transaction_type: TransactionType
    sender: str
    receiver: str
    amount: float
    gas_price: float
    gas_limit: int
    data: Dict[str, Any]
    signature: str
    block_number: int
    status: VerificationStatus
    timestamp: datetime
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class BlockchainBlock:
    """Blockchain block"""
    block_id: str
    block_number: int
    previous_hash: str
    merkle_root: str
    timestamp: datetime
    nonce: int
    difficulty: int
    transactions: List[str]
    validator: str
    block_hash: str
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class AIModelRecord:
    """AI model record on blockchain"""
    record_id: str
    model_id: str
    model_hash: str
    performance_metrics: Dict[str, float]
    verification_status: VerificationStatus
    verifiers: List[str]
    consensus_score: float
    block_number: int
    transaction_hash: str
    timestamp: datetime
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


class BlockchainAIVerificationSystem:
    """Advanced blockchain-based AI verification system"""
    
    def __init__(self, blockchain_type: BlockchainType = BlockchainType.PRIVATE, max_nodes: int = 1000):
        self.blockchain_type = blockchain_type
        self.max_nodes = max_nodes
        
        self.blockchain_nodes: Dict[str, BlockchainNode] = {}
        self.smart_contracts: Dict[str, SmartContract] = {}
        self.transactions: Dict[str, BlockchainTransaction] = {}
        self.blocks: List[BlockchainBlock] = []
        self.ai_model_records: Dict[str, AIModelRecord] = {}
        
        # Blockchain components
        self.consensus_algorithm: ConsensusAlgorithm = ConsensusAlgorithm.PROOF_OF_STAKE
        self.current_block_number: int = 0
        self.difficulty: int = 1
        self.block_time: int = 15  # seconds
        
        # Verification system
        self.verification_threshold: float = 0.67
        self.consensus_required: int = 3
        
        # Initialize blockchain components
        self._initialize_blockchain_components()
        
        # Start blockchain services
        self._start_blockchain_services()
    
    async def register_blockchain_node(self, 
                                     node_id: str,
                                     node_type: str,
                                     network_address: str,
                                     public_key: str,
                                     stake_amount: float = 0.0,
                                     is_validator: bool = False) -> BlockchainNode:
        """Register blockchain node"""
        try:
            node = BlockchainNode(
                node_id=node_id,
                node_type=node_type,
                network_address=network_address,
                public_key=public_key,
                stake_amount=stake_amount,
                reputation_score=0.5,
                is_validator=is_validator,
                last_activity=datetime.now(),
                performance_metrics={}
            )
            
            self.blockchain_nodes[node_id] = node
            
            logger.info(f"Registered blockchain node: {node_id}")
            
            return node
            
        except Exception as e:
            logger.error(f"Error registering blockchain node: {str(e)}")
            raise e
    
    async def deploy_smart_contract(self, 
                                  contract_type: SmartContractType,
                                  creator: str,
                                  bytecode: str,
                                  abi: Dict[str, Any],
                                  gas_limit: int = 1000000) -> SmartContract:
        """Deploy smart contract"""
        try:
            contract_id = hashlib.md5(f"{contract_type}_{creator}_{datetime.now()}_{uuid.uuid4()}".encode()).hexdigest()
            contract_address = f"0x{contract_id[:40]}"
            
            contract = SmartContract(
                contract_id=contract_id,
                contract_type=contract_type,
                contract_address=contract_address,
                bytecode=bytecode,
                abi=abi,
                creator=creator,
                version="1.0.0",
                gas_limit=gas_limit,
                execution_count=0,
                last_execution=datetime.now()
            )
            
            self.smart_contracts[contract_id] = contract
            
            # Create deployment transaction
            await self._create_transaction(
                transaction_type=TransactionType.MODEL_REGISTRATION,
                sender=creator,
                receiver=contract_address,
                amount=0.0,
                data={"contract_deployment": True, "contract_id": contract_id}
            )
            
            logger.info(f"Deployed smart contract: {contract_type.value} ({contract_id})")
            
            return contract
            
        except Exception as e:
            logger.error(f"Error deploying smart contract: {str(e)}")
            raise e
    
    async def register_ai_model(self, 
                              model_id: str,
                              model_data: Dict[str, Any],
                              performance_metrics: Dict[str, float],
                              verifier: str) -> AIModelRecord:
        """Register AI model on blockchain"""
        try:
            # Calculate model hash
            model_hash = hashlib.sha256(json.dumps(model_data, sort_keys=True).encode()).hexdigest()
            
            # Create model record
            record_id = hashlib.md5(f"{model_id}_{model_hash}_{datetime.now()}".encode()).hexdigest()
            
            model_record = AIModelRecord(
                record_id=record_id,
                model_id=model_id,
                model_hash=model_hash,
                performance_metrics=performance_metrics,
                verification_status=VerificationStatus.PENDING,
                verifiers=[verifier],
                consensus_score=0.0,
                block_number=0,
                transaction_hash="",
                timestamp=datetime.now()
            )
            
            self.ai_model_records[record_id] = model_record
            
            # Create registration transaction
            transaction = await self._create_transaction(
                transaction_type=TransactionType.MODEL_REGISTRATION,
                sender=verifier,
                receiver="0x0000000000000000000000000000000000000000",
                amount=0.0,
                data={
                    "model_registration": True,
                    "model_id": model_id,
                    "model_hash": model_hash,
                    "performance_metrics": performance_metrics
                }
            )
            
            model_record.transaction_hash = transaction.transaction_id
            
            # Start verification process
            await self._initiate_verification_process(model_record)
            
            logger.info(f"Registered AI model: {model_id} ({record_id})")
            
            return model_record
            
        except Exception as e:
            logger.error(f"Error registering AI model: {str(e)}")
            raise e
    
    async def verify_ai_model(self, 
                            record_id: str,
                            verifier: str,
                            verification_result: bool,
                            verification_data: Dict[str, Any]) -> bool:
        """Verify AI model"""
        try:
            if record_id not in self.ai_model_records:
                raise ValueError(f"Model record {record_id} not found")
            
            model_record = self.ai_model_records[record_id]
            
            # Add verifier if not already present
            if verifier not in model_record.verifiers:
                model_record.verifiers.append(verifier)
            
            # Create verification transaction
            await self._create_transaction(
                transaction_type=TransactionType.VERIFICATION_REQUEST,
                sender=verifier,
                receiver=record_id,
                amount=0.0,
                data={
                    "verification": True,
                    "record_id": record_id,
                    "verification_result": verification_result,
                    "verification_data": verification_data
                }
            )
            
            # Update consensus score
            await self._update_consensus_score(model_record, verification_result)
            
            # Check if consensus reached
            if await self._check_consensus_reached(model_record):
                model_record.verification_status = VerificationStatus.VERIFIED
                await self._finalize_verification(model_record)
            
            logger.info(f"Verified AI model: {record_id} by {verifier}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error verifying AI model: {str(e)}")
            return False
    
    async def create_block(self, 
                         validator: str,
                         transactions: List[str]) -> BlockchainBlock:
        """Create new blockchain block"""
        try:
            # Get previous block
            previous_hash = self.blocks[-1].block_hash if self.blocks else "0" * 64
            
            # Calculate merkle root
            merkle_root = await self._calculate_merkle_root(transactions)
            
            # Create block
            block_id = hashlib.md5(f"{self.current_block_number}_{previous_hash}_{datetime.now()}".encode()).hexdigest()
            
            block = BlockchainBlock(
                block_id=block_id,
                block_number=self.current_block_number,
                previous_hash=previous_hash,
                merkle_root=merkle_root,
                timestamp=datetime.now(),
                nonce=0,
                difficulty=self.difficulty,
                transactions=transactions,
                validator=validator,
                block_hash=""
            )
            
            # Mine block
            block.block_hash = await self._mine_block(block)
            
            # Add block to chain
            self.blocks.append(block)
            self.current_block_number += 1
            
            # Update model records with block number
            await self._update_model_records_with_block(block)
            
            logger.info(f"Created block: {block.block_number} ({block.block_id})")
            
            return block
            
        except Exception as e:
            logger.error(f"Error creating block: {str(e)}")
            raise e
    
    async def get_blockchain_analytics(self, 
                                     time_range_hours: int = 24) -> Dict[str, Any]:
        """Get blockchain analytics"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=time_range_hours)
            
            # Filter recent data
            recent_blocks = [b for b in self.blocks if b.timestamp >= cutoff_time]
            recent_transactions = [t for t in self.transactions.values() if t.timestamp >= cutoff_time]
            recent_model_records = [r for r in self.ai_model_records.values() if r.timestamp >= cutoff_time]
            
            analytics = {
                "blockchain_info": {
                    "blockchain_type": self.blockchain_type.value,
                    "consensus_algorithm": self.consensus_algorithm.value,
                    "total_blocks": len(self.blocks),
                    "total_transactions": len(self.transactions),
                    "total_nodes": len(self.blockchain_nodes),
                    "total_contracts": len(self.smart_contracts)
                },
                "recent_activity": {
                    "blocks_created": len(recent_blocks),
                    "transactions_processed": len(recent_transactions),
                    "models_registered": len(recent_model_records),
                    "average_block_time": await self._calculate_average_block_time(recent_blocks),
                    "transaction_throughput": len(recent_transactions) / time_range_hours
                },
                "verification_metrics": {
                    "total_models": len(self.ai_model_records),
                    "verified_models": len([r for r in self.ai_model_records.values() if r.verification_status == VerificationStatus.VERIFIED]),
                    "pending_verifications": len([r for r in self.ai_model_records.values() if r.verification_status == VerificationStatus.PENDING]),
                    "average_consensus_score": np.mean([r.consensus_score for r in self.ai_model_records.values()]),
                    "verification_success_rate": len([r for r in self.ai_model_records.values() if r.verification_status == VerificationStatus.VERIFIED]) / len(self.ai_model_records) if self.ai_model_records else 0
                },
                "network_health": {
                    "active_nodes": len([n for n in self.blockchain_nodes.values() if n.last_activity >= cutoff_time]),
                    "validator_nodes": len([n for n in self.blockchain_nodes.values() if n.is_validator]),
                    "average_stake": np.mean([n.stake_amount for n in self.blockchain_nodes.values()]),
                    "average_reputation": np.mean([n.reputation_score for n in self.blockchain_nodes.values()]),
                    "network_decentralization": await self._calculate_decentralization()
                },
                "smart_contract_metrics": {
                    "total_contracts": len(self.smart_contracts),
                    "contract_types": {ct.value: len([c for c in self.smart_contracts.values() if c.contract_type == ct]) for ct in SmartContractType},
                    "total_executions": sum(c.execution_count for c in self.smart_contracts.values()),
                    "average_gas_usage": np.mean([c.gas_limit for c in self.smart_contracts.values()])
                },
                "security_metrics": {
                    "block_validation_rate": await self._calculate_validation_rate(),
                    "consensus_accuracy": await self._calculate_consensus_accuracy(),
                    "attack_resistance": await self._calculate_attack_resistance(),
                    "immutability_score": await self._calculate_immutability_score()
                }
            }
            
            return analytics
            
        except Exception as e:
            logger.error(f"Error getting blockchain analytics: {str(e)}")
            return {"error": str(e)}
    
    # Private helper methods
    def _initialize_blockchain_components(self) -> None:
        """Initialize blockchain components"""
        try:
            # Initialize consensus algorithm based on blockchain type
            if self.blockchain_type == BlockchainType.ETHEREUM:
                self.consensus_algorithm = ConsensusAlgorithm.PROOF_OF_STAKE
            elif self.blockchain_type == BlockchainType.BITCOIN:
                self.consensus_algorithm = ConsensusAlgorithm.PROOF_OF_WORK
            elif self.blockchain_type == BlockchainType.HYPERLEDGER:
                self.consensus_algorithm = ConsensusAlgorithm.PROOF_OF_AUTHORITY
            else:
                self.consensus_algorithm = ConsensusAlgorithm.PROOF_OF_STAKE
            
            # Initialize genesis block
            genesis_block = BlockchainBlock(
                block_id="genesis",
                block_number=0,
                previous_hash="0" * 64,
                merkle_root="0" * 64,
                timestamp=datetime.now(),
                nonce=0,
                difficulty=1,
                transactions=[],
                validator="genesis",
                block_hash="0" * 64
            )
            
            self.blocks.append(genesis_block)
            
            logger.info(f"Initialized blockchain components: {self.blockchain_type.value}, {self.consensus_algorithm.value}")
            
        except Exception as e:
            logger.error(f"Error initializing blockchain components: {str(e)}")
    
    async def _create_transaction(self, 
                                transaction_type: TransactionType,
                                sender: str,
                                receiver: str,
                                amount: float,
                                data: Dict[str, Any]) -> BlockchainTransaction:
        """Create blockchain transaction"""
        try:
            transaction_id = hashlib.md5(f"{transaction_type}_{sender}_{receiver}_{datetime.now()}_{uuid.uuid4()}".encode()).hexdigest()
            
            transaction = BlockchainTransaction(
                transaction_id=transaction_id,
                transaction_type=transaction_type,
                sender=sender,
                receiver=receiver,
                amount=amount,
                gas_price=0.0001,
                gas_limit=21000,
                data=data,
                signature="",
                block_number=0,
                status=VerificationStatus.PENDING,
                timestamp=datetime.now()
            )
            
            self.transactions[transaction_id] = transaction
            
            return transaction
            
        except Exception as e:
            logger.error(f"Error creating transaction: {str(e)}")
            raise e
    
    async def _initiate_verification_process(self, model_record: AIModelRecord) -> None:
        """Initiate verification process for model"""
        try:
            # Select verifiers
            verifiers = await self._select_verifiers()
            
            # Send verification requests
            for verifier in verifiers:
                await self._send_verification_request(model_record, verifier)
            
        except Exception as e:
            logger.error(f"Error initiating verification process: {str(e)}")
    
    async def _select_verifiers(self) -> List[str]:
        """Select verifiers for consensus"""
        try:
            # Select validators with highest stake and reputation
            validators = [
                (node_id, node) for node_id, node in self.blockchain_nodes.items()
                if node.is_validator and node.reputation_score > 0.5
            ]
            
            # Sort by stake amount and reputation
            validators.sort(key=lambda x: x[1].stake_amount * x[1].reputation_score, reverse=True)
            
            # Select top verifiers
            selected_verifiers = [node_id for node_id, _ in validators[:self.consensus_required]]
            
            return selected_verifiers
            
        except Exception as e:
            logger.error(f"Error selecting verifiers: {str(e)}")
            return []
    
    async def _send_verification_request(self, model_record: AIModelRecord, verifier: str) -> None:
        """Send verification request to verifier"""
        try:
            # Simulate verification request
            logger.info(f"Sent verification request for {model_record.record_id} to {verifier}")
            
        except Exception as e:
            logger.error(f"Error sending verification request: {str(e)}")
    
    async def _update_consensus_score(self, model_record: AIModelRecord, verification_result: bool) -> None:
        """Update consensus score for model"""
        try:
            # Calculate consensus score based on verifications
            positive_verifications = sum(1 for _ in model_record.verifiers)  # Simplified
            total_verifications = len(model_record.verifiers)
            
            if total_verifications > 0:
                model_record.consensus_score = positive_verifications / total_verifications
            
        except Exception as e:
            logger.error(f"Error updating consensus score: {str(e)}")
    
    async def _check_consensus_reached(self, model_record: AIModelRecord) -> bool:
        """Check if consensus is reached"""
        try:
            return (
                len(model_record.verifiers) >= self.consensus_required and
                model_record.consensus_score >= self.verification_threshold
            )
            
        except Exception as e:
            logger.error(f"Error checking consensus: {str(e)}")
            return False
    
    async def _finalize_verification(self, model_record: AIModelRecord) -> None:
        """Finalize verification process"""
        try:
            # Create finalization transaction
            await self._create_transaction(
                transaction_type=TransactionType.AUDIT_LOG,
                sender="consensus",
                receiver=model_record.record_id,
                amount=0.0,
                data={
                    "verification_finalized": True,
                    "record_id": model_record.record_id,
                    "consensus_score": model_record.consensus_score,
                    "verifiers": model_record.verifiers
                }
            )
            
            logger.info(f"Finalized verification for {model_record.record_id}")
            
        except Exception as e:
            logger.error(f"Error finalizing verification: {str(e)}")
    
    async def _calculate_merkle_root(self, transactions: List[str]) -> str:
        """Calculate merkle root for transactions"""
        try:
            if not transactions:
                return "0" * 64
            
            # Simple merkle root calculation
            hashes = [hashlib.sha256(tx.encode()).hexdigest() for tx in transactions]
            
            while len(hashes) > 1:
                new_hashes = []
                for i in range(0, len(hashes), 2):
                    if i + 1 < len(hashes):
                        combined = hashes[i] + hashes[i + 1]
                    else:
                        combined = hashes[i] + hashes[i]
                    new_hashes.append(hashlib.sha256(combined.encode()).hexdigest())
                hashes = new_hashes
            
            return hashes[0]
            
        except Exception as e:
            logger.error(f"Error calculating merkle root: {str(e)}")
            return "0" * 64
    
    async def _mine_block(self, block: BlockchainBlock) -> str:
        """Mine block using consensus algorithm"""
        try:
            if self.consensus_algorithm == ConsensusAlgorithm.PROOF_OF_WORK:
                return await self._proof_of_work_mining(block)
            elif self.consensus_algorithm == ConsensusAlgorithm.PROOF_OF_STAKE:
                return await self._proof_of_stake_mining(block)
            else:
                # Simple hash for other algorithms
                block_data = f"{block.block_number}{block.previous_hash}{block.merkle_root}{block.timestamp}{block.nonce}"
                return hashlib.sha256(block_data.encode()).hexdigest()
                
        except Exception as e:
            logger.error(f"Error mining block: {str(e)}")
            return "0" * 64
    
    async def _proof_of_work_mining(self, block: BlockchainBlock) -> str:
        """Proof of work mining"""
        try:
            target = "0" * self.difficulty
            
            while True:
                block_data = f"{block.block_number}{block.previous_hash}{block.merkle_root}{block.timestamp}{block.nonce}"
                block_hash = hashlib.sha256(block_data.encode()).hexdigest()
                
                if block_hash.startswith(target):
                    return block_hash
                
                block.nonce += 1
                
        except Exception as e:
            logger.error(f"Error in proof of work mining: {str(e)}")
            return "0" * 64
    
    async def _proof_of_stake_mining(self, block: BlockchainBlock) -> str:
        """Proof of stake mining"""
        try:
            # Select validator based on stake
            validators = [(node_id, node) for node_id, node in self.blockchain_nodes.items() if node.is_validator]
            
            if not validators:
                return "0" * 64
            
            # Weighted selection based on stake
            total_stake = sum(node.stake_amount for _, node in validators)
            if total_stake == 0:
                return "0" * 64
            
            # Simple block hash
            block_data = f"{block.block_number}{block.previous_hash}{block.merkle_root}{block.timestamp}{block.validator}"
            return hashlib.sha256(block_data.encode()).hexdigest()
            
        except Exception as e:
            logger.error(f"Error in proof of stake mining: {str(e)}")
            return "0" * 64
    
    async def _update_model_records_with_block(self, block: BlockchainBlock) -> None:
        """Update model records with block number"""
        try:
            for transaction_id in block.transactions:
                if transaction_id in self.transactions:
                    transaction = self.transactions[transaction_id]
                    transaction.block_number = block.block_number
                    
                    # Update model records
                    if transaction.transaction_type == TransactionType.MODEL_REGISTRATION:
                        for record in self.ai_model_records.values():
                            if record.transaction_hash == transaction_id:
                                record.block_number = block.block_number
                                break
            
        except Exception as e:
            logger.error(f"Error updating model records with block: {str(e)}")
    
    async def _calculate_average_block_time(self, blocks: List[BlockchainBlock]) -> float:
        """Calculate average block time"""
        try:
            if len(blocks) < 2:
                return 0.0
            
            total_time = (blocks[-1].timestamp - blocks[0].timestamp).total_seconds()
            return total_time / (len(blocks) - 1)
            
        except Exception as e:
            logger.error(f"Error calculating average block time: {str(e)}")
            return 0.0
    
    async def _calculate_decentralization(self) -> float:
        """Calculate network decentralization score"""
        try:
            if not self.blockchain_nodes:
                return 0.0
            
            # Calculate Gini coefficient for stake distribution
            stakes = [node.stake_amount for node in self.blockchain_nodes.values()]
            stakes.sort()
            
            n = len(stakes)
            if n == 0:
                return 0.0
            
            # Gini coefficient calculation
            cumsum = np.cumsum(stakes)
            gini = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n if cumsum[-1] > 0 else 0
            
            # Decentralization score (1 - Gini coefficient)
            return 1 - gini
            
        except Exception as e:
            logger.error(f"Error calculating decentralization: {str(e)}")
            return 0.0
    
    async def _calculate_validation_rate(self) -> float:
        """Calculate block validation rate"""
        try:
            if not self.blocks:
                return 0.0
            
            # Simulate validation rate
            return np.random.uniform(0.95, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating validation rate: {str(e)}")
            return 0.0
    
    async def _calculate_consensus_accuracy(self) -> float:
        """Calculate consensus accuracy"""
        try:
            if not self.ai_model_records:
                return 0.0
            
            verified_models = [r for r in self.ai_model_records.values() if r.verification_status == VerificationStatus.VERIFIED]
            return len(verified_models) / len(self.ai_model_records)
            
        except Exception as e:
            logger.error(f"Error calculating consensus accuracy: {str(e)}")
            return 0.0
    
    async def _calculate_attack_resistance(self) -> float:
        """Calculate attack resistance score"""
        try:
            # Factors: decentralization, stake distribution, validator diversity
            decentralization = await self._calculate_decentralization()
            validator_count = len([n for n in self.blockchain_nodes.values() if n.is_validator])
            max_validators = 100  # Assume max 100 validators
            
            validator_diversity = min(validator_count / max_validators, 1.0)
            
            # Weighted score
            attack_resistance = (decentralization * 0.4 + validator_diversity * 0.6)
            
            return min(attack_resistance, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating attack resistance: {str(e)}")
            return 0.0
    
    async def _calculate_immutability_score(self) -> float:
        """Calculate immutability score"""
        try:
            if not self.blocks:
                return 0.0
            
            # Factors: block depth, consensus strength, network size
            block_depth = len(self.blocks)
            consensus_strength = self.verification_threshold
            network_size = len(self.blockchain_nodes)
            
            # Normalize factors
            depth_score = min(block_depth / 100, 1.0)  # Max score at 100 blocks
            consensus_score = consensus_strength
            network_score = min(network_size / 50, 1.0)  # Max score at 50 nodes
            
            # Weighted immutability score
            immutability = (depth_score * 0.3 + consensus_score * 0.4 + network_score * 0.3)
            
            return min(immutability, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating immutability score: {str(e)}")
            return 0.0
    
    def _start_blockchain_services(self) -> None:
        """Start blockchain services"""
        try:
            # Start block mining service
            asyncio.create_task(self._block_mining_service())
            
            # Start consensus service
            asyncio.create_task(self._consensus_service())
            
            # Start verification service
            asyncio.create_task(self._verification_service())
            
            logger.info("Started blockchain services")
            
        except Exception as e:
            logger.error(f"Error starting blockchain services: {str(e)}")
    
    async def _block_mining_service(self) -> None:
        """Block mining service"""
        try:
            while True:
                await asyncio.sleep(self.block_time)
                
                # Check for pending transactions
                pending_transactions = [
                    tx_id for tx_id, tx in self.transactions.items()
                    if tx.block_number == 0
                ]
                
                if pending_transactions:
                    # Select validator
                    validator = await self._select_validator()
                    
                    # Create block
                    await self.create_block(validator, pending_transactions[:10])  # Limit to 10 transactions per block
                
        except Exception as e:
            logger.error(f"Error in block mining service: {str(e)}")
    
    async def _consensus_service(self) -> None:
        """Consensus service"""
        try:
            while True:
                await asyncio.sleep(30)  # Consensus every 30 seconds
                
                # Update consensus metrics
                # Validate blocks
                # Update node reputations
                
        except Exception as e:
            logger.error(f"Error in consensus service: {str(e)}")
    
    async def _verification_service(self) -> None:
        """Verification service"""
        try:
            while True:
                await asyncio.sleep(60)  # Verification every minute
                
                # Process pending verifications
                # Update verification statuses
                # Clean up expired verifications
                
        except Exception as e:
            logger.error(f"Error in verification service: {str(e)}")
    
    async def _select_validator(self) -> str:
        """Select validator for block creation"""
        try:
            validators = [(node_id, node) for node_id, node in self.blockchain_nodes.items() if node.is_validator]
            
            if not validators:
                return "default_validator"
            
            # Weighted selection based on stake
            total_stake = sum(node.stake_amount for _, node in validators)
            if total_stake == 0:
                return validators[0][0]
            
            # Simple weighted selection
            weights = [node.stake_amount / total_stake for _, node in validators]
            selected_index = np.random.choice(len(validators), p=weights)
            
            return validators[selected_index][0]
            
        except Exception as e:
            logger.error(f"Error selecting validator: {str(e)}")
            return "default_validator"


# Global blockchain system instance
_blockchain_system: Optional[BlockchainAIVerificationSystem] = None


def get_blockchain_system(blockchain_type: BlockchainType = BlockchainType.PRIVATE, max_nodes: int = 1000) -> BlockchainAIVerificationSystem:
    """Get or create global blockchain system instance"""
    global _blockchain_system
    if _blockchain_system is None:
        _blockchain_system = BlockchainAIVerificationSystem(blockchain_type, max_nodes)
    return _blockchain_system


# Example usage
async def main():
    """Example usage of the blockchain AI verification system"""
    blockchain_system = get_blockchain_system()
    
    # Register blockchain nodes
    node1 = await blockchain_system.register_blockchain_node(
        node_id="node_1",
        node_type="validator",
        network_address="192.168.1.100:8080",
        public_key="0x1234567890abcdef",
        stake_amount=1000.0,
        is_validator=True
    )
    print(f"Registered node: {node1.node_id}")
    
    node2 = await blockchain_system.register_blockchain_node(
        node_id="node_2",
        node_type="validator",
        network_address="192.168.1.101:8080",
        public_key="0xabcdef1234567890",
        stake_amount=1500.0,
        is_validator=True
    )
    print(f"Registered node: {node2.node_id}")
    
    # Deploy smart contract
    contract = await blockchain_system.deploy_smart_contract(
        contract_type=SmartContractType.MODEL_VERIFICATION,
        creator="node_1",
        bytecode="0x608060405234801561001057600080fd5b50",
        abi={"name": "verifyModel", "type": "function", "inputs": []},
        gas_limit=1000000
    )
    print(f"Deployed contract: {contract.contract_id}")
    
    # Register AI model
    model_record = await blockchain_system.register_ai_model(
        model_id="model_1",
        model_data={"architecture": "CNN", "layers": 10},
        performance_metrics={"accuracy": 0.95, "precision": 0.93, "recall": 0.91},
        verifier="node_1"
    )
    print(f"Registered model: {model_record.record_id}")
    
    # Verify AI model
    verification_result = await blockchain_system.verify_ai_model(
        record_id=model_record.record_id,
        verifier="node_2",
        verification_result=True,
        verification_data={"validation_accuracy": 0.94, "test_passed": True}
    )
    print(f"Verification result: {verification_result}")
    
    # Get analytics
    analytics = await blockchain_system.get_blockchain_analytics()
    print(f"Blockchain analytics:")
    print(f"  Total blocks: {analytics['blockchain_info']['total_blocks']}")
    print(f"  Total transactions: {analytics['blockchain_info']['total_transactions']}")
    print(f"  Verified models: {analytics['verification_metrics']['verified_models']}")
    print(f"  Network decentralization: {analytics['network_health']['network_decentralization']:.2f}")


if __name__ == "__main__":
    asyncio.run(main())

























