"""
Blockchain Types and Definitions
================================

Type definitions for blockchain integration and smart contracts.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
import uuid
import hashlib

class BlockchainType(Enum):
    """Blockchain network types."""
    ETHEREUM = "ethereum"
    BITCOIN = "bitcoin"
    HYPERLEDGER = "hyperledger"
    POLKADOT = "polkadot"
    CARDANO = "cardano"
    SOLANA = "solana"
    BINANCE_SMART_CHAIN = "bsc"
    POLYGON = "polygon"
    AVALANCHE = "avalanche"
    CUSTOM = "custom"

class TransactionType(Enum):
    """Transaction types."""
    TRANSFER = "transfer"
    SMART_CONTRACT_DEPLOY = "contract_deploy"
    SMART_CONTRACT_CALL = "contract_call"
    AUDIT_LOG = "audit_log"
    DATA_STORAGE = "data_storage"
    IDENTITY_VERIFICATION = "identity_verification"
    ASSET_MINT = "asset_mint"
    ASSET_BURN = "asset_burn"
    VOTING = "voting"
    GOVERNANCE = "governance"

class SmartContractType(Enum):
    """Smart contract types."""
    ERC20_TOKEN = "erc20_token"
    ERC721_NFT = "erc721_nft"
    ERC1155_MULTI_TOKEN = "erc1155_multi_token"
    AUDIT_CONTRACT = "audit_contract"
    IDENTITY_CONTRACT = "identity_contract"
    GOVERNANCE_CONTRACT = "governance_contract"
    ORACLE_CONTRACT = "oracle_contract"
    ESCROW_CONTRACT = "escrow_contract"
    CUSTOM = "custom"

class ConsensusAlgorithm(Enum):
    """Consensus algorithms."""
    PROOF_OF_WORK = "pow"
    PROOF_OF_STAKE = "pos"
    DELEGATED_PROOF_OF_STAKE = "dpos"
    PROOF_OF_AUTHORITY = "poa"
    PROOF_OF_HISTORY = "poh"
    PRACTICAL_BYZANTINE_FAULT_TOLERANCE = "pbft"
    RAFT = "raft"
    TENDERMINT = "tendermint"

@dataclass
class NetworkNode:
    """Blockchain network node."""
    id: str
    address: str
    port: int
    node_type: str = "full"  # full, light, archive
    is_miner: bool = False
    is_validator: bool = False
    stake_amount: float = 0.0
    reputation_score: float = 1.0
    last_seen: datetime = field(default_factory=datetime.now)
    status: str = "online"  # online, offline, syncing
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Transaction:
    """Blockchain transaction."""
    id: str
    hash: str
    from_address: str
    to_address: str
    amount: float
    gas_price: float
    gas_limit: int
    nonce: int
    transaction_type: TransactionType
    data: bytes = b""
    signature: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    block_number: Optional[int] = None
    block_hash: Optional[str] = None
    status: str = "pending"  # pending, confirmed, failed
    confirmations: int = 0
    fee: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __init__(self, from_addr: str, to_addr: str, amount: float, tx_type: TransactionType, **kwargs):
        self.id = str(uuid.uuid4())
        self.from_address = from_addr
        self.to_address = to_addr
        self.amount = amount
        self.transaction_type = tx_type
        self.gas_price = kwargs.get("gas_price", 0.0)
        self.gas_limit = kwargs.get("gas_limit", 21000)
        self.nonce = kwargs.get("nonce", 0)
        self.data = kwargs.get("data", b"")
        self.timestamp = datetime.now()
        self.status = "pending"
        self.confirmations = 0
        self.fee = 0.0
        self.metadata = kwargs.get("metadata", {})
        self.hash = self._calculate_hash()
    
    def _calculate_hash(self) -> str:
        """Calculate transaction hash."""
        data = f"{self.from_address}{self.to_address}{self.amount}{self.nonce}{self.timestamp.isoformat()}"
        return hashlib.sha256(data.encode()).hexdigest()

@dataclass
class Block:
    """Blockchain block."""
    number: int
    hash: str
    previous_hash: str
    timestamp: datetime
    transactions: List[Transaction] = field(default_factory=list)
    merkle_root: str = ""
    nonce: int = 0
    difficulty: float = 0.0
    gas_used: int = 0
    gas_limit: int = 0
    size: int = 0
    miner: str = ""
    validator: str = ""
    consensus_data: Dict[str, Any] = field(default_factory=dict)
    
    def __init__(self, block_number: int, prev_hash: str, **kwargs):
        self.number = block_number
        self.previous_hash = prev_hash
        self.timestamp = datetime.now()
        self.transactions = kwargs.get("transactions", [])
        self.nonce = kwargs.get("nonce", 0)
        self.difficulty = kwargs.get("difficulty", 0.0)
        self.gas_used = kwargs.get("gas_used", 0)
        self.gas_limit = kwargs.get("gas_limit", 0)
        self.size = kwargs.get("size", 0)
        self.miner = kwargs.get("miner", "")
        self.validator = kwargs.get("validator", "")
        self.consensus_data = kwargs.get("consensus_data", {})
        self.hash = self._calculate_hash()
        self.merkle_root = self._calculate_merkle_root()
    
    def _calculate_hash(self) -> str:
        """Calculate block hash."""
        data = f"{self.number}{self.previous_hash}{self.timestamp.isoformat()}{self.nonce}"
        return hashlib.sha256(data.encode()).hexdigest()
    
    def _calculate_merkle_root(self) -> str:
        """Calculate Merkle root of transactions."""
        if not self.transactions:
            return ""
        
        tx_hashes = [tx.hash for tx in self.transactions]
        while len(tx_hashes) > 1:
            next_level = []
            for i in range(0, len(tx_hashes), 2):
                left = tx_hashes[i]
                right = tx_hashes[i + 1] if i + 1 < len(tx_hashes) else left
                combined = hashlib.sha256((left + right).encode()).hexdigest()
                next_level.append(combined)
            tx_hashes = next_level
        
        return tx_hashes[0] if tx_hashes else ""

@dataclass
class SmartContract:
    """Smart contract definition."""
    id: str
    name: str
    address: str
    contract_type: SmartContractType
    bytecode: bytes
    abi: Dict[str, Any] = field(default_factory=dict)
    source_code: str = ""
    compiler_version: str = ""
    deployment_tx: str = ""
    deployment_block: int = 0
    creator: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    version: str = "1.0.0"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_function_signature(self, function_name: str) -> Optional[str]:
        """Get function signature from ABI."""
        if "functions" in self.abi:
            for func in self.abi["functions"]:
                if func.get("name") == function_name:
                    return func.get("signature", "")
        return None
    
    def get_event_signature(self, event_name: str) -> Optional[str]:
        """Get event signature from ABI."""
        if "events" in self.abi:
            for event in self.abi["events"]:
                if event.get("name") == event_name:
                    return event.get("signature", "")
        return None

@dataclass
class ContractTemplate:
    """Smart contract template."""
    id: str
    name: str
    description: str
    contract_type: SmartContractType
    template_code: str
    parameters: List[Dict[str, Any]] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    version: str = "1.0.0"
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    usage_count: int = 0
    rating: float = 0.0

@dataclass
class ContractExecution:
    """Smart contract execution."""
    id: str
    contract_address: str
    function_name: str
    parameters: List[Any] = field(default_factory=list)
    transaction_hash: str = ""
    gas_used: int = 0
    gas_price: float = 0.0
    execution_time: float = 0.0
    success: bool = False
    return_value: Any = None
    error_message: Optional[str] = None
    block_number: Optional[int] = None
    timestamp: datetime = field(default_factory=datetime.now)
    caller: str = ""
    value: float = 0.0

@dataclass
class AuditLog:
    """Immutable audit log entry."""
    id: str
    event_type: str
    entity_type: str
    entity_id: str
    action: str
    actor: str
    timestamp: datetime = field(default_factory=datetime.now)
    data: Dict[str, Any] = field(default_factory=dict)
    hash: str = ""
    previous_hash: str = ""
    block_number: Optional[int] = None
    transaction_hash: Optional[str] = None
    verified: bool = False
    
    def __init__(self, event_type: str, entity_type: str, entity_id: str, action: str, actor: str, **kwargs):
        self.id = str(uuid.uuid4())
        self.event_type = event_type
        self.entity_type = entity_type
        self.entity_id = entity_id
        self.action = action
        self.actor = actor
        self.timestamp = datetime.now()
        self.data = kwargs.get("data", {})
        self.previous_hash = kwargs.get("previous_hash", "")
        self.hash = self._calculate_hash()
        self.verified = False
    
    def _calculate_hash(self) -> str:
        """Calculate audit log hash."""
        data = f"{self.id}{self.event_type}{self.entity_type}{self.entity_id}{self.action}{self.actor}{self.timestamp.isoformat()}{self.previous_hash}"
        return hashlib.sha256(data.encode()).hexdigest()

@dataclass
class BlockchainNetwork:
    """Blockchain network configuration."""
    id: str
    name: str
    blockchain_type: BlockchainType
    network_id: int
    rpc_url: str
    ws_url: str
    chain_id: int
    consensus_algorithm: ConsensusAlgorithm
    block_time: int = 15  # seconds
    gas_limit: int = 8000000
    gas_price: float = 0.0
    nodes: List[NetworkNode] = field(default_factory=list)
    contracts: List[SmartContract] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

@dataclass
class BlockchainMetrics:
    """Blockchain performance metrics."""
    total_transactions: int = 0
    confirmed_transactions: int = 0
    pending_transactions: int = 0
    failed_transactions: int = 0
    average_confirmation_time: float = 0.0
    average_gas_price: float = 0.0
    network_hash_rate: float = 0.0
    difficulty: float = 0.0
    block_height: int = 0
    active_nodes: int = 0
    total_nodes: int = 0
    contract_deployments: int = 0
    contract_calls: int = 0
    audit_logs: int = 0
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class DecentralizedIdentity:
    """Decentralized identity (DID)."""
    did: str
    public_key: str
    private_key: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    credentials: List[Dict[str, Any]] = field(default_factory=list)
    verifiable_credentials: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    revoked: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TokenAsset:
    """Token/asset on blockchain."""
    id: str
    name: str
    symbol: str
    contract_address: str
    token_type: str  # ERC20, ERC721, ERC1155
    total_supply: float = 0.0
    decimals: int = 18
    owner: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    transfers: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class GovernanceProposal:
    """Blockchain governance proposal."""
    id: str
    title: str
    description: str
    proposer: str
    contract_address: str
    function_call: str
    parameters: List[Any] = field(default_factory=list)
    voting_start: datetime = field(default_factory=datetime.now)
    voting_end: datetime = field(default_factory=datetime.now)
    quorum: float = 0.0
    support_threshold: float = 0.0
    votes_for: int = 0
    votes_against: int = 0
    total_votes: int = 0
    executed: bool = False
    execution_tx: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
