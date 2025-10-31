#!/usr/bin/env python3
"""
Blockchain Integration System

Advanced blockchain integration with:
- Smart contract integration
- Cryptocurrency payments
- NFT minting and management
- Decentralized storage
- Blockchain analytics
- Cross-chain interoperability
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Union, Callable, Awaitable
import asyncio
import time
import json
import uuid
import hashlib
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import structlog
from collections import defaultdict, deque
import aiohttp
from web3 import Web3
from eth_account import Account
import ipfshttpclient

logger = structlog.get_logger("blockchain_integration")

# =============================================================================
# BLOCKCHAIN MODELS
# =============================================================================

class BlockchainType(Enum):
    """Blockchain types."""
    ETHEREUM = "ethereum"
    POLYGON = "polygon"
    BINANCE_SMART_CHAIN = "binance_smart_chain"
    SOLANA = "solana"
    CARDANO = "cardano"
    POLKADOT = "polkadot"
    AVALANCHE = "avalanche"
    ARBITRUM = "arbitrum"

class TransactionStatus(Enum):
    """Transaction status."""
    PENDING = "pending"
    CONFIRMED = "confirmed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class NFTStandard(Enum):
    """NFT standards."""
    ERC721 = "erc721"
    ERC1155 = "erc1155"
    SPL_TOKEN = "spl_token"
    METADATA = "metadata"

@dataclass
class BlockchainConfig:
    """Blockchain configuration."""
    blockchain_id: str
    name: str
    blockchain_type: BlockchainType
    rpc_url: str
    chain_id: int
    gas_price: int
    gas_limit: int
    private_key: Optional[str]
    contract_address: Optional[str]
    abi: Optional[Dict[str, Any]]
    ipfs_gateway: str
    enabled: bool
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "blockchain_id": self.blockchain_id,
            "name": self.name,
            "blockchain_type": self.blockchain_type.value,
            "rpc_url": self.rpc_url,
            "chain_id": self.chain_id,
            "gas_price": self.gas_price,
            "gas_limit": self.gas_limit,
            "contract_address": self.contract_address,
            "ipfs_gateway": self.ipfs_gateway,
            "enabled": self.enabled
        }

@dataclass
class Transaction:
    """Blockchain transaction."""
    transaction_id: str
    blockchain_id: str
    from_address: str
    to_address: str
    value: int
    gas_price: int
    gas_limit: int
    data: Optional[str]
    nonce: int
    status: TransactionStatus
    hash: Optional[str]
    block_number: Optional[int]
    created_at: datetime
    confirmed_at: Optional[datetime]
    
    def __post_init__(self):
        if not self.transaction_id:
            self.transaction_id = str(uuid.uuid4())
        if not self.created_at:
            self.created_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "transaction_id": self.transaction_id,
            "blockchain_id": self.blockchain_id,
            "from_address": self.from_address,
            "to_address": self.to_address,
            "value": self.value,
            "gas_price": self.gas_price,
            "gas_limit": self.gas_limit,
            "data": self.data,
            "nonce": self.nonce,
            "status": self.status.value,
            "hash": self.hash,
            "block_number": self.block_number,
            "created_at": self.created_at.isoformat(),
            "confirmed_at": self.confirmed_at.isoformat() if self.confirmed_at else None
        }

@dataclass
class NFTMetadata:
    """NFT metadata."""
    nft_id: str
    name: str
    description: str
    image_url: str
    attributes: List[Dict[str, Any]]
    external_url: Optional[str]
    animation_url: Optional[str]
    background_color: Optional[str]
    youtube_url: Optional[str]
    created_at: datetime
    
    def __post_init__(self):
        if not self.nft_id:
            self.nft_id = str(uuid.uuid4())
        if not self.created_at:
            self.created_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "nft_id": self.nft_id,
            "name": self.name,
            "description": self.description,
            "image_url": self.image_url,
            "attributes": self.attributes,
            "external_url": self.external_url,
            "animation_url": self.animation_url,
            "background_color": self.background_color,
            "youtube_url": self.youtube_url,
            "created_at": self.created_at.isoformat()
        }

@dataclass
class SmartContract:
    """Smart contract information."""
    contract_id: str
    name: str
    address: str
    abi: Dict[str, Any]
    blockchain_id: str
    deployed_at: datetime
    version: str
    functions: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "contract_id": self.contract_id,
            "name": self.name,
            "address": self.address,
            "abi": self.abi,
            "blockchain_id": self.blockchain_id,
            "deployed_at": self.deployed_at.isoformat(),
            "version": self.version,
            "functions": self.functions
        }

# =============================================================================
# BLOCKCHAIN CLIENT
# =============================================================================

class BlockchainClient:
    """Blockchain client for interacting with blockchains."""
    
    def __init__(self, config: BlockchainConfig):
        self.config = config
        self.web3: Optional[Web3] = None
        self.account: Optional[Account] = None
        self.ipfs_client: Optional[Any] = None
        self.contract: Optional[Any] = None
        
        # Statistics
        self.stats = {
            'total_transactions': 0,
            'successful_transactions': 0,
            'failed_transactions': 0,
            'total_gas_used': 0,
            'average_gas_price': 0.0,
            'last_block_number': 0
        }
        
        # Transaction tracking
        self.pending_transactions: Dict[str, Transaction] = {}
        self.transaction_history: deque = deque(maxlen=10000)
    
    async def start(self) -> None:
        """Start the blockchain client."""
        try:
            # Initialize Web3
            self.web3 = Web3(Web3.HTTPProvider(self.config.rpc_url))
            
            if not self.web3.is_connected():
                raise ConnectionError(f"Failed to connect to {self.config.name}")
            
            # Initialize account if private key provided
            if self.config.private_key:
                self.account = Account.from_key(self.config.private_key)
            
            # Initialize IPFS client
            self.ipfs_client = ipfshttpclient.connect(self.config.ipfs_gateway)
            
            # Initialize smart contract if address provided
            if self.config.contract_address and self.config.abi:
                self.contract = self.web3.eth.contract(
                    address=self.config.contract_address,
                    abi=self.config.abi
                )
            
            logger.info(
                "Blockchain client started",
                blockchain_id=self.config.blockchain_id,
                name=self.config.name,
                chain_id=self.config.chain_id
            )
        
        except Exception as e:
            logger.error("Failed to start blockchain client", error=str(e))
            raise
    
    async def stop(self) -> None:
        """Stop the blockchain client."""
        if self.ipfs_client:
            self.ipfs_client.close()
        
        logger.info("Blockchain client stopped", blockchain_id=self.config.blockchain_id)
    
    async def get_balance(self, address: str) -> int:
        """Get balance of an address."""
        if not self.web3:
            raise RuntimeError("Blockchain client not initialized")
        
        balance = self.web3.eth.get_balance(address)
        return balance
    
    async def get_transaction_count(self, address: str) -> int:
        """Get transaction count (nonce) for an address."""
        if not self.web3:
            raise RuntimeError("Blockchain client not initialized")
        
        nonce = self.web3.eth.get_transaction_count(address)
        return nonce
    
    async def send_transaction(self, to_address: str, value: int, data: Optional[str] = None) -> Transaction:
        """Send a transaction."""
        if not self.web3 or not self.account:
            raise RuntimeError("Blockchain client not initialized")
        
        # Get nonce
        nonce = await self.get_transaction_count(self.account.address)
        
        # Create transaction
        transaction = Transaction(
            transaction_id=str(uuid.uuid4()),
            blockchain_id=self.config.blockchain_id,
            from_address=self.account.address,
            to_address=to_address,
            value=value,
            gas_price=self.config.gas_price,
            gas_limit=self.config.gas_limit,
            data=data,
            nonce=nonce,
            status=TransactionStatus.PENDING
        )
        
        # Build transaction
        tx_dict = {
            'nonce': nonce,
            'gasPrice': self.config.gas_price,
            'gas': self.config.gas_limit,
            'to': to_address,
            'value': value,
            'data': data or b''
        }
        
        # Sign transaction
        signed_tx = self.web3.eth.account.sign_transaction(tx_dict, self.config.private_key)
        
        # Send transaction
        tx_hash = self.web3.eth.send_raw_transaction(signed_tx.rawTransaction)
        transaction.hash = tx_hash.hex()
        
        # Track transaction
        self.pending_transactions[transaction.transaction_id] = transaction
        self.transaction_history.append(transaction)
        
        # Update statistics
        self.stats['total_transactions'] += 1
        
        logger.info(
            "Transaction sent",
            transaction_id=transaction.transaction_id,
            hash=transaction.hash,
            from_address=transaction.from_address,
            to_address=transaction.to_address,
            value=transaction.value
        )
        
        return transaction
    
    async def call_contract_function(self, function_name: str, *args, **kwargs) -> Any:
        """Call a smart contract function."""
        if not self.contract:
            raise RuntimeError("Smart contract not initialized")
        
        try:
            function = getattr(self.contract.functions, function_name)
            result = function(*args, **kwargs).call()
            return result
        
        except Exception as e:
            logger.error("Contract function call failed", function=function_name, error=str(e))
            raise
    
    async def execute_contract_function(self, function_name: str, *args, **kwargs) -> Transaction:
        """Execute a smart contract function (write operation)."""
        if not self.contract or not self.account:
            raise RuntimeError("Smart contract or account not initialized")
        
        try:
            # Build transaction
            function = getattr(self.contract.functions, function_name)
            tx = function(*args, **kwargs).build_transaction({
                'from': self.account.address,
                'gas': self.config.gas_limit,
                'gasPrice': self.config.gas_price,
                'nonce': await self.get_transaction_count(self.account.address)
            })
            
            # Sign and send transaction
            signed_tx = self.web3.eth.account.sign_transaction(tx, self.config.private_key)
            tx_hash = self.web3.eth.send_raw_transaction(signed_tx.rawTransaction)
            
            # Create transaction record
            transaction = Transaction(
                transaction_id=str(uuid.uuid4()),
                blockchain_id=self.config.blockchain_id,
                from_address=self.account.address,
                to_address=self.config.contract_address,
                value=0,
                gas_price=self.config.gas_price,
                gas_limit=self.config.gas_limit,
                data=tx['data'].hex(),
                nonce=tx['nonce'],
                status=TransactionStatus.PENDING,
                hash=tx_hash.hex()
            )
            
            # Track transaction
            self.pending_transactions[transaction.transaction_id] = transaction
            self.transaction_history.append(transaction)
            
            # Update statistics
            self.stats['total_transactions'] += 1
            
            logger.info(
                "Contract function executed",
                transaction_id=transaction.transaction_id,
                function=function_name,
                hash=transaction.hash
            )
            
            return transaction
        
        except Exception as e:
            logger.error("Contract function execution failed", function=function_name, error=str(e))
            raise
    
    async def upload_to_ipfs(self, data: bytes, filename: Optional[str] = None) -> str:
        """Upload data to IPFS."""
        if not self.ipfs_client:
            raise RuntimeError("IPFS client not initialized")
        
        try:
            result = self.ipfs_client.add_bytes(data)
            ipfs_hash = result['Hash']
            
            logger.info("Data uploaded to IPFS", hash=ipfs_hash, filename=filename)
            return ipfs_hash
        
        except Exception as e:
            logger.error("IPFS upload failed", error=str(e))
            raise
    
    async def upload_json_to_ipfs(self, data: Dict[str, Any]) -> str:
        """Upload JSON data to IPFS."""
        json_data = json.dumps(data, indent=2).encode('utf-8')
        return await self.upload_to_ipfs(json_data, "metadata.json")
    
    async def get_from_ipfs(self, ipfs_hash: str) -> bytes:
        """Get data from IPFS."""
        if not self.ipfs_client:
            raise RuntimeError("IPFS client not initialized")
        
        try:
            data = self.ipfs_client.cat(ipfs_hash)
            return data
        
        except Exception as e:
            logger.error("IPFS retrieval failed", hash=ipfs_hash, error=str(e))
            raise
    
    async def get_json_from_ipfs(self, ipfs_hash: str) -> Dict[str, Any]:
        """Get JSON data from IPFS."""
        data = await self.get_from_ipfs(ipfs_hash)
        return json.loads(data.decode('utf-8'))
    
    async def check_transaction_status(self, transaction_id: str) -> TransactionStatus:
        """Check transaction status."""
        transaction = self.pending_transactions.get(transaction_id)
        if not transaction or not transaction.hash:
            return TransactionStatus.FAILED
        
        try:
            # Get transaction receipt
            receipt = self.web3.eth.get_transaction_receipt(transaction.hash)
            
            if receipt.status == 1:
                transaction.status = TransactionStatus.CONFIRMED
                transaction.block_number = receipt.blockNumber
                transaction.confirmed_at = datetime.utcnow()
                
                # Update statistics
                self.stats['successful_transactions'] += 1
                self.stats['total_gas_used'] += receipt.gasUsed
                
                # Remove from pending
                del self.pending_transactions[transaction_id]
                
                logger.info(
                    "Transaction confirmed",
                    transaction_id=transaction_id,
                    block_number=transaction.block_number,
                    gas_used=receipt.gasUsed
                )
            
            else:
                transaction.status = TransactionStatus.FAILED
                self.stats['failed_transactions'] += 1
                del self.pending_transactions[transaction_id]
                
                logger.warning("Transaction failed", transaction_id=transaction_id)
            
            return transaction.status
        
        except Exception as e:
            logger.error("Failed to check transaction status", transaction_id=transaction_id, error=str(e))
            return TransactionStatus.PENDING
    
    async def get_latest_block_number(self) -> int:
        """Get latest block number."""
        if not self.web3:
            raise RuntimeError("Blockchain client not initialized")
        
        block_number = self.web3.eth.block_number
        self.stats['last_block_number'] = block_number
        return block_number
    
    def get_client_stats(self) -> Dict[str, Any]:
        """Get client statistics."""
        return {
            **self.stats,
            'pending_transactions': len(self.pending_transactions),
            'transaction_history_size': len(self.transaction_history),
            'blockchain_id': self.config.blockchain_id,
            'chain_id': self.config.chain_id
        }

# =============================================================================
# NFT MANAGER
# =============================================================================

class NFTManager:
    """NFT management system."""
    
    def __init__(self, blockchain_client: BlockchainClient):
        self.blockchain_client = blockchain_client
        self.nft_metadata: Dict[str, NFTMetadata] = {}
        self.nft_ownership: Dict[str, str] = {}  # nft_id -> owner_address
        
        # Statistics
        self.stats = {
            'total_nfts': 0,
            'minted_nfts': 0,
            'transferred_nfts': 0,
            'burned_nfts': 0
        }
    
    async def mint_nft(self, metadata: NFTMetadata, owner_address: str) -> str:
        """Mint a new NFT."""
        try:
            # Upload metadata to IPFS
            metadata_dict = metadata.to_dict()
            ipfs_hash = await self.blockchain_client.upload_json_to_ipfs(metadata_dict)
            
            # Mint NFT on blockchain
            transaction = await self.blockchain_client.execute_contract_function(
                'mint',
                owner_address,
                ipfs_hash
            )
            
            # Store NFT metadata
            self.nft_metadata[metadata.nft_id] = metadata
            self.nft_ownership[metadata.nft_id] = owner_address
            
            # Update statistics
            self.stats['total_nfts'] += 1
            self.stats['minted_nfts'] += 1
            
            logger.info(
                "NFT minted",
                nft_id=metadata.nft_id,
                owner=owner_address,
                transaction_id=transaction.transaction_id,
                ipfs_hash=ipfs_hash
            )
            
            return metadata.nft_id
        
        except Exception as e:
            logger.error("NFT minting failed", nft_id=metadata.nft_id, error=str(e))
            raise
    
    async def transfer_nft(self, nft_id: str, from_address: str, to_address: str) -> str:
        """Transfer NFT ownership."""
        try:
            # Check ownership
            if self.nft_ownership.get(nft_id) != from_address:
                raise ValueError(f"Address {from_address} does not own NFT {nft_id}")
            
            # Transfer on blockchain
            transaction = await self.blockchain_client.execute_contract_function(
                'transferFrom',
                from_address,
                to_address,
                nft_id
            )
            
            # Update ownership
            self.nft_ownership[nft_id] = to_address
            
            # Update statistics
            self.stats['transferred_nfts'] += 1
            
            logger.info(
                "NFT transferred",
                nft_id=nft_id,
                from_address=from_address,
                to_address=to_address,
                transaction_id=transaction.transaction_id
            )
            
            return transaction.transaction_id
        
        except Exception as e:
            logger.error("NFT transfer failed", nft_id=nft_id, error=str(e))
            raise
    
    async def burn_nft(self, nft_id: str, owner_address: str) -> str:
        """Burn (destroy) an NFT."""
        try:
            # Check ownership
            if self.nft_ownership.get(nft_id) != owner_address:
                raise ValueError(f"Address {owner_address} does not own NFT {nft_id}")
            
            # Burn on blockchain
            transaction = await self.blockchain_client.execute_contract_function(
                'burn',
                nft_id
            )
            
            # Remove from records
            if nft_id in self.nft_metadata:
                del self.nft_metadata[nft_id]
            if nft_id in self.nft_ownership:
                del self.nft_ownership[nft_id]
            
            # Update statistics
            self.stats['burned_nfts'] += 1
            
            logger.info(
                "NFT burned",
                nft_id=nft_id,
                owner=owner_address,
                transaction_id=transaction.transaction_id
            )
            
            return transaction.transaction_id
        
        except Exception as e:
            logger.error("NFT burning failed", nft_id=nft_id, error=str(e))
            raise
    
    def get_nft_metadata(self, nft_id: str) -> Optional[NFTMetadata]:
        """Get NFT metadata."""
        return self.nft_metadata.get(nft_id)
    
    def get_nft_owner(self, nft_id: str) -> Optional[str]:
        """Get NFT owner."""
        return self.nft_ownership.get(nft_id)
    
    def get_owner_nfts(self, owner_address: str) -> List[str]:
        """Get all NFTs owned by an address."""
        return [
            nft_id for nft_id, owner in self.nft_ownership.items()
            if owner == owner_address
        ]
    
    def get_nft_stats(self) -> Dict[str, Any]:
        """Get NFT statistics."""
        return {
            **self.stats,
            'unique_owners': len(set(self.nft_ownership.values())),
            'nft_metadata_count': len(self.nft_metadata)
        }

# =============================================================================
# BLOCKCHAIN MANAGER
# =============================================================================

class BlockchainManager:
    """Manager for multiple blockchain integrations."""
    
    def __init__(self):
        self.blockchains: Dict[str, BlockchainClient] = {}
        self.nft_managers: Dict[str, NFTManager] = {}
        self.smart_contracts: Dict[str, SmartContract] = {}
        
        # Statistics
        self.stats = {
            'total_blockchains': 0,
            'active_blockchains': 0,
            'total_transactions': 0,
            'total_nfts': 0,
            'total_contracts': 0
        }
    
    async def start(self) -> None:
        """Start all blockchain clients."""
        for client in self.blockchains.values():
            await client.start()
        
        logger.info("Blockchain manager started", blockchain_count=len(self.blockchains))
    
    async def stop(self) -> None:
        """Stop all blockchain clients."""
        for client in self.blockchains.values():
            await client.stop()
        
        logger.info("Blockchain manager stopped")
    
    def add_blockchain(self, config: BlockchainConfig) -> BlockchainClient:
        """Add a blockchain client."""
        client = BlockchainClient(config)
        self.blockchains[config.blockchain_id] = client
        self.nft_managers[config.blockchain_id] = NFTManager(client)
        
        self.stats['total_blockchains'] += 1
        if config.enabled:
            self.stats['active_blockchains'] += 1
        
        logger.info(
            "Blockchain added",
            blockchain_id=config.blockchain_id,
            name=config.name,
            type=config.blockchain_type.value
        )
        
        return client
    
    def remove_blockchain(self, blockchain_id: str) -> bool:
        """Remove a blockchain client."""
        if blockchain_id in self.blockchains:
            del self.blockchains[blockchain_id]
            del self.nft_managers[blockchain_id]
            
            self.stats['total_blockchains'] -= 1
            
            logger.info("Blockchain removed", blockchain_id=blockchain_id)
            return True
        
        return False
    
    def get_blockchain_client(self, blockchain_id: str) -> Optional[BlockchainClient]:
        """Get blockchain client."""
        return self.blockchains.get(blockchain_id)
    
    def get_nft_manager(self, blockchain_id: str) -> Optional[NFTManager]:
        """Get NFT manager for blockchain."""
        return self.nft_managers.get(blockchain_id)
    
    def add_smart_contract(self, contract: SmartContract) -> None:
        """Add smart contract."""
        self.smart_contracts[contract.contract_id] = contract
        self.stats['total_contracts'] += 1
        
        logger.info(
            "Smart contract added",
            contract_id=contract.contract_id,
            name=contract.name,
            address=contract.address
        )
    
    def get_smart_contract(self, contract_id: str) -> Optional[SmartContract]:
        """Get smart contract."""
        return self.smart_contracts.get(contract_id)
    
    def get_manager_stats(self) -> Dict[str, Any]:
        """Get manager statistics."""
        blockchain_stats = {}
        for blockchain_id, client in self.blockchains.items():
            blockchain_stats[blockchain_id] = client.get_client_stats()
        
        nft_stats = {}
        for blockchain_id, nft_manager in self.nft_managers.items():
            nft_stats[blockchain_id] = nft_manager.get_nft_stats()
        
        return {
            **self.stats,
            'blockchains': blockchain_stats,
            'nft_managers': nft_stats,
            'smart_contracts': {
                contract_id: contract.to_dict()
                for contract_id, contract in self.smart_contracts.items()
            }
        }

# =============================================================================
# GLOBAL BLOCKCHAIN INSTANCES
# =============================================================================

# Global blockchain manager
blockchain_manager = BlockchainManager()

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'BlockchainType',
    'TransactionStatus',
    'NFTStandard',
    'BlockchainConfig',
    'Transaction',
    'NFTMetadata',
    'SmartContract',
    'BlockchainClient',
    'NFTManager',
    'BlockchainManager',
    'blockchain_manager'
]





























