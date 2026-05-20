"""
Blockchain Integration System for Facebook Posts
Content verification, immutability, and decentralized storage
"""

import asyncio
import json
import time
import hashlib
import uuid
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import logging
from collections import defaultdict, deque
import aiohttp
import base64
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend

logger = logging.getLogger(__name__)


# Pure functions for blockchain integration

class BlockchainType(str, Enum):
    ETHEREUM = "ethereum"
    POLYGON = "polygon"
    BINANCE_SMART_CHAIN = "bsc"
    SOLANA = "solana"
    CUSTOM = "custom"


class ContentVerificationStatus(str, Enum):
    PENDING = "pending"
    VERIFIED = "verified"
    FAILED = "failed"
    EXPIRED = "expired"


class TransactionType(str, Enum):
    CONTENT_CREATION = "content_creation"
    CONTENT_UPDATE = "content_update"
    CONTENT_DELETION = "content_deletion"
    VERIFICATION = "verification"
    OWNERSHIP_TRANSFER = "ownership_transfer"


@dataclass(frozen=True)
class ContentHash:
    """Immutable content hash - pure data structure"""
    content_id: str
    content_hash: str
    merkle_root: str
    timestamp: datetime
    block_number: Optional[int]
    transaction_hash: Optional[str]
    verification_status: ContentVerificationStatus
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary - pure function"""
        return {
            "content_id": content_id,
            "content_hash": content_hash,
            "merkle_root": merkle_root,
            "timestamp": timestamp.isoformat(),
            "block_number": block_number,
            "transaction_hash": transaction_hash,
            "verification_status": verification_status.value
        }


@dataclass(frozen=True)
class BlockchainTransaction:
    """Immutable blockchain transaction - pure data structure"""
    transaction_id: str
    transaction_type: TransactionType
    content_id: str
    from_address: str
    to_address: str
    data: Dict[str, Any]
    gas_used: int
    gas_price: int
    block_number: int
    transaction_hash: str
    timestamp: datetime
    status: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary - pure function"""
        return {
            "transaction_id": transaction_id,
            "transaction_type": transaction_type.value,
            "content_id": content_id,
            "from_address": from_address,
            "to_address": to_address,
            "data": data,
            "gas_used": gas_used,
            "gas_price": gas_price,
            "block_number": block_number,
            "transaction_hash": transaction_hash,
            "timestamp": timestamp.isoformat(),
            "status": status
        }


@dataclass(frozen=True)
class SmartContract:
    """Immutable smart contract - pure data structure"""
    contract_address: str
    contract_name: str
    blockchain_type: BlockchainType
    abi: Dict[str, Any]
    bytecode: str
    deployed_at: datetime
    version: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary - pure data structure"""
        return {
            "contract_address": contract_address,
            "contract_name": contract_name,
            "blockchain_type": blockchain_type.value,
            "abi": abi,
            "bytecode": bytecode,
            "deployed_at": deployed_at.isoformat(),
            "version": version
        }


def calculate_content_hash(content: str, metadata: Dict[str, Any]) -> str:
    """Calculate content hash - pure function"""
    # Combine content and metadata
    combined_data = {
        "content": content,
        "metadata": metadata,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    # Create hash
    data_string = json.dumps(combined_data, sort_keys=True)
    return hashlib.sha256(data_string.encode()).hexdigest()


def calculate_merkle_root(hashes: List[str]) -> str:
    """Calculate Merkle root - pure function"""
    if not hashes:
        return ""
    
    if len(hashes) == 1:
        return hashes[0]
    
    # Create pairs and hash them
    new_hashes = []
    for i in range(0, len(hashes), 2):
        if i + 1 < len(hashes):
            combined = hashes[i] + hashes[i + 1]
        else:
            combined = hashes[i] + hashes[i]  # Duplicate last hash if odd number
        
        new_hash = hashlib.sha256(combined.encode()).hexdigest()
        new_hashes.append(new_hash)
    
    # Recursively calculate root
    return calculate_merkle_root(new_hashes)


def generate_wallet_address() -> Tuple[str, str]:
    """Generate wallet address and private key - pure function"""
    # Generate RSA key pair
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
        backend=default_backend()
    )
    
    public_key = private_key.public_key()
    
    # Serialize keys
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    )
    
    public_pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    )
    
    # Generate addresses (simplified)
    private_key_hash = hashlib.sha256(private_pem).hexdigest()
    public_key_hash = hashlib.sha256(public_pem).hexdigest()
    
    wallet_address = f"0x{public_key_hash[:40]}"
    private_key_hex = private_key_hash
    
    return wallet_address, private_key_hex


def verify_content_integrity(
    content: str,
    stored_hash: str,
    metadata: Dict[str, Any]
) -> bool:
    """Verify content integrity - pure function"""
    try:
        # Calculate current hash
        current_hash = calculate_content_hash(content, metadata)
        
        # Compare with stored hash
        return current_hash == stored_hash
        
    except Exception as e:
        logger.error(f"Error verifying content integrity: {str(e)}")
        return False


def create_content_hash(
    content_id: str,
    content: str,
    metadata: Dict[str, Any]
) -> ContentHash:
    """Create content hash - pure function"""
    content_hash = calculate_content_hash(content, metadata)
    
    # Create Merkle root (simplified - in practice, use actual Merkle tree)
    merkle_root = calculate_merkle_root([content_hash])
    
    return ContentHash(
        content_id=content_id,
        content_hash=content_hash,
        merkle_root=merkle_root,
        timestamp=datetime.utcnow(),
        block_number=None,
        transaction_hash=None,
        verification_status=ContentVerificationStatus.PENDING
    )


# Blockchain Integration System Class

class BlockchainIntegrationSystem:
    """Blockchain Integration System for content verification and immutability"""
    
    def __init__(
        self,
        blockchain_type: BlockchainType = BlockchainType.ETHEREUM,
        rpc_url: Optional[str] = None,
        private_key: Optional[str] = None
    ):
        self.blockchain_type = blockchain_type
        self.rpc_url = rpc_url or self._get_default_rpc_url()
        self.private_key = private_key
        
        # Wallet and accounts
        self.wallet_address: Optional[str] = None
        self.accounts: Dict[str, str] = {}  # address -> private_key
        
        # Smart contracts
        self.smart_contracts: Dict[str, SmartContract] = {}
        
        # Content registry
        self.content_hashes: Dict[str, ContentHash] = {}
        self.transactions: Dict[str, BlockchainTransaction] = {}
        
        # HTTP client for blockchain communication
        self.http_client: Optional[aiohttp.ClientSession] = None
        
        # Statistics
        self.stats = {
            "total_transactions": 0,
            "successful_transactions": 0,
            "failed_transactions": 0,
            "content_verified": 0,
            "gas_used_total": 0,
            "average_gas_price": 0.0
        }
        
        # Background tasks
        self.verification_task: Optional[asyncio.Task] = None
        self.is_running = False
    
    def _get_default_rpc_url(self) -> str:
        """Get default RPC URL - pure function"""
        rpc_urls = {
            BlockchainType.ETHEREUM: "https://mainnet.infura.io/v3/YOUR_PROJECT_ID",
            BlockchainType.POLYGON: "https://polygon-rpc.com",
            BlockchainType.BINANCE_SMART_CHAIN: "https://bsc-dataseed.binance.org",
            BlockchainType.SOLANA: "https://api.mainnet-beta.solana.com"
        }
        
        return rpc_urls.get(self.blockchain_type, "https://localhost:8545")
    
    async def start(self) -> None:
        """Start blockchain integration system"""
        if self.is_running:
            return
        
        try:
            # Initialize HTTP client
            self.http_client = aiohttp.ClientSession()
            
            # Generate or load wallet
            if not self.wallet_address:
                self.wallet_address, self.private_key = generate_wallet_address()
                self.accounts[self.wallet_address] = self.private_key
            
            # Deploy smart contracts
            await self._deploy_smart_contracts()
            
            # Start background tasks
            self.is_running = True
            self.verification_task = asyncio.create_task(self._verification_loop())
            
            logger.info("Blockchain integration system started")
            
        except Exception as e:
            logger.error(f"Error starting blockchain integration system: {str(e)}")
            raise
    
    async def stop(self) -> None:
        """Stop blockchain integration system"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        if self.verification_task:
            self.verification_task.cancel()
        
        if self.http_client:
            await self.http_client.close()
        
        logger.info("Blockchain integration system stopped")
    
    async def _deploy_smart_contracts(self) -> None:
        """Deploy smart contracts"""
        try:
            # Deploy content verification contract
            content_contract = await self._deploy_content_verification_contract()
            self.smart_contracts["content_verification"] = content_contract
            
            # Deploy ownership contract
            ownership_contract = await self._deploy_ownership_contract()
            self.smart_contracts["ownership"] = ownership_contract
            
            logger.info("Smart contracts deployed successfully")
            
        except Exception as e:
            logger.error(f"Error deploying smart contracts: {str(e)}")
    
    async def _deploy_content_verification_contract(self) -> SmartContract:
        """Deploy content verification smart contract"""
        try:
            # Contract ABI (simplified)
            abi = [
                {
                    "inputs": [
                        {"name": "contentId", "type": "string"},
                        {"name": "contentHash", "type": "string"},
                        {"name": "merkleRoot", "type": "string"}
                    ],
                    "name": "verifyContent",
                    "outputs": [{"name": "verified", "type": "bool"}],
                    "stateMutability": "nonpayable",
                    "type": "function"
                },
                {
                    "inputs": [{"name": "contentId", "type": "string"}],
                    "name": "getContentHash",
                    "outputs": [{"name": "contentHash", "type": "string"}],
                    "stateMutability": "view",
                    "type": "function"
                }
            ]
            
            # Deploy contract (simplified)
            contract_address = f"0x{hashlib.sha256(f'content_verification_{int(time.time())}'.encode()).hexdigest()[:40]}"
            
            contract = SmartContract(
                contract_address=contract_address,
                contract_name="ContentVerification",
                blockchain_type=self.blockchain_type,
                abi=abi,
                bytecode="0x608060405234801561001057600080fd5b50...",  # Simplified
                deployed_at=datetime.utcnow(),
                version="1.0.0"
            )
            
            return contract
            
        except Exception as e:
            logger.error(f"Error deploying content verification contract: {str(e)}")
            raise
    
    async def _deploy_ownership_contract(self) -> SmartContract:
        """Deploy ownership smart contract"""
        try:
            # Contract ABI (simplified)
            abi = [
                {
                    "inputs": [
                        {"name": "contentId", "type": "string"},
                        {"name": "owner", "type": "address"}
                    ],
                    "name": "setOwner",
                    "outputs": [],
                    "stateMutability": "nonpayable",
                    "type": "function"
                },
                {
                    "inputs": [{"name": "contentId", "type": "string"}],
                    "name": "getOwner",
                    "outputs": [{"name": "owner", "type": "address"}],
                    "stateMutability": "view",
                    "type": "function"
                }
            ]
            
            # Deploy contract (simplified)
            contract_address = f"0x{hashlib.sha256(f'ownership_{int(time.time())}'.encode()).hexdigest()[:40]}"
            
            contract = SmartContract(
                contract_address=contract_address,
                contract_name="Ownership",
                blockchain_type=self.blockchain_type,
                abi=abi,
                bytecode="0x608060405234801561001057600080fd5b50...",  # Simplified
                deployed_at=datetime.utcnow(),
                version="1.0.0"
            )
            
            return contract
            
        except Exception as e:
            logger.error(f"Error deploying ownership contract: {str(e)}")
            raise
    
    async def register_content(
        self,
        content_id: str,
        content: str,
        metadata: Dict[str, Any]
    ) -> ContentHash:
        """Register content on blockchain"""
        try:
            # Create content hash
            content_hash = create_content_hash(content_id, content, metadata)
            
            # Store locally
            self.content_hashes[content_id] = content_hash
            
            # Create blockchain transaction
            transaction = await self._create_content_transaction(
                TransactionType.CONTENT_CREATION,
                content_id,
                content_hash
            )
            
            # Update content hash with transaction info
            updated_hash = ContentHash(
                content_id=content_hash.content_id,
                content_hash=content_hash.content_hash,
                merkle_root=content_hash.merkle_root,
                timestamp=content_hash.timestamp,
                block_number=transaction.block_number,
                transaction_hash=transaction.transaction_hash,
                verification_status=ContentVerificationStatus.VERIFIED
            )
            
            self.content_hashes[content_id] = updated_hash
            
            # Update statistics
            self.stats["total_transactions"] += 1
            self.stats["successful_transactions"] += 1
            self.stats["content_verified"] += 1
            
            logger.info(f"Content registered on blockchain: {content_id}")
            
            return updated_hash
            
        except Exception as e:
            logger.error(f"Error registering content: {str(e)}")
            raise
    
    async def verify_content(
        self,
        content_id: str,
        content: str,
        metadata: Dict[str, Any]
    ) -> bool:
        """Verify content integrity"""
        try:
            # Get stored hash
            stored_hash = self.content_hashes.get(content_id)
            if not stored_hash:
                return False
            
            # Verify integrity
            is_valid = verify_content_integrity(content, stored_hash.content_hash, metadata)
            
            if is_valid:
                # Update verification status
                updated_hash = ContentHash(
                    content_id=stored_hash.content_id,
                    content_hash=stored_hash.content_hash,
                    merkle_root=stored_hash.merkle_root,
                    timestamp=stored_hash.timestamp,
                    block_number=stored_hash.block_number,
                    transaction_hash=stored_hash.transaction_hash,
                    verification_status=ContentVerificationStatus.VERIFIED
                )
                
                self.content_hashes[content_id] = updated_hash
                self.stats["content_verified"] += 1
            
            return is_valid
            
        except Exception as e:
            logger.error(f"Error verifying content: {str(e)}")
            return False
    
    async def transfer_ownership(
        self,
        content_id: str,
        from_address: str,
        to_address: str
    ) -> BlockchainTransaction:
        """Transfer content ownership"""
        try:
            # Create ownership transfer transaction
            transaction = await self._create_ownership_transaction(
                content_id,
                from_address,
                to_address
            )
            
            # Update statistics
            self.stats["total_transactions"] += 1
            self.stats["successful_transactions"] += 1
            
            logger.info(f"Ownership transferred: {content_id} from {from_address} to {to_address}")
            
            return transaction
            
        except Exception as e:
            logger.error(f"Error transferring ownership: {str(e)}")
            raise
    
    async def _create_content_transaction(
        self,
        transaction_type: TransactionType,
        content_id: str,
        content_hash: ContentHash
    ) -> BlockchainTransaction:
        """Create content-related transaction"""
        try:
            # Generate transaction hash
            transaction_hash = hashlib.sha256(
                f"{content_id}_{content_hash.content_hash}_{int(time.time())}".encode()
            ).hexdigest()
            
            # Create transaction
            transaction = BlockchainTransaction(
                transaction_id=f"tx_{uuid.uuid4().hex[:8]}",
                transaction_type=transaction_type,
                content_id=content_id,
                from_address=self.wallet_address or "0x0",
                to_address=self.smart_contracts["content_verification"].contract_address,
                data={
                    "content_hash": content_hash.content_hash,
                    "merkle_root": content_hash.merkle_root,
                    "timestamp": content_hash.timestamp.isoformat()
                },
                gas_used=21000,  # Standard gas limit
                gas_price=20000000000,  # 20 Gwei
                block_number=int(time.time()) % 1000000,  # Simulated block number
                transaction_hash=transaction_hash,
                timestamp=datetime.utcnow(),
                status="success"
            )
            
            # Store transaction
            self.transactions[transaction.transaction_id] = transaction
            
            return transaction
            
        except Exception as e:
            logger.error(f"Error creating content transaction: {str(e)}")
            raise
    
    async def _create_ownership_transaction(
        self,
        content_id: str,
        from_address: str,
        to_address: str
    ) -> BlockchainTransaction:
        """Create ownership transfer transaction"""
        try:
            # Generate transaction hash
            transaction_hash = hashlib.sha256(
                f"{content_id}_{from_address}_{to_address}_{int(time.time())}".encode()
            ).hexdigest()
            
            # Create transaction
            transaction = BlockchainTransaction(
                transaction_id=f"tx_{uuid.uuid4().hex[:8]}",
                transaction_type=TransactionType.OWNERSHIP_TRANSFER,
                content_id=content_id,
                from_address=from_address,
                to_address=to_address,
                data={
                    "content_id": content_id,
                    "previous_owner": from_address,
                    "new_owner": to_address
                },
                gas_used=25000,
                gas_price=20000000000,
                block_number=int(time.time()) % 1000000,
                transaction_hash=transaction_hash,
                timestamp=datetime.utcnow(),
                status="success"
            )
            
            # Store transaction
            self.transactions[transaction.transaction_id] = transaction
            
            return transaction
            
        except Exception as e:
            logger.error(f"Error creating ownership transaction: {str(e)}")
            raise
    
    async def _verification_loop(self) -> None:
        """Background verification loop"""
        while self.is_running:
            try:
                # Process pending verifications
                for content_id, content_hash in self.content_hashes.items():
                    if content_hash.verification_status == ContentVerificationStatus.PENDING:
                        # Simulate verification process
                        await asyncio.sleep(1)
                        
                        # Update status to verified
                        updated_hash = ContentHash(
                            content_id=content_hash.content_id,
                            content_hash=content_hash.content_hash,
                            merkle_root=content_hash.merkle_root,
                            timestamp=content_hash.timestamp,
                            block_number=content_hash.block_number,
                            transaction_hash=content_hash.transaction_hash,
                            verification_status=ContentVerificationStatus.VERIFIED
                        )
                        
                        self.content_hashes[content_id] = updated_hash
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in verification loop: {str(e)}")
                await asyncio.sleep(10)
    
    def get_content_hash(self, content_id: str) -> Optional[ContentHash]:
        """Get content hash by ID"""
        return self.content_hashes.get(content_id)
    
    def get_transaction(self, transaction_id: str) -> Optional[BlockchainTransaction]:
        """Get transaction by ID"""
        return self.transactions.get(transaction_id)
    
    def get_verified_content(self) -> List[ContentHash]:
        """Get all verified content"""
        return [
            content_hash for content_hash in self.content_hashes.values()
            if content_hash.verification_status == ContentVerificationStatus.VERIFIED
        ]
    
    def get_blockchain_statistics(self) -> Dict[str, Any]:
        """Get blockchain statistics"""
        return {
            "statistics": self.stats.copy(),
            "blockchain_type": self.blockchain_type.value,
            "wallet_address": self.wallet_address,
            "smart_contracts": {
                name: contract.to_dict()
                for name, contract in self.smart_contracts.items()
            },
            "content_hashes": {
                content_id: content_hash.to_dict()
                for content_id, content_hash in self.content_hashes.items()
            },
            "transactions": {
                tx_id: transaction.to_dict()
                for tx_id, transaction in self.transactions.items()
            }
        }


# Factory functions

def create_blockchain_integration_system(
    blockchain_type: BlockchainType = BlockchainType.ETHEREUM,
    rpc_url: Optional[str] = None,
    private_key: Optional[str] = None
) -> BlockchainIntegrationSystem:
    """Create blockchain integration system - pure function"""
    return BlockchainIntegrationSystem(blockchain_type, rpc_url, private_key)


async def get_blockchain_integration_system() -> BlockchainIntegrationSystem:
    """Get blockchain integration system instance"""
    system = create_blockchain_integration_system()
    await system.start()
    return system

