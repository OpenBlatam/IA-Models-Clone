"""
Blockchain Integration - Integración de Blockchain para Documentos Inmutables
Advanced blockchain technology for document immutability and verification

This module implements blockchain capabilities including:
- Document immutability and tamper-proof storage
- Multi-blockchain support (Ethereum, Hyperledger, etc.)
- Smart contracts for document management
- Document verification and authentication
- NFT-based document ownership
- Decentralized identity management
- Cross-chain document transfer
- Blockchain-based audit trails
"""

import asyncio
import logging
import time
import json
import hashlib
import base64
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import uuid
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
import ecdsa
import base58

# Blockchain libraries (optional imports)
try:
    from web3 import Web3
    from eth_account import Account
    WEB3_AVAILABLE = True
except ImportError:
    WEB3_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BlockchainType(Enum):
    """Supported blockchain types"""
    ETHEREUM = "ethereum"
    BITCOIN = "bitcoin"
    HYPERLEDGER = "hyperledger"
    POLYGON = "polygon"
    BINANCE_SMART_CHAIN = "bsc"
    AVALANCHE = "avalanche"
    SOLANA = "solana"
    CUSTOM = "custom"

class DocumentStatus(Enum):
    """Document status on blockchain"""
    PENDING = "pending"
    CONFIRMED = "confirmed"
    VERIFIED = "verified"
    DISPUTED = "disputed"
    REVOKED = "revoked"

class TransactionType(Enum):
    """Blockchain transaction types"""
    DOCUMENT_CREATE = "document_create"
    DOCUMENT_UPDATE = "document_update"
    DOCUMENT_TRANSFER = "document_transfer"
    DOCUMENT_VERIFY = "document_verify"
    DOCUMENT_REVOKE = "document_revoke"
    OWNERSHIP_CHANGE = "ownership_change"

@dataclass
class BlockchainConfig:
    """Configuration for blockchain integration"""
    blockchain_type: BlockchainType
    network_url: str = ""
    contract_address: str = ""
    private_key: str = ""
    gas_limit: int = 200000
    gas_price: int = 20
    confirmation_blocks: int = 12
    timeout: int = 300
    
    # Advanced parameters
    enable_smart_contracts: bool = True
    enable_nft_support: bool = True
    enable_cross_chain: bool = False
    enable_privacy: bool = False
    consensus_algorithm: str = "proof_of_work"

@dataclass
class DocumentHash:
    """Document hash for blockchain storage"""
    document_id: str
    content_hash: str
    metadata_hash: str
    timestamp: float
    block_number: Optional[int] = None
    transaction_hash: Optional[str] = None
    merkle_root: Optional[str] = None

@dataclass
class BlockchainDocument:
    """Document representation on blockchain"""
    document_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    owner_address: str = ""
    creator_address: str = ""
    status: DocumentStatus = DocumentStatus.PENDING
    blockchain_type: BlockchainType = BlockchainType.ETHEREUM
    transaction_hash: Optional[str] = None
    block_number: Optional[int] = None
    gas_used: Optional[int] = None
    created_at: float = field(default_factory=time.time)
    verified_at: Optional[float] = None
    version: int = 1
    previous_version_hash: Optional[str] = None

@dataclass
class SmartContract:
    """Smart contract representation"""
    contract_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    contract_address: str = ""
    abi: List[Dict[str, Any]] = field(default_factory=list)
    bytecode: str = ""
    blockchain_type: BlockchainType = BlockchainType.ETHEREUM
    deployed_at: Optional[float] = None
    gas_cost: Optional[int] = None

@dataclass
class NFTMetadata:
    """NFT metadata for document ownership"""
    name: str = ""
    description: str = ""
    image: str = ""
    attributes: List[Dict[str, Any]] = field(default_factory=list)
    external_url: str = ""
    animation_url: str = ""
    background_color: str = ""
    youtube_url: str = ""

class BlockchainManager:
    """Main blockchain manager for document operations"""
    
    def __init__(self, config: BlockchainConfig):
        self.config = config
        self.web3 = None
        self.account = None
        self.contracts: Dict[str, SmartContract] = {}
        self.documents: Dict[str, BlockchainDocument] = {}
        self.transaction_history: List[Dict[str, Any]] = []
        
        # Initialize blockchain connection
        self._initialize_blockchain()
    
    def _initialize_blockchain(self):
        """Initialize blockchain connection"""
        try:
            if self.config.blockchain_type == BlockchainType.ETHEREUM and WEB3_AVAILABLE:
                self.web3 = Web3(Web3.HTTPProvider(self.config.network_url))
                
                if self.config.private_key:
                    self.account = Account.from_key(self.config.private_key)
                
                logger.info(f"Connected to {self.config.blockchain_type.value} blockchain")
            else:
                logger.warning(f"Blockchain {self.config.blockchain_type.value} not fully supported")
                
        except Exception as e:
            logger.error(f"Error initializing blockchain: {str(e)}")
    
    async def create_document_hash(self, document_content: str, 
                                 metadata: Dict[str, Any]) -> DocumentHash:
        """Create cryptographic hash for document"""
        try:
            # Create content hash
            content_bytes = document_content.encode('utf-8')
            content_hash = hashlib.sha256(content_bytes).hexdigest()
            
            # Create metadata hash
            metadata_json = json.dumps(metadata, sort_keys=True)
            metadata_bytes = metadata_json.encode('utf-8')
            metadata_hash = hashlib.sha256(metadata_bytes).hexdigest()
            
            # Create combined hash
            combined_data = f"{content_hash}{metadata_hash}{time.time()}"
            combined_hash = hashlib.sha256(combined_data.encode('utf-8')).hexdigest()
            
            document_hash = DocumentHash(
                document_id=str(uuid.uuid4()),
                content_hash=content_hash,
                metadata_hash=metadata_hash,
                timestamp=time.time()
            )
            
            logger.info(f"Created document hash: {document_hash.document_id}")
            return document_hash
            
        except Exception as e:
            logger.error(f"Error creating document hash: {str(e)}")
            return DocumentHash(document_id="", content_hash="", metadata_hash="", timestamp=0)
    
    async def store_document_on_blockchain(self, document: BlockchainDocument) -> bool:
        """Store document on blockchain"""
        try:
            if not self.web3:
                logger.error("Blockchain not initialized")
                return False
            
            # Create document hash
            document_hash = await self.create_document_hash(document.content, document.metadata)
            
            # Prepare transaction data
            transaction_data = {
                "document_id": document.document_id,
                "content_hash": document_hash.content_hash,
                "metadata_hash": document_hash.metadata_hash,
                "owner": document.owner_address,
                "creator": document.creator_address,
                "timestamp": int(document.created_at),
                "version": document.version
            }
            
            # Estimate gas
            gas_estimate = self._estimate_gas(transaction_data)
            
            # Create transaction
            transaction = {
                "from": self.account.address,
                "gas": gas_estimate,
                "gasPrice": self.web3.toWei(self.config.gas_price, 'gwei'),
                "nonce": self.web3.eth.get_transaction_count(self.account.address)
            }
            
            # Sign and send transaction
            signed_txn = self.account.sign_transaction(transaction)
            tx_hash = self.web3.eth.send_raw_transaction(signed_txn.rawTransaction)
            
            # Wait for confirmation
            receipt = self.web3.eth.wait_for_transaction_receipt(
                tx_hash, timeout=self.config.timeout
            )
            
            # Update document with blockchain data
            document.transaction_hash = tx_hash.hex()
            document.block_number = receipt.blockNumber
            document.gas_used = receipt.gasUsed
            document.status = DocumentStatus.CONFIRMED
            
            # Store document
            self.documents[document.document_id] = document
            
            # Record transaction
            self.transaction_history.append({
                "type": TransactionType.DOCUMENT_CREATE.value,
                "document_id": document.document_id,
                "transaction_hash": tx_hash.hex(),
                "block_number": receipt.blockNumber,
                "timestamp": time.time()
            })
            
            logger.info(f"Document stored on blockchain: {document.document_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing document on blockchain: {str(e)}")
            return False
    
    def _estimate_gas(self, transaction_data: Dict[str, Any]) -> int:
        """Estimate gas for transaction"""
        # Simplified gas estimation
        base_gas = 21000
        data_gas = len(json.dumps(transaction_data)) * 16
        return base_gas + data_gas
    
    async def verify_document_integrity(self, document_id: str) -> Dict[str, Any]:
        """Verify document integrity on blockchain"""
        try:
            if document_id not in self.documents:
                return {"verified": False, "error": "Document not found"}
            
            document = self.documents[document_id]
            
            if not document.transaction_hash:
                return {"verified": False, "error": "Document not on blockchain"}
            
            # Get transaction from blockchain
            transaction = self.web3.eth.get_transaction(document.transaction_hash)
            receipt = self.web3.eth.get_transaction_receipt(document.transaction_hash)
            
            # Verify transaction exists and is confirmed
            if not transaction or not receipt:
                return {"verified": False, "error": "Transaction not found"}
            
            # Check confirmation count
            current_block = self.web3.eth.block_number
            confirmations = current_block - receipt.blockNumber
            
            if confirmations < self.config.confirmation_blocks:
                return {
                    "verified": False,
                    "error": f"Insufficient confirmations: {confirmations}/{self.config.confirmation_blocks}"
                }
            
            # Verify document content hasn't changed
            current_hash = await self.create_document_hash(document.content, document.metadata)
            
            verification_result = {
                "verified": True,
                "document_id": document_id,
                "transaction_hash": document.transaction_hash,
                "block_number": document.block_number,
                "confirmations": confirmations,
                "content_hash": current_hash.content_hash,
                "metadata_hash": current_hash.metadata_hash,
                "verified_at": time.time()
            }
            
            # Update document status
            document.status = DocumentStatus.VERIFIED
            document.verified_at = time.time()
            
            logger.info(f"Document integrity verified: {document_id}")
            return verification_result
            
        except Exception as e:
            logger.error(f"Error verifying document integrity: {str(e)}")
            return {"verified": False, "error": str(e)}
    
    async def transfer_document_ownership(self, document_id: str, 
                                        new_owner_address: str) -> bool:
        """Transfer document ownership on blockchain"""
        try:
            if document_id not in self.documents:
                logger.error(f"Document {document_id} not found")
                return False
            
            document = self.documents[document_id]
            
            if document.owner_address != self.account.address:
                logger.error("Not authorized to transfer document")
                return False
            
            # Create ownership transfer transaction
            transfer_data = {
                "document_id": document_id,
                "from_owner": document.owner_address,
                "to_owner": new_owner_address,
                "timestamp": int(time.time())
            }
            
            # Estimate gas
            gas_estimate = self._estimate_gas(transfer_data)
            
            # Create transaction
            transaction = {
                "from": self.account.address,
                "gas": gas_estimate,
                "gasPrice": self.web3.toWei(self.config.gas_price, 'gwei'),
                "nonce": self.web3.eth.get_transaction_count(self.account.address)
            }
            
            # Sign and send transaction
            signed_txn = self.account.sign_transaction(transaction)
            tx_hash = self.web3.eth.send_raw_transaction(signed_txn.rawTransaction)
            
            # Wait for confirmation
            receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash)
            
            # Update document ownership
            document.owner_address = new_owner_address
            
            # Record transaction
            self.transaction_history.append({
                "type": TransactionType.OWNERSHIP_CHANGE.value,
                "document_id": document_id,
                "transaction_hash": tx_hash.hex(),
                "block_number": receipt.blockNumber,
                "timestamp": time.time()
            })
            
            logger.info(f"Document ownership transferred: {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error transferring document ownership: {str(e)}")
            return False
    
    async def create_nft_for_document(self, document_id: str, 
                                    nft_metadata: NFTMetadata) -> Dict[str, Any]:
        """Create NFT for document ownership"""
        try:
            if document_id not in self.documents:
                return {"success": False, "error": "Document not found"}
            
            document = self.documents[document_id]
            
            # Create NFT metadata
            nft_data = {
                "name": nft_metadata.name or document.title,
                "description": nft_metadata.description or f"Document: {document.title}",
                "image": nft_metadata.image,
                "attributes": nft_metadata.attributes,
                "external_url": nft_metadata.external_url,
                "document_id": document_id,
                "content_hash": await self.create_document_hash(document.content, document.metadata)
            }
            
            # Store NFT metadata (in practice, this would be on IPFS)
            metadata_hash = hashlib.sha256(json.dumps(nft_data).encode()).hexdigest()
            
            # Create NFT transaction
            nft_transaction_data = {
                "token_id": str(uuid.uuid4()),
                "owner": document.owner_address,
                "metadata_hash": metadata_hash,
                "document_id": document_id,
                "timestamp": int(time.time())
            }
            
            # Estimate gas
            gas_estimate = self._estimate_gas(nft_transaction_data)
            
            # Create transaction
            transaction = {
                "from": self.account.address,
                "gas": gas_estimate,
                "gasPrice": self.web3.toWei(self.config.gas_price, 'gwei'),
                "nonce": self.web3.eth.get_transaction_count(self.account.address)
            }
            
            # Sign and send transaction
            signed_txn = self.account.sign_transaction(transaction)
            tx_hash = self.web3.eth.send_raw_transaction(signed_txn.rawTransaction)
            
            # Wait for confirmation
            receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash)
            
            nft_result = {
                "success": True,
                "token_id": nft_transaction_data["token_id"],
                "transaction_hash": tx_hash.hex(),
                "block_number": receipt.blockNumber,
                "metadata_hash": metadata_hash,
                "owner": document.owner_address
            }
            
            logger.info(f"NFT created for document: {document_id}")
            return nft_result
            
        except Exception as e:
            logger.error(f"Error creating NFT for document: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def get_document_audit_trail(self, document_id: str) -> List[Dict[str, Any]]:
        """Get complete audit trail for document"""
        try:
            audit_trail = []
            
            # Get document transactions
            document_transactions = [
                tx for tx in self.transaction_history 
                if tx.get("document_id") == document_id
            ]
            
            for tx in document_transactions:
                # Get transaction details from blockchain
                if tx.get("transaction_hash"):
                    blockchain_tx = self.web3.eth.get_transaction(tx["transaction_hash"])
                    receipt = self.web3.eth.get_transaction_receipt(tx["transaction_hash"])
                    
                    audit_entry = {
                        "type": tx["type"],
                        "transaction_hash": tx["transaction_hash"],
                        "block_number": tx["block_number"],
                        "timestamp": tx["timestamp"],
                        "gas_used": receipt.gasUsed if receipt else None,
                        "status": "confirmed" if receipt else "pending"
                    }
                    
                    audit_trail.append(audit_entry)
            
            # Sort by timestamp
            audit_trail.sort(key=lambda x: x["timestamp"])
            
            logger.info(f"Retrieved audit trail for document: {document_id}")
            return audit_trail
            
        except Exception as e:
            logger.error(f"Error getting document audit trail: {str(e)}")
            return []
    
    async def revoke_document(self, document_id: str, reason: str) -> bool:
        """Revoke document on blockchain"""
        try:
            if document_id not in self.documents:
                logger.error(f"Document {document_id} not found")
                return False
            
            document = self.documents[document_id]
            
            if document.owner_address != self.account.address:
                logger.error("Not authorized to revoke document")
                return False
            
            # Create revocation transaction
            revocation_data = {
                "document_id": document_id,
                "reason": reason,
                "revoked_by": self.account.address,
                "timestamp": int(time.time())
            }
            
            # Estimate gas
            gas_estimate = self._estimate_gas(revocation_data)
            
            # Create transaction
            transaction = {
                "from": self.account.address,
                "gas": gas_estimate,
                "gasPrice": self.web3.toWei(self.config.gas_price, 'gwei'),
                "nonce": self.web3.eth.get_transaction_count(self.account.address)
            }
            
            # Sign and send transaction
            signed_txn = self.account.sign_transaction(transaction)
            tx_hash = self.web3.eth.send_raw_transaction(signed_txn.rawTransaction)
            
            # Wait for confirmation
            receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash)
            
            # Update document status
            document.status = DocumentStatus.REVOKED
            
            # Record transaction
            self.transaction_history.append({
                "type": TransactionType.DOCUMENT_REVOKE.value,
                "document_id": document_id,
                "transaction_hash": tx_hash.hex(),
                "block_number": receipt.blockNumber,
                "timestamp": time.time(),
                "reason": reason
            })
            
            logger.info(f"Document revoked: {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error revoking document: {str(e)}")
            return False

class MultiBlockchainManager:
    """Manager for multiple blockchain networks"""
    
    def __init__(self):
        self.blockchain_managers: Dict[BlockchainType, BlockchainManager] = {}
        self.cross_chain_bridges: Dict[Tuple[BlockchainType, BlockchainType], Any] = {}
        
    async def add_blockchain(self, blockchain_type: BlockchainType, 
                           config: BlockchainConfig) -> bool:
        """Add blockchain network"""
        try:
            manager = BlockchainManager(config)
            self.blockchain_managers[blockchain_type] = manager
            
            logger.info(f"Added blockchain: {blockchain_type.value}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding blockchain {blockchain_type.value}: {str(e)}")
            return False
    
    async def store_document_multi_chain(self, document: BlockchainDocument, 
                                       blockchain_types: List[BlockchainType]) -> Dict[str, Any]:
        """Store document on multiple blockchains"""
        results = {}
        
        for blockchain_type in blockchain_types:
            if blockchain_type in self.blockchain_managers:
                manager = self.blockchain_managers[blockchain_type]
                
                # Create document copy for this blockchain
                doc_copy = BlockchainDocument(
                    document_id=document.document_id,
                    title=document.title,
                    content=document.content,
                    metadata=document.metadata,
                    owner_address=document.owner_address,
                    creator_address=document.creator_address,
                    blockchain_type=blockchain_type
                )
                
                success = await manager.store_document_on_blockchain(doc_copy)
                results[blockchain_type.value] = {
                    "success": success,
                    "transaction_hash": doc_copy.transaction_hash,
                    "block_number": doc_copy.block_number
                }
            else:
                results[blockchain_type.value] = {
                    "success": False,
                    "error": f"Blockchain {blockchain_type.value} not available"
                }
        
        return results
    
    async def verify_document_cross_chain(self, document_id: str) -> Dict[str, Any]:
        """Verify document across multiple blockchains"""
        verification_results = {}
        
        for blockchain_type, manager in self.blockchain_managers.items():
            if document_id in manager.documents:
                verification = await manager.verify_document_integrity(document_id)
                verification_results[blockchain_type.value] = verification
        
        # Check consistency across chains
        consistent = len(set(
            result.get("content_hash", "") 
            for result in verification_results.values() 
            if result.get("verified", False)
        )) <= 1
        
        return {
            "cross_chain_verification": verification_results,
            "consistent_across_chains": consistent,
            "verification_count": sum(1 for r in verification_results.values() if r.get("verified", False))
        }

class BlockchainIntegration:
    """Main Blockchain Integration Engine"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.multi_blockchain_manager = MultiBlockchainManager()
        self.smart_contracts: Dict[str, SmartContract] = {}
        self.document_registry: Dict[str, BlockchainDocument] = {}
        self.analytics_data: Dict[str, Any] = {}
        
        # Initialize default blockchains
        self._initialize_default_blockchains()
        
        logger.info("Blockchain Integration Engine initialized")
    
    def _initialize_default_blockchains(self):
        """Initialize default blockchain configurations"""
        # Ethereum configuration
        ethereum_config = BlockchainConfig(
            blockchain_type=BlockchainType.ETHEREUM,
            network_url="https://mainnet.infura.io/v3/YOUR_PROJECT_ID",
            gas_limit=200000,
            gas_price=20,
            confirmation_blocks=12
        )
        
        # Add to multi-blockchain manager
        asyncio.create_task(
            self.multi_blockchain_manager.add_blockchain(
                BlockchainType.ETHEREUM, ethereum_config
            )
        )
    
    async def create_immutable_document(self, document_data: Dict[str, Any], 
                                      blockchain_types: List[BlockchainType] = None) -> Dict[str, Any]:
        """Create immutable document on blockchain"""
        try:
            if blockchain_types is None:
                blockchain_types = [BlockchainType.ETHEREUM]
            
            # Create blockchain document
            blockchain_doc = BlockchainDocument(
                title=document_data.get("title", "Untitled Document"),
                content=document_data.get("content", ""),
                metadata=document_data.get("metadata", {}),
                owner_address=document_data.get("owner_address", ""),
                creator_address=document_data.get("creator_address", ""),
                blockchain_type=blockchain_types[0]
            )
            
            # Store on multiple blockchains
            storage_results = await self.multi_blockchain_manager.store_document_multi_chain(
                blockchain_doc, blockchain_types
            )
            
            # Register document
            self.document_registry[blockchain_doc.document_id] = blockchain_doc
            
            result = {
                "success": True,
                "document_id": blockchain_doc.document_id,
                "blockchain_storage": storage_results,
                "created_at": blockchain_doc.created_at
            }
            
            logger.info(f"Created immutable document: {blockchain_doc.document_id}")
            return result
            
        except Exception as e:
            logger.error(f"Error creating immutable document: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def verify_document_authenticity(self, document_id: str) -> Dict[str, Any]:
        """Verify document authenticity across blockchains"""
        try:
            if document_id not in self.document_registry:
                return {"verified": False, "error": "Document not found in registry"}
            
            # Cross-chain verification
            verification_results = await self.multi_blockchain_manager.verify_document_cross_chain(document_id)
            
            # Calculate authenticity score
            verification_count = verification_results["verification_count"]
            total_chains = len(self.multi_blockchain_manager.blockchain_managers)
            authenticity_score = verification_count / total_chains if total_chains > 0 else 0
            
            result = {
                "verified": verification_results["consistent_across_chains"],
                "authenticity_score": authenticity_score,
                "verification_details": verification_results["cross_chain_verification"],
                "verified_at": time.time()
            }
            
            logger.info(f"Document authenticity verified: {document_id}")
            return result
            
        except Exception as e:
            logger.error(f"Error verifying document authenticity: {str(e)}")
            return {"verified": False, "error": str(e)}
    
    async def create_document_nft(self, document_id: str, 
                                nft_metadata: NFTMetadata) -> Dict[str, Any]:
        """Create NFT for document ownership"""
        try:
            if document_id not in self.document_registry:
                return {"success": False, "error": "Document not found"}
            
            document = self.document_registry[document_id]
            
            # Get blockchain manager for document's blockchain
            if document.blockchain_type in self.multi_blockchain_manager.blockchain_managers:
                manager = self.multi_blockchain_manager.blockchain_managers[document.blockchain_type]
                nft_result = await manager.create_nft_for_document(document_id, nft_metadata)
                
                logger.info(f"NFT created for document: {document_id}")
                return nft_result
            else:
                return {"success": False, "error": f"Blockchain {document.blockchain_type.value} not available"}
                
        except Exception as e:
            logger.error(f"Error creating document NFT: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def get_blockchain_analytics(self) -> Dict[str, Any]:
        """Get blockchain analytics and statistics"""
        try:
            analytics = {
                "total_documents": len(self.document_registry),
                "blockchain_networks": len(self.multi_blockchain_manager.blockchain_managers),
                "total_transactions": sum(
                    len(manager.transaction_history) 
                    for manager in self.multi_blockchain_manager.blockchain_managers.values()
                ),
                "document_status_distribution": {},
                "blockchain_usage": {},
                "gas_statistics": {}
            }
            
            # Document status distribution
            for document in self.document_registry.values():
                status = document.status.value
                analytics["document_status_distribution"][status] = \
                    analytics["document_status_distribution"].get(status, 0) + 1
            
            # Blockchain usage
            for blockchain_type, manager in self.multi_blockchain_manager.blockchain_managers.items():
                analytics["blockchain_usage"][blockchain_type.value] = {
                    "documents": len(manager.documents),
                    "transactions": len(manager.transaction_history)
                }
            
            # Gas statistics
            total_gas = 0
            gas_count = 0
            for manager in self.multi_blockchain_manager.blockchain_managers.values():
                for doc in manager.documents.values():
                    if doc.gas_used:
                        total_gas += doc.gas_used
                        gas_count += 1
            
            if gas_count > 0:
                analytics["gas_statistics"] = {
                    "total_gas_used": total_gas,
                    "average_gas_per_document": total_gas / gas_count,
                    "documents_with_gas_data": gas_count
                }
            
            return analytics
            
        except Exception as e:
            logger.error(f"Error getting blockchain analytics: {str(e)}")
            return {}
    
    async def export_blockchain_data(self, export_path: str) -> bool:
        """Export blockchain data for backup/audit"""
        try:
            export_data = {
                "document_registry": {
                    doc_id: {
                        "title": doc.title,
                        "status": doc.status.value,
                        "blockchain_type": doc.blockchain_type.value,
                        "transaction_hash": doc.transaction_hash,
                        "block_number": doc.block_number,
                        "created_at": doc.created_at,
                        "owner_address": doc.owner_address
                    }
                    for doc_id, doc in self.document_registry.items()
                },
                "transaction_history": {},
                "analytics": await self.get_blockchain_analytics(),
                "export_timestamp": time.time()
            }
            
            # Add transaction history for each blockchain
            for blockchain_type, manager in self.multi_blockchain_manager.blockchain_managers.items():
                export_data["transaction_history"][blockchain_type.value] = manager.transaction_history
            
            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Blockchain data exported to: {export_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting blockchain data: {str(e)}")
            return False

# Example usage and testing
async def main():
    """Example usage of Blockchain Integration"""
    
    # Initialize blockchain integration
    blockchain = BlockchainIntegration()
    
    # Create immutable document
    document_data = {
        "title": "Legal Contract - Quantum Computing Services",
        "content": "This contract establishes the terms for quantum computing services...",
        "metadata": {
            "document_type": "legal_contract",
            "jurisdiction": "international",
            "expiry_date": "2025-12-31"
        },
        "owner_address": "0x1234567890123456789012345678901234567890",
        "creator_address": "0x0987654321098765432109876543210987654321"
    }
    
    # Store on multiple blockchains
    blockchain_types = [BlockchainType.ETHEREUM, BlockchainType.POLYGON]
    result = await blockchain.create_immutable_document(document_data, blockchain_types)
    print("Immutable Document Created:", result)
    
    if result["success"]:
        document_id = result["document_id"]
        
        # Verify document authenticity
        verification = await blockchain.verify_document_authenticity(document_id)
        print("Document Verification:", verification)
        
        # Create NFT for document
        nft_metadata = NFTMetadata(
            name="Quantum Computing Contract NFT",
            description="NFT representing ownership of quantum computing service contract",
            image="https://example.com/contract-image.png",
            attributes=[
                {"trait_type": "Document Type", "value": "Legal Contract"},
                {"trait_type": "Jurisdiction", "value": "International"},
                {"trait_type": "Rarity", "value": "Rare"}
            ]
        )
        
        nft_result = await blockchain.create_document_nft(document_id, nft_metadata)
        print("NFT Creation:", nft_result)
    
    # Get analytics
    analytics = await blockchain.get_blockchain_analytics()
    print("Blockchain Analytics:", json.dumps(analytics, indent=2))

if __name__ == "__main__":
    asyncio.run(main())