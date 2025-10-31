"""
BUL Blockchain Document Verification
===================================

Blockchain integration for document verification, integrity, and authenticity.
"""

import asyncio
import json
import hashlib
import hmac
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, asdict
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
import requests
from web3 import Web3
from eth_account import Account
import ipfshttpclient
import cryptography
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.backends import default_backend

from ..utils import get_logger, get_cache_manager
from ..config import get_config
from ..security import get_encryption

logger = get_logger(__name__)

class VerificationStatus(str, Enum):
    """Document verification status"""
    PENDING = "pending"
    VERIFIED = "verified"
    FAILED = "failed"
    EXPIRED = "expired"
    REVOKED = "revoked"

class BlockchainNetwork(str, Enum):
    """Blockchain networks"""
    ETHEREUM_MAINNET = "ethereum_mainnet"
    ETHEREUM_TESTNET = "ethereum_testnet"
    POLYGON = "polygon"
    BSC = "bsc"
    ARBITRUM = "arbitrum"
    LOCAL = "local"

class DocumentType(str, Enum):
    """Document types for verification"""
    CONTRACT = "contract"
    CERTIFICATE = "certificate"
    REPORT = "report"
    PROPOSAL = "proposal"
    AGREEMENT = "agreement"
    POLICY = "policy"
    MANUAL = "manual"
    SPECIFICATION = "specification"

@dataclass
class DocumentHash:
    """Document hash information"""
    content_hash: str
    metadata_hash: str
    combined_hash: str
    algorithm: str
    timestamp: datetime
    block_number: Optional[int] = None
    transaction_hash: Optional[str] = None

@dataclass
class VerificationRecord:
    """Document verification record"""
    id: str
    document_id: str
    document_hash: DocumentHash
    verifier_address: str
    verification_status: VerificationStatus
    verification_timestamp: datetime
    blockchain_network: BlockchainNetwork
    transaction_hash: Optional[str] = None
    block_number: Optional[int] = None
    gas_used: Optional[int] = None
    verification_fee: Optional[float] = None
    metadata: Dict[str, Any] = None

@dataclass
class SmartContract:
    """Smart contract information"""
    address: str
    abi: List[Dict[str, Any]]
    network: BlockchainNetwork
    deployed_at: datetime
    version: str
    functions: List[str]

class BlockchainDocumentVerifier:
    """Blockchain-based document verification system"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.cache_manager = get_cache_manager()
        self.config = get_config()
        self.encryption = get_encryption()
        
        # Blockchain connections
        self.web3_connections: Dict[BlockchainNetwork, Web3] = {}
        self.contracts: Dict[BlockchainNetwork, SmartContract] = {}
        
        # IPFS client
        self.ipfs_client = None
        
        # Verification records
        self.verification_records: Dict[str, VerificationRecord] = {}
        
        # Initialize blockchain connections
        self._initialize_blockchain_connections()
        self._initialize_ipfs()
        self._deploy_contracts()
    
    def _initialize_blockchain_connections(self):
        """Initialize blockchain network connections"""
        try:
            # Ethereum Mainnet
            if self.config.blockchain.ethereum_mainnet_url:
                self.web3_connections[BlockchainNetwork.ETHEREUM_MAINNET] = Web3(
                    Web3.HTTPProvider(self.config.blockchain.ethereum_mainnet_url)
                )
            
            # Ethereum Testnet (Goerli)
            if self.config.blockchain.ethereum_testnet_url:
                self.web3_connections[BlockchainNetwork.ETHEREUM_TESTNET] = Web3(
                    Web3.HTTPProvider(self.config.blockchain.ethereum_testnet_url)
                )
            
            # Polygon
            if self.config.blockchain.polygon_url:
                self.web3_connections[BlockchainNetwork.POLYGON] = Web3(
                    Web3.HTTPProvider(self.config.blockchain.polygon_url)
                )
            
            # BSC
            if self.config.blockchain.bsc_url:
                self.web3_connections[BlockchainNetwork.BSC] = Web3(
                    Web3.HTTPProvider(self.config.blockchain.bsc_url)
                )
            
            # Local development
            self.web3_connections[BlockchainNetwork.LOCAL] = Web3(
                Web3.HTTPProvider("http://localhost:8545")
            )
            
            self.logger.info(f"Initialized {len(self.web3_connections)} blockchain connections")
        
        except Exception as e:
            self.logger.error(f"Failed to initialize blockchain connections: {e}")
    
    def _initialize_ipfs(self):
        """Initialize IPFS client"""
        try:
            self.ipfs_client = ipfshttpclient.connect(
                '/ip4/127.0.0.1/tcp/5001/http'
            )
            self.logger.info("IPFS client initialized successfully")
        except Exception as e:
            self.logger.warning(f"Failed to initialize IPFS client: {e}")
            self.ipfs_client = None
    
    def _deploy_contracts(self):
        """Deploy or connect to smart contracts"""
        try:
            # Document verification contract ABI
            contract_abi = [
                {
                    "inputs": [
                        {"internalType": "string", "name": "documentId", "type": "string"},
                        {"internalType": "bytes32", "name": "documentHash", "type": "bytes32"},
                        {"internalType": "string", "name": "metadata", "type": "string"}
                    ],
                    "name": "verifyDocument",
                    "outputs": [{"internalType": "bool", "name": "", "type": "bool"}],
                    "stateMutability": "nonpayable",
                    "type": "function"
                },
                {
                    "inputs": [
                        {"internalType": "string", "name": "documentId", "type": "string"}
                    ],
                    "name": "getVerificationStatus",
                    "outputs": [
                        {"internalType": "bool", "name": "isVerified", "type": "bool"},
                        {"internalType": "uint256", "name": "timestamp", "type": "uint256"},
                        {"internalType": "address", "name": "verifier", "type": "address"}
                    ],
                    "stateMutability": "view",
                    "type": "function"
                },
                {
                    "inputs": [
                        {"internalType": "string", "name": "documentId", "type": "string"}
                    ],
                    "name": "revokeVerification",
                    "outputs": [{"internalType": "bool", "name": "", "type": "bool"}],
                    "stateMutability": "nonpayable",
                    "type": "function"
                },
                {
                    "anonymous": False,
                    "inputs": [
                        {"indexed": True, "internalType": "string", "name": "documentId", "type": "string"},
                        {"indexed": True, "internalType": "bytes32", "name": "documentHash", "type": "bytes32"},
                        {"indexed": False, "internalType": "address", "name": "verifier", "type": "address"},
                        {"indexed": False, "internalType": "uint256", "name": "timestamp", "type": "uint256"}
                    ],
                    "name": "DocumentVerified",
                    "type": "event"
                }
            ]
            
            # Deploy contracts on each network
            for network, web3 in self.web3_connections.items():
                try:
                    if network == BlockchainNetwork.LOCAL:
                        # For local development, use a mock contract
                        contract_address = "0x1234567890123456789012345678901234567890"
                    else:
                        # In production, deploy actual contracts
                        contract_address = self._deploy_contract(web3, contract_abi, network)
                    
                    self.contracts[network] = SmartContract(
                        address=contract_address,
                        abi=contract_abi,
                        network=network,
                        deployed_at=datetime.now(),
                        version="1.0.0",
                        functions=["verifyDocument", "getVerificationStatus", "revokeVerification"]
                    )
                    
                    self.logger.info(f"Contract deployed on {network.value}: {contract_address}")
                
                except Exception as e:
                    self.logger.error(f"Failed to deploy contract on {network.value}: {e}")
        
        except Exception as e:
            self.logger.error(f"Failed to deploy contracts: {e}")
    
    def _deploy_contract(self, web3: Web3, abi: List[Dict[str, Any]], network: BlockchainNetwork) -> str:
        """Deploy smart contract to blockchain"""
        try:
            # Contract bytecode (simplified - in production, use actual compiled contract)
            bytecode = "0x608060405234801561001057600080fd5b50600436106100415760003560e01c8063..."
            
            # Get account for deployment
            account = Account.from_key(self.config.blockchain.private_key)
            
            # Create contract
            contract = web3.eth.contract(abi=abi, bytecode=bytecode)
            
            # Deploy contract
            transaction = contract.constructor().build_transaction({
                'from': account.address,
                'gas': 2000000,
                'gasPrice': web3.eth.gas_price,
                'nonce': web3.eth.get_transaction_count(account.address)
            })
            
            # Sign and send transaction
            signed_txn = web3.eth.account.sign_transaction(transaction, self.config.blockchain.private_key)
            tx_hash = web3.eth.send_raw_transaction(signed_txn.rawTransaction)
            
            # Wait for transaction receipt
            receipt = web3.eth.wait_for_transaction_receipt(tx_hash)
            
            return receipt.contractAddress
        
        except Exception as e:
            self.logger.error(f"Failed to deploy contract: {e}")
            return "0x0000000000000000000000000000000000000000"
    
    async def generate_document_hash(
        self,
        content: str,
        metadata: Dict[str, Any]
    ) -> DocumentHash:
        """Generate cryptographic hash for document"""
        try:
            # Generate content hash
            content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
            
            # Generate metadata hash
            metadata_json = json.dumps(metadata, sort_keys=True)
            metadata_hash = hashlib.sha256(metadata_json.encode('utf-8')).hexdigest()
            
            # Generate combined hash
            combined_data = f"{content_hash}{metadata_hash}"
            combined_hash = hashlib.sha256(combined_data.encode('utf-8')).hexdigest()
            
            return DocumentHash(
                content_hash=content_hash,
                metadata_hash=metadata_hash,
                combined_hash=combined_hash,
                algorithm="SHA-256",
                timestamp=datetime.now()
            )
        
        except Exception as e:
            self.logger.error(f"Error generating document hash: {e}")
            raise
    
    async def store_document_ipfs(
        self,
        content: str,
        metadata: Dict[str, Any]
    ) -> str:
        """Store document in IPFS"""
        try:
            if not self.ipfs_client:
                raise Exception("IPFS client not available")
            
            # Prepare document data
            document_data = {
                "content": content,
                "metadata": metadata,
                "timestamp": datetime.now().isoformat(),
                "version": "1.0"
            }
            
            # Add to IPFS
            result = self.ipfs_client.add_json(document_data)
            ipfs_hash = result['Hash']
            
            self.logger.info(f"Document stored in IPFS: {ipfs_hash}")
            return ipfs_hash
        
        except Exception as e:
            self.logger.error(f"Error storing document in IPFS: {e}")
            raise
    
    async def verify_document_blockchain(
        self,
        document_id: str,
        document_hash: DocumentHash,
        network: BlockchainNetwork = BlockchainNetwork.ETHEREUM_TESTNET
    ) -> VerificationRecord:
        """Verify document on blockchain"""
        try:
            if network not in self.web3_connections:
                raise Exception(f"Network {network.value} not available")
            
            web3 = self.web3_connections[network]
            contract = self.contracts[network]
            
            # Get account
            account = Account.from_key(self.config.blockchain.private_key)
            
            # Create contract instance
            contract_instance = web3.eth.contract(
                address=contract.address,
                abi=contract.abi
            )
            
            # Prepare metadata
            metadata = {
                "document_id": document_id,
                "content_hash": document_hash.content_hash,
                "metadata_hash": document_hash.metadata_hash,
                "algorithm": document_hash.algorithm,
                "timestamp": document_hash.timestamp.isoformat()
            }
            
            # Call verifyDocument function
            transaction = contract_instance.functions.verifyDocument(
                document_id,
                web3.toBytes(hexstr=document_hash.combined_hash),
                json.dumps(metadata)
            ).build_transaction({
                'from': account.address,
                'gas': 200000,
                'gasPrice': web3.eth.gas_price,
                'nonce': web3.eth.get_transaction_count(account.address)
            })
            
            # Sign and send transaction
            signed_txn = web3.eth.account.sign_transaction(transaction, self.config.blockchain.private_key)
            tx_hash = web3.eth.send_raw_transaction(signed_txn.rawTransaction)
            
            # Wait for transaction receipt
            receipt = web3.eth.wait_for_transaction_receipt(tx_hash)
            
            # Create verification record
            verification_record = VerificationRecord(
                id=f"ver_{document_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                document_id=document_id,
                document_hash=document_hash,
                verifier_address=account.address,
                verification_status=VerificationStatus.VERIFIED,
                verification_timestamp=datetime.now(),
                blockchain_network=network,
                transaction_hash=tx_hash.hex(),
                block_number=receipt.blockNumber,
                gas_used=receipt.gasUsed,
                verification_fee=float(receipt.gasUsed * web3.eth.gas_price) / 1e18,
                metadata=metadata
            )
            
            # Store verification record
            self.verification_records[verification_record.id] = verification_record
            
            # Update document hash with blockchain info
            document_hash.block_number = receipt.blockNumber
            document_hash.transaction_hash = tx_hash.hex()
            
            self.logger.info(f"Document verified on blockchain: {document_id}")
            return verification_record
        
        except Exception as e:
            self.logger.error(f"Error verifying document on blockchain: {e}")
            raise
    
    async def check_verification_status(
        self,
        document_id: str,
        network: BlockchainNetwork = BlockchainNetwork.ETHEREUM_TESTNET
    ) -> Optional[VerificationRecord]:
        """Check document verification status on blockchain"""
        try:
            if network not in self.web3_connections:
                return None
            
            web3 = self.web3_connections[network]
            contract = self.contracts[network]
            
            # Create contract instance
            contract_instance = web3.eth.contract(
                address=contract.address,
                abi=contract.abi
            )
            
            # Call getVerificationStatus function
            result = contract_instance.functions.getVerificationStatus(document_id).call()
            
            is_verified, timestamp, verifier = result
            
            if is_verified:
                # Find verification record
                for record in self.verification_records.values():
                    if record.document_id == document_id and record.blockchain_network == network:
                        return record
            
            return None
        
        except Exception as e:
            self.logger.error(f"Error checking verification status: {e}")
            return None
    
    async def revoke_verification(
        self,
        document_id: str,
        network: BlockchainNetwork = BlockchainNetwork.ETHEREUM_TESTNET
    ) -> bool:
        """Revoke document verification on blockchain"""
        try:
            if network not in self.web3_connections:
                return False
            
            web3 = self.web3_connections[network]
            contract = self.contracts[network]
            
            # Get account
            account = Account.from_key(self.config.blockchain.private_key)
            
            # Create contract instance
            contract_instance = web3.eth.contract(
                address=contract.address,
                abi=contract.abi
            )
            
            # Call revokeVerification function
            transaction = contract_instance.functions.revokeVerification(document_id).build_transaction({
                'from': account.address,
                'gas': 100000,
                'gasPrice': web3.eth.gas_price,
                'nonce': web3.eth.get_transaction_count(account.address)
            })
            
            # Sign and send transaction
            signed_txn = web3.eth.account.sign_transaction(transaction, self.config.blockchain.private_key)
            tx_hash = web3.eth.send_raw_transaction(signed_txn.rawTransaction)
            
            # Wait for transaction receipt
            receipt = web3.eth.wait_for_transaction_receipt(tx_hash)
            
            # Update verification record
            for record in self.verification_records.values():
                if record.document_id == document_id and record.blockchain_network == network:
                    record.verification_status = VerificationStatus.REVOKED
                    break
            
            self.logger.info(f"Document verification revoked: {document_id}")
            return True
        
        except Exception as e:
            self.logger.error(f"Error revoking verification: {e}")
            return False
    
    async def verify_document_integrity(
        self,
        document_id: str,
        content: str,
        metadata: Dict[str, Any],
        network: BlockchainNetwork = BlockchainNetwork.ETHEREUM_TESTNET
    ) -> Dict[str, Any]:
        """Complete document verification process"""
        try:
            # Generate document hash
            document_hash = await self.generate_document_hash(content, metadata)
            
            # Store in IPFS
            ipfs_hash = await self.store_document_ipfs(content, metadata)
            
            # Verify on blockchain
            verification_record = await self.verify_document_blockchain(
                document_id, document_hash, network
            )
            
            return {
                "document_id": document_id,
                "verification_id": verification_record.id,
                "document_hash": asdict(document_hash),
                "ipfs_hash": ipfs_hash,
                "verification_record": asdict(verification_record),
                "verification_url": f"https://etherscan.io/tx/{verification_record.transaction_hash}",
                "success": True
            }
        
        except Exception as e:
            self.logger.error(f"Error in document verification: {e}")
            return {
                "document_id": document_id,
                "error": str(e),
                "success": False
            }
    
    async def get_verification_history(
        self,
        document_id: Optional[str] = None,
        network: Optional[BlockchainNetwork] = None
    ) -> List[VerificationRecord]:
        """Get verification history"""
        try:
            records = list(self.verification_records.values())
            
            # Filter by document ID
            if document_id:
                records = [r for r in records if r.document_id == document_id]
            
            # Filter by network
            if network:
                records = [r for r in records if r.blockchain_network == network]
            
            # Sort by timestamp
            records.sort(key=lambda x: x.verification_timestamp, reverse=True)
            
            return records
        
        except Exception as e:
            self.logger.error(f"Error getting verification history: {e}")
            return []
    
    async def get_blockchain_stats(self) -> Dict[str, Any]:
        """Get blockchain verification statistics"""
        try:
            total_verifications = len(self.verification_records)
            
            # Count by network
            network_counts = {}
            for record in self.verification_records.values():
                network = record.blockchain_network.value
                network_counts[network] = network_counts.get(network, 0) + 1
            
            # Count by status
            status_counts = {}
            for record in self.verification_records.values():
                status = record.verification_status.value
                status_counts[status] = status_counts.get(status, 0) + 1
            
            # Calculate total fees
            total_fees = sum(
                record.verification_fee or 0
                for record in self.verification_records.values()
            )
            
            return {
                "total_verifications": total_verifications,
                "network_distribution": network_counts,
                "status_distribution": status_counts,
                "total_fees_paid": round(total_fees, 6),
                "available_networks": [network.value for network in self.web3_connections.keys()],
                "contract_addresses": {
                    network.value: contract.address
                    for network, contract in self.contracts.items()
                }
            }
        
        except Exception as e:
            self.logger.error(f"Error getting blockchain stats: {e}")
            return {}

# Global blockchain verifier
_blockchain_verifier: Optional[BlockchainDocumentVerifier] = None

def get_blockchain_verifier() -> BlockchainDocumentVerifier:
    """Get the global blockchain verifier"""
    global _blockchain_verifier
    if _blockchain_verifier is None:
        _blockchain_verifier = BlockchainDocumentVerifier()
    return _blockchain_verifier

# Blockchain router
blockchain_router = APIRouter(prefix="/blockchain", tags=["Blockchain Verification"])

@blockchain_router.post("/verify-document")
async def verify_document_endpoint(
    document_id: str = Field(..., description="Document ID"),
    content: str = Field(..., description="Document content"),
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata"),
    network: BlockchainNetwork = Field(BlockchainNetwork.ETHEREUM_TESTNET, description="Blockchain network")
):
    """Verify document on blockchain"""
    try:
        verifier = get_blockchain_verifier()
        result = await verifier.verify_document_integrity(document_id, content, metadata, network)
        return result
    
    except Exception as e:
        logger.error(f"Error verifying document: {e}")
        raise HTTPException(status_code=500, detail="Failed to verify document")

@blockchain_router.get("/verification-status/{document_id}")
async def get_verification_status_endpoint(
    document_id: str,
    network: BlockchainNetwork = BlockchainNetwork.ETHEREUM_TESTNET
):
    """Get document verification status"""
    try:
        verifier = get_blockchain_verifier()
        status = await verifier.check_verification_status(document_id, network)
        
        if status:
            return {"verification": asdict(status), "verified": True}
        else:
            return {"verified": False, "message": "Document not verified"}
    
    except Exception as e:
        logger.error(f"Error getting verification status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get verification status")

@blockchain_router.post("/revoke-verification/{document_id}")
async def revoke_verification_endpoint(
    document_id: str,
    network: BlockchainNetwork = BlockchainNetwork.ETHEREUM_TESTNET
):
    """Revoke document verification"""
    try:
        verifier = get_blockchain_verifier()
        success = await verifier.revoke_verification(document_id, network)
        
        if success:
            return {"success": True, "message": "Verification revoked successfully"}
        else:
            return {"success": False, "message": "Failed to revoke verification"}
    
    except Exception as e:
        logger.error(f"Error revoking verification: {e}")
        raise HTTPException(status_code=500, detail="Failed to revoke verification")

@blockchain_router.get("/verification-history")
async def get_verification_history_endpoint(
    document_id: Optional[str] = None,
    network: Optional[BlockchainNetwork] = None
):
    """Get verification history"""
    try:
        verifier = get_blockchain_verifier()
        history = await verifier.get_verification_history(document_id, network)
        return {"verifications": [asdict(record) for record in history]}
    
    except Exception as e:
        logger.error(f"Error getting verification history: {e}")
        raise HTTPException(status_code=500, detail="Failed to get verification history")

@blockchain_router.get("/stats")
async def get_blockchain_stats_endpoint():
    """Get blockchain verification statistics"""
    try:
        verifier = get_blockchain_verifier()
        stats = await verifier.get_blockchain_stats()
        return {"stats": stats}
    
    except Exception as e:
        logger.error(f"Error getting blockchain stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get blockchain stats")

@blockchain_router.post("/generate-hash")
async def generate_document_hash_endpoint(
    content: str = Field(..., description="Document content"),
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")
):
    """Generate document hash"""
    try:
        verifier = get_blockchain_verifier()
        document_hash = await verifier.generate_document_hash(content, metadata)
        return {"hash": asdict(document_hash)}
    
    except Exception as e:
        logger.error(f"Error generating document hash: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate document hash")


