"""
Blockchain Document Verifier for Export IA
==========================================

Advanced blockchain-based document verification system that ensures
document integrity, authenticity, and immutability using distributed
ledger technology.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import uuid
import os
import hashlib
import base64
from pathlib import Path
import ecdsa
from ecdsa import SigningKey, VerifyingKey, SECP256k1
import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.backends import default_backend

logger = logging.getLogger(__name__)

class BlockchainType(Enum):
    """Types of blockchain networks."""
    ETHEREUM = "ethereum"
    BITCOIN = "bitcoin"
    POLYGON = "polygon"
    BSC = "bsc"
    ARBITRUM = "arbitrum"
    OPTIMISM = "optimism"
    LOCAL = "local"
    MOCK = "mock"

class VerificationStatus(Enum):
    """Document verification status."""
    PENDING = "pending"
    VERIFIED = "verified"
    FAILED = "failed"
    EXPIRED = "expired"
    REVOKED = "revoked"

class DocumentIntegrityLevel(Enum):
    """Levels of document integrity verification."""
    BASIC = "basic"          # Hash verification only
    STANDARD = "standard"    # Hash + signature verification
    ADVANCED = "advanced"    # Hash + signature + timestamp verification
    ENTERPRISE = "enterprise" # Full blockchain verification with smart contracts

@dataclass
class DocumentHash:
    """Document hash information."""
    content_hash: str
    metadata_hash: str
    combined_hash: str
    algorithm: str = "SHA-256"
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class DigitalSignature:
    """Digital signature information."""
    signature: str
    public_key: str
    algorithm: str = "ECDSA"
    timestamp: datetime = field(default_factory=datetime.now)
    signer_id: str = ""

@dataclass
class BlockchainTransaction:
    """Blockchain transaction information."""
    transaction_hash: str
    block_number: int
    block_hash: str
    gas_used: int
    gas_price: int
    timestamp: datetime
    network: BlockchainType
    contract_address: Optional[str] = None

@dataclass
class DocumentVerification:
    """Document verification record."""
    id: str
    document_id: str
    document_hash: DocumentHash
    digital_signature: Optional[DigitalSignature]
    blockchain_transaction: Optional[BlockchainTransaction]
    verification_status: VerificationStatus
    integrity_level: DocumentIntegrityLevel
    created_at: datetime
    verified_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BlockchainConfig:
    """Blockchain configuration."""
    network: BlockchainType = BlockchainType.MOCK
    rpc_url: str = ""
    private_key: str = ""
    contract_address: str = ""
    gas_limit: int = 100000
    gas_price: int = 20
    verification_timeout: int = 300  # 5 minutes
    integrity_level: DocumentIntegrityLevel = DocumentIntegrityLevel.STANDARD

class BlockchainDocumentVerifier:
    """Blockchain-based document verification system."""
    
    def __init__(self, config: Optional[BlockchainConfig] = None):
        self.config = config or BlockchainConfig()
        self.verifications: Dict[str, DocumentVerification] = {}
        self.key_pairs: Dict[str, Tuple[str, str]] = {}  # signer_id -> (private_key, public_key)
        
        # Initialize blockchain connection
        self._initialize_blockchain()
        
        # Generate default key pair
        self._generate_key_pair("default_signer")
        
        logger.info(f"Blockchain Document Verifier initialized for {self.config.network.value}")
    
    def _initialize_blockchain(self):
        """Initialize blockchain connection."""
        try:
            if self.config.network == BlockchainType.MOCK:
                logger.info("Using mock blockchain for testing")
            elif self.config.network == BlockchainType.LOCAL:
                logger.info("Connecting to local blockchain")
            else:
                logger.info(f"Connecting to {self.config.network.value} network")
                # In production, would initialize actual blockchain connection
                
        except Exception as e:
            logger.error(f"Failed to initialize blockchain: {e}")
            # Fallback to mock blockchain
            self.config.network = BlockchainType.MOCK
    
    def _generate_key_pair(self, signer_id: str) -> Tuple[str, str]:
        """Generate ECDSA key pair for signing."""
        try:
            # Generate private key
            private_key = SigningKey.generate(curve=SECP256k1)
            public_key = private_key.get_verifying_key()
            
            # Convert to strings
            private_key_str = private_key.to_string().hex()
            public_key_str = public_key.to_string().hex()
            
            # Store key pair
            self.key_pairs[signer_id] = (private_key_str, public_key_str)
            
            logger.info(f"Generated key pair for signer: {signer_id}")
            return private_key_str, public_key_str
            
        except Exception as e:
            logger.error(f"Failed to generate key pair: {e}")
            raise
    
    def _calculate_document_hash(self, content: str, metadata: Dict[str, Any]) -> DocumentHash:
        """Calculate document hashes."""
        try:
            # Content hash
            content_bytes = content.encode('utf-8')
            content_hash = hashlib.sha256(content_bytes).hexdigest()
            
            # Metadata hash
            metadata_json = json.dumps(metadata, sort_keys=True)
            metadata_bytes = metadata_json.encode('utf-8')
            metadata_hash = hashlib.sha256(metadata_bytes).hexdigest()
            
            # Combined hash
            combined_data = f"{content_hash}{metadata_hash}".encode('utf-8')
            combined_hash = hashlib.sha256(combined_data).hexdigest()
            
            return DocumentHash(
                content_hash=content_hash,
                metadata_hash=metadata_hash,
                combined_hash=combined_hash
            )
            
        except Exception as e:
            logger.error(f"Failed to calculate document hash: {e}")
            raise
    
    def _sign_document(self, document_hash: DocumentHash, signer_id: str) -> DigitalSignature:
        """Sign document hash with digital signature."""
        try:
            if signer_id not in self.key_pairs:
                self._generate_key_pair(signer_id)
            
            private_key_str, public_key_str = self.key_pairs[signer_id]
            
            # Create signing key
            private_key = SigningKey.from_string(bytes.fromhex(private_key_str), curve=SECP256k1)
            
            # Sign the combined hash
            signature_bytes = private_key.sign(document_hash.combined_hash.encode('utf-8'))
            signature = signature_bytes.hex()
            
            return DigitalSignature(
                signature=signature,
                public_key=public_key_str,
                signer_id=signer_id
            )
            
        except Exception as e:
            logger.error(f"Failed to sign document: {e}")
            raise
    
    def _verify_signature(self, document_hash: DocumentHash, signature: DigitalSignature) -> bool:
        """Verify digital signature."""
        try:
            # Create verifying key
            public_key = VerifyingKey.from_string(
                bytes.fromhex(signature.public_key), 
                curve=SECP256k1
            )
            
            # Verify signature
            is_valid = public_key.verify(
                bytes.fromhex(signature.signature),
                document_hash.combined_hash.encode('utf-8')
            )
            
            return is_valid
            
        except Exception as e:
            logger.error(f"Signature verification failed: {e}")
            return False
    
    async def _submit_to_blockchain(self, document_hash: DocumentHash, signature: DigitalSignature) -> Optional[BlockchainTransaction]:
        """Submit document verification to blockchain."""
        try:
            if self.config.network == BlockchainType.MOCK:
                # Mock blockchain transaction
                return BlockchainTransaction(
                    transaction_hash=f"0x{hashlib.sha256(f'{document_hash.combined_hash}{signature.signature}'.encode()).hexdigest()}",
                    block_number=12345,
                    block_hash=f"0x{hashlib.sha256(f'block_{12345}'.encode()).hexdigest()}",
                    gas_used=21000,
                    gas_price=self.config.gas_price,
                    timestamp=datetime.now(),
                    network=self.config.network,
                    contract_address=self.config.contract_address
                )
            
            elif self.config.network == BlockchainType.LOCAL:
                # Local blockchain submission (simplified)
                return BlockchainTransaction(
                    transaction_hash=f"0x{hashlib.sha256(f'{document_hash.combined_hash}{signature.signature}'.encode()).hexdigest()}",
                    block_number=1,
                    block_hash=f"0x{hashlib.sha256(f'local_block_{1}'.encode()).hexdigest()}",
                    gas_used=50000,
                    gas_price=self.config.gas_price,
                    timestamp=datetime.now(),
                    network=self.config.network,
                    contract_address=self.config.contract_address
                )
            
            else:
                # Real blockchain submission would go here
                # This would involve actual smart contract interaction
                logger.warning(f"Real blockchain submission not implemented for {self.config.network.value}")
                return None
                
        except Exception as e:
            logger.error(f"Blockchain submission failed: {e}")
            return None
    
    async def verify_document(
        self,
        document_id: str,
        content: str,
        metadata: Dict[str, Any],
        signer_id: str = "default_signer",
        integrity_level: Optional[DocumentIntegrityLevel] = None
    ) -> DocumentVerification:
        """Verify document integrity using blockchain."""
        
        verification_id = str(uuid.uuid4())
        integrity_level = integrity_level or self.config.integrity_level
        
        logger.info(f"Starting document verification: {document_id}")
        logger.info(f"Integrity level: {integrity_level.value}")
        
        try:
            # Step 1: Calculate document hash
            document_hash = self._calculate_document_hash(content, metadata)
            
            # Step 2: Create digital signature
            digital_signature = None
            if integrity_level in [DocumentIntegrityLevel.STANDARD, DocumentIntegrityLevel.ADVANCED, DocumentIntegrityLevel.ENTERPRISE]:
                digital_signature = self._sign_document(document_hash, signer_id)
            
            # Step 3: Submit to blockchain
            blockchain_transaction = None
            if integrity_level in [DocumentIntegrityLevel.ADVANCED, DocumentIntegrityLevel.ENTERPRISE]:
                blockchain_transaction = await self._submit_to_blockchain(document_hash, digital_signature)
            
            # Step 4: Create verification record
            verification = DocumentVerification(
                id=verification_id,
                document_id=document_id,
                document_hash=document_hash,
                digital_signature=digital_signature,
                blockchain_transaction=blockchain_transaction,
                verification_status=VerificationStatus.VERIFIED,
                integrity_level=integrity_level,
                created_at=datetime.now(),
                verified_at=datetime.now(),
                expires_at=datetime.now() + timedelta(days=365),  # 1 year validity
                metadata={
                    "content_length": len(content),
                    "metadata_keys": list(metadata.keys()),
                    "signer_id": signer_id,
                    "network": self.config.network.value
                }
            )
            
            # Store verification
            self.verifications[verification_id] = verification
            
            logger.info(f"Document verification completed: {verification_id}")
            logger.info(f"Transaction hash: {blockchain_transaction.transaction_hash if blockchain_transaction else 'N/A'}")
            
            return verification
            
        except Exception as e:
            logger.error(f"Document verification failed: {e}")
            
            # Create failed verification record
            failed_verification = DocumentVerification(
                id=verification_id,
                document_id=document_id,
                document_hash=DocumentHash("", "", ""),
                verification_status=VerificationStatus.FAILED,
                integrity_level=integrity_level,
                created_at=datetime.now(),
                metadata={"error": str(e)}
            )
            
            self.verifications[verification_id] = failed_verification
            return failed_verification
    
    async def verify_document_integrity(
        self,
        verification_id: str,
        content: str,
        metadata: Dict[str, Any]
    ) -> bool:
        """Verify document integrity against stored verification."""
        
        if verification_id not in self.verifications:
            logger.error(f"Verification not found: {verification_id}")
            return False
        
        verification = self.verifications[verification_id]
        
        if verification.verification_status != VerificationStatus.VERIFIED:
            logger.error(f"Verification not in verified status: {verification.verification_status}")
            return False
        
        # Check if verification has expired
        if verification.expires_at and datetime.now() > verification.expires_at:
            logger.error("Verification has expired")
            verification.verification_status = VerificationStatus.EXPIRED
            return False
        
        try:
            # Recalculate document hash
            current_hash = self._calculate_document_hash(content, metadata)
            
            # Compare hashes
            if current_hash.combined_hash != verification.document_hash.combined_hash:
                logger.error("Document hash mismatch - document has been modified")
                return False
            
            # Verify digital signature if present
            if verification.digital_signature:
                if not self._verify_signature(current_hash, verification.digital_signature):
                    logger.error("Digital signature verification failed")
                    return False
            
            logger.info("Document integrity verification successful")
            return True
            
        except Exception as e:
            logger.error(f"Integrity verification failed: {e}")
            return False
    
    def get_verification(self, verification_id: str) -> Optional[DocumentVerification]:
        """Get verification record by ID."""
        return self.verifications.get(verification_id)
    
    def list_verifications(self, document_id: Optional[str] = None) -> List[DocumentVerification]:
        """List verification records."""
        verifications = list(self.verifications.values())
        
        if document_id:
            verifications = [v for v in verifications if v.document_id == document_id]
        
        return sorted(verifications, key=lambda x: x.created_at, reverse=True)
    
    def revoke_verification(self, verification_id: str, reason: str = "") -> bool:
        """Revoke a verification."""
        if verification_id not in self.verifications:
            return False
        
        verification = self.verifications[verification_id]
        verification.verification_status = VerificationStatus.REVOKED
        verification.metadata["revocation_reason"] = reason
        verification.metadata["revoked_at"] = datetime.now().isoformat()
        
        logger.info(f"Verification revoked: {verification_id}, reason: {reason}")
        return True
    
    def get_verification_statistics(self) -> Dict[str, Any]:
        """Get verification statistics."""
        verifications = list(self.verifications.values())
        
        if not verifications:
            return {"message": "No verifications found"}
        
        status_counts = {}
        for status in VerificationStatus:
            status_counts[status.value] = len([v for v in verifications if v.verification_status == status])
        
        integrity_counts = {}
        for level in DocumentIntegrityLevel:
            integrity_counts[level.value] = len([v for v in verifications if v.integrity_level == level])
        
        return {
            "total_verifications": len(verifications),
            "status_distribution": status_counts,
            "integrity_level_distribution": integrity_counts,
            "verified_documents": len([v for v in verifications if v.verification_status == VerificationStatus.VERIFIED]),
            "expired_verifications": len([v for v in verifications if v.verification_status == VerificationStatus.EXPIRED]),
            "revoked_verifications": len([v for v in verifications if v.verification_status == VerificationStatus.REVOKED]),
            "network": self.config.network.value,
            "active_key_pairs": len(self.key_pairs)
        }
    
    def export_verification_certificate(self, verification_id: str) -> Optional[Dict[str, Any]]:
        """Export verification certificate."""
        verification = self.get_verification(verification_id)
        if not verification or verification.verification_status != VerificationStatus.VERIFIED:
            return None
        
        certificate = {
            "certificate_id": verification_id,
            "document_id": verification.document_id,
            "verification_status": verification.verification_status.value,
            "integrity_level": verification.integrity_level.value,
            "created_at": verification.created_at.isoformat(),
            "verified_at": verification.verified_at.isoformat() if verification.verified_at else None,
            "expires_at": verification.expires_at.isoformat() if verification.expires_at else None,
            "document_hash": {
                "content_hash": verification.document_hash.content_hash,
                "metadata_hash": verification.document_hash.metadata_hash,
                "combined_hash": verification.document_hash.combined_hash,
                "algorithm": verification.document_hash.algorithm
            },
            "digital_signature": {
                "signature": verification.digital_signature.signature,
                "public_key": verification.digital_signature.public_key,
                "algorithm": verification.digital_signature.algorithm,
                "signer_id": verification.digital_signature.signer_id
            } if verification.digital_signature else None,
            "blockchain_transaction": {
                "transaction_hash": verification.blockchain_transaction.transaction_hash,
                "block_number": verification.blockchain_transaction.block_number,
                "block_hash": verification.blockchain_transaction.block_hash,
                "network": verification.blockchain_transaction.network.value,
                "timestamp": verification.blockchain_transaction.timestamp.isoformat()
            } if verification.blockchain_transaction else None,
            "metadata": verification.metadata
        }
        
        return certificate

# Global blockchain verifier instance
_global_blockchain_verifier: Optional[BlockchainDocumentVerifier] = None

def get_global_blockchain_verifier() -> BlockchainDocumentVerifier:
    """Get the global blockchain verifier instance."""
    global _global_blockchain_verifier
    if _global_blockchain_verifier is None:
        _global_blockchain_verifier = BlockchainDocumentVerifier()
    return _global_blockchain_verifier



























