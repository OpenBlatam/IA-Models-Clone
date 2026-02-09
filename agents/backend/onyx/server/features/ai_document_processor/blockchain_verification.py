"""
Blockchain Document Verification and Smart Contracts Module
"""

import asyncio
import logging
import time
import hashlib
import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import uuid
from pathlib import Path

from web3 import Web3
from eth_account import Account
import ipfshttpclient
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.backends import default_backend

from config import settings
from models import ProcessingStatus

logger = logging.getLogger(__name__)


class BlockchainVerification:
    """Blockchain Document Verification and Smart Contracts Engine"""
    
    def __init__(self):
        self.web3 = None
        self.contracts = {}
        self.ipfs_client = None
        self.initialized = False
        
    async def initialize(self):
        """Initialize blockchain verification system"""
        if self.initialized:
            return
            
        try:
            logger.info("Initializing Blockchain Verification System...")
            
            # Initialize Web3 connection
            if hasattr(settings, 'ethereum_rpc_url') and settings.ethereum_rpc_url:
                self.web3 = Web3(Web3.HTTPProvider(settings.ethereum_rpc_url))
                
                if self.web3.is_connected():
                    logger.info("Connected to Ethereum network")
                else:
                    logger.warning("Failed to connect to Ethereum network")
            
            # Initialize IPFS client
            if hasattr(settings, 'ipfs_url') and settings.ipfs_url:
                try:
                    self.ipfs_client = ipfshttpclient.connect(settings.ipfs_url)
                    logger.info("Connected to IPFS network")
                except Exception as e:
                    logger.warning(f"Failed to connect to IPFS: {e}")
            
            # Load smart contracts
            await self._load_smart_contracts()
            
            self.initialized = True
            logger.info("Blockchain Verification System initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing blockchain verification: {e}")
            raise
    
    async def _load_smart_contracts(self):
        """Load smart contracts for document verification"""
        try:
            # Document Verification Contract ABI
            document_verification_abi = [
                {
                    "inputs": [
                        {"name": "documentHash", "type": "bytes32"},
                        {"name": "owner", "type": "address"},
                        {"name": "metadata", "type": "string"}
                    ],
                    "name": "registerDocument",
                    "outputs": [{"name": "", "type": "bool"}],
                    "stateMutability": "nonpayable",
                    "type": "function"
                },
                {
                    "inputs": [{"name": "documentHash", "type": "bytes32"}],
                    "name": "verifyDocument",
                    "outputs": [
                        {"name": "isValid", "type": "bool"},
                        {"name": "owner", "type": "address"},
                        {"name": "timestamp", "type": "uint256"}
                    ],
                    "stateMutability": "view",
                    "type": "function"
                },
                {
                    "inputs": [{"name": "documentHash", "type": "bytes32"}],
                    "name": "getDocumentInfo",
                    "outputs": [
                        {"name": "owner", "type": "address"},
                        {"name": "timestamp", "type": "uint256"},
                        {"name": "metadata", "type": "string"}
                    ],
                    "stateMutability": "view",
                    "type": "function"
                }
            ]
            
            # Load contract if address is provided
            if hasattr(settings, 'document_verification_contract_address') and settings.document_verification_contract_address:
                self.contracts['document_verification'] = self.web3.eth.contract(
                    address=settings.document_verification_contract_address,
                    abi=document_verification_abi
                )
                logger.info("Document verification contract loaded")
            
        except Exception as e:
            logger.error(f"Error loading smart contracts: {e}")
    
    async def register_document_on_blockchain(self, document_path: str, 
                                            owner_address: str,
                                            metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Register document on blockchain for verification"""
        try:
            if not self.initialized:
                await self.initialize()
            
            # Calculate document hash
            document_hash = await self._calculate_document_hash(document_path)
            
            # Upload document to IPFS if available
            ipfs_hash = None
            if self.ipfs_client:
                ipfs_hash = await self._upload_to_ipfs(document_path)
            
            # Prepare metadata
            if metadata is None:
                metadata = {}
            
            metadata.update({
                "ipfs_hash": ipfs_hash,
                "timestamp": datetime.now().isoformat(),
                "file_size": Path(document_path).stat().st_size
            })
            
            # Register on blockchain
            if 'document_verification' in self.contracts:
                tx_hash = await self._register_on_blockchain(
                    document_hash, owner_address, json.dumps(metadata)
                )
                
                return {
                    "document_hash": document_hash.hex(),
                    "ipfs_hash": ipfs_hash,
                    "transaction_hash": tx_hash,
                    "owner_address": owner_address,
                    "metadata": metadata,
                    "status": "registered"
                }
            else:
                return {
                    "document_hash": document_hash.hex(),
                    "ipfs_hash": ipfs_hash,
                    "owner_address": owner_address,
                    "metadata": metadata,
                    "status": "registered_offline"
                }
                
        except Exception as e:
            logger.error(f"Error registering document on blockchain: {e}")
            return {"error": str(e), "status": "failed"}
    
    async def verify_document_on_blockchain(self, document_path: str) -> Dict[str, Any]:
        """Verify document authenticity on blockchain"""
        try:
            if not self.initialized:
                await self.initialize()
            
            # Calculate document hash
            document_hash = await self._calculate_document_hash(document_path)
            
            # Verify on blockchain
            if 'document_verification' in self.contracts:
                verification_result = await self._verify_on_blockchain(document_hash)
                
                return {
                    "document_hash": document_hash.hex(),
                    "is_valid": verification_result["isValid"],
                    "owner": verification_result["owner"],
                    "timestamp": verification_result["timestamp"],
                    "status": "verified"
                }
            else:
                return {
                    "document_hash": document_hash.hex(),
                    "is_valid": False,
                    "status": "verification_unavailable"
                }
                
        except Exception as e:
            logger.error(f"Error verifying document on blockchain: {e}")
            return {"error": str(e), "status": "failed"}
    
    async def create_digital_signature(self, document_path: str, 
                                     private_key: str) -> Dict[str, Any]:
        """Create digital signature for document"""
        try:
            # Load private key
            private_key_obj = serialization.load_pem_private_key(
                private_key.encode(),
                password=None,
                backend=default_backend()
            )
            
            # Read document
            with open(document_path, 'rb') as f:
                document_data = f.read()
            
            # Create signature
            signature = private_key_obj.sign(
                document_data,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            
            # Get public key
            public_key = private_key_obj.public_key()
            public_key_pem = public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )
            
            return {
                "signature": signature.hex(),
                "public_key": public_key_pem.decode(),
                "document_hash": hashlib.sha256(document_data).hexdigest(),
                "status": "signed"
            }
            
        except Exception as e:
            logger.error(f"Error creating digital signature: {e}")
            return {"error": str(e), "status": "failed"}
    
    async def verify_digital_signature(self, document_path: str, 
                                     signature: str, 
                                     public_key: str) -> Dict[str, Any]:
        """Verify digital signature of document"""
        try:
            # Load public key
            public_key_obj = serialization.load_pem_public_key(
                public_key.encode(),
                backend=default_backend()
            )
            
            # Read document
            with open(document_path, 'rb') as f:
                document_data = f.read()
            
            # Verify signature
            try:
                public_key_obj.verify(
                    bytes.fromhex(signature),
                    document_data,
                    padding.PSS(
                        mgf=padding.MGF1(hashes.SHA256()),
                        salt_length=padding.PSS.MAX_LENGTH
                    ),
                    hashes.SHA256()
                )
                
                return {
                    "is_valid": True,
                    "document_hash": hashlib.sha256(document_data).hexdigest(),
                    "status": "verified"
                }
                
            except Exception:
                return {
                    "is_valid": False,
                    "document_hash": hashlib.sha256(document_data).hexdigest(),
                    "status": "verification_failed"
                }
                
        except Exception as e:
            logger.error(f"Error verifying digital signature: {e}")
            return {"error": str(e), "status": "failed"}
    
    async def create_merkle_tree(self, documents: List[str]) -> Dict[str, Any]:
        """Create Merkle tree for multiple documents"""
        try:
            # Calculate hashes for all documents
            document_hashes = []
            for doc_path in documents:
                doc_hash = await self._calculate_document_hash(doc_path)
                document_hashes.append(doc_hash)
            
            # Build Merkle tree
            merkle_tree = self._build_merkle_tree(document_hashes)
            
            return {
                "merkle_root": merkle_tree["root"].hex(),
                "tree_depth": merkle_tree["depth"],
                "leaf_count": len(document_hashes),
                "document_hashes": [h.hex() for h in document_hashes],
                "tree_structure": merkle_tree["structure"],
                "status": "created"
            }
            
        except Exception as e:
            logger.error(f"Error creating Merkle tree: {e}")
            return {"error": str(e), "status": "failed"}
    
    async def verify_merkle_proof(self, document_path: str, 
                                merkle_root: str, 
                                proof: List[str]) -> Dict[str, Any]:
        """Verify document against Merkle tree proof"""
        try:
            # Calculate document hash
            document_hash = await self._calculate_document_hash(document_path)
            
            # Verify Merkle proof
            is_valid = self._verify_merkle_proof(
                document_hash, 
                bytes.fromhex(merkle_root), 
                [bytes.fromhex(p) for p in proof]
            )
            
            return {
                "document_hash": document_hash.hex(),
                "merkle_root": merkle_root,
                "is_valid": is_valid,
                "status": "verified" if is_valid else "verification_failed"
            }
            
        except Exception as e:
            logger.error(f"Error verifying Merkle proof: {e}")
            return {"error": str(e), "status": "failed"}
    
    async def create_nft_metadata(self, document_path: str, 
                                metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Create NFT metadata for document"""
        try:
            # Calculate document hash
            document_hash = await self._calculate_document_hash(document_path)
            
            # Upload to IPFS if available
            ipfs_hash = None
            if self.ipfs_client:
                ipfs_hash = await self._upload_to_ipfs(document_path)
            
            # Create NFT metadata
            nft_metadata = {
                "name": metadata.get("name", "Document NFT"),
                "description": metadata.get("description", "Digitally verified document"),
                "image": f"ipfs://{ipfs_hash}" if ipfs_hash else None,
                "attributes": [
                    {"trait_type": "Document Hash", "value": document_hash.hex()},
                    {"trait_type": "File Size", "value": Path(document_path).stat().st_size},
                    {"trait_type": "Created", "value": datetime.now().isoformat()}
                ],
                "external_url": metadata.get("external_url"),
                "background_color": metadata.get("background_color", "ffffff")
            }
            
            # Upload metadata to IPFS
            metadata_ipfs_hash = None
            if self.ipfs_client:
                metadata_json = json.dumps(nft_metadata, indent=2)
                metadata_ipfs_hash = self.ipfs_client.add_str(metadata_json)
            
            return {
                "nft_metadata": nft_metadata,
                "metadata_ipfs_hash": metadata_ipfs_hash,
                "document_hash": document_hash.hex(),
                "document_ipfs_hash": ipfs_hash,
                "status": "created"
            }
            
        except Exception as e:
            logger.error(f"Error creating NFT metadata: {e}")
            return {"error": str(e), "status": "failed"}
    
    async def _calculate_document_hash(self, document_path: str) -> bytes:
        """Calculate SHA-256 hash of document"""
        try:
            sha256_hash = hashlib.sha256()
            with open(document_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
            return sha256_hash.digest()
        except Exception as e:
            logger.error(f"Error calculating document hash: {e}")
            raise
    
    async def _upload_to_ipfs(self, file_path: str) -> str:
        """Upload file to IPFS"""
        try:
            if not self.ipfs_client:
                return None
            
            result = self.ipfs_client.add(file_path)
            return result['Hash']
            
        except Exception as e:
            logger.error(f"Error uploading to IPFS: {e}")
            return None
    
    async def _register_on_blockchain(self, document_hash: bytes, 
                                    owner_address: str, 
                                    metadata: str) -> str:
        """Register document on blockchain"""
        try:
            # Get contract
            contract = self.contracts['document_verification']
            
            # Prepare transaction
            tx = contract.functions.registerDocument(
                document_hash,
                owner_address,
                metadata
            ).build_transaction({
                'from': owner_address,
                'gas': 200000,
                'gasPrice': self.web3.eth.gas_price,
                'nonce': self.web3.eth.get_transaction_count(owner_address)
            })
            
            # Sign and send transaction
            # Note: In production, you'd use a proper wallet integration
            # This is a simplified example
            tx_hash = self.web3.eth.send_raw_transaction(tx.rawTransaction)
            
            return tx_hash.hex()
            
        except Exception as e:
            logger.error(f"Error registering on blockchain: {e}")
            raise
    
    async def _verify_on_blockchain(self, document_hash: bytes) -> Dict[str, Any]:
        """Verify document on blockchain"""
        try:
            contract = self.contracts['document_verification']
            
            result = contract.functions.verifyDocument(document_hash).call()
            
            return {
                "isValid": result[0],
                "owner": result[1],
                "timestamp": result[2]
            }
            
        except Exception as e:
            logger.error(f"Error verifying on blockchain: {e}")
            raise
    
    def _build_merkle_tree(self, hashes: List[bytes]) -> Dict[str, Any]:
        """Build Merkle tree from list of hashes"""
        try:
            if not hashes:
                return {"root": b"", "depth": 0, "structure": []}
            
            # Ensure even number of hashes
            if len(hashes) % 2 == 1:
                hashes.append(hashes[-1])
            
            current_level = hashes
            tree_structure = [current_level]
            
            while len(current_level) > 1:
                next_level = []
                for i in range(0, len(current_level), 2):
                    left = current_level[i]
                    right = current_level[i + 1] if i + 1 < len(current_level) else current_level[i]
                    
                    # Combine and hash
                    combined = left + right
                    parent_hash = hashlib.sha256(combined).digest()
                    next_level.append(parent_hash)
                
                current_level = next_level
                tree_structure.append(current_level)
            
            return {
                "root": current_level[0],
                "depth": len(tree_structure) - 1,
                "structure": [[h.hex() for h in level] for level in tree_structure]
            }
            
        except Exception as e:
            logger.error(f"Error building Merkle tree: {e}")
            raise
    
    def _verify_merkle_proof(self, leaf_hash: bytes, 
                           merkle_root: bytes, 
                           proof: List[bytes]) -> bool:
        """Verify Merkle proof"""
        try:
            current_hash = leaf_hash
            
            for sibling_hash in proof:
                # Determine order (left or right)
                if current_hash < sibling_hash:
                    combined = current_hash + sibling_hash
                else:
                    combined = sibling_hash + current_hash
                
                current_hash = hashlib.sha256(combined).digest()
            
            return current_hash == merkle_root
            
        except Exception as e:
            logger.error(f"Error verifying Merkle proof: {e}")
            return False


# Global blockchain verification instance
blockchain_verification = BlockchainVerification()


async def initialize_blockchain_verification():
    """Initialize the blockchain verification system"""
    await blockchain_verification.initialize()














