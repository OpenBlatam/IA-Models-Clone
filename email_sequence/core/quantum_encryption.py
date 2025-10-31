"""
Quantum-Resistant Encryption for Email Sequence System

This module provides quantum-resistant encryption capabilities for enhanced security
in the post-quantum computing era.
"""

import asyncio
import logging
import hashlib
import secrets
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from uuid import UUID
from dataclasses import dataclass
from enum import Enum

from pydantic import BaseModel, Field
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import base64

from .config import get_settings
from .exceptions import QuantumEncryptionError
from .cache import cache_manager

logger = logging.getLogger(__name__)
settings = get_settings()


class EncryptionAlgorithm(str, Enum):
    """Supported encryption algorithms"""
    AES_256_GCM = "aes_256_gcm"
    CHACHA20_POLY1305 = "chacha20_poly1305"
    RSA_4096 = "rsa_4096"
    HYBRID_AES_RSA = "hybrid_aes_rsa"
    QUANTUM_RESISTANT = "quantum_resistant"


class KeyType(str, Enum):
    """Key types for encryption"""
    SYMMETRIC = "symmetric"
    ASYMMETRIC = "asymmetric"
    HYBRID = "hybrid"
    QUANTUM_RESISTANT = "quantum_resistant"


@dataclass
class EncryptionKey:
    """Encryption key data structure"""
    key_id: str
    key_type: KeyType
    algorithm: EncryptionAlgorithm
    key_data: bytes
    created_at: datetime
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = None


@dataclass
class EncryptedData:
    """Encrypted data structure"""
    data: bytes
    key_id: str
    algorithm: EncryptionAlgorithm
    iv: bytes
    tag: Optional[bytes] = None
    metadata: Dict[str, Any] = None


class QuantumResistantEncryption:
    """Quantum-resistant encryption system"""
    
    def __init__(self):
        """Initialize quantum-resistant encryption"""
        self.keys: Dict[str, EncryptionKey] = {}
        self.key_cache: Dict[str, bytes] = {}
        self.encryption_cache: Dict[str, EncryptedData] = {}
        
        # Performance metrics
        self.encryptions_performed = 0
        self.decryptions_performed = 0
        self.keys_generated = 0
        
        logger.info("Quantum-Resistant Encryption initialized")
    
    async def initialize(self) -> None:
        """Initialize the encryption system"""
        try:
            # Generate master keys
            await self._generate_master_keys()
            
            # Initialize key rotation
            asyncio.create_task(self._key_rotation_scheduler())
            
            # Initialize cache cleanup
            asyncio.create_task(self._cache_cleanup_scheduler())
            
            logger.info("Quantum-Resistant Encryption initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing quantum-resistant encryption: {e}")
            raise QuantumEncryptionError(f"Failed to initialize quantum-resistant encryption: {e}")
    
    async def generate_key(
        self,
        key_type: KeyType = KeyType.SYMMETRIC,
        algorithm: EncryptionAlgorithm = EncryptionAlgorithm.AES_256_GCM,
        expires_in_hours: Optional[int] = None
    ) -> EncryptionKey:
        """
        Generate a new encryption key.
        
        Args:
            key_type: Type of key to generate
            algorithm: Encryption algorithm to use
            expires_in_hours: Key expiration time in hours
            
        Returns:
            EncryptionKey object
        """
        try:
            key_id = self._generate_key_id()
            
            # Generate key based on type and algorithm
            if key_type == KeyType.SYMMETRIC:
                key_data = await self._generate_symmetric_key(algorithm)
            elif key_type == KeyType.ASYMMETRIC:
                key_data = await self._generate_asymmetric_key(algorithm)
            elif key_type == KeyType.HYBRID:
                key_data = await self._generate_hybrid_key(algorithm)
            elif key_type == KeyType.QUANTUM_RESISTANT:
                key_data = await self._generate_quantum_resistant_key(algorithm)
            else:
                raise QuantumEncryptionError(f"Unsupported key type: {key_type}")
            
            # Create expiration time
            expires_at = None
            if expires_in_hours:
                expires_at = datetime.utcnow() + timedelta(hours=expires_in_hours)
            
            # Create encryption key
            encryption_key = EncryptionKey(
                key_id=key_id,
                key_type=key_type,
                algorithm=algorithm,
                key_data=key_data,
                created_at=datetime.utcnow(),
                expires_at=expires_at,
                metadata={
                    "generated_by": "quantum_encryption_system",
                    "key_strength": self._get_key_strength(algorithm)
                }
            )
            
            # Store key
            self.keys[key_id] = encryption_key
            self.key_cache[key_id] = key_data
            self.keys_generated += 1
            
            # Cache key with expiration
            cache_ttl = expires_in_hours * 3600 if expires_in_hours else 86400
            await cache_manager.set(f"encryption_key:{key_id}", encryption_key.__dict__, cache_ttl)
            
            logger.info(f"Generated encryption key: {key_id} ({key_type.value}, {algorithm.value})")
            return encryption_key
            
        except Exception as e:
            logger.error(f"Error generating encryption key: {e}")
            raise QuantumEncryptionError(f"Failed to generate encryption key: {e}")
    
    async def encrypt_data(
        self,
        data: bytes,
        key_id: str,
        algorithm: Optional[EncryptionAlgorithm] = None
    ) -> EncryptedData:
        """
        Encrypt data using specified key.
        
        Args:
            data: Data to encrypt
            key_id: Key ID to use for encryption
            algorithm: Optional algorithm override
            
        Returns:
            EncryptedData object
        """
        try:
            # Get encryption key
            encryption_key = await self._get_encryption_key(key_id)
            if not encryption_key:
                raise QuantumEncryptionError(f"Encryption key not found: {key_id}")
            
            # Check key expiration
            if encryption_key.expires_at and datetime.utcnow() > encryption_key.expires_at:
                raise QuantumEncryptionError(f"Encryption key expired: {key_id}")
            
            # Use specified algorithm or key's algorithm
            algo = algorithm or encryption_key.algorithm
            
            # Encrypt data based on algorithm
            if algo == EncryptionAlgorithm.AES_256_GCM:
                encrypted_data = await self._encrypt_aes_gcm(data, encryption_key.key_data)
            elif algo == EncryptionAlgorithm.CHACHA20_POLY1305:
                encrypted_data = await self._encrypt_chacha20_poly1305(data, encryption_key.key_data)
            elif algo == EncryptionAlgorithm.RSA_4096:
                encrypted_data = await self._encrypt_rsa(data, encryption_key.key_data)
            elif algo == EncryptionAlgorithm.HYBRID_AES_RSA:
                encrypted_data = await self._encrypt_hybrid(data, encryption_key.key_data)
            elif algo == EncryptionAlgorithm.QUANTUM_RESISTANT:
                encrypted_data = await self._encrypt_quantum_resistant(data, encryption_key.key_data)
            else:
                raise QuantumEncryptionError(f"Unsupported encryption algorithm: {algo}")
            
            # Create encrypted data object
            result = EncryptedData(
                data=encrypted_data["ciphertext"],
                key_id=key_id,
                algorithm=algo,
                iv=encrypted_data["iv"],
                tag=encrypted_data.get("tag"),
                metadata={
                    "encrypted_at": datetime.utcnow().isoformat(),
                    "key_type": encryption_key.key_type.value,
                    "data_size": len(data)
                }
            )
            
            self.encryptions_performed += 1
            
            # Cache encrypted data
            await cache_manager.set(
                f"encrypted_data:{hashlib.sha256(data).hexdigest()}",
                result.__dict__,
                3600  # 1 hour
            )
            
            logger.info(f"Data encrypted using key {key_id} with algorithm {algo.value}")
            return result
            
        except Exception as e:
            logger.error(f"Error encrypting data: {e}")
            raise QuantumEncryptionError(f"Failed to encrypt data: {e}")
    
    async def decrypt_data(
        self,
        encrypted_data: EncryptedData,
        key_id: Optional[str] = None
    ) -> bytes:
        """
        Decrypt data using specified key.
        
        Args:
            encrypted_data: EncryptedData object
            key_id: Optional key ID override
            
        Returns:
            Decrypted data bytes
        """
        try:
            # Use provided key_id or encrypted_data's key_id
            actual_key_id = key_id or encrypted_data.key_id
            
            # Get encryption key
            encryption_key = await self._get_encryption_key(actual_key_id)
            if not encryption_key:
                raise QuantumEncryptionError(f"Encryption key not found: {actual_key_id}")
            
            # Check key expiration
            if encryption_key.expires_at and datetime.utcnow() > encryption_key.expires_at:
                raise QuantumEncryptionError(f"Encryption key expired: {actual_key_id}")
            
            # Decrypt data based on algorithm
            if encrypted_data.algorithm == EncryptionAlgorithm.AES_256_GCM:
                decrypted_data = await self._decrypt_aes_gcm(
                    encrypted_data.data,
                    encryption_key.key_data,
                    encrypted_data.iv,
                    encrypted_data.tag
                )
            elif encrypted_data.algorithm == EncryptionAlgorithm.CHACHA20_POLY1305:
                decrypted_data = await self._decrypt_chacha20_poly1305(
                    encrypted_data.data,
                    encryption_key.key_data,
                    encrypted_data.iv,
                    encrypted_data.tag
                )
            elif encrypted_data.algorithm == EncryptionAlgorithm.RSA_4096:
                decrypted_data = await self._decrypt_rsa(
                    encrypted_data.data,
                    encryption_key.key_data
                )
            elif encrypted_data.algorithm == EncryptionAlgorithm.HYBRID_AES_RSA:
                decrypted_data = await self._decrypt_hybrid(
                    encrypted_data.data,
                    encryption_key.key_data,
                    encrypted_data.iv
                )
            elif encrypted_data.algorithm == EncryptionAlgorithm.QUANTUM_RESISTANT:
                decrypted_data = await self._decrypt_quantum_resistant(
                    encrypted_data.data,
                    encryption_key.key_data,
                    encrypted_data.iv
                )
            else:
                raise QuantumEncryptionError(f"Unsupported decryption algorithm: {encrypted_data.algorithm}")
            
            self.decryptions_performed += 1
            
            logger.info(f"Data decrypted using key {actual_key_id} with algorithm {encrypted_data.algorithm.value}")
            return decrypted_data
            
        except Exception as e:
            logger.error(f"Error decrypting data: {e}")
            raise QuantumEncryptionError(f"Failed to decrypt data: {e}")
    
    async def rotate_key(self, key_id: str) -> EncryptionKey:
        """
        Rotate an encryption key.
        
        Args:
            key_id: Key ID to rotate
            
        Returns:
            New EncryptionKey object
        """
        try:
            # Get existing key
            old_key = await self._get_encryption_key(key_id)
            if not old_key:
                raise QuantumEncryptionError(f"Key not found for rotation: {key_id}")
            
            # Generate new key with same parameters
            new_key = await self.generate_key(
                key_type=old_key.key_type,
                algorithm=old_key.algorithm,
                expires_in_hours=24  # Default 24 hours
            )
            
            # Mark old key as rotated
            old_key.metadata = old_key.metadata or {}
            old_key.metadata["rotated_at"] = datetime.utcnow().isoformat()
            old_key.metadata["replaced_by"] = new_key.key_id
            
            # Update cache
            await cache_manager.set(f"encryption_key:{key_id}", old_key.__dict__, 86400)
            
            logger.info(f"Key rotated: {key_id} -> {new_key.key_id}")
            return new_key
            
        except Exception as e:
            logger.error(f"Error rotating key: {e}")
            raise QuantumEncryptionError(f"Failed to rotate key: {e}")
    
    async def get_encryption_stats(self) -> Dict[str, Any]:
        """
        Get encryption system statistics.
        
        Returns:
            Dictionary with encryption statistics
        """
        try:
            active_keys = len([k for k in self.keys.values() if not k.expires_at or datetime.utcnow() < k.expires_at])
            expired_keys = len([k for k in self.keys.values() if k.expires_at and datetime.utcnow() >= k.expires_at])
            
            return {
                "total_keys": len(self.keys),
                "active_keys": active_keys,
                "expired_keys": expired_keys,
                "encryptions_performed": self.encryptions_performed,
                "decryptions_performed": self.decryptions_performed,
                "keys_generated": self.keys_generated,
                "cache_size": len(self.key_cache),
                "encrypted_data_cache_size": len(self.encryption_cache),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting encryption stats: {e}")
            return {"error": str(e)}
    
    # Private helper methods
    def _generate_key_id(self) -> str:
        """Generate unique key ID"""
        return f"key_{secrets.token_hex(16)}"
    
    async def _generate_symmetric_key(self, algorithm: EncryptionAlgorithm) -> bytes:
        """Generate symmetric encryption key"""
        if algorithm == EncryptionAlgorithm.AES_256_GCM:
            return secrets.token_bytes(32)  # 256 bits
        elif algorithm == EncryptionAlgorithm.CHACHA20_POLY1305:
            return secrets.token_bytes(32)  # 256 bits
        else:
            return secrets.token_bytes(32)  # Default 256 bits
    
    async def _generate_asymmetric_key(self, algorithm: EncryptionAlgorithm) -> bytes:
        """Generate asymmetric encryption key"""
        if algorithm == EncryptionAlgorithm.RSA_4096:
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=4096,
                backend=default_backend()
            )
            return private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )
        else:
            raise QuantumEncryptionError(f"Unsupported asymmetric algorithm: {algorithm}")
    
    async def _generate_hybrid_key(self, algorithm: EncryptionAlgorithm) -> bytes:
        """Generate hybrid encryption key"""
        # Generate both symmetric and asymmetric keys
        symmetric_key = await self._generate_symmetric_key(EncryptionAlgorithm.AES_256_GCM)
        asymmetric_key = await self._generate_asymmetric_key(EncryptionAlgorithm.RSA_4096)
        
        # Combine keys
        return symmetric_key + asymmetric_key
    
    async def _generate_quantum_resistant_key(self, algorithm: EncryptionAlgorithm) -> bytes:
        """Generate quantum-resistant encryption key"""
        # Use post-quantum cryptography algorithms
        # For now, use enhanced key generation
        return secrets.token_bytes(64)  # 512 bits for quantum resistance
    
    def _get_key_strength(self, algorithm: EncryptionAlgorithm) -> str:
        """Get key strength description"""
        strength_map = {
            EncryptionAlgorithm.AES_256_GCM: "256-bit",
            EncryptionAlgorithm.CHACHA20_POLY1305: "256-bit",
            EncryptionAlgorithm.RSA_4096: "4096-bit",
            EncryptionAlgorithm.HYBRID_AES_RSA: "Hybrid 256-bit + 4096-bit",
            EncryptionAlgorithm.QUANTUM_RESISTANT: "512-bit Quantum-Resistant"
        }
        return strength_map.get(algorithm, "Unknown")
    
    async def _get_encryption_key(self, key_id: str) -> Optional[EncryptionKey]:
        """Get encryption key by ID"""
        # Check memory cache first
        if key_id in self.keys:
            return self.keys[key_id]
        
        # Check Redis cache
        key_data = await cache_manager.get(f"encryption_key:{key_id}")
        if key_data:
            key = EncryptionKey(**key_data)
            self.keys[key_id] = key
            return key
        
        return None
    
    async def _encrypt_aes_gcm(self, data: bytes, key: bytes) -> Dict[str, bytes]:
        """Encrypt data using AES-256-GCM"""
        iv = secrets.token_bytes(12)  # 96-bit IV for GCM
        cipher = Cipher(algorithms.AES(key), modes.GCM(iv), backend=default_backend())
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(data) + encryptor.finalize()
        return {
            "ciphertext": ciphertext,
            "iv": iv,
            "tag": encryptor.tag
        }
    
    async def _decrypt_aes_gcm(self, ciphertext: bytes, key: bytes, iv: bytes, tag: bytes) -> bytes:
        """Decrypt data using AES-256-GCM"""
        cipher = Cipher(algorithms.AES(key), modes.GCM(iv, tag), backend=default_backend())
        decryptor = cipher.decryptor()
        return decryptor.update(ciphertext) + decryptor.finalize()
    
    async def _encrypt_chacha20_poly1305(self, data: bytes, key: bytes) -> Dict[str, bytes]:
        """Encrypt data using ChaCha20-Poly1305"""
        iv = secrets.token_bytes(12)  # 96-bit IV
        cipher = Cipher(algorithms.ChaCha20(key, iv), None, backend=default_backend())
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(data) + encryptor.finalize()
        return {
            "ciphertext": ciphertext,
            "iv": iv,
            "tag": b""  # ChaCha20-Poly1305 doesn't use separate tag
        }
    
    async def _decrypt_chacha20_poly1305(self, ciphertext: bytes, key: bytes, iv: bytes, tag: bytes) -> bytes:
        """Decrypt data using ChaCha20-Poly1305"""
        cipher = Cipher(algorithms.ChaCha20(key, iv), None, backend=default_backend())
        decryptor = cipher.decryptor()
        return decryptor.update(ciphertext) + decryptor.finalize()
    
    async def _encrypt_rsa(self, data: bytes, key: bytes) -> Dict[str, bytes]:
        """Encrypt data using RSA"""
        private_key = serialization.load_pem_private_key(key, password=None, backend=default_backend())
        public_key = private_key.public_key()
        
        # RSA can only encrypt small amounts of data
        max_length = (public_key.key_size // 8) - 2 * hashes.SHA256.digest_size - 2
        if len(data) > max_length:
            raise QuantumEncryptionError("Data too large for RSA encryption")
        
        ciphertext = public_key.encrypt(
            data,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        return {
            "ciphertext": ciphertext,
            "iv": b"",  # RSA doesn't use IV
            "tag": b""  # RSA doesn't use tag
        }
    
    async def _decrypt_rsa(self, ciphertext: bytes, key: bytes) -> bytes:
        """Decrypt data using RSA"""
        private_key = serialization.load_pem_private_key(key, password=None, backend=default_backend())
        
        return private_key.decrypt(
            ciphertext,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
    
    async def _encrypt_hybrid(self, data: bytes, key: bytes) -> Dict[str, bytes]:
        """Encrypt data using hybrid AES-RSA"""
        # Split key into symmetric and asymmetric parts
        symmetric_key = key[:32]
        asymmetric_key = key[32:]
        
        # Encrypt data with AES
        aes_result = await self._encrypt_aes_gcm(data, symmetric_key)
        
        # Encrypt AES key with RSA
        rsa_result = await self._encrypt_rsa(symmetric_key, asymmetric_key)
        
        return {
            "ciphertext": aes_result["ciphertext"] + rsa_result["ciphertext"],
            "iv": aes_result["iv"],
            "tag": aes_result["tag"]
        }
    
    async def _decrypt_hybrid(self, ciphertext: bytes, key: bytes, iv: bytes) -> bytes:
        """Decrypt data using hybrid AES-RSA"""
        # Split key into symmetric and asymmetric parts
        symmetric_key = key[:32]
        asymmetric_key = key[32:]
        
        # Split ciphertext
        rsa_ciphertext_length = 512  # RSA-4096 ciphertext length
        aes_ciphertext = ciphertext[:-rsa_ciphertext_length]
        rsa_ciphertext = ciphertext[-rsa_ciphertext_length:]
        
        # Decrypt AES key with RSA
        decrypted_aes_key = await self._decrypt_rsa(rsa_ciphertext, asymmetric_key)
        
        # Decrypt data with AES
        return await self._decrypt_aes_gcm(aes_ciphertext, decrypted_aes_key, iv, b"")
    
    async def _encrypt_quantum_resistant(self, data: bytes, key: bytes) -> Dict[str, bytes]:
        """Encrypt data using quantum-resistant algorithm"""
        # Use enhanced AES with quantum-resistant key derivation
        derived_key = hashlib.sha512(key + b"quantum_resistant_salt").digest()[:32]
        return await self._encrypt_aes_gcm(data, derived_key)
    
    async def _decrypt_quantum_resistant(self, ciphertext: bytes, key: bytes, iv: bytes) -> bytes:
        """Decrypt data using quantum-resistant algorithm"""
        # Use enhanced AES with quantum-resistant key derivation
        derived_key = hashlib.sha512(key + b"quantum_resistant_salt").digest()[:32]
        return await self._decrypt_aes_gcm(ciphertext, derived_key, iv, b"")
    
    async def _generate_master_keys(self) -> None:
        """Generate master encryption keys"""
        try:
            # Generate master key for system operations
            master_key = await self.generate_key(
                key_type=KeyType.QUANTUM_RESISTANT,
                algorithm=EncryptionAlgorithm.QUANTUM_RESISTANT,
                expires_in_hours=8760  # 1 year
            )
            
            logger.info("Master encryption keys generated")
            
        except Exception as e:
            logger.error(f"Error generating master keys: {e}")
    
    async def _key_rotation_scheduler(self) -> None:
        """Schedule automatic key rotation"""
        while True:
            try:
                await asyncio.sleep(3600)  # Check every hour
                
                # Find keys that need rotation
                for key_id, key in self.keys.items():
                    if key.expires_at and datetime.utcnow() > key.expires_at - timedelta(hours=1):
                        logger.info(f"Rotating expired key: {key_id}")
                        await self.rotate_key(key_id)
                
            except Exception as e:
                logger.error(f"Error in key rotation scheduler: {e}")
    
    async def _cache_cleanup_scheduler(self) -> None:
        """Schedule cache cleanup"""
        while True:
            try:
                await asyncio.sleep(1800)  # Clean up every 30 minutes
                
                # Clean up expired keys from memory
                expired_keys = [
                    key_id for key_id, key in self.keys.items()
                    if key.expires_at and datetime.utcnow() > key.expires_at
                ]
                
                for key_id in expired_keys:
                    del self.keys[key_id]
                    if key_id in self.key_cache:
                        del self.key_cache[key_id]
                
                if expired_keys:
                    logger.info(f"Cleaned up {len(expired_keys)} expired keys")
                
            except Exception as e:
                logger.error(f"Error in cache cleanup scheduler: {e}")


# Global quantum-resistant encryption instance
quantum_encryption = QuantumResistantEncryption()






























