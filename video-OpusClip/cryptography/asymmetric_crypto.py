#!/usr/bin/env python3
"""
Asymmetric Cryptography for Video-OpusClip
Asymmetric encryption, digital signatures, and key exchange
"""

import os
import base64
import hashlib
from typing import Dict, List, Any, Optional, Union, Tuple
from enum import Enum
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, ec, ed25519, x25519, dsa
from cryptography.hazmat.primitives.asymmetric import padding, utils
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.backends import default_backend
from cryptography.exceptions import InvalidKey

logger = logging.getLogger(__name__)


class AsymmetricAlgorithm(Enum):
    """Asymmetric encryption algorithms"""
    RSA = "rsa"
    ECC = "ecc"
    DSA = "dsa"
    ED25519 = "ed25519"
    X25519 = "x25519"


class KeyFormat(Enum):
    """Key formats"""
    PEM = "pem"
    DER = "der"
    SSH = "ssh"
    JWK = "jwk"


@dataclass
class KeyPair:
    """Asymmetric key pair"""
    private_key: bytes
    public_key: bytes
    algorithm: AsymmetricAlgorithm
    key_size: int
    format: KeyFormat = KeyFormat.PEM


@dataclass
class SignatureResult:
    """Result of signature operation"""
    signature: bytes
    algorithm: AsymmetricAlgorithm
    hash_algorithm: str
    key_size: int


@dataclass
class VerificationResult:
    """Result of signature verification"""
    verified: bool
    algorithm: AsymmetricAlgorithm
    hash_algorithm: str


class AsymmetricCryptoBase(ABC):
    """Base class for asymmetric cryptography"""
    
    def __init__(self, algorithm: AsymmetricAlgorithm, key_size: int = 2048):
        self.algorithm = algorithm
        self.key_size = key_size
        self.backend = default_backend()
    
    @abstractmethod
    def generate_key_pair(self) -> KeyPair:
        """Generate key pair"""
        pass
    
    @abstractmethod
    def encrypt(self, data: bytes, public_key: bytes) -> bytes:
        """Encrypt data with public key"""
        pass
    
    @abstractmethod
    def decrypt(self, data: bytes, private_key: bytes) -> bytes:
        """Decrypt data with private key"""
        pass
    
    @abstractmethod
    def sign(self, data: bytes, private_key: bytes, hash_algorithm: str = "sha256") -> SignatureResult:
        """Sign data with private key"""
        pass
    
    @abstractmethod
    def verify(self, data: bytes, signature: bytes, public_key: bytes, hash_algorithm: str = "sha256") -> VerificationResult:
        """Verify signature with public key"""
        pass


class RSAEncryption(AsymmetricCryptoBase):
    """RSA encryption implementation"""
    
    def __init__(self, key_size: int = 2048):
        super().__init__(AsymmetricAlgorithm.RSA, key_size)
    
    def generate_key_pair(self) -> KeyPair:
        """Generate RSA key pair"""
        try:
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=self.key_size,
                backend=self.backend
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
            
            return KeyPair(
                private_key=private_pem,
                public_key=public_pem,
                algorithm=self.algorithm,
                key_size=self.key_size
            )
            
        except Exception as e:
            logger.error(f"RSA key pair generation failed: {e}")
            raise
    
    def encrypt(self, data: bytes, public_key: bytes) -> bytes:
        """Encrypt data with RSA public key"""
        try:
            # Load public key
            if isinstance(public_key, bytes):
                public_key_obj = serialization.load_pem_public_key(public_key, backend=self.backend)
            else:
                public_key_obj = public_key
            
            # Encrypt with OAEP padding
            ciphertext = public_key_obj.encrypt(
                data,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            
            return ciphertext
            
        except Exception as e:
            logger.error(f"RSA encryption failed: {e}")
            raise
    
    def decrypt(self, data: bytes, private_key: bytes) -> bytes:
        """Decrypt data with RSA private key"""
        try:
            # Load private key
            if isinstance(private_key, bytes):
                private_key_obj = serialization.load_pem_private_key(
                    private_key,
                    password=None,
                    backend=self.backend
                )
            else:
                private_key_obj = private_key
            
            # Decrypt with OAEP padding
            plaintext = private_key_obj.decrypt(
                data,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            
            return plaintext
            
        except Exception as e:
            logger.error(f"RSA decryption failed: {e}")
            raise
    
    def sign(self, data: bytes, private_key: bytes, hash_algorithm: str = "sha256") -> SignatureResult:
        """Sign data with RSA private key"""
        try:
            # Load private key
            if isinstance(private_key, bytes):
                private_key_obj = serialization.load_pem_private_key(
                    private_key,
                    password=None,
                    backend=self.backend
                )
            else:
                private_key_obj = private_key
            
            # Choose hash algorithm
            hash_algo = getattr(hashes, hash_algorithm.upper())()
            
            # Sign with PSS padding
            signature = private_key_obj.sign(
                data,
                padding.PSS(
                    mgf=padding.MGF1(algorithm=hash_algo),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hash_algo
            )
            
            return SignatureResult(
                signature=signature,
                algorithm=self.algorithm,
                hash_algorithm=hash_algorithm,
                key_size=self.key_size
            )
            
        except Exception as e:
            logger.error(f"RSA signing failed: {e}")
            raise
    
    def verify(self, data: bytes, signature: bytes, public_key: bytes, hash_algorithm: str = "sha256") -> VerificationResult:
        """Verify signature with RSA public key"""
        try:
            # Load public key
            if isinstance(public_key, bytes):
                public_key_obj = serialization.load_pem_public_key(public_key, backend=self.backend)
            else:
                public_key_obj = public_key
            
            # Choose hash algorithm
            hash_algo = getattr(hashes, hash_algorithm.upper())()
            
            # Verify with PSS padding
            public_key_obj.verify(
                signature,
                data,
                padding.PSS(
                    mgf=padding.MGF1(algorithm=hash_algo),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hash_algo
            )
            
            return VerificationResult(
                verified=True,
                algorithm=self.algorithm,
                hash_algorithm=hash_algorithm
            )
            
        except Exception as e:
            logger.error(f"RSA verification failed: {e}")
            return VerificationResult(
                verified=False,
                algorithm=self.algorithm,
                hash_algorithm=hash_algorithm
            )


class ECCEncryption(AsymmetricCryptoBase):
    """ECC encryption implementation"""
    
    def __init__(self, curve: str = "secp256r1"):
        super().__init__(AsymmetricAlgorithm.ECC, 256)  # ECC key size is curve-dependent
        self.curve = curve
        self.curve_obj = getattr(ec, curve.upper())()
    
    def generate_key_pair(self) -> KeyPair:
        """Generate ECC key pair"""
        try:
            private_key = ec.generate_private_key(
                self.curve_obj,
                backend=self.backend
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
            
            return KeyPair(
                private_key=private_pem,
                public_key=public_pem,
                algorithm=self.algorithm,
                key_size=self.key_size
            )
            
        except Exception as e:
            logger.error(f"ECC key pair generation failed: {e}")
            raise
    
    def encrypt(self, data: bytes, public_key: bytes) -> bytes:
        """Encrypt data with ECC public key (using ECDH)"""
        try:
            # Load public key
            if isinstance(public_key, bytes):
                public_key_obj = serialization.load_pem_public_key(public_key, backend=self.backend)
            else:
                public_key_obj = public_key
            
            # Generate ephemeral key pair
            ephemeral_private_key = ec.generate_private_key(
                self.curve_obj,
                backend=self.backend
            )
            
            # Perform ECDH key exchange
            shared_key = ephemeral_private_key.exchange(
                ec.ECDH(),
                public_key_obj
            )
            
            # Derive encryption key
            derived_key = HKDF(
                algorithm=hashes.SHA256(),
                length=32,
                salt=None,
                info=b"ecc_encryption",
                backend=self.backend
            ).derive(shared_key)
            
            # Encrypt data with derived key (using AES)
            from .symmetric_crypto import AESEncryption
            aes = AESEncryption()
            encrypted_data = aes.encrypt(data, derived_key)
            
            # Combine ephemeral public key and encrypted data
            ephemeral_public_key = ephemeral_private_key.public_key()
            ephemeral_public_bytes = ephemeral_public_key.public_bytes(
                encoding=serialization.Encoding.DER,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )
            
            return ephemeral_public_bytes + encrypted_data.ciphertext
            
        except Exception as e:
            logger.error(f"ECC encryption failed: {e}")
            raise
    
    def decrypt(self, data: bytes, private_key: bytes) -> bytes:
        """Decrypt data with ECC private key"""
        try:
            # Load private key
            if isinstance(private_key, bytes):
                private_key_obj = serialization.load_pem_private_key(
                    private_key,
                    password=None,
                    backend=self.backend
                )
            else:
                private_key_obj = private_key
            
            # Extract ephemeral public key and encrypted data
            ephemeral_public_key_size = 91  # Approximate size for secp256r1
            ephemeral_public_bytes = data[:ephemeral_public_key_size]
            encrypted_data = data[ephemeral_public_key_size:]
            
            # Load ephemeral public key
            ephemeral_public_key = serialization.load_der_public_key(
                ephemeral_public_bytes,
                backend=self.backend
            )
            
            # Perform ECDH key exchange
            shared_key = private_key_obj.exchange(
                ec.ECDH(),
                ephemeral_public_key
            )
            
            # Derive decryption key
            derived_key = HKDF(
                algorithm=hashes.SHA256(),
                length=32,
                salt=None,
                info=b"ecc_encryption",
                backend=self.backend
            ).derive(shared_key)
            
            # Decrypt data with derived key
            from .symmetric_crypto import AESEncryption
            aes = AESEncryption()
            decrypted_data = aes.decrypt(encrypted_data, derived_key)
            
            return decrypted_data.plaintext
            
        except Exception as e:
            logger.error(f"ECC decryption failed: {e}")
            raise
    
    def sign(self, data: bytes, private_key: bytes, hash_algorithm: str = "sha256") -> SignatureResult:
        """Sign data with ECC private key"""
        try:
            # Load private key
            if isinstance(private_key, bytes):
                private_key_obj = serialization.load_pem_private_key(
                    private_key,
                    password=None,
                    backend=self.backend
                )
            else:
                private_key_obj = private_key
            
            # Choose hash algorithm
            hash_algo = getattr(hashes, hash_algorithm.upper())()
            
            # Sign
            signature = private_key_obj.sign(
                data,
                ec.ECDSA(hash_algo)
            )
            
            return SignatureResult(
                signature=signature,
                algorithm=self.algorithm,
                hash_algorithm=hash_algorithm,
                key_size=self.key_size
            )
            
        except Exception as e:
            logger.error(f"ECC signing failed: {e}")
            raise
    
    def verify(self, data: bytes, signature: bytes, public_key: bytes, hash_algorithm: str = "sha256") -> VerificationResult:
        """Verify signature with ECC public key"""
        try:
            # Load public key
            if isinstance(public_key, bytes):
                public_key_obj = serialization.load_pem_public_key(public_key, backend=self.backend)
            else:
                public_key_obj = public_key
            
            # Choose hash algorithm
            hash_algo = getattr(hashes, hash_algorithm.upper())()
            
            # Verify
            public_key_obj.verify(
                signature,
                data,
                ec.ECDSA(hash_algo)
            )
            
            return VerificationResult(
                verified=True,
                algorithm=self.algorithm,
                hash_algorithm=hash_algorithm
            )
            
        except Exception as e:
            logger.error(f"ECC verification failed: {e}")
            return VerificationResult(
                verified=False,
                algorithm=self.algorithm,
                hash_algorithm=hash_algorithm
            )


class DSAEncryption(AsymmetricCryptoBase):
    """DSA encryption implementation (signature only)"""
    
    def __init__(self, key_size: int = 2048):
        super().__init__(AsymmetricAlgorithm.DSA, key_size)
    
    def generate_key_pair(self) -> KeyPair:
        """Generate DSA key pair"""
        try:
            private_key = dsa.generate_private_key(
                key_size=self.key_size,
                backend=self.backend
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
            
            return KeyPair(
                private_key=private_pem,
                public_key=public_pem,
                algorithm=self.algorithm,
                key_size=self.key_size
            )
            
        except Exception as e:
            logger.error(f"DSA key pair generation failed: {e}")
            raise
    
    def encrypt(self, data: bytes, public_key: bytes) -> bytes:
        """DSA does not support encryption"""
        raise NotImplementedError("DSA does not support encryption")
    
    def decrypt(self, data: bytes, private_key: bytes) -> bytes:
        """DSA does not support decryption"""
        raise NotImplementedError("DSA does not support decryption")
    
    def sign(self, data: bytes, private_key: bytes, hash_algorithm: str = "sha256") -> SignatureResult:
        """Sign data with DSA private key"""
        try:
            # Load private key
            if isinstance(private_key, bytes):
                private_key_obj = serialization.load_pem_private_key(
                    private_key,
                    password=None,
                    backend=self.backend
                )
            else:
                private_key_obj = private_key
            
            # Choose hash algorithm
            hash_algo = getattr(hashes, hash_algorithm.upper())()
            
            # Sign
            signature = private_key_obj.sign(
                data,
                hash_algo
            )
            
            return SignatureResult(
                signature=signature,
                algorithm=self.algorithm,
                hash_algorithm=hash_algorithm,
                key_size=self.key_size
            )
            
        except Exception as e:
            logger.error(f"DSA signing failed: {e}")
            raise
    
    def verify(self, data: bytes, signature: bytes, public_key: bytes, hash_algorithm: str = "sha256") -> VerificationResult:
        """Verify signature with DSA public key"""
        try:
            # Load public key
            if isinstance(public_key, bytes):
                public_key_obj = serialization.load_pem_public_key(public_key, backend=self.backend)
            else:
                public_key_obj = public_key
            
            # Choose hash algorithm
            hash_algo = getattr(hashes, hash_algorithm.upper())()
            
            # Verify
            public_key_obj.verify(
                signature,
                data,
                hash_algo
            )
            
            return VerificationResult(
                verified=True,
                algorithm=self.algorithm,
                hash_algorithm=hash_algorithm
            )
            
        except Exception as e:
            logger.error(f"DSA verification failed: {e}")
            return VerificationResult(
                verified=False,
                algorithm=self.algorithm,
                hash_algorithm=hash_algorithm
            )


class Ed25519Encryption(AsymmetricCryptoBase):
    """Ed25519 encryption implementation (signature only)"""
    
    def __init__(self):
        super().__init__(AsymmetricAlgorithm.ED25519, 256)
    
    def generate_key_pair(self) -> KeyPair:
        """Generate Ed25519 key pair"""
        try:
            private_key = ed25519.Ed25519PrivateKey.generate()
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
            
            return KeyPair(
                private_key=private_pem,
                public_key=public_pem,
                algorithm=self.algorithm,
                key_size=self.key_size
            )
            
        except Exception as e:
            logger.error(f"Ed25519 key pair generation failed: {e}")
            raise
    
    def encrypt(self, data: bytes, public_key: bytes) -> bytes:
        """Ed25519 does not support encryption"""
        raise NotImplementedError("Ed25519 does not support encryption")
    
    def decrypt(self, data: bytes, private_key: bytes) -> bytes:
        """Ed25519 does not support decryption"""
        raise NotImplementedError("Ed25519 does not support decryption")
    
    def sign(self, data: bytes, private_key: bytes, hash_algorithm: str = "sha256") -> SignatureResult:
        """Sign data with Ed25519 private key"""
        try:
            # Load private key
            if isinstance(private_key, bytes):
                private_key_obj = serialization.load_pem_private_key(
                    private_key,
                    password=None,
                    backend=self.backend
                )
            else:
                private_key_obj = private_key
            
            # Sign (Ed25519 has built-in hashing)
            signature = private_key_obj.sign(data)
            
            return SignatureResult(
                signature=signature,
                algorithm=self.algorithm,
                hash_algorithm="ed25519",  # Ed25519 has built-in hashing
                key_size=self.key_size
            )
            
        except Exception as e:
            logger.error(f"Ed25519 signing failed: {e}")
            raise
    
    def verify(self, data: bytes, signature: bytes, public_key: bytes, hash_algorithm: str = "sha256") -> VerificationResult:
        """Verify signature with Ed25519 public key"""
        try:
            # Load public key
            if isinstance(public_key, bytes):
                public_key_obj = serialization.load_pem_public_key(public_key, backend=self.backend)
            else:
                public_key_obj = public_key
            
            # Verify (Ed25519 has built-in hashing)
            public_key_obj.verify(signature, data)
            
            return VerificationResult(
                verified=True,
                algorithm=self.algorithm,
                hash_algorithm="ed25519"
            )
            
        except Exception as e:
            logger.error(f"Ed25519 verification failed: {e}")
            return VerificationResult(
                verified=False,
                algorithm=self.algorithm,
                hash_algorithm="ed25519"
            )


class X25519Encryption(AsymmetricCryptoBase):
    """X25519 encryption implementation (key exchange only)"""
    
    def __init__(self):
        super().__init__(AsymmetricAlgorithm.X25519, 256)
    
    def generate_key_pair(self) -> KeyPair:
        """Generate X25519 key pair"""
        try:
            private_key = x25519.X25519PrivateKey.generate()
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
            
            return KeyPair(
                private_key=private_pem,
                public_key=public_pem,
                algorithm=self.algorithm,
                key_size=self.key_size
            )
            
        except Exception as e:
            logger.error(f"X25519 key pair generation failed: {e}")
            raise
    
    def encrypt(self, data: bytes, public_key: bytes) -> bytes:
        """X25519 does not support direct encryption"""
        raise NotImplementedError("X25519 does not support direct encryption")
    
    def decrypt(self, data: bytes, private_key: bytes) -> bytes:
        """X25519 does not support direct decryption"""
        raise NotImplementedError("X25519 does not support direct decryption")
    
    def sign(self, data: bytes, private_key: bytes, hash_algorithm: str = "sha256") -> SignatureResult:
        """X25519 does not support signing"""
        raise NotImplementedError("X25519 does not support signing")
    
    def verify(self, data: bytes, signature: bytes, public_key: bytes, hash_algorithm: str = "sha256") -> VerificationResult:
        """X25519 does not support signature verification"""
        raise NotImplementedError("X25519 does not support signature verification")
    
    def exchange(self, private_key: bytes, public_key: bytes) -> bytes:
        """Perform X25519 key exchange"""
        try:
            # Load keys
            if isinstance(private_key, bytes):
                private_key_obj = serialization.load_pem_private_key(
                    private_key,
                    password=None,
                    backend=self.backend
                )
            else:
                private_key_obj = private_key
            
            if isinstance(public_key, bytes):
                public_key_obj = serialization.load_pem_public_key(public_key, backend=self.backend)
            else:
                public_key_obj = public_key
            
            # Perform key exchange
            shared_key = private_key_obj.exchange(public_key_obj)
            return shared_key
            
        except Exception as e:
            logger.error(f"X25519 key exchange failed: {e}")
            raise


class AsymmetricCryptoService:
    """Main asymmetric cryptography service"""
    
    def __init__(self):
        self.algorithms: Dict[AsymmetricAlgorithm, AsymmetricCryptoBase] = {
            AsymmetricAlgorithm.RSA: RSAEncryption(),
            AsymmetricAlgorithm.ECC: ECCEncryption(),
            AsymmetricAlgorithm.DSA: DSAEncryption(),
            AsymmetricAlgorithm.ED25519: Ed25519Encryption(),
            AsymmetricAlgorithm.X25519: X25519Encryption()
        }
    
    def generate_key_pair(self, algorithm: AsymmetricAlgorithm = AsymmetricAlgorithm.RSA) -> KeyPair:
        """Generate key pair for specified algorithm"""
        if algorithm not in self.algorithms:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        
        crypto = self.algorithms[algorithm]
        return crypto.generate_key_pair()
    
    def encrypt(self, data: bytes, public_key: bytes, algorithm: AsymmetricAlgorithm = AsymmetricAlgorithm.RSA) -> bytes:
        """Encrypt data with public key"""
        if algorithm not in self.algorithms:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        
        crypto = self.algorithms[algorithm]
        return crypto.encrypt(data, public_key)
    
    def decrypt(self, data: bytes, private_key: bytes, algorithm: AsymmetricAlgorithm = AsymmetricAlgorithm.RSA) -> bytes:
        """Decrypt data with private key"""
        if algorithm not in self.algorithms:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        
        crypto = self.algorithms[algorithm]
        return crypto.decrypt(data, private_key)
    
    def sign(self, data: bytes, private_key: bytes, algorithm: AsymmetricAlgorithm = AsymmetricAlgorithm.RSA, hash_algorithm: str = "sha256") -> SignatureResult:
        """Sign data with private key"""
        if algorithm not in self.algorithms:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        
        crypto = self.algorithms[algorithm]
        return crypto.sign(data, private_key, hash_algorithm)
    
    def verify(self, data: bytes, signature: bytes, public_key: bytes, algorithm: AsymmetricAlgorithm = AsymmetricAlgorithm.RSA, hash_algorithm: str = "sha256") -> VerificationResult:
        """Verify signature with public key"""
        if algorithm not in self.algorithms:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        
        crypto = self.algorithms[algorithm]
        return crypto.verify(data, signature, public_key, hash_algorithm)
    
    def get_supported_algorithms(self) -> List[AsymmetricAlgorithm]:
        """Get list of supported algorithms"""
        return list(self.algorithms.keys())


# Convenience functions
def generate_key_pair(algorithm: AsymmetricAlgorithm = AsymmetricAlgorithm.RSA) -> KeyPair:
    """Convenience function for generating key pairs"""
    service = AsymmetricCryptoService()
    return service.generate_key_pair(algorithm)


def encrypt_asymmetric(data: bytes, public_key: bytes, algorithm: AsymmetricAlgorithm = AsymmetricAlgorithm.RSA) -> bytes:
    """Convenience function for asymmetric encryption"""
    service = AsymmetricCryptoService()
    return service.encrypt(data, public_key, algorithm)


def decrypt_asymmetric(data: bytes, private_key: bytes, algorithm: AsymmetricAlgorithm = AsymmetricAlgorithm.RSA) -> bytes:
    """Convenience function for asymmetric decryption"""
    service = AsymmetricCryptoService()
    return service.decrypt(data, private_key, algorithm)


def sign_data(data: bytes, private_key: bytes, algorithm: AsymmetricAlgorithm = AsymmetricAlgorithm.RSA, hash_algorithm: str = "sha256") -> SignatureResult:
    """Convenience function for signing data"""
    service = AsymmetricCryptoService()
    return service.sign(data, private_key, algorithm, hash_algorithm)


def verify_signature(data: bytes, signature: bytes, public_key: bytes, algorithm: AsymmetricAlgorithm = AsymmetricAlgorithm.RSA, hash_algorithm: str = "sha256") -> VerificationResult:
    """Convenience function for verifying signatures"""
    service = AsymmetricCryptoService()
    return service.verify(data, signature, public_key, algorithm, hash_algorithm)


# Example usage
if __name__ == "__main__":
    # Example asymmetric cryptography
    print("🔐 Asymmetric Cryptography Example")
    
    # Test data
    test_data = b"Hello, Video-OpusClip! This is a test message for asymmetric encryption."
    
    # Test RSA encryption
    print("\n" + "="*60)
    print("RSA ENCRYPTION")
    print("="*60)
    
    rsa_key_pair = generate_key_pair(AsymmetricAlgorithm.RSA)
    print(f"✅ RSA key pair generated: {rsa_key_pair.key_size} bits")
    
    rsa_encrypted = encrypt_asymmetric(test_data, rsa_key_pair.public_key, AsymmetricAlgorithm.RSA)
    print(f"✅ RSA encrypted: {len(rsa_encrypted)} bytes")
    
    rsa_decrypted = decrypt_asymmetric(rsa_encrypted, rsa_key_pair.private_key, AsymmetricAlgorithm.RSA)
    print(f"✅ RSA decrypted: {rsa_decrypted.decode()}")
    
    # Test RSA signing
    rsa_signature = sign_data(test_data, rsa_key_pair.private_key, AsymmetricAlgorithm.RSA)
    print(f"✅ RSA signature: {len(rsa_signature.signature)} bytes")
    
    rsa_verified = verify_signature(test_data, rsa_signature.signature, rsa_key_pair.public_key, AsymmetricAlgorithm.RSA)
    print(f"✅ RSA verification: {rsa_verified.verified}")
    
    # Test ECC encryption
    print("\n" + "="*60)
    print("ECC ENCRYPTION")
    print("="*60)
    
    ecc_key_pair = generate_key_pair(AsymmetricAlgorithm.ECC)
    print(f"✅ ECC key pair generated: {ecc_key_pair.key_size} bits")
    
    ecc_encrypted = encrypt_asymmetric(test_data, ecc_key_pair.public_key, AsymmetricAlgorithm.ECC)
    print(f"✅ ECC encrypted: {len(ecc_encrypted)} bytes")
    
    ecc_decrypted = decrypt_asymmetric(ecc_encrypted, ecc_key_pair.private_key, AsymmetricAlgorithm.ECC)
    print(f"✅ ECC decrypted: {ecc_decrypted.decode()}")
    
    # Test ECC signing
    ecc_signature = sign_data(test_data, ecc_key_pair.private_key, AsymmetricAlgorithm.ECC)
    print(f"✅ ECC signature: {len(ecc_signature.signature)} bytes")
    
    ecc_verified = verify_signature(test_data, ecc_signature.signature, ecc_key_pair.public_key, AsymmetricAlgorithm.ECC)
    print(f"✅ ECC verification: {ecc_verified.verified}")
    
    # Test Ed25519 signing
    print("\n" + "="*60)
    print("ED25519 SIGNING")
    print("="*60)
    
    ed25519_key_pair = generate_key_pair(AsymmetricAlgorithm.ED25519)
    print(f"✅ Ed25519 key pair generated: {ed25519_key_pair.key_size} bits")
    
    ed25519_signature = sign_data(test_data, ed25519_key_pair.private_key, AsymmetricAlgorithm.ED25519)
    print(f"✅ Ed25519 signature: {len(ed25519_signature.signature)} bytes")
    
    ed25519_verified = verify_signature(test_data, ed25519_signature.signature, ed25519_key_pair.public_key, AsymmetricAlgorithm.ED25519)
    print(f"✅ Ed25519 verification: {ed25519_verified.verified}")
    
    # Test X25519 key exchange
    print("\n" + "="*60)
    print("X25519 KEY EXCHANGE")
    print("="*60)
    
    x25519_key_pair1 = generate_key_pair(AsymmetricAlgorithm.X25519)
    x25519_key_pair2 = generate_key_pair(AsymmetricAlgorithm.X25519)
    print(f"✅ X25519 key pairs generated")
    
    shared_key1 = x25519_key_pair1.private_key.exchange(x25519_key_pair2.public_key)
    shared_key2 = x25519_key_pair2.private_key.exchange(x25519_key_pair1.public_key)
    print(f"✅ X25519 key exchange: {shared_key1 == shared_key2}")
    print(f"   Shared key: {shared_key1.hex()[:32]}...")
    
    print("\n✅ Asymmetric cryptography example completed!") 