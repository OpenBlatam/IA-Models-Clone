#!/usr/bin/env python3
"""
Symmetric Cryptography for Video-OpusClip
Symmetric encryption algorithms and operations
"""

import os
import base64
import hashlib
from typing import Dict, List, Any, Optional, Union, Tuple
from enum import Enum
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import hashes, hmac
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.padding import PKCS7
from cryptography.fernet import Fernet
from cryptography.hazmat.backends import default_backend

logger = logging.getLogger(__name__)


class SymmetricAlgorithm(Enum):
    """Symmetric encryption algorithms"""
    AES = "aes"
    CHACHA20 = "chacha20"
    FERNET = "fernet"
    BLOWFISH = "blowfish"
    TWOFISH = "twofish"


class KeySize(Enum):
    """Key sizes for symmetric algorithms"""
    AES_128 = 128
    AES_192 = 192
    AES_256 = 256
    CHACHA20_256 = 256
    BLOWFISH_448 = 448
    TWOFISH_256 = 256


class EncryptionMode(Enum):
    """Encryption modes"""
    CBC = "cbc"
    GCM = "gcm"
    CTR = "ctr"
    CFB = "cfb"
    OFB = "ofb"


@dataclass
class EncryptionResult:
    """Result of encryption operation"""
    ciphertext: bytes
    iv: Optional[bytes] = None
    tag: Optional[bytes] = None
    salt: Optional[bytes] = None
    algorithm: SymmetricAlgorithm = SymmetricAlgorithm.AES
    mode: EncryptionMode = EncryptionMode.GCM
    key_size: KeySize = KeySize.AES_256


@dataclass
class DecryptionResult:
    """Result of decryption operation"""
    plaintext: bytes
    verified: bool = True
    algorithm: SymmetricAlgorithm = SymmetricAlgorithm.AES
    mode: EncryptionMode = EncryptionMode.GCM


class SymmetricCryptoBase(ABC):
    """Base class for symmetric cryptography"""
    
    def __init__(self, algorithm: SymmetricAlgorithm, key_size: KeySize):
        self.algorithm = algorithm
        self.key_size = key_size
        self.backend = default_backend()
    
    @abstractmethod
    def encrypt(self, data: bytes, key: bytes, **kwargs) -> EncryptionResult:
        """Encrypt data"""
        pass
    
    @abstractmethod
    def decrypt(self, data: bytes, key: bytes, **kwargs) -> DecryptionResult:
        """Decrypt data"""
        pass
    
    def generate_key(self, password: Optional[str] = None, salt: Optional[bytes] = None) -> bytes:
        """Generate encryption key"""
        if password:
            if not salt:
                salt = os.urandom(16)
            
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=self.key_size.value // 8,
                salt=salt,
                iterations=100000,
                backend=self.backend
            )
            return kdf.derive(password.encode())
        else:
            return os.urandom(self.key_size.value // 8)
    
    def generate_iv(self, size: int = 16) -> bytes:
        """Generate initialization vector"""
        return os.urandom(size)


class AESEncryption(SymmetricCryptoBase):
    """AES encryption implementation"""
    
    def __init__(self, key_size: KeySize = KeySize.AES_256, mode: EncryptionMode = EncryptionMode.GCM):
        super().__init__(SymmetricAlgorithm.AES, key_size)
        self.mode = mode
    
    def encrypt(self, data: bytes, key: bytes, iv: Optional[bytes] = None) -> EncryptionResult:
        """Encrypt data using AES"""
        try:
            if not iv:
                iv = self.generate_iv()
            
            if self.mode == EncryptionMode.GCM:
                cipher = Cipher(
                    algorithms.AES(key),
                    modes.GCM(iv),
                    backend=self.backend
                )
                encryptor = cipher.encryptor()
                ciphertext = encryptor.update(data) + encryptor.finalize()
                
                return EncryptionResult(
                    ciphertext=ciphertext,
                    iv=iv,
                    tag=encryptor.tag,
                    algorithm=self.algorithm,
                    mode=self.mode,
                    key_size=self.key_size
                )
            
            elif self.mode == EncryptionMode.CBC:
                cipher = Cipher(
                    algorithms.AES(key),
                    modes.CBC(iv),
                    backend=self.backend
                )
                encryptor = cipher.encryptor()
                padder = PKCS7(128).padder()
                padded_data = padder.update(data) + padder.finalize()
                ciphertext = encryptor.update(padded_data) + encryptor.finalize()
                
                return EncryptionResult(
                    ciphertext=ciphertext,
                    iv=iv,
                    algorithm=self.algorithm,
                    mode=self.mode,
                    key_size=self.key_size
                )
            
            elif self.mode == EncryptionMode.CTR:
                cipher = Cipher(
                    algorithms.AES(key),
                    modes.CTR(iv),
                    backend=self.backend
                )
                encryptor = cipher.encryptor()
                ciphertext = encryptor.update(data) + encryptor.finalize()
                
                return EncryptionResult(
                    ciphertext=ciphertext,
                    iv=iv,
                    algorithm=self.algorithm,
                    mode=self.mode,
                    key_size=self.key_size
                )
            
            else:
                raise ValueError(f"Unsupported AES mode: {self.mode}")
                
        except Exception as e:
            logger.error(f"AES encryption failed: {e}")
            raise
    
    def decrypt(self, data: bytes, key: bytes, iv: bytes, tag: Optional[bytes] = None) -> DecryptionResult:
        """Decrypt data using AES"""
        try:
            if self.mode == EncryptionMode.GCM:
                cipher = Cipher(
                    algorithms.AES(key),
                    modes.GCM(iv, tag),
                    backend=self.backend
                )
                decryptor = cipher.decryptor()
                plaintext = decryptor.update(data) + decryptor.finalize()
                
                return DecryptionResult(
                    plaintext=plaintext,
                    verified=True,
                    algorithm=self.algorithm,
                    mode=self.mode
                )
            
            elif self.mode == EncryptionMode.CBC:
                cipher = Cipher(
                    algorithms.AES(key),
                    modes.CBC(iv),
                    backend=self.backend
                )
                decryptor = cipher.decryptor()
                padded_plaintext = decryptor.update(data) + decryptor.finalize()
                unpadder = PKCS7(128).unpadder()
                plaintext = unpadder.update(padded_plaintext) + unpadder.finalize()
                
                return DecryptionResult(
                    plaintext=plaintext,
                    verified=True,
                    algorithm=self.algorithm,
                    mode=self.mode
                )
            
            elif self.mode == EncryptionMode.CTR:
                cipher = Cipher(
                    algorithms.AES(key),
                    modes.CTR(iv),
                    backend=self.backend
                )
                decryptor = cipher.decryptor()
                plaintext = decryptor.update(data) + decryptor.finalize()
                
                return DecryptionResult(
                    plaintext=plaintext,
                    verified=True,
                    algorithm=self.algorithm,
                    mode=self.mode
                )
            
            else:
                raise ValueError(f"Unsupported AES mode: {self.mode}")
                
        except Exception as e:
            logger.error(f"AES decryption failed: {e}")
            raise


class ChaCha20Encryption(SymmetricCryptoBase):
    """ChaCha20 encryption implementation"""
    
    def __init__(self):
        super().__init__(SymmetricAlgorithm.CHACHA20, KeySize.CHACHA20_256)
    
    def encrypt(self, data: bytes, key: bytes, nonce: Optional[bytes] = None) -> EncryptionResult:
        """Encrypt data using ChaCha20"""
        try:
            if not nonce:
                nonce = os.urandom(12)
            
            cipher = Cipher(
                algorithms.ChaCha20(key, nonce),
                mode=None,
                backend=self.backend
            )
            encryptor = cipher.encryptor()
            ciphertext = encryptor.update(data) + encryptor.finalize()
            
            return EncryptionResult(
                ciphertext=ciphertext,
                iv=nonce,
                algorithm=self.algorithm,
                mode=EncryptionMode.CTR,  # ChaCha20 is similar to CTR mode
                key_size=self.key_size
            )
            
        except Exception as e:
            logger.error(f"ChaCha20 encryption failed: {e}")
            raise
    
    def decrypt(self, data: bytes, key: bytes, nonce: bytes) -> DecryptionResult:
        """Decrypt data using ChaCha20"""
        try:
            cipher = Cipher(
                algorithms.ChaCha20(key, nonce),
                mode=None,
                backend=self.backend
            )
            decryptor = cipher.decryptor()
            plaintext = decryptor.update(data) + decryptor.finalize()
            
            return DecryptionResult(
                plaintext=plaintext,
                verified=True,
                algorithm=self.algorithm,
                mode=EncryptionMode.CTR
            )
            
        except Exception as e:
            logger.error(f"ChaCha20 decryption failed: {e}")
            raise


class FernetEncryption(SymmetricCryptoBase):
    """Fernet encryption implementation"""
    
    def __init__(self):
        super().__init__(SymmetricAlgorithm.FERNET, KeySize.AES_256)
    
    def encrypt(self, data: bytes, key: bytes, **kwargs) -> EncryptionResult:
        """Encrypt data using Fernet"""
        try:
            # Fernet expects a base64-encoded key
            if isinstance(key, bytes):
                key_b64 = base64.urlsafe_b64encode(key)
            else:
                key_b64 = key
            
            fernet = Fernet(key_b64)
            ciphertext = fernet.encrypt(data)
            
            return EncryptionResult(
                ciphertext=ciphertext,
                algorithm=self.algorithm,
                mode=EncryptionMode.CBC,  # Fernet uses AES-128 in CBC mode
                key_size=KeySize.AES_128
            )
            
        except Exception as e:
            logger.error(f"Fernet encryption failed: {e}")
            raise
    
    def decrypt(self, data: bytes, key: bytes, **kwargs) -> DecryptionResult:
        """Decrypt data using Fernet"""
        try:
            # Fernet expects a base64-encoded key
            if isinstance(key, bytes):
                key_b64 = base64.urlsafe_b64encode(key)
            else:
                key_b64 = key
            
            fernet = Fernet(key_b64)
            plaintext = fernet.decrypt(data)
            
            return DecryptionResult(
                plaintext=plaintext,
                verified=True,
                algorithm=self.algorithm,
                mode=EncryptionMode.CBC
            )
            
        except Exception as e:
            logger.error(f"Fernet decryption failed: {e}")
            raise


class BlowfishEncryption(SymmetricCryptoBase):
    """Blowfish encryption implementation"""
    
    def __init__(self):
        super().__init__(SymmetricAlgorithm.BLOWFISH, KeySize.BLOWFISH_448)
    
    def encrypt(self, data: bytes, key: bytes, iv: Optional[bytes] = None) -> EncryptionResult:
        """Encrypt data using Blowfish"""
        try:
            if not iv:
                iv = self.generate_iv(8)  # Blowfish uses 64-bit blocks
            
            cipher = Cipher(
                algorithms.Blowfish(key),
                modes.CBC(iv),
                backend=self.backend
            )
            encryptor = cipher.encryptor()
            padder = PKCS7(64).padder()  # Blowfish block size is 64 bits
            padded_data = padder.update(data) + padder.finalize()
            ciphertext = encryptor.update(padded_data) + encryptor.finalize()
            
            return EncryptionResult(
                ciphertext=ciphertext,
                iv=iv,
                algorithm=self.algorithm,
                mode=EncryptionMode.CBC,
                key_size=self.key_size
            )
            
        except Exception as e:
            logger.error(f"Blowfish encryption failed: {e}")
            raise
    
    def decrypt(self, data: bytes, key: bytes, iv: bytes, **kwargs) -> DecryptionResult:
        """Decrypt data using Blowfish"""
        try:
            cipher = Cipher(
                algorithms.Blowfish(key),
                modes.CBC(iv),
                backend=self.backend
            )
            decryptor = cipher.decryptor()
            padded_plaintext = decryptor.update(data) + decryptor.finalize()
            unpadder = PKCS7(64).unpadder()
            plaintext = unpadder.update(padded_plaintext) + unpadder.finalize()
            
            return DecryptionResult(
                plaintext=plaintext,
                verified=True,
                algorithm=self.algorithm,
                mode=EncryptionMode.CBC
            )
            
        except Exception as e:
            logger.error(f"Blowfish decryption failed: {e}")
            raise


class TwofishEncryption(SymmetricCryptoBase):
    """Twofish encryption implementation (placeholder)"""
    
    def __init__(self):
        super().__init__(SymmetricAlgorithm.TWOFISH, KeySize.TWOFISH_256)
        logger.warning("Twofish implementation is a placeholder - using AES-256 instead")
    
    def encrypt(self, data: bytes, key: bytes, **kwargs) -> EncryptionResult:
        """Encrypt data using Twofish (fallback to AES)"""
        # Fallback to AES-256 since Twofish is not available in cryptography library
        aes = AESEncryption(KeySize.AES_256, EncryptionMode.GCM)
        return aes.encrypt(data, key, **kwargs)
    
    def decrypt(self, data: bytes, key: bytes, **kwargs) -> DecryptionResult:
        """Decrypt data using Twofish (fallback to AES)"""
        # Fallback to AES-256 since Twofish is not available in cryptography library
        aes = AESEncryption(KeySize.AES_256, EncryptionMode.GCM)
        return aes.decrypt(data, key, **kwargs)


class SymmetricCryptoService:
    """Main symmetric cryptography service"""
    
    def __init__(self):
        self.algorithms: Dict[SymmetricAlgorithm, SymmetricCryptoBase] = {
            SymmetricAlgorithm.AES: AESEncryption(),
            SymmetricAlgorithm.CHACHA20: ChaCha20Encryption(),
            SymmetricAlgorithm.FERNET: FernetEncryption(),
            SymmetricAlgorithm.BLOWFISH: BlowfishEncryption(),
            SymmetricAlgorithm.TWOFISH: TwofishEncryption()
        }
    
    def encrypt(
        self,
        data: bytes,
        algorithm: SymmetricAlgorithm = SymmetricAlgorithm.AES,
        key: Optional[bytes] = None,
        password: Optional[str] = None,
        **kwargs
    ) -> EncryptionResult:
        """Encrypt data using specified algorithm"""
        if algorithm not in self.algorithms:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        
        crypto = self.algorithms[algorithm]
        
        if key is None:
            if password is None:
                raise ValueError("Either key or password must be provided")
            key = crypto.generate_key(password)
        
        return crypto.encrypt(data, key, **kwargs)
    
    def decrypt(
        self,
        data: bytes,
        key: bytes,
        algorithm: SymmetricAlgorithm = SymmetricAlgorithm.AES,
        **kwargs
    ) -> DecryptionResult:
        """Decrypt data using specified algorithm"""
        if algorithm not in self.algorithms:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        
        crypto = self.algorithms[algorithm]
        return crypto.decrypt(data, key, **kwargs)
    
    def generate_key(
        self,
        algorithm: SymmetricAlgorithm = SymmetricAlgorithm.AES,
        password: Optional[str] = None
    ) -> bytes:
        """Generate key for specified algorithm"""
        if algorithm not in self.algorithms:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        
        crypto = self.algorithms[algorithm]
        return crypto.generate_key(password)
    
    def get_supported_algorithms(self) -> List[SymmetricAlgorithm]:
        """Get list of supported algorithms"""
        return list(self.algorithms.keys())


# Convenience functions
def encrypt_symmetric(
    data: bytes,
    algorithm: SymmetricAlgorithm = SymmetricAlgorithm.AES,
    key: Optional[bytes] = None,
    password: Optional[str] = None,
    **kwargs
) -> EncryptionResult:
    """Convenience function for symmetric encryption"""
    service = SymmetricCryptoService()
    return service.encrypt(data, algorithm, key, password, **kwargs)


def decrypt_symmetric(
    data: bytes,
    key: bytes,
    algorithm: SymmetricAlgorithm = SymmetricAlgorithm.AES,
    **kwargs
) -> DecryptionResult:
    """Convenience function for symmetric decryption"""
    service = SymmetricCryptoService()
    return service.decrypt(data, key, algorithm, **kwargs)


def generate_symmetric_key(
    algorithm: SymmetricAlgorithm = SymmetricAlgorithm.AES,
    password: Optional[str] = None
) -> bytes:
    """Convenience function for generating symmetric keys"""
    service = SymmetricCryptoService()
    return service.generate_key(algorithm, password)


# Example usage
if __name__ == "__main__":
    # Example symmetric encryption
    print("🔐 Symmetric Cryptography Example")
    
    # Test data
    test_data = b"Hello, Video-OpusClip! This is a test message for symmetric encryption."
    
    # Test AES encryption
    print("\n" + "="*60)
    print("AES ENCRYPTION")
    print("="*60)
    
    aes_key = generate_symmetric_key(SymmetricAlgorithm.AES)
    aes_result = encrypt_symmetric(test_data, SymmetricAlgorithm.AES, aes_key)
    print(f"✅ AES encrypted: {len(aes_result.ciphertext)} bytes")
    print(f"   IV: {aes_result.iv.hex()[:16]}...")
    print(f"   Tag: {aes_result.tag.hex()[:16]}..." if aes_result.tag else "   Tag: None")
    
    aes_decrypted = decrypt_symmetric(aes_result.ciphertext, aes_key, SymmetricAlgorithm.AES, 
                                     iv=aes_result.iv, tag=aes_result.tag)
    print(f"✅ AES decrypted: {aes_decrypted.plaintext.decode()}")
    
    # Test ChaCha20 encryption
    print("\n" + "="*60)
    print("CHACHA20 ENCRYPTION")
    print("="*60)
    
    chacha_key = generate_symmetric_key(SymmetricAlgorithm.CHACHA20)
    chacha_result = encrypt_symmetric(test_data, SymmetricAlgorithm.CHACHA20, chacha_key)
    print(f"✅ ChaCha20 encrypted: {len(chacha_result.ciphertext)} bytes")
    print(f"   Nonce: {chacha_result.iv.hex()[:16]}...")
    
    chacha_decrypted = decrypt_symmetric(chacha_result.ciphertext, chacha_key, SymmetricAlgorithm.CHACHA20,
                                        nonce=chacha_result.iv)
    print(f"✅ ChaCha20 decrypted: {chacha_decrypted.plaintext.decode()}")
    
    # Test Fernet encryption
    print("\n" + "="*60)
    print("FERNET ENCRYPTION")
    print("="*60)
    
    fernet_key = generate_symmetric_key(SymmetricAlgorithm.FERNET)
    fernet_result = encrypt_symmetric(test_data, SymmetricAlgorithm.FERNET, fernet_key)
    print(f"✅ Fernet encrypted: {len(fernet_result.ciphertext)} bytes")
    
    fernet_decrypted = decrypt_symmetric(fernet_result.ciphertext, fernet_key, SymmetricAlgorithm.FERNET)
    print(f"✅ Fernet decrypted: {fernet_decrypted.plaintext.decode()}")
    
    # Test password-based encryption
    print("\n" + "="*60)
    print("PASSWORD-BASED ENCRYPTION")
    print("="*60)
    
    password = "my-secure-password-123"
    password_result = encrypt_symmetric(test_data, SymmetricAlgorithm.AES, password=password)
    print(f"✅ Password-based encrypted: {len(password_result.ciphertext)} bytes")
    print(f"   Salt: {password_result.salt.hex()[:16]}..." if password_result.salt else "   Salt: None")
    
    password_decrypted = decrypt_symmetric(password_result.ciphertext, password_result.salt, SymmetricAlgorithm.AES,
                                         iv=password_result.iv, tag=password_result.tag)
    print(f"✅ Password-based decrypted: {password_decrypted.plaintext.decode()}")
    
    print("\n✅ Symmetric cryptography example completed!") 