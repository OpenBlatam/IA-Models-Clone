#!/usr/bin/env python3
"""
Hash Functions for Video-OpusClip
Hash functions, password hashing, and salt generation
"""

import os
import base64
import hashlib
from typing import Dict, List, Any, Optional, Union, Tuple
from enum import Enum
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
from cryptography.hazmat.backends import default_backend
import bcrypt
import argon2

logger = logging.getLogger(__name__)


class HashAlgorithm(Enum):
    """Hash algorithms"""
    SHA256 = "sha256"
    SHA512 = "sha512"
    BLAKE2B = "blake2b"
    ARGON2 = "argon2"
    BCRYPT = "bcrypt"
    SCRYPT = "scrypt"
    PBKDF2 = "pbkdf2"


class SaltGenerator(Enum):
    """Salt generation methods"""
    RANDOM = "random"
    CRYPTO = "crypto"
    SYSTEM = "system"


@dataclass
class HashResult:
    """Result of hashing operation"""
    hash_value: bytes
    salt: Optional[bytes] = None
    algorithm: HashAlgorithm = HashAlgorithm.SHA256
    iterations: Optional[int] = None
    memory_cost: Optional[int] = None
    parallelism: Optional[int] = None


@dataclass
class VerificationResult:
    """Result of hash verification"""
    verified: bool
    algorithm: HashAlgorithm
    message: str = ""


class HashBase(ABC):
    """Base class for hash functions"""
    
    def __init__(self, algorithm: HashAlgorithm):
        self.algorithm = algorithm
        self.backend = default_backend()
    
    @abstractmethod
    def hash(self, data: bytes, salt: Optional[bytes] = None, **kwargs) -> HashResult:
        """Hash data"""
        pass
    
    @abstractmethod
    def verify(self, data: bytes, hash_value: bytes, salt: Optional[bytes] = None, **kwargs) -> VerificationResult:
        """Verify hash"""
        pass
    
    def generate_salt(self, size: int = 16, method: SaltGenerator = SaltGenerator.CRYPTO) -> bytes:
        """Generate salt"""
        if method == SaltGenerator.CRYPTO:
            return os.urandom(size)
        elif method == SaltGenerator.SYSTEM:
            return os.urandom(size)
        else:
            return os.urandom(size)


class SHA256Hash(HashBase):
    """SHA-256 hash implementation"""
    
    def __init__(self):
        super().__init__(HashAlgorithm.SHA256)
    
    def hash(self, data: bytes, salt: Optional[bytes] = None, **kwargs) -> HashResult:
        """Hash data with SHA-256"""
        try:
            if salt:
                data_with_salt = salt + data
            else:
                data_with_salt = data
            
            hash_value = hashlib.sha256(data_with_salt).digest()
            
            return HashResult(
                hash_value=hash_value,
                salt=salt,
                algorithm=self.algorithm
            )
            
        except Exception as e:
            logger.error(f"SHA-256 hashing failed: {e}")
            raise
    
    def verify(self, data: bytes, hash_value: bytes, salt: Optional[bytes] = None, **kwargs) -> VerificationResult:
        """Verify SHA-256 hash"""
        try:
            computed_hash = self.hash(data, salt).hash_value
            verified = computed_hash == hash_value
            
            return VerificationResult(
                verified=verified,
                algorithm=self.algorithm,
                message="Hash verification successful" if verified else "Hash verification failed"
            )
            
        except Exception as e:
            logger.error(f"SHA-256 verification failed: {e}")
            return VerificationResult(
                verified=False,
                algorithm=self.algorithm,
                message=f"Verification error: {e}"
            )


class SHA512Hash(HashBase):
    """SHA-512 hash implementation"""
    
    def __init__(self):
        super().__init__(HashAlgorithm.SHA512)
    
    def hash(self, data: bytes, salt: Optional[bytes] = None, **kwargs) -> HashResult:
        """Hash data with SHA-512"""
        try:
            if salt:
                data_with_salt = salt + data
            else:
                data_with_salt = data
            
            hash_value = hashlib.sha512(data_with_salt).digest()
            
            return HashResult(
                hash_value=hash_value,
                salt=salt,
                algorithm=self.algorithm
            )
            
        except Exception as e:
            logger.error(f"SHA-512 hashing failed: {e}")
            raise
    
    def verify(self, data: bytes, hash_value: bytes, salt: Optional[bytes] = None, **kwargs) -> VerificationResult:
        """Verify SHA-512 hash"""
        try:
            computed_hash = self.hash(data, salt).hash_value
            verified = computed_hash == hash_value
            
            return VerificationResult(
                verified=verified,
                algorithm=self.algorithm,
                message="Hash verification successful" if verified else "Hash verification failed"
            )
            
        except Exception as e:
            logger.error(f"SHA-512 verification failed: {e}")
            return VerificationResult(
                verified=False,
                algorithm=self.algorithm,
                message=f"Verification error: {e}"
            )


class Blake2bHash(HashBase):
    """Blake2b hash implementation"""
    
    def __init__(self, digest_size: int = 64):
        super().__init__(HashAlgorithm.BLAKE2B)
        self.digest_size = digest_size
    
    def hash(self, data: bytes, salt: Optional[bytes] = None, **kwargs) -> HashResult:
        """Hash data with Blake2b"""
        try:
            if salt:
                hash_value = hashlib.blake2b(data, salt=salt, digest_size=self.digest_size).digest()
            else:
                hash_value = hashlib.blake2b(data, digest_size=self.digest_size).digest()
            
            return HashResult(
                hash_value=hash_value,
                salt=salt,
                algorithm=self.algorithm
            )
            
        except Exception as e:
            logger.error(f"Blake2b hashing failed: {e}")
            raise
    
    def verify(self, data: bytes, hash_value: bytes, salt: Optional[bytes] = None, **kwargs) -> VerificationResult:
        """Verify Blake2b hash"""
        try:
            computed_hash = self.hash(data, salt).hash_value
            verified = computed_hash == hash_value
            
            return VerificationResult(
                verified=verified,
                algorithm=self.algorithm,
                message="Hash verification successful" if verified else "Hash verification failed"
            )
            
        except Exception as e:
            logger.error(f"Blake2b verification failed: {e}")
            return VerificationResult(
                verified=False,
                algorithm=self.algorithm,
                message=f"Verification error: {e}"
            )


class Argon2Hash(HashBase):
    """Argon2 hash implementation"""
    
    def __init__(self, time_cost: int = 3, memory_cost: int = 65536, parallelism: int = 4):
        super().__init__(HashAlgorithm.ARGON2)
        self.time_cost = time_cost
        self.memory_cost = memory_cost
        self.parallelism = parallelism
        self.argon2_hasher = argon2.PasswordHasher(
            time_cost=time_cost,
            memory_cost=memory_cost,
            parallelism=parallelism
        )
    
    def hash(self, data: bytes, salt: Optional[bytes] = None, **kwargs) -> HashResult:
        """Hash data with Argon2"""
        try:
            # Argon2 expects string input
            data_str = data.decode('utf-8') if isinstance(data, bytes) else str(data)
            
            hash_value = self.argon2_hasher.hash(data_str)
            
            return HashResult(
                hash_value=hash_value.encode(),
                salt=salt,
                algorithm=self.algorithm,
                iterations=self.time_cost,
                memory_cost=self.memory_cost,
                parallelism=self.parallelism
            )
            
        except Exception as e:
            logger.error(f"Argon2 hashing failed: {e}")
            raise
    
    def verify(self, data: bytes, hash_value: bytes, salt: Optional[bytes] = None, **kwargs) -> VerificationResult:
        """Verify Argon2 hash"""
        try:
            # Argon2 expects string input
            data_str = data.decode('utf-8') if isinstance(data, bytes) else str(data)
            hash_str = hash_value.decode('utf-8') if isinstance(hash_value, bytes) else str(hash_value)
            
            verified = self.argon2_hasher.verify(hash_str, data_str)
            
            return VerificationResult(
                verified=verified,
                algorithm=self.algorithm,
                message="Hash verification successful" if verified else "Hash verification failed"
            )
            
        except Exception as e:
            logger.error(f"Argon2 verification failed: {e}")
            return VerificationResult(
                verified=False,
                algorithm=self.algorithm,
                message=f"Verification error: {e}"
            )


class BcryptHash(HashBase):
    """Bcrypt hash implementation"""
    
    def __init__(self, rounds: int = 12):
        super().__init__(HashAlgorithm.BCRYPT)
        self.rounds = rounds
    
    def hash(self, data: bytes, salt: Optional[bytes] = None, **kwargs) -> HashResult:
        """Hash data with bcrypt"""
        try:
            # Bcrypt expects string input
            data_str = data.decode('utf-8') if isinstance(data, bytes) else str(data)
            
            if salt:
                # Use provided salt
                hash_value = bcrypt.hashpw(data_str.encode(), salt)
            else:
                # Generate new salt
                hash_value = bcrypt.hashpw(data_str.encode(), bcrypt.gensalt(self.rounds))
            
            return HashResult(
                hash_value=hash_value,
                salt=salt,
                algorithm=self.algorithm,
                iterations=self.rounds
            )
            
        except Exception as e:
            logger.error(f"Bcrypt hashing failed: {e}")
            raise
    
    def verify(self, data: bytes, hash_value: bytes, salt: Optional[bytes] = None, **kwargs) -> VerificationResult:
        """Verify bcrypt hash"""
        try:
            # Bcrypt expects string input
            data_str = data.decode('utf-8') if isinstance(data, bytes) else str(data)
            
            verified = bcrypt.checkpw(data_str.encode(), hash_value)
            
            return VerificationResult(
                verified=verified,
                algorithm=self.algorithm,
                message="Hash verification successful" if verified else "Hash verification failed"
            )
            
        except Exception as e:
            logger.error(f"Bcrypt verification failed: {e}")
            return VerificationResult(
                verified=False,
                algorithm=self.algorithm,
                message=f"Verification error: {e}"
            )


class ScryptHash(HashBase):
    """Scrypt hash implementation"""
    
    def __init__(self, n: int = 16384, r: int = 8, p: int = 1):
        super().__init__(HashAlgorithm.SCRYPT)
        self.n = n  # CPU/memory cost
        self.r = r  # Block size
        self.p = p  # Parallelization
    
    def hash(self, data: bytes, salt: Optional[bytes] = None, **kwargs) -> HashResult:
        """Hash data with scrypt"""
        try:
            if not salt:
                salt = self.generate_salt(16)
            
            kdf = Scrypt(
                salt=salt,
                length=32,
                n=self.n,
                r=self.r,
                p=self.p,
                backend=self.backend
            )
            
            hash_value = kdf.derive(data)
            
            return HashResult(
                hash_value=hash_value,
                salt=salt,
                algorithm=self.algorithm,
                iterations=self.n
            )
            
        except Exception as e:
            logger.error(f"Scrypt hashing failed: {e}")
            raise
    
    def verify(self, data: bytes, hash_value: bytes, salt: Optional[bytes] = None, **kwargs) -> VerificationResult:
        """Verify scrypt hash"""
        try:
            if not salt:
                return VerificationResult(
                    verified=False,
                    algorithm=self.algorithm,
                    message="Salt is required for scrypt verification"
                )
            
            kdf = Scrypt(
                salt=salt,
                length=32,
                n=self.n,
                r=self.r,
                p=self.p,
                backend=self.backend
            )
            
            try:
                kdf.verify(data, hash_value)
                verified = True
                message = "Hash verification successful"
            except Exception:
                verified = False
                message = "Hash verification failed"
            
            return VerificationResult(
                verified=verified,
                algorithm=self.algorithm,
                message=message
            )
            
        except Exception as e:
            logger.error(f"Scrypt verification failed: {e}")
            return VerificationResult(
                verified=False,
                algorithm=self.algorithm,
                message=f"Verification error: {e}"
            )


class PBKDF2Hash(HashBase):
    """PBKDF2 hash implementation"""
    
    def __init__(self, iterations: int = 100000):
        super().__init__(HashAlgorithm.PBKDF2)
        self.iterations = iterations
    
    def hash(self, data: bytes, salt: Optional[bytes] = None, **kwargs) -> HashResult:
        """Hash data with PBKDF2"""
        try:
            if not salt:
                salt = self.generate_salt(16)
            
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=self.iterations,
                backend=self.backend
            )
            
            hash_value = kdf.derive(data)
            
            return HashResult(
                hash_value=hash_value,
                salt=salt,
                algorithm=self.algorithm,
                iterations=self.iterations
            )
            
        except Exception as e:
            logger.error(f"PBKDF2 hashing failed: {e}")
            raise
    
    def verify(self, data: bytes, hash_value: bytes, salt: Optional[bytes] = None, **kwargs) -> VerificationResult:
        """Verify PBKDF2 hash"""
        try:
            if not salt:
                return VerificationResult(
                    verified=False,
                    algorithm=self.algorithm,
                    message="Salt is required for PBKDF2 verification"
                )
            
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=self.iterations,
                backend=self.backend
            )
            
            try:
                kdf.verify(data, hash_value)
                verified = True
                message = "Hash verification successful"
            except Exception:
                verified = False
                message = "Hash verification failed"
            
            return VerificationResult(
                verified=verified,
                algorithm=self.algorithm,
                message=message
            )
            
        except Exception as e:
            logger.error(f"PBKDF2 verification failed: {e}")
            return VerificationResult(
                verified=False,
                algorithm=self.algorithm,
                message=f"Verification error: {e}"
            )


class HashService:
    """Main hash service"""
    
    def __init__(self):
        self.algorithms: Dict[HashAlgorithm, HashBase] = {
            HashAlgorithm.SHA256: SHA256Hash(),
            HashAlgorithm.SHA512: SHA512Hash(),
            HashAlgorithm.BLAKE2B: Blake2bHash(),
            HashAlgorithm.ARGON2: Argon2Hash(),
            HashAlgorithm.BCRYPT: BcryptHash(),
            HashAlgorithm.SCRYPT: ScryptHash(),
            HashAlgorithm.PBKDF2: PBKDF2Hash()
        }
    
    def hash(
        self,
        data: bytes,
        algorithm: HashAlgorithm = HashAlgorithm.SHA256,
        salt: Optional[bytes] = None,
        **kwargs
    ) -> HashResult:
        """Hash data using specified algorithm"""
        if algorithm not in self.algorithms:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        
        hash_func = self.algorithms[algorithm]
        return hash_func.hash(data, salt, **kwargs)
    
    def verify(
        self,
        data: bytes,
        hash_value: bytes,
        algorithm: HashAlgorithm = HashAlgorithm.SHA256,
        salt: Optional[bytes] = None,
        **kwargs
    ) -> VerificationResult:
        """Verify hash using specified algorithm"""
        if algorithm not in self.algorithms:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        
        hash_func = self.algorithms[algorithm]
        return hash_func.verify(data, hash_value, salt, **kwargs)
    
    def get_supported_algorithms(self) -> List[HashAlgorithm]:
        """Get list of supported algorithms"""
        return list(self.algorithms.keys())


# Convenience functions
def hash_data(
    data: bytes,
    algorithm: HashAlgorithm = HashAlgorithm.SHA256,
    salt: Optional[bytes] = None,
    **kwargs
) -> HashResult:
    """Convenience function for hashing data"""
    service = HashService()
    return service.hash(data, algorithm, salt, **kwargs)


def verify_hash(
    data: bytes,
    hash_value: bytes,
    algorithm: HashAlgorithm = HashAlgorithm.SHA256,
    salt: Optional[bytes] = None,
    **kwargs
) -> VerificationResult:
    """Convenience function for verifying hashes"""
    service = HashService()
    return service.verify(data, hash_value, algorithm, salt, **kwargs)


def bcrypt_hash(data: bytes, rounds: int = 12) -> HashResult:
    """Convenience function for bcrypt hashing"""
    bcrypt_hash_func = BcryptHash(rounds)
    return bcrypt_hash_func.hash(data)


def scrypt_hash(data: bytes, n: int = 16384, r: int = 8, p: int = 1) -> HashResult:
    """Convenience function for scrypt hashing"""
    scrypt_hash_func = ScryptHash(n, r, p)
    return scrypt_hash_func.hash(data)


def pbkdf2_hash(data: bytes, iterations: int = 100000) -> HashResult:
    """Convenience function for PBKDF2 hashing"""
    pbkdf2_hash_func = PBKDF2Hash(iterations)
    return pbkdf2_hash_func.hash(data)


# Example usage
if __name__ == "__main__":
    # Example hash functions
    print("🔐 Hash Functions Example")
    
    # Test data
    test_data = b"Hello, Video-OpusClip! This is a test message for hashing."
    password = b"my-secure-password-123"
    
    # Test SHA-256
    print("\n" + "="*60)
    print("SHA-256 HASHING")
    print("="*60)
    
    sha256_result = hash_data(test_data, HashAlgorithm.SHA256)
    print(f"✅ SHA-256 hash: {sha256_result.hash_value.hex()}")
    
    sha256_verified = verify_hash(test_data, sha256_result.hash_value, HashAlgorithm.SHA256)
    print(f"✅ SHA-256 verification: {sha256_verified.verified}")
    
    # Test SHA-512
    print("\n" + "="*60)
    print("SHA-512 HASHING")
    print("="*60)
    
    sha512_result = hash_data(test_data, HashAlgorithm.SHA512)
    print(f"✅ SHA-512 hash: {sha512_result.hash_value.hex()}")
    
    sha512_verified = verify_hash(test_data, sha512_result.hash_value, HashAlgorithm.SHA512)
    print(f"✅ SHA-512 verification: {sha512_verified.verified}")
    
    # Test Blake2b
    print("\n" + "="*60)
    print("BLAKE2B HASHING")
    print("="*60)
    
    blake2b_result = hash_data(test_data, HashAlgorithm.BLAKE2B)
    print(f"✅ Blake2b hash: {blake2b_result.hash_value.hex()}")
    
    blake2b_verified = verify_hash(test_data, blake2b_result.hash_value, HashAlgorithm.BLAKE2B)
    print(f"✅ Blake2b verification: {blake2b_verified.verified}")
    
    # Test Argon2 (password hashing)
    print("\n" + "="*60)
    print("ARGON2 PASSWORD HASHING")
    print("="*60)
    
    argon2_result = hash_data(password, HashAlgorithm.ARGON2)
    print(f"✅ Argon2 hash: {argon2_result.hash_value.decode()[:50]}...")
    print(f"   Memory cost: {argon2_result.memory_cost}")
    print(f"   Parallelism: {argon2_result.parallelism}")
    
    argon2_verified = verify_hash(password, argon2_result.hash_value, HashAlgorithm.ARGON2)
    print(f"✅ Argon2 verification: {argon2_verified.verified}")
    
    # Test bcrypt (password hashing)
    print("\n" + "="*60)
    print("BCRYPT PASSWORD HASHING")
    print("="*60)
    
    bcrypt_result = bcrypt_hash(password)
    print(f"✅ Bcrypt hash: {bcrypt_result.hash_value.decode()[:50]}...")
    print(f"   Rounds: {bcrypt_result.iterations}")
    
    bcrypt_verified = verify_hash(password, bcrypt_result.hash_value, HashAlgorithm.BCRYPT)
    print(f"✅ Bcrypt verification: {bcrypt_verified.verified}")
    
    # Test scrypt (password hashing)
    print("\n" + "="*60)
    print("SCRYPT PASSWORD HASHING")
    print("="*60)
    
    scrypt_result = scrypt_hash(password)
    print(f"✅ Scrypt hash: {scrypt_result.hash_value.hex()}")
    print(f"   Salt: {scrypt_result.salt.hex()}")
    print(f"   Iterations: {scrypt_result.iterations}")
    
    scrypt_verified = verify_hash(password, scrypt_result.hash_value, HashAlgorithm.SCRYPT, salt=scrypt_result.salt)
    print(f"✅ Scrypt verification: {scrypt_verified.verified}")
    
    # Test PBKDF2 (password hashing)
    print("\n" + "="*60)
    print("PBKDF2 PASSWORD HASHING")
    print("="*60)
    
    pbkdf2_result = pbkdf2_hash(password)
    print(f"✅ PBKDF2 hash: {pbkdf2_result.hash_value.hex()}")
    print(f"   Salt: {pbkdf2_result.salt.hex()}")
    print(f"   Iterations: {pbkdf2_result.iterations}")
    
    pbkdf2_verified = verify_hash(password, pbkdf2_result.hash_value, HashAlgorithm.PBKDF2, salt=pbkdf2_result.salt)
    print(f"✅ PBKDF2 verification: {pbkdf2_verified.verified}")
    
    # Test salt generation
    print("\n" + "="*60)
    print("SALT GENERATION")
    print("="*60)
    
    salt1 = os.urandom(16)
    salt2 = os.urandom(16)
    print(f"✅ Salt 1: {salt1.hex()}")
    print(f"✅ Salt 2: {salt2.hex()}")
    print(f"✅ Salts different: {salt1 != salt2}")
    
    print("\n✅ Hash functions example completed!") 