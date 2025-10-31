#!/usr/bin/env python3
"""
Crypto Helpers Module for Video-OpusClip
Cryptographic utilities and security functions
"""

import asyncio
import hashlib
import hmac
import secrets
import base64
import json
from typing import Dict, List, Any, Optional, Union, Tuple, Bytes
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
import os
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend

class HashAlgorithm(str, Enum):
    """Supported hash algorithms"""
    MD5 = "md5"
    SHA1 = "sha1"
    SHA256 = "sha256"
    SHA512 = "sha512"
    BLAKE2B = "blake2b"
    BLAKE2S = "blake2s"

class EncryptionAlgorithm(str, Enum):
    """Supported encryption algorithms"""
    AES = "aes"
    RSA = "rsa"
    FERNET = "fernet"
    CHACHA20 = "chacha20"

@dataclass
class CryptoConfig:
    """Configuration for cryptographic operations"""
    hash_algorithm: HashAlgorithm = HashAlgorithm.SHA256
    encryption_algorithm: EncryptionAlgorithm = EncryptionAlgorithm.AES
    key_size: int = 256
    salt_size: int = 32
    iterations: int = 100000
    encoding: str = "utf-8"

class CryptoHelpers:
    """Cryptographic utilities and security functions"""
    
    def __init__(self, config: Optional[CryptoConfig] = None):
        self.config = config or CryptoConfig()
        self._fernet_key: Optional[bytes] = None
        self._rsa_private_key: Optional[rsa.RSAPrivateKey] = None
        self._rsa_public_key: Optional[rsa.RSAPublicKey] = None
    
    def generate_salt(self, size: Optional[int] = None) -> bytes:
        """Generate a cryptographically secure salt"""
        salt_size = size or self.config.salt_size
        return secrets.token_bytes(salt_size)
    
    def generate_key(self, size: Optional[int] = None) -> bytes:
        """Generate a cryptographically secure key"""
        key_size = size or self.config.key_size
        return secrets.token_bytes(key_size // 8)
    
    def hash_data(self, data: Union[str, bytes], algorithm: Optional[HashAlgorithm] = None) -> str:
        """Hash data using specified algorithm"""
        if isinstance(data, str):
            data = data.encode(self.config.encoding)
        
        algo = algorithm or self.config.hash_algorithm
        
        if algo == HashAlgorithm.MD5:
            return hashlib.md5(data).hexdigest()
        elif algo == HashAlgorithm.SHA1:
            return hashlib.sha1(data).hexdigest()
        elif algo == HashAlgorithm.SHA256:
            return hashlib.sha256(data).hexdigest()
        elif algo == HashAlgorithm.SHA512:
            return hashlib.sha512(data).hexdigest()
        elif algo == HashAlgorithm.BLAKE2B:
            return hashlib.blake2b(data).hexdigest()
        elif algo == HashAlgorithm.BLAKE2S:
            return hashlib.blake2s(data).hexdigest()
        else:
            raise ValueError(f"Unsupported hash algorithm: {algo}")
    
    def hash_password(self, password: str, salt: Optional[bytes] = None) -> Dict[str, str]:
        """Hash password with salt using PBKDF2"""
        if salt is None:
            salt = self.generate_salt()
        
        # Use PBKDF2 for password hashing
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=self.config.iterations,
            backend=default_backend()
        )
        
        key = kdf.derive(password.encode(self.config.encoding))
        
        return {
            "hash": base64.b64encode(key).decode('ascii'),
            "salt": base64.b64encode(salt).decode('ascii'),
            "iterations": str(self.config.iterations)
        }
    
    def verify_password(self, password: str, stored_hash: str, stored_salt: str, iterations: int) -> bool:
        """Verify password against stored hash"""
        try:
            salt = base64.b64decode(stored_salt)
            expected_hash = base64.b64decode(stored_hash)
            
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=iterations,
                backend=default_backend()
            )
            
            kdf.verify(password.encode(self.config.encoding), expected_hash)
            return True
        except Exception:
            return False
    
    def generate_hmac(self, data: Union[str, bytes], key: Union[str, bytes], algorithm: Optional[HashAlgorithm] = None) -> str:
        """Generate HMAC for data"""
        if isinstance(data, str):
            data = data.encode(self.config.encoding)
        if isinstance(key, str):
            key = key.encode(self.config.encoding)
        
        algo = algorithm or self.config.hash_algorithm
        
        if algo == HashAlgorithm.SHA256:
            return hmac.new(key, data, hashlib.sha256).hexdigest()
        elif algo == HashAlgorithm.SHA512:
            return hmac.new(key, data, hashlib.sha512).hexdigest()
        else:
            raise ValueError(f"Unsupported HMAC algorithm: {algo}")
    
    def verify_hmac(self, data: Union[str, bytes], key: Union[str, bytes], expected_hmac: str, algorithm: Optional[HashAlgorithm] = None) -> bool:
        """Verify HMAC for data"""
        actual_hmac = self.generate_hmac(data, key, algorithm)
        return hmac.compare_digest(actual_hmac, expected_hmac)
    
    def generate_fernet_key(self) -> bytes:
        """Generate Fernet key for symmetric encryption"""
        if self._fernet_key is None:
            self._fernet_key = Fernet.generate_key()
        return self._fernet_key
    
    def encrypt_fernet(self, data: Union[str, bytes], key: Optional[bytes] = None) -> str:
        """Encrypt data using Fernet (symmetric encryption)"""
        if isinstance(data, str):
            data = data.encode(self.config.encoding)
        
        if key is None:
            key = self.generate_fernet_key()
        
        fernet = Fernet(key)
        encrypted_data = fernet.encrypt(data)
        return base64.b64encode(encrypted_data).decode('ascii')
    
    def decrypt_fernet(self, encrypted_data: str, key: bytes) -> str:
        """Decrypt data using Fernet"""
        fernet = Fernet(key)
        encrypted_bytes = base64.b64decode(encrypted_data)
        decrypted_data = fernet.decrypt(encrypted_bytes)
        return decrypted_data.decode(self.config.encoding)
    
    def generate_rsa_keypair(self, key_size: int = 2048) -> Tuple[bytes, bytes]:
        """Generate RSA key pair"""
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=key_size,
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
        
        return private_pem, public_pem
    
    def encrypt_rsa(self, data: Union[str, bytes], public_key_pem: bytes) -> str:
        """Encrypt data using RSA public key"""
        if isinstance(data, str):
            data = data.encode(self.config.encoding)
        
        public_key = serialization.load_pem_public_key(public_key_pem, backend=default_backend())
        
        encrypted_data = public_key.encrypt(
            data,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        return base64.b64encode(encrypted_data).decode('ascii')
    
    def decrypt_rsa(self, encrypted_data: str, private_key_pem: bytes) -> str:
        """Decrypt data using RSA private key"""
        private_key = serialization.load_pem_private_key(private_key_pem, backend=default_backend())
        
        encrypted_bytes = base64.b64decode(encrypted_data)
        decrypted_data = private_key.decrypt(
            encrypted_bytes,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        return decrypted_data.decode(self.config.encoding)
    
    def encrypt_aes(self, data: Union[str, bytes], key: Optional[bytes] = None, iv: Optional[bytes] = None) -> Dict[str, str]:
        """Encrypt data using AES"""
        if isinstance(data, str):
            data = data.encode(self.config.encoding)
        
        if key is None:
            key = self.generate_key(256)
        if iv is None:
            iv = secrets.token_bytes(16)
        
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
        encryptor = cipher.encryptor()
        
        # Pad data to block size
        padded_data = self._pad_data(data, 16)
        encrypted_data = encryptor.update(padded_data) + encryptor.finalize()
        
        return {
            "encrypted_data": base64.b64encode(encrypted_data).decode('ascii'),
            "key": base64.b64encode(key).decode('ascii'),
            "iv": base64.b64encode(iv).decode('ascii')
        }
    
    def decrypt_aes(self, encrypted_data: str, key: str, iv: str) -> str:
        """Decrypt data using AES"""
        encrypted_bytes = base64.b64decode(encrypted_data)
        key_bytes = base64.b64decode(key)
        iv_bytes = base64.b64decode(iv)
        
        cipher = Cipher(algorithms.AES(key_bytes), modes.CBC(iv_bytes), backend=default_backend())
        decryptor = cipher.decryptor()
        
        decrypted_data = decryptor.update(encrypted_bytes) + decryptor.finalize()
        unpadded_data = self._unpad_data(decrypted_data)
        
        return unpadded_data.decode(self.config.encoding)
    
    def _pad_data(self, data: bytes, block_size: int) -> bytes:
        """Pad data to block size using PKCS7"""
        padding_length = block_size - (len(data) % block_size)
        padding = bytes([padding_length] * padding_length)
        return data + padding
    
    def _unpad_data(self, data: bytes) -> bytes:
        """Remove PKCS7 padding"""
        padding_length = data[-1]
        return data[:-padding_length]
    
    def generate_secure_token(self, length: int = 32) -> str:
        """Generate a secure random token"""
        return secrets.token_urlsafe(length)
    
    def generate_uuid(self) -> str:
        """Generate a UUID"""
        import uuid
        return str(uuid.uuid4())
    
    def hash_file(self, file_path: str, algorithm: Optional[HashAlgorithm] = None) -> str:
        """Calculate hash of a file"""
        algo = algorithm or self.config.hash_algorithm
        
        if algo == HashAlgorithm.SHA256:
            hash_obj = hashlib.sha256()
        elif algo == HashAlgorithm.SHA512:
            hash_obj = hashlib.sha512()
        elif algo == HashAlgorithm.MD5:
            hash_obj = hashlib.md5()
        else:
            raise ValueError(f"Unsupported hash algorithm for files: {algo}")
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_obj.update(chunk)
        
        return hash_obj.hexdigest()
    
    async def hash_file_async(self, file_path: str, algorithm: Optional[HashAlgorithm] = None) -> str:
        """Calculate hash of a file asynchronously"""
        return await asyncio.to_thread(self.hash_file, file_path, algorithm)
    
    def encrypt_file(self, file_path: str, output_path: str, key: Optional[bytes] = None) -> Dict[str, str]:
        """Encrypt a file"""
        if key is None:
            key = self.generate_key(256)
        
        iv = secrets.token_bytes(16)
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
        encryptor = cipher.encryptor()
        
        with open(file_path, 'rb') as input_file, open(output_path, 'wb') as output_file:
            while True:
                chunk = input_file.read(4096)
                if not chunk:
                    break
                
                # Pad the last chunk if needed
                if len(chunk) < 4096:
                    chunk = self._pad_data(chunk, 16)
                
                encrypted_chunk = encryptor.update(chunk)
                output_file.write(encrypted_chunk)
            
            # Write final block
            final_chunk = encryptor.finalize()
            output_file.write(final_chunk)
        
        return {
            "key": base64.b64encode(key).decode('ascii'),
            "iv": base64.b64encode(iv).decode('ascii')
        }
    
    def decrypt_file(self, file_path: str, output_path: str, key: str, iv: str) -> None:
        """Decrypt a file"""
        key_bytes = base64.b64decode(key)
        iv_bytes = base64.b64decode(iv)
        
        cipher = Cipher(algorithms.AES(key_bytes), modes.CBC(iv_bytes), backend=default_backend())
        decryptor = cipher.decryptor()
        
        with open(file_path, 'rb') as input_file, open(output_path, 'wb') as output_file:
            while True:
                chunk = input_file.read(4096)
                if not chunk:
                    break
                
                decrypted_chunk = decryptor.update(chunk)
                output_file.write(decrypted_chunk)
            
            # Handle final block
            final_chunk = decryptor.finalize()
            if final_chunk:
                unpadded_chunk = self._unpad_data(final_chunk)
                output_file.write(unpadded_chunk)
    
    def create_digital_signature(self, data: Union[str, bytes], private_key_pem: bytes) -> str:
        """Create digital signature"""
        if isinstance(data, str):
            data = data.encode(self.config.encoding)
        
        private_key = serialization.load_pem_private_key(private_key_pem, backend=default_backend())
        
        signature = private_key.sign(
            data,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        
        return base64.b64encode(signature).decode('ascii')
    
    def verify_digital_signature(self, data: Union[str, bytes], signature: str, public_key_pem: bytes) -> bool:
        """Verify digital signature"""
        if isinstance(data, str):
            data = data.encode(self.config.encoding)
        
        try:
            public_key = serialization.load_pem_public_key(public_key_pem, backend=default_backend())
            signature_bytes = base64.b64decode(signature)
            
            public_key.verify(
                signature_bytes,
                data,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True
        except Exception:
            return False
    
    def generate_secure_password(self, length: int = 16, include_symbols: bool = True) -> str:
        """Generate a secure password"""
        import string
        
        characters = string.ascii_letters + string.digits
        if include_symbols:
            characters += "!@#$%^&*()_+-=[]{}|;:,.<>?"
        
        return ''.join(secrets.choice(characters) for _ in range(length))
    
    def validate_password_strength(self, password: str) -> Dict[str, Any]:
        """Validate password strength"""
        import re
        
        score = 0
        feedback = []
        
        # Length check
        if len(password) >= 8:
            score += 1
        else:
            feedback.append("Password should be at least 8 characters long")
        
        # Uppercase check
        if re.search(r'[A-Z]', password):
            score += 1
        else:
            feedback.append("Password should contain at least one uppercase letter")
        
        # Lowercase check
        if re.search(r'[a-z]', password):
            score += 1
        else:
            feedback.append("Password should contain at least one lowercase letter")
        
        # Digit check
        if re.search(r'\d', password):
            score += 1
        else:
            feedback.append("Password should contain at least one digit")
        
        # Symbol check
        if re.search(r'[!@#$%^&*()_+\-=\[\]{}|;:,.<>?]', password):
            score += 1
        else:
            feedback.append("Password should contain at least one special character")
        
        # Strength assessment
        if score >= 5:
            strength = "Very Strong"
        elif score >= 4:
            strength = "Strong"
        elif score >= 3:
            strength = "Moderate"
        elif score >= 2:
            strength = "Weak"
        else:
            strength = "Very Weak"
        
        return {
            "score": score,
            "max_score": 5,
            "strength": strength,
            "feedback": feedback,
            "is_strong": score >= 4
        }

# Example usage
async def main():
    """Example usage of crypto helpers"""
    print("🔐 Crypto Helpers Example")
    
    # Create crypto helpers
    crypto = CryptoHelpers()
    
    # Password hashing
    password = "my_secure_password"
    hash_result = crypto.hash_password(password)
    print(f"Password hash: {hash_result['hash']}")
    print(f"Salt: {hash_result['salt']}")
    
    # Verify password
    is_valid = crypto.verify_password(password, hash_result['hash'], hash_result['salt'], crypto.config.iterations)
    print(f"Password verification: {is_valid}")
    
    # Generate HMAC
    data = "sensitive_data"
    key = "secret_key"
    hmac_result = crypto.generate_hmac(data, key)
    print(f"HMAC: {hmac_result}")
    
    # Fernet encryption
    message = "Hello, encrypted world!"
    encrypted = crypto.encrypt_fernet(message)
    print(f"Encrypted: {encrypted}")
    
    # Generate RSA keypair
    private_key, public_key = crypto.generate_rsa_keypair()
    print(f"RSA Private Key: {private_key[:50]}...")
    print(f"RSA Public Key: {public_key[:50]}...")
    
    # RSA encryption
    rsa_encrypted = crypto.encrypt_rsa("Secret message", public_key)
    print(f"RSA Encrypted: {rsa_encrypted}")
    
    # AES encryption
    aes_result = crypto.encrypt_aes("AES encrypted data")
    print(f"AES Encrypted: {aes_result['encrypted_data']}")
    
    # Generate secure token
    token = crypto.generate_secure_token()
    print(f"Secure Token: {token}")
    
    # Password strength validation
    weak_password = "123"
    strong_password = "MySecureP@ssw0rd!"
    
    weak_result = crypto.validate_password_strength(weak_password)
    strong_result = crypto.validate_password_strength(strong_password)
    
    print(f"Weak password strength: {weak_result['strength']} (Score: {weak_result['score']})")
    print(f"Strong password strength: {strong_result['strength']} (Score: {strong_result['score']})")
    
    # Generate secure password
    secure_password = crypto.generate_secure_password(20, include_symbols=True)
    print(f"Generated secure password: {secure_password}")

if __name__ == "__main__":
    asyncio.run(main()) 