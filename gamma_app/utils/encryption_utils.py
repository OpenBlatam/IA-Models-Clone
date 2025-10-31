"""
Gamma App - Encryption Utilities
Advanced encryption and security utilities
"""

import os
import base64
import hashlib
import secrets
from typing import Union, Optional, Dict, Any
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
import logging

logger = logging.getLogger(__name__)

class EncryptionManager:
    """Advanced encryption manager"""
    
    def __init__(self, master_key: Optional[str] = None):
        self.master_key = master_key or os.getenv("MASTER_ENCRYPTION_KEY")
        if not self.master_key:
            self.master_key = self._generate_master_key()
        
        self.fernet = self._create_fernet()
        self.rsa_private_key = None
        self.rsa_public_key = None
        self._load_or_generate_rsa_keys()
    
    def _generate_master_key(self) -> str:
        """Generate a new master key"""
        return Fernet.generate_key().decode()
    
    def _create_fernet(self) -> Fernet:
        """Create Fernet encryption instance"""
        # Derive key from master key
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'gamma_app_salt',
            iterations=100000,
            backend=default_backend()
        )
        key = base64.urlsafe_b64encode(kdf.derive(self.master_key.encode()))
        return Fernet(key)
    
    def _load_or_generate_rsa_keys(self):
        """Load or generate RSA key pair"""
        try:
            # Try to load existing keys
            self._load_rsa_keys()
        except FileNotFoundError:
            # Generate new keys
            self._generate_rsa_keys()
    
    def _generate_rsa_keys(self):
        """Generate new RSA key pair"""
        self.rsa_private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )
        self.rsa_public_key = self.rsa_private_key.public_key()
        
        # Save keys
        self._save_rsa_keys()
    
    def _save_rsa_keys(self):
        """Save RSA keys to files"""
        try:
            # Save private key
            with open("rsa_private_key.pem", "wb") as f:
                f.write(self.rsa_private_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption()
                ))
            
            # Save public key
            with open("rsa_public_key.pem", "wb") as f:
                f.write(self.rsa_public_key.public_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo
                ))
        except Exception as e:
            logger.error(f"Error saving RSA keys: {e}")
    
    def _load_rsa_keys(self):
        """Load RSA keys from files"""
        # Load private key
        with open("rsa_private_key.pem", "rb") as f:
            self.rsa_private_key = serialization.load_pem_private_key(
                f.read(), password=None, backend=default_backend()
            )
        
        # Load public key
        with open("rsa_public_key.pem", "rb") as f:
            self.rsa_public_key = serialization.load_pem_public_key(
                f.read(), backend=default_backend()
            )
    
    def encrypt_symmetric(self, data: Union[str, bytes]) -> str:
        """Encrypt data using symmetric encryption"""
        try:
            if isinstance(data, str):
                data = data.encode()
            
            encrypted_data = self.fernet.encrypt(data)
            return base64.urlsafe_b64encode(encrypted_data).decode()
        except Exception as e:
            logger.error(f"Error encrypting data: {e}")
            raise
    
    def decrypt_symmetric(self, encrypted_data: str) -> str:
        """Decrypt data using symmetric encryption"""
        try:
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode())
            decrypted_data = self.fernet.decrypt(encrypted_bytes)
            return decrypted_data.decode()
        except Exception as e:
            logger.error(f"Error decrypting data: {e}")
            raise
    
    def encrypt_asymmetric(self, data: Union[str, bytes]) -> str:
        """Encrypt data using asymmetric encryption"""
        try:
            if isinstance(data, str):
                data = data.encode()
            
            encrypted_data = self.rsa_public_key.encrypt(
                data,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            return base64.urlsafe_b64encode(encrypted_data).decode()
        except Exception as e:
            logger.error(f"Error encrypting with RSA: {e}")
            raise
    
    def decrypt_asymmetric(self, encrypted_data: str) -> str:
        """Decrypt data using asymmetric encryption"""
        try:
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode())
            decrypted_data = self.rsa_private_key.decrypt(
                encrypted_bytes,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            return decrypted_data.decode()
        except Exception as e:
            logger.error(f"Error decrypting with RSA: {e}")
            raise
    
    def hash_password(self, password: str, salt: Optional[str] = None) -> Dict[str, str]:
        """Hash password with salt"""
        if salt is None:
            salt = secrets.token_hex(32)
        
        # Use PBKDF2 for password hashing
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt.encode(),
            iterations=100000,
            backend=default_backend()
        )
        
        password_hash = base64.urlsafe_b64encode(
            kdf.derive(password.encode())
        ).decode()
        
        return {
            "hash": password_hash,
            "salt": salt
        }
    
    def verify_password(self, password: str, password_hash: str, salt: str) -> bool:
        """Verify password against hash"""
        try:
            computed_hash = self.hash_password(password, salt)["hash"]
            return secrets.compare_digest(password_hash, computed_hash)
        except Exception as e:
            logger.error(f"Error verifying password: {e}")
            return False
    
    def generate_token(self, length: int = 32) -> str:
        """Generate secure random token"""
        return secrets.token_urlsafe(length)
    
    def generate_api_key(self) -> str:
        """Generate API key"""
        return f"gamma_{secrets.token_urlsafe(32)}"
    
    def create_hmac(self, data: str, key: str) -> str:
        """Create HMAC signature"""
        hmac = hashlib.pbkdf2_hmac(
            'sha256',
            data.encode(),
            key.encode(),
            100000
        )
        return base64.urlsafe_b64encode(hmac).decode()
    
    def verify_hmac(self, data: str, signature: str, key: str) -> bool:
        """Verify HMAC signature"""
        try:
            expected_signature = self.create_hmac(data, key)
            return secrets.compare_digest(signature, expected_signature)
        except Exception as e:
            logger.error(f"Error verifying HMAC: {e}")
            return False
    
    def encrypt_file(self, file_path: str, output_path: Optional[str] = None) -> str:
        """Encrypt file"""
        try:
            if output_path is None:
                output_path = f"{file_path}.encrypted"
            
            with open(file_path, 'rb') as f:
                file_data = f.read()
            
            encrypted_data = self.fernet.encrypt(file_data)
            
            with open(output_path, 'wb') as f:
                f.write(encrypted_data)
            
            return output_path
        except Exception as e:
            logger.error(f"Error encrypting file: {e}")
            raise
    
    def decrypt_file(self, encrypted_file_path: str, output_path: Optional[str] = None) -> str:
        """Decrypt file"""
        try:
            if output_path is None:
                output_path = encrypted_file_path.replace('.encrypted', '')
            
            with open(encrypted_file_path, 'rb') as f:
                encrypted_data = f.read()
            
            decrypted_data = self.fernet.decrypt(encrypted_data)
            
            with open(output_path, 'wb') as f:
                f.write(decrypted_data)
            
            return output_path
        except Exception as e:
            logger.error(f"Error decrypting file: {e}")
            raise
    
    def get_public_key_pem(self) -> str:
        """Get public key in PEM format"""
        return self.rsa_public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        ).decode()
    
    def encrypt_sensitive_data(self, data: Dict[str, Any]) -> Dict[str, str]:
        """Encrypt sensitive data in dictionary"""
        encrypted_data = {}
        
        for key, value in data.items():
            if isinstance(value, str):
                encrypted_data[key] = self.encrypt_symmetric(value)
            else:
                encrypted_data[key] = self.encrypt_symmetric(str(value))
        
        return encrypted_data
    
    def decrypt_sensitive_data(self, encrypted_data: Dict[str, str]) -> Dict[str, str]:
        """Decrypt sensitive data in dictionary"""
        decrypted_data = {}
        
        for key, value in encrypted_data.items():
            decrypted_data[key] = self.decrypt_symmetric(value)
        
        return decrypted_data

class DataMasking:
    """Data masking utilities"""
    
    @staticmethod
    def mask_email(email: str) -> str:
        """Mask email address"""
        if '@' not in email:
            return email
        
        local, domain = email.split('@', 1)
        if len(local) <= 2:
            masked_local = '*' * len(local)
        else:
            masked_local = local[0] + '*' * (len(local) - 2) + local[-1]
        
        return f"{masked_local}@{domain}"
    
    @staticmethod
    def mask_phone(phone: str) -> str:
        """Mask phone number"""
        if len(phone) <= 4:
            return '*' * len(phone)
        
        return phone[:2] + '*' * (len(phone) - 4) + phone[-2:]
    
    @staticmethod
    def mask_credit_card(card_number: str) -> str:
        """Mask credit card number"""
        if len(card_number) <= 4:
            return '*' * len(card_number)
        
        return '*' * (len(card_number) - 4) + card_number[-4:]
    
    @staticmethod
    def mask_ssn(ssn: str) -> str:
        """Mask Social Security Number"""
        if len(ssn) != 9:
            return '*' * len(ssn)
        
        return '***-**-' + ssn[-4:]

class SecureRandom:
    """Secure random number generation"""
    
    @staticmethod
    def random_string(length: int = 16) -> str:
        """Generate secure random string"""
        return secrets.token_urlsafe(length)
    
    @staticmethod
    def random_int(min_val: int = 0, max_val: int = 100) -> int:
        """Generate secure random integer"""
        return secrets.randbelow(max_val - min_val + 1) + min_val
    
    @staticmethod
    def random_bytes(length: int = 32) -> bytes:
        """Generate secure random bytes"""
        return secrets.token_bytes(length)
    
    @staticmethod
    def random_hex(length: int = 32) -> str:
        """Generate secure random hex string"""
        return secrets.token_hex(length)

# Global encryption manager instance
encryption_manager = EncryptionManager()

def encrypt_data(data: Union[str, bytes]) -> str:
    """Encrypt data using global encryption manager"""
    return encryption_manager.encrypt_symmetric(data)

def decrypt_data(encrypted_data: str) -> str:
    """Decrypt data using global encryption manager"""
    return encryption_manager.decrypt_symmetric(encrypted_data)

def hash_password(password: str, salt: Optional[str] = None) -> Dict[str, str]:
    """Hash password using global encryption manager"""
    return encryption_manager.hash_password(password, salt)

def verify_password(password: str, password_hash: str, salt: str) -> bool:
    """Verify password using global encryption manager"""
    return encryption_manager.verify_password(password, password_hash, salt)

def generate_token(length: int = 32) -> str:
    """Generate secure token"""
    return encryption_manager.generate_token(length)

def generate_api_key() -> str:
    """Generate API key"""
    return encryption_manager.generate_api_key()

























