from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from typing import Optional, Union
from pydantic import BaseModel, field_validator
import structlog
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, ed25519, padding
from cryptography.hazmat.backends import default_backend
import base64
        import hashlib
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Digital signature utilities for cybersecurity tools.
"""

logger = structlog.get_logger(__name__)

class SignatureInput(BaseModel):
    """Input model for data signing."""
    data: Union[str, bytes]
    private_key: str
    algorithm: str = "RSA-SHA256"
    encoding: str = "utf-8"
    
    @field_validator('data')
    def validate_data(cls, v) -> bool:
        if not v:
            raise ValueError("Data cannot be empty")
        return v
    
    @field_validator('private_key')
    def validate_private_key(cls, v) -> bool:
        if not v:
            raise ValueError("Private key cannot be empty")
        return v
    
    @field_validator('algorithm')
    def validate_algorithm(cls, v) -> bool:
        valid_algorithms = ["RSA-SHA256", "RSA-SHA512", "Ed25519"]
        if v not in valid_algorithms:
            raise ValueError(f"Algorithm must be one of: {valid_algorithms}")
        return v

class VerificationInput(BaseModel):
    """Input model for signature verification."""
    data: Union[str, bytes]
    signature: str
    public_key: str
    algorithm: str = "RSA-SHA256"
    encoding: str = "utf-8"
    
    @field_validator('data')
    def validate_data(cls, v) -> bool:
        if not v:
            raise ValueError("Data cannot be empty")
        return v
    
    @field_validator('signature')
    def validate_signature(cls, v) -> bool:
        if not v:
            raise ValueError("Signature cannot be empty")
        return v
    
    @field_validator('public_key')
    def validate_public_key(cls, v) -> bool:
        if not v:
            raise ValueError("Public key cannot be empty")
        return v

class SignatureResult(BaseModel):
    """Result model for signature operations."""
    signature: str
    algorithm: str
    data_hash: Optional[str] = None
    is_successful: bool
    error_message: Optional[str] = None

class VerificationResult(BaseModel):
    """Result model for verification operations."""
    is_valid: bool
    algorithm: str
    data_hash: Optional[str] = None
    is_successful: bool
    error_message: Optional[str] = None

def sign_data(input_data: SignatureInput) -> SignatureResult:
    """
    RORO: Receive SignatureInput, return SignatureResult
    
    Sign data using the specified algorithm.
    """
    try:
        # Convert data to bytes if it's a string
        if isinstance(input_data.data, str):
            data_bytes = input_data.data.encode(input_data.encoding)
        else:
            data_bytes = input_data.data
        
        if input_data.algorithm == "RSA-SHA256":
            return sign_with_rsa_sha256(data_bytes, input_data.private_key, input_data.encoding)
        elif input_data.algorithm == "RSA-SHA512":
            return sign_with_rsa_sha512(data_bytes, input_data.private_key, input_data.encoding)
        elif input_data.algorithm == "Ed25519":
            return sign_with_ed25519(data_bytes, input_data.private_key, input_data.encoding)
        else:
            raise ValueError(f"Unsupported algorithm: {input_data.algorithm}")
            
    except Exception as e:
        logger.error("Data signing failed", error=str(e))
        return SignatureResult(
            signature="",
            algorithm=input_data.algorithm,
            is_successful=False,
            error_message=str(e)
        )

def verify_signature(input_data: VerificationInput) -> VerificationResult:
    """
    RORO: Receive VerificationInput, return VerificationResult
    
    Verify a signature using the specified algorithm.
    """
    try:
        # Convert data to bytes if it's a string
        if isinstance(input_data.data, str):
            data_bytes = input_data.data.encode(input_data.encoding)
        else:
            data_bytes = input_data.data
        
        if input_data.algorithm == "RSA-SHA256":
            return verify_with_rsa_sha256(data_bytes, input_data.signature, input_data.public_key, input_data.encoding)
        elif input_data.algorithm == "RSA-SHA512":
            return verify_with_rsa_sha512(data_bytes, input_data.signature, input_data.public_key, input_data.encoding)
        elif input_data.algorithm == "Ed25519":
            return verify_with_ed25519(data_bytes, input_data.signature, input_data.public_key, input_data.encoding)
        else:
            raise ValueError(f"Unsupported algorithm: {input_data.algorithm}")
            
    except Exception as e:
        logger.error("Signature verification failed", error=str(e))
        return VerificationResult(
            is_valid=False,
            algorithm=input_data.algorithm,
            is_successful=False,
            error_message=str(e)
        )

def sign_with_rsa_sha256(data_bytes: bytes, private_key_str: str, encoding: str) -> SignatureResult:
    """Sign data using RSA-SHA256."""
    try:
        # Load private key
        private_key_bytes = private_key_str.encode(encoding)
        private_key = serialization.load_pem_private_key(
            private_key_bytes,
            password=None,
            backend=default_backend()
        )
        
        # Generate hash
        hash_obj = hashes.Hash(hashes.SHA256(), backend=default_backend())
        hash_obj.update(data_bytes)
        data_hash = hash_obj.finalize()
        
        # Sign the hash
        signature = private_key.sign(
            data_hash,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        
        return SignatureResult(
            signature=base64.b64encode(signature).decode(encoding),
            algorithm="RSA-SHA256",
            data_hash=base64.b64encode(data_hash).decode(encoding),
            is_successful=True
        )
        
    except Exception as e:
        logger.error("RSA-SHA256 signing failed", error=str(e))
        raise

def verify_with_rsa_sha256(data_bytes: bytes, signature_str: str, public_key_str: str, encoding: str) -> VerificationResult:
    """Verify signature using RSA-SHA256."""
    try:
        # Load public key
        public_key_bytes = public_key_str.encode(encoding)
        public_key = serialization.load_pem_public_key(
            public_key_bytes,
            backend=default_backend()
        )
        
        # Decode signature
        signature = base64.b64decode(signature_str.encode(encoding))
        
        # Generate hash
        hash_obj = hashes.Hash(hashes.SHA256(), backend=default_backend())
        hash_obj.update(data_bytes)
        data_hash = hash_obj.finalize()
        
        # Verify signature
        try:
            public_key.verify(
                signature,
                data_hash,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            is_valid = True
        except Exception:
            is_valid = False
        
        return VerificationResult(
            is_valid=is_valid,
            algorithm="RSA-SHA256",
            data_hash=base64.b64encode(data_hash).decode(encoding),
            is_successful=True
        )
        
    except Exception as e:
        logger.error("RSA-SHA256 verification failed", error=str(e))
        raise

def sign_with_rsa_sha512(data_bytes: bytes, private_key_str: str, encoding: str) -> SignatureResult:
    """Sign data using RSA-SHA512."""
    try:
        # Load private key
        private_key_bytes = private_key_str.encode(encoding)
        private_key = serialization.load_pem_private_key(
            private_key_bytes,
            password=None,
            backend=default_backend()
        )
        
        # Generate hash
        hash_obj = hashes.Hash(hashes.SHA512(), backend=default_backend())
        hash_obj.update(data_bytes)
        data_hash = hash_obj.finalize()
        
        # Sign the hash
        signature = private_key.sign(
            data_hash,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA512()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA512()
        )
        
        return SignatureResult(
            signature=base64.b64encode(signature).decode(encoding),
            algorithm="RSA-SHA512",
            data_hash=base64.b64encode(data_hash).decode(encoding),
            is_successful=True
        )
        
    except Exception as e:
        logger.error("RSA-SHA512 signing failed", error=str(e))
        raise

def verify_with_rsa_sha512(data_bytes: bytes, signature_str: str, public_key_str: str, encoding: str) -> VerificationResult:
    """Verify signature using RSA-SHA512."""
    try:
        # Load public key
        public_key_bytes = public_key_str.encode(encoding)
        public_key = serialization.load_pem_public_key(
            public_key_bytes,
            backend=default_backend()
        )
        
        # Decode signature
        signature = base64.b64decode(signature_str.encode(encoding))
        
        # Generate hash
        hash_obj = hashes.Hash(hashes.SHA512(), backend=default_backend())
        hash_obj.update(data_bytes)
        data_hash = hash_obj.finalize()
        
        # Verify signature
        try:
            public_key.verify(
                signature,
                data_hash,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA512()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA512()
            )
            is_valid = True
        except Exception:
            is_valid = False
        
        return VerificationResult(
            is_valid=is_valid,
            algorithm="RSA-SHA512",
            data_hash=base64.b64encode(data_hash).decode(encoding),
            is_successful=True
        )
        
    except Exception as e:
        logger.error("RSA-SHA512 verification failed", error=str(e))
        raise

def sign_with_ed25519(data_bytes: bytes, private_key_str: str, encoding: str) -> SignatureResult:
    """Sign data using Ed25519."""
    try:
        # Load private key
        private_key_bytes = private_key_str.encode(encoding)
        private_key = serialization.load_pem_private_key(
            private_key_bytes,
            password=None,
            backend=default_backend()
        )
        
        # Sign the data directly (Ed25519 doesn't require separate hashing)
        signature = private_key.sign(data_bytes)
        
        return SignatureResult(
            signature=base64.b64encode(signature).decode(encoding),
            algorithm="Ed25519",
            is_successful=True
        )
        
    except Exception as e:
        logger.error("Ed25519 signing failed", error=str(e))
        raise

def verify_with_ed25519(data_bytes: bytes, signature_str: str, public_key_str: str, encoding: str) -> VerificationResult:
    """Verify signature using Ed25519."""
    try:
        # Load public key
        public_key_bytes = public_key_str.encode(encoding)
        public_key = serialization.load_pem_public_key(
            public_key_bytes,
            backend=default_backend()
        )
        
        # Decode signature
        signature = base64.b64decode(signature_str.encode(encoding))
        
        # Verify signature
        try:
            public_key.verify(signature, data_bytes)
            is_valid = True
        except Exception:
            is_valid = False
        
        return VerificationResult(
            is_valid=is_valid,
            algorithm="Ed25519",
            is_successful=True
        )
        
    except Exception as e:
        logger.error("Ed25519 verification failed", error=str(e))
        raise

def generate_message_digest(data: Union[str, bytes], algorithm: str = "SHA-256", encoding: str = "utf-8") -> str:
    """
    Generate a message digest (hash) of the data.
    
    Args:
        data: Data to hash
        algorithm: Hash algorithm to use
        encoding: String encoding if data is string
        
    Returns:
        Hex digest of the data
    """
    try:
        
        # Convert data to bytes if it's a string
        if isinstance(data, str):
            data_bytes = data.encode(encoding)
        else:
            data_bytes = data
        
        # Create hash object
        if algorithm.upper() == "SHA-256":
            hash_obj = hashlib.sha256()
        elif algorithm.upper() == "SHA-512":
            hash_obj = hashlib.sha512()
        elif algorithm.upper() == "MD5":
            hash_obj = hashlib.md5()
        else:
            raise ValueError(f"Unsupported hash algorithm: {algorithm}")
        
        # Update hash with data
        hash_obj.update(data_bytes)
        
        # Return hex digest
        return hash_obj.hexdigest()
        
    except Exception as e:
        logger.error("Message digest generation failed", error=str(e))
        raise 