"""
Cryptography Module for Video-OpusClip
Comprehensive cryptographic operations for symmetric and asymmetric encryption
"""

from .symmetric_crypto import (
    SymmetricCryptoService, AESEncryption, ChaCha20Encryption,
    FernetEncryption, BlowfishEncryption, TwofishEncryption,
    encrypt_symmetric, decrypt_symmetric, generate_symmetric_key,
    SymmetricAlgorithm, KeySize, EncryptionMode
)

from .asymmetric_crypto import (
    AsymmetricCryptoService, RSAEncryption, ECCEncryption,
    DSAEncryption, Ed25519Encryption, X25519Encryption,
    encrypt_asymmetric, decrypt_asymmetric, sign_data, verify_signature,
    generate_key_pair, AsymmetricAlgorithm, KeyFormat
)

from .hash_functions import (
    HashService, SHA256Hash, SHA512Hash, Blake2bHash, Argon2Hash,
    bcrypt_hash, scrypt_hash, pbkdf2_hash, hash_data, verify_hash,
    HashAlgorithm, SaltGenerator
)

from .key_management import (
    KeyManager, KeyStore, KeyRotation, KeyDerivation,
    generate_master_key, derive_key, rotate_keys, backup_keys,
    KeyType, KeyUsage, KeyMetadata
)

from .certificate_management import (
    CertificateManager, X509Certificate, SelfSignedCertificate,
    CertificateAuthority, CertificateChain, create_certificate,
    sign_certificate, verify_certificate, CertificateInfo
)

from .secure_random import (
    SecureRandom, CryptoRandom, SystemRandom, HardwareRandom,
    generate_random_bytes, generate_random_string, generate_random_number,
    RandomSource, EntropySource
)

from .crypto_utils import (
    CryptoUtils, Base64Encoder, HexEncoder, PEMEncoder,
    encode_data, decode_data, format_key, parse_key,
    EncodingFormat, KeyFormat
)

from .crypto_config import (
    CryptoConfig, SecurityLevel, KeySize, AlgorithmSuite,
    load_crypto_config, save_crypto_config, validate_crypto_config
)

from .crypto_validation import (
    CryptoValidator, AlgorithmValidator, KeyValidator,
    CertificateValidator, validate_algorithm, validate_key,
    validate_certificate, ValidationResult
)

from .crypto_logging import (
    CryptoLogger, SecurityLogger, AuditLogger,
    log_crypto_operation, log_security_event, log_audit_trail,
    LogLevel, SecurityEvent
)

from .crypto_exceptions import (
    CryptoError, EncryptionError, DecryptionError,
    KeyError, CertificateError, ValidationError,
    AlgorithmError, ConfigurationError
)

__all__ = [
    # Symmetric Cryptography
    'SymmetricCryptoService', 'AESEncryption', 'ChaCha20Encryption',
    'FernetEncryption', 'BlowfishEncryption', 'TwofishEncryption',
    'encrypt_symmetric', 'decrypt_symmetric', 'generate_symmetric_key',
    'SymmetricAlgorithm', 'KeySize', 'EncryptionMode',
    
    # Asymmetric Cryptography
    'AsymmetricCryptoService', 'RSAEncryption', 'ECCEncryption',
    'DSAEncryption', 'Ed25519Encryption', 'X25519Encryption',
    'encrypt_asymmetric', 'decrypt_asymmetric', 'sign_data', 'verify_signature',
    'generate_key_pair', 'AsymmetricAlgorithm', 'KeyFormat',
    
    # Hash Functions
    'HashService', 'SHA256Hash', 'SHA512Hash', 'Blake2bHash', 'Argon2Hash',
    'bcrypt_hash', 'scrypt_hash', 'pbkdf2_hash', 'hash_data', 'verify_hash',
    'HashAlgorithm', 'SaltGenerator',
    
    # Key Management
    'KeyManager', 'KeyStore', 'KeyRotation', 'KeyDerivation',
    'generate_master_key', 'derive_key', 'rotate_keys', 'backup_keys',
    'KeyType', 'KeyUsage', 'KeyMetadata',
    
    # Certificate Management
    'CertificateManager', 'X509Certificate', 'SelfSignedCertificate',
    'CertificateAuthority', 'CertificateChain', 'create_certificate',
    'sign_certificate', 'verify_certificate', 'CertificateInfo',
    
    # Secure Random
    'SecureRandom', 'CryptoRandom', 'SystemRandom', 'HardwareRandom',
    'generate_random_bytes', 'generate_random_string', 'generate_random_number',
    'RandomSource', 'EntropySource',
    
    # Crypto Utils
    'CryptoUtils', 'Base64Encoder', 'HexEncoder', 'PEMEncoder',
    'encode_data', 'decode_data', 'format_key', 'parse_key',
    'EncodingFormat', 'KeyFormat',
    
    # Crypto Config
    'CryptoConfig', 'SecurityLevel', 'KeySize', 'AlgorithmSuite',
    'load_crypto_config', 'save_crypto_config', 'validate_crypto_config',
    
    # Crypto Validation
    'CryptoValidator', 'AlgorithmValidator', 'KeyValidator',
    'CertificateValidator', 'validate_algorithm', 'validate_key',
    'validate_certificate', 'ValidationResult',
    
    # Crypto Logging
    'CryptoLogger', 'SecurityLogger', 'AuditLogger',
    'log_crypto_operation', 'log_security_event', 'log_audit_trail',
    'LogLevel', 'SecurityEvent',
    
    # Crypto Exceptions
    'CryptoError', 'EncryptionError', 'DecryptionError',
    'KeyError', 'CertificateError', 'ValidationError',
    'AlgorithmError', 'ConfigurationError'
] 