# Cryptography Module for Video-OpusClip

Comprehensive cryptographic operations for symmetric and asymmetric encryption, hashing, and key management.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Symmetric Cryptography](#symmetric-cryptography)
- [Asymmetric Cryptography](#asymmetric-cryptography)
- [Hash Functions](#hash-functions)
- [Key Management](#key-management)
- [Certificate Management](#certificate-management)
- [Secure Random](#secure-random)
- [Crypto Utils](#crypto-utils)
- [Best Practices](#best-practices)
- [Examples](#examples)
- [API Reference](#api-reference)

## 🎯 Overview

The Cryptography module provides a comprehensive set of cryptographic operations for the Video-OpusClip project, including:

- **Symmetric Encryption**: AES, ChaCha20, Fernet, Blowfish, Twofish
- **Asymmetric Encryption**: RSA, ECC, DSA, Ed25519, X25519
- **Hash Functions**: SHA-256, SHA-512, Blake2b, Argon2, bcrypt, scrypt, PBKDF2
- **Key Management**: Key generation, storage, rotation, and derivation
- **Certificate Management**: X.509 certificates and certificate authorities
- **Secure Random**: Cryptographically secure random number generation
- **Crypto Utils**: Encoding, decoding, and formatting utilities

## ✨ Features

### 🔐 Symmetric Cryptography
- **Multiple Algorithms**: AES, ChaCha20, Fernet, Blowfish, Twofish
- **Encryption Modes**: CBC, GCM, CTR, CFB, OFB
- **Key Sizes**: 128, 192, 256 bits
- **Password-based Key Derivation**: PBKDF2 with configurable iterations

### 🔑 Asymmetric Cryptography
- **RSA**: Encryption, decryption, signing, verification
- **ECC**: Elliptic curve cryptography with multiple curves
- **DSA**: Digital signature algorithm
- **Ed25519**: Modern elliptic curve digital signature
- **X25519**: Elliptic curve key exchange

### 🔒 Hash Functions
- **Cryptographic Hashes**: SHA-256, SHA-512, Blake2b
- **Password Hashing**: Argon2, bcrypt, scrypt, PBKDF2
- **Salt Generation**: Secure salt generation with multiple methods
- **Hash Verification**: Built-in verification for all hash types

### 🔧 Key Management
- **Key Generation**: Secure key generation for all algorithms
- **Key Storage**: Secure key storage with encryption
- **Key Rotation**: Automated key rotation policies
- **Key Derivation**: Key derivation from master keys

### 📜 Certificate Management
- **X.509 Certificates**: Certificate creation and validation
- **Certificate Authorities**: CA setup and management
- **Certificate Chains**: Chain validation and management
- **Self-signed Certificates**: Self-signed certificate generation

### 🎲 Secure Random
- **Multiple Sources**: Crypto, system, and hardware random sources
- **Entropy Management**: Entropy source management
- **Random Generation**: Bytes, strings, and numbers

### 🛠️ Crypto Utils
- **Encoding Formats**: Base64, Hex, PEM, DER, JSON
- **Key Formats**: PEM, DER, SSH, JWK
- **Format Conversion**: Convert between different formats
- **Validation**: Format validation and data integrity

## 🛠️ Installation

The cryptography module requires the following dependencies:

```bash
# Core cryptography dependencies
pip install cryptography

# Password hashing dependencies
pip install bcrypt argon2-cffi

# Additional utilities
pip install pycryptodome  # For additional algorithms
```

## 🚀 Quick Start

### Basic Symmetric Encryption

```python
from cryptography import encrypt_symmetric, decrypt_symmetric, SymmetricAlgorithm

# Encrypt data with AES-256-GCM
data = b"Hello, Video-OpusClip!"
encrypted = encrypt_symmetric(data, algorithm=SymmetricAlgorithm.AES)
print(f"Encrypted: {encrypted.ciphertext.hex()}")

# Decrypt data
decrypted = decrypt_symmetric(encrypted.ciphertext, encrypted.iv, encrypted.tag)
print(f"Decrypted: {decrypted.plaintext.decode()}")
```

### Basic Asymmetric Encryption

```python
from cryptography import generate_key_pair, encrypt_asymmetric, decrypt_asymmetric, AsymmetricAlgorithm

# Generate RSA key pair
key_pair = generate_key_pair(AsymmetricAlgorithm.RSA)
print(f"Public key: {key_pair.public_key[:100]}...")

# Encrypt with public key
data = b"Secret message"
encrypted = encrypt_asymmetric(data, key_pair.public_key, AsymmetricAlgorithm.RSA)

# Decrypt with private key
decrypted = decrypt_asymmetric(encrypted, key_pair.private_key, AsymmetricAlgorithm.RSA)
print(f"Decrypted: {decrypted.decode()}")
```

### Password Hashing

```python
from cryptography import hash_data, verify_hash, HashAlgorithm

# Hash password with Argon2
password = b"my-secure-password"
hashed = hash_data(password, HashAlgorithm.ARGON2)
print(f"Hash: {hashed.hash_value.decode()[:50]}...")

# Verify password
verified = verify_hash(password, hashed.hash_value, HashAlgorithm.ARGON2)
print(f"Verified: {verified.verified}")
```

## 🔐 Symmetric Cryptography

### Supported Algorithms

```python
from cryptography import SymmetricAlgorithm, EncryptionMode, KeySize

# AES with different modes
aes_gcm = AESEncryption(KeySize.AES_256, EncryptionMode.GCM)
aes_cbc = AESEncryption(KeySize.AES_256, EncryptionMode.CBC)
aes_ctr = AESEncryption(KeySize.AES_256, EncryptionMode.CTR)

# ChaCha20
chacha20 = ChaCha20Encryption()

# Fernet (AES-128-CBC with HMAC)
fernet = FernetEncryption()

# Blowfish
blowfish = BlowfishEncryption()
```

### Encryption Examples

```python
from cryptography import SymmetricCryptoService

service = SymmetricCryptoService()

# AES-GCM encryption
data = b"Sensitive data"
key = service.generate_key(SymmetricAlgorithm.AES)
result = service.encrypt(data, SymmetricAlgorithm.AES, key)

print(f"Ciphertext: {result.ciphertext.hex()}")
print(f"IV: {result.iv.hex()}")
print(f"Tag: {result.tag.hex()}")

# ChaCha20 encryption
chacha_result = service.encrypt(data, SymmetricAlgorithm.CHACHA20, key)
print(f"ChaCha20 ciphertext: {chacha_result.ciphertext.hex()}")
```

### Password-based Encryption

```python
# Encrypt with password
password = "my-secure-password"
result = service.encrypt(data, SymmetricAlgorithm.AES, password=password)

print(f"Salt: {result.salt.hex()}")
print(f"Ciphertext: {result.ciphertext.hex()}")

# Decrypt with password
decrypted = service.decrypt(result.ciphertext, result.salt, SymmetricAlgorithm.AES,
                           iv=result.iv, tag=result.tag)
print(f"Decrypted: {decrypted.plaintext.decode()}")
```

## 🔑 Asymmetric Cryptography

### Key Generation

```python
from cryptography import AsymmetricCryptoService

service = AsymmetricCryptoService()

# Generate RSA key pair
rsa_keys = service.generate_key_pair(AsymmetricAlgorithm.RSA)
print(f"RSA private key: {len(rsa_keys.private_key)} bytes")
print(f"RSA public key: {len(rsa_keys.public_key)} bytes")

# Generate ECC key pair
ecc_keys = service.generate_key_pair(AsymmetricAlgorithm.ECC)
print(f"ECC private key: {len(ecc_keys.private_key)} bytes")
print(f"ECC public key: {len(ecc_keys.public_key)} bytes")

# Generate Ed25519 key pair
ed25519_keys = service.generate_key_pair(AsymmetricAlgorithm.ED25519)
print(f"Ed25519 private key: {len(ed25519_keys.private_key)} bytes")
print(f"Ed25519 public key: {len(ed25519_keys.public_key)} bytes")
```

### Encryption and Decryption

```python
# RSA encryption
data = b"Secret message"
encrypted = service.encrypt(data, rsa_keys.public_key, AsymmetricAlgorithm.RSA)
decrypted = service.decrypt(encrypted, rsa_keys.private_key, AsymmetricAlgorithm.RSA)
print(f"RSA decrypted: {decrypted.decode()}")

# ECC encryption (using ECDH)
ecc_encrypted = service.encrypt(data, ecc_keys.public_key, AsymmetricAlgorithm.ECC)
ecc_decrypted = service.decrypt(ecc_encrypted, ecc_keys.private_key, AsymmetricAlgorithm.ECC)
print(f"ECC decrypted: {ecc_decrypted.decode()}")
```

### Digital Signatures

```python
# RSA signing
data = b"Message to sign"
signature = service.sign(data, rsa_keys.private_key, AsymmetricAlgorithm.RSA)
verified = service.verify(data, signature.signature, rsa_keys.public_key, AsymmetricAlgorithm.RSA)
print(f"RSA signature verified: {verified.verified}")

# Ed25519 signing
ed25519_signature = service.sign(data, ed25519_keys.private_key, AsymmetricAlgorithm.ED25519)
ed25519_verified = service.verify(data, ed25519_signature.signature, ed25519_keys.public_key, AsymmetricAlgorithm.ED25519)
print(f"Ed25519 signature verified: {ed25519_verified.verified}")
```

### Key Exchange

```python
# X25519 key exchange
x25519_keys1 = service.generate_key_pair(AsymmetricAlgorithm.X25519)
x25519_keys2 = service.generate_key_pair(AsymmetricAlgorithm.X25519)

# Perform key exchange
x25519_crypto = service.algorithms[AsymmetricAlgorithm.X25519]
shared_key1 = x25519_crypto.exchange(x25519_keys1.private_key, x25519_keys2.public_key)
shared_key2 = x25519_crypto.exchange(x25519_keys2.private_key, x25519_keys1.public_key)

print(f"Key exchange successful: {shared_key1 == shared_key2}")
print(f"Shared key: {shared_key1.hex()[:32]}...")
```

## 🔒 Hash Functions

### Cryptographic Hashes

```python
from cryptography import HashService

service = HashService()

# SHA-256 hashing
data = b"Data to hash"
sha256_hash = service.hash(data, HashAlgorithm.SHA256)
print(f"SHA-256: {sha256_hash.hash_value.hex()}")

# SHA-512 hashing
sha512_hash = service.hash(data, HashAlgorithm.SHA512)
print(f"SHA-512: {sha512_hash.hash_value.hex()}")

# Blake2b hashing
blake2b_hash = service.hash(data, HashAlgorithm.BLAKE2B)
print(f"Blake2b: {blake2b_hash.hash_value.hex()}")
```

### Password Hashing

```python
# Argon2 password hashing
password = b"my-secure-password"
argon2_hash = service.hash(password, HashAlgorithm.ARGON2)
print(f"Argon2 hash: {argon2_hash.hash_value.decode()[:50]}...")
print(f"Memory cost: {argon2_hash.memory_cost}")
print(f"Parallelism: {argon2_hash.parallelism}")

# Bcrypt password hashing
bcrypt_hash = service.hash(password, HashAlgorithm.BCRYPT)
print(f"Bcrypt hash: {bcrypt_hash.hash_value.decode()[:50]}...")
print(f"Rounds: {bcrypt_hash.iterations}")

# Scrypt password hashing
scrypt_hash = service.hash(password, HashAlgorithm.SCRYPT)
print(f"Scrypt hash: {scrypt_hash.hash_value.hex()}")
print(f"Salt: {scrypt_hash.salt.hex()}")
print(f"Iterations: {scrypt_hash.iterations}")

# PBKDF2 password hashing
pbkdf2_hash = service.hash(password, HashAlgorithm.PBKDF2)
print(f"PBKDF2 hash: {pbkdf2_hash.hash_value.hex()}")
print(f"Salt: {pbkdf2_hash.salt.hex()}")
print(f"Iterations: {pbkdf2_hash.iterations}")
```

### Hash Verification

```python
# Verify Argon2 hash
argon2_verified = service.verify(password, argon2_hash.hash_value, HashAlgorithm.ARGON2)
print(f"Argon2 verified: {argon2_verified.verified}")

# Verify bcrypt hash
bcrypt_verified = service.verify(password, bcrypt_hash.hash_value, HashAlgorithm.BCRYPT)
print(f"Bcrypt verified: {bcrypt_verified.verified}")

# Verify scrypt hash
scrypt_verified = service.verify(password, scrypt_hash.hash_value, HashAlgorithm.SCRYPT, salt=scrypt_hash.salt)
print(f"Scrypt verified: {scrypt_verified.verified}")

# Verify PBKDF2 hash
pbkdf2_verified = service.verify(password, pbkdf2_hash.hash_value, HashAlgorithm.PBKDF2, salt=pbkdf2_hash.salt)
print(f"PBKDF2 verified: {pbkdf2_verified.verified}")
```

## 🔧 Key Management

### Key Generation

```python
from cryptography import KeyManager

manager = KeyManager()

# Generate master key
master_key = manager.generate_master_key()
print(f"Master key: {master_key.hex()}")

# Generate symmetric key
symmetric_key = manager.generate_key(KeyType.SYMMETRIC, KeySize.AES_256)
print(f"Symmetric key: {symmetric_key.hex()}")

# Generate asymmetric key pair
asymmetric_keys = manager.generate_key_pair(KeyType.RSA, KeySize.RSA_2048)
print(f"RSA private key: {len(asymmetric_keys.private_key)} bytes")
print(f"RSA public key: {len(asymmetric_keys.public_key)} bytes")
```

### Key Derivation

```python
# Derive key from master key
derived_key = manager.derive_key(master_key, "application-key", KeySize.AES_256)
print(f"Derived key: {derived_key.hex()}")

# Derive multiple keys
user_key = manager.derive_key(master_key, "user-key", KeySize.AES_256)
session_key = manager.derive_key(master_key, "session-key", KeySize.AES_256)
print(f"User key: {user_key.hex()}")
print(f"Session key: {session_key.hex()}")
```

### Key Storage

```python
# Store key securely
key_id = manager.store_key(symmetric_key, KeyUsage.ENCRYPTION)
print(f"Stored key ID: {key_id}")

# Retrieve key
retrieved_key = manager.get_key(key_id)
print(f"Retrieved key: {retrieved_key.hex()}")

# List all keys
all_keys = manager.list_keys()
print(f"All keys: {list(all_keys.keys())}")
```

### Key Rotation

```python
# Rotate keys
new_key = manager.rotate_key(key_id)
print(f"New key ID: {new_key}")

# Backup keys
backup_data = manager.backup_keys()
print(f"Backup data size: {len(backup_data)} bytes")

# Restore keys
manager.restore_keys(backup_data)
print("Keys restored successfully")
```

## 📜 Certificate Management

### Certificate Creation

```python
from cryptography import CertificateManager

manager = CertificateManager()

# Create self-signed certificate
cert_info = CertificateInfo(
    common_name="example.com",
    organization="Example Org",
    country="US",
    state="CA",
    locality="San Francisco"
)

cert = manager.create_self_signed_certificate(cert_info, key_pair.private_key)
print(f"Certificate: {cert.pem[:100]}...")

# Create certificate authority
ca_cert = manager.create_certificate_authority(cert_info, ca_key_pair.private_key)
print(f"CA Certificate: {ca_cert.pem[:100]}...")
```

### Certificate Signing

```python
# Sign certificate with CA
signed_cert = manager.sign_certificate(cert, ca_cert, ca_key_pair.private_key)
print(f"Signed certificate: {signed_cert.pem[:100]}...")

# Verify certificate
verified = manager.verify_certificate(signed_cert, ca_cert)
print(f"Certificate verified: {verified}")
```

### Certificate Chain

```python
# Create certificate chain
chain = CertificateChain([signed_cert, ca_cert])
print(f"Chain length: {len(chain.certificates)}")

# Verify chain
chain_verified = manager.verify_certificate_chain(chain)
print(f"Chain verified: {chain_verified}")
```

## 🎲 Secure Random

### Random Generation

```python
from cryptography import SecureRandom

random = SecureRandom()

# Generate random bytes
random_bytes = random.generate_bytes(32)
print(f"Random bytes: {random_bytes.hex()}")

# Generate random string
random_string = random.generate_string(16)
print(f"Random string: {random_string}")

# Generate random number
random_number = random.generate_number(1, 100)
print(f"Random number: {random_number}")
```

### Multiple Sources

```python
# Crypto random
crypto_random = CryptoRandom()
crypto_bytes = crypto_random.generate_bytes(32)
print(f"Crypto random: {crypto_bytes.hex()}")

# System random
system_random = SystemRandom()
system_bytes = system_random.generate_bytes(32)
print(f"System random: {system_bytes.hex()}")

# Hardware random (if available)
try:
    hardware_random = HardwareRandom()
    hardware_bytes = hardware_random.generate_bytes(32)
    print(f"Hardware random: {hardware_bytes.hex()}")
except Exception as e:
    print(f"Hardware random not available: {e}")
```

## 🛠️ Crypto Utils

### Data Encoding

```python
from cryptography import CryptoUtils

utils = CryptoUtils()

# Base64 encoding
base64_data = utils.encode_data(test_data, EncodingFormat.BASE64)
print(f"Base64: {base64_data}")

# Hex encoding
hex_data = utils.encode_data(test_data, EncodingFormat.HEX)
print(f"Hex: {hex_data}")

# PEM encoding
pem_data = utils.encode_data(test_data, EncodingFormat.PEM)
print(f"PEM:\n{pem_data}")

# JSON encoding
json_data = utils.encode_data(test_data, EncodingFormat.JSON)
print(f"JSON: {json_data}")
```

### Key Formatting

```python
# Format key as PEM
pem_key = utils.format_key(key, KeyFormat.PEM, "RSA")
print(f"PEM key:\n{pem_key}")

# Format key as SSH
ssh_key = utils.format_key(key, KeyFormat.SSH, "rsa")
print(f"SSH key: {ssh_key}")

# Format key as JWK
jwk_key = utils.format_key(key, KeyFormat.JWK, "oct")
print(f"JWK key: {jwk_key}")
```

### Format Conversion

```python
# Convert between formats
base64_to_hex = utils.convert_format(base64_data, EncodingFormat.BASE64, EncodingFormat.HEX)
print(f"Base64 to Hex: {base64_to_hex}")

hex_to_base64 = utils.convert_format(hex_data, EncodingFormat.HEX, EncodingFormat.BASE64)
print(f"Hex to Base64: {hex_to_base64}")
```

### Format Validation

```python
# Validate formats
is_valid_base64 = utils.validate_format(base64_data, EncodingFormat.BASE64)
print(f"Valid Base64: {is_valid_base64}")

is_valid_hex = utils.validate_format(hex_data, EncodingFormat.HEX)
print(f"Valid Hex: {is_valid_hex}")

is_valid_invalid = utils.validate_format("invalid-data", EncodingFormat.BASE64)
print(f"Valid invalid data: {is_valid_invalid}")
```

## 📋 Best Practices

### 1. Algorithm Selection

```python
# ✅ Good: Use modern, secure algorithms
encrypted = encrypt_symmetric(data, algorithm=SymmetricAlgorithm.AES)  # AES-256-GCM
hashed = hash_data(password, HashAlgorithm.ARGON2)  # Argon2 for passwords
signed = sign_data(data, private_key, AsymmetricAlgorithm.ED25519)  # Ed25519 for signatures

# ❌ Bad: Use deprecated algorithms
encrypted = encrypt_symmetric(data, algorithm=SymmetricAlgorithm.BLOWFISH)  # Deprecated
hashed = hash_data(password, HashAlgorithm.SHA256)  # Not for passwords
signed = sign_data(data, private_key, AsymmetricAlgorithm.DSA)  # Deprecated
```

### 2. Key Management

```python
# ✅ Good: Use secure key generation
key = generate_symmetric_key(SymmetricAlgorithm.AES)
key_pair = generate_key_pair(AsymmetricAlgorithm.RSA)

# ✅ Good: Use key derivation
derived_key = derive_key(master_key, "application-key", KeySize.AES_256)

# ✅ Good: Rotate keys regularly
new_key = rotate_key(key_id)

# ❌ Bad: Use weak keys
key = b"weak-key-123"  # Predictable
key = os.urandom(8)  # Too short
```

### 3. Password Hashing

```python
# ✅ Good: Use dedicated password hashing
hashed = hash_data(password, HashAlgorithm.ARGON2)
hashed = hash_data(password, HashAlgorithm.BCRYPT)

# ✅ Good: Use appropriate parameters
argon2_hash = hash_data(password, HashAlgorithm.ARGON2, 
                       time_cost=3, memory_cost=65536, parallelism=4)

# ❌ Bad: Use cryptographic hashes for passwords
hashed = hash_data(password, HashAlgorithm.SHA256)  # Too fast
```

### 4. Random Generation

```python
# ✅ Good: Use cryptographically secure random
random_bytes = generate_random_bytes(32)
random_string = generate_random_string(16)

# ✅ Good: Use appropriate entropy sources
crypto_random = CryptoRandom()
random_bytes = crypto_random.generate_bytes(32)

# ❌ Bad: Use predictable random
import random
random_bytes = bytes([random.randint(0, 255) for _ in range(32)])  # Predictable
```

### 5. Error Handling

```python
# ✅ Good: Handle cryptographic errors gracefully
try:
    decrypted = decrypt_symmetric(encrypted_data, key)
except Exception as e:
    logger.error(f"Decryption failed: {e}")
    # Handle gracefully

# ✅ Good: Validate inputs
if not key or len(key) < 16:
    raise ValueError("Invalid key")

# ❌ Bad: Ignore errors
decrypted = decrypt_symmetric(encrypted_data, key)  # May fail silently
```

### 6. Secure Storage

```python
# ✅ Good: Encrypt sensitive data at rest
encrypted_key = encrypt_symmetric(key, master_key)
store_encrypted(encrypted_key)

# ✅ Good: Use secure key storage
key_store = KeyStore(encryption_key=master_key)
key_id = key_store.store(key)

# ❌ Bad: Store keys in plain text
with open("keys.txt", "w") as f:
    f.write(key.decode())  # Insecure
```

## 📚 Examples

### Complete Encryption Example

```python
from cryptography import (
    SymmetricCryptoService, AsymmetricCryptoService,
    HashService, KeyManager, SecureRandom
)

# Initialize services
symmetric_service = SymmetricCryptoService()
asymmetric_service = AsymmetricCryptoService()
hash_service = HashService()
key_manager = KeyManager()
random = SecureRandom()

# Generate keys
master_key = key_manager.generate_master_key()
symmetric_key = key_manager.generate_key(KeyType.SYMMETRIC, KeySize.AES_256)
key_pair = key_manager.generate_key_pair(KeyType.RSA, KeySize.RSA_2048)

# Encrypt data
data = b"Sensitive data to encrypt"
encrypted = symmetric_service.encrypt(data, SymmetricAlgorithm.AES, symmetric_key)

# Sign data
signature = asymmetric_service.sign(data, key_pair.private_key, AsymmetricAlgorithm.RSA)

# Hash password
password = b"user-password"
hashed = hash_service.hash(password, HashAlgorithm.ARGON2)

# Store keys securely
key_id = key_manager.store_key(symmetric_key, KeyUsage.ENCRYPTION)

print(f"Encrypted: {encrypted.ciphertext.hex()}")
print(f"Signature: {signature.signature.hex()}")
print(f"Password hash: {hashed.hash_value.decode()[:50]}...")
print(f"Key ID: {key_id}")
```

### Secure Communication Example

```python
from cryptography import (
    generate_key_pair, encrypt_asymmetric, decrypt_asymmetric,
    sign_data, verify_signature, AsymmetricAlgorithm
)

# Alice generates key pair
alice_keys = generate_key_pair(AsymmetricAlgorithm.RSA)

# Bob generates key pair
bob_keys = generate_key_pair(AsymmetricAlgorithm.RSA)

# Alice sends encrypted message to Bob
message = b"Hello Bob, this is a secret message!"
encrypted_message = encrypt_asymmetric(message, bob_keys.public_key, AsymmetricAlgorithm.RSA)

# Alice signs the encrypted message
signature = sign_data(encrypted_message, alice_keys.private_key, AsymmetricAlgorithm.RSA)

# Bob receives and verifies
verified = verify_signature(encrypted_message, signature.signature, alice_keys.public_key, AsymmetricAlgorithm.RSA)
if verified.verified:
    decrypted_message = decrypt_asymmetric(encrypted_message, bob_keys.private_key, AsymmetricAlgorithm.RSA)
    print(f"Message from Alice: {decrypted_message.decode()}")
else:
    print("Message verification failed!")
```

### Password Management Example

```python
from cryptography import hash_data, verify_hash, HashAlgorithm

class PasswordManager:
    def __init__(self):
        self.users = {}
    
    def register_user(self, username: str, password: str):
        """Register a new user with hashed password"""
        password_bytes = password.encode()
        hashed = hash_data(password_bytes, HashAlgorithm.ARGON2)
        
        self.users[username] = {
            'hash': hashed.hash_value,
            'salt': hashed.salt,
            'algorithm': hashed.algorithm
        }
        print(f"User {username} registered successfully")
    
    def authenticate_user(self, username: str, password: str) -> bool:
        """Authenticate user with password"""
        if username not in self.users:
            return False
        
        user_data = self.users[username]
        password_bytes = password.encode()
        
        verified = verify_hash(password_bytes, user_data['hash'], user_data['algorithm'])
        return verified.verified

# Usage
password_manager = PasswordManager()

# Register users
password_manager.register_user("alice", "secure-password-123")
password_manager.register_user("bob", "another-secure-password")

# Authenticate users
print(f"Alice auth: {password_manager.authenticate_user('alice', 'secure-password-123')}")
print(f"Alice wrong password: {password_manager.authenticate_user('alice', 'wrong-password')}")
print(f"Bob auth: {password_manager.authenticate_user('bob', 'another-secure-password')}")
```

## 📖 API Reference

### SymmetricCryptoService

```python
class SymmetricCryptoService:
    def encrypt(self, data: bytes, algorithm: SymmetricAlgorithm, key: Optional[bytes] = None, password: Optional[str] = None, **kwargs) -> EncryptionResult
    def decrypt(self, data: bytes, key: bytes, algorithm: SymmetricAlgorithm, **kwargs) -> DecryptionResult
    def generate_key(self, algorithm: SymmetricAlgorithm, password: Optional[str] = None) -> bytes
    def get_supported_algorithms(self) -> List[SymmetricAlgorithm]
```

### AsymmetricCryptoService

```python
class AsymmetricCryptoService:
    def generate_key_pair(self, algorithm: AsymmetricAlgorithm) -> KeyPair
    def encrypt(self, data: bytes, public_key: bytes, algorithm: AsymmetricAlgorithm) -> bytes
    def decrypt(self, data: bytes, private_key: bytes, algorithm: AsymmetricAlgorithm) -> bytes
    def sign(self, data: bytes, private_key: bytes, algorithm: AsymmetricAlgorithm, hash_algorithm: str = "sha256") -> SignatureResult
    def verify(self, data: bytes, signature: bytes, public_key: bytes, algorithm: AsymmetricAlgorithm, hash_algorithm: str = "sha256") -> VerificationResult
    def get_supported_algorithms(self) -> List[AsymmetricAlgorithm]
```

### HashService

```python
class HashService:
    def hash(self, data: bytes, algorithm: HashAlgorithm, salt: Optional[bytes] = None, **kwargs) -> HashResult
    def verify(self, data: bytes, hash_value: bytes, algorithm: HashAlgorithm, salt: Optional[bytes] = None, **kwargs) -> VerificationResult
    def get_supported_algorithms(self) -> List[HashAlgorithm]
```

### KeyManager

```python
class KeyManager:
    def generate_master_key(self) -> bytes
    def generate_key(self, key_type: KeyType, key_size: KeySize) -> bytes
    def generate_key_pair(self, key_type: KeyType, key_size: KeySize) -> KeyPair
    def derive_key(self, master_key: bytes, purpose: str, key_size: KeySize) -> bytes
    def store_key(self, key: bytes, usage: KeyUsage) -> str
    def get_key(self, key_id: str) -> bytes
    def rotate_key(self, key_id: str) -> str
    def backup_keys(self) -> bytes
    def restore_keys(self, backup_data: bytes)
```

### CryptoUtils

```python
class CryptoUtils:
    def encode_data(self, data: bytes, format: EncodingFormat, **kwargs) -> Union[str, bytes]
    def decode_data(self, data: Union[str, bytes], format: EncodingFormat, **kwargs) -> bytes
    def format_key(self, key: bytes, format: KeyFormat, key_type: str = "RSA", **kwargs) -> Union[str, bytes]
    def parse_key(self, key_data: Union[str, bytes], format: KeyFormat, **kwargs) -> bytes
    def convert_format(self, data: Union[str, bytes], from_format: EncodingFormat, to_format: EncodingFormat, **kwargs) -> Union[str, bytes]
    def validate_format(self, data: Union[str, bytes], format: EncodingFormat, **kwargs) -> bool
    def get_data_info(self, data: Union[str, bytes], format: EncodingFormat, **kwargs) -> Dict[str, Any]
```

### Convenience Functions

```python
# Symmetric cryptography
encrypt_symmetric(data: bytes, algorithm: SymmetricAlgorithm = SymmetricAlgorithm.AES, key: Optional[bytes] = None, password: Optional[str] = None, **kwargs) -> EncryptionResult
decrypt_symmetric(data: bytes, key: bytes, algorithm: SymmetricAlgorithm = SymmetricAlgorithm.AES, **kwargs) -> DecryptionResult
generate_symmetric_key(algorithm: SymmetricAlgorithm = SymmetricAlgorithm.AES, password: Optional[str] = None) -> bytes

# Asymmetric cryptography
generate_key_pair(algorithm: AsymmetricAlgorithm = AsymmetricAlgorithm.RSA) -> KeyPair
encrypt_asymmetric(data: bytes, public_key: bytes, algorithm: AsymmetricAlgorithm = AsymmetricAlgorithm.RSA) -> bytes
decrypt_asymmetric(data: bytes, private_key: bytes, algorithm: AsymmetricAlgorithm = AsymmetricAlgorithm.RSA) -> bytes
sign_data(data: bytes, private_key: bytes, algorithm: AsymmetricAlgorithm = AsymmetricAlgorithm.RSA, hash_algorithm: str = "sha256") -> SignatureResult
verify_signature(data: bytes, signature: bytes, public_key: bytes, algorithm: AsymmetricAlgorithm = AsymmetricAlgorithm.RSA, hash_algorithm: str = "sha256") -> VerificationResult

# Hash functions
hash_data(data: bytes, algorithm: HashAlgorithm = HashAlgorithm.SHA256, salt: Optional[bytes] = None, **kwargs) -> HashResult
verify_hash(data: bytes, hash_value: bytes, algorithm: HashAlgorithm = HashAlgorithm.SHA256, salt: Optional[bytes] = None, **kwargs) -> VerificationResult
bcrypt_hash(data: bytes, rounds: int = 12) -> HashResult
scrypt_hash(data: bytes, n: int = 16384, r: int = 8, p: int = 1) -> HashResult
pbkdf2_hash(data: bytes, iterations: int = 100000) -> HashResult

# Crypto utils
encode_data(data: bytes, format: EncodingFormat = EncodingFormat.BASE64, **kwargs) -> Union[str, bytes]
decode_data(data: Union[str, bytes], format: EncodingFormat = EncodingFormat.BASE64, **kwargs) -> bytes
format_key(key: bytes, format: KeyFormat = KeyFormat.PEM, key_type: str = "RSA", **kwargs) -> Union[str, bytes]
parse_key(key_data: Union[str, bytes], format: KeyFormat = KeyFormat.PEM, **kwargs) -> bytes
convert_format(data: Union[str, bytes], from_format: EncodingFormat, to_format: EncodingFormat, **kwargs) -> Union[str, bytes]
validate_format(data: Union[str, bytes], format: EncodingFormat, **kwargs) -> bool
```

## 🤝 Contributing

When contributing to the cryptography module:

1. **Follow security best practices** for all cryptographic operations
2. **Use established algorithms** and avoid implementing custom cryptography
3. **Add comprehensive tests** for all new functionality
4. **Update documentation** for new features
5. **Use type hints** for all new functions and classes
6. **Follow the error handling patterns** established in the module

## 📄 License

This module is part of the Video-OpusClip project and follows the same license terms. 