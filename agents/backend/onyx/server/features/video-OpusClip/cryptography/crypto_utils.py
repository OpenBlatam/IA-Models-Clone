#!/usr/bin/env python3
"""
Crypto Utils for Video-OpusClip
Utility functions for encoding, decoding, and formatting cryptographic data
"""

import base64
import binascii
from typing import Dict, List, Any, Optional, Union, Tuple
from enum import Enum
from dataclasses import dataclass
import json
import logging

logger = logging.getLogger(__name__)


class EncodingFormat(Enum):
    """Data encoding formats"""
    BASE64 = "base64"
    BASE64_URLSAFE = "base64_urlsafe"
    HEX = "hex"
    PEM = "pem"
    DER = "der"
    JSON = "json"
    RAW = "raw"


class KeyFormat(Enum):
    """Key formats"""
    PEM = "pem"
    DER = "der"
    SSH = "ssh"
    JWK = "jwk"
    RAW = "raw"


@dataclass
class EncodedData:
    """Encoded data container"""
    data: bytes
    format: EncodingFormat
    encoding: str = "utf-8"
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class Base64Encoder:
    """Base64 encoding utilities"""
    
    @staticmethod
    def encode(data: bytes, urlsafe: bool = False) -> str:
        """Encode data to base64"""
        try:
            if urlsafe:
                return base64.urlsafe_b64encode(data).decode('utf-8')
            else:
                return base64.b64encode(data).decode('utf-8')
        except Exception as e:
            logger.error(f"Base64 encoding failed: {e}")
            raise
    
    @staticmethod
    def decode(data: str, urlsafe: bool = False) -> bytes:
        """Decode data from base64"""
        try:
            if urlsafe:
                return base64.urlsafe_b64decode(data)
            else:
                return base64.b64decode(data)
        except Exception as e:
            logger.error(f"Base64 decoding failed: {e}")
            raise
    
    @staticmethod
    def encode_with_padding(data: bytes, urlsafe: bool = False) -> str:
        """Encode data to base64 with padding"""
        try:
            if urlsafe:
                return base64.urlsafe_b64encode(data).decode('utf-8')
            else:
                return base64.b64encode(data).decode('utf-8')
        except Exception as e:
            logger.error(f"Base64 encoding with padding failed: {e}")
            raise
    
    @staticmethod
    def decode_with_padding(data: str, urlsafe: bool = False) -> bytes:
        """Decode data from base64 with padding"""
        try:
            # Add padding if needed
            missing_padding = len(data) % 4
            if missing_padding:
                data += '=' * (4 - missing_padding)
            
            if urlsafe:
                return base64.urlsafe_b64decode(data)
            else:
                return base64.b64decode(data)
        except Exception as e:
            logger.error(f"Base64 decoding with padding failed: {e}")
            raise


class HexEncoder:
    """Hexadecimal encoding utilities"""
    
    @staticmethod
    def encode(data: bytes, separator: str = "") -> str:
        """Encode data to hexadecimal"""
        try:
            return data.hex()
        except Exception as e:
            logger.error(f"Hex encoding failed: {e}")
            raise
    
    @staticmethod
    def decode(data: str, separator: str = "") -> bytes:
        """Decode data from hexadecimal"""
        try:
            # Remove separators if present
            clean_data = data.replace(separator, "")
            return binascii.unhexlify(clean_data)
        except Exception as e:
            logger.error(f"Hex decoding failed: {e}")
            raise
    
    @staticmethod
    def encode_with_separator(data: bytes, separator: str = ":", group_size: int = 2) -> str:
        """Encode data to hexadecimal with separators"""
        try:
            hex_str = data.hex()
            if separator and group_size > 0:
                groups = [hex_str[i:i+group_size] for i in range(0, len(hex_str), group_size)]
                return separator.join(groups)
            return hex_str
        except Exception as e:
            logger.error(f"Hex encoding with separator failed: {e}")
            raise
    
    @staticmethod
    def decode_with_separator(data: str, separator: str = ":") -> bytes:
        """Decode data from hexadecimal with separators"""
        try:
            clean_data = data.replace(separator, "")
            return binascii.unhexlify(clean_data)
        except Exception as e:
            logger.error(f"Hex decoding with separator failed: {e}")
            raise


class PEMEncoder:
    """PEM encoding utilities"""
    
    @staticmethod
    def encode(data: bytes, header: str = "-----BEGIN DATA-----", footer: str = "-----END DATA-----") -> str:
        """Encode data to PEM format"""
        try:
            base64_data = base64.b64encode(data).decode('utf-8')
            
            # Split into lines of 64 characters
            lines = [base64_data[i:i+64] for i in range(0, len(base64_data), 64)]
            
            pem_lines = [header]
            pem_lines.extend(lines)
            pem_lines.append(footer)
            
            return '\n'.join(pem_lines)
        except Exception as e:
            logger.error(f"PEM encoding failed: {e}")
            raise
    
    @staticmethod
    def decode(data: str, header: str = "-----BEGIN DATA-----", footer: str = "-----END DATA-----") -> bytes:
        """Decode data from PEM format"""
        try:
            lines = data.strip().split('\n')
            
            # Remove header and footer
            if lines[0] == header:
                lines = lines[1:]
            if lines[-1] == footer:
                lines = lines[:-1]
            
            # Join lines and decode
            base64_data = ''.join(lines)
            return base64.b64decode(base64_data)
        except Exception as e:
            logger.error(f"PEM decoding failed: {e}")
            raise
    
    @staticmethod
    def encode_private_key(data: bytes, algorithm: str = "RSA") -> str:
        """Encode private key to PEM format"""
        header = f"-----BEGIN {algorithm.upper()} PRIVATE KEY-----"
        footer = f"-----END {algorithm.upper()} PRIVATE KEY-----"
        return PEMEncoder.encode(data, header, footer)
    
    @staticmethod
    def encode_public_key(data: bytes, algorithm: str = "RSA") -> str:
        """Encode public key to PEM format"""
        header = f"-----BEGIN {algorithm.upper()} PUBLIC KEY-----"
        footer = f"-----END {algorithm.upper()} PUBLIC KEY-----"
        return PEMEncoder.encode(data, header, footer)
    
    @staticmethod
    def encode_certificate(data: bytes) -> str:
        """Encode certificate to PEM format"""
        header = "-----BEGIN CERTIFICATE-----"
        footer = "-----END CERTIFICATE-----"
        return PEMEncoder.encode(data, header, footer)


class CryptoUtils:
    """Main crypto utilities class"""
    
    def __init__(self):
        self.base64_encoder = Base64Encoder()
        self.hex_encoder = HexEncoder()
        self.pem_encoder = PEMEncoder()
    
    def encode_data(
        self,
        data: bytes,
        format: EncodingFormat = EncodingFormat.BASE64,
        **kwargs
    ) -> Union[str, bytes]:
        """Encode data in specified format"""
        try:
            if format == EncodingFormat.BASE64:
                return self.base64_encoder.encode(data, urlsafe=False)
            elif format == EncodingFormat.BASE64_URLSAFE:
                return self.base64_encoder.encode(data, urlsafe=True)
            elif format == EncodingFormat.HEX:
                return self.hex_encoder.encode(data)
            elif format == EncodingFormat.PEM:
                return self.pem_encoder.encode(data)
            elif format == EncodingFormat.JSON:
                return json.dumps({"data": self.base64_encoder.encode(data)})
            elif format == EncodingFormat.RAW:
                return data
            else:
                raise ValueError(f"Unsupported encoding format: {format}")
                
        except Exception as e:
            logger.error(f"Data encoding failed: {e}")
            raise
    
    def decode_data(
        self,
        data: Union[str, bytes],
        format: EncodingFormat = EncodingFormat.BASE64,
        **kwargs
    ) -> bytes:
        """Decode data from specified format"""
        try:
            if format == EncodingFormat.BASE64:
                return self.base64_encoder.decode(data, urlsafe=False)
            elif format == EncodingFormat.BASE64_URLSAFE:
                return self.base64_encoder.decode(data, urlsafe=True)
            elif format == EncodingFormat.HEX:
                return self.hex_encoder.decode(data)
            elif format == EncodingFormat.PEM:
                return self.pem_encoder.decode(data)
            elif format == EncodingFormat.JSON:
                json_data = json.loads(data)
                return self.base64_encoder.decode(json_data["data"])
            elif format == EncodingFormat.RAW:
                return data if isinstance(data, bytes) else data.encode()
            else:
                raise ValueError(f"Unsupported encoding format: {format}")
                
        except Exception as e:
            logger.error(f"Data decoding failed: {e}")
            raise
    
    def format_key(
        self,
        key: bytes,
        format: KeyFormat = KeyFormat.PEM,
        key_type: str = "RSA",
        **kwargs
    ) -> Union[str, bytes]:
        """Format key in specified format"""
        try:
            if format == KeyFormat.PEM:
                if "private" in key_type.lower():
                    return self.pem_encoder.encode_private_key(key, key_type)
                elif "public" in key_type.lower():
                    return self.pem_encoder.encode_public_key(key, key_type)
                else:
                    return self.pem_encoder.encode(key)
            elif format == KeyFormat.DER:
                return key
            elif format == KeyFormat.SSH:
                # SSH format is similar to PEM but with different headers
                if "public" in key_type.lower():
                    return f"ssh-{key_type.lower()} {self.base64_encoder.encode(key)}"
                else:
                    raise ValueError("SSH format is only supported for public keys")
            elif format == KeyFormat.JWK:
                # JWK format
                jwk = {
                    "kty": key_type.upper(),
                    "k": self.base64_encoder.encode(key, urlsafe=True),
                    "alg": f"{key_type.upper()}256"
                }
                return json.dumps(jwk)
            elif format == KeyFormat.RAW:
                return key
            else:
                raise ValueError(f"Unsupported key format: {format}")
                
        except Exception as e:
            logger.error(f"Key formatting failed: {e}")
            raise
    
    def parse_key(
        self,
        key_data: Union[str, bytes],
        format: KeyFormat = KeyFormat.PEM,
        **kwargs
    ) -> bytes:
        """Parse key from specified format"""
        try:
            if format == KeyFormat.PEM:
                return self.pem_encoder.decode(key_data)
            elif format == KeyFormat.DER:
                return key_data if isinstance(key_data, bytes) else key_data.encode()
            elif format == KeyFormat.SSH:
                # Parse SSH format
                if isinstance(key_data, str) and key_data.startswith("ssh-"):
                    parts = key_data.split()
                    if len(parts) >= 2:
                        return self.base64_encoder.decode(parts[1])
                raise ValueError("Invalid SSH key format")
            elif format == KeyFormat.JWK:
                # Parse JWK format
                jwk = json.loads(key_data)
                return self.base64_encoder.decode(jwk["k"], urlsafe=True)
            elif format == KeyFormat.RAW:
                return key_data if isinstance(key_data, bytes) else key_data.encode()
            else:
                raise ValueError(f"Unsupported key format: {format}")
                
        except Exception as e:
            logger.error(f"Key parsing failed: {e}")
            raise
    
    def convert_format(
        self,
        data: Union[str, bytes],
        from_format: EncodingFormat,
        to_format: EncodingFormat,
        **kwargs
    ) -> Union[str, bytes]:
        """Convert data between formats"""
        try:
            # Decode from source format
            decoded_data = self.decode_data(data, from_format)
            
            # Encode to target format
            return self.encode_data(decoded_data, to_format, **kwargs)
            
        except Exception as e:
            logger.error(f"Format conversion failed: {e}")
            raise
    
    def validate_format(
        self,
        data: Union[str, bytes],
        format: EncodingFormat,
        **kwargs
    ) -> bool:
        """Validate if data is in specified format"""
        try:
            # Try to decode the data
            self.decode_data(data, format, **kwargs)
            return True
        except Exception:
            return False
    
    def get_data_info(
        self,
        data: Union[str, bytes],
        format: EncodingFormat,
        **kwargs
    ) -> Dict[str, Any]:
        """Get information about encoded data"""
        try:
            decoded_data = self.decode_data(data, format, **kwargs)
            
            info = {
                "format": format.value,
                "original_size": len(data) if isinstance(data, str) else len(data),
                "decoded_size": len(decoded_data),
                "is_valid": True
            }
            
            if format == EncodingFormat.HEX:
                info["hex_length"] = len(data) if isinstance(data, str) else len(data) * 2
            elif format == EncodingFormat.BASE64:
                info["base64_length"] = len(data) if isinstance(data, str) else len(data)
            
            return info
            
        except Exception as e:
            return {
                "format": format.value,
                "is_valid": False,
                "error": str(e)
            }


# Convenience functions
def encode_data(
    data: bytes,
    format: EncodingFormat = EncodingFormat.BASE64,
    **kwargs
) -> Union[str, bytes]:
    """Convenience function for encoding data"""
    utils = CryptoUtils()
    return utils.encode_data(data, format, **kwargs)


def decode_data(
    data: Union[str, bytes],
    format: EncodingFormat = EncodingFormat.BASE64,
    **kwargs
) -> bytes:
    """Convenience function for decoding data"""
    utils = CryptoUtils()
    return utils.decode_data(data, format, **kwargs)


def format_key(
    key: bytes,
    format: KeyFormat = KeyFormat.PEM,
    key_type: str = "RSA",
    **kwargs
) -> Union[str, bytes]:
    """Convenience function for formatting keys"""
    utils = CryptoUtils()
    return utils.format_key(key, format, key_type, **kwargs)


def parse_key(
    key_data: Union[str, bytes],
    format: KeyFormat = KeyFormat.PEM,
    **kwargs
) -> bytes:
    """Convenience function for parsing keys"""
    utils = CryptoUtils()
    return utils.parse_key(key_data, format, **kwargs)


def convert_format(
    data: Union[str, bytes],
    from_format: EncodingFormat,
    to_format: EncodingFormat,
    **kwargs
) -> Union[str, bytes]:
    """Convenience function for converting between formats"""
    utils = CryptoUtils()
    return utils.convert_format(data, from_format, to_format, **kwargs)


def validate_format(
    data: Union[str, bytes],
    format: EncodingFormat,
    **kwargs
) -> bool:
    """Convenience function for validating formats"""
    utils = CryptoUtils()
    return utils.validate_format(data, format, **kwargs)


# Example usage
if __name__ == "__main__":
    # Example crypto utilities
    print("🔧 Crypto Utils Example")
    
    # Test data
    test_data = b"Hello, Video-OpusClip! This is test data for encoding and decoding."
    
    # Test Base64 encoding
    print("\n" + "="*60)
    print("BASE64 ENCODING")
    print("="*60)
    
    base64_encoded = encode_data(test_data, EncodingFormat.BASE64)
    print(f"✅ Base64 encoded: {base64_encoded}")
    
    base64_decoded = decode_data(base64_encoded, EncodingFormat.BASE64)
    print(f"✅ Base64 decoded: {base64_decoded.decode()}")
    
    # Test Base64 URL-safe encoding
    print("\n" + "="*60)
    print("BASE64 URL-SAFE ENCODING")
    print("="*60)
    
    base64_url_encoded = encode_data(test_data, EncodingFormat.BASE64_URLSAFE)
    print(f"✅ Base64 URL-safe encoded: {base64_url_encoded}")
    
    base64_url_decoded = decode_data(base64_url_encoded, EncodingFormat.BASE64_URLSAFE)
    print(f"✅ Base64 URL-safe decoded: {base64_url_decoded.decode()}")
    
    # Test Hex encoding
    print("\n" + "="*60)
    print("HEX ENCODING")
    print("="*60)
    
    hex_encoded = encode_data(test_data, EncodingFormat.HEX)
    print(f"✅ Hex encoded: {hex_encoded}")
    
    hex_decoded = decode_data(hex_encoded, EncodingFormat.HEX)
    print(f"✅ Hex decoded: {hex_decoded.decode()}")
    
    # Test PEM encoding
    print("\n" + "="*60)
    print("PEM ENCODING")
    print("="*60)
    
    pem_encoded = encode_data(test_data, EncodingFormat.PEM)
    print(f"✅ PEM encoded:\n{pem_encoded}")
    
    pem_decoded = decode_data(pem_encoded, EncodingFormat.PEM)
    print(f"✅ PEM decoded: {pem_decoded.decode()}")
    
    # Test JSON encoding
    print("\n" + "="*60)
    print("JSON ENCODING")
    print("="*60)
    
    json_encoded = encode_data(test_data, EncodingFormat.JSON)
    print(f"✅ JSON encoded: {json_encoded}")
    
    json_decoded = decode_data(json_encoded, EncodingFormat.JSON)
    print(f"✅ JSON decoded: {json_decoded.decode()}")
    
    # Test key formatting
    print("\n" + "="*60)
    print("KEY FORMATTING")
    print("="*60)
    
    # Generate a dummy key
    dummy_key = b"dummy-key-data-for-testing-purposes-only"
    
    pem_key = format_key(dummy_key, KeyFormat.PEM, "RSA")
    print(f"✅ PEM key:\n{pem_key}")
    
    ssh_key = format_key(dummy_key, KeyFormat.SSH, "rsa")
    print(f"✅ SSH key: {ssh_key}")
    
    jwk_key = format_key(dummy_key, KeyFormat.JWK, "oct")
    print(f"✅ JWK key: {jwk_key}")
    
    # Test format conversion
    print("\n" + "="*60)
    print("FORMAT CONVERSION")
    print("="*60)
    
    # Convert from Base64 to Hex
    base64_to_hex = convert_format(base64_encoded, EncodingFormat.BASE64, EncodingFormat.HEX)
    print(f"✅ Base64 to Hex: {base64_to_hex}")
    
    # Convert from Hex to Base64
    hex_to_base64 = convert_format(hex_encoded, EncodingFormat.HEX, EncodingFormat.BASE64)
    print(f"✅ Hex to Base64: {hex_to_base64}")
    
    # Test format validation
    print("\n" + "="*60)
    print("FORMAT VALIDATION")
    print("="*60)
    
    is_valid_base64 = validate_format(base64_encoded, EncodingFormat.BASE64)
    print(f"✅ Valid Base64: {is_valid_base64}")
    
    is_valid_hex = validate_format(hex_encoded, EncodingFormat.HEX)
    print(f"✅ Valid Hex: {is_valid_hex}")
    
    is_valid_invalid = validate_format("invalid-data", EncodingFormat.BASE64)
    print(f"✅ Valid invalid data: {is_valid_invalid}")
    
    # Test data info
    print("\n" + "="*60)
    print("DATA INFO")
    print("="*60)
    
    utils = CryptoUtils()
    base64_info = utils.get_data_info(base64_encoded, EncodingFormat.BASE64)
    print(f"✅ Base64 info: {base64_info}")
    
    hex_info = utils.get_data_info(hex_encoded, EncodingFormat.HEX)
    print(f"✅ Hex info: {hex_info}")
    
    print("\n✅ Crypto utils example completed!") 