"""
Encoding Utilities for Piel Mejorador AI SAM3
============================================

Unified encoding and decoding utilities.
"""

import base64
import logging
from typing import Union, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class EncodingUtils:
    """Unified encoding utilities."""
    
    @staticmethod
    def encode_base64(data: Union[str, bytes]) -> str:
        """
        Encode data to base64 string.
        
        Args:
            data: String or bytes to encode
            
        Returns:
            Base64 encoded string
        """
        if isinstance(data, str):
            data = data.encode('utf-8')
        return base64.b64encode(data).decode('utf-8')
    
    @staticmethod
    def decode_base64(encoded: str) -> bytes:
        """
        Decode base64 string to bytes.
        
        Args:
            encoded: Base64 encoded string
            
        Returns:
            Decoded bytes
        """
        return base64.b64decode(encoded)
    
    @staticmethod
    def encode_base64_file(file_path: Union[str, Path]) -> str:
        """
        Encode file to base64 string.
        
        Args:
            file_path: Path to file
            
        Returns:
            Base64 encoded string
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')
    
    @staticmethod
    def decode_base64_to_file(
        encoded: str,
        output_path: Union[str, Path]
    ) -> Path:
        """
        Decode base64 string to file.
        
        Args:
            encoded: Base64 encoded string
            output_path: Path to output file
            
        Returns:
            Path to created file
        """
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        
        decoded = base64.b64decode(encoded)
        with open(output, "wb") as f:
            f.write(decoded)
        
        return output
    
    @staticmethod
    def encode_url_safe(data: Union[str, bytes]) -> str:
        """
        Encode data to URL-safe base64 string.
        
        Args:
            data: String or bytes to encode
            
        Returns:
            URL-safe base64 encoded string
        """
        if isinstance(data, str):
            data = data.encode('utf-8')
        return base64.urlsafe_b64encode(data).decode('utf-8')
    
    @staticmethod
    def decode_url_safe(encoded: str) -> bytes:
        """
        Decode URL-safe base64 string to bytes.
        
        Args:
            encoded: URL-safe base64 encoded string
            
        Returns:
            Decoded bytes
        """
        return base64.urlsafe_b64decode(encoded)
    
    @staticmethod
    def encode_hex(data: Union[str, bytes]) -> str:
        """
        Encode data to hexadecimal string.
        
        Args:
            data: String or bytes to encode
            
        Returns:
            Hexadecimal string
        """
        if isinstance(data, str):
            data = data.encode('utf-8')
        return data.hex()
    
    @staticmethod
    def decode_hex(encoded: str) -> bytes:
        """
        Decode hexadecimal string to bytes.
        
        Args:
            encoded: Hexadecimal string
            
        Returns:
            Decoded bytes
        """
        return bytes.fromhex(encoded)


# Convenience functions
def encode_base64(data: Union[str, bytes]) -> str:
    """Encode to base64."""
    return EncodingUtils.encode_base64(data)


def decode_base64(encoded: str) -> bytes:
    """Decode from base64."""
    return EncodingUtils.decode_base64(encoded)


def encode_base64_file(file_path: Union[str, Path]) -> str:
    """Encode file to base64."""
    return EncodingUtils.encode_base64_file(file_path)


def decode_base64_to_file(encoded: str, output_path: Union[str, Path]) -> Path:
    """Decode base64 to file."""
    return EncodingUtils.decode_base64_to_file(encoded, output_path)




