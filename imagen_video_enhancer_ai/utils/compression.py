"""
Compression utilities for Imagen Video Enhancer AI
===================================================

Compress and decompress results and files.
"""

import gzip
import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class CompressionManager:
    """
    Manages compression of results and files.
    
    Features:
    - JSON compression
    - File compression
    - Automatic compression levels
    """
    
    @staticmethod
    def compress_json(
        data: Dict[str, Any],
        output_path: str,
        compression_level: int = 6
    ) -> str:
        """
        Compress JSON data to gzip.
        
        Args:
            data: Data dictionary
            output_path: Output file path
            compression_level: Compression level (1-9)
            
        Returns:
            Path to compressed file
        """
        output_file = Path(output_path)
        if not output_file.suffix == ".gz":
            output_file = output_file.with_suffix(output_file.suffix + ".gz")
        
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        json_str = json.dumps(data, ensure_ascii=False)
        json_bytes = json_str.encode('utf-8')
        
        with gzip.open(output_file, 'wb', compresslevel=compression_level) as f:
            f.write(json_bytes)
        
        original_size = len(json_bytes)
        compressed_size = output_file.stat().st_size
        ratio = (1 - compressed_size / original_size) * 100 if original_size > 0 else 0
        
        logger.info(
            f"Compressed JSON: {original_size} -> {compressed_size} bytes "
            f"({ratio:.1f}% reduction)"
        )
        
        return str(output_file)
    
    @staticmethod
    def decompress_json(compressed_path: str) -> Dict[str, Any]:
        """
        Decompress gzipped JSON file.
        
        Args:
            compressed_path: Path to compressed file
            
        Returns:
            Decompressed data dictionary
        """
        with gzip.open(compressed_path, 'rb') as f:
            json_bytes = f.read()
        
        json_str = json_bytes.decode('utf-8')
        return json.loads(json_str)
    
    @staticmethod
    def compress_file(
        file_path: str,
        output_path: Optional[str] = None,
        compression_level: int = 6
    ) -> str:
        """
        Compress a file to gzip.
        
        Args:
            file_path: Path to file to compress
            output_path: Optional output path (defaults to file_path.gz)
            compression_level: Compression level (1-9)
            
        Returns:
            Path to compressed file
        """
        input_file = Path(file_path)
        
        if output_path:
            output_file = Path(output_path)
        else:
            output_file = input_file.with_suffix(input_file.suffix + ".gz")
        
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        original_size = input_file.stat().st_size
        
        with open(input_file, 'rb') as f_in:
            with gzip.open(output_file, 'wb', compresslevel=compression_level) as f_out:
                f_out.writelines(f_in)
        
        compressed_size = output_file.stat().st_size
        ratio = (1 - compressed_size / original_size) * 100 if original_size > 0 else 0
        
        logger.info(
            f"Compressed file: {original_size} -> {compressed_size} bytes "
            f"({ratio:.1f}% reduction)"
        )
        
        return str(output_file)
    
    @staticmethod
    def decompress_file(
        compressed_path: str,
        output_path: Optional[str] = None
    ) -> str:
        """
        Decompress a gzipped file.
        
        Args:
            compressed_path: Path to compressed file
            output_path: Optional output path
            
        Returns:
            Path to decompressed file
        """
        compressed_file = Path(compressed_path)
        
        if output_path:
            output_file = Path(output_path)
        else:
            # Remove .gz extension
            if compressed_file.suffix == ".gz":
                output_file = compressed_file.with_suffix("")
            else:
                output_file = compressed_file.parent / compressed_file.stem
        
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with gzip.open(compressed_file, 'rb') as f_in:
            with open(output_file, 'wb') as f_out:
                f_out.write(f_in.read())
        
        logger.info(f"Decompressed file: {compressed_file} -> {output_file}")
        return str(output_file)




