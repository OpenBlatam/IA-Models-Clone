"""
Sistema de Compresión Inteligente
==================================

Sistema para comprimir documentos y resultados de manera inteligente.
"""

import logging
import gzip
import zlib
import bz2
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import json

logger = logging.getLogger(__name__)


class CompressionMethod(Enum):
    """Métodos de compresión"""
    GZIP = "gzip"
    ZLIB = "zlib"
    BZ2 = "bz2"
    NONE = "none"


@dataclass
class CompressionResult:
    """Resultado de compresión"""
    original_size: int
    compressed_size: int
    compression_ratio: float
    method: CompressionMethod
    time_taken: float


class IntelligentCompressor:
    """
    Compresor inteligente
    
    Comprime documentos y resultados de análisis
    de manera eficiente y selectiva.
    """
    
    def __init__(self):
        """Inicializar compresor"""
        logger.info("IntelligentCompressor inicializado")
    
    def compress_data(
        self,
        data: Any,
        method: CompressionMethod = CompressionMethod.GZIP
    ) -> tuple[bytes, CompressionResult]:
        """
        Comprimir datos
        
        Args:
            data: Datos a comprimir (dict, str, bytes)
            method: Método de compresión
        
        Returns:
            Tupla (datos_comprimidos, resultado)
        """
        import time
        start_time = time.time()
        
        # Convertir a bytes si es necesario
        if isinstance(data, dict):
            data_bytes = json.dumps(data, ensure_ascii=False).encode('utf-8')
        elif isinstance(data, str):
            data_bytes = data.encode('utf-8')
        else:
            data_bytes = data
        
        original_size = len(data_bytes)
        
        # Comprimir según método
        if method == CompressionMethod.GZIP:
            compressed = gzip.compress(data_bytes)
        elif method == CompressionMethod.ZLIB:
            compressed = zlib.compress(data_bytes)
        elif method == CompressionMethod.BZ2:
            compressed = bz2.compress(data_bytes)
        else:
            compressed = data_bytes
        
        compressed_size = len(compressed)
        compression_ratio = compressed_size / original_size if original_size > 0 else 0.0
        time_taken = time.time() - start_time
        
        result = CompressionResult(
            original_size=original_size,
            compressed_size=compressed_size,
            compression_ratio=compression_ratio,
            method=method,
            time_taken=time_taken
        )
        
        return compressed, result
    
    def decompress_data(
        self,
        compressed_data: bytes,
        method: CompressionMethod = CompressionMethod.GZIP
    ) -> bytes:
        """
        Descomprimir datos
        
        Args:
            compressed_data: Datos comprimidos
            method: Método de compresión usado
        
        Returns:
            Datos descomprimidos
        """
        if method == CompressionMethod.GZIP:
            return gzip.decompress(compressed_data)
        elif method == CompressionMethod.ZLIB:
            return zlib.decompress(compressed_data)
        elif method == CompressionMethod.BZ2:
            return bz2.decompress(compressed_data)
        else:
            return compressed_data
    
    def compress_analysis_result(
        self,
        analysis_result: Dict[str, Any],
        compress_large_fields: bool = True,
        threshold: int = 1000
    ) -> Dict[str, Any]:
        """
        Comprimir resultado de análisis de manera inteligente
        
        Args:
            analysis_result: Resultado de análisis
            compress_large_fields: Si True, comprime campos grandes
            threshold: Umbral de tamaño para comprimir
        
        Returns:
            Resultado con campos comprimidos
        """
        compressed_result = {}
        
        for key, value in analysis_result.items():
            if isinstance(value, str) and len(value) > threshold and compress_large_fields:
                # Comprimir campos grandes
                compressed, comp_result = self.compress_data(value)
                compressed_result[f"{key}_compressed"] = compressed.hex()  # Convertir a hex para JSON
                compressed_result[f"{key}_compression"] = comp_result.compression_ratio
            elif isinstance(value, dict):
                # Comprimir recursivamente
                compressed_result[key] = self.compress_analysis_result(value, compress_large_fields, threshold)
            else:
                compressed_result[key] = value
        
        return compressed_result
    
    def select_best_method(
        self,
        data: bytes,
        sample_size: int = 10000
    ) -> CompressionMethod:
        """
        Seleccionar mejor método de compresión
        
        Args:
            data: Datos a comprimir
            sample_size: Tamaño de muestra para probar
        
        Returns:
            Mejor método de compresión
        """
        sample = data[:sample_size] if len(data) > sample_size else data
        
        results = {}
        for method in [CompressionMethod.GZIP, CompressionMethod.ZLIB, CompressionMethod.BZ2]:
            try:
                _, result = self.compress_data(sample, method)
                results[method] = result.compression_ratio
            except:
                continue
        
        if results:
            best_method = min(results.items(), key=lambda x: x[1])[0]
            return best_method
        
        return CompressionMethod.GZIP
















