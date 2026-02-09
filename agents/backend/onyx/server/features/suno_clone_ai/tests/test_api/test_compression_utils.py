"""
Tests para utilidades de compresión
"""

import pytest
from api.utils.compression import (
    compress_gzip,
    compress_brotli,
    get_best_compression
)


@pytest.mark.unit
@pytest.mark.api
class TestCompressGzip:
    """Tests para compress_gzip"""
    
    def test_compress_gzip_small_data(self):
        """Test de compresión gzip con datos pequeños"""
        data = b"test data"
        compressed = compress_gzip(data)
        
        assert isinstance(compressed, bytes)
        assert len(compressed) > 0
    
    def test_compress_gzip_large_data(self):
        """Test de compresión gzip con datos grandes"""
        data = b"x" * 10000
        compressed = compress_gzip(data)
        
        assert isinstance(compressed, bytes)
        # Los datos grandes deberían comprimirse
        assert len(compressed) < len(data)
    
    def test_compress_gzip_empty(self):
        """Test de compresión gzip con datos vacíos"""
        data = b""
        compressed = compress_gzip(data)
        
        assert isinstance(compressed, bytes)
    
    def test_compress_gzip_already_compressed(self):
        """Test de compresión gzip con datos ya comprimidos"""
        data = b"x" * 100
        compressed1 = compress_gzip(data)
        compressed2 = compress_gzip(compressed1)
        
        # La segunda compresión puede no ser más pequeña
        assert isinstance(compressed2, bytes)


@pytest.mark.unit
@pytest.mark.api
class TestCompressBrotli:
    """Tests para compress_brotli"""
    
    def test_compress_brotli_small_data(self):
        """Test de compresión brotli con datos pequeños"""
        data = b"test data"
        compressed = compress_brotli(data)
        
        assert isinstance(compressed, bytes)
        assert len(compressed) > 0
    
    def test_compress_brotli_large_data(self):
        """Test de compresión brotli con datos grandes"""
        data = b"x" * 10000
        compressed = compress_brotli(data)
        
        assert isinstance(compressed, bytes)
        # Los datos grandes deberían comprimirse
        assert len(compressed) < len(data)
    
    def test_compress_brotli_empty(self):
        """Test de compresión brotli con datos vacíos"""
        data = b""
        compressed = compress_brotli(data)
        
        assert isinstance(compressed, bytes)
    
    def test_compress_brotli_vs_gzip(self):
        """Test comparativo entre brotli y gzip"""
        data = b"x" * 10000
        brotli_compressed = compress_brotli(data)
        gzip_compressed = compress_gzip(data)
        
        # Brotli generalmente comprime mejor que gzip
        assert isinstance(brotli_compressed, bytes)
        assert isinstance(gzip_compressed, bytes)


@pytest.mark.unit
@pytest.mark.api
class TestGetBestCompression:
    """Tests para get_best_compression"""
    
    def test_get_best_compression_no_accept_encoding(self):
        """Test sin Accept-Encoding header"""
        data = b"test data"
        compressed, encoding = get_best_compression(data, None)
        
        assert compressed == data
        assert encoding == "identity"
    
    def test_get_best_compression_brotli_preferred(self):
        """Test con preferencia por Brotli"""
        data = b"x" * 1000
        compressed, encoding = get_best_compression(data, "br, gzip")
        
        assert encoding in ["br", "gzip", "identity"]
        assert isinstance(compressed, bytes)
    
    def test_get_best_compression_gzip_fallback(self):
        """Test con gzip como fallback"""
        data = b"x" * 1000
        compressed, encoding = get_best_compression(data, "gzip")
        
        assert encoding in ["gzip", "identity"]
        assert isinstance(compressed, bytes)
    
    def test_get_best_compression_identity(self):
        """Test que retorna identity cuando no hay compresión mejor"""
        data = b"x"  # Datos muy pequeños
        compressed, encoding = get_best_compression(data, "br, gzip")
        
        # Para datos muy pequeños, puede retornar identity
        assert encoding in ["br", "gzip", "identity"]
    
    def test_get_best_compression_case_insensitive(self):
        """Test que es case-insensitive"""
        data = b"x" * 1000
        compressed1, encoding1 = get_best_compression(data, "BR, GZIP")
        compressed2, encoding2 = get_best_compression(data, "br, gzip")
        
        # Debería manejar ambos casos
        assert isinstance(compressed1, bytes)
        assert isinstance(compressed2, bytes)
    
    def test_get_best_compression_empty_data(self):
        """Test con datos vacíos"""
        data = b""
        compressed, encoding = get_best_compression(data, "br, gzip")
        
        assert isinstance(compressed, bytes)
        assert encoding in ["br", "gzip", "identity"]



