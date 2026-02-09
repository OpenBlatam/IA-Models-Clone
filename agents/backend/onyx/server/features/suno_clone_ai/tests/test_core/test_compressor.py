"""
Tests para compressor
"""

import pytest
import tempfile
import os
from unittest.mock import patch

from core.compression.compressor import Compressor
from test_helpers import BaseServiceTestCase, StandardTestMixin


class TestCompressor(BaseServiceTestCase, StandardTestMixin):
    """Tests para Compressor"""
    
    @pytest.fixture
    def compressor_gzip(self):
        """Fixture para Compressor con gzip"""
        return Compressor(algorithm="gzip")
    
    @pytest.fixture
    def temp_file(self):
        """Archivo temporal"""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_path = f.name
            yield temp_path
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_compressor_init(self, compressor_gzip):
        """Test de inicialización"""
        assert compressor_gzip.algorithm == "gzip"
    
    def test_compress_decompress_gzip(self, compressor_gzip):
        """Test de compresión y descompresión con gzip"""
        data = {"key": "value", "number": 123, "list": [1, 2, 3]}
        
        compressed = compressor_gzip.compress(data)
        decompressed = compressor_gzip.decompress(compressed)
        
        assert decompressed == data
        assert len(compressed) < len(str(data).encode())  # Debería ser más pequeño
    
    @pytest.mark.parametrize("data", [
        {"key": "value"},
        [1, 2, 3, 4, 5],
        "test string",
        12345,
        {"nested": {"deep": {"value": 42}}}
    ])
    def test_compress_decompress_different_types(self, compressor_gzip, data):
        """Test de compresión con diferentes tipos de datos"""
        compressed = compressor_gzip.compress(data)
        decompressed = compressor_gzip.decompress(compressed)
        
        assert decompressed == data
    
    def test_compress_to_file(self, compressor_gzip, temp_file):
        """Test de compresión a archivo"""
        data = {"key": "value"}
        
        compressed = compressor_gzip.compress(data, file_path=temp_file)
        
        assert os.path.exists(temp_file)
        assert len(compressed) > 0
    
    def test_decompress_from_file(self, compressor_gzip, temp_file):
        """Test de descompresión desde archivo"""
        data = {"key": "value"}
        
        compressor_gzip.compress(data, file_path=temp_file)
        decompressed = compressor_gzip.decompress(None, file_path=temp_file)
        
        assert decompressed == data
    
    @patch('core.compression.compressor.LZ4_AVAILABLE', True)
    def test_compress_lz4(self):
        """Test de compresión con lz4"""
        with patch('core.compression.compressor.lz4'):
            compressor = Compressor(algorithm="lz4")
            data = {"key": "value"}
            
            # Mock lz4 compression
            import core.compression.compressor as comp_module
            comp_module.lz4.frame.compress = lambda x: b"compressed"
            comp_module.lz4.frame.decompress = lambda x: x
            
            compressed = compressor.compress(data)
            
            assert compressed is not None
    
    def test_compress_unknown_algorithm(self):
        """Test de compresión con algoritmo desconocido"""
        compressor = Compressor(algorithm="unknown")
        data = {"key": "value"}
        
        with pytest.raises(ValueError, match="Unknown algorithm"):
            compressor.compress(data)
    
    def test_decompress_unknown_algorithm(self):
        """Test de descompresión con algoritmo desconocido"""
        compressor = Compressor(algorithm="unknown")
        
        with pytest.raises(ValueError, match="Unknown algorithm"):
            compressor.decompress(b"data")



