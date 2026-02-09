"""
Tests refactorizados para funciones helper del core
Usando clases base y helpers para eliminar duplicación
"""

import pytest
import json
import uuid
from pathlib import Path
import tempfile
import os
import asyncio

from core.helpers import (
    generate_id,
    hash_string,
    safe_json_loads,
    safe_json_dumps,
    format_duration,
    format_file_size,
    ensure_directory,
    chunk_list,
    merge_dicts,
    get_nested_value,
    set_nested_value,
    sanitize_filename,
    retry_on_failure
)
from test_helpers import BaseServiceTestCase, StandardTestMixin


class TestGenerateIDRefactored(BaseServiceTestCase, StandardTestMixin):
    """Tests refactorizados para generate_id"""
    
    def test_generate_id_no_prefix(self):
        """Test de generación sin prefijo"""
        id_str = generate_id()
        
        assert isinstance(id_str, str)
        assert len(id_str) > 0
        self.assert_valid_uuid(id_str)
    
    def test_generate_id_with_prefix(self):
        """Test de generación con prefijo"""
        id_str = generate_id(prefix="song")
        
        assert isinstance(id_str, str)
        assert id_str.startswith("song_")
        uuid_part = id_str.split("_", 1)[1]
        self.assert_valid_uuid(uuid_part)
    
    def test_generate_id_unique(self):
        """Test de que genera IDs únicos"""
        ids = [generate_id() for _ in range(100)]
        assert len(ids) == len(set(ids))


class TestHashStringRefactored(BaseServiceTestCase, StandardTestMixin):
    """Tests refactorizados para hash_string"""
    
    @pytest.mark.parametrize("algorithm,expected_length", [
        ("sha256", 64),
        ("md5", 32),
        ("sha1", 40)
    ])
    def test_hash_string_algorithms(self, algorithm, expected_length):
        """Test de hash con diferentes algoritmos"""
        result = hash_string("test", algorithm=algorithm)
        
        assert isinstance(result, str)
        assert len(result) == expected_length
    
    def test_hash_string_consistency(self):
        """Test de consistencia del hash"""
        value = "test_string"
        hash1 = hash_string(value)
        hash2 = hash_string(value)
        
        assert hash1 == hash2
    
    def test_hash_string_different_values(self):
        """Test de que valores diferentes producen hashes diferentes"""
        hash1 = hash_string("value1")
        hash2 = hash_string("value2")
        
        assert hash1 != hash2


class TestSafeJSONRefactored(BaseServiceTestCase, StandardTestMixin):
    """Tests refactorizados para safe_json_loads y safe_json_dumps"""
    
    def test_safe_json_loads_valid(self):
        """Test de carga de JSON válido"""
        json_str = '{"key": "value", "number": 123}'
        result = safe_json_loads(json_str)
        
        assert isinstance(result, dict)
        assert result["key"] == "value"
        assert result["number"] == 123
    
    def test_safe_json_loads_invalid(self):
        """Test de carga de JSON inválido"""
        json_str = '{"key": "value"'
        result = safe_json_loads(json_str, default={"error": True})
        
        assert result == {"error": True}
    
    def test_safe_json_loads_default_none(self):
        """Test de carga con default None"""
        json_str = 'invalid json'
        result = safe_json_loads(json_str)
        
        assert result is None
    
    def test_safe_json_dumps_valid(self):
        """Test de serialización válida"""
        data = {"key": "value", "number": 123}
        result = safe_json_dumps(data)
        
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert parsed == data
    
    def test_safe_json_dumps_with_datetime(self):
        """Test de serialización con datetime"""
        from datetime import datetime
        data = {"timestamp": datetime.now()}
        result = safe_json_dumps(data)
        
        assert isinstance(result, str)
        assert "timestamp" in result


class TestFormatFunctionsRefactored(BaseServiceTestCase, StandardTestMixin):
    """Tests refactorizados para funciones de formato"""
    
    @pytest.mark.parametrize("seconds,expected", [
        (45.0, "0:45"),
        (125.0, "2:05"),
        (0.0, "0:00"),
        (3661.0, "61:01")
    ])
    def test_format_duration(self, seconds, expected):
        """Test de formato de duración"""
        result = format_duration(seconds)
        assert result == expected
    
    @pytest.mark.parametrize("bytes_size,expected_unit", [
        (512, "B"),
        (2048, "KB"),
        (1048576, "MB"),
        (1073741824, "GB")
    ])
    def test_format_file_size(self, bytes_size, expected_unit):
        """Test de formato de tamaño de archivo"""
        result = format_file_size(bytes_size)
        assert expected_unit in result
        assert isinstance(result, str)


class TestDirectoryOperationsRefactored(BaseServiceTestCase, StandardTestMixin):
    """Tests refactorizados para operaciones de directorio"""
    
    def test_ensure_directory_new(self):
        """Test de creación de directorio nuevo"""
        with tempfile.TemporaryDirectory() as temp_dir:
            new_dir = os.path.join(temp_dir, "new_subdir")
            result = ensure_directory(new_dir)
            
            assert isinstance(result, Path)
            assert os.path.exists(new_dir)
    
    def test_ensure_directory_exists(self):
        """Test cuando el directorio ya existe"""
        with tempfile.TemporaryDirectory() as temp_dir:
            result1 = ensure_directory(temp_dir)
            result2 = ensure_directory(temp_dir)
            
            assert isinstance(result1, Path)
            assert isinstance(result2, Path)


class TestListOperationsRefactored(BaseServiceTestCase, StandardTestMixin):
    """Tests refactorizados para operaciones de listas"""
    
    @pytest.mark.parametrize("items,chunk_size,expected_chunks", [
        (list(range(10)), 5, 2),
        (list(range(10)), 3, 4),
        ([], 5, 0),
        (list(range(5)), 10, 1)
    ])
    def test_chunk_list(self, items, chunk_size, expected_chunks):
        """Test de chunking de lista"""
        chunks = chunk_list(items, chunk_size)
        
        assert len(chunks) == expected_chunks
        if items:
            assert sum(len(chunk) for chunk in chunks) == len(items)


class TestDictOperationsRefactored(BaseServiceTestCase, StandardTestMixin):
    """Tests refactorizados para operaciones de diccionarios"""
    
    def test_merge_dicts_two(self):
        """Test de fusión de dos diccionarios"""
        dict1 = {"a": 1, "b": 2}
        dict2 = {"c": 3, "d": 4}
        
        result = merge_dicts(dict1, dict2)
        
        assert result == {"a": 1, "b": 2, "c": 3, "d": 4}
    
    def test_merge_dicts_overwrite(self):
        """Test de fusión con sobrescritura"""
        dict1 = {"a": 1, "b": 2}
        dict2 = {"b": 3, "c": 4}
        
        result = merge_dicts(dict1, dict2)
        
        assert result["b"] == 3  # Último valor gana
    
    @pytest.mark.parametrize("path,expected", [
        ("key", "value"),
        ("user.profile.name", "John"),
        ("nonexistent", None)
    ])
    def test_get_nested_value(self, path, expected):
        """Test de obtención de valor anidado"""
        data = {
            "key": "value",
            "user": {
                "profile": {
                    "name": "John"
                }
            }
        }
        result = get_nested_value(data, path, default=None)
        
        if expected is not None:
            assert result == expected
    
    def test_set_nested_value(self):
        """Test de establecimiento de valor anidado"""
        data = {}
        set_nested_value(data, "user.profile.name", "John")
        
        assert data["user"]["profile"]["name"] == "John"


class TestSanitizeFilenameRefactored(BaseServiceTestCase, StandardTestMixin):
    """Tests refactorizados para sanitize_filename"""
    
    def test_sanitize_filename_normal(self):
        """Test de sanitización de nombre normal"""
        result = sanitize_filename("normal_file.txt")
        assert result == "normal_file.txt"
    
    def test_sanitize_filename_special_chars(self):
        """Test de sanitización con caracteres especiales"""
        result = sanitize_filename('file<>:"/\\|?*.txt')
        
        dangerous_chars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*']
        for char in dangerous_chars:
            assert char not in result
    
    def test_sanitize_filename_long(self):
        """Test de sanitización de nombre largo"""
        long_name = "a" * 300 + ".txt"
        result = sanitize_filename(long_name)
        
        assert len(result) <= 255


class TestRetryOnFailureRefactored(BaseServiceTestCase, StandardTestMixin):
    """Tests refactorizados para retry_on_failure decorator"""
    
    @pytest.mark.asyncio
    async def test_retry_on_failure_success(self):
        """Test de retry con éxito inmediato"""
        call_count = 0
        
        @retry_on_failure(max_retries=3)
        async def test_func():
            nonlocal call_count
            call_count += 1
            return "success"
        
        result = await test_func()
        
        assert result == "success"
        assert call_count == 1
    
    @pytest.mark.asyncio
    async def test_retry_on_failure_retries(self):
        """Test de retry con reintentos"""
        call_count = 0
        
        @retry_on_failure(max_retries=3, delay=0.01)
        async def test_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Error")
            return "success"
        
        result = await test_func()
        
        assert result == "success"
        assert call_count == 3
    
    @pytest.mark.asyncio
    async def test_retry_on_failure_max_retries(self):
        """Test de retry con máximo de reintentos alcanzado"""
        call_count = 0
        
        @retry_on_failure(max_retries=3, delay=0.01)
        async def test_func():
            nonlocal call_count
            call_count += 1
            raise ValueError("Error")
        
        with pytest.raises(ValueError):
            await test_func()
        
        assert call_count == 3



