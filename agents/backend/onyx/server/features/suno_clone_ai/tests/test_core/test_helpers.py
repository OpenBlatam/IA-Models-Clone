"""
Tests para funciones helper del core
"""

import pytest
import json
import uuid
from pathlib import Path
import tempfile
import os

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
import asyncio


@pytest.mark.unit
@pytest.mark.core
class TestGenerateID:
    """Tests para generate_id"""
    
    def test_generate_id_no_prefix(self):
        """Test de generación sin prefijo"""
        id_str = generate_id()
        
        assert isinstance(id_str, str)
        assert len(id_str) > 0
        # Verificar que es un UUID válido
        uuid.UUID(id_str)
    
    def test_generate_id_with_prefix(self):
        """Test de generación con prefijo"""
        id_str = generate_id(prefix="song")
        
        assert isinstance(id_str, str)
        assert id_str.startswith("song_")
        # Verificar que el resto es un UUID válido
        uuid_part = id_str.split("_", 1)[1]
        uuid.UUID(uuid_part)
    
    def test_generate_id_unique(self):
        """Test de que genera IDs únicos"""
        ids = [generate_id() for _ in range(100)]
        
        assert len(ids) == len(set(ids))  # Todos únicos


@pytest.mark.unit
@pytest.mark.core
class TestHashString:
    """Tests para hash_string"""
    
    def test_hash_string_sha256(self):
        """Test de hash con SHA256"""
        result = hash_string("test", algorithm="sha256")
        
        assert isinstance(result, str)
        assert len(result) == 64  # SHA256 produce 64 caracteres hex
    
    def test_hash_string_md5(self):
        """Test de hash con MD5"""
        result = hash_string("test", algorithm="md5")
        
        assert isinstance(result, str)
        assert len(result) == 32  # MD5 produce 32 caracteres hex
    
    def test_hash_string_sha1(self):
        """Test de hash con SHA1"""
        result = hash_string("test", algorithm="sha1")
        
        assert isinstance(result, str)
        assert len(result) == 40  # SHA1 produce 40 caracteres hex
    
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


@pytest.mark.unit
@pytest.mark.core
class TestSafeJSON:
    """Tests para safe_json_loads y safe_json_dumps"""
    
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
    
    def test_safe_json_dumps_invalid(self):
        """Test de serialización inválida"""
        # Objeto que no se puede serializar
        class Unserializable:
            pass
        
        data = {"obj": Unserializable()}
        result = safe_json_dumps(data, default="{}")
        
        assert result == "{}"


@pytest.mark.unit
@pytest.mark.core
class TestFormatDuration:
    """Tests para format_duration"""
    
    def test_format_duration_seconds(self):
        """Test de formato de segundos"""
        result = format_duration(45.0)
        
        assert result == "0:45"
    
    def test_format_duration_minutes(self):
        """Test de formato de minutos"""
        result = format_duration(125.0)
        
        assert result == "2:05"
    
    def test_format_duration_zero(self):
        """Test de formato de cero"""
        result = format_duration(0.0)
        
        assert result == "0:00"
    
    def test_format_duration_large(self):
        """Test de formato de duración grande"""
        result = format_duration(3661.0)
        
        assert result == "61:01"


@pytest.mark.unit
@pytest.mark.core
class TestFormatFileSize:
    """Tests para format_file_size"""
    
    def test_format_file_size_bytes(self):
        """Test de formato en bytes"""
        result = format_file_size(512)
        
        assert "B" in result
        assert "512" in result
    
    def test_format_file_size_kb(self):
        """Test de formato en KB"""
        result = format_file_size(2048)
        
        assert "KB" in result
    
    def test_format_file_size_mb(self):
        """Test de formato en MB"""
        result = format_file_size(1048576)
        
        assert "MB" in result
    
    def test_format_file_size_gb(self):
        """Test de formato en GB"""
        result = format_file_size(1073741824)
        
        assert "GB" in result


@pytest.mark.unit
@pytest.mark.core
class TestEnsureDirectory:
    """Tests para ensure_directory"""
    
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
            assert os.path.exists(temp_dir)


@pytest.mark.unit
@pytest.mark.core
class TestChunkList:
    """Tests para chunk_list"""
    
    def test_chunk_list_exact(self):
        """Test de chunking exacto"""
        items = list(range(10))
        chunks = chunk_list(items, chunk_size=5)
        
        assert len(chunks) == 2
        assert chunks[0] == [0, 1, 2, 3, 4]
        assert chunks[1] == [5, 6, 7, 8, 9]
    
    def test_chunk_list_remainder(self):
        """Test de chunking con resto"""
        items = list(range(10))
        chunks = chunk_list(items, chunk_size=3)
        
        assert len(chunks) == 4
        assert len(chunks[-1]) == 1
    
    def test_chunk_list_empty(self):
        """Test de chunking de lista vacía"""
        chunks = chunk_list([], chunk_size=5)
        
        assert chunks == []
    
    def test_chunk_list_single_chunk(self):
        """Test de chunking en un solo chunk"""
        items = list(range(5))
        chunks = chunk_list(items, chunk_size=10)
        
        assert len(chunks) == 1
        assert chunks[0] == items


@pytest.mark.unit
@pytest.mark.core
class TestMergeDicts:
    """Tests para merge_dicts"""
    
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
        assert result["c"] == 4
    
    def test_merge_dicts_multiple(self):
        """Test de fusión de múltiples diccionarios"""
        dict1 = {"a": 1}
        dict2 = {"b": 2}
        dict3 = {"c": 3}
        
        result = merge_dicts(dict1, dict2, dict3)
        
        assert result == {"a": 1, "b": 2, "c": 3}


@pytest.mark.unit
@pytest.mark.core
class TestNestedValue:
    """Tests para get_nested_value y set_nested_value"""
    
    def test_get_nested_value_simple(self):
        """Test de obtención de valor simple"""
        data = {"key": "value"}
        result = get_nested_value(data, "key")
        
        assert result == "value"
    
    def test_get_nested_value_nested(self):
        """Test de obtención de valor anidado"""
        data = {"user": {"profile": {"name": "John"}}}
        result = get_nested_value(data, "user.profile.name")
        
        assert result == "John"
    
    def test_get_nested_value_default(self):
        """Test de obtención con default"""
        data = {"key": "value"}
        result = get_nested_value(data, "nonexistent", default="default")
        
        assert result == "default"
    
    def test_set_nested_value_simple(self):
        """Test de establecimiento de valor simple"""
        data = {}
        set_nested_value(data, "key", "value")
        
        assert data["key"] == "value"
    
    def test_set_nested_value_nested(self):
        """Test de establecimiento de valor anidado"""
        data = {}
        set_nested_value(data, "user.profile.name", "John")
        
        assert data["user"]["profile"]["name"] == "John"
    
    def test_set_nested_value_existing(self):
        """Test de establecimiento en estructura existente"""
        data = {"user": {"profile": {}}}
        set_nested_value(data, "user.profile.name", "John")
        
        assert data["user"]["profile"]["name"] == "John"


@pytest.mark.unit
@pytest.mark.core
class TestSanitizeFilename:
    """Tests para sanitize_filename"""
    
    def test_sanitize_filename_normal(self):
        """Test de sanitización de nombre normal"""
        result = sanitize_filename("normal_file.txt")
        
        assert result == "normal_file.txt"
    
    def test_sanitize_filename_special_chars(self):
        """Test de sanitización con caracteres especiales"""
        result = sanitize_filename('file<>:"/\\|?*.txt')
        
        assert "<" not in result
        assert ">" not in result
        assert ":" not in result
        assert "/" not in result
        assert "\\" not in result
    
    def test_sanitize_filename_long(self):
        """Test de sanitización de nombre largo"""
        long_name = "a" * 300 + ".txt"
        result = sanitize_filename(long_name)
        
        assert len(result) <= 255


@pytest.mark.unit
@pytest.mark.core
class TestRetryOnFailure:
    """Tests para retry_on_failure decorator"""
    
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
