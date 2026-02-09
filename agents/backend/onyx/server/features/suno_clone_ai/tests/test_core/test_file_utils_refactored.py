"""
Tests refactorizados para utilidades de archivos
Usando clases base y helpers
"""

import pytest
import json
import pickle
import yaml
import tempfile
import os
from pathlib import Path

from core.utils.file_utils import (
    FileManager,
    ensure_dir,
    save_json,
    load_json
)
from test_helpers import BaseServiceTestCase, StandardTestMixin


class TestFileManagerRefactored(BaseServiceTestCase, StandardTestMixin):
    """Tests refactorizados para FileManager"""
    
    @pytest.fixture
    def temp_dir(self):
        """Directorio temporal"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    def test_ensure_dir_new(self, temp_dir):
        """Test de creación de directorio nuevo"""
        new_dir = os.path.join(temp_dir, "new_subdir")
        result = FileManager.ensure_dir(new_dir)
        
        assert isinstance(result, Path)
        assert os.path.exists(new_dir)
        assert os.path.isdir(new_dir)
    
    def test_ensure_dir_exists(self, temp_dir):
        """Test cuando el directorio ya existe"""
        result1 = FileManager.ensure_dir(temp_dir)
        result2 = FileManager.ensure_dir(temp_dir)
        
        assert isinstance(result1, Path)
        assert isinstance(result2, Path)
        assert os.path.exists(temp_dir)
    
    def test_ensure_dir_nested(self, temp_dir):
        """Test de creación de directorio anidado"""
        nested_dir = os.path.join(temp_dir, "level1", "level2", "level3")
        result = FileManager.ensure_dir(nested_dir)
        
        assert os.path.exists(nested_dir)
        assert isinstance(result, Path)
    
    @pytest.mark.parametrize("data", [
        {"key": "value", "number": 123},
        {"list": [1, 2, 3], "nested": {"a": 1}},
        {"empty": None}
    ])
    def test_save_load_json(self, temp_dir, data):
        """Test de guardar y cargar JSON"""
        file_path = os.path.join(temp_dir, "test.json")
        
        FileManager.save_json(data, file_path)
        assert os.path.exists(file_path)
        
        loaded = FileManager.load_json(file_path)
        assert loaded == data
    
    def test_save_json_custom_indent(self, temp_dir):
        """Test de guardar JSON con indent personalizado"""
        file_path = os.path.join(temp_dir, "test.json")
        data = {"key": "value"}
        
        FileManager.save_json(data, file_path, indent=4)
        
        assert os.path.exists(file_path)
        with open(file_path, 'r') as f:
            content = f.read()
            assert "    " in content
    
    def test_save_json_creates_dir(self, temp_dir):
        """Test de que crea el directorio si no existe"""
        file_path = os.path.join(temp_dir, "subdir", "test.json")
        data = {"key": "value"}
        
        FileManager.save_json(data, file_path)
        
        assert os.path.exists(file_path)
    
    @pytest.mark.parametrize("data", [
        {"key": "value", "number": 123},
        {"nested": {"a": 1, "b": 2}}
    ])
    def test_save_load_yaml(self, temp_dir, data):
        """Test de guardar y cargar YAML"""
        file_path = os.path.join(temp_dir, "test.yaml")
        
        FileManager.save_yaml(data, file_path)
        assert os.path.exists(file_path)
        
        loaded = FileManager.load_yaml(file_path)
        assert loaded == data
    
    @pytest.mark.parametrize("data", [
        {"key": "value", "number": 123},
        {"list": [1, 2, 3]}
    ])
    def test_save_load_pickle(self, temp_dir, data):
        """Test de guardar y cargar pickle"""
        file_path = os.path.join(temp_dir, "test.pkl")
        
        FileManager.save_pickle(data, file_path)
        assert os.path.exists(file_path)
        
        loaded = FileManager.load_pickle(file_path)
        assert loaded == data
    
    def test_save_pickle_complex_object(self, temp_dir):
        """Test de guardar objeto complejo en pickle"""
        file_path = os.path.join(temp_dir, "test.pkl")
        
        class CustomObject:
            def __init__(self, value):
                self.value = value
        
        obj = CustomObject("test")
        FileManager.save_pickle(obj, file_path)
        
        loaded = FileManager.load_pickle(file_path)
        assert isinstance(loaded, CustomObject)
        assert loaded.value == "test"


class TestHelperFunctionsRefactored(BaseServiceTestCase, StandardTestMixin):
    """Tests refactorizados para funciones helper"""
    
    @pytest.fixture
    def temp_dir(self):
        """Directorio temporal"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    def test_ensure_dir_function(self, temp_dir):
        """Test de función ensure_dir"""
        new_dir = os.path.join(temp_dir, "helper_dir")
        result = ensure_dir(new_dir)
        
        assert isinstance(result, Path)
        assert os.path.exists(new_dir)
    
    def test_save_load_json_functions(self, temp_dir):
        """Test de funciones save_json y load_json"""
        file_path = os.path.join(temp_dir, "test.json")
        data = {"key": "value", "number": 123}
        
        save_json(data, file_path)
        assert os.path.exists(file_path)
        
        loaded = load_json(file_path)
        assert loaded == data



