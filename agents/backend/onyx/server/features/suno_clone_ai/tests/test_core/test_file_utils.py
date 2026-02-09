"""
Tests para utilidades de archivos
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


@pytest.fixture
def temp_dir():
    """Directorio temporal"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.mark.unit
@pytest.mark.core
class TestFileManager:
    """Tests para FileManager"""
    
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
    
    def test_save_json(self, temp_dir):
        """Test de guardar JSON"""
        file_path = os.path.join(temp_dir, "test.json")
        data = {"key": "value", "number": 123, "list": [1, 2, 3]}
        
        FileManager.save_json(data, file_path)
        
        assert os.path.exists(file_path)
        with open(file_path, 'r') as f:
            loaded = json.load(f)
        assert loaded == data
    
    def test_save_json_custom_indent(self, temp_dir):
        """Test de guardar JSON con indent personalizado"""
        file_path = os.path.join(temp_dir, "test.json")
        data = {"key": "value"}
        
        FileManager.save_json(data, file_path, indent=4)
        
        assert os.path.exists(file_path)
        with open(file_path, 'r') as f:
            content = f.read()
            # Verificar que tiene indentación
            assert "    " in content
    
    def test_save_json_creates_dir(self, temp_dir):
        """Test de que crea el directorio si no existe"""
        file_path = os.path.join(temp_dir, "subdir", "test.json")
        data = {"key": "value"}
        
        FileManager.save_json(data, file_path)
        
        assert os.path.exists(file_path)
    
    def test_load_json(self, temp_dir):
        """Test de cargar JSON"""
        file_path = os.path.join(temp_dir, "test.json")
        data = {"key": "value", "number": 123}
        
        with open(file_path, 'w') as f:
            json.dump(data, f)
        
        loaded = FileManager.load_json(file_path)
        
        assert loaded == data
    
    def test_save_yaml(self, temp_dir):
        """Test de guardar YAML"""
        file_path = os.path.join(temp_dir, "test.yaml")
        data = {"key": "value", "number": 123, "nested": {"a": 1, "b": 2}}
        
        FileManager.save_yaml(data, file_path)
        
        assert os.path.exists(file_path)
        with open(file_path, 'r') as f:
            loaded = yaml.safe_load(f)
        assert loaded == data
    
    def test_load_yaml(self, temp_dir):
        """Test de cargar YAML"""
        file_path = os.path.join(temp_dir, "test.yaml")
        data = {"key": "value", "number": 123}
        
        with open(file_path, 'w') as f:
            yaml.dump(data, f)
        
        loaded = FileManager.load_yaml(file_path)
        
        assert loaded == data
    
    def test_save_pickle(self, temp_dir):
        """Test de guardar pickle"""
        file_path = os.path.join(temp_dir, "test.pkl")
        data = {"key": "value", "number": 123, "list": [1, 2, 3]}
        
        FileManager.save_pickle(data, file_path)
        
        assert os.path.exists(file_path)
        with open(file_path, 'rb') as f:
            loaded = pickle.load(f)
        assert loaded == data
    
    def test_load_pickle(self, temp_dir):
        """Test de cargar pickle"""
        file_path = os.path.join(temp_dir, "test.pkl")
        data = {"key": "value", "number": 123}
        
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
        
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


@pytest.mark.unit
@pytest.mark.core
class TestHelperFunctions:
    """Tests para funciones helper"""
    
    def test_ensure_dir_function(self, temp_dir):
        """Test de función ensure_dir"""
        new_dir = os.path.join(temp_dir, "helper_dir")
        result = ensure_dir(new_dir)
        
        assert isinstance(result, Path)
        assert os.path.exists(new_dir)
    
    def test_save_json_function(self, temp_dir):
        """Test de función save_json"""
        file_path = os.path.join(temp_dir, "test.json")
        data = {"key": "value"}
        
        save_json(data, file_path)
        
        assert os.path.exists(file_path)
    
    def test_load_json_function(self, temp_dir):
        """Test de función load_json"""
        file_path = os.path.join(temp_dir, "test.json")
        data = {"key": "value"}
        
        with open(file_path, 'w') as f:
            json.dump(data, f)
        
        loaded = load_json(file_path)
        
        assert loaded == data



