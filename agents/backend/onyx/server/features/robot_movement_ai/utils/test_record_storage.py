"""
Tests unitarios para RecordStorage
===================================

Tests completos para verificar que todas las mejoras de refactorización
funcionan correctamente.
"""

import unittest
import tempfile
import os
import json
from pathlib import Path
from record_storage import RecordStorage


class TestRecordStorage(unittest.TestCase):
    """Tests para la clase RecordStorage refactorizada."""
    
    def setUp(self):
        """Configurar test fixture antes de cada test."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.temp_dir, "test_records.json")
        self.storage = RecordStorage(self.test_file)
    
    def tearDown(self):
        """Limpiar después de cada test."""
        if os.path.exists(self.test_file):
            os.remove(self.test_file)
        os.rmdir(self.temp_dir)
    
    def test_initialization_creates_file(self):
        """Test que la inicialización crea el archivo correctamente."""
        self.assertTrue(os.path.exists(self.test_file))
        
        with open(self.test_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.assertIn('records', data)
        self.assertEqual(data['records'], [])
    
    def test_write_valid_records(self):
        """Test escribir registros válidos."""
        records = [
            {"id": "1", "name": "Test 1", "value": 100},
            {"id": "2", "name": "Test 2", "value": 200}
        ]
        
        result = self.storage.write(records)
        self.assertTrue(result)
        
        written = self.storage.read()
        self.assertEqual(len(written), 2)
        self.assertEqual(written[0]["id"], "1")
    
    def test_write_invalid_type_raises_error(self):
        """Test que escribir tipo inválido lanza TypeError."""
        with self.assertRaises(TypeError):
            self.storage.write("not a list")
        
        with self.assertRaises(ValueError):
            self.storage.write([{"id": "1"}, "not a dict"])
    
    def test_read_empty_file(self):
        """Test leer archivo vacío."""
        records = self.storage.read()
        self.assertEqual(records, [])
    
    def test_read_nonexistent_file(self):
        """Test leer archivo que no existe."""
        new_storage = RecordStorage(os.path.join(self.temp_dir, "nonexistent.json"))
        records = new_storage.read()
        self.assertEqual(records, [])
    
    def test_read_corrupted_json_raises_error(self):
        """Test que leer JSON corrupto lanza error."""
        with open(self.test_file, 'w', encoding='utf-8') as f:
            f.write("invalid json content")
        
        with self.assertRaises(RuntimeError):
            self.storage.read()
    
    def test_update_existing_record(self):
        """Test actualizar un registro existente."""
        records = [
            {"id": "1", "name": "Original", "age": 25},
            {"id": "2", "name": "Other", "age": 30}
        ]
        self.storage.write(records)
        
        result = self.storage.update("1", {"age": 26, "city": "Madrid"})
        self.assertTrue(result)
        
        updated = self.storage.get("1")
        self.assertEqual(updated["age"], 26)
        self.assertEqual(updated["city"], "Madrid")
        self.assertEqual(updated["name"], "Original")
        self.assertEqual(updated["id"], "1")
    
    def test_update_preserves_id(self):
        """Test que update preserva el ID del registro."""
        records = [{"id": "1", "name": "Test"}]
        self.storage.write(records)
        
        self.storage.update("1", {"id": "should_not_change", "name": "Updated"})
        
        updated = self.storage.get("1")
        self.assertEqual(updated["id"], "1")
        self.assertEqual(updated["name"], "Updated")
    
    def test_update_nonexistent_record(self):
        """Test actualizar registro que no existe."""
        records = [{"id": "1", "name": "Test"}]
        self.storage.write(records)
        
        result = self.storage.update("999", {"name": "New"})
        self.assertFalse(result)
    
    def test_update_invalid_id_type_raises_error(self):
        """Test que update con ID inválido lanza error."""
        with self.assertRaises(TypeError):
            self.storage.update(123, {"test": "value"})
        
        with self.assertRaises(ValueError):
            self.storage.update("", {"test": "value"})
        
        with self.assertRaises(ValueError):
            self.storage.update("   ", {"test": "value"})
    
    def test_update_invalid_updates_type_raises_error(self):
        """Test que update con updates inválido lanza error."""
        with self.assertRaises(TypeError):
            self.storage.update("1", "not a dict")
        
        result = self.storage.update("1", {})
        self.assertFalse(result)
    
    def test_add_new_record(self):
        """Test agregar nuevo registro."""
        new_record = {"id": "1", "name": "New Record", "value": 100}
        result = self.storage.add(new_record)
        self.assertTrue(result)
        
        record = self.storage.get("1")
        self.assertIsNotNone(record)
        self.assertEqual(record["name"], "New Record")
    
    def test_add_duplicate_id_fails(self):
        """Test que agregar registro con ID duplicado falla."""
        record1 = {"id": "1", "name": "First"}
        record2 = {"id": "1", "name": "Second"}
        
        self.storage.add(record1)
        result = self.storage.add(record2)
        self.assertFalse(result)
    
    def test_add_missing_id_raises_error(self):
        """Test que agregar registro sin ID lanza error."""
        with self.assertRaises(ValueError):
            self.storage.add({"name": "No ID"})
    
    def test_delete_existing_record(self):
        """Test eliminar registro existente."""
        records = [
            {"id": "1", "name": "To Delete"},
            {"id": "2", "name": "Keep"}
        ]
        self.storage.write(records)
        
        result = self.storage.delete("1")
        self.assertTrue(result)
        
        remaining = self.storage.read()
        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining[0]["id"], "2")
    
    def test_delete_nonexistent_record(self):
        """Test eliminar registro que no existe."""
        result = self.storage.delete("999")
        self.assertFalse(result)
    
    def test_get_existing_record(self):
        """Test obtener registro existente."""
        records = [{"id": "1", "name": "Test", "value": 100}]
        self.storage.write(records)
        
        record = self.storage.get("1")
        self.assertIsNotNone(record)
        self.assertEqual(record["name"], "Test")
    
    def test_get_nonexistent_record(self):
        """Test obtener registro que no existe."""
        record = self.storage.get("999")
        self.assertIsNone(record)
    
    def test_context_manager_usage(self):
        """Test que se usan context managers correctamente."""
        records = [{"id": "1", "name": "Test"}]
        self.storage.write(records)
        
        self.assertTrue(os.path.exists(self.test_file))
        
        with open(self.test_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.assertIn('records', data)
    
    def test_complex_update_scenario(self):
        """Test escenario complejo de actualización."""
        initial_records = [
            {"id": "1", "name": "User 1", "age": 25, "city": "Madrid"},
            {"id": "2", "name": "User 2", "age": 30, "city": "Barcelona"},
            {"id": "3", "name": "User 3", "age": 28, "city": "Valencia"}
        ]
        self.storage.write(initial_records)
        
        self.storage.update("2", {"age": 31, "status": "active"})
        
        updated = self.storage.get("2")
        self.assertEqual(updated["age"], 31)
        self.assertEqual(updated["status"], "active")
        self.assertEqual(updated["name"], "User 2")
        self.assertEqual(updated["city"], "Barcelona")
        self.assertEqual(updated["id"], "2")
        
        all_records = self.storage.read()
        self.assertEqual(len(all_records), 3)


class TestRecordStorageEdgeCases(unittest.TestCase):
    """Tests para casos extremos y edge cases."""
    
    def setUp(self):
        """Configurar test fixture."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.temp_dir, "test_edge.json")
        self.storage = RecordStorage(self.test_file)
    
    def tearDown(self):
        """Limpiar después de cada test."""
        if os.path.exists(self.test_file):
            os.remove(self.test_file)
        os.rmdir(self.temp_dir)
    
    def test_empty_string_id(self):
        """Test con ID de cadena vacía."""
        with self.assertRaises(ValueError):
            self.storage.update("", {"test": "value"})
    
    def test_whitespace_only_id(self):
        """Test con ID que solo tiene espacios."""
        with self.assertRaises(ValueError):
            self.storage.update("   ", {"test": "value"})
    
    def test_very_large_record(self):
        """Test con registro muy grande."""
        large_data = {"id": "1", "data": "x" * 10000}
        result = self.storage.add(large_data)
        self.assertTrue(result)
        
        retrieved = self.storage.get("1")
        self.assertEqual(len(retrieved["data"]), 10000)
    
    def test_special_characters_in_id(self):
        """Test con caracteres especiales en ID."""
        record = {"id": "test-123_456", "name": "Special ID"}
        self.storage.add(record)
        
        retrieved = self.storage.get("test-123_456")
        self.assertIsNotNone(retrieved)
    
    def test_unicode_characters(self):
        """Test con caracteres Unicode."""
        record = {"id": "1", "name": "José María", "city": "São Paulo"}
        self.storage.add(record)
        
        retrieved = self.storage.get("1")
        self.assertEqual(retrieved["name"], "José María")
        self.assertEqual(retrieved["city"], "São Paulo")
    
    def test_nested_structures(self):
        """Test con estructuras anidadas."""
        record = {
            "id": "1",
            "name": "Test",
            "metadata": {
                "tags": ["tag1", "tag2"],
                "settings": {"option1": True, "option2": False}
            }
        }
        self.storage.add(record)
        
        retrieved = self.storage.get("1")
        self.assertIn("metadata", retrieved)
        self.assertEqual(len(retrieved["metadata"]["tags"]), 2)


if __name__ == '__main__':
    unittest.main()
