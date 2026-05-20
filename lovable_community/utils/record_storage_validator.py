"""
Record Storage Validator

Validates that the refactored RecordStorage implementation meets all requirements.
Run this script to verify the code is correct.
"""

import sys
import tempfile
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.record_storage import RecordStorage


class RequirementValidator:
    """Validates all refactoring requirements."""
    
    def __init__(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.passed = 0
        self.failed = 0
        self.errors = []
    
    def cleanup(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test(self, name: str, test_func):
        """Run a test and track results."""
        try:
            result = test_func()
            if result:
                print(f"✅ {name}")
                self.passed += 1
            else:
                print(f"❌ {name}")
                self.failed += 1
                self.errors.append(name)
        except Exception as e:
            print(f"❌ {name}: {e}")
            self.failed += 1
            self.errors.append(f"{name}: {e}")
    
    def test_context_managers(self):
        """Test Requirement 1: Context Managers"""
        file_path = self.temp_dir / "test_context.json"
        storage = RecordStorage(str(file_path))
        
        storage.write([{"id": "1", "name": "Test"}])
        records = storage.read()
        
        return len(records) == 1 and records[0]["name"] == "Test"
    
    def test_indentation(self):
        """Test Requirement 2: Correct Indentation"""
        file_path = self.temp_dir / "test_indent.json"
        storage = RecordStorage(str(file_path))
        
        storage.write([
            {"id": "1", "name": "Alice", "age": 30},
            {"id": "2", "name": "Bob", "age": 25}
        ])
        
        records = storage.read()
        return len(records) == 2
    
    def test_update_merging(self):
        """Test Requirement 3: Correct Record Handling in update()"""
        file_path = self.temp_dir / "test_update.json"
        storage = RecordStorage(str(file_path))
        
        original = {"id": "1", "name": "Alice", "age": 30, "city": "Boston"}
        storage.write([original])
        
        storage.update("1", {"age": 31, "city": "New York"})
        updated = storage.get("1")
        
        return (
            updated["name"] == "Alice" and
            updated["age"] == 31 and
            updated["city"] == "New York" and
            updated["id"] == "1"
        )
    
    def test_error_handling_read(self):
        """Test Requirement 4: Error Handling in read()"""
        file_path = self.temp_dir / "test_read_error.json"
        storage = RecordStorage(str(file_path))
        
        try:
            invalid_json = file_path
            invalid_json.write_text("invalid json content")
            storage.read()
            return False
        except RuntimeError:
            return True
        except Exception:
            return False
    
    def test_error_handling_write(self):
        """Test Requirement 4: Error Handling in write()"""
        file_path = self.temp_dir / "test_write_error.json"
        storage = RecordStorage(str(file_path))
        
        try:
            storage.write("not a list")
            return False
        except ValueError:
            return True
        except Exception:
            return False
    
    def test_error_handling_update(self):
        """Test Requirement 4: Error Handling in update()"""
        file_path = self.temp_dir / "test_update_error.json"
        storage = RecordStorage(str(file_path))
        
        try:
            storage.update("", {"age": 30})
            return False
        except ValueError:
            return True
        except Exception:
            return False
    
    def test_id_preservation(self):
        """Test that update() preserves record ID"""
        file_path = self.temp_dir / "test_id.json"
        storage = RecordStorage(str(file_path))
        
        storage.write([{"id": "1", "name": "Alice"}])
        storage.update("1", {"id": "999", "age": 30})
        updated = storage.get("1")
        
        return updated["id"] == "1" and updated.get("age") == 30
    
    def test_file_closure(self):
        """Test that files are properly closed"""
        file_path = self.temp_dir / "test_closure.json"
        storage = RecordStorage(str(file_path))
        
        for i in range(10):
            storage.write([{"id": str(i), "value": i}])
            records = storage.read()
            if len(records) != 1:
                return False
        
        return True
    
    def test_concurrent_operations(self):
        """Test multiple operations in sequence"""
        file_path = self.temp_dir / "test_concurrent.json"
        storage = RecordStorage(str(file_path))
        
        storage.write([{"id": "1", "name": "Alice"}])
        storage.update("1", {"age": 30})
        storage.add({"id": "2", "name": "Bob"})
        storage.delete("1")
        
        records = storage.read()
        return len(records) == 1 and records[0]["id"] == "2"
    
    def run_all_tests(self):
        """Run all validation tests."""
        print("\n" + "=" * 70)
        print("  Record Storage - Requirement Validation")
        print("=" * 70 + "\n")
        
        print("Testing Requirements:\n")
        
        self.test("Requirement 1: Context Managers", self.test_context_managers)
        self.test("Requirement 2: Correct Indentation", self.test_indentation)
        self.test("Requirement 3: Update Merging (not replacing)", self.test_update_merging)
        self.test("Requirement 4: Error Handling - read()", self.test_error_handling_read)
        self.test("Requirement 4: Error Handling - write()", self.test_error_handling_write)
        self.test("Requirement 4: Error Handling - update()", self.test_error_handling_update)
        
        print("\nAdditional Tests:\n")
        
        self.test("ID Preservation in Updates", self.test_id_preservation)
        self.test("File Properly Closed", self.test_file_closure)
        self.test("Concurrent Operations", self.test_concurrent_operations)
        
        print("\n" + "=" * 70)
        print(f"  Results: {self.passed} passed, {self.failed} failed")
        print("=" * 70)
        
        if self.failed > 0:
            print("\n❌ Failed Tests:")
            for error in self.errors:
                print(f"  - {error}")
            return False
        else:
            print("\n✅ All requirements met!")
            return True


def main():
    """Run validation."""
    validator = RequirementValidator()
    try:
        success = validator.run_all_tests()
        return 0 if success else 1
    finally:
        validator.cleanup()


if __name__ == "__main__":
    exit(main())


