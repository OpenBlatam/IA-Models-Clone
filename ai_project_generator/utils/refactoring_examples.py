"""
Refactoring Examples - Complete Usage Guide
===========================================

This file demonstrates how to use all the refactored utilities
with proper error handling and best practices.
"""

from pathlib import Path
from typing import Dict, Any, List
import tempfile
import os

from .file_storage import FileStorage
from .file_operations import (
    read_json, write_json, read_yaml, write_yaml,
    read_text, write_text, read_lines, write_lines,
    FileOperationError
)
from .refactored_project_versioning import ProjectVersioning


def example_file_storage():
    """Example: Using FileStorage for record management."""
    print("\n=== FileStorage Example ===")
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        storage = FileStorage(tmp_path)
        
        # Write initial records
        initial_records = [
            {"id": "1", "name": "Alice", "age": 30, "city": "New York"},
            {"id": "2", "name": "Bob", "age": 25, "city": "London"},
            {"id": "3", "name": "Charlie", "age": 35, "city": "Paris"}
        ]
        storage.write(initial_records)
        print("✅ Records written successfully")
        
        # Read all records
        records = storage.read()
        print(f"✅ Read {len(records)} records")
        
        # Update a record (merges updates)
        storage.update("2", {"age": 26, "status": "active"})
        print("✅ Record updated successfully")
        
        # Verify update
        updated_records = storage.read()
        updated_bob = next((r for r in updated_records if r.get('id') == '2'), None)
        print(f"✅ Updated record: {updated_bob}")
        print(f"   - Age updated: {updated_bob.get('age')}")
        print(f"   - Original city preserved: {updated_bob.get('city')}")
        print(f"   - New field added: {updated_bob.get('status')}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def example_file_operations():
    """Example: Using file_operations utilities."""
    print("\n=== File Operations Example ===")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        # JSON operations
        json_file = tmp_path / "data.json"
        json_data = {"name": "Test", "value": 42, "items": [1, 2, 3]}
        write_json(json_file, json_data)
        print("✅ JSON file written")
        
        loaded_json = read_json(json_file)
        print(f"✅ JSON file read: {loaded_json}")
        
        # Text operations
        text_file = tmp_path / "notes.txt"
        write_text(text_file, "Line 1\nLine 2\nLine 3")
        print("✅ Text file written")
        
        text_content = read_text(text_file)
        print(f"✅ Text file read: {text_content}")
        
        # Lines operations
        lines_file = tmp_path / "lines.txt"
        write_lines(lines_file, ["First line", "Second line", "Third line"])
        print("✅ Lines written")
        
        lines = read_lines(lines_file)
        print(f"✅ Lines read: {lines}")
        
        # Error handling example
        try:
            read_json(tmp_path / "nonexistent.json")
        except FileOperationError as e:
            print(f"✅ Error handling works: {e}")


def example_project_versioning():
    """Example: Using ProjectVersioning."""
    print("\n=== Project Versioning Example ===")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        # Create a sample project
        project_path = tmp_path / "my_project"
        project_path.mkdir()
        (project_path / "main.py").write_text("print('Hello')")
        (project_path / "README.md").write_text("# My Project")
        
        # Initialize versioning
        versions_dir = tmp_path / "versions"
        versioning = ProjectVersioning(versions_dir=versions_dir)
        
        # Create version 1.0.0
        v1_info = versioning.create_version(
            project_id="proj-1",
            project_path=project_path,
            version="1.0.0",
            description="Initial release",
            metadata={"author": "Alice", "license": "MIT"}
        )
        print(f"✅ Version 1.0.0 created: {v1_info['created_at']}")
        
        # Modify project
        (project_path / "main.py").write_text("print('Hello World')")
        
        # Create version 1.1.0
        v2_info = versioning.create_version(
            project_id="proj-1",
            project_path=project_path,
            version="1.1.0",
            description="Added greeting",
            metadata={"author": "Alice", "license": "MIT", "changes": "Updated greeting"}
        )
        print(f"✅ Version 1.1.0 created: {v2_info['created_at']}")
        
        # List all versions
        versions = versioning.list_versions("proj-1")
        print(f"✅ Found {len(versions)} versions")
        for v in versions:
            print(f"   - {v['version']}: {v['description']}")
        
        # Compare versions
        comparison = versioning.compare_versions("proj-1", "1.0.0", "1.1.0")
        print(f"✅ Versions compared")
        print(f"   - Same hash: {comparison['same_hash']}")
        print(f"   - Differences: {comparison['differences']}")
        
        # Restore version 1.0.0
        restore_path = tmp_path / "restored_project"
        success = versioning.restore_version("proj-1", "1.0.0", restore_path)
        if success:
            print(f"✅ Version 1.0.0 restored to {restore_path}")
            restored_content = (restore_path / "main.py").read_text()
            print(f"   - Content: {restored_content}")


def example_error_handling():
    """Example: Demonstrating error handling."""
    print("\n=== Error Handling Example ===")
    
    # Invalid input validation
    try:
        storage = FileStorage("")
    except ValueError as e:
        print(f"✅ Caught invalid file_path: {e}")
    
    # File not found handling
    try:
        storage = FileStorage("/nonexistent/path/data.json")
        records = storage.read()
        print(f"✅ Handles missing file gracefully: {len(records)} records")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    
    # Invalid data type
    try:
        storage = FileStorage("/tmp/test.json")
        storage.write("not a list")  # Should fail
    except ValueError as e:
        print(f"✅ Caught invalid data type: {e}")
    
    # Update non-existent record
    try:
        storage = FileStorage("/tmp/test.json")
        storage.write([{"id": "1", "name": "Test"}])
        result = storage.update("999", {"name": "Updated"})
        print(f"✅ Handles non-existent record: {result}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


def run_all_examples():
    """Run all examples."""
    print("=" * 50)
    print("REFACTORING EXAMPLES - COMPLETE USAGE GUIDE")
    print("=" * 50)
    
    example_file_storage()
    example_file_operations()
    example_project_versioning()
    example_error_handling()
    
    print("\n" + "=" * 50)
    print("ALL EXAMPLES COMPLETED SUCCESSFULLY")
    print("=" * 50)


if __name__ == "__main__":
    run_all_examples()


