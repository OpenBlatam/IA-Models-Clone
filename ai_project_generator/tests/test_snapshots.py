"""
Snapshot testing for regression detection
"""

import pytest
from pathlib import Path
from typing import Dict, Any
import json
import hashlib


class SnapshotManager:
    """Manager for snapshot testing"""
    
    def __init__(self, snapshots_dir: Path):
        self.snapshots_dir = Path(snapshots_dir)
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
    
    def save_snapshot(self, name: str, data: Any) -> Path:
        """Save a snapshot"""
        snapshot_file = self.snapshots_dir / f"{name}.snapshot"
        
        if isinstance(data, dict):
            content = json.dumps(data, indent=2, default=str, ensure_ascii=False)
        else:
            content = str(data)
        
        snapshot_file.write_text(content, encoding="utf-8")
        return snapshot_file
    
    def load_snapshot(self, name: str) -> str:
        """Load a snapshot"""
        snapshot_file = self.snapshots_dir / f"{name}.snapshot"
        if snapshot_file.exists():
            return snapshot_file.read_text(encoding="utf-8")
        return None
    
    def compare_snapshot(self, name: str, current_data: Any) -> Dict[str, Any]:
        """Compare current data with snapshot"""
        snapshot_content = self.load_snapshot(name)
        
        if isinstance(current_data, dict):
            current_content = json.dumps(current_data, indent=2, default=str, ensure_ascii=False)
        else:
            current_content = str(current_data)
        
        if snapshot_content is None:
            return {
                "match": False,
                "error": "Snapshot not found",
                "current": current_content
            }
        
        current_hash = hashlib.md5(current_content.encode()).hexdigest()
        snapshot_hash = hashlib.md5(snapshot_content.encode()).hexdigest()
        
        return {
            "match": current_hash == snapshot_hash,
            "current_hash": current_hash,
            "snapshot_hash": snapshot_hash,
            "identical": current_content == snapshot_content
        }


@pytest.fixture
def snapshot_manager(temp_dir):
    """Fixture for snapshot manager"""
    return SnapshotManager(temp_dir / "snapshots")


class TestSnapshots:
    """Tests for snapshot testing"""
    
    def test_save_and_load_snapshot(self, snapshot_manager):
        """Test saving and loading snapshots"""
        data = {
            "project_id": "test-123",
            "name": "test_project",
            "status": "completed"
        }
        
        snapshot_manager.save_snapshot("test_project", data)
        loaded = snapshot_manager.load_snapshot("test_project")
        
        assert loaded is not None
        assert "test-123" in loaded
    
    def test_compare_snapshot_match(self, snapshot_manager):
        """Test comparing snapshot that matches"""
        data = {"key": "value"}
        
        snapshot_manager.save_snapshot("test", data)
        comparison = snapshot_manager.compare_snapshot("test", data)
        
        assert comparison["match"] is True
        assert comparison["identical"] is True
    
    def test_compare_snapshot_mismatch(self, snapshot_manager):
        """Test comparing snapshot that doesn't match"""
        original = {"key": "value1"}
        modified = {"key": "value2"}
        
        snapshot_manager.save_snapshot("test", original)
        comparison = snapshot_manager.compare_snapshot("test", modified)
        
        assert comparison["match"] is False
        assert comparison["identical"] is False

