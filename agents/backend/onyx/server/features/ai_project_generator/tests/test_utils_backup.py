"""
Tests for BackupManager utility
"""

import pytest
import asyncio
import json
import tarfile
from pathlib import Path

from ..utils.backup_manager import BackupManager


class TestBackupManager:
    """Test suite for BackupManager"""

    def test_init(self, temp_dir):
        """Test BackupManager initialization"""
        backup_dir = temp_dir / "backups"
        manager = BackupManager(backup_dir=backup_dir)
        assert manager.backup_dir == backup_dir
        assert backup_dir.exists()

    def test_init_default_dir(self):
        """Test BackupManager with default directory"""
        manager = BackupManager()
        assert manager.backup_dir.exists()

    @pytest.mark.asyncio
    async def test_create_backup(self, temp_dir):
        """Test creating a backup"""
        manager = BackupManager(backup_dir=temp_dir / "backups")
        
        # Create test projects directory
        projects_dir = temp_dir / "projects"
        projects_dir.mkdir()
        (projects_dir / "project1").mkdir()
        (projects_dir / "project1" / "README.md").write_text("# Project 1")
        
        backup_info = await manager.create_backup(projects_dir)
        
        assert backup_info["backup_name"].startswith("backup_")
        assert "backup_path" in backup_info
        assert backup_info["size_bytes"] > 0
        assert backup_info["includes"]["projects"] is True
        
        # Verify backup file exists
        backup_path = Path(backup_info["backup_path"])
        assert backup_path.exists()

    @pytest.mark.asyncio
    async def test_create_backup_with_cache(self, temp_dir):
        """Test creating backup with cache"""
        manager = BackupManager(backup_dir=temp_dir / "backups")
        
        projects_dir = temp_dir / "projects"
        projects_dir.mkdir()
        
        # Create cache directory
        cache_dir = temp_dir / "cache"
        cache_dir.mkdir()
        (cache_dir / "cache1.json").write_text('{"test": "data"}')
        
        backup_info = await manager.create_backup(
            projects_dir,
            include_cache=True
        )
        
        assert backup_info["includes"]["cache"] is True

    @pytest.mark.asyncio
    async def test_create_backup_with_queue(self, temp_dir):
        """Test creating backup with queue"""
        manager = BackupManager(backup_dir=temp_dir / "backups")
        
        projects_dir = temp_dir / "projects"
        projects_dir.mkdir()
        
        # Create queue file
        queue_file = temp_dir / "project_queue.json"
        queue_file.write_text('{"queue": []}')
        
        backup_info = await manager.create_backup(
            projects_dir,
            include_queue=True
        )
        
        assert backup_info["includes"]["queue"] is True

    @pytest.mark.asyncio
    async def test_restore_backup(self, temp_dir):
        """Test restoring a backup"""
        manager = BackupManager(backup_dir=temp_dir / "backups")
        
        # Create and backup projects
        projects_dir = temp_dir / "projects"
        projects_dir.mkdir()
        (projects_dir / "test_project").mkdir()
        (projects_dir / "test_project" / "file.txt").write_text("test content")
        
        backup_info = await manager.create_backup(projects_dir)
        
        # Restore to new location
        restore_dir = temp_dir / "restored"
        restore_info = await manager.restore_backup(
            backup_path=Path(backup_info["backup_path"]),
            target_dir=restore_dir
        )
        
        assert restore_info["success"] is True
        assert (restore_dir / "projects" / "test_project" / "file.txt").exists()

    @pytest.mark.asyncio
    async def test_list_backups(self, temp_dir):
        """Test listing backups"""
        manager = BackupManager(backup_dir=temp_dir / "backups")
        
        projects_dir = temp_dir / "projects"
        projects_dir.mkdir()
        
        # Create multiple backups
        await manager.create_backup(projects_dir)
        await manager.create_backup(projects_dir)
        
        backups = await manager.list_backups()
        
        assert len(backups) >= 2

    @pytest.mark.asyncio
    async def test_delete_backup(self, temp_dir):
        """Test deleting a backup"""
        manager = BackupManager(backup_dir=temp_dir / "backups")
        
        projects_dir = temp_dir / "projects"
        projects_dir.mkdir()
        
        backup_info = await manager.create_backup(projects_dir)
        backup_path = Path(backup_info["backup_path"])
        assert backup_path.exists()
        
        result = await manager.delete_backup(backup_info["backup_name"])
        
        assert result is True
        assert not backup_path.exists()

    @pytest.mark.asyncio
    async def test_backup_metadata(self, temp_dir):
        """Test backup metadata is saved"""
        manager = BackupManager(backup_dir=temp_dir / "backups")
        
        projects_dir = temp_dir / "projects"
        projects_dir.mkdir()
        
        backup_info = await manager.create_backup(projects_dir)
        
        # Check metadata file exists
        metadata_path = manager.backup_dir / f"{backup_info['backup_name']}_info.json"
        assert metadata_path.exists()
        
        metadata = json.loads(metadata_path.read_text())
        assert metadata["backup_name"] == backup_info["backup_name"]

