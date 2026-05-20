"""
Advanced Test Result Backup System
Automated backup with scheduling, versioning, and retention policies
"""

import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import hashlib


class AdvancedTestResultBackup:
    """Advanced backup system for test results"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.results_dir = project_root / "test_results"
        self.backup_dir = project_root / "backups"
        self.backup_dir.mkdir(exist_ok=True)
        self.config_file = project_root / "config" / "backup_config.json"
        self.config = self._load_config()
    
    def _load_config(self) -> Dict:
        """Load backup configuration"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                return self._default_config()
        return self._default_config()
    
    def _default_config(self) -> Dict:
        """Default backup configuration"""
        return {
            'enabled': True,
            'schedule': 'daily',
            'retention_days': 30,
            'compression': True,
            'versioning': True,
            'backup_on_failure': True
        }
    
    def _save_config(self):
        """Save backup configuration"""
        self.config_file.parent.mkdir(exist_ok=True)
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2)
    
    def create_backup(
        self,
        backup_name: str = None,
        include_metadata: bool = True
    ) -> Dict:
        """Create a backup of test results"""
        if backup_name is None:
            backup_name = datetime.now().strftime("backup_%Y%m%d_%H%M%S")
        
        backup_path = self.backup_dir / backup_name
        backup_path.mkdir(exist_ok=True)
        
        # Copy result files
        copied_files = []
        total_size = 0
        
        for result_file in self.results_dir.glob("*.json"):
            try:
                dest_file = backup_path / result_file.name
                shutil.copy2(result_file, dest_file)
                copied_files.append(result_file.name)
                total_size += result_file.stat().st_size
            except Exception as e:
                print(f"Error backing up {result_file}: {e}")
        
        # Create metadata
        metadata = {
            'backup_name': backup_name,
            'timestamp': datetime.now().isoformat(),
            'files_count': len(copied_files),
            'total_size': total_size,
            'files': copied_files
        }
        
        if include_metadata:
            metadata_file = backup_path / "backup_metadata.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
        
        # Calculate checksum
        checksum = self._calculate_backup_checksum(backup_path)
        metadata['checksum'] = checksum
        
        # Save metadata
        if include_metadata:
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
        
        print(f"✅ Backup created: {backup_name} ({len(copied_files)} files, {total_size / 1024 / 1024:.2f} MB)")
        
        return metadata
    
    def _calculate_backup_checksum(self, backup_path: Path) -> str:
        """Calculate checksum for backup"""
        hasher = hashlib.sha256()
        
        for file_path in sorted(backup_path.glob("*.json")):
            with open(file_path, 'rb') as f:
                hasher.update(f.read())
        
        return hasher.hexdigest()
    
    def restore_backup(
        self,
        backup_name: str,
        target_dir: Path = None
    ) -> bool:
        """Restore from backup"""
        backup_path = self.backup_dir / backup_name
        
        if not backup_path.exists():
            print(f"❌ Backup not found: {backup_name}")
            return False
        
        if target_dir is None:
            target_dir = self.results_dir
        
        target_dir.mkdir(exist_ok=True)
        
        restored_count = 0
        
        for backup_file in backup_path.glob("*.json"):
            if backup_file.name == "backup_metadata.json":
                continue
            
            try:
                dest_file = target_dir / backup_file.name
                shutil.copy2(backup_file, dest_file)
                restored_count += 1
            except Exception as e:
                print(f"Error restoring {backup_file}: {e}")
        
        print(f"✅ Restored {restored_count} files from backup: {backup_name}")
        return True
    
    def list_backups(self) -> List[Dict]:
        """List all available backups"""
        backups = []
        
        for backup_dir in self.backup_dir.iterdir():
            if backup_dir.is_dir():
                metadata_file = backup_dir / "backup_metadata.json"
                
                if metadata_file.exists():
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    backups.append(metadata)
                else:
                    # Backup without metadata
                    backups.append({
                        'backup_name': backup_dir.name,
                        'timestamp': datetime.fromtimestamp(backup_dir.stat().st_mtime).isoformat(),
                        'files_count': len(list(backup_dir.glob("*.json")))
                    })
        
        return sorted(backups, key=lambda x: x.get('timestamp', ''), reverse=True)
    
    def cleanup_old_backups(self, retention_days: int = None):
        """Clean up old backups based on retention policy"""
        if retention_days is None:
            retention_days = self.config.get('retention_days', 30)
        
        cutoff = datetime.now() - timedelta(days=retention_days)
        
        removed_count = 0
        
        for backup_dir in self.backup_dir.iterdir():
            if backup_dir.is_dir():
                backup_time = datetime.fromtimestamp(backup_dir.stat().st_mtime)
                
                if backup_time < cutoff:
                    try:
                        shutil.rmtree(backup_dir)
                        removed_count += 1
                        print(f"🗑️ Removed old backup: {backup_dir.name}")
                    except Exception as e:
                        print(f"Error removing {backup_dir}: {e}")
        
        print(f"✅ Cleaned up {removed_count} old backups")
        return removed_count
    
    def verify_backup(self, backup_name: str) -> bool:
        """Verify backup integrity"""
        backup_path = self.backup_dir / backup_name
        
        if not backup_path.exists():
            print(f"❌ Backup not found: {backup_name}")
            return False
        
        metadata_file = backup_path / "backup_metadata.json"
        
        if not metadata_file.exists():
            print(f"⚠️ No metadata found for backup: {backup_name}")
            return False
        
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # Verify checksum
        current_checksum = self._calculate_backup_checksum(backup_path)
        stored_checksum = metadata.get('checksum')
        
        if current_checksum == stored_checksum:
            print(f"✅ Backup verified: {backup_name}")
            return True
        else:
            print(f"❌ Backup checksum mismatch: {backup_name}")
            return False


def main():
    """CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Advanced Test Result Backup')
    parser.add_argument('--backup', action='store_true', help='Create backup')
    parser.add_argument('--restore', type=str, help='Restore backup')
    parser.add_argument('--list', action='store_true', help='List backups')
    parser.add_argument('--verify', type=str, help='Verify backup')
    parser.add_argument('--cleanup', action='store_true', help='Clean up old backups')
    parser.add_argument('--project-root', type=str, help='Project root directory')
    
    args = parser.parse_args()
    
    project_root = Path(args.project_root) if args.project_root else Path(__file__).parent
    
    backup_system = AdvancedTestResultBackup(project_root)
    
    if args.backup:
        print("💾 Creating backup...")
        backup_system.create_backup()
    elif args.restore:
        print(f"📥 Restoring backup: {args.restore}")
        backup_system.restore_backup(args.restore)
    elif args.list:
        print("📋 Listing backups...")
        backups = backup_system.list_backups()
        for backup in backups[:20]:
            print(f"  {backup['backup_name']}: {backup.get('files_count', 0)} files")
    elif args.verify:
        backup_system.verify_backup(args.verify)
    elif args.cleanup:
        print("🧹 Cleaning up old backups...")
        backup_system.cleanup_old_backups()
    else:
        print("Use --help to see available options")


if __name__ == '__main__':
    main()

