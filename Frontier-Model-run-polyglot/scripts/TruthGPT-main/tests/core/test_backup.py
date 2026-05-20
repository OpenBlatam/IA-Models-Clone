"""
Test Result Backup and Restore
Backup and restore test results and history
"""

import json
import shutil
import zipfile
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

class TestResultBackup:
    """Backup and restore test results"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.results_dir = project_root / "test_results"
        self.history_file = project_root / "test_history.json"
        self.metrics_file = project_root / "test_metrics.json"
        self.db_file = project_root / "test_results.db"
        self.backup_dir = project_root / "backups"
        self.backup_dir.mkdir(exist_ok=True)
    
    def create_backup(self, backup_name: str = None) -> Path:
        """Create backup of all test data"""
        if backup_name is None:
            backup_name = datetime.now().strftime("backup_%Y%m%d_%H%M%S")
        
        backup_path = self.backup_dir / f"{backup_name}.zip"
        
        with zipfile.ZipFile(backup_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Backup JSON results
            if self.results_dir.exists():
                for result_file in self.results_dir.glob("*.json"):
                    zipf.write(result_file, f"results/{result_file.name}")
            
            # Backup history
            if self.history_file.exists():
                zipf.write(self.history_file, "test_history.json")
            
            # Backup metrics
            if self.metrics_file.exists():
                zipf.write(self.metrics_file, "test_metrics.json")
            
            # Backup database
            if self.db_file.exists():
                zipf.write(self.db_file, "test_results.db")
            
            # Backup configuration
            config_files = [
                ".coveragerc",
                "alert_config.json"
            ]
            for config_file in config_files:
                config_path = self.project_root / config_file
                if config_path.exists():
                    zipf.write(config_path, config_file)
            
            # Create manifest
            manifest = {
                'backup_name': backup_name,
                'timestamp': datetime.now().isoformat(),
                'files': [f.name for f in zipf.namelist()]
            }
            zipf.writestr("manifest.json", json.dumps(manifest, indent=2))
        
        return backup_path
    
    def list_backups(self) -> List[Dict]:
        """List all available backups"""
        backups = []
        for backup_file in sorted(self.backup_dir.glob("*.zip"), reverse=True):
            try:
                with zipfile.ZipFile(backup_file, 'r') as zipf:
                    manifest_str = zipf.read("manifest.json").decode('utf-8')
                    manifest = json.loads(manifest_str)
                    manifest['file'] = backup_file.name
                    manifest['size_mb'] = backup_file.stat().st_size / (1024 * 1024)
                    backups.append(manifest)
            except Exception:
                # Try to get basic info
                backups.append({
                    'file': backup_file.name,
                    'timestamp': datetime.fromtimestamp(backup_file.stat().st_mtime).isoformat(),
                    'size_mb': backup_file.stat().st_size / (1024 * 1024)
                })
        
        return backups
    
    def restore_backup(self, backup_name: str, restore_db: bool = True) -> bool:
        """Restore from backup"""
        backup_path = self.backup_dir / f"{backup_name}.zip"
        
        if not backup_path.exists():
            print(f"❌ Backup not found: {backup_name}")
            return False
        
        try:
            with zipfile.ZipFile(backup_path, 'r') as zipf:
                # Read manifest
                manifest_str = zipf.read("manifest.json").decode('utf-8')
                manifest = json.loads(manifest_str)
                
                # Restore files
                for file_info in zipf.infolist():
                    if file_info.filename == "manifest.json":
                        continue
                    
                    # Extract to appropriate location
                    if file_info.filename.startswith("results/"):
                        target = self.results_dir / file_info.filename.replace("results/", "")
                        self.results_dir.mkdir(exist_ok=True)
                        zipf.extract(file_info, self.project_root)
                        (self.project_root / file_info.filename).rename(target)
                    else:
                        zipf.extract(file_info, self.project_root)
                
                print(f"✅ Restored backup: {backup_name}")
                print(f"   Files restored: {len(manifest.get('files', []))}")
                return True
        except Exception as e:
            print(f"❌ Error restoring backup: {e}")
            return False
    
    def cleanup_old_backups(self, keep_days: int = 30):
        """Clean up old backups"""
        cutoff_date = datetime.now().timestamp() - (keep_days * 24 * 60 * 60)
        
        removed = 0
        for backup_file in self.backup_dir.glob("*.zip"):
            if backup_file.stat().st_mtime < cutoff_date:
                backup_file.unlink()
                removed += 1
        
        return removed

def main():
    """Example usage"""
    from pathlib import Path
    project_root = Path(__file__).parent.parent
    
    backup = TestResultBackup(project_root)
    
    # Create backup
    backup_path = backup.create_backup()
    print(f"✅ Created backup: {backup_path}")
    
    # List backups
    backups = backup.list_backups()
    print(f"\n📦 Available backups: {len(backups)}")
    for b in backups[:5]:
        print(f"  • {b.get('backup_name', b['file'])} - {b.get('size_mb', 0):.2f} MB")
    
    # Cleanup old backups
    removed = backup.cleanup_old_backups(keep_days=30)
    if removed > 0:
        print(f"\n🗑️  Removed {removed} old backups")

if __name__ == "__main__":
    main()







