"""
BUL Backup Manager
=================

Comprehensive backup and restore management for the BUL system.
"""

import os
import sys
import json
import shutil
import gzip
import tarfile
import zipfile
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import hashlib

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BackupManager:
    """Comprehensive backup and restore management."""
    
    def __init__(self, backup_dir: str = "backups"):
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(exist_ok=True)
        
        self.backup_config = {
            'include_files': [
                'bul_optimized.py',
                'config_optimized.py',
                'modules/',
                'generated_documents/',
                'logs/',
                '.env',
                'requirements_optimized.txt'
            ],
            'exclude_patterns': [
                '__pycache__/',
                '*.pyc',
                '*.pyo',
                '*.pyd',
                '.git/',
                'node_modules/',
                '*.log',
                '*.tmp'
            ],
            'retention_days': 30,
            'compression': True
        }
    
    def create_backup(self, backup_name: str = None, include_data: bool = True) -> Dict[str, Any]:
        """Create a comprehensive backup."""
        if not backup_name:
            backup_name = f"bul_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        print(f"💾 Creating backup: {backup_name}")
        
        backup_path = self.backup_dir / f"{backup_name}.tar.gz"
        temp_dir = self.backup_dir / "temp" / backup_name
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Copy files to temp directory
            copied_files = []
            for item in self.backup_config['include_files']:
                source = Path(item)
                if source.exists():
                    dest = temp_dir / item
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    
                    if source.is_file():
                        shutil.copy2(source, dest)
                        copied_files.append(str(item))
                    elif source.is_dir():
                        shutil.copytree(source, dest, dirs_exist_ok=True)
                        copied_files.append(f"{item}/")
            
            # Create backup metadata
            metadata = {
                'backup_name': backup_name,
                'created_at': datetime.now().isoformat(),
                'files_included': copied_files,
                'include_data': include_data,
                'system_info': self._get_system_info()
            }
            
            with open(temp_dir / 'backup_metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Create compressed archive
            with tarfile.open(backup_path, 'w:gz') as tar:
                tar.add(temp_dir, arcname=backup_name)
            
            # Clean up temp directory
            shutil.rmtree(temp_dir.parent)
            
            # Calculate backup size and checksum
            backup_size = backup_path.stat().st_size
            backup_checksum = self._calculate_checksum(backup_path)
            
            result = {
                'backup_name': backup_name,
                'backup_path': str(backup_path),
                'backup_size': backup_size,
                'backup_checksum': backup_checksum,
                'files_included': len(copied_files),
                'created_at': metadata['created_at'],
                'success': True
            }
            
            print(f"✅ Backup created successfully: {backup_path}")
            print(f"   Size: {self._format_size(backup_size)}")
            print(f"   Files: {len(copied_files)}")
            
            return result
            
        except Exception as e:
            print(f"❌ Error creating backup: {e}")
            # Clean up on error
            if temp_dir.exists():
                shutil.rmtree(temp_dir.parent, ignore_errors=True)
            return {'success': False, 'error': str(e)}
    
    def restore_backup(self, backup_name: str, restore_path: str = ".") -> Dict[str, Any]:
        """Restore from a backup."""
        print(f"🔄 Restoring backup: {backup_name}")
        
        backup_path = self.backup_dir / f"{backup_name}.tar.gz"
        if not backup_path.exists():
            return {'success': False, 'error': f'Backup not found: {backup_path}'}
        
        restore_path = Path(restore_path)
        restore_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # Extract backup
            with tarfile.open(backup_path, 'r:gz') as tar:
                tar.extractall(restore_path)
            
            # Read metadata
            metadata_path = restore_path / backup_name / 'backup_metadata.json'
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                print(f"✅ Backup restored successfully")
                print(f"   Original backup date: {metadata['created_at']}")
                print(f"   Files restored: {len(metadata['files_included'])}")
                
                return {
                    'success': True,
                    'backup_name': backup_name,
                    'restore_path': str(restore_path),
                    'metadata': metadata
                }
            else:
                return {'success': False, 'error': 'Backup metadata not found'}
                
        except Exception as e:
            print(f"❌ Error restoring backup: {e}")
            return {'success': False, 'error': str(e)}
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """List all available backups."""
        backups = []
        
        for backup_file in self.backup_dir.glob("*.tar.gz"):
            try:
                stat = backup_file.stat()
                backup_name = backup_file.stem
                
                # Try to get metadata
                metadata = self._get_backup_metadata(backup_name)
                
                backup_info = {
                    'name': backup_name,
                    'path': str(backup_file),
                    'size': stat.st_size,
                    'created_at': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    'modified_at': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    'metadata': metadata
                }
                
                backups.append(backup_info)
                
            except Exception as e:
                logger.warning(f"Could not read backup info for {backup_file}: {e}")
        
        # Sort by creation date (newest first)
        backups.sort(key=lambda x: x['created_at'], reverse=True)
        
        return backups
    
    def cleanup_old_backups(self) -> Dict[str, Any]:
        """Clean up old backups based on retention policy."""
        print("🧹 Cleaning up old backups...")
        
        retention_days = self.backup_config['retention_days']
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        
        backups = self.list_backups()
        deleted_backups = []
        
        for backup in backups:
            backup_date = datetime.fromisoformat(backup['created_at'])
            if backup_date < cutoff_date:
                try:
                    backup_path = Path(backup['path'])
                    backup_path.unlink()
                    deleted_backups.append(backup['name'])
                    print(f"   🗑️  Deleted old backup: {backup['name']}")
                except Exception as e:
                    logger.warning(f"Could not delete backup {backup['name']}: {e}")
        
        return {
            'deleted_count': len(deleted_backups),
            'deleted_backups': deleted_backups,
            'retention_days': retention_days
        }
    
    def verify_backup(self, backup_name: str) -> Dict[str, Any]:
        """Verify backup integrity."""
        print(f"🔍 Verifying backup: {backup_name}")
        
        backup_path = self.backup_dir / f"{backup_name}.tar.gz"
        if not backup_path.exists():
            return {'success': False, 'error': f'Backup not found: {backup_path}'}
        
        try:
            # Check if archive can be opened
            with tarfile.open(backup_path, 'r:gz') as tar:
                members = tar.getmembers()
            
            # Get metadata
            metadata = self._get_backup_metadata(backup_name)
            
            # Calculate current checksum
            current_checksum = self._calculate_checksum(backup_path)
            
            result = {
                'success': True,
                'backup_name': backup_name,
                'file_count': len(members),
                'metadata': metadata,
                'checksum': current_checksum,
                'size': backup_path.stat().st_size
            }
            
            print(f"✅ Backup verification successful")
            print(f"   Files: {len(members)}")
            print(f"   Size: {self._format_size(result['size'])}")
            
            return result
            
        except Exception as e:
            print(f"❌ Backup verification failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def schedule_backup(self, schedule_type: str = 'daily') -> Dict[str, Any]:
        """Create backup schedule configuration."""
        print(f"⏰ Creating {schedule_type} backup schedule...")
        
        schedules = {
            'daily': {
                'cron': '0 2 * * *',  # 2 AM daily
                'description': 'Daily backup at 2 AM'
            },
            'weekly': {
                'cron': '0 2 * * 0',  # 2 AM on Sunday
                'description': 'Weekly backup on Sunday at 2 AM'
            },
            'monthly': {
                'cron': '0 2 1 * *',  # 2 AM on 1st of month
                'description': 'Monthly backup on 1st at 2 AM'
            }
        }
        
        if schedule_type not in schedules:
            return {'success': False, 'error': f'Unknown schedule type: {schedule_type}'}
        
        schedule_config = schedules[schedule_type]
        
        # Create cron job entry
        cron_entry = f"{schedule_config['cron']} cd {os.getcwd()} && python backup_manager.py --create --schedule {schedule_type}"
        
        # Create systemd timer (alternative)
        systemd_timer = f"""[Unit]
Description=BUL {schedule_type.title()} Backup
Requires=bul-backup.service

[Timer]
OnCalendar={schedule_type}
Persistent=true

[Install]
WantedBy=timers.target
"""
        
        systemd_service = f"""[Unit]
Description=BUL Backup Service

[Service]
Type=oneshot
WorkingDirectory={os.getcwd()}
ExecStart=/usr/bin/python3 backup_manager.py --create --schedule {schedule_type}
User=bul
Group=bul
"""
        
        try:
            # Save configurations
            with open(f'backup_schedule_{schedule_type}.cron', 'w') as f:
                f.write(cron_entry)
            
            with open(f'bul-backup-{schedule_type}.timer', 'w') as f:
                f.write(systemd_timer)
            
            with open(f'bul-backup.service', 'w') as f:
                f.write(systemd_service)
            
            result = {
                'success': True,
                'schedule_type': schedule_type,
                'cron_entry': cron_entry,
                'description': schedule_config['description'],
                'files_created': [
                    f'backup_schedule_{schedule_type}.cron',
                    f'bul-backup-{schedule_type}.timer',
                    'bul-backup.service'
                ]
            }
            
            print(f"✅ {schedule_type.title()} backup schedule created")
            print(f"   Cron: {schedule_config['cron']}")
            print(f"   Description: {schedule_config['description']}")
            
            return result
            
        except Exception as e:
            print(f"❌ Error creating backup schedule: {e}")
            return {'success': False, 'error': str(e)}
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information for backup metadata."""
        return {
            'python_version': sys.version,
            'platform': sys.platform,
            'working_directory': os.getcwd(),
            'timestamp': datetime.now().isoformat()
        }
    
    def _get_backup_metadata(self, backup_name: str) -> Optional[Dict[str, Any]]:
        """Get metadata from backup file."""
        backup_path = self.backup_dir / f"{backup_name}.tar.gz"
        
        try:
            with tarfile.open(backup_path, 'r:gz') as tar:
                metadata_file = tar.extractfile(f"{backup_name}/backup_metadata.json")
                if metadata_file:
                    return json.loads(metadata_file.read().decode('utf-8'))
        except Exception:
            pass
        
        return None
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    def _format_size(self, size_bytes: int) -> str:
        """Format file size in human readable format."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"
    
    def generate_backup_report(self) -> str:
        """Generate backup status report."""
        backups = self.list_backups()
        cleanup_result = self.cleanup_old_backups()
        
        report = f"""
BUL Backup Status Report
========================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

BACKUP CONFIGURATION
-------------------
Backup Directory: {self.backup_dir}
Retention Days: {self.backup_config['retention_days']}
Compression: {self.backup_config['compression']}

INCLUDED FILES
--------------
{chr(10).join(f"- {file}" for file in self.backup_config['include_files'])}

EXCLUDED PATTERNS
-----------------
{chr(10).join(f"- {pattern}" for pattern in self.backup_config['exclude_patterns'])}

AVAILABLE BACKUPS
-----------------
Total Backups: {len(backups)}
"""
        
        if backups:
            total_size = sum(backup['size'] for backup in backups)
            report += f"Total Size: {self._format_size(total_size)}\n\n"
            
            for backup in backups[:10]:  # Show last 10 backups
                report += f"""
{backup['name']}:
  Created: {backup['created_at']}
  Size: {self._format_size(backup['size'])}
  Files: {len(backup.get('metadata', {}).get('files_included', []))}
"""
        else:
            report += "No backups found.\n"
        
        report += f"""
CLEANUP RESULTS
--------------
Deleted Backups: {cleanup_result['deleted_count']}
Retention Policy: {cleanup_result['retention_days']} days
"""
        
        if cleanup_result['deleted_backups']:
            report += "Deleted:\n"
            for backup_name in cleanup_result['deleted_backups']:
                report += f"  - {backup_name}\n"
        
        return report

def main():
    """Main backup manager function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="BUL Backup Manager")
    parser.add_argument("--create", action="store_true", help="Create a new backup")
    parser.add_argument("--restore", help="Restore from backup (specify backup name)")
    parser.add_argument("--list", action="store_true", help="List all backups")
    parser.add_argument("--verify", help="Verify backup integrity (specify backup name)")
    parser.add_argument("--cleanup", action="store_true", help="Clean up old backups")
    parser.add_argument("--schedule", choices=['daily', 'weekly', 'monthly'], help="Create backup schedule")
    parser.add_argument("--report", action="store_true", help="Generate backup report")
    parser.add_argument("--name", help="Custom backup name")
    parser.add_argument("--backup-dir", default="backups", help="Backup directory")
    
    args = parser.parse_args()
    
    manager = BackupManager(args.backup_dir)
    
    print("💾 BUL Backup Manager")
    print("=" * 30)
    
    if args.create:
        result = manager.create_backup(args.name)
        if result['success']:
            print(f"✅ Backup created: {result['backup_name']}")
        else:
            print(f"❌ Backup failed: {result['error']}")
            return 1
    
    elif args.restore:
        result = manager.restore_backup(args.restore)
        if result['success']:
            print(f"✅ Backup restored: {result['backup_name']}")
        else:
            print(f"❌ Restore failed: {result['error']}")
            return 1
    
    elif args.list:
        backups = manager.list_backups()
        if backups:
            print(f"\n📋 Available Backups ({len(backups)}):")
            print("-" * 50)
            for backup in backups:
                print(f"{backup['name']:<30} {manager._format_size(backup['size']):<10} {backup['created_at']}")
        else:
            print("No backups found.")
    
    elif args.verify:
        result = manager.verify_backup(args.verify)
        if result['success']:
            print(f"✅ Backup verified: {result['backup_name']}")
        else:
            print(f"❌ Verification failed: {result['error']}")
            return 1
    
    elif args.cleanup:
        result = manager.cleanup_old_backups()
        print(f"🧹 Cleanup completed: {result['deleted_count']} backups deleted")
    
    elif args.schedule:
        result = manager.schedule_backup(args.schedule)
        if result['success']:
            print(f"✅ {args.schedule.title()} schedule created")
        else:
            print(f"❌ Schedule creation failed: {result['error']}")
            return 1
    
    elif args.report:
        report = manager.generate_backup_report()
        print(report)
        
        # Save report
        report_file = f"backup_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        print(f"📄 Report saved to: {report_file}")
    
    else:
        parser.print_help()
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
