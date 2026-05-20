"""
Backup script for database and data.
"""

import os
import sys
import shutil
from pathlib import Path
from datetime import datetime


def backup_database(backup_dir: str = "backups"):
    """Backup database and data files."""
    backup_path = Path(backup_dir)
    backup_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"backup_{timestamp}"
    backup_location = backup_path / backup_name
    
    # Directories to backup
    directories_to_backup = [
        "data/tutor_db",
        "conversations",
        "reports"
    ]
    
    print(f"Creating backup: {backup_name}")
    
    for directory in directories_to_backup:
        source = Path(directory)
        if source.exists():
            dest = backup_location / directory
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source, dest, dirs_exist_ok=True)
            print(f"✓ Backed up: {directory}")
    
    # Create backup info file
    info_file = backup_location / "backup_info.txt"
    with open(info_file, "w") as f:
        f.write(f"Backup created: {datetime.now().isoformat()}\n")
        f.write(f"Backup name: {backup_name}\n")
        f.write(f"Directories backed up: {', '.join(directories_to_backup)}\n")
    
    print(f"✓ Backup completed: {backup_location}")
    return backup_location


def list_backups(backup_dir: str = "backups"):
    """List all available backups."""
    backup_path = Path(backup_dir)
    
    if not backup_path.exists():
        print("No backups directory found")
        return []
    
    backups = [d for d in backup_path.iterdir() if d.is_dir() and d.name.startswith("backup_")]
    backups.sort(reverse=True)
    
    print(f"Found {len(backups)} backups:")
    for backup in backups:
        print(f"  - {backup.name}")
    
    return backups


def restore_backup(backup_name: str, backup_dir: str = "backups"):
    """Restore from a backup."""
    backup_path = Path(backup_dir) / backup_name
    
    if not backup_path.exists():
        print(f"Error: Backup {backup_name} not found")
        return False
    
    print(f"Restoring from backup: {backup_name}")
    
    # Restore directories
    for item in backup_path.iterdir():
        if item.is_dir():
            dest = Path(item.name)
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(item, dest)
            print(f"✓ Restored: {item.name}")
    
    print("✓ Restore completed")
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Backup and restore tool")
    parser.add_argument("action", choices=["backup", "list", "restore"], help="Action to perform")
    parser.add_argument("--name", help="Backup name for restore")
    parser.add_argument("--dir", default="backups", help="Backup directory")
    
    args = parser.parse_args()
    
    if args.action == "backup":
        backup_database(args.dir)
    elif args.action == "list":
        list_backups(args.dir)
    elif args.action == "restore":
        if not args.name:
            print("Error: --name required for restore")
            sys.exit(1)
        restore_backup(args.name, args.dir)






