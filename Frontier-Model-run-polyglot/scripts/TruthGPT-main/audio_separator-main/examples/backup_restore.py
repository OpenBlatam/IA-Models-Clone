"""
Examples of backup and restore features.
"""

from audio_separator.utils.backup_utils import (
    backup_separation_results,
    restore_separation_results,
    list_backups
)


def example_backup():
    """Example of backing up separation results."""
    print("Example 1: Backup Results")
    print("-" * 50)
    
    backup_path = backup_separation_results(
        "separated",
        "backups",
        include_metadata=True
    )
    
    print(f"Backed up to: {backup_path}")
    print()


def example_list_backups():
    """Example of listing backups."""
    print("Example 2: List Backups")
    print("-" * 50)
    
    backups = list_backups("backups")
    
    print(f"Found {len(backups)} backup(s):")
    for i, backup in enumerate(backups, 1):
        print(f"\n{i}. {backup['path']}")
        print(f"   Timestamp: {backup['timestamp']}")
        if backup['source_dir']:
            print(f"   Source: {backup['source_dir']}")
    print()


def example_restore():
    """Example of restoring from backup."""
    print("Example 3: Restore Results")
    print("-" * 50)
    
    # List backups first
    backups = list_backups("backups")
    if not backups:
        print("No backups found")
        return
    
    # Restore from most recent backup
    latest_backup = backups[0]['path']
    
    restore_separation_results(
        latest_backup,
        "restored",
        overwrite=True
    )
    
    print(f"Restored from: {latest_backup}")
    print()


if __name__ == "__main__":
    example_backup()
    example_list_backups()
    example_restore()

