#!/usr/bin/env python3
"""
Backup Tasks Script
===================

Script to backup tasks and results.
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from imagen_video_enhancer_ai.utils.backup import BackupManager
from imagen_video_enhancer_ai import EnhancerAgent, EnhancerConfig


async def main():
    parser = argparse.ArgumentParser(description="Backup tasks and results")
    parser.add_argument("--output-dir", default="output", help="Output directory")
    parser.add_argument("--backup-dir", default="backups", help="Backup directory")
    parser.add_argument("--name", help="Backup name (defaults to timestamp)")
    parser.add_argument("--compress", action="store_true", help="Compress backup")
    parser.add_argument("--list", action="store_true", help="List backups")
    parser.add_argument("--restore", help="Restore backup by name")
    parser.add_argument("--target-dir", help="Target directory for restore")
    parser.add_argument("--delete", help="Delete backup by name")
    
    args = parser.parse_args()
    
    backup_manager = BackupManager(backup_dir=args.backup_dir)
    
    if args.list:
        backups = backup_manager.list_backups()
        print(f"\nFound {len(backups)} backups:\n")
        for backup in backups:
            print(f"  {backup['name']}")
            print(f"    Path: {backup['path']}")
            print(f"    Size: {backup['size_mb']:.2f} MB")
            print(f"    Created: {backup['created']}")
            print()
        return
    
    if args.delete:
        if backup_manager.delete_backup(args.delete):
            print(f"Deleted backup: {args.delete}")
        else:
            print(f"Failed to delete backup: {args.delete}")
        return
    
    if args.restore:
        if not args.target_dir:
            print("Error: --target-dir required for restore")
            return
        
        try:
            restored = backup_manager.restore_backup(
                args.restore,
                args.target_dir,
                overwrite=True
            )
            print(f"Restored backup to: {restored}")
        except Exception as e:
            print(f"Error restoring backup: {e}")
        return
    
    # Create backup
    try:
        config = EnhancerConfig()
        agent = EnhancerAgent(config=config, output_dir=args.output_dir)
        
        backup_path = backup_manager.create_backup(
            source_dir=args.output_dir,
            backup_name=args.name,
            compress=args.compress
        )
        
        print(f"Backup created: {backup_path}")
        
        await agent.close()
    except Exception as e:
        print(f"Error creating backup: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())




