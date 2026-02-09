"""
Script to backup and restore separation results.
"""

import argparse
import sys
from pathlib import Path
from audio_separator.utils.backup_utils import (
    backup_separation_results,
    restore_separation_results,
    list_backups
)
from audio_separator.logger import logger


def main():
    parser = argparse.ArgumentParser(
        description="Backup and restore separation results"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Backup command
    backup_parser = subparsers.add_parser("backup", help="Backup separation results")
    backup_parser.add_argument("results_dir", type=str, help="Results directory to backup")
    backup_parser.add_argument("-o", "--output", type=str, default="backups", help="Backup directory")
    backup_parser.add_argument("--no-metadata", action="store_true", help="Don't include metadata")
    
    # Restore command
    restore_parser = subparsers.add_parser("restore", help="Restore separation results")
    restore_parser.add_argument("backup_dir", type=str, help="Backup directory")
    restore_parser.add_argument("-o", "--output", type=str, required=True, help="Restore directory")
    restore_parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List available backups")
    list_parser.add_argument("backup_dir", type=str, help="Backup directory")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        if args.command == "backup":
            backup_path = backup_separation_results(
                args.results_dir,
                args.output,
                include_metadata=not args.no_metadata
            )
            print(f"Backup created at: {backup_path}")
            return 0
        
        elif args.command == "restore":
            restore_separation_results(
                args.backup_dir,
                args.output,
                overwrite=args.overwrite
            )
            print(f"Results restored to: {args.output}")
            return 0
        
        elif args.command == "list":
            backups = list_backups(args.backup_dir)
            if not backups:
                print("No backups found")
                return 0
            
            print(f"\nFound {len(backups)} backup(s):\n")
            for i, backup in enumerate(backups, 1):
                print(f"{i}. {Path(backup['path']).name}")
                print(f"   Timestamp: {backup['timestamp']}")
                if backup['source_dir']:
                    print(f"   Source: {backup['source_dir']}")
                print()
            return 0
        
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        logger.exception("Backup/restore failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

