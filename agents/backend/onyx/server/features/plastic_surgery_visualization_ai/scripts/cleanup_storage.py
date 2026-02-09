"""Script to cleanup old storage files."""

from pathlib import Path
from datetime import datetime, timedelta
import argparse

from config.settings import settings
from utils.logger import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)


def cleanup_old_files(
    directory: Path,
    days_old: int = 30,
    dry_run: bool = False
) -> tuple[int, int]:
    """
    Cleanup files older than specified days.
    
    Args:
        directory: Directory to clean
        days_old: Delete files older than this many days
        dry_run: If True, don't actually delete files
        
    Returns:
        Tuple of (files_deleted, bytes_freed)
    """
    if not directory.exists():
        logger.warning(f"Directory does not exist: {directory}")
        return 0, 0
    
    cutoff_date = datetime.now() - timedelta(days=days_old)
    files_deleted = 0
    bytes_freed = 0
    
    for file_path in directory.rglob("*"):
        if file_path.is_file():
            file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
            
            if file_mtime < cutoff_date:
                file_size = file_path.stat().st_size
                
                if dry_run:
                    logger.info(f"Would delete: {file_path} ({file_size} bytes)")
                else:
                    try:
                        file_path.unlink()
                        files_deleted += 1
                        bytes_freed += file_size
                        logger.info(f"Deleted: {file_path}")
                    except Exception as e:
                        logger.error(f"Error deleting {file_path}: {e}")
    
    return files_deleted, bytes_freed


def main():
    """Main cleanup function."""
    parser = argparse.ArgumentParser(description="Cleanup old storage files")
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Delete files older than this many days (default: 30)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without actually deleting"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory to clean (default: from settings)"
    )
    parser.add_argument(
        "--upload-dir",
        type=str,
        help="Upload directory to clean (default: from settings)"
    )
    
    args = parser.parse_args()
    
    directories = []
    
    if args.output_dir:
        directories.append(Path(args.output_dir))
    else:
        directories.append(Path(settings.output_dir))
    
    if args.upload_dir:
        directories.append(Path(args.upload_dir))
    else:
        directories.append(Path(settings.upload_dir))
    
    total_files = 0
    total_bytes = 0
    
    for directory in directories:
        logger.info(f"Cleaning directory: {directory}")
        files, bytes_freed = cleanup_old_files(
            directory,
            days_old=args.days,
            dry_run=args.dry_run
        )
        total_files += files
        total_bytes += bytes_freed
    
    if args.dry_run:
        print(f"\nDry run complete. Would delete {total_files} files ({total_bytes / 1024 / 1024:.2f} MB)")
    else:
        print(f"\nCleanup complete. Deleted {total_files} files ({total_bytes / 1024 / 1024:.2f} MB)")


if __name__ == "__main__":
    main()

