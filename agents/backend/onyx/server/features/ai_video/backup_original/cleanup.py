from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import os
import sys
import shutil
import logging
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any
from core.utils import cleanup_old_files, get_memory_usage, get_cpu_usage
from core.exceptions import AIVideoError
from typing import Any, List, Dict, Optional
import asyncio
#!/usr/bin/env python3
"""
AI Video System - Cleanup Script

Production-ready cleanup script for maintaining system health and performance.
"""


# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))


logger = logging.getLogger(__name__)


class AIVideoCleanup:
    """
    Production-ready cleanup utility for the AI Video System.
    
    This class provides:
    - File cleanup and maintenance
    - Log rotation and cleanup
    - Cache cleanup
    - Resource monitoring
    - System optimization
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize cleanup utility.
        
        Args:
            config_file: Path to configuration file
        """
        self.config_file = config_file
        self.cleanup_stats = {
            'files_removed': 0,
            'bytes_freed': 0,
            'directories_cleaned': 0,
            'errors': 0
        }
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    
    def cleanup_temp_files(self, temp_dir: str, max_age_days: int = 1) -> Dict[str, Any]:
        """
        Clean up temporary files.
        
        Args:
            temp_dir: Temporary directory path
            max_age_days: Maximum age in days
            
        Returns:
            Dict[str, Any]: Cleanup statistics
        """
        logger.info(f"🧹 Cleaning temporary files in {temp_dir}")
        
        try:
            temp_path = Path(temp_dir)
            if not temp_path.exists():
                logger.warning(f"Temporary directory does not exist: {temp_dir}")
                return {'files_removed': 0, 'bytes_freed': 0}
            
            files_removed = 0
            bytes_freed = 0
            cutoff_time = datetime.now() - timedelta(days=max_age_days)
            
            for file_path in temp_path.rglob("*"):
                if file_path.is_file():
                    try:
                        file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                        if file_time < cutoff_time:
                            file_size = file_path.stat().st_size
                            file_path.unlink()
                            files_removed += 1
                            bytes_freed += file_size
                            logger.debug(f"Removed temp file: {file_path}")
                    except Exception as e:
                        logger.warning(f"Failed to remove temp file {file_path}: {e}")
                        self.cleanup_stats['errors'] += 1
            
            self.cleanup_stats['files_removed'] += files_removed
            self.cleanup_stats['bytes_freed'] += bytes_freed
            
            logger.info(f"✅ Removed {files_removed} temp files, freed {bytes_freed} bytes")
            return {'files_removed': files_removed, 'bytes_freed': bytes_freed}
            
        except Exception as e:
            logger.error(f"❌ Failed to cleanup temp files: {e}")
            self.cleanup_stats['errors'] += 1
            return {'files_removed': 0, 'bytes_freed': 0}
    
    def cleanup_old_logs(self, log_dir: str, max_age_days: int = 30) -> Dict[str, Any]:
        """
        Clean up old log files.
        
        Args:
            log_dir: Log directory path
            max_age_days: Maximum age in days
            
        Returns:
            Dict[str, Any]: Cleanup statistics
        """
        logger.info(f"📋 Cleaning old logs in {log_dir}")
        
        try:
            log_path = Path(log_dir)
            if not log_path.exists():
                logger.warning(f"Log directory does not exist: {log_dir}")
                return {'files_removed': 0, 'bytes_freed': 0}
            
            files_removed = 0
            bytes_freed = 0
            cutoff_time = datetime.now() - timedelta(days=max_age_days)
            
            for file_path in log_path.glob("*.log*"):
                if file_path.is_file():
                    try:
                        file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                        if file_time < cutoff_time:
                            file_size = file_path.stat().st_size
                            file_path.unlink()
                            files_removed += 1
                            bytes_freed += file_size
                            logger.debug(f"Removed old log: {file_path}")
                    except Exception as e:
                        logger.warning(f"Failed to remove old log {file_path}: {e}")
                        self.cleanup_stats['errors'] += 1
            
            self.cleanup_stats['files_removed'] += files_removed
            self.cleanup_stats['bytes_freed'] += bytes_freed
            
            logger.info(f"✅ Removed {files_removed} old logs, freed {bytes_freed} bytes")
            return {'files_removed': files_removed, 'bytes_freed': bytes_freed}
            
        except Exception as e:
            logger.error(f"❌ Failed to cleanup old logs: {e}")
            self.cleanup_stats['errors'] += 1
            return {'files_removed': 0, 'bytes_freed': 0}
    
    def cleanup_cache(self, cache_dir: str, max_age_hours: int = 24) -> Dict[str, Any]:
        """
        Clean up cache files.
        
        Args:
            cache_dir: Cache directory path
            max_age_hours: Maximum age in hours
            
        Returns:
            Dict[str, Any]: Cleanup statistics
        """
        logger.info(f"🗂️ Cleaning cache in {cache_dir}")
        
        try:
            cache_path = Path(cache_dir)
            if not cache_path.exists():
                logger.warning(f"Cache directory does not exist: {cache_dir}")
                return {'files_removed': 0, 'bytes_freed': 0}
            
            files_removed = 0
            bytes_freed = 0
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
            
            for file_path in cache_path.rglob("*"):
                if file_path.is_file():
                    try:
                        file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                        if file_time < cutoff_time:
                            file_size = file_path.stat().st_size
                            file_path.unlink()
                            files_removed += 1
                            bytes_freed += file_size
                            logger.debug(f"Removed cache file: {file_path}")
                    except Exception as e:
                        logger.warning(f"Failed to remove cache file {file_path}: {e}")
                        self.cleanup_stats['errors'] += 1
            
            self.cleanup_stats['files_removed'] += files_removed
            self.cleanup_stats['bytes_freed'] += bytes_freed
            
            logger.info(f"✅ Removed {files_removed} cache files, freed {bytes_freed} bytes")
            return {'files_removed': files_removed, 'bytes_freed': bytes_freed}
            
        except Exception as e:
            logger.error(f"❌ Failed to cleanup cache: {e}")
            self.cleanup_stats['errors'] += 1
            return {'files_removed': 0, 'bytes_freed': 0}
    
    def cleanup_storage(self, storage_dir: str, max_age_days: int = 7) -> Dict[str, Any]:
        """
        Clean up old storage files.
        
        Args:
            storage_dir: Storage directory path
            max_age_days: Maximum age in days
            
        Returns:
            Dict[str, Any]: Cleanup statistics
        """
        logger.info(f"💾 Cleaning storage in {storage_dir}")
        
        try:
            storage_path = Path(storage_dir)
            if not storage_path.exists():
                logger.warning(f"Storage directory does not exist: {storage_dir}")
                return {'files_removed': 0, 'bytes_freed': 0}
            
            files_removed = 0
            bytes_freed = 0
            cutoff_time = datetime.now() - timedelta(days=max_age_days)
            
            # Clean up old videos
            videos_dir = storage_path / "videos"
            if videos_dir.exists():
                for file_path in videos_dir.glob("*"):
                    if file_path.is_file():
                        try:
                            file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                            if file_time < cutoff_time:
                                file_size = file_path.stat().st_size
                                file_path.unlink()
                                files_removed += 1
                                bytes_freed += file_size
                                logger.debug(f"Removed old video: {file_path}")
                        except Exception as e:
                            logger.warning(f"Failed to remove old video {file_path}: {e}")
                            self.cleanup_stats['errors'] += 1
            
            # Clean up old thumbnails
            thumbnails_dir = storage_path / "thumbnails"
            if thumbnails_dir.exists():
                for file_path in thumbnails_dir.glob("*"):
                    if file_path.is_file():
                        try:
                            file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                            if file_time < cutoff_time:
                                file_size = file_path.stat().st_size
                                file_path.unlink()
                                files_removed += 1
                                bytes_freed += file_size
                                logger.debug(f"Removed old thumbnail: {file_path}")
                        except Exception as e:
                            logger.warning(f"Failed to remove old thumbnail {file_path}: {e}")
                            self.cleanup_stats['errors'] += 1
            
            self.cleanup_stats['files_removed'] += files_removed
            self.cleanup_stats['bytes_freed'] += bytes_freed
            
            logger.info(f"✅ Removed {files_removed} storage files, freed {bytes_freed} bytes")
            return {'files_removed': files_removed, 'bytes_freed': bytes_freed}
            
        except Exception as e:
            logger.error(f"❌ Failed to cleanup storage: {e}")
            self.cleanup_stats['errors'] += 1
            return {'files_removed': 0, 'bytes_freed': 0}
    
    def cleanup_empty_directories(self, base_dir: str) -> int:
        """
        Remove empty directories.
        
        Args:
            base_dir: Base directory to search
            
        Returns:
            int: Number of directories removed
        """
        logger.info(f"📁 Cleaning empty directories in {base_dir}")
        
        try:
            base_path = Path(base_dir)
            if not base_path.exists():
                logger.warning(f"Base directory does not exist: {base_dir}")
                return 0
            
            directories_removed = 0
            
            # Walk through directories in reverse order (deepest first)
            for dir_path in sorted(base_path.rglob("*"), key=lambda x: len(x.parts), reverse=True):
                if dir_path.is_dir():
                    try:
                        # Check if directory is empty
                        if not any(dir_path.iterdir()):
                            dir_path.rmdir()
                            directories_removed += 1
                            logger.debug(f"Removed empty directory: {dir_path}")
                    except Exception as e:
                        logger.warning(f"Failed to remove empty directory {dir_path}: {e}")
                        self.cleanup_stats['errors'] += 1
            
            self.cleanup_stats['directories_cleaned'] += directories_removed
            
            logger.info(f"✅ Removed {directories_removed} empty directories")
            return directories_removed
            
        except Exception as e:
            logger.error(f"❌ Failed to cleanup empty directories: {e}")
            self.cleanup_stats['errors'] += 1
            return 0
    
    def get_system_stats(self) -> Dict[str, Any]:
        """
        Get system resource statistics.
        
        Returns:
            Dict[str, Any]: System statistics
        """
        try:
            memory_usage = get_memory_usage()
            cpu_usage = get_cpu_usage()
            
            return {
                'memory_usage': memory_usage,
                'cpu_usage': cpu_usage,
                'cleanup_stats': self.cleanup_stats
            }
        except Exception as e:
            logger.error(f"Failed to get system stats: {e}")
            return {'error': str(e)}
    
    def run_full_cleanup(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run full cleanup with configuration.
        
        Args:
            config: Cleanup configuration
            
        Returns:
            Dict[str, Any]: Cleanup results
        """
        logger.info("🚀 Starting full system cleanup...")
        
        results = {
            'temp_files': {},
            'logs': {},
            'cache': {},
            'storage': {},
            'empty_directories': 0,
            'system_stats': {},
            'errors': 0
        }
        
        try:
            # Cleanup temp files
            if config.get('cleanup_temp', True):
                results['temp_files'] = self.cleanup_temp_files(
                    config.get('temp_dir', './temp'),
                    config.get('temp_max_age_days', 1)
                )
            
            # Cleanup old logs
            if config.get('cleanup_logs', True):
                results['logs'] = self.cleanup_old_logs(
                    config.get('log_dir', './logs'),
                    config.get('log_max_age_days', 30)
                )
            
            # Cleanup cache
            if config.get('cleanup_cache', True):
                results['cache'] = self.cleanup_cache(
                    config.get('cache_dir', './cache'),
                    config.get('cache_max_age_hours', 24)
                )
            
            # Cleanup storage
            if config.get('cleanup_storage', True):
                results['storage'] = self.cleanup_storage(
                    config.get('storage_dir', './storage'),
                    config.get('storage_max_age_days', 7)
                )
            
            # Cleanup empty directories
            if config.get('cleanup_empty_dirs', True):
                results['empty_directories'] = self.cleanup_empty_directories(
                    config.get('base_dir', '.')
                )
            
            # Get system stats
            results['system_stats'] = self.get_system_stats()
            results['errors'] = self.cleanup_stats['errors']
            
            logger.info("✅ Full cleanup completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"❌ Full cleanup failed: {e}")
            results['errors'] += 1
            return results


def main():
    """Main cleanup function."""
    parser = argparse.ArgumentParser(description="AI Video System Cleanup")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--temp-dir", default="./temp", help="Temporary directory")
    parser.add_argument("--log-dir", default="./logs", help="Log directory")
    parser.add_argument("--storage-dir", default="./storage", help="Storage directory")
    parser.add_argument("--cache-dir", default="./cache", help="Cache directory")
    parser.add_argument("--temp-age", type=int, default=1, help="Temp files max age (days)")
    parser.add_argument("--log-age", type=int, default=30, help="Log files max age (days)")
    parser.add_argument("--storage-age", type=int, default=7, help="Storage files max age (days)")
    parser.add_argument("--cache-age", type=int, default=24, help="Cache files max age (hours)")
    parser.add_argument("--no-temp", action="store_true", help="Skip temp file cleanup")
    parser.add_argument("--no-logs", action="store_true", help="Skip log cleanup")
    parser.add_argument("--no-storage", action="store_true", help="Skip storage cleanup")
    parser.add_argument("--no-cache", action="store_true", help="Skip cache cleanup")
    parser.add_argument("--no-empty-dirs", action="store_true", help="Skip empty directory cleanup")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be cleaned without doing it")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Setup verbose logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create cleanup configuration
    config = {
        'temp_dir': args.temp_dir,
        'log_dir': args.log_dir,
        'storage_dir': args.storage_dir,
        'cache_dir': args.cache_dir,
        'temp_max_age_days': args.temp_age,
        'log_max_age_days': args.log_age,
        'storage_max_age_days': args.storage_age,
        'cache_max_age_hours': args.cache_age,
        'cleanup_temp': not args.no_temp,
        'cleanup_logs': not args.no_logs,
        'cleanup_storage': not args.no_storage,
        'cleanup_cache': not args.no_cache,
        'cleanup_empty_dirs': not args.no_empty_dirs,
        'dry_run': args.dry_run
    }
    
    # Create cleanup utility
    cleanup = AIVideoCleanup(args.config)
    
    if args.dry_run:
        print("🔍 Dry run mode - showing what would be cleaned:")
        # In dry run mode, just show what would be cleaned
        for key, value in config.items():
            if key.startswith('cleanup_') and value:
                print(f"  - {key}: {value}")
        return
    
    try:
        # Run cleanup
        results = cleanup.run_full_cleanup(config)
        
        # Print results
        print("\n" + "="*50)
        print("🧹 Cleanup Results")
        print("="*50)
        
        total_files = 0
        total_bytes = 0
        
        for cleanup_type, stats in results.items():
            if isinstance(stats, dict) and 'files_removed' in stats:
                files = stats['files_removed']
                bytes_freed = stats['bytes_freed']
                total_files += files
                total_bytes += bytes_freed
                print(f"{cleanup_type.replace('_', ' ').title()}: {files} files, {bytes_freed} bytes")
        
        print(f"\n📊 Summary:")
        print(f"  Total files removed: {total_files}")
        print(f"  Total bytes freed: {total_bytes}")
        print(f"  Empty directories removed: {results['empty_directories']}")
        print(f"  Errors encountered: {results['errors']}")
        
        if results['system_stats']:
            print(f"\n💻 System Stats:")
            for key, value in results['system_stats'].items():
                print(f"  {key}: {value}")
        
    except Exception as e:
        print(f"❌ Cleanup failed: {e}")
        sys.exit(1)


match __name__:
    case "__main__":
    main() 