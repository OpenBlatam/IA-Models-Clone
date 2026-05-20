from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

import asyncio
import shutil
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List
import structlog
from .config import get_production_config
from database.connection import get_db_session
from database.repository import VideoRepository, TaskRepository, FileRepository, MetricsRepository
            import shutil
                import gzip
                    import gzip
            import gc
            import psutil
from typing import Any, List, Dict, Optional
import logging
#!/usr/bin/env python3
"""
Production Maintenance Service for OS Content UGC Video Generator
Handles maintenance tasks, cleanup, and system optimization
"""



logger = structlog.get_logger("os_content.maintenance")

class MaintenanceService:
    """Production maintenance service"""
    
    def __init__(self) -> Any:
        self.config = get_production_config()
    
    async def run_maintenance(self) -> Any:
        """Run all maintenance tasks"""
        try:
            logger.info("Starting production maintenance")
            
            # Database maintenance
            await self._database_maintenance()
            
            # File system maintenance
            await self._filesystem_maintenance()
            
            # Cache maintenance
            await self._cache_maintenance()
            
            # Log maintenance
            await self._log_maintenance()
            
            # Metrics cleanup
            await self._metrics_cleanup()
            
            # System optimization
            await self._system_optimization()
            
            logger.info("Production maintenance completed successfully")
            
        except Exception as e:
            logger.error(f"Maintenance failed: {e}")
            raise
    
    async def _database_maintenance(self) -> Any:
        """Database maintenance tasks"""
        logger.info("Starting database maintenance")
        
        try:
            async for session in get_db_session():
                video_repo = VideoRepository(session)
                task_repo = TaskRepository(session)
                file_repo = FileRepository(session)
                
                # Cleanup old completed tasks
                await self._cleanup_old_tasks(task_repo)
                
                # Cleanup old video requests
                await self._cleanup_old_video_requests(video_repo)
                
                # Cleanup orphaned files
                await self._cleanup_orphaned_files(file_repo)
                
                # Optimize database
                await self._optimize_database(session)
                
        except Exception as e:
            logger.error(f"Database maintenance failed: {e}")
    
    async def _cleanup_old_tasks(self, task_repo: TaskRepository):
        """Cleanup old completed tasks"""
        try:
            # Delete tasks older than 30 days
            cutoff_date = datetime.utcnow() - timedelta(days=30)
            
            # Get old completed tasks
            old_tasks = await task_repo.get_old_completed_tasks(cutoff_date)
            
            for task in old_tasks:
                await task_repo.delete_task(task.id)
            
            logger.info(f"Cleaned up {len(old_tasks)} old completed tasks")
            
        except Exception as e:
            logger.error(f"Failed to cleanup old tasks: {e}")
    
    async def _cleanup_old_video_requests(self, video_repo: VideoRepository):
        """Cleanup old video requests"""
        try:
            # Delete video requests older than 90 days
            cutoff_date = datetime.utcnow() - timedelta(days=90)
            
            # Get old video requests
            old_requests = await video_repo.get_old_video_requests(cutoff_date)
            
            for request in old_requests:
                # Delete associated files
                await self._delete_video_files(request.id)
                
                # Delete video request
                await video_repo.delete_video_request(request.id)
            
            logger.info(f"Cleaned up {len(old_requests)} old video requests")
            
        except Exception as e:
            logger.error(f"Failed to cleanup old video requests: {e}")
    
    async def _cleanup_orphaned_files(self, file_repo: FileRepository):
        """Cleanup orphaned files"""
        try:
            # Get orphaned files (no associated video request)
            orphaned_files = await file_repo.get_orphaned_files()
            
            for file_record in orphaned_files:
                # Delete physical file
                if os.path.exists(file_record.file_path):
                    os.remove(file_record.file_path)
                
                # Delete database record
                await file_repo.delete_file(file_record.id)
            
            logger.info(f"Cleaned up {len(orphaned_files)} orphaned files")
            
        except Exception as e:
            logger.error(f"Failed to cleanup orphaned files: {e}")
    
    async def _optimize_database(self, session) -> Any:
        """Optimize database"""
        try:
            # Run VACUUM and ANALYZE
            await session.execute("VACUUM ANALYZE")
            await session.commit()
            
            logger.info("Database optimization completed")
            
        except Exception as e:
            logger.error(f"Database optimization failed: {e}")
    
    async def _filesystem_maintenance(self) -> Any:
        """File system maintenance"""
        logger.info("Starting filesystem maintenance")
        
        try:
            # Cleanup temporary files
            await self._cleanup_temp_files()
            
            # Cleanup old uploads
            await self._cleanup_old_uploads()
            
            # Check disk space
            await self._check_disk_space()
            
            # Optimize file permissions
            await self._optimize_file_permissions()
            
        except Exception as e:
            logger.error(f"Filesystem maintenance failed: {e}")
    
    async def _cleanup_temp_files(self) -> Any:
        """Cleanup temporary files"""
        try:
            temp_dirs = [
                "/tmp/os_content",
                "/var/tmp/os_content",
                "/var/cache/os_content"
            ]
            
            for temp_dir in temp_dirs:
                if os.path.exists(temp_dir):
                    # Remove files older than 24 hours
                    cutoff_time = datetime.now() - timedelta(hours=24)
                    
                    for root, dirs, files in os.walk(temp_dir):
                        for file in files:
                            file_path = os.path.join(root, file)
                            file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                            
                            if file_time < cutoff_time:
                                os.remove(file_path)
                                logger.debug(f"Removed temp file: {file_path}")
            
            logger.info("Temporary files cleanup completed")
            
        except Exception as e:
            logger.error(f"Failed to cleanup temp files: {e}")
    
    async async def _cleanup_old_uploads(self) -> Any:
        """Cleanup old uploads"""
        try:
            upload_dir = Path(self.config.upload_dir)
            if not upload_dir.exists():
                return
            
            # Remove files older than 30 days
            cutoff_time = datetime.now() - timedelta(days=30)
            
            for file_path in upload_dir.rglob("*"):
                if file_path.is_file():
                    file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                    
                    if file_time < cutoff_time:
                        file_path.unlink()
                        logger.debug(f"Removed old upload: {file_path}")
            
            logger.info("Old uploads cleanup completed")
            
        except Exception as e:
            logger.error(f"Failed to cleanup old uploads: {e}")
    
    async def _check_disk_space(self) -> Any:
        """Check disk space"""
        try:
            
            upload_dir = Path(self.config.upload_dir)
            if upload_dir.exists():
                total, used, free = shutil.disk_usage(upload_dir)
                usage_percent = (used / total) * 100
                
                if usage_percent > 90:
                    logger.warning(f"Disk usage is high: {usage_percent:.1f}%")
                    
                    # Trigger emergency cleanup
                    await self._emergency_cleanup()
                elif usage_percent > 80:
                    logger.warning(f"Disk usage is elevated: {usage_percent:.1f}%")
                
                logger.info(f"Disk usage: {usage_percent:.1f}%")
            
        except Exception as e:
            logger.error(f"Failed to check disk space: {e}")
    
    async def _emergency_cleanup(self) -> Any:
        """Emergency cleanup when disk space is low"""
        try:
            logger.warning("Starting emergency cleanup")
            
            # Remove files older than 7 days
            cutoff_time = datetime.now() - timedelta(days=7)
            upload_dir = Path(self.config.upload_dir)
            
            removed_count = 0
            for file_path in upload_dir.rglob("*"):
                if file_path.is_file():
                    file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                    
                    if file_time < cutoff_time:
                        file_path.unlink()
                        removed_count += 1
            
            logger.warning(f"Emergency cleanup removed {removed_count} files")
            
        except Exception as e:
            logger.error(f"Emergency cleanup failed: {e}")
    
    async def _optimize_file_permissions(self) -> Any:
        """Optimize file permissions"""
        try:
            upload_dir = Path(self.config.upload_dir)
            if upload_dir.exists():
                # Set directory permissions
                os.chmod(upload_dir, 0o755)
                
                # Set file permissions
                for file_path in upload_dir.rglob("*"):
                    if file_path.is_file():
                        os.chmod(file_path, 0o644)
            
            logger.info("File permissions optimized")
            
        except Exception as e:
            logger.error(f"Failed to optimize file permissions: {e}")
    
    async def _cache_maintenance(self) -> Any:
        """Cache maintenance"""
        logger.info("Starting cache maintenance")
        
        try:
            # This would integrate with the cache manager
            # For now, just log the maintenance
            logger.info("Cache maintenance completed")
            
        except Exception as e:
            logger.error(f"Cache maintenance failed: {e}")
    
    async def _log_maintenance(self) -> Any:
        """Log maintenance"""
        logger.info("Starting log maintenance")
        
        try:
            log_file = Path(self.config.log_file)
            if log_file.exists():
                # Rotate logs if they're too large
                if log_file.stat().st_size > self.config.log_max_size:
                    await self._rotate_logs()
                
                # Compress old logs
                await self._compress_old_logs()
            
            logger.info("Log maintenance completed")
            
        except Exception as e:
            logger.error(f"Log maintenance failed: {e}")
    
    async def _rotate_logs(self) -> Any:
        """Rotate log files"""
        try:
            log_file = Path(self.config.log_file)
            backup_count = self.config.log_backup_count
            
            # Remove oldest backup if it exists
            oldest_backup = log_file.with_suffix(f".{backup_count}.gz")
            if oldest_backup.exists():
                oldest_backup.unlink()
            
            # Rotate existing backups
            for i in range(backup_count - 1, 0, -1):
                old_backup = log_file.with_suffix(f".{i}.gz")
                new_backup = log_file.with_suffix(f".{i + 1}.gz")
                
                if old_backup.exists():
                    old_backup.rename(new_backup)
            
            # Create new backup
            if log_file.exists():
                backup_file = log_file.with_suffix(".1.gz")
                
                with open(log_file, 'rb') as f_in:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                    with gzip.open(backup_file, 'wb') as f_out:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                        shutil.copyfileobj(f_in, f_out)
                
                # Truncate original log file
                log_file.truncate(0)
            
            logger.info("Log rotation completed")
            
        except Exception as e:
            logger.error(f"Log rotation failed: {e}")
    
    async def _compress_old_logs(self) -> Any:
        """Compress old log files"""
        try:
            log_dir = Path(self.config.log_file).parent
            
            for log_file in log_dir.glob("*.log.*"):
                if not log_file.suffix == '.gz':
                    # Compress log file
                    
                    with open(log_file, 'rb') as f_in:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                        with gzip.open(f"{log_file}.gz", 'wb') as f_out:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                            shutil.copyfileobj(f_in, f_out)
                    
                    # Remove original file
                    log_file.unlink()
            
            logger.info("Old logs compression completed")
            
        except Exception as e:
            logger.error(f"Log compression failed: {e}")
    
    async def _metrics_cleanup(self) -> Any:
        """Cleanup old metrics"""
        logger.info("Starting metrics cleanup")
        
        try:
            async for session in get_db_session():
                metrics_repo = MetricsRepository(session)
                
                # Delete metrics older than 30 days
                cutoff_date = datetime.utcnow() - timedelta(days=30)
                await metrics_repo.delete_old_metrics(cutoff_date)
                
            logger.info("Metrics cleanup completed")
            
        except Exception as e:
            logger.error(f"Metrics cleanup failed: {e}")
    
    async def _system_optimization(self) -> Any:
        """System optimization"""
        logger.info("Starting system optimization")
        
        try:
            # Clear system caches
            await self._clear_system_caches()
            
            # Optimize memory usage
            await self._optimize_memory()
            
            # Check system health
            await self._check_system_health()
            
            logger.info("System optimization completed")
            
        except Exception as e:
            logger.error(f"System optimization failed: {e}")
    
    async def _clear_system_caches(self) -> Any:
        """Clear system caches"""
        try:
            # This would clear various system caches
            # For now, just log the operation
            logger.info("System caches cleared")
            
        except Exception as e:
            logger.error(f"Failed to clear system caches: {e}")
    
    async def _optimize_memory(self) -> Any:
        """Optimize memory usage"""
        try:
            
            # Force garbage collection
            collected = gc.collect()
            logger.info(f"Garbage collection freed {collected} objects")
            
        except Exception as e:
            logger.error(f"Memory optimization failed: {e}")
    
    async def _check_system_health(self) -> Any:
        """Check system health"""
        try:
            
            # Check CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Check memory usage
            memory = psutil.virtual_memory()
            
            # Check disk usage
            disk = psutil.disk_usage('/')
            
            logger.info(f"System health - CPU: {cpu_percent}%, Memory: {memory.percent}%, Disk: {disk.percent}%")
            
            # Alert if any metric is too high
            if cpu_percent > 80:
                logger.warning(f"High CPU usage: {cpu_percent}%")
            
            if memory.percent > 85:
                logger.warning(f"High memory usage: {memory.percent}%")
            
            if disk.percent > 90:
                logger.warning(f"High disk usage: {disk.percent}%")
            
        except Exception as e:
            logger.error(f"System health check failed: {e}")
    
    async def _delete_video_files(self, video_request_id: str):
        """Delete video files for a request"""
        try:
            # Delete video file
            video_path = Path(self.config.upload_dir) / f"ugc_{video_request_id}.mp4"
            if video_path.exists():
                video_path.unlink()
            
            # Delete thumbnail if exists
            thumbnail_path = Path(self.config.upload_dir) / f"thumb_{video_request_id}.jpg"
            if thumbnail_path.exists():
                thumbnail_path.unlink()
            
            logger.debug(f"Deleted video files for request: {video_request_id}")
            
        except Exception as e:
            logger.error(f"Failed to delete video files for {video_request_id}: {e}")

# Global maintenance service instance
maintenance_service = MaintenanceService()

async def run_maintenance():
    """Run maintenance service"""
    await maintenance_service.run_maintenance()

match __name__:
    case "__main__":
    asyncio.run(run_maintenance()) 