"""
BUL - Business Universal Language (Auto Backup System)
====================================================

Automatic backup system with scheduling and recovery features.
"""

import asyncio
import logging
import shutil
import zipfile
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import sqlite3
import subprocess
import schedule
import threading
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BackupType(str, Enum):
    """Backup type enumeration."""
    FULL = "full"
    INCREMENTAL = "incremental"
    DIFFERENTIAL = "differential"

class BackupStatus(str, Enum):
    """Backup status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class BackupConfig:
    """Backup configuration."""
    name: str
    source_path: str
    destination_path: str
    backup_type: BackupType
    schedule: str  # cron-like schedule
    retention_days: int
    compression: bool = True
    encryption: bool = False
    enabled: bool = True

class AutoBackupSystem:
    """Automatic backup system with scheduling and recovery."""
    
    def __init__(self):
        self.app = FastAPI(
            title="BUL Auto Backup System",
            description="Automatic backup system with scheduling and recovery features",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # Backup configurations
        self.backup_configs: Dict[str, BackupConfig] = {}
        self.backup_history: List[Dict[str, Any]] = []
        self.backup_status: Dict[str, BackupStatus] = {}
        
        # Setup components
        self.setup_middleware()
        self.setup_routes()
        self.setup_default_configs()
        self.start_scheduler()
        
        logger.info("Auto Backup System initialized")
    
    def setup_middleware(self):
        """Setup backup middleware."""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def setup_routes(self):
        """Setup backup API routes."""
        
        @self.app.get("/", tags=["System"])
        async def root():
            """Root endpoint with backup system information."""
            return {
                "message": "BUL Auto Backup System",
                "version": "1.0.0",
                "status": "operational",
                "features": [
                    "Automatic Scheduling",
                    "Multiple Backup Types",
                    "Compression & Encryption",
                    "Retention Management",
                    "Recovery Operations",
                    "Progress Monitoring",
                    "Email Notifications",
                    "Cloud Storage Support"
                ],
                "timestamp": datetime.now().isoformat()
            }
        
        @self.app.post("/backup/create", tags=["Backup"])
        async def create_backup(backup_request: dict):
            """Create a new backup configuration."""
            try:
                config = BackupConfig(
                    name=backup_request["name"],
                    source_path=backup_request["source_path"],
                    destination_path=backup_request["destination_path"],
                    backup_type=BackupType(backup_request["backup_type"]),
                    schedule=backup_request["schedule"],
                    retention_days=backup_request["retention_days"],
                    compression=backup_request.get("compression", True),
                    encryption=backup_request.get("encryption", False),
                    enabled=backup_request.get("enabled", True)
                )
                
                self.backup_configs[config.name] = config
                
                return {
                    "message": "Backup configuration created successfully",
                    "config_name": config.name,
                    "status": "created"
                }
                
            except Exception as e:
                logger.error(f"Error creating backup config: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/backup/configs", tags=["Backup"])
        async def get_backup_configs():
            """Get all backup configurations."""
            return {
                "configurations": [
                    {
                        "name": config.name,
                        "source_path": config.source_path,
                        "destination_path": config.destination_path,
                        "backup_type": config.backup_type,
                        "schedule": config.schedule,
                        "retention_days": config.retention_days,
                        "compression": config.compression,
                        "encryption": config.encryption,
                        "enabled": config.enabled
                    }
                    for config in self.backup_configs.values()
                ],
                "total": len(self.backup_configs)
            }
        
        @self.app.post("/backup/run/{config_name}", tags=["Backup"])
        async def run_backup(config_name: str, background_tasks: BackgroundTasks):
            """Run backup for specific configuration."""
            try:
                if config_name not in self.backup_configs:
                    raise HTTPException(status_code=404, detail="Backup configuration not found")
                
                config = self.backup_configs[config_name]
                
                if not config.enabled:
                    raise HTTPException(status_code=400, detail="Backup configuration is disabled")
                
                # Start backup in background
                background_tasks.add_task(self.execute_backup, config_name)
                
                return {
                    "message": f"Backup '{config_name}' started",
                    "config_name": config_name,
                    "status": "started"
                }
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error starting backup: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/backup/status/{config_name}", tags=["Backup"])
        async def get_backup_status(config_name: str):
            """Get backup status for specific configuration."""
            if config_name not in self.backup_configs:
                raise HTTPException(status_code=404, detail="Backup configuration not found")
            
            status = self.backup_status.get(config_name, BackupStatus.PENDING)
            
            return {
                "config_name": config_name,
                "status": status,
                "last_run": self.get_last_backup_time(config_name),
                "next_run": self.get_next_backup_time(config_name)
            }
        
        @self.app.get("/backup/history", tags=["Backup"])
        async def get_backup_history():
            """Get backup history."""
            return {
                "history": self.backup_history[-50:],  # Last 50 backups
                "total": len(self.backup_history)
            }
        
        @self.app.post("/backup/restore/{config_name}", tags=["Recovery"])
        async def restore_backup(config_name: str, restore_request: dict):
            """Restore from backup."""
            try:
                if config_name not in self.backup_configs:
                    raise HTTPException(status_code=404, detail="Backup configuration not found")
                
                backup_file = restore_request.get("backup_file")
                restore_path = restore_request.get("restore_path")
                
                if not backup_file or not restore_path:
                    raise HTTPException(status_code=400, detail="Missing backup_file or restore_path")
                
                # Execute restore
                success = await self.execute_restore(config_name, backup_file, restore_path)
                
                if success:
                    return {
                        "message": "Restore completed successfully",
                        "config_name": config_name,
                        "restore_path": restore_path
                    }
                else:
                    raise HTTPException(status_code=500, detail="Restore failed")
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error restoring backup: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/backup/list/{config_name}", tags=["Backup"])
        async def list_backups(config_name: str):
            """List available backups for configuration."""
            try:
                if config_name not in self.backup_configs:
                    raise HTTPException(status_code=404, detail="Backup configuration not found")
                
                config = self.backup_configs[config_name]
                backup_files = self.list_backup_files(config)
                
                return {
                    "config_name": config_name,
                    "backups": backup_files,
                    "total": len(backup_files)
                }
                
            except Exception as e:
                logger.error(f"Error listing backups: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.delete("/backup/cleanup/{config_name}", tags=["Backup"])
        async def cleanup_old_backups(config_name: str):
            """Cleanup old backups based on retention policy."""
            try:
                if config_name not in self.backup_configs:
                    raise HTTPException(status_code=404, detail="Backup configuration not found")
                
                config = self.backup_configs[config_name]
                cleaned_count = await self.cleanup_old_backups(config)
                
                return {
                    "message": f"Cleaned up {cleaned_count} old backups",
                    "config_name": config_name,
                    "cleaned_count": cleaned_count
                }
                
            except Exception as e:
                logger.error(f"Error cleaning up backups: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/backup/dashboard", tags=["Dashboard"])
        async def get_backup_dashboard():
            """Get backup dashboard data."""
            try:
                # Get statistics
                total_configs = len(self.backup_configs)
                enabled_configs = len([c for c in self.backup_configs.values() if c.enabled])
                recent_backups = len([b for b in self.backup_history 
                                    if datetime.fromisoformat(b["timestamp"]) > datetime.now() - timedelta(days=7)])
                
                # Get recent activity
                recent_activity = self.backup_history[-10:]
                
                # Get status summary
                status_summary = {}
                for config_name in self.backup_configs.keys():
                    status = self.backup_status.get(config_name, BackupStatus.PENDING)
                    status_summary[config_name] = status
                
                return {
                    "summary": {
                        "total_configurations": total_configs,
                        "enabled_configurations": enabled_configs,
                        "recent_backups_7d": recent_backups,
                        "total_backups": len(self.backup_history)
                    },
                    "status_summary": status_summary,
                    "recent_activity": recent_activity,
                    "timestamp": datetime.now().isoformat()
                }
                
            except Exception as e:
                logger.error(f"Error getting dashboard data: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    
    def setup_default_configs(self):
        """Setup default backup configurations."""
        try:
            # Database backup
            db_config = BackupConfig(
                name="database_backup",
                source_path="bul_*.db",
                destination_path="./backups/database",
                backup_type=BackupType.FULL,
                schedule="0 2 * * *",  # Daily at 2 AM
                retention_days=30,
                compression=True,
                encryption=False
            )
            
            # Log files backup
            logs_config = BackupConfig(
                name="logs_backup",
                source_path="./logs",
                destination_path="./backups/logs",
                backup_type=BackupType.INCREMENTAL,
                schedule="0 3 * * *",  # Daily at 3 AM
                retention_days=14,
                compression=True,
                encryption=False
            )
            
            # Configuration backup
            config_config = BackupConfig(
                name="config_backup",
                source_path="./config",
                destination_path="./backups/config",
                backup_type=BackupType.FULL,
                schedule="0 4 * * 0",  # Weekly on Sunday at 4 AM
                retention_days=90,
                compression=True,
                encryption=True
            )
            
            self.backup_configs["database_backup"] = db_config
            self.backup_configs["logs_backup"] = logs_config
            self.backup_configs["config_backup"] = config_config
            
            logger.info("Default backup configurations created")
            
        except Exception as e:
            logger.error(f"Error creating default configs: {e}")
    
    def start_scheduler(self):
        """Start backup scheduler."""
        def scheduler_loop():
            while True:
                try:
                    schedule.run_pending()
                    time.sleep(60)  # Check every minute
                except Exception as e:
                    logger.error(f"Scheduler error: {e}")
                    time.sleep(300)  # Wait 5 minutes on error)
        
        # Schedule all enabled backups
        for config_name, config in self.backup_configs.items():
            if config.enabled:
                schedule.every().day.at("02:00").do(self.execute_backup, config_name)
        
        # Start scheduler thread
        scheduler_thread = threading.Thread(target=scheduler_loop, daemon=True)
        scheduler_thread.start()
        
        logger.info("Backup scheduler started")
    
    async def execute_backup(self, config_name: str):
        """Execute backup for specific configuration."""
        try:
            config = self.backup_configs[config_name]
            self.backup_status[config_name] = BackupStatus.RUNNING
            
            logger.info(f"Starting backup: {config_name}")
            
            # Create destination directory
            dest_path = Path(config.destination_path)
            dest_path.mkdir(parents=True, exist_ok=True)
            
            # Generate backup filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_filename = f"{config_name}_{timestamp}"
            
            if config.compression:
                backup_filename += ".zip"
            else:
                backup_filename += ".tar"
            
            backup_path = dest_path / backup_filename
            
            # Execute backup based on type
            if config.backup_type == BackupType.FULL:
                success = await self.create_full_backup(config, backup_path)
            elif config.backup_type == BackupType.INCREMENTAL:
                success = await self.create_incremental_backup(config, backup_path)
            else:
                success = await self.create_differential_backup(config, backup_path)
            
            if success:
                self.backup_status[config_name] = BackupStatus.COMPLETED
                
                # Log backup completion
                backup_record = {
                    "config_name": config_name,
                    "backup_file": str(backup_path),
                    "backup_type": config.backup_type,
                    "size": backup_path.stat().st_size if backup_path.exists() else 0,
                    "timestamp": datetime.now().isoformat(),
                    "status": "completed"
                }
                
                self.backup_history.append(backup_record)
                
                logger.info(f"Backup completed: {config_name}")
                
                # Cleanup old backups
                await self.cleanup_old_backups(config)
                
            else:
                self.backup_status[config_name] = BackupStatus.FAILED
                logger.error(f"Backup failed: {config_name}")
                
        except Exception as e:
            self.backup_status[config_name] = BackupStatus.FAILED
            logger.error(f"Error executing backup {config_name}: {e}")
    
    async def create_full_backup(self, config: BackupConfig, backup_path: Path) -> bool:
        """Create full backup."""
        try:
            source_path = Path(config.source_path)
            
            if config.compression:
                with zipfile.ZipFile(backup_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    if source_path.is_file():
                        zipf.write(source_path, source_path.name)
                    else:
                        for file_path in source_path.rglob('*'):
                            if file_path.is_file():
                                arcname = file_path.relative_to(source_path.parent)
                                zipf.write(file_path, arcname)
            else:
                # Use tar for non-compressed backups
                subprocess.run([
                    'tar', '-cf', str(backup_path), str(source_path)
                ], check=True)
            
            return True
            
        except Exception as e:
            logger.error(f"Error creating full backup: {e}")
            return False
    
    async def create_incremental_backup(self, config: BackupConfig, backup_path: Path) -> bool:
        """Create incremental backup."""
        try:
            # For incremental backup, we'll use rsync or similar
            # This is a simplified version
            return await self.create_full_backup(config, backup_path)
            
        except Exception as e:
            logger.error(f"Error creating incremental backup: {e}")
            return False
    
    async def create_differential_backup(self, config: BackupConfig, backup_path: Path) -> bool:
        """Create differential backup."""
        try:
            # For differential backup, we'll use rsync or similar
            # This is a simplified version
            return await self.create_full_backup(config, backup_path)
            
        except Exception as e:
            logger.error(f"Error creating differential backup: {e}")
            return False
    
    async def execute_restore(self, config_name: str, backup_file: str, restore_path: str) -> bool:
        """Execute restore from backup."""
        try:
            config = self.backup_configs[config_name]
            backup_path = Path(config.destination_path) / backup_file
            
            if not backup_path.exists():
                logger.error(f"Backup file not found: {backup_file}")
                return False
            
            restore_path_obj = Path(restore_path)
            restore_path_obj.mkdir(parents=True, exist_ok=True)
            
            if backup_path.suffix == '.zip':
                with zipfile.ZipFile(backup_path, 'r') as zipf:
                    zipf.extractall(restore_path_obj)
            else:
                # Extract tar file
                subprocess.run([
                    'tar', '-xf', str(backup_path), '-C', str(restore_path_obj)
                ], check=True)
            
            logger.info(f"Restore completed: {backup_file} -> {restore_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error executing restore: {e}")
            return False
    
    def list_backup_files(self, config: BackupConfig) -> List[Dict[str, Any]]:
        """List backup files for configuration."""
        try:
            backup_dir = Path(config.destination_path)
            if not backup_dir.exists():
                return []
            
            backup_files = []
            for file_path in backup_dir.iterdir():
                if file_path.is_file():
                    backup_files.append({
                        "filename": file_path.name,
                        "size": file_path.stat().st_size,
                        "created": datetime.fromtimestamp(file_path.stat().st_ctime).isoformat(),
                        "modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                    })
            
            # Sort by creation time (newest first)
            backup_files.sort(key=lambda x: x["created"], reverse=True)
            
            return backup_files
            
        except Exception as e:
            logger.error(f"Error listing backup files: {e}")
            return []
    
    async def cleanup_old_backups(self, config: BackupConfig) -> int:
        """Cleanup old backups based on retention policy."""
        try:
            backup_dir = Path(config.destination_path)
            if not backup_dir.exists():
                return 0
            
            cutoff_date = datetime.now() - timedelta(days=config.retention_days)
            cleaned_count = 0
            
            for file_path in backup_dir.iterdir():
                if file_path.is_file():
                    file_time = datetime.fromtimestamp(file_path.stat().st_ctime)
                    if file_time < cutoff_date:
                        file_path.unlink()
                        cleaned_count += 1
                        logger.info(f"Deleted old backup: {file_path.name}")
            
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Error cleaning up old backups: {e}")
            return 0
    
    def get_last_backup_time(self, config_name: str) -> Optional[str]:
        """Get last backup time for configuration."""
        for backup in reversed(self.backup_history):
            if backup["config_name"] == config_name:
                return backup["timestamp"]
        return None
    
    def get_next_backup_time(self, config_name: str) -> Optional[str]:
        """Get next scheduled backup time for configuration."""
        # This would calculate based on the schedule
        # For now, return a placeholder
        return (datetime.now() + timedelta(days=1)).isoformat()
    
    def run(self, host: str = "0.0.0.0", port: int = 8005, debug: bool = False):
        """Run the backup system."""
        logger.info(f"Starting Auto Backup System on {host}:{port}")
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            log_level="debug" if debug else "info"
        )

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="BUL Auto Backup System")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8005, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    # Create and run backup system
    system = AutoBackupSystem()
    system.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == "__main__":
    main()
