"""
Backup System for AI Document Processor
Real, working backup and recovery features for document processing
"""

import asyncio
import logging
import json
import time
import os
import shutil
import zipfile
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import hashlib

logger = logging.getLogger(__name__)

class BackupSystem:
    """Real working backup system for AI document processing"""
    
    def __init__(self):
        self.backup_dir = "backups"
        self.max_backups = 10
        self.backup_interval = 3600  # 1 hour in seconds
        
        # Backup stats
        self.stats = {
            "total_backups": 0,
            "successful_backups": 0,
            "failed_backups": 0,
            "last_backup_time": None,
            "start_time": time.time()
        }
        
        # Ensure backup directory exists
        os.makedirs(self.backup_dir, exist_ok=True)
    
    async def create_backup(self, backup_type: str = "full", 
                          include_data: bool = True) -> Dict[str, Any]:
        """Create backup of system data"""
        try:
            backup_id = self._generate_backup_id()
            backup_timestamp = datetime.now().isoformat()
            
            backup_info = {
                "backup_id": backup_id,
                "type": backup_type,
                "timestamp": backup_timestamp,
                "status": "in_progress",
                "files_backed_up": [],
                "data_backed_up": {},
                "backup_size": 0,
                "checksum": ""
            }
            
            # Create backup directory
            backup_path = os.path.join(self.backup_dir, backup_id)
            os.makedirs(backup_path, exist_ok=True)
            
            # Backup configuration files
            config_files = [
                "real_working_requirements.txt",
                "requirements.txt"
            ]
            
            for config_file in config_files:
                if os.path.exists(config_file):
                    shutil.copy2(config_file, backup_path)
                    backup_info["files_backed_up"].append(config_file)
            
            # Backup application files
            app_files = [
                "real_working_processor.py",
                "advanced_real_processor.py",
                "document_upload_processor.py",
                "monitoring_system.py",
                "security_system.py",
                "notification_system.py",
                "analytics_system.py"
            ]
            
            for app_file in app_files:
                if os.path.exists(app_file):
                    shutil.copy2(app_file, backup_path)
                    backup_info["files_backed_up"].append(app_file)
            
            # Backup route files
            route_files = [
                "improved_real_routes.py",
                "advanced_real_routes.py",
                "upload_routes.py",
                "monitoring_routes.py",
                "security_routes.py",
                "notification_routes.py",
                "analytics_routes.py"
            ]
            
            for route_file in route_files:
                if os.path.exists(route_file):
                    shutil.copy2(route_file, backup_path)
                    backup_info["files_backed_up"].append(route_file)
            
            # Backup application files
            app_files = [
                "improved_real_app.py",
                "complete_real_app.py",
                "ultimate_real_app.py",
                "final_real_app.py"
            ]
            
            for app_file in app_files:
                if os.path.exists(app_file):
                    shutil.copy2(app_file, backup_path)
                    backup_info["files_backed_up"].append(app_file)
            
            # Backup documentation
            doc_files = [
                "README_REAL.md",
                "README_IMPROVED.md",
                "README_COMPLETE_REAL.md",
                "README_ULTIMATE_REAL.md",
                "README_FINAL_REAL.md"
            ]
            
            for doc_file in doc_files:
                if os.path.exists(doc_file):
                    shutil.copy2(doc_file, backup_path)
                    backup_info["files_backed_up"].append(doc_file)
            
            # Backup installation scripts
            install_files = [
                "install_real.py",
                "install_improved.py",
                "install_complete_real.py",
                "install_ultimate_real.py"
            ]
            
            for install_file in install_files:
                if os.path.exists(install_file):
                    shutil.copy2(install_file, backup_path)
                    backup_info["files_backed_up"].append(install_file)
            
            # Backup data if requested
            if include_data:
                data_backup = await self._backup_data(backup_path)
                backup_info["data_backed_up"] = data_backup
            
            # Create backup archive
            archive_path = f"{backup_path}.zip"
            with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(backup_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, backup_path)
                        zipf.write(file_path, arcname)
            
            # Calculate backup size and checksum
            backup_size = os.path.getsize(archive_path)
            backup_info["backup_size"] = backup_size
            
            with open(archive_path, 'rb') as f:
                backup_info["checksum"] = hashlib.md5(f.read()).hexdigest()
            
            # Clean up temporary directory
            shutil.rmtree(backup_path)
            
            # Update backup info
            backup_info["status"] = "completed"
            backup_info["archive_path"] = archive_path
            
            # Save backup metadata
            metadata_path = f"{archive_path}.json"
            with open(metadata_path, 'w') as f:
                json.dump(backup_info, f, indent=2)
            
            # Update stats
            self.stats["total_backups"] += 1
            self.stats["successful_backups"] += 1
            self.stats["last_backup_time"] = backup_timestamp
            
            # Clean old backups
            await self._clean_old_backups()
            
            return backup_info
            
        except Exception as e:
            self.stats["failed_backups"] += 1
            logger.error(f"Error creating backup: {e}")
            return {
                "error": str(e),
                "status": "failed",
                "timestamp": datetime.now().isoformat()
            }
    
    async def _backup_data(self, backup_path: str) -> Dict[str, Any]:
        """Backup system data"""
        try:
            data_backup = {
                "timestamp": datetime.now().isoformat(),
                "data_files": []
            }
            
            # Backup cache files if they exist
            cache_files = ["cache.json", "analytics.json", "logs.json"]
            for cache_file in cache_files:
                if os.path.exists(cache_file):
                    shutil.copy2(cache_file, backup_path)
                    data_backup["data_files"].append(cache_file)
            
            # Backup configuration data
            config_data = {
                "backup_timestamp": datetime.now().isoformat(),
                "system_info": {
                    "python_version": os.sys.version,
                    "platform": os.name,
                    "working_directory": os.getcwd()
                }
            }
            
            config_file_path = os.path.join(backup_path, "system_config.json")
            with open(config_file_path, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            data_backup["data_files"].append("system_config.json")
            
            return data_backup
            
        except Exception as e:
            logger.error(f"Error backing up data: {e}")
            return {"error": str(e)}
    
    async def restore_backup(self, backup_id: str, restore_path: str = ".") -> Dict[str, Any]:
        """Restore from backup"""
        try:
            # Find backup archive
            archive_path = os.path.join(self.backup_dir, f"{backup_id}.zip")
            metadata_path = f"{archive_path}.json"
            
            if not os.path.exists(archive_path):
                return {"error": f"Backup {backup_id} not found"}
            
            # Load backup metadata
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    backup_info = json.load(f)
            else:
                backup_info = {"backup_id": backup_id}
            
            # Extract backup
            with zipfile.ZipFile(archive_path, 'r') as zipf:
                zipf.extractall(restore_path)
            
            return {
                "status": "completed",
                "backup_id": backup_id,
                "restore_path": restore_path,
                "files_restored": backup_info.get("files_backed_up", []),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error restoring backup: {e}")
            return {"error": str(e)}
    
    async def list_backups(self) -> List[Dict[str, Any]]:
        """List available backups"""
        try:
            backups = []
            
            for filename in os.listdir(self.backup_dir):
                if filename.endswith('.zip'):
                    backup_id = filename[:-4]  # Remove .zip extension
                    metadata_path = os.path.join(self.backup_dir, f"{backup_id}.zip.json")
                    
                    backup_info = {
                        "backup_id": backup_id,
                        "archive_path": os.path.join(self.backup_dir, filename),
                        "file_size": os.path.getsize(os.path.join(self.backup_dir, filename)),
                        "created_time": datetime.fromtimestamp(
                            os.path.getctime(os.path.join(self.backup_dir, filename))
                        ).isoformat()
                    }
                    
                    # Load metadata if available
                    if os.path.exists(metadata_path):
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                            backup_info.update(metadata)
                    
                    backups.append(backup_info)
            
            # Sort by creation time (newest first)
            backups.sort(key=lambda x: x.get("created_time", ""), reverse=True)
            
            return backups
            
        except Exception as e:
            logger.error(f"Error listing backups: {e}")
            return []
    
    async def delete_backup(self, backup_id: str) -> Dict[str, Any]:
        """Delete backup"""
        try:
            archive_path = os.path.join(self.backup_dir, f"{backup_id}.zip")
            metadata_path = f"{archive_path}.json"
            
            if not os.path.exists(archive_path):
                return {"error": f"Backup {backup_id} not found"}
            
            # Delete archive and metadata
            os.remove(archive_path)
            if os.path.exists(metadata_path):
                os.remove(metadata_path)
            
            return {
                "status": "deleted",
                "backup_id": backup_id,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error deleting backup: {e}")
            return {"error": str(e)}
    
    async def _clean_old_backups(self):
        """Clean old backups to maintain max_backups limit"""
        try:
            backups = await self.list_backups()
            
            if len(backups) > self.max_backups:
                # Delete oldest backups
                backups_to_delete = backups[self.max_backups:]
                for backup in backups_to_delete:
                    await self.delete_backup(backup["backup_id"])
                    logger.info(f"Deleted old backup: {backup['backup_id']}")
            
        except Exception as e:
            logger.error(f"Error cleaning old backups: {e}")
    
    def _generate_backup_id(self) -> str:
        """Generate unique backup ID"""
        timestamp = int(time.time())
        return f"backup_{timestamp}"
    
    def get_backup_stats(self) -> Dict[str, Any]:
        """Get backup statistics"""
        uptime = time.time() - self.stats["start_time"]
        return {
            "stats": self.stats.copy(),
            "uptime_seconds": round(uptime, 2),
            "uptime_hours": round(uptime / 3600, 2),
            "backup_directory": self.backup_dir,
            "max_backups": self.max_backups,
            "backup_interval": self.backup_interval
        }
    
    def get_backup_config(self) -> Dict[str, Any]:
        """Get backup configuration"""
        return {
            "backup_directory": self.backup_dir,
            "max_backups": self.max_backups,
            "backup_interval": self.backup_interval,
            "features": {
                "full_backup": True,
                "incremental_backup": False,
                "data_backup": True,
                "config_backup": True,
                "app_backup": True,
                "doc_backup": True,
                "install_backup": True,
                "automatic_cleanup": True
            }
        }

# Global instance
backup_system = BackupSystem()













