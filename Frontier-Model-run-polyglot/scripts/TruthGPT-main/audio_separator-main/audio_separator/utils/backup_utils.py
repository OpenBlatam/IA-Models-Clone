"""
Backup and restore utilities.
"""

import shutil
import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

from ..logger import logger


def backup_separation_results(
    results_dir: str,
    backup_dir: str,
    include_metadata: bool = True
) -> str:
    """
    Backup separation results to a backup directory.
    
    Args:
        results_dir: Directory containing separation results
        backup_dir: Backup directory
        include_metadata: Include metadata files
        
    Returns:
        Path to backup directory
    """
    results_dir = Path(results_dir)
    backup_dir = Path(backup_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"backup_{timestamp}"
    backup_path.mkdir(parents=True, exist_ok=True)
    
    # Copy files
    if results_dir.exists():
        shutil.copytree(results_dir, backup_path / "results", dirs_exist_ok=True)
        logger.info(f"Backed up results to {backup_path}")
    
    # Save metadata
    if include_metadata:
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "source_dir": str(results_dir),
            "backup_dir": str(backup_path)
        }
        
        metadata_path = backup_path / "backup_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    return str(backup_path)


def restore_separation_results(
    backup_dir: str,
    restore_dir: str,
    overwrite: bool = False
):
    """
    Restore separation results from backup.
    
    Args:
        backup_dir: Backup directory
        restore_dir: Directory to restore to
        overwrite: Overwrite existing files
    """
    backup_dir = Path(backup_dir)
    restore_dir = Path(restore_dir)
    
    if not backup_dir.exists():
        raise FileNotFoundError(f"Backup directory not found: {backup_dir}")
    
    results_backup = backup_dir / "results"
    
    if not results_backup.exists():
        raise FileNotFoundError(f"Results not found in backup: {results_backup}")
    
    if restore_dir.exists() and not overwrite:
        raise FileExistsError(f"Restore directory exists: {restore_dir}")
    
    restore_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(results_backup, restore_dir, dirs_exist_ok=overwrite)
    
    logger.info(f"Restored results to {restore_dir}")


def list_backups(backup_dir: str) -> List[Dict[str, str]]:
    """
    List available backups.
    
    Args:
        backup_dir: Backup directory
        
    Returns:
        List of backup information dictionaries
    """
    backup_dir = Path(backup_dir)
    
    if not backup_dir.exists():
        return []
    
    backups = []
    
    for backup_path in backup_dir.glob("backup_*"):
        if backup_path.is_dir():
            metadata_path = backup_path / "backup_metadata.json"
            
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    backups.append({
                        "path": str(backup_path),
                        "timestamp": metadata.get("timestamp", ""),
                        "source_dir": metadata.get("source_dir", "")
                    })
            else:
                backups.append({
                    "path": str(backup_path),
                    "timestamp": backup_path.name.replace("backup_", ""),
                    "source_dir": ""
                })
    
    # Sort by timestamp (newest first)
    backups.sort(key=lambda x: x["timestamp"], reverse=True)
    
    return backups

