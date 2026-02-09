"""
Backup and restore utilities.
"""

import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from .file_helpers import datetime_to_iso
import logging

from .file_helpers import (
    ensure_directory_exists,
    get_timestamp_filename,
    write_json_file,
    read_json_file,
    get_iso_timestamp
)

logger = logging.getLogger(__name__)


def backup_database(db_path: str, backup_dir: str = "backups") -> str:
    """
    Backup SQLite database.
    
    Args:
        db_path: Path to database file
        backup_dir: Directory to store backups
    
    Returns:
        Path to backup file
    """
    db_file = Path(db_path)
    if not db_file.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")
    
    backup_file = get_timestamp_filename("maintenance_db", "db", backup_dir)
    ensure_directory_exists(backup_file)
    
    shutil.copy2(db_file, backup_file)
    logger.info(f"Database backed up to {backup_file}")
    
    return backup_file


def backup_conversations(
    conversations: Dict[str, List[Dict[str, Any]]],
    backup_dir: str = "backups"
) -> str:
    """
    Backup conversations to JSON.
    
    Args:
        conversations: Dictionary of conversations
        backup_dir: Directory to store backups
    
    Returns:
        Path to backup file
    """
    backup_file = get_timestamp_filename("conversations", "json", backup_dir)
    
    backup_data = {
        "backup_timestamp": get_iso_timestamp(),
        "conversation_count": len(conversations),
        "conversations": conversations
    }
    
    write_json_file(backup_data, backup_file)
    logger.info(f"Conversations backed up to {backup_file}")
    return backup_file


def restore_conversations(backup_file: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Restore conversations from backup.
    
    Args:
        backup_file: Path to backup file
    
    Returns:
        Dictionary of conversations
    """
    backup_data = read_json_file(backup_file)
    logger.info(f"Conversations restored from {backup_file}")
    return backup_data.get("conversations", {})


def list_backups(backup_dir: str = "backups") -> List[Dict[str, Any]]:
    """
    List available backups.
    
    Args:
        backup_dir: Directory containing backups
    
    Returns:
        List of backup information
    """
    backup_path = Path(backup_dir)
    if not backup_path.exists():
        return []
    
    backups = []
    for file in backup_path.glob("*.db"):
        stat = file.stat()
        backups.append({
            "type": "database",
            "file": str(file),
            "size": stat.st_size,
            "created": datetime_to_iso(datetime.fromtimestamp(stat.st_mtime))
        })
    
    for file in backup_path.glob("*.json"):
        stat = file.stat()
        backups.append({
            "type": "conversations",
            "file": str(file),
            "size": stat.st_size,
            "created": datetime_to_iso(datetime.fromtimestamp(stat.st_mtime))
        })
    
    return sorted(backups, key=lambda x: x["created"], reverse=True)






