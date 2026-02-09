"""
Admin API endpoints for system management.
Refactored to use BaseRouter for reduced duplication.
"""

from fastapi import Depends
from pydantic import BaseModel, Field
from typing import Dict, Any

from .base_router import BaseRouter
from .exceptions import ValidationError
from ..core.plugin_system import PluginManager
from ..utils.backup_utils import backup_database, backup_conversations, restore_conversations, list_backups
from ..utils.metrics import metrics_collector
from .dependencies import get_rate_limiter, get_conversation_manager

# Create base router instance
base = BaseRouter(
    prefix="/api/admin",
    tags=["Admin"],
    require_authentication=True,
    require_rate_limit=False
)

router = base.router

plugin_manager = PluginManager()


class BackupRequest(BaseModel):
    """Request to create backup."""
    backup_type: str = Field(..., description="Type of backup: 'database' or 'conversations'")


class RestoreRequest(BaseModel):
    """Request to restore from backup."""
    backup_file: str = Field(..., description="Path to backup file")


@router.post("/backup")
@base.timed_endpoint("create_backup")
async def create_backup(
    request: BackupRequest,
    _: Dict = Depends(base.get_auth_dependency())
) -> Dict[str, Any]:
    """Create a backup."""
    base.log_request("create_backup", backup_type=request.backup_type)
    
    if request.backup_type == "database":
        backup_file = backup_database("data/maintenance.db")
        return base.success({
            "backup_file": backup_file,
            "type": "database"
        })
    elif request.backup_type == "conversations":
        conversations = base.conversation_manager.conversations
        backup_file = backup_conversations(conversations)
        return base.success({
            "backup_file": backup_file,
            "type": "conversations"
        })
    else:
        raise ValidationError("Invalid backup type. Use 'database' or 'conversations'")


@router.post("/restore")
@base.timed_endpoint("restore_backup")
async def restore_backup(
    request: RestoreRequest,
    _: Dict = Depends(base.get_auth_dependency())
) -> Dict[str, Any]:
    """Restore from backup."""
    base.log_request("restore_backup", backup_file=request.backup_file)
    
    if request.backup_file.endswith(".json"):
        conversations = restore_conversations(request.backup_file)
        base.conversation_manager.conversations.update(conversations)
        return base.success(None, message=f"Restored {len(conversations)} conversations")
    else:
        raise ValidationError("Only JSON conversation backups can be restored via API")


@router.get("/backups")
@base.timed_endpoint("list_backups")
async def list_backups_endpoint(
    _: Dict = Depends(base.get_auth_dependency())
) -> Dict[str, Any]:
    """List available backups."""
    base.log_request("list_backups")
    
    backups = list_backups()
    return base.success({
        "backups": backups,
        "count": len(backups)
    })


@router.get("/plugins")
@base.timed_endpoint("list_plugins")
async def list_plugins(
    _: Dict = Depends(base.get_auth_dependency())
) -> Dict[str, Any]:
    """List registered plugins."""
    base.log_request("list_plugins")
    
    plugins = plugin_manager.list_plugins()
    return base.success({
        "plugins": plugins,
        "count": len(plugins)
    })


@router.get("/system-stats")
@base.timed_endpoint("get_system_stats")
async def get_system_stats(
    _: Dict = Depends(base.get_auth_dependency()),
    rate_limiter = Depends(get_rate_limiter),
    conversation_manager = Depends(get_conversation_manager)
) -> Dict[str, Any]:
    """Get comprehensive system statistics."""
    base.log_request("get_system_stats")
    
    stats = {
        "metrics": metrics_collector.get_stats(),
        "rate_limiting": rate_limiter.get_stats(),
        "conversations": {
            "active": len(conversation_manager.conversations),
            "total_messages": sum(
                len(msgs) for msgs in conversation_manager.conversations.values()
            )
        },
        "database": {
            "path": str(base.database.db_path),
            "exists": base.database.db_path.exists()
        },
        "plugins": {
            "count": len(plugin_manager.plugins),
            "hooks": len(plugin_manager.hooks)
        }
    }
    
    return base.success(stats)






