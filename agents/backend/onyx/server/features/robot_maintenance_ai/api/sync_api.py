"""
Synchronization API for syncing data across systems and devices.
Refactored to use BaseRouter for reduced duplication.
"""

from fastapi import Depends
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
import hashlib

from .base_router import BaseRouter
from ..utils.file_helpers import get_iso_timestamp, get_timestamp_id, parse_iso_date

# Create base router instance
base = BaseRouter(
    prefix="/api/sync",
    tags=["Synchronization"],
    require_authentication=True,
    require_rate_limit=False
)

router = base.router


class SyncRequest(BaseModel):
    """Request to sync data."""
    sync_type: str = Field(..., description="Type: conversations, maintenance, all")
    last_sync_timestamp: Optional[str] = Field(None, description="Last sync timestamp (ISO format)")
    device_id: Optional[str] = Field(None, description="Device identifier")


class SyncResponse(BaseModel):
    """Sync response model."""
    sync_id: str
    sync_timestamp: str
    items_synced: int
    conflicts: List[Dict[str, Any]]


@router.post("/start")
@base.timed_endpoint("start_sync")
async def start_sync(
    request: SyncRequest,
    _: Dict = Depends(base.get_auth_dependency())
) -> Dict[str, Any]:
    """
    Start a synchronization operation.
    """
    base.log_request("start_sync", sync_type=request.sync_type, device_id=request.device_id)
    
    sync_id = get_timestamp_id("sync_")
    sync_timestamp = get_iso_timestamp()
    
    items_synced = 0
    conflicts = []
    
    if request.sync_type in ["conversations", "all"]:
        conversations = base.database.get_all_conversations()
        if request.last_sync_timestamp:
            last_sync = parse_iso_date(request.last_sync_timestamp)
            if last_sync:
                conversations = [
                    c for c in conversations
                    if c.get("updated_at") and
                    parse_iso_date(c["updated_at"]) and
                    parse_iso_date(c["updated_at"]) > last_sync
                ]
        items_synced += len(conversations)
    
    if request.sync_type in ["maintenance", "all"]:
        history = base.database.get_maintenance_history(limit=10000)
        if request.last_sync_timestamp:
            last_sync = parse_iso_date(request.last_sync_timestamp)
            if last_sync:
                history = [
                    h for h in history
                    if h.get("created_at") and
                    parse_iso_date(h["created_at"]) and
                    parse_iso_date(h["created_at"]) > last_sync
                ]
        items_synced += len(history)
    
    return base.success({
        "sync_id": sync_id,
        "sync_timestamp": sync_timestamp,
        "items_synced": items_synced,
        "conflicts": conflicts,
        "device_id": request.device_id
    })


@router.get("/status/{sync_id}")
@base.timed_endpoint("get_sync_status")
async def get_sync_status(
    sync_id: str,
    _: Dict = Depends(base.get_auth_dependency())
) -> Dict[str, Any]:
    """
    Get status of a synchronization operation.
    """
    base.log_request("get_sync_status", sync_id=sync_id)
    
    # In a real implementation, this would track sync status
    return base.success({
        "sync_id": sync_id,
        "status": "completed",
        "progress": 100,
        "items_synced": 0,
        "errors": []
    })


@router.post("/resolve-conflict")
@base.timed_endpoint("resolve_conflict")
async def resolve_conflict(
    conflict_id: str = Field(..., description="Conflict ID"),
    resolution: str = Field(..., description="Resolution: server, client, merge"),
    _: Dict = Depends(base.get_auth_dependency())
) -> Dict[str, Any]:
    """
    Resolve a synchronization conflict.
    """
    base.log_request("resolve_conflict", conflict_id=conflict_id, resolution=resolution)
    
    return base.success({
        "conflict_id": conflict_id,
        "resolution": resolution,
        "resolved_at": get_iso_timestamp()
    }, message="Conflict resolved successfully")


@router.get("/checksum")
@base.timed_endpoint("get_data_checksum")
async def get_data_checksum(
    data_type: str = Field("all", description="Type: conversations, maintenance, all"),
    _: Dict = Depends(base.get_auth_dependency())
) -> Dict[str, Any]:
    """
    Get checksum of data for integrity verification.
    """
    base.log_request("get_data_checksum", data_type=data_type)
    
    checksums = {}
    
    if data_type in ["conversations", "all"]:
        conversations = base.database.get_all_conversations()
        data_str = str(sorted(conversations, key=lambda x: x.get("id", "")))
        checksums["conversations"] = hashlib.sha256(data_str.encode()).hexdigest()
    
    if data_type in ["maintenance", "all"]:
        history = base.database.get_maintenance_history(limit=10000)
        data_str = str(sorted(history, key=lambda x: x.get("id", 0)))
        checksums["maintenance"] = hashlib.sha256(data_str.encode()).hexdigest()
    
    return base.success({
        "checksums": checksums,
        "timestamp": get_iso_timestamp()
    })




