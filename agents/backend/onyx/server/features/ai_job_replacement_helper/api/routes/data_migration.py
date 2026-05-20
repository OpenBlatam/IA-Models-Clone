"""
Data Migration endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.data_migration import DataMigrationService

router = APIRouter()
migration_service = DataMigrationService()


@router.post("/run/{migration_id}")
async def run_migration(
    migration_id: str,
    dry_run: bool = False
) -> Dict[str, Any]:
    """Ejecutar migración"""
    try:
        result = await migration_service.run_migration(migration_id, dry_run)
        return {
            "migration_id": result.migration_id,
            "status": result.status.value,
            "records_processed": result.records_processed,
            "duration_seconds": result.duration_seconds,
            "error_message": result.error_message,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rollback/{migration_id}")
async def rollback_migration(migration_id: str) -> Dict[str, Any]:
    """Revertir migración"""
    try:
        result = await migration_service.rollback_migration(migration_id)
        return {
            "migration_id": result.migration_id,
            "status": result.status.value,
            "duration_seconds": result.duration_seconds,
            "error_message": result.error_message,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history")
async def get_migration_history() -> Dict[str, Any]:
    """Obtener historial de migraciones"""
    try:
        history = migration_service.get_migration_history()
        return {
            "history": history,
            "total": len(history),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




