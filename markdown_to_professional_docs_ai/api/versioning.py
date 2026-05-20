"""Document versioning endpoints"""
from fastapi import APIRouter, HTTPException, Form
from typing import Optional
from utils.document_versioning import get_document_versioning

router = APIRouter(prefix="/version", tags=["Versioning"])


@router.post("/create")
async def create_version(
    document_path: str = Form(...),
    description: Optional[str] = Form(None)
):
    """Create a new version of a document"""
    versioning = get_document_versioning()
    version = versioning.create_version(document_path, description)
    
    return {
        "status": "success",
        "version_id": version["version_id"],
        "document_path": document_path
    }


@router.get("/{version_id}")
async def get_version(version_id: str):
    """Get version information"""
    versioning = get_document_versioning()
    version = versioning.get_version(version_id)
    
    if not version:
        raise HTTPException(status_code=404, detail="Version not found")
    
    return version


@router.get("s")
async def list_versions(
    document_path: Optional[str] = None,
    limit: int = 100
):
    """List all versions"""
    versioning = get_document_versioning()
    versions = versioning.list_versions(document_path, limit)
    
    return {
        "versions": versions,
        "total": len(versions)
    }


@router.post("/{version_id}/restore")
async def restore_version(version_id: str):
    """Restore a document to a specific version"""
    versioning = get_document_versioning()
    result = versioning.restore_version(version_id)
    
    if result:
        return {
            "status": "success",
            "version_id": version_id,
            "restored_path": result
        }
    else:
        raise HTTPException(status_code=400, detail="Failed to restore version")

