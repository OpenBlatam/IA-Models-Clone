"""Collaboration endpoints"""
from fastapi import APIRouter
from utils.collaboration import get_collaboration_tracker

router = APIRouter(prefix="/collaboration", tags=["Collaboration"])


@router.get("/{document_path:path}")
async def get_collaboration_info(document_path: str):
    """Get collaboration information for document"""
    tracker = get_collaboration_tracker()
    info = tracker.get_collaboration_info(document_path)
    
    return info

