"""Annotations endpoints"""
from fastapi import APIRouter, HTTPException, Form
from typing import Optional
from utils.annotations import get_annotation_manager

router = APIRouter(prefix="/annotations", tags=["Annotations"])


@router.post("/add")
async def add_annotation(
    document_path: str = Form(...),
    annotation_type: str = Form(...),  # comment, highlight, note
    content: str = Form(...),
    position: Optional[str] = Form(None),  # JSON string
    author: Optional[str] = Form(None)
):
    """Add annotation to document"""
    import json
    
    annotation_manager = get_annotation_manager()
    
    position_dict = None
    if position:
        try:
            position_dict = json.loads(position)
        except json.JSONDecodeError:
            pass
    
    annotation = annotation_manager.add_annotation(
        document_path,
        annotation_type,
        content,
        position_dict,
        author
    )
    
    return {
        "status": "success",
        "annotation_id": annotation.get("id"),
        "document_path": document_path
    }


@router.get("/{document_path:path}")
async def get_annotations(document_path: str):
    """Get annotations for document"""
    annotation_manager = get_annotation_manager()
    annotations = annotation_manager.get_annotations(document_path)
    
    return {
        "document_path": document_path,
        "annotations": annotations,
        "total": len(annotations)
    }

