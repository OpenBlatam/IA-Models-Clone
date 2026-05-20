"""Document validation endpoints"""
from fastapi import APIRouter, HTTPException, Form
from utils.document_validator import get_document_validator
from utils.document_compressor import get_document_compressor

router = APIRouter(prefix="/validate", tags=["Validation"])


@router.post("")
async def validate_document(
    document_path: str = Form(...)
):
    """Validate document"""
    validator = get_document_validator()
    validation = validator.validate_document(document_path)
    
    return validation


@router.post("/compress")
async def compress_document(
    document_path: str = Form(...),
    compression_level: str = Form("medium")
):
    """Compress document"""
    compressor = get_document_compressor()
    compressed_path = compressor.compress_document(document_path)
    
    return {
        "status": "success",
        "original_path": document_path,
        "compressed_path": compressed_path
    }

