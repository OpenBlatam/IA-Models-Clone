"""File processing utilities."""

import re
from pathlib import Path
from typing import Dict, Any, Optional
import PyPDF2
import fitz  # PyMuPDF


def validate_pdf_file(file_content: bytes, filename: str) -> Dict[str, Any]:
    """Validate PDF file and return validation result."""
    if not file_content:
        return {"valid": False, "error": "Empty file"}
    
    if not filename.lower().endswith('.pdf'):
        return {"valid": False, "error": "Invalid file type"}
    
    try:
        # Try to open with PyPDF2
        import io
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
        page_count = len(pdf_reader.pages)
        
        return {
            "valid": True,
            "page_count": page_count,
            "file_size": len(file_content)
        }
    except Exception as e:
        return {"valid": False, "error": f"Invalid PDF: {str(e)}"}


def extract_metadata(file_content: bytes) -> Dict[str, Any]:
    """Extract metadata from PDF file."""
    try:
        import io
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
        metadata = pdf_reader.metadata or {}
        
        return {
            "title": metadata.get("/Title", ""),
            "author": metadata.get("/Author", ""),
            "subject": metadata.get("/Subject", ""),
            "creator": metadata.get("/Creator", ""),
            "producer": metadata.get("/Producer", ""),
            "creation_date": metadata.get("/CreationDate", ""),
            "modification_date": metadata.get("/ModDate", "")
        }
    except Exception:
        return {}


def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe storage."""
    # Remove dangerous characters
    filename = re.sub(r'[<>:"|?*\\/]', '', filename)
    # Remove leading/trailing spaces and dots
    filename = filename.strip(' .')
    # Limit length
    if len(filename) > 255:
        filename = filename[:255]
    return filename or "unnamed_file.pdf"


def get_file_hash(file_content: bytes) -> str:
    """Generate hash for file content."""
    import hashlib
    return hashlib.md5(file_content).hexdigest()


def is_pdf_corrupted(file_content: bytes) -> bool:
    """Check if PDF file is corrupted."""
    try:
        import io
        PyPDF2.PdfReader(io.BytesIO(file_content))
        return False
    except Exception:
        return True
