"""Batch processing endpoints"""
from fastapi import APIRouter, HTTPException, Form
from typing import List
from utils.batch_processor import get_batch_processor
from services.markdown_parser import MarkdownParser
from utils.security import get_security_sanitizer
import json

router = APIRouter(prefix="/batch", tags=["Batch"])


@router.post("/convert")
async def batch_convert(
    markdown_contents: str = Form(...),  # JSON array string
    output_format: str = Form(...),
    include_charts: bool = Form(True),
    include_tables: bool = Form(True)
):
    """Batch convert multiple markdown contents"""
    try:
        contents = json.loads(markdown_contents)
        if not isinstance(contents, list):
            raise ValueError("markdown_contents must be a JSON array")
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in markdown_contents")
    
    batch_processor = get_batch_processor()
    parser = MarkdownParser()
    sanitizer = get_security_sanitizer()
    
    # Parse all contents
    parsed_contents = []
    for content in contents:
        sanitized = sanitizer.sanitize_markdown(content)
        parsed = parser.parse(sanitized)
        parsed_contents.append(parsed)
    
    # Process batch
    results = await batch_processor.process_batch(
        parsed_contents,
        output_format,
        include_charts,
        include_tables
    )
    
    return {
        "status": "success",
        "total": len(results),
        "results": results
    }

