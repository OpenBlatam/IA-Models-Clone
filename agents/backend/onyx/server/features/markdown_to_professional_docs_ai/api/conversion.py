"""Conversion endpoints"""
from fastapi import APIRouter, HTTPException, Form, File, UploadFile
from fastapi.responses import FileResponse
from typing import Optional
from pathlib import Path
import os

from config import settings
from services.converter_service import ConverterService
from services.markdown_parser import MarkdownParser
from utils.exceptions import ConversionException
from utils.validators import validate_format, validate_markdown_content
from utils.security import get_security_sanitizer
from utils.cache import get_cache
from utils.metrics import get_metrics, TimingContext
from utils.webhooks import get_webhook_client

router = APIRouter(prefix="/convert", tags=["Conversion"])

# Initialize services (singleton pattern)
_converter_service = None
_markdown_parser = None
_cache = None
_metrics = None
_webhook_client = None


def get_converter_service():
    """Get converter service instance"""
    global _converter_service
    if _converter_service is None:
        _converter_service = ConverterService()
    return _converter_service


def get_markdown_parser():
    """Get markdown parser instance"""
    global _markdown_parser
    if _markdown_parser is None:
        _markdown_parser = MarkdownParser()
    return _markdown_parser


def get_cache_instance():
    """Get cache instance"""
    global _cache
    if _cache is None:
        _cache = get_cache()
    return _cache


def get_metrics_instance():
    """Get metrics instance"""
    global _metrics
    if _metrics is None:
        _metrics = get_metrics()
    return _metrics


def get_webhook_client_instance():
    """Get webhook client instance"""
    global _webhook_client
    if _webhook_client is None:
        _webhook_client = get_webhook_client()
    return _webhook_client


@router.post("")
async def convert_markdown(
    markdown_content: str = Form(...),
    output_format: str = Form(...),
    include_charts: bool = Form(True),
    include_tables: bool = Form(True),
    custom_styling: Optional[str] = Form(None)
):
    """Convert Markdown content to specified format"""
    try:
        # Validate inputs
        validate_format(output_format)
        validate_markdown_content(markdown_content)
        
        # Get service instances
        sanitizer = get_security_sanitizer()
        cache = get_cache_instance()
        metrics = get_metrics_instance()
        webhook_client = get_webhook_client_instance()
        markdown_parser = get_markdown_parser()
        converter_service = get_converter_service()
        
        # Sanitize markdown
        sanitized = sanitizer.sanitize_markdown(markdown_content)
        
        # Check cache
        cache_key = f"{hash(sanitized)}:{output_format}"
        cached_path = cache.get(cache_key)
        if cached_path and os.path.exists(cached_path):
            metrics.increment("conversions.cache_hit")
            return FileResponse(
                cached_path,
                media_type="application/octet-stream",
                filename=Path(cached_path).name
            )
        
        # Send webhook
        await webhook_client.send_webhook("conversion.started", {
            "format": output_format
        })
        
        # Parse markdown
        with TimingContext(metrics, "parsing.duration"):
            parsed_content = markdown_parser.parse(sanitized)
        
        # Convert
        with TimingContext(metrics, "conversion.duration"):
            output_path = await converter_service.convert(
                parsed_content=parsed_content,
                output_format=output_format,
                include_charts=include_charts,
                include_tables=include_tables,
                custom_styling=custom_styling
            )
        
        # Cache result
        cache.set(cache_key, output_path)
        metrics.increment("conversions.total")
        metrics.increment(f"conversions.{output_format}")
        
        # Send webhook
        await webhook_client.send_webhook("conversion.completed", {
            "format": output_format,
            "output_path": output_path
        })
        
        return FileResponse(
            output_path,
            media_type="application/octet-stream",
            filename=Path(output_path).name
        )
    except ConversionException as e:
        metrics = get_metrics_instance()
        webhook_client = get_webhook_client_instance()
        metrics.increment("conversions.errors")
        await webhook_client.send_webhook("conversion.failed", {
            "error": str(e)
        })
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        metrics = get_metrics_instance()
        metrics.increment("conversions.errors")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/file")
async def convert_file(
    file: UploadFile = File(...),
    output_format: str = Form(...),
    include_charts: bool = Form(True),
    include_tables: bool = Form(True)
):
    """Convert Markdown file to specified format"""
    try:
        # Read file content
        content = await file.read()
        markdown_content = content.decode('utf-8')
        
        return await convert_markdown(
            markdown_content=markdown_content,
            output_format=output_format,
            include_charts=include_charts,
            include_tables=include_tables
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

