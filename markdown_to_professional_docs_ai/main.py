"""Markdown to Professional Documents AI - FastAPI Application"""
import logging
import os
from pathlib import Path
from typing import List, Optional
"""Markdown to Professional Documents AI - FastAPI Application"""
import logging
import os
import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.openapi.utils import get_openapi
import uvicorn

from config import settings
from services.converter_service import ConverterService
from services.markdown_parser import MarkdownParser
from utils.exceptions import (
    MarkdownConverterException,
    InvalidFormatException,
    ParsingException,
    ConversionException,
    FileSizeException,
    ValidationException
)
from utils.validators import (
    validate_format,
    validate_markdown_content,
    validate_file_size,
    validate_output_format
)
from utils.cache import get_cache
from utils.metrics import get_metrics, TimingContext
from utils.rate_limiter import get_rate_limiter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Ensure output and temp directories exist
os.makedirs(settings.output_dir, exist_ok=True)
os.makedirs(settings.temp_dir, exist_ok=True)

# Initialize FastAPI app
app = FastAPI(
    title=settings.app_name,
    description="Convert Markdown files to professional document formats: Excel, PDF, Word, Tableau, Power BI, and more",
    version="2.3.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
converter_service = ConverterService()
markdown_parser = MarkdownParser()
rate_limiter = get_rate_limiter(
    max_requests=settings.rate_limit_requests,
    window_seconds=settings.rate_limit_window
)
metrics = get_metrics()

# Include routers
from api import (
    root_router,
    health_router,
    conversion_router,
    formats_router,
    cache_router,
    metrics_router,
    templates_router,
    validation_router,
    security_router
)

app.include_router(root_router)
app.include_router(conversion_router)
app.include_router(health_router)
app.include_router(formats_router)
app.include_router(cache_router)
app.include_router(metrics_router)
app.include_router(templates_router)
app.include_router(validation_router)
app.include_router(security_router)


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Rate limiting middleware"""
    # Skip rate limiting for health and metrics endpoints
    if request.url.path in ["/health", "/metrics", "/cache/stats"]:
        return await call_next(request)
    
    # Get client identifier (IP address)
    client_ip = request.client.host if request.client else "unknown"
    
    # Check rate limit
    allowed, remaining = rate_limiter.is_allowed(client_ip)
    
    if not allowed:
        metrics.increment("rate_limit.exceeded")
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={
                "error": "Rate limit exceeded",
                "message": "Too many requests. Please try again later.",
                "retry_after": 60
            },
            headers={"Retry-After": "60"}
        )
    
    # Add rate limit headers
    response = await call_next(request)
    response.headers["X-RateLimit-Remaining"] = str(remaining)
    response.headers["X-RateLimit-Limit"] = "100"
    
    return response


class ConversionRequest(BaseModel):
    """Request model for text-based conversion"""
    markdown_content: str = Field(..., description="Markdown content to convert")
    output_format: str = Field(..., description="Output format: excel, pdf, word, tableau, powerbi, html")
    include_charts: bool = Field(True, description="Include charts and diagrams")
    include_tables: bool = Field(True, description="Include tables")
    custom_styling: Optional[dict] = Field(None, description="Custom styling options")
    template: Optional[str] = Field("professional", description="Template name (professional, modern, classic)")
    language: Optional[str] = Field("en", description="Language code (en, es, fr, de, etc.)")
    webhook_url: Optional[str] = Field(None, description="Webhook URL for async notifications")
    webhook_secret: Optional[str] = Field(None, description="Webhook secret for signing")


class BatchConversionRequest(BaseModel):
    """Request model for batch conversion"""
    markdown_contents: List[str] = Field(..., min_items=1, max_items=10)
    output_format: str
    include_charts: bool = True
    include_tables: bool = True


class ConversionResponse(BaseModel):
    """Response model for conversion"""
    success: bool
    message: str
    output_file: Optional[str] = None
    file_size: Optional[int] = None
    format: Optional[str] = None


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "description": "Convert Markdown to professional document formats",
        "supported_formats": [
            "excel", "pdf", "word", "tableau", "powerbi", 
            "html", "ppt", "latex", "rtf", "epub", "odt"
        ],
        "endpoints": {
            "convert": "/convert",
            "convert_file": "/convert/file",
            "batch_convert": "/convert/batch",
            "health": "/health",
            "formats": "/formats"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    from utils.health_checks import get_health_checker
    
    health_checker = get_health_checker()
    health_status = await health_checker.run_checks()
    
    cache = get_cache()
    cache_stats = cache.get_stats()
    
    return {
        **health_status,
        "service": settings.app_name,
        "version": settings.app_version,
        "cache": {
            "enabled": True,
            "entries": cache_stats["entries"],
            "size_mb": round(cache_stats["total_size"] / (1024 * 1024), 2)
        }
    }


@app.post("/cache/clear")
async def clear_cache():
    """Clear conversion cache"""
    cache = get_cache()
    cleared = cache.clear()
    return {
        "status": "success",
        "message": f"Cleared {cleared} cache entries"
    }


@app.get("/cache/stats")
async def cache_stats():
    """Get cache statistics"""
    cache = get_cache()
    return cache.get_stats()


@app.get("/metrics")
async def get_metrics_endpoint():
    """Get service metrics"""
    return metrics.get_metrics()


@app.post("/metrics/reset")
async def reset_metrics():
    """Reset all metrics"""
    metrics.reset()
    return {"status": "success", "message": "Metrics reset"}


@app.post("/validate")
async def validate_document(
    file_path: str = Form(...),
    expected_format: str = Form(...)
):
    """Validate a generated document"""
    validator = get_document_validator()
    result = validator.validate_document(file_path, expected_format)
    return result


@app.post("/compress")
async def compress_document(
    file_path: str = Form(...),
    method: str = Form("zip"),
    output_path: Optional[str] = Form(None)
):
    """Compress a document"""
    compressor = get_document_compressor()
    compressed_path = compressor.compress_file(file_path, output_path, method)
    
    if compressed_path:
        return {
            "status": "success",
            "original_path": file_path,
            "compressed_path": compressed_path,
            "method": method
        }
    else:
        raise HTTPException(status_code=500, detail="Compression failed")


@app.post("/version/create")
async def create_version(
    document_path: str = Form(...),
    metadata: Optional[str] = Form(None)
):
    """Create a version of a document"""
    from utils.document_versioning import get_document_versioning
    import json
    
    versioning = get_document_versioning()
    metadata_dict = json.loads(metadata) if metadata else None
    
    version_id = versioning.create_version(document_path, metadata_dict)
    
    return {
        "status": "success",
        "version_id": version_id,
        "document_path": document_path
    }


@app.get("/version/{version_id}")
async def get_version(version_id: str):
    """Get version information"""
    from utils.document_versioning import get_document_versioning
    
    versioning = get_document_versioning()
    version_info = versioning.get_version(version_id)
    
    if version_info:
        return version_info
    else:
        raise HTTPException(status_code=404, detail="Version not found")


@app.get("/versions")
async def list_versions(document_path: Optional[str] = None):
    """List all versions"""
    from utils.document_versioning import get_document_versioning
    
    versioning = get_document_versioning()
    versions = versioning.list_versions(document_path)
    
    return {
        "count": len(versions),
        "versions": versions
    }


@app.post("/version/{version_id}/restore")
async def restore_version(
    version_id: str,
    output_path: Optional[str] = Form(None)
):
    """Restore a version"""
    from utils.document_versioning import get_document_versioning
    
    versioning = get_document_versioning()
    restored_path = versioning.restore_version(version_id, output_path)
    
    if restored_path:
        return {
            "status": "success",
            "version_id": version_id,
            "restored_path": restored_path
        }
    else:
        raise HTTPException(status_code=500, detail="Restore failed")


@app.post("/backup/create")
async def create_backup(
    source_path: str = Form(...),
    backup_name: Optional[str] = Form(None)
):
    """Create a backup"""
    from utils.backup_manager import get_backup_manager
    
    backup_manager = get_backup_manager()
    backup_path = backup_manager.create_backup(source_path, backup_name)
    
    return {
        "status": "success",
        "backup_path": backup_path,
        "source_path": source_path
    }


@app.get("/backups")
async def list_backups(source_path: Optional[str] = None):
    """List all backups"""
    from utils.backup_manager import get_backup_manager
    
    backup_manager = get_backup_manager()
    backups = backup_manager.list_backups(source_path)
    
    return {
        "count": len(backups),
        "backups": backups
    }


@app.post("/backup/{backup_name}/restore")
async def restore_backup(
    backup_name: str,
    output_path: Optional[str] = Form(None)
):
    """Restore from backup"""
    from utils.backup_manager import get_backup_manager
    
    backup_manager = get_backup_manager()
    restored_path = backup_manager.restore_backup(backup_name, output_path)
    
    if restored_path:
        return {
            "status": "success",
            "backup_name": backup_name,
            "restored_path": restored_path
        }
    else:
        raise HTTPException(status_code=500, detail="Restore failed")


@app.post("/annotations/add")
async def add_annotation(
    document_path: str = Form(...),
    annotation_type: str = Form(...),
    content: str = Form(...),
    position: Optional[str] = Form(None),
    author: Optional[str] = Form(None)
):
    """Add annotation to document"""
    from utils.annotations import get_annotation_manager
    import json
    
    annotation_manager = get_annotation_manager()
    position_dict = json.loads(position) if position else None
    
    annotation_id = annotation_manager.add_annotation(
        document_path,
        annotation_type,
        content,
        position_dict,
        author
    )
    
    return {
        "status": "success",
        "annotation_id": annotation_id
    }


@app.get("/annotations/{document_path:path}")
async def get_annotations(document_path: str):
    """Get annotations for document"""
    from utils.annotations import get_annotation_manager
    
    annotation_manager = get_annotation_manager()
    annotations = annotation_manager.get_annotations(document_path)
    
    return {
        "count": len(annotations),
        "annotations": annotations
    }


@app.get("/search")
async def search_documents(
    query: str,
    limit: int = 10
):
    """Search documents"""
    from utils.search_index import get_search_index
    
    search_index = get_search_index()
    results = search_index.search(query, limit)
    
    return {
        "query": query,
        "count": len(results),
        "results": results
    }


@app.get("/analytics/report")
async def get_analytics_report(
    period: str = "daily"  # daily, weekly, monthly
):
    """Get analytics report"""
    from utils.analytics import get_analytics_engine
    
    analytics = get_analytics_engine()
    
    if period == "daily":
        report = analytics.generate_daily_report()
    elif period == "weekly":
        report = analytics.generate_weekly_report()
    elif period == "monthly":
        report = analytics.generate_monthly_report()
    else:
        report = analytics.generate_daily_report()
    
    return report


@app.get("/collaboration/{document_path:path}")
async def get_collaboration_info(document_path: str):
    """Get collaboration information for document"""
    from utils.collaboration import get_collaboration_tracker
    
    collaboration = get_collaboration_tracker()
    
    return {
        "change_history": collaboration.get_change_history(document_path),
        "collaborators": collaboration.get_collaborators(document_path),
        "statistics": collaboration.get_document_statistics(document_path)
    }


@app.get("/plugins")
async def list_plugins():
    """List all registered plugins"""
    from utils.plugin_system import get_plugin_manager
    
    plugin_manager = get_plugin_manager()
    return {
        "plugins": plugin_manager.list_plugins()
    }


@app.get("/scheduler/tasks")
async def list_scheduled_tasks():
    """List all scheduled tasks"""
    from utils.scheduler import get_scheduler
    
    scheduler = get_scheduler()
    return {
        "tasks": scheduler.list_tasks()
    }


@app.get("/scheduler/task/{task_id}")
async def get_task_status(task_id: str):
    """Get task status"""
    from utils.scheduler import get_scheduler
    
    scheduler = get_scheduler()
    task_status = scheduler.get_task_status(task_id)
    
    if task_status:
        return task_status
    else:
        raise HTTPException(status_code=404, detail="Task not found")


@app.post("/scheduler/task/{task_id}/cancel")
async def cancel_task(task_id: str):
    """Cancel scheduled task"""
    from utils.scheduler import get_scheduler
    
    scheduler = get_scheduler()
    cancelled = scheduler.cancel_task(task_id)
    
    if cancelled:
        return {"status": "success", "message": f"Task {task_id} cancelled"}
    else:
        raise HTTPException(status_code=404, detail="Task not found")


@app.get("/permissions/roles")
async def list_roles():
    """List all roles"""
    from utils.permissions import get_permission_manager
    
    permission_manager = get_permission_manager()
    return {
        "roles": permission_manager.list_roles()
    }


@app.post("/permissions/user/{user_id}/role")
async def assign_user_role(
    user_id: str,
    role_name: str = Form(...)
):
    """Assign role to user"""
    from utils.permissions import get_permission_manager
    
    permission_manager = get_permission_manager()
    assigned = permission_manager.assign_role(user_id, role_name)
    
    if assigned:
        return {
            "status": "success",
            "user_id": user_id,
            "role": role_name
        }
    else:
        raise HTTPException(status_code=400, detail="Invalid role name")


@app.post("/compare")
async def compare_documents(
    doc1_path: str = Form(...),
    doc2_path: str = Form(...),
    comparison_type: str = Form("content")
):
    """Compare two documents"""
    from utils.document_comparator import get_document_comparator
    
    comparator = get_document_comparator()
    result = comparator.compare_documents(doc1_path, doc2_path, comparison_type)
    
    return result


@app.post("/translate")
async def translate_content(
    content: str = Form(...),
    target_language: str = Form(...),
    source_language: Optional[str] = Form(None)
):
    """Translate content"""
    from utils.translator import get_translator
    
    translator = get_translator()
    result = translator.translate_content(content, target_language, source_language)
    
    return result


@app.post("/sign")
async def sign_document(
    document_path: str = Form(...),
    signer_name: str = Form(...),
    private_key: Optional[str] = Form(None)
):
    """Sign a document"""
    from utils.digital_signature import get_signature_manager
    
    signature_manager = get_signature_manager()
    signature = signature_manager.sign_document(document_path, signer_name, private_key)
    
    return {
        "status": "success",
        "signature": signature
    }


@app.post("/verify")
async def verify_signature(
    document_path: str = Form(...)
):
    """Verify document signature"""
    from utils.digital_signature import get_signature_manager
    
    signature_manager = get_signature_manager()
    verification = signature_manager.verify_signature(document_path)
    
    return verification


@app.post("/export/multiple")
async def export_multiple_formats(
    markdown_content: str = Form(...),
    output_formats: str = Form(...),  # Comma-separated
    include_charts: bool = Form(True),
    include_tables: bool = Form(True)
):
    """Export to multiple formats simultaneously"""
    from utils.multi_export import get_multi_exporter
    from services.markdown_parser import MarkdownParser
    from utils.security import get_security_sanitizer
    
    # Parse markdown
    sanitizer = get_security_sanitizer()
    sanitized = sanitizer.sanitize_markdown(markdown_content)
    parser = MarkdownParser()
    parsed_content = parser.parse(sanitized)
    
    # Parse formats
    formats = [f.strip() for f in output_formats.split(",")]
    
    # Export
    exporter = get_multi_exporter()
    results = await exporter.export_to_multiple_formats(
        parsed_content,
        formats,
        "multi_export",
        include_charts,
        include_tables
    )
    
    return results


@app.post("/export/package")
async def create_format_package(
    markdown_content: str = Form(...),
    output_formats: str = Form(...),
    package_name: str = Form("package"),
    include_charts: bool = Form(True),
    include_tables: bool = Form(True)
):
    """Create package with multiple format exports"""
    from utils.multi_export import get_multi_exporter
    from services.markdown_parser import MarkdownParser
    from utils.security import get_security_sanitizer
    
    # Parse markdown
    sanitizer = get_security_sanitizer()
    sanitized = sanitizer.sanitize_markdown(markdown_content)
    parser = MarkdownParser()
    parsed_content = parser.parse(sanitized)
    
    # Parse formats
    formats = [f.strip() for f in output_formats.split(",")]
    
    # Create package
    exporter = get_multi_exporter()
    package_path = await exporter.create_format_package(
        parsed_content,
        formats,
        package_name,
        include_charts,
        include_tables
    )
    
    if package_path:
        return {
            "status": "success",
            "package_path": package_path
        }
    else:
        raise HTTPException(status_code=500, detail="Package creation failed")


@app.get("/templates")
async def get_templates():
    """Get list of available templates"""
    from utils.advanced_templates import get_advanced_template_manager
    
    template_manager = get_template_manager()
    advanced_manager = get_advanced_template_manager()
    
    return {
        "basic_templates": template_manager.list_templates(),
        "advanced_templates": advanced_manager.list_templates(),
        "default": "professional"
    }


@app.post("/templates/create")
async def create_template(
    template_name: str = Form(...),
    template_config: str = Form(...)  # JSON string
):
    """Create a new template"""
    from utils.advanced_templates import get_advanced_template_manager
    import json
    
    advanced_manager = get_advanced_template_manager()
    try:
        config = json.loads(template_config)
        success = advanced_manager.create_template(template_name, config)
        
        if success:
            return {
                "status": "success",
                "template_name": template_name
            }
        else:
            raise HTTPException(status_code=400, detail="Template creation failed")
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in template_config")


@app.delete("/templates/{template_name}")
async def delete_template(template_name: str):
    """Delete a template"""
    from utils.advanced_templates import get_advanced_template_manager
    
    advanced_manager = get_advanced_template_manager()
    success = advanced_manager.delete_template(template_name)
    
    if success:
        return {
            "status": "success",
            "template_name": template_name
        }
    else:
        raise HTTPException(status_code=400, detail="Template deletion failed or template is built-in")


@app.get("/languages")
async def get_languages():
    """Get list of supported languages"""
    from utils.i18n import Language
    return {
        "languages": [lang.value for lang in Language],
        "default": "en"
    }


@app.get("/formats")
async def get_supported_formats():
    """Get list of supported output formats"""
    return {
        "formats": {
            "excel": {
                "description": "Microsoft Excel (.xlsx)",
                "features": ["tables", "charts", "formulas", "formatting"]
            },
            "pdf": {
                "description": "Portable Document Format (.pdf)",
                "features": ["text", "tables", "charts", "diagrams", "images"]
            },
            "word": {
                "description": "Microsoft Word (.docx)",
                "features": ["text", "tables", "images", "formatting", "styles"]
            },
            "tableau": {
                "description": "Tableau Workbook (.twb)",
                "features": ["data connections", "visualizations", "dashboards"]
            },
            "powerbi": {
                "description": "Power BI Report (.pbix)",
                "features": ["data models", "visualizations", "reports"]
            },
            "html": {
                "description": "HTML Document (.html)",
                "features": ["interactive", "charts", "responsive"]
            },
            "ppt": {
                "description": "PowerPoint Presentation (.pptx)",
                "features": ["slides", "charts", "images", "animations"]
            },
            "odt": {
                "description": "OpenDocument Text (.odt)",
                "features": ["text", "tables", "formatting"]
            },
            "rtf": {
                "description": "Rich Text Format (.rtf)",
                "features": ["text", "tables", "formatting", "colors"]
            },
            "latex": {
                "description": "LaTeX Document (.tex)",
                "features": ["math", "tables", "professional typesetting", "academic"]
            },
            "tex": {
                "description": "LaTeX Document (.tex)",
                "features": ["math", "tables", "professional typesetting", "academic"]
            },
            "epub": {
                "description": "EPUB E-book (.epub)",
                "features": ["ebook", "navigation", "reflowable", "mobile-friendly"]
            }
        }
    }


@app.post("/convert", response_model=ConversionResponse)
async def convert_markdown(request: ConversionRequest):
    """Convert Markdown content to specified format"""
    conversion_id = str(uuid.uuid4())
    
    with TimingContext("conversion.total"):
        metrics.increment("conversions.requested")
        
        # Detect language if not specified
        language = request.language or detect_language(request.markdown_content)
        
        # Get template
        template_manager = get_template_manager()
        template = template_manager.get_template(request.template or "professional")
        
        # Merge with custom styling
        if request.custom_styling:
            template = template_manager.merge_template(request.template or "professional", request.custom_styling)
        
        # Send webhook if provided
        webhook_client = None
        if request.webhook_url:
            webhook_client = get_webhook_client()
            await webhook_client.send_conversion_started(
                request.webhook_url,
                conversion_id,
                request.output_format,
                request.webhook_secret
            )
        
        try:
            # Validate inputs
            validate_markdown_content(request.markdown_content)
            output_format = validate_output_format(request.output_format)
            metrics.increment(f"conversions.format.{output_format}")
            
            # Check cache
            cache = get_cache()
            cache_options = {
                "include_charts": request.include_charts,
                "include_tables": request.include_tables,
                "custom_styling": template,
                "language": language
            }
            
            with TimingContext("conversion.cache_check"):
                cached_path = cache.get(request.markdown_content, output_format, cache_options)
            
            if cached_path:
                metrics.increment("conversions.cache_hit")
                file_size = os.path.getsize(cached_path) if os.path.exists(cached_path) else 0
                
                # Send webhook
                if webhook_client:
                    await webhook_client.send_conversion_completed(
                        request.webhook_url,
                        conversion_id,
                        output_format,
                        cached_path,
                        file_size,
                        request.webhook_secret
                    )
                
                return ConversionResponse(
                    success=True,
                    message=f"Successfully converted to {output_format} (cached)",
                    output_file=cached_path,
                    file_size=file_size,
                    format=output_format
                )
            
            metrics.increment("conversions.cache_miss")
            
            # Sanitize and parse markdown
            sanitizer = get_security_sanitizer()
            sanitized_content = sanitizer.sanitize_markdown(request.markdown_content)
            
            with TimingContext("conversion.parsing"):
                parsed_content = markdown_parser.parse(sanitized_content)
            
            # Record parsing statistics
            stats = parsed_content.get("statistics", {})
            metrics.record_value("parsing.words", stats.get("total_words", 0))
            metrics.record_value("parsing.tables", stats.get("total_tables", 0))
            
            # Convert to requested format
            with TimingContext(f"conversion.{output_format}"):
                output_path = await converter_service.convert(
                    parsed_content=parsed_content,
                    output_format=output_format,
                    include_charts=request.include_charts,
                    include_tables=request.include_tables,
                    custom_styling=template
                )
            
            # Cache result
            cache.set(request.markdown_content, output_format, cache_options, output_path)
            
            # Get file size
            file_size = os.path.getsize(output_path) if os.path.exists(output_path) else 0
            metrics.record_value("conversion.output_size", file_size)
            
            # Validate document
            validator = get_document_validator()
            validation_result = validator.validate_document(output_path, output_format)
            if not validation_result["valid"]:
                logger.warning(f"Document validation warnings: {validation_result.get('warnings', [])}")
            
            # Index document for search
            from utils.search_index import get_search_index
            search_index = get_search_index()
            search_index.index_document(
                output_path,
                request.markdown_content,
                {"format": output_format, "template": request.template}
            )
            
            # Track change
            from utils.collaboration import get_collaboration_tracker
            collaboration = get_collaboration_tracker()
            collaboration.track_change(
                output_path,
                "create",
                {"format": output_format, "file_size": file_size},
                author=request.custom_styling.get("author") if request.custom_styling else None
            )
            
            metrics.increment("conversions.success")
            
            # Send webhook
            if webhook_client:
                await webhook_client.send_conversion_completed(
                    request.webhook_url,
                    conversion_id,
                    output_format,
                    output_path,
                    file_size,
                    request.webhook_secret
                )
            
            return ConversionResponse(
                success=True,
                message=f"Successfully converted to {output_format}",
                output_file=output_path,
                file_size=file_size,
                format=output_format
            )
        except (InvalidFormatException, ValidationException, ParsingException) as e:
            logger.warning(f"Validation error: {str(e)}")
            if webhook_client and request.webhook_url:
                await webhook_client.send_conversion_failed(
                    request.webhook_url,
                    conversion_id,
                    request.output_format,
                    str(e),
                    request.webhook_secret
                )
            raise HTTPException(status_code=e.status_code, detail=e.message)
        except ConversionException as e:
            logger.error(f"Conversion error: {str(e)}", exc_info=True)
            if webhook_client and request.webhook_url:
                await webhook_client.send_conversion_failed(
                    request.webhook_url,
                    conversion_id,
                    request.output_format,
                    str(e),
                    request.webhook_secret
                )
            raise HTTPException(status_code=e.status_code, detail=e.message)
        except MarkdownConverterException as e:
            logger.error(f"Converter error: {str(e)}", exc_info=True)
            if webhook_client and request.webhook_url:
                await webhook_client.send_conversion_failed(
                    request.webhook_url,
                    conversion_id,
                    request.output_format,
                    str(e),
                    request.webhook_secret
                )
            raise HTTPException(status_code=e.status_code, detail=e.message)
        except Exception as e:
            metrics.increment("conversions.error")
            logger.error(f"Unexpected error: {str(e)}", exc_info=True)
            if webhook_client and request.webhook_url:
                await webhook_client.send_conversion_failed(
                    request.webhook_url,
                    conversion_id,
                    request.output_format,
                    str(e),
                    request.webhook_secret
                )
            raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/convert/file")
async def convert_markdown_file(
    file: UploadFile = File(...),
    output_format: str = Form(...),
    include_charts: bool = Form(True),
    include_tables: bool = Form(True)
):
    """Convert uploaded Markdown file to specified format"""
    try:
        # Validate file type
        if not file.filename or not file.filename.endswith(('.md', '.markdown')):
            raise ValidationException("File must be a Markdown file (.md or .markdown)")
        
        # Validate file size
        contents = await file.read()
        validate_file_size(len(contents), settings.max_file_size)
        
        # Validate format
        output_format = validate_output_format(output_format)
        
        # Parse markdown
        try:
            markdown_content = contents.decode('utf-8')
        except UnicodeDecodeError:
            raise ValidationException("File must be valid UTF-8 encoded text")
        
        validate_markdown_content(markdown_content)
        
        # Sanitize content
        sanitizer = get_security_sanitizer()
        sanitized_content = sanitizer.sanitize_markdown(markdown_content)
        
        parsed_content = markdown_parser.parse(sanitized_content)
        
        # Convert to requested format
        output_path = await converter_service.convert(
            parsed_content=parsed_content,
            output_format=output_format,
            include_charts=include_charts,
            include_tables=include_tables
        )
        
        # Return file
        return FileResponse(
            path=output_path,
            filename=Path(output_path).name,
            media_type="application/octet-stream"
        )
    except (InvalidFormatException, ValidationException, FileSizeException) as e:
        logger.warning(f"Validation error: {str(e)}")
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except ParsingException as e:
        logger.error(f"Parsing error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except ConversionException as e:
        logger.error(f"Conversion error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except HTTPException:
        raise
    except Exception as e:
        metrics.increment("conversions.error")
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/convert/batch", response_model=List[ConversionResponse])
async def batch_convert(request: BatchConversionRequest):
    """Convert multiple Markdown contents to specified format"""
    results = []
    
        for idx, markdown_content in enumerate(request.markdown_contents):
        try:
            # Sanitize and parse markdown
            sanitizer = get_security_sanitizer()
            sanitized_content = sanitizer.sanitize_markdown(markdown_content)
            parsed_content = markdown_parser.parse(sanitized_content)
            
            # Convert to requested format
            output_path = await converter_service.convert(
                parsed_content=parsed_content,
                output_format=request.output_format,
                include_charts=request.include_charts,
                include_tables=request.include_tables,
                filename_suffix=f"_batch_{idx+1}"
            )
            
            file_size = os.path.getsize(output_path) if os.path.exists(output_path) else 0
            
            results.append(ConversionResponse(
                success=True,
                message=f"Successfully converted item {idx+1} to {request.output_format}",
                output_file=output_path,
                file_size=file_size,
                format=request.output_format
            ))
        except Exception as e:
            logger.error(f"Batch conversion error for item {idx+1}: {str(e)}")
            results.append(ConversionResponse(
                success=False,
                message=f"Error converting item {idx+1}: {str(e)}",
                output_file=None,
                file_size=None,
                format=request.output_format
            ))
    
    return results


@app.post("/convert/repository")
async def convert_repository(
    repository_path: str = Form(...),
    output_format: str = Form(...),
    include_charts: bool = Form(True),
    include_tables: bool = Form(True),
    recursive: bool = Form(True)
):
    """Convert all Markdown files in a repository to specified format"""
    try:
        repo_path = Path(repository_path)
        if not repo_path.exists():
            raise HTTPException(status_code=404, detail="Repository path not found")
        
        # Find all markdown files
        pattern = "**/*.md" if recursive else "*.md"
        md_files = list(repo_path.glob(pattern))
        
        if not md_files:
            raise HTTPException(status_code=404, detail="No Markdown files found in repository")
        
        results = []
        for md_file in md_files:
            try:
                # Read and parse markdown
                with open(md_file, 'r', encoding='utf-8') as f:
                    markdown_content = f.read()
                
                # Sanitize content
                sanitizer = get_security_sanitizer()
                sanitized_content = sanitizer.sanitize_markdown(markdown_content)
                parsed_content = markdown_parser.parse(sanitized_content)
                
                # Convert to requested format
                output_path = await converter_service.convert(
                    parsed_content=parsed_content,
                    output_format=output_format,
                    include_charts=include_charts,
                    include_tables=include_tables,
                    filename_suffix=f"_{md_file.stem}"
                )
                
                file_size = os.path.getsize(output_path) if os.path.exists(output_path) else 0
                
                results.append({
                    "source_file": str(md_file),
                    "output_file": output_path,
                    "file_size": file_size,
                    "success": True
                })
            except Exception as e:
                logger.error(f"Error converting {md_file}: {str(e)}")
                results.append({
                    "source_file": str(md_file),
                    "output_file": None,
                    "file_size": None,
                    "success": False,
                    "error": str(e)
                })
        
        return {
            "total_files": len(md_files),
            "successful": sum(1 for r in results if r.get("success")),
            "failed": sum(1 for r in results if not r.get("success")),
            "results": results
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Repository conversion error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """Dashboard web interface"""
    from pathlib import Path
    dashboard_path = Path(__file__).parent / "templates" / "dashboard.html"
    if dashboard_path.exists():
        return dashboard_path.read_text()
    else:
        return HTMLResponse(content="<h1>Dashboard not found</h1>", status_code=404)


@app.post("/compare")
async def compare_documents(
    doc1_path: str = Form(...),
    doc2_path: str = Form(...),
    comparison_type: str = Form("content")
):
    """Compare two documents"""
    from utils.document_comparator import get_document_comparator
    
    comparator = get_document_comparator()
    result = comparator.compare_documents(doc1_path, doc2_path, comparison_type)
    
    return result


@app.post("/translate")
async def translate_content(
    content: str = Form(...),
    target_language: str = Form(...),
    source_language: Optional[str] = Form(None)
):
    """Translate content"""
    from utils.translator import get_translator
    
    translator = get_translator()
    result = translator.translate_content(content, target_language, source_language)
    
    return result


@app.post("/sign")
async def sign_document(
    document_path: str = Form(...),
    signer_name: str = Form(...),
    private_key: Optional[str] = Form(None)
):
    """Sign a document"""
    from utils.digital_signature import get_signature_manager
    
    signature_manager = get_signature_manager()
    signature = signature_manager.sign_document(document_path, signer_name, private_key)
    
    return {
        "status": "success",
        "signature": signature
    }


@app.post("/verify")
async def verify_signature(
    document_path: str = Form(...)
):
    """Verify document signature"""
    from utils.digital_signature import get_signature_manager
    
    signature_manager = get_signature_manager()
    verification = signature_manager.verify_signature(document_path)
    
    return verification


@app.post("/export/multiple")
async def export_multiple_formats(
    markdown_content: str = Form(...),
    output_formats: str = Form(...),  # Comma-separated
    include_charts: bool = Form(True),
    include_tables: bool = Form(True)
):
    """Export to multiple formats simultaneously"""
    from utils.multi_export import get_multi_exporter
    from services.markdown_parser import MarkdownParser
    from utils.security import get_security_sanitizer
    
    # Parse markdown
    sanitizer = get_security_sanitizer()
    sanitized = sanitizer.sanitize_markdown(markdown_content)
    parser = MarkdownParser()
    parsed_content = parser.parse(sanitized)
    
    # Parse formats
    formats = [f.strip() for f in output_formats.split(",")]
    
    # Export
    exporter = get_multi_exporter()
    results = await exporter.export_to_multiple_formats(
        parsed_content,
        formats,
        "multi_export",
        include_charts,
        include_tables
    )
    
    return results


@app.post("/export/package")
async def create_format_package(
    markdown_content: str = Form(...),
    output_formats: str = Form(...),
    package_name: str = Form("package"),
    include_charts: bool = Form(True),
    include_tables: bool = Form(True)
):
    """Create package with multiple format exports"""
    from utils.multi_export import get_multi_exporter
    from services.markdown_parser import MarkdownParser
    from utils.security import get_security_sanitizer
    
    # Parse markdown
    sanitizer = get_security_sanitizer()
    sanitized = sanitizer.sanitize_markdown(markdown_content)
    parser = MarkdownParser()
    parsed_content = parser.parse(sanitized)
    
    # Parse formats
    formats = [f.strip() for f in output_formats.split(",")]
    
    # Create package
    exporter = get_multi_exporter()
    package_path = await exporter.create_format_package(
        parsed_content,
        formats,
        package_name,
        include_charts,
        include_tables
    )
    
    if package_path:
        return {
            "status": "success",
            "package_path": package_path
        }
    else:
        raise HTTPException(status_code=500, detail="Package creation failed")


@app.post("/workflow/create")
async def create_workflow(
    workflow_name: str = Form(...),
    workflow_steps: str = Form(...)  # JSON string
):
    """Create a new workflow"""
    from utils.workflow_engine import get_workflow_engine
    import json
    
    workflow_engine = get_workflow_engine()
    try:
        steps = json.loads(workflow_steps)
        success = workflow_engine.create_workflow(workflow_name, steps)
        
        if success:
            return {
                "status": "success",
                "workflow_name": workflow_name
            }
        else:
            raise HTTPException(status_code=400, detail="Workflow creation failed")
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in workflow_steps")


@app.post("/workflow/execute")
async def execute_workflow(
    workflow_name: str = Form(...),
    initial_data: str = Form("{}")  # JSON string
):
    """Execute a workflow"""
    from utils.workflow_engine import get_workflow_engine
    import json
    
    workflow_engine = get_workflow_engine()
    try:
        data = json.loads(initial_data)
        execution_id = await workflow_engine.execute_workflow(workflow_name, data)
        
        return {
            "status": "started",
            "execution_id": execution_id,
            "workflow_name": workflow_name
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in initial_data")


@app.get("/workflow/execution/{execution_id}")
async def get_workflow_execution(execution_id: str):
    """Get workflow execution status"""
    from utils.workflow_engine import get_workflow_engine, WorkflowStatus
    
    workflow_engine = get_workflow_engine()
    execution = workflow_engine.get_execution(execution_id)
    
    if execution:
        return {
            "execution_id": execution.workflow_id,
            "status": execution.status.value,
            "current_step": execution.current_step,
            "results": execution.results,
            "started_at": execution.started_at.isoformat(),
            "completed_at": execution.completed_at.isoformat() if execution.completed_at else None,
            "error": execution.error
        }
    else:
        raise HTTPException(status_code=404, detail="Execution not found")


@app.get("/workflows")
async def list_workflows():
    """List all workflows"""
    from utils.workflow_engine import get_workflow_engine
    
    workflow_engine = get_workflow_engine()
    workflows = workflow_engine.list_workflows()
    
    return {
        "workflows": workflows
    }


@app.post("/ai/suggest")
async def get_ai_suggestions(
    markdown_content: str = Form(...)
):
    """Get AI-powered suggestions"""
    from utils.ai_suggestions import get_ai_engine
    from services.markdown_parser import MarkdownParser
    from utils.security import get_security_sanitizer
    
    # Parse markdown
    sanitizer = get_security_sanitizer()
    sanitized = sanitizer.sanitize_markdown(markdown_content)
    parser = MarkdownParser()
    parsed_content = parser.parse(sanitized)
    
    # Get AI suggestions
    ai_engine = get_ai_engine()
    suggestions = ai_engine.suggest_improvements(parsed_content)
    
    return suggestions


@app.post("/ai/analyze")
async def analyze_content(
    markdown_content: str = Form(...)
):
    """Analyze content and generate summary"""
    from utils.ai_suggestions import get_ai_engine
    from services.markdown_parser import MarkdownParser
    from utils.security import get_security_sanitizer
    
    # Parse markdown
    sanitizer = get_security_sanitizer()
    sanitized = sanitizer.sanitize_markdown(markdown_content)
    parser = MarkdownParser()
    parsed_content = parser.parse(sanitized)
    
    # Analyze
    ai_engine = get_ai_engine()
    summary = ai_engine.generate_summary(parsed_content)
    
    return summary


@app.post("/cloud/upload")
async def upload_to_cloud(
    file_path: str = Form(...),
    provider: str = Form(...),
    destination: Optional[str] = Form(None)
):
    """Upload file to cloud storage"""
    from utils.cloud_storage import get_cloud_storage
    
    cloud_storage = get_cloud_storage()
    
    if provider == "s3":
        result = cloud_storage.upload_to_s3(file_path, destination)
    elif provider == "gdrive":
        result = cloud_storage.upload_to_gdrive(file_path, destination)
    elif provider == "azure":
        result = cloud_storage.upload_to_azure(file_path, destination or "documents")
    else:
        raise HTTPException(status_code=400, detail=f"Unknown provider: {provider}")
    
    if result.get("success"):
        return result
    else:
        raise HTTPException(status_code=500, detail=result.get("error", "Upload failed"))


@app.get("/cloud/providers")
async def list_cloud_providers():
    """List available cloud storage providers"""
    from utils.cloud_storage import get_cloud_storage
    
    cloud_storage = get_cloud_storage()
    providers = cloud_storage.list_providers()
    
    provider_info = {}
    for provider in providers:
        provider_info[provider] = cloud_storage.get_provider_info(provider)
    
    return {
        "providers": provider_info
    }


@app.post("/review")
async def review_document(
    document_path: str = Form(...),
    reviewer: str = Form(...),
    markdown_content: Optional[str] = Form(None)
):
    """Review a document"""
    from utils.document_review import get_document_reviewer
    from services.markdown_parser import MarkdownParser
    from utils.security import get_security_sanitizer
    
    reviewer_instance = get_document_reviewer()
    
    parsed_content = None
    if markdown_content:
        sanitizer = get_security_sanitizer()
        sanitized = sanitizer.sanitize_markdown(markdown_content)
        parser = MarkdownParser()
        parsed_content = parser.parse(sanitized)
    
    review = reviewer_instance.review_document(document_path, reviewer, parsed_content)
    
    return {
        "document_path": review.document_path,
        "reviewer": review.reviewer,
        "status": review.status,
        "score": review.score,
        "comments": [
            {
                "reviewer": c.reviewer,
                "comment": c.comment,
                "line_number": c.line_number,
                "severity": c.severity,
                "timestamp": c.timestamp.isoformat()
            }
            for c in review.comments
        ],
        "reviewed_at": review.reviewed_at.isoformat()
    }


@app.get("/review/{document_path:path}")
async def get_document_reviews(document_path: str):
    """Get all reviews for a document"""
    from utils.document_review import get_document_reviewer
    
    reviewer = get_document_reviewer()
    reviews = reviewer.get_reviews(document_path)
    
    return {
        "document_path": document_path,
        "reviews": [
            {
                "reviewer": r.reviewer,
                "status": r.status,
                "score": r.score,
                "reviewed_at": r.reviewed_at.isoformat(),
                "comments_count": len(r.comments)
            }
            for r in reviews
        ],
        "total_reviews": len(reviews)
    }


@app.post("/integrations/register")
async def register_integration(
    name: str = Form(...),
    base_url: str = Form(...),
    api_key: Optional[str] = Form(None)
):
    """Register external API integration"""
    from utils.api_client import get_integration_manager
    
    integration_manager = get_integration_manager()
    success = integration_manager.register_integration(name, base_url, api_key)
    
    if success:
        return {
            "status": "success",
            "integration": name
        }
    else:
        raise HTTPException(status_code=400, detail="Integration registration failed")


@app.get("/integrations")
async def list_integrations():
    """List registered integrations"""
    from utils.api_client import get_integration_manager
    
    integration_manager = get_integration_manager()
    integrations = integration_manager.list_integrations()
    
    return {
        "integrations": integrations
    }


@app.post("/webhooks/register")
async def register_webhook(
    webhook_id: str = Form(...),
    url: str = Form(...),
    secret: Optional[str] = Form(None),
    events: Optional[str] = Form(None),  # Comma-separated
    retry_count: int = Form(3),
    retry_delay: int = Form(5)
):
    """Register webhook"""
    from utils.advanced_webhooks import get_advanced_webhook_client, WebhookEvent, WebhookConfig
    
    webhook_client = get_advanced_webhook_client()
    
    # Parse events
    event_list = []
    if events:
        for event_str in events.split(","):
            try:
                event_list.append(WebhookEvent(event_str.strip()))
            except ValueError:
                pass
    
    config = WebhookConfig(
        url=url,
        secret=secret,
        events=event_list if event_list else None,
        retry_count=retry_count,
        retry_delay=retry_delay
    )
    
    success = webhook_client.register_webhook(webhook_id, config)
    
    if success:
        return {
            "status": "success",
            "webhook_id": webhook_id
        }
    else:
        raise HTTPException(status_code=400, detail="Webhook registration failed")


@app.get("/webhooks")
async def list_webhooks():
    """List registered webhooks"""
    from utils.advanced_webhooks import get_advanced_webhook_client
    
    webhook_client = get_advanced_webhook_client()
    webhooks = webhook_client.list_webhooks()
    
    return {
        "webhooks": webhooks
    }


@app.get("/webhooks/deliveries")
async def get_webhook_deliveries(
    webhook_id: Optional[str] = None,
    event: Optional[str] = None,
    limit: int = 100
):
    """Get webhook deliveries"""
    from utils.advanced_webhooks import get_advanced_webhook_client, WebhookEvent
    
    webhook_client = get_advanced_webhook_client()
    
    event_enum = None
    if event:
        try:
            event_enum = WebhookEvent(event)
        except ValueError:
            pass
    
    deliveries = webhook_client.get_deliveries(webhook_id, event_enum, limit)
    
    return {
        "deliveries": [
            {
                "webhook_id": d.webhook_id,
                "event": d.event.value,
                "status": d.status,
                "attempts": d.attempts,
                "response_code": d.response_code,
                "created_at": d.created_at.isoformat()
            }
            for d in deliveries
        ]
    }


@app.post("/pipeline/create")
async def create_pipeline(
    pipeline_name: str = Form(...)
):
    """Create a new data pipeline"""
    from utils.data_pipeline import get_pipeline_manager
    
    pipeline_manager = get_pipeline_manager()
    pipeline = pipeline_manager.create_pipeline(pipeline_name)
    
    return {
        "status": "success",
        "pipeline_name": pipeline.name
    }


@app.post("/pipeline/{pipeline_name}/step")
async def add_pipeline_step(
    pipeline_name: str,
    step_name: str = Form(...),
    transform_type: str = Form(...),
    function_code: str = Form(...)  # Would need to be serialized function
):
    """Add step to pipeline"""
    # This would require function serialization - simplified for now
    return {
        "status": "not_implemented",
        "message": "Pipeline step addition requires function serialization"
    }


@app.post("/pipeline/{pipeline_name}/execute")
async def execute_pipeline(
    pipeline_name: str,
    data: str = Form(...)  # JSON string
):
    """Execute a pipeline"""
    from utils.data_pipeline import get_pipeline_manager
    import json
    
    pipeline_manager = get_pipeline_manager()
    
    try:
        input_data = json.loads(data)
        result = pipeline_manager.execute_pipeline(pipeline_name, input_data)
        
        return {
            "status": "success",
            "result": result
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in data")


@app.get("/pipelines")
async def list_pipelines():
    """List all pipelines"""
    from utils.data_pipeline import get_pipeline_manager
    
    pipeline_manager = get_pipeline_manager()
    pipelines = pipeline_manager.list_pipelines()
    
    return {
        "pipelines": pipelines
    }


@app.post("/tests/run")
async def run_tests():
    """Run all tests"""
    from utils.testing import get_test_runner
    
    test_runner = get_test_runner()
    results = await test_runner.run_all_tests()
    
    return {
        "results": [
            {
                "test_name": r.test_name,
                "passed": r.passed,
                "error": r.error,
                "execution_time": r.execution_time,
                "timestamp": r.timestamp.isoformat()
            }
            for r in results
        ],
        "summary": test_runner.get_test_summary()
    }


@app.get("/tests/summary")
async def get_test_summary():
    """Get test summary"""
    from utils.testing import get_test_runner
    
    test_runner = get_test_runner()
    summary = test_runner.get_test_summary()
    
    return summary


@app.get("/metrics/prometheus")
async def prometheus_metrics():
    """Prometheus metrics endpoint"""
    from utils.prometheus_metrics import get_prometheus_metrics
    from fastapi.responses import Response
    
    metrics = get_prometheus_metrics()
    metrics_data = metrics.get_metrics()
    
    return Response(
        content=metrics_data,
        media_type="text/plain; version=0.0.4; charset=utf-8"
    )


@app.post("/rate-limit/user/{user_id}/limits")
async def set_user_rate_limits(
    user_id: str,
    requests_per_minute: Optional[int] = Form(None),
    requests_per_hour: Optional[int] = Form(None),
    requests_per_day: Optional[int] = Form(None)
):
    """Set custom rate limits for user"""
    from utils.advanced_rate_limiter import get_user_rate_limiter
    
    rate_limiter = get_user_rate_limiter()
    rate_limiter.set_user_limits(
        user_id,
        requests_per_minute,
        requests_per_hour,
        requests_per_day
    )
    
    return {
        "status": "success",
        "user_id": user_id,
        "limits": {
            "per_minute": requests_per_minute,
            "per_hour": requests_per_hour,
            "per_day": requests_per_day
        }
    }


@app.get("/rate-limit/user/{user_id}/stats")
async def get_user_rate_limit_stats(user_id: str):
    """Get user rate limit statistics"""
    from utils.advanced_rate_limiter import get_user_rate_limiter
    
    rate_limiter = get_user_rate_limiter()
    stats = rate_limiter.get_user_stats(user_id)
    
    return stats


@app.post("/rate-limit/user/{user_id}/reset")
async def reset_user_rate_limit(user_id: str):
    """Reset user rate limit"""
    from utils.advanced_rate_limiter import get_user_rate_limiter
    
    rate_limiter = get_user_rate_limiter()
    rate_limiter.reset_user(user_id)
    
    return {
        "status": "success",
        "user_id": user_id
    }


@app.post("/queue/task")
async def enqueue_task(
    task_type: str = Form(...),
    payload: str = Form(...),  # JSON string
    priority: int = Form(0)
):
    """Enqueue a task for asynchronous processing"""
    from utils.task_queue import get_task_queue
    import json
    
    task_queue = get_task_queue()
    
    try:
        task_payload = json.loads(payload)
        task_id = await task_queue.enqueue(task_type, task_payload, priority)
        
        return {
            "status": "queued",
            "task_id": task_id
        }
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in payload")


@app.get("/queue/task/{task_id}")
async def get_task_status(task_id: str):
    """Get task status"""
    from utils.task_queue import get_task_queue, TaskStatus
    
    task_queue = get_task_queue()
    task = task_queue.get_task(task_id)
    
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return {
        "task_id": task.task_id,
        "task_type": task.task_type,
        "status": task.status.value,
        "priority": task.priority,
        "created_at": task.created_at.isoformat(),
        "started_at": task.started_at.isoformat() if task.started_at else None,
        "completed_at": task.completed_at.isoformat() if task.completed_at else None,
        "result": task.result,
        "error": task.error
    }


@app.get("/queue/tasks")
async def list_tasks(
    status: Optional[str] = None,
    limit: int = 100
):
    """List tasks"""
    from utils.task_queue import get_task_queue, TaskStatus
    
    task_queue = get_task_queue()
    
    status_enum = None
    if status:
        try:
            status_enum = TaskStatus(status)
        except ValueError:
            pass
    
    tasks = task_queue.get_tasks(status_enum, limit)
    
    return {
        "tasks": [
            {
                "task_id": t.task_id,
                "task_type": t.task_type,
                "status": t.status.value,
                "priority": t.priority,
                "created_at": t.created_at.isoformat()
            }
            for t in tasks
        ],
        "total": len(tasks)
    }


@app.post("/auth/login")
async def login(
    username: str = Form(...),
    password: str = Form(...)
):
    """Login and get JWT token"""
    from utils.jwt_auth import get_jwt_auth
    from utils.audit_log import get_audit_logger, AuditAction
    from fastapi import Request
    
    # Placeholder authentication - in production would verify against user database
    # For now, accept any username/password
    
    jwt_auth = get_jwt_auth()
    access_token = jwt_auth.create_access_token(
        user_id=username,
        username=username,
        roles=["user"]
    )
    refresh_token = jwt_auth.create_refresh_token(username)
    
    # Audit log
    audit_logger = get_audit_logger()
    audit_logger.log(
        action=AuditAction.LOGIN,
        resource="auth",
        user_id=username,
        success=True
    )
    
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer"
    }


@app.post("/auth/refresh")
async def refresh_token(
    refresh_token: str = Form(...)
):
    """Refresh access token"""
    from utils.jwt_auth import get_jwt_auth
    
    jwt_auth = get_jwt_auth()
    new_access_token = jwt_auth.refresh_access_token(refresh_token)
    
    if not new_access_token:
        raise HTTPException(status_code=401, detail="Invalid refresh token")
    
    return {
        "access_token": new_access_token,
        "token_type": "bearer"
    }


@app.get("/audit/logs")
async def get_audit_logs(
    user_id: Optional[str] = None,
    action: Optional[str] = None,
    resource: Optional[str] = None,
    limit: int = 100
):
    """Get audit logs"""
    from utils.audit_log import get_audit_logger, AuditAction
    
    audit_logger = get_audit_logger()
    
    action_enum = None
    if action:
        try:
            action_enum = AuditAction(action)
        except ValueError:
            pass
    
    logs = audit_logger.get_logs(user_id, action_enum, resource, limit)
    
    return {
        "logs": [
            {
                "timestamp": log.timestamp.isoformat(),
                "user_id": log.user_id,
                "action": log.action.value,
                "resource": log.resource,
                "resource_id": log.resource_id,
                "details": log.details,
                "ip_address": log.ip_address,
                "success": log.success,
                "error_message": log.error_message
            }
            for log in logs
        ],
        "total": len(logs)
    }


@app.get("/config")
async def get_config(key: Optional[str] = None):
    """Get configuration"""
    from utils.dynamic_config import get_dynamic_config
    
    config = get_dynamic_config()
    
    if key:
        value = config.get(key)
        return {key: value}
    else:
        return {"config": config.config}


@app.post("/config")
async def set_config(
    key: str = Form(...),
    value: str = Form(...)  # JSON string
):
    """Set configuration"""
    from utils.dynamic_config import get_dynamic_config
    import json
    
    config = get_dynamic_config()
    
    try:
        parsed_value = json.loads(value)
        config.set(key, parsed_value)
        
        return {
            "status": "success",
            "key": key,
            "value": parsed_value
        }
    except json.JSONDecodeError:
        # Try as string
        config.set(key, value)
        return {
            "status": "success",
            "key": key,
            "value": value
        }


@app.get("/cache/redis/stats")
async def get_redis_cache_stats():
    """Get Redis cache statistics"""
    from utils.redis_cache import get_redis_cache
    
    redis_cache = get_redis_cache()
    if not redis_cache:
        return {
            "enabled": False,
            "message": "Redis not configured"
        }
    
    stats = redis_cache.get_stats()
    return stats


@app.post("/images/optimize")
async def optimize_image(
    image_path: str = Form(...),
    quality: str = Form("medium"),
    max_width: Optional[int] = Form(None),
    max_height: Optional[int] = Form(None),
    format: Optional[str] = Form(None)
):
    """Optimize image"""
    from utils.advanced_image_optimizer import get_image_optimizer
    
    optimizer = get_image_optimizer()
    result = optimizer.optimize_image(
        image_path,
        quality=quality,
        max_width=max_width,
        max_height=max_height,
        format=format
    )
    
    return result


@app.get("/images/info")
async def get_image_info(image_path: str):
    """Get image information"""
    from utils.advanced_image_optimizer import get_image_optimizer
    
    optimizer = get_image_optimizer()
    info = optimizer.get_image_info(image_path)
    
    return info


@app.post("/preview/generate")
async def generate_preview(
    document_path: str = Form(...),
    format: str = Form("png"),
    page: int = Form(1),
    width: int = Form(800),
    height: int = Form(1000)
):
    """Generate document preview"""
    from utils.document_preview import get_preview_generator
    
    generator = get_preview_generator()
    result = generator.generate_preview(
        document_path,
        format=format,
        page=page,
        width=width,
        height=height
    )
    
    return result


@app.post("/watermark/add")
async def add_watermark(
    document_path: str = Form(...),
    watermark_config: str = Form(...),  # JSON string
    output_path: Optional[str] = Form(None)
):
    """Add watermark to document"""
    from utils.advanced_watermark import get_watermarker
    import json
    
    watermarker = get_watermarker()
    
    try:
        config = json.loads(watermark_config)
        result = watermarker.add_watermark(document_path, config, output_path)
        return result
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in watermark_config")


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.port,
        reload=True,
        log_level="info"
    )

