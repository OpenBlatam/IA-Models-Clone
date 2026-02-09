from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query, Path
from fastapi.responses import JSONResponse, StreamingResponse
import asyncio
from application.use_cases.analyze_url import AnalyzeURLUseCase
from application.use_cases.analyze_urls_batch import AnalyzeURLsBatchUseCase
from application.dto.analyze_url_request import AnalyzeURLRequest
from application.dto.analyze_urls_batch_request import AnalyzeURLsBatchRequest
from application.dto.analyze_url_response import AnalyzeURLResponse
from presentation.validators.url_validator import URLValidator
from presentation.middleware.performance_middleware import PerformanceMiddleware
from shared.core.dependencies import get_analyze_url_use_case, get_analyze_urls_batch_use_case
from shared.core.logging import get_logger
from shared.core.metrics import REQUEST_COUNTER, REQUEST_DURATION, ERROR_COUNTER
        import urllib.parse
        import urllib.parse
        import csv
        import io
from typing import Any, List, Dict, Optional
import logging
"""
SEO API Routes
FastAPI routes for SEO analysis endpoints
"""



logger = get_logger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/seo", tags=["SEO Analysis"])


@router.post("/analyze", response_model=AnalyzeURLResponse)
async def analyze_url(
    request: AnalyzeURLRequest,
    use_case: AnalyzeURLUseCase = Depends(get_analyze_url_use_case),
    validator: URLValidator = Depends()
):
    """
    Analyze single URL for SEO
    
    This endpoint analyzes a single URL and returns comprehensive SEO data
    including meta tags, links, content analysis, and recommendations.
    """
    try:
        # Validate request
        validator.validate(request)
        
        # Execute use case
        result = await use_case.execute(request)
        
        logger.info(
            "URL analysis completed successfully",
            url=request.url,
            score=result.score,
            grade=result.grade
        )
        
        return result
        
    except Exception as e:
        ERROR_COUNTER.labels(endpoint="analyze_url", error_type=type(e).__name__).inc()
        logger.error("Error in analyze URL endpoint", url=request.url, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analyze")
async def analyze_url_get(
    url: str = Query(..., description="URL to analyze"),
    include_content: bool = Query(True, description="Include content analysis"),
    include_links: bool = Query(True, description="Include links analysis"),
    include_meta: bool = Query(True, description="Include meta tags analysis"),
    max_links: int = Query(100, description="Maximum number of links to analyze", ge=0, le=1000),
    timeout: float = Query(10.0, description="Request timeout in seconds", ge=1.0, le=60.0),
    use_case: AnalyzeURLUseCase = Depends(get_analyze_url_use_case),
    validator: URLValidator = Depends()
):
    """
    Analyze single URL via GET request
    
    This endpoint provides the same functionality as POST /analyze but via GET request
    for easier integration and testing.
    """
    try:
        # Create request object
        request = AnalyzeURLRequest(
            url=url,
            include_content=include_content,
            include_links=include_links,
            include_meta=include_meta,
            max_links=max_links,
            timeout=timeout
        )
        
        # Validate request
        validator.validate(request)
        
        # Execute use case
        result = await use_case.execute(request)
        
        logger.info(
            "URL analysis completed successfully (GET)",
            url=url,
            score=result.score,
            grade=result.grade
        )
        
        return result
        
    except Exception as e:
        ERROR_COUNTER.labels(endpoint="analyze_url_get", error_type=type(e).__name__).inc()
        logger.error("Error in analyze URL GET endpoint", url=url, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze/batch")
async def analyze_urls_batch(
    request: AnalyzeURLsBatchRequest,
    use_case: AnalyzeURLsBatchUseCase = Depends(get_analyze_urls_batch_use_case),
    background_tasks: BackgroundTasks
):
    """
    Analyze multiple URLs concurrently
    
    This endpoint analyzes multiple URLs in parallel and returns
    comprehensive results for each URL.
    """
    try:
        # Execute use case
        result = await use_case.execute(request)
        
        # Add background task for cleanup
        background_tasks.add_task(use_case.cleanup)
        
        logger.info(
            "Batch URL analysis completed successfully",
            total_urls=result["total_urls"],
            successful=result["successful"],
            failed=result["failed"],
            processing_time=result["processing_time"]
        )
        
        return result
        
    except Exception as e:
        ERROR_COUNTER.labels(endpoint="analyze_urls_batch", error_type=type(e).__name__).inc()
        logger.error("Error in batch analysis endpoint", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analyze/batch")
async def analyze_urls_batch_get(
    urls: str = Query(..., description="Comma-separated URLs to analyze"),
    include_content: bool = Query(True, description="Include content analysis"),
    include_links: bool = Query(True, description="Include links analysis"),
    include_meta: bool = Query(True, description="Include meta tags analysis"),
    max_links: int = Query(100, description="Maximum number of links to analyze", ge=0, le=1000),
    timeout: float = Query(10.0, description="Request timeout in seconds", ge=1.0, le=60.0),
    use_case: AnalyzeURLsBatchUseCase = Depends(get_analyze_urls_batch_use_case),
    background_tasks: BackgroundTasks
):
    """
    Analyze multiple URLs via GET request
    
    This endpoint provides the same functionality as POST /analyze/batch but via GET request.
    URLs should be comma-separated.
    """
    try:
        # Parse URLs
        url_list = [url.strip() for url in urls.split(',') if url.strip()]
        
        # Create request object
        request = AnalyzeURLsBatchRequest(
            urls=url_list,
            include_content=include_content,
            include_links=include_links,
            include_meta=include_meta,
            max_links=max_links,
            timeout=timeout
        )
        
        # Execute use case
        result = await use_case.execute(request)
        
        # Add background task for cleanup
        background_tasks.add_task(use_case.cleanup)
        
        logger.info(
            "Batch URL analysis completed successfully (GET)",
            total_urls=result["total_urls"],
            successful=result["successful"],
            failed=result["failed"]
        )
        
        return result
        
    except Exception as e:
        ERROR_COUNTER.labels(endpoint="analyze_urls_batch_get", error_type=type(e).__name__).inc()
        logger.error("Error in batch analysis GET endpoint", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analyze/{url:path}")
async def analyze_url_path(
    url: str = Path(..., description="URL to analyze"),
    include_content: bool = Query(True, description="Include content analysis"),
    include_links: bool = Query(True, description="Include links analysis"),
    include_meta: bool = Query(True, description="Include meta tags analysis"),
    max_links: int = Query(100, description="Maximum number of links to analyze", ge=0, le=1000),
    timeout: float = Query(10.0, description="Request timeout in seconds", ge=1.0, le=60.0),
    use_case: AnalyzeURLUseCase = Depends(get_analyze_url_use_case),
    validator: URLValidator = Depends()
):
    """
    Analyze URL via path parameter
    
    This endpoint allows analyzing URLs via path parameter for RESTful design.
    """
    try:
        # Decode URL from path
        decoded_url = urllib.parse.unquote(url)
        
        # Create request object
        request = AnalyzeURLRequest(
            url=decoded_url,
            include_content=include_content,
            include_links=include_links,
            include_meta=include_meta,
            max_links=max_links,
            timeout=timeout
        )
        
        # Validate request
        validator.validate(request)
        
        # Execute use case
        result = await use_case.execute(request)
        
        logger.info(
            "URL analysis completed successfully (path)",
            url=decoded_url,
            score=result.score,
            grade=result.grade
        )
        
        return result
        
    except Exception as e:
        ERROR_COUNTER.labels(endpoint="analyze_url_path", error_type=type(e).__name__).inc()
        logger.error("Error in analyze URL path endpoint", url=url, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/compare")
async def compare_urls(
    url1: str = Query(..., description="First URL to compare"),
    url2: str = Query(..., description="Second URL to compare"),
    include_content: bool = Query(True, description="Include content analysis"),
    include_links: bool = Query(True, description="Include links analysis"),
    include_meta: bool = Query(True, description="Include meta tags analysis"),
    max_links: int = Query(100, description="Maximum number of links to analyze", ge=0, le=1000),
    timeout: float = Query(10.0, description="Request timeout in seconds", ge=1.0, le=60.0),
    use_case: AnalyzeURLUseCase = Depends(get_analyze_url_use_case),
    validator: URLValidator = Depends()
):
    """
    Compare two URLs for SEO analysis
    
    This endpoint analyzes two URLs and provides a comparison
    of their SEO performance.
    """
    try:
        # Analyze both URLs
        request1 = AnalyzeURLRequest(
            url=url1,
            include_content=include_content,
            include_links=include_links,
            include_meta=include_meta,
            max_links=max_links,
            timeout=timeout
        )
        
        request2 = AnalyzeURLRequest(
            url=url2,
            include_content=include_content,
            include_links=include_links,
            include_meta=include_meta,
            max_links=max_links,
            timeout=timeout
        )
        
        # Validate requests
        validator.validate(request1)
        validator.validate(request2)
        
        # Execute analyses concurrently
        results = await asyncio.gather(
            use_case.execute(request1),
            use_case.execute(request2)
        )
        
        # Create comparison
        comparison = {
            "url1": results[0],
            "url2": results[1],
            "comparison": {
                "score_difference": results[0].score - results[1].score,
                "content_length_difference": results[0].content_length - results[1].content_length,
                "links_difference": len(results[0].links) - len(results[1].links),
                "meta_tags_difference": len(results[0].meta_tags) - len(results[1].meta_tags),
                "processing_time_difference": results[0].processing_time - results[1].processing_time
            }
        }
        
        logger.info(
            "URL comparison completed successfully",
            url1=url1,
            url2=url2,
            score1=results[0].score,
            score2=results[1].score
        )
        
        return comparison
        
    except Exception as e:
        ERROR_COUNTER.labels(endpoint="compare_urls", error_type=type(e).__name__).inc()
        logger.error("Error in URL comparison endpoint", url1=url1, url2=url2, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/export/{url:path}")
async def export_analysis(
    url: str = Path(..., description="URL to analyze"),
    format: str = Query("json", description="Export format (json, csv, xml)"),
    include_content: bool = Query(True, description="Include content analysis"),
    include_links: bool = Query(True, description="Include links analysis"),
    include_meta: bool = Query(True, description="Include meta tags analysis"),
    max_links: int = Query(100, description="Maximum number of links to analyze", ge=0, le=1000),
    timeout: float = Query(10.0, description="Request timeout in seconds", ge=1.0, le=60.0),
    use_case: AnalyzeURLUseCase = Depends(get_analyze_url_use_case),
    validator: URLValidator = Depends()
):
    """
    Export SEO analysis in various formats
    
    This endpoint analyzes a URL and exports the results in JSON, CSV, or XML format.
    """
    try:
        # Decode URL from path
        decoded_url = urllib.parse.unquote(url)
        
        # Create request object
        request = AnalyzeURLRequest(
            url=decoded_url,
            include_content=include_content,
            include_links=include_links,
            include_meta=include_meta,
            max_links=max_links,
            timeout=timeout
        )
        
        # Validate request
        validator.validate(request)
        
        # Execute use case
        result = await use_case.execute(request)
        
        # Export in requested format
        if format.lower() == "csv":
            return StreamingResponse(
                iter([self._to_csv(result)]),
                media_type="text/csv",
                headers={"Content-Disposition": f"attachment; filename=seo_analysis_{decoded_url.replace('://', '_').replace('/', '_')}.csv"}
            )
        elif format.lower() == "xml":
            return StreamingResponse(
                iter([self._to_xml(result)]),
                media_type="application/xml",
                headers={"Content-Disposition": f"attachment; filename=seo_analysis_{decoded_url.replace('://', '_').replace('/', '_')}.xml"}
            )
        else:
            return result
        
    except Exception as e:
        ERROR_COUNTER.labels(endpoint="export_analysis", error_type=type(e).__name__).inc()
        logger.error("Error in export analysis endpoint", url=url, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
    
    def _to_csv(self, result: AnalyzeURLResponse) -> str:
        """Convert result to CSV format"""
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow([
            "URL", "Title", "Description", "Keywords", "Score", "Grade", "Level",
            "Content Length", "Links Count", "Meta Tags Count", "Processing Time",
            "Issues Count", "Recommendations Count", "Timestamp"
        ])
        
        # Write data
        writer.writerow([
            result.url,
            result.title or "",
            result.description or "",
            result.keywords or "",
            result.score,
            result.grade,
            result.level,
            result.content_length,
            len(result.links),
            len(result.meta_tags),
            result.processing_time,
            len(result.issues),
            len(result.recommendations),
            result.timestamp
        ])
        
        return output.getvalue()
    
    def _to_xml(self, result: AnalyzeURLResponse) -> str:
        """Convert result to XML format"""
        xml_parts = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<seo_analysis>',
            f'  <url>{result.url}</url>',
            f'  <title>{result.title or ""}</title>',
            f'  <description>{result.description or ""}</description>',
            f'  <keywords>{result.keywords or ""}</keywords>',
            f'  <score>{result.score}</score>',
            f'  <grade>{result.grade}</grade>',
            f'  <level>{result.level}</level>',
            f'  <content_length>{result.content_length}</content_length>',
            f'  <processing_time>{result.processing_time}</processing_time>',
            '  <links>',
        ]
        
        for link in result.links:
            xml_parts.append(f'    <link>{link}</link>')
        
        xml_parts.extend([
            '  </links>',
            '  <meta_tags>',
        ])
        
        for name, value in result.meta_tags.items():
            xml_parts.append(f'    <meta name="{name}">{value}</meta>')
        
        xml_parts.extend([
            '  </meta_tags>',
            '  <issues>',
        ])
        
        for issue in result.issues:
            xml_parts.append(f'    <issue>{issue}</issue>')
        
        xml_parts.extend([
            '  </issues>',
            '  <recommendations>',
        ])
        
        for recommendation in result.recommendations:
            xml_parts.append(f'    <recommendation>{recommendation}</recommendation>')
        
        xml_parts.extend([
            '  </recommendations>',
            f'  <timestamp>{result.timestamp}</timestamp>',
            '</seo_analysis>'
        ])
        
        return '\n'.join(xml_parts) 