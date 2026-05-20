"""
Document Upload Routes
Real, working API endpoints for document upload and processing
"""

import logging
from fastapi import APIRouter, HTTPException, Form, File, UploadFile
from fastapi.responses import JSONResponse
from typing import Optional, List
import asyncio
from document_upload_processor import document_upload_processor
from real_working_processor import real_working_processor
from advanced_real_processor import advanced_real_processor

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/upload", tags=["Document Upload Processing"])

@router.post("/process-document")
async def process_document(
    file: UploadFile = File(...),
    analysis_type: str = Form("basic")
):
    """Upload and process document file"""
    try:
        # Check if file format is supported
        if not document_upload_processor.is_format_supported(file.filename):
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file format: {file.filename}"
            )
        
        # Read file content
        file_content = await file.read()
        
        # Parse document
        parse_result = await document_upload_processor.process_uploaded_document(
            file_content, file.filename
        )
        
        if not parse_result["processing_successful"]:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to parse document: {parse_result.get('error', 'Unknown error')}"
            )
        
        # Process with AI based on analysis type
        if analysis_type == "basic":
            ai_result = await real_working_processor.process_text(
                parse_result["text_content"], "analyze"
            )
        elif analysis_type == "advanced":
            ai_result = await advanced_real_processor.process_text_advanced(
                parse_result["text_content"], "analyze", use_cache=True
            )
        else:
            raise HTTPException(
                status_code=400,
                detail="Invalid analysis_type. Must be 'basic' or 'advanced'"
            )
        
        # Combine results
        result = {
            "document_info": {
                "filename": parse_result["filename"],
                "file_type": parse_result["file_type"],
                "file_size": parse_result["file_size"],
                "metadata": parse_result["metadata"]
            },
            "ai_analysis": ai_result,
            "processing_time": parse_result["processing_time"]
        }
        
        return JSONResponse(content=result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing uploaded document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/process-document-basic")
async def process_document_basic(
    file: UploadFile = File(...)
):
    """Upload and process document with basic AI analysis"""
    try:
        # Check if file format is supported
        if not document_upload_processor.is_format_supported(file.filename):
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file format: {file.filename}"
            )
        
        # Read file content
        file_content = await file.read()
        
        # Parse document
        parse_result = await document_upload_processor.process_uploaded_document(
            file_content, file.filename
        )
        
        if not parse_result["processing_successful"]:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to parse document: {parse_result.get('error', 'Unknown error')}"
            )
        
        # Process with basic AI
        ai_result = await real_working_processor.process_text(
            parse_result["text_content"], "analyze"
        )
        
        # Combine results
        result = {
            "document_info": {
                "filename": parse_result["filename"],
                "file_type": parse_result["file_type"],
                "file_size": parse_result["file_size"],
                "metadata": parse_result["metadata"]
            },
            "basic_analysis": ai_result,
            "processing_time": parse_result["processing_time"]
        }
        
        return JSONResponse(content=result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing uploaded document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/process-document-advanced")
async def process_document_advanced(
    file: UploadFile = File(...),
    use_cache: bool = Form(True)
):
    """Upload and process document with advanced AI analysis"""
    try:
        # Check if file format is supported
        if not document_upload_processor.is_format_supported(file.filename):
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file format: {file.filename}"
            )
        
        # Read file content
        file_content = await file.read()
        
        # Parse document
        parse_result = await document_upload_processor.process_uploaded_document(
            file_content, file.filename
        )
        
        if not parse_result["processing_successful"]:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to parse document: {parse_result.get('error', 'Unknown error')}"
            )
        
        # Process with advanced AI
        ai_result = await advanced_real_processor.process_text_advanced(
            parse_result["text_content"], "analyze", use_cache
        )
        
        # Combine results
        result = {
            "document_info": {
                "filename": parse_result["filename"],
                "file_type": parse_result["file_type"],
                "file_size": parse_result["file_size"],
                "metadata": parse_result["metadata"]
            },
            "advanced_analysis": ai_result,
            "processing_time": parse_result["processing_time"]
        }
        
        return JSONResponse(content=result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing uploaded document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch-upload")
async def batch_upload(
    files: List[UploadFile] = File(...),
    analysis_type: str = Form("basic")
):
    """Upload and process multiple documents"""
    try:
        results = []
        
        for i, file in enumerate(files):
            try:
                # Check if file format is supported
                if not document_upload_processor.is_format_supported(file.filename):
                    results.append({
                        "batch_index": i,
                        "filename": file.filename,
                        "error": f"Unsupported file format: {file.filename}",
                        "status": "error"
                    })
                    continue
                
                # Read file content
                file_content = await file.read()
                
                # Parse document
                parse_result = await document_upload_processor.process_uploaded_document(
                    file_content, file.filename
                )
                
                if not parse_result["processing_successful"]:
                    results.append({
                        "batch_index": i,
                        "filename": file.filename,
                        "error": parse_result.get('error', 'Unknown error'),
                        "status": "error"
                    })
                    continue
                
                # Process with AI based on analysis type
                if analysis_type == "basic":
                    ai_result = await real_working_processor.process_text(
                        parse_result["text_content"], "analyze"
                    )
                elif analysis_type == "advanced":
                    ai_result = await advanced_real_processor.process_text_advanced(
                        parse_result["text_content"], "analyze", use_cache=True
                    )
                else:
                    results.append({
                        "batch_index": i,
                        "filename": file.filename,
                        "error": "Invalid analysis_type",
                        "status": "error"
                    })
                    continue
                
                # Combine results
                result = {
                    "batch_index": i,
                    "filename": parse_result["filename"],
                    "file_type": parse_result["file_type"],
                    "file_size": parse_result["file_size"],
                    "ai_analysis": ai_result,
                    "processing_time": parse_result["processing_time"],
                    "status": "success"
                }
                
                results.append(result)
                
            except Exception as e:
                results.append({
                    "batch_index": i,
                    "filename": file.filename,
                    "error": str(e),
                    "status": "error"
                })
        
        return JSONResponse(content={
            "batch_results": results,
            "total_processed": len(results),
            "successful": len([r for r in results if r.get("status") == "success"]),
            "failed": len([r for r in results if r.get("status") == "error"])
        })
        
    except Exception as e:
        logger.error(f"Error in batch upload: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/supported-formats")
async def get_supported_formats():
    """Get list of supported file formats"""
    try:
        formats = document_upload_processor.get_supported_formats()
        return JSONResponse(content={
            "supported_formats": formats,
            "total_formats": len([f for f in formats.values() if f])
        })
    except Exception as e:
        logger.error(f"Error getting supported formats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/upload-stats")
async def get_upload_stats():
    """Get upload processing statistics"""
    try:
        stats = document_upload_processor.get_stats()
        return JSONResponse(content=stats)
    except Exception as e:
        logger.error(f"Error getting upload stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health-upload")
async def health_check_upload():
    """Upload processing health check"""
    try:
        stats = document_upload_processor.get_stats()
        formats = document_upload_processor.get_supported_formats()
        
        return JSONResponse(content={
            "status": "healthy",
            "service": "Document Upload Processor",
            "version": "1.0.0",
            "features": {
                "pdf_processing": formats["pdf"],
                "docx_processing": formats["docx"],
                "excel_processing": formats["xlsx"],
                "powerpoint_processing": formats["pptx"],
                "text_processing": formats["txt"],
                "ocr_processing": formats["image"]
            },
            "processing_stats": stats["stats"],
            "supported_formats": formats
        })
    except Exception as e:
        logger.error(f"Error in upload health check: {e}")
        raise HTTPException(status_code=500, detail=str(e))













