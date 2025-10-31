"""
AI Integration System - API Endpoints
REST API endpoints for managing AI content integrations
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import asyncio
import uuid
import logging
from datetime import datetime

from .integration_engine import (
    AIIntegrationEngine, 
    IntegrationRequest, 
    IntegrationResult, 
    ContentType, 
    IntegrationStatus,
    integration_engine
)

# Create router
router = APIRouter(prefix="/ai-integration", tags=["AI Integration"])

# Pydantic models for API
class ContentDataModel(BaseModel):
    """Content data model for API requests"""
    title: str
    content: str
    tags: Optional[List[str]] = []
    author: Optional[str] = None
    category: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = {}

class IntegrationRequestModel(BaseModel):
    """Integration request model for API"""
    content_id: str = Field(..., description="Unique identifier for the content")
    content_type: str = Field(..., description="Type of content (blog_post, email_campaign, etc.)")
    content_data: ContentDataModel
    target_platforms: List[str] = Field(..., description="List of target platforms")
    priority: int = Field(default=1, ge=1, le=10, description="Priority level (1-10)")
    max_retries: int = Field(default=3, ge=0, le=10, description="Maximum retry attempts")

class IntegrationStatusModel(BaseModel):
    """Integration status response model"""
    content_id: str
    status: str
    results: List[Dict[str, Any]]
    created_at: datetime
    updated_at: datetime

class PlatformTestModel(BaseModel):
    """Platform connection test model"""
    platform: str
    status: bool
    message: str

class BulkIntegrationModel(BaseModel):
    """Bulk integration request model"""
    requests: List[IntegrationRequestModel]
    batch_id: Optional[str] = None

# API Endpoints

@router.post("/integrate", response_model=Dict[str, str])
async def create_integration(
    request: IntegrationRequestModel,
    background_tasks: BackgroundTasks
):
    """
    Create a new integration request
    """
    try:
        # Validate content type
        try:
            content_type = ContentType(request.content_type)
        except ValueError:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid content type. Valid types: {[ct.value for ct in ContentType]}"
            )
        
        # Create integration request
        integration_request = IntegrationRequest(
            content_id=request.content_id,
            content_type=content_type,
            content_data=request.content_data.dict(),
            target_platforms=request.target_platforms,
            priority=request.priority,
            max_retries=request.max_retries,
            metadata={"api_request": True, "timestamp": datetime.utcnow().isoformat()}
        )
        
        # Add to queue
        await integration_engine.add_integration_request(integration_request)
        
        # Process in background
        background_tasks.add_task(integration_engine.process_single_request, integration_request)
        
        return {
            "message": "Integration request created successfully",
            "content_id": request.content_id,
            "status": "queued"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/integrate/bulk", response_model=Dict[str, str])
async def create_bulk_integration(
    bulk_request: BulkIntegrationModel,
    background_tasks: BackgroundTasks
):
    """
    Create multiple integration requests in bulk
    """
    try:
        batch_id = bulk_request.batch_id or str(uuid.uuid4())
        processed_count = 0
        
        for request in bulk_request.requests:
            try:
                # Validate content type
                content_type = ContentType(request.content_type)
                
                # Create integration request
                integration_request = IntegrationRequest(
                    content_id=request.content_id,
                    content_type=content_type,
                    content_data=request.content_data.dict(),
                    target_platforms=request.target_platforms,
                    priority=request.priority,
                    max_retries=request.max_retries,
                    metadata={
                        "api_request": True, 
                        "batch_id": batch_id,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                )
                
                # Add to queue
                await integration_engine.add_integration_request(integration_request)
                processed_count += 1
                
            except Exception as e:
                logger.error(f"Error processing bulk request {request.content_id}: {str(e)}")
                continue
        
        # Process all requests in background
        background_tasks.add_task(integration_engine.process_integration_queue)
        
        return {
            "message": f"Bulk integration request created successfully",
            "batch_id": batch_id,
            "processed_requests": processed_count,
            "total_requests": len(bulk_request.requests),
            "status": "queued"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status/{content_id}", response_model=IntegrationStatusModel)
async def get_integration_status(content_id: str):
    """
    Get integration status for a specific content item
    """
    try:
        status_data = await integration_engine.get_integration_status(content_id)
        
        if status_data["status"] == "not_found":
            raise HTTPException(status_code=404, detail="Integration not found")
        
        return IntegrationStatusModel(
            content_id=content_id,
            status=status_data["overall_status"],
            results=status_data["results"],
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status", response_model=List[IntegrationStatusModel])
async def get_all_integration_status():
    """
    Get status of all integrations
    """
    try:
        all_status = []
        for content_id in integration_engine.results.keys():
            status_data = await integration_engine.get_integration_status(content_id)
            all_status.append(IntegrationStatusModel(
                content_id=content_id,
                status=status_data["overall_status"],
                results=status_data["results"],
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            ))
        
        return all_status
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/platforms", response_model=List[str])
async def get_available_platforms():
    """
    Get list of available integration platforms
    """
    try:
        return integration_engine.get_available_platforms()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/platforms/{platform}/test", response_model=PlatformTestModel)
async def test_platform_connection(platform: str):
    """
    Test connection to a specific platform
    """
    try:
        is_connected = await integration_engine.test_connection(platform)
        
        return PlatformTestModel(
            platform=platform,
            status=is_connected,
            message="Connection successful" if is_connected else "Connection failed"
        )
        
    except Exception as e:
        return PlatformTestModel(
            platform=platform,
            status=False,
            message=f"Connection test error: {str(e)}"
        )

@router.get("/queue/status", response_model=Dict[str, Any])
async def get_queue_status():
    """
    Get current queue status
    """
    try:
        return {
            "queue_length": len(integration_engine.integration_queue),
            "pending_requests": [
                {
                    "content_id": req.content_id,
                    "content_type": req.content_type.value,
                    "target_platforms": req.target_platforms,
                    "priority": req.priority,
                    "retry_count": req.retry_count
                }
                for req in integration_engine.integration_queue
            ],
            "engine_running": integration_engine.is_running,
            "total_results": len(integration_engine.results)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/queue/process")
async def process_queue(background_tasks: BackgroundTasks):
    """
    Manually trigger queue processing
    """
    try:
        background_tasks.add_task(integration_engine.process_integration_queue)
        return {"message": "Queue processing started", "status": "processing"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/status/{content_id}")
async def delete_integration_status(content_id: str):
    """
    Delete integration status for a specific content item
    """
    try:
        if content_id in integration_engine.results:
            del integration_engine.results[content_id]
            return {"message": f"Integration status deleted for {content_id}"}
        else:
            raise HTTPException(status_code=404, detail="Integration not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    try:
        return {
            "status": "healthy",
            "engine_running": integration_engine.is_running,
            "available_platforms": len(integration_engine.get_available_platforms()),
            "queue_length": len(integration_engine.integration_queue),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Webhook endpoints for platform callbacks
@router.post("/webhooks/{platform}")
async def handle_platform_webhook(platform: str, payload: Dict[str, Any]):
    """
    Handle webhooks from integrated platforms
    """
    try:
        # Log webhook for debugging
        logger.info(f"Received webhook from {platform}: {payload}")
        
        # Process webhook based on platform
        if platform == "salesforce":
            # Handle Salesforce webhook
            pass
        elif platform == "mailchimp":
            # Handle Mailchimp webhook
            pass
        elif platform == "wordpress":
            # Handle WordPress webhook
            pass
        
        return {"message": f"Webhook processed for {platform}", "status": "success"}
        
    except Exception as e:
        logger.error(f"Error processing webhook from {platform}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Export the router
__all__ = ["router"]
