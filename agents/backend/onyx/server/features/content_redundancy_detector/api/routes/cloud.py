"""
Cloud Router - Cloud storage and integration endpoints
"""

import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Query, Path
from fastapi.responses import JSONResponse

try:
    from cloud_integration import cloud_manager, CloudConfig, CloudProvider
except ImportError:
    logging.warning("cloud_integration module not available")
    cloud_manager = None
    CloudConfig = None
    CloudProvider = None

from ...core.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/cloud", tags=["Cloud"])


@router.post("/config/add", response_model=Dict[str, Any])
async def add_cloud_config(config_data: Dict[str, Any]) -> JSONResponse:
    """Add cloud configuration"""
    logger.info("Cloud configuration addition requested")
    
    if not cloud_manager or not CloudConfig or not CloudProvider:
        raise HTTPException(status_code=503, detail="Cloud integration not available")
    
    try:
        name = config_data.get("name")
        provider = config_data.get("provider")
        region = config_data.get("region")
        credentials = config_data.get("credentials", {})
        
        if not all([name, provider, region]):
            raise ValueError("Name, provider, and region are required")
        
        cloud_config = CloudConfig(
            provider=CloudProvider(provider),
            region=region,
            credentials=credentials
        )
        
        cloud_manager.add_cloud_config(name, cloud_config)
        
        return JSONResponse(content={
            "success": True,
            "data": {
                "message": "Cloud configuration added successfully",
                "config_name": name,
                "provider": provider,
                "region": region
            },
            "error": None
        })
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Add cloud config error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/upload", response_model=Dict[str, Any])
async def upload_to_cloud(upload_data: Dict[str, Any]) -> JSONResponse:
    """Upload data to cloud storage"""
    logger.info("Cloud upload requested")
    
    if not cloud_manager:
        raise HTTPException(status_code=503, detail="Cloud integration not available")
    
    try:
        config_name = upload_data.get("config_name")
        bucket = upload_data.get("bucket")
        data = upload_data.get("data", {})
        
        if not all([config_name, bucket]):
            raise ValueError("Config name and bucket are required")
        
        success = await cloud_manager.upload_analysis_result(config_name, bucket, data)
        
        if not success:
            raise HTTPException(status_code=500, detail="Upload failed")
        
        return JSONResponse(content={
            "success": True,
            "data": {
                "message": "Data uploaded successfully",
                "config_name": config_name,
                "bucket": bucket
            },
            "error": None
        })
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Cloud upload error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/download", response_model=Dict[str, Any])
async def download_from_cloud(download_data: Dict[str, Any]) -> JSONResponse:
    """Download data from cloud storage"""
    logger.info("Cloud download requested")
    
    if not cloud_manager:
        raise HTTPException(status_code=503, detail="Cloud integration not available")
    
    try:
        config_name = download_data.get("config_name")
        bucket = download_data.get("bucket")
        key = download_data.get("key")
        
        if not all([config_name, bucket, key]):
            raise ValueError("Config name, bucket, and key are required")
        
        data = await cloud_manager.download_analysis_result(config_name, bucket, key)
        
        if not data:
            raise HTTPException(status_code=404, detail="Data not found")
        
        return JSONResponse(content={
            "success": True,
            "data": {
                "message": "Data downloaded successfully",
                "data": data,
                "config_name": config_name,
                "bucket": bucket,
                "key": key
            },
            "error": None
        })
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Cloud download error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/backup", response_model=Dict[str, Any])
async def backup_to_cloud(backup_data: Dict[str, Any]) -> JSONResponse:
    """Backup data to cloud storage"""
    logger.info("Cloud backup requested")
    
    if not cloud_manager:
        raise HTTPException(status_code=503, detail="Cloud integration not available")
    
    try:
        config_name = backup_data.get("config_name")
        bucket = backup_data.get("bucket")
        data = backup_data.get("data", {})
        
        if not all([config_name, bucket]):
            raise ValueError("Config name and bucket are required")
        
        success = await cloud_manager.backup_system_data(config_name, bucket, data)
        
        if not success:
            raise HTTPException(status_code=500, detail="Backup failed")
        
        return JSONResponse(content={
            "success": True,
            "data": {
                "message": "Data backed up successfully",
                "config_name": config_name,
                "bucket": bucket
            },
            "error": None
        })
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Cloud backup error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/list/{config_name}/{bucket}", response_model=Dict[str, Any])
async def list_cloud_files(
    config_name: str = Path(...),
    bucket: str = Path(...),
    prefix: str = Query(default="")
) -> JSONResponse:
    """List files in cloud storage"""
    logger.info(f"Cloud file listing requested: {config_name}/{bucket}")
    
    if not cloud_manager:
        raise HTTPException(status_code=503, detail="Cloud integration not available")
    
    try:
        files = await cloud_manager.list_analysis_results(config_name, bucket, prefix)
        
        return JSONResponse(content={
            "success": True,
            "data": {
                "config_name": config_name,
                "bucket": bucket,
                "prefix": prefix,
                "files": files,
                "count": len(files)
            },
            "error": None
        })
    except Exception as e:
        logger.error(f"Cloud file listing error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/cleanup/{config_name}/{bucket}", response_model=Dict[str, Any])
async def cleanup_cloud_files(
    config_name: str = Path(...),
    bucket: str = Path(...),
    days_old: int = Query(default=30, ge=1)
) -> JSONResponse:
    """Cleanup old files in cloud storage"""
    logger.info(f"Cloud cleanup requested: {config_name}/{bucket}")
    
    if not cloud_manager:
        raise HTTPException(status_code=503, detail="Cloud integration not available")
    
    try:
        deleted_count = await cloud_manager.delete_old_analysis_results(config_name, bucket, days_old)
        
        return JSONResponse(content={
            "success": True,
            "data": {
                "message": "Cloud cleanup completed",
                "config_name": config_name,
                "bucket": bucket,
                "days_old": days_old,
                "deleted_count": deleted_count
            },
            "error": None
        })
    except Exception as e:
        logger.error(f"Cloud cleanup error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/configs", response_model=Dict[str, Any])
async def get_cloud_configs() -> JSONResponse:
    """Get all cloud configurations"""
    logger.info("Cloud configurations requested")
    
    if not cloud_manager:
        raise HTTPException(status_code=503, detail="Cloud integration not available")
    
    try:
        configs = cloud_manager.get_cloud_configs()
        stats = cloud_manager.get_cloud_stats()
        
        return JSONResponse(content={
            "success": True,
            "data": {
                "configs": [
                    {
                        "name": name,
                        "provider": config.provider.value if hasattr(config.provider, 'value') else str(config.provider),
                        "region": config.region,
                        "endpoint": config.endpoint if hasattr(config, 'endpoint') else None
                    }
                    for name, config in configs.items()
                ],
                "stats": stats
            },
            "error": None
        })
    except Exception as e:
        logger.error(f"Get cloud configs error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")






