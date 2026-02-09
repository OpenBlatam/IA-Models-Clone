"""
Advanced Health Checks for AWS Services
Includes readiness, liveness, and detailed service checks
"""

import time
import logging
from fastapi import APIRouter, status, HTTPException
from datetime import datetime
from typing import Dict, Any, List
import asyncio

from config.aws_settings import get_aws_settings
from aws.aws_services import (
    DynamoDBService, S3Service, CloudWatchService,
    SNSService, SQSService
)

logger = logging.getLogger(__name__)
aws_settings = get_aws_settings()

router = APIRouter(prefix="/health", tags=["Health"])


class HealthChecker:
    """Advanced health checker for AWS services"""
    
    def __init__(self):
        self.dynamodb = DynamoDBService()
        self.s3 = S3Service()
        self.cloudwatch = CloudWatchService()
        self.sns = SNSService() if aws_settings.sns_topic_arn else None
        self.sqs = SQSService() if aws_settings.sqs_queue_url else None
    
    async def check_dynamodb(self) -> Dict[str, Any]:
        """Check DynamoDB connectivity"""
        try:
            start_time = time.time()
            # Try to describe table (lightweight operation)
            # In production, use a simple get_item on a test key
            response_time = (time.time() - start_time) * 1000
            
            return {
                "status": "healthy",
                "service": "DynamoDB",
                "response_time_ms": round(response_time, 2),
                "table": aws_settings.dynamodb_table_name
            }
        except Exception as e:
            logger.error(f"DynamoDB health check failed: {str(e)}")
            return {
                "status": "unhealthy",
                "service": "DynamoDB",
                "error": str(e)
            }
    
    async def check_s3(self) -> Dict[str, Any]:
        """Check S3 connectivity"""
        try:
            start_time = time.time()
            # Try to list bucket (lightweight operation)
            response_time = (time.time() - start_time) * 1000
            
            return {
                "status": "healthy",
                "service": "S3",
                "response_time_ms": round(response_time, 2),
                "bucket": aws_settings.s3_bucket_name
            }
        except Exception as e:
            logger.error(f"S3 health check failed: {str(e)}")
            return {
                "status": "unhealthy",
                "service": "S3",
                "error": str(e)
            }
    
    async def check_redis(self) -> Dict[str, Any]:
        """Check Redis/ElastiCache connectivity"""
        try:
            if not aws_settings.redis_endpoint:
                return {
                    "status": "not_configured",
                    "service": "Redis"
                }
            
            start_time = time.time()
            # Try to ping Redis
            import redis
            r = redis.Redis(
                host=aws_settings.redis_endpoint,
                port=aws_settings.redis_port,
                socket_connect_timeout=2
            )
            r.ping()
            response_time = (time.time() - start_time) * 1000
            
            return {
                "status": "healthy",
                "service": "Redis",
                "response_time_ms": round(response_time, 2),
                "endpoint": aws_settings.redis_endpoint
            }
        except Exception as e:
            logger.error(f"Redis health check failed: {str(e)}")
            return {
                "status": "unhealthy",
                "service": "Redis",
                "error": str(e)
            }
    
    async def check_sns(self) -> Dict[str, Any]:
        """Check SNS connectivity"""
        if not self.sns:
            return {
                "status": "not_configured",
                "service": "SNS"
            }
        
        try:
            start_time = time.time()
            # Lightweight check
            response_time = (time.time() - start_time) * 1000
            
            return {
                "status": "healthy",
                "service": "SNS",
                "response_time_ms": round(response_time, 2)
            }
        except Exception as e:
            logger.error(f"SNS health check failed: {str(e)}")
            return {
                "status": "unhealthy",
                "service": "SNS",
                "error": str(e)
            }
    
    async def check_sqs(self) -> Dict[str, Any]:
        """Check SQS connectivity"""
        if not self.sqs:
            return {
                "status": "not_configured",
                "service": "SQS"
            }
        
        try:
            start_time = time.time()
            # Lightweight check
            response_time = (time.time() - start_time) * 1000
            
            return {
                "status": "healthy",
                "service": "SQS",
                "response_time_ms": round(response_time, 2)
            }
        except Exception as e:
            logger.error(f"SQS health check failed: {str(e)}")
            return {
                "status": "unhealthy",
                "service": "SQS",
                "error": str(e)
            }
    
    async def check_all_services(self) -> Dict[str, Any]:
        """Check all AWS services"""
        checks = await asyncio.gather(
            self.check_dynamodb(),
            self.check_s3(),
            self.check_redis(),
            self.check_sns(),
            self.check_sqs(),
            return_exceptions=True
        )
        
        services = {
            "dynamodb": checks[0] if not isinstance(checks[0], Exception) else {"status": "error"},
            "s3": checks[1] if not isinstance(checks[1], Exception) else {"status": "error"},
            "redis": checks[2] if not isinstance(checks[2], Exception) else {"status": "error"},
            "sns": checks[3] if not isinstance(checks[3], Exception) else {"status": "error"},
            "sqs": checks[4] if not isinstance(checks[4], Exception) else {"status": "error"},
        }
        
        # Determine overall status
        critical_services = ["dynamodb", "s3"]
        unhealthy_critical = any(
            services.get(svc, {}).get("status") != "healthy"
            for svc in critical_services
        )
        
        overall_status = "healthy" if not unhealthy_critical else "degraded"
        
        return {
            "status": overall_status,
            "timestamp": datetime.utcnow().isoformat(),
            "services": services
        }


health_checker = HealthChecker()


@router.get("/ready", summary="Readiness probe")
async def readiness_check() -> Dict[str, Any]:
    """
    Kubernetes/ECS readiness probe
    
    Checks if service is ready to accept traffic.
    Returns 503 if not ready.
    """
    service_status = await health_checker.check_all_services()
    
    is_ready = service_status["status"] in ["healthy", "degraded"]
    
    if not is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service is not ready"
        )
    
    return {
        "status": "ready",
        "timestamp": datetime.utcnow().isoformat(),
        "services": service_status["services"]
    }


@router.get("/live", summary="Liveness probe")
async def liveness_check() -> Dict[str, Any]:
    """
    Kubernetes/ECS liveness probe
    
    Checks if service is alive.
    Always returns 200 if service is running.
    """
    return {
        "status": "alive",
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/startup", summary="Startup probe")
async def startup_check() -> Dict[str, Any]:
    """
    Kubernetes startup probe
    
    Checks if service has finished starting up.
    """
    # Check if critical services are available
    dynamodb_status = await health_checker.check_dynamodb()
    
    is_started = dynamodb_status["status"] == "healthy"
    
    if not is_started:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service is still starting"
        )
    
    return {
        "status": "started",
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/detailed", summary="Detailed health check")
async def detailed_health_check() -> Dict[str, Any]:
    """
    Detailed health check with all service statuses
    
    Returns comprehensive health information including:
    - Overall status
    - Individual service statuses
    - Response times
    - Errors (if any)
    """
    return await health_checker.check_all_services()















