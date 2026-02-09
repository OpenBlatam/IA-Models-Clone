"""
Analytics Background Tasks
Background tasks for tracking analytics events
"""

import logging
from typing import Optional

from ...services.analytics_service import AnalyticsService

logger = logging.getLogger(__name__)

async def track_content_generation(
    user_id: str,
    content_type: str,
    processing_time: float,
    analytics_service: Optional[AnalyticsService] = None
) -> None:
    """Track content generation analytics"""
    try:
        if analytics_service:
            await analytics_service.track_content_creation(
                user_id=user_id,
                content_id="",
                content_type=content_type,
                processing_time=processing_time,
                quality_score=0.0,
                metadata={"processing_time": processing_time}
            )
        logger.info(
            "Tracked content generation",
            extra={
                "user_id": user_id,
                "content_type": content_type,
                "processing_time": processing_time
            }
        )
    except Exception as e:
        logger.error(
            "Error tracking content generation",
            extra={
                "user_id": user_id,
                "content_type": content_type,
                "error": str(e)
            },
            exc_info=True
        )

async def track_export(
    user_id: str,
    content_type: str,
    output_format: str,
    analytics_service: Optional[AnalyticsService] = None
) -> None:
    """Track export analytics"""
    try:
        if analytics_service:
            await analytics_service.track_content_export(
                user_id=user_id,
                content_id="",
                export_format=output_format,
                metadata={"content_type": content_type}
            )
        logger.info(
            "Tracked export",
            extra={
                "user_id": user_id,
                "content_type": content_type,
                "output_format": output_format
            }
        )
    except Exception as e:
        logger.error(
            "Error tracking export",
            extra={
                "user_id": user_id,
                "content_type": content_type,
                "error": str(e)
            },
            exc_info=True
        )







