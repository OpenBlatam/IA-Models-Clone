from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

from typing import Optional, List, Dict, Any
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import and_
from onyx.db.ads import AdsGeneration, BackgroundRemoval, AdsAnalytics
from onyx.db.engine import get_session_context_manager
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Database service for ads functionality.
"""


class AdsDBService:
    """Service for handling ads-related database operations."""

    @staticmethod
    async def create_ads_generation(
        user_id: int,
        url: str,
        type: str,
        content: Dict[str, Any],
        prompt: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        brand_voice: Optional[Dict[str, Any]] = None,
        audience_profile: Optional[Dict[str, Any]] = None,
        project_context: Optional[Dict[str, Any]] = None
    ) -> AdsGeneration:
        """Create a new ads generation record."""
        async with get_session_context_manager() as session:
            ads_generation = AdsGeneration(
                user_id=user_id,
                url=url,
                type=type,
                prompt=prompt,
                content=content,
                metadata=metadata,
                brand_voice=brand_voice,
                audience_profile=audience_profile,
                project_context=project_context
            )
            session.add(ads_generation)
            await session.commit()
            await session.refresh(ads_generation)
            return ads_generation

    @staticmethod
    async def get_ads_generation(
        user_id: int,
        ads_id: int
    ) -> Optional[AdsGeneration]:
        """Get an ads generation record by ID."""
        async with get_session_context_manager() as session:
            return await session.query(AdsGeneration).filter(
                and_(
                    AdsGeneration.id == ads_id,
                    AdsGeneration.user_id == user_id,
                    AdsGeneration.is_deleted == False
                )
            ).first()

    @staticmethod
    async def list_ads_generations(
        user_id: int,
        type: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[AdsGeneration]:
        """List ads generation records for a user."""
        async with get_session_context_manager() as session:
            query = session.query(AdsGeneration).filter(
                and_(
                    AdsGeneration.user_id == user_id,
                    AdsGeneration.is_deleted == False
                )
            )
            if type:
                query = query.filter(AdsGeneration.type == type)
            return await query.order_by(AdsGeneration.created_at.desc()).offset(offset).limit(limit).all()

    @staticmethod
    async def create_background_removal(
        user_id: int,
        processed_image_url: str,
        original_image_url: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        image_settings: Optional[Dict[str, Any]] = None,
        content_sources: Optional[List[Dict[str, Any]]] = None
    ) -> BackgroundRemoval:
        """Create a new background removal record."""
        async with get_session_context_manager() as session:
            bg_removal = BackgroundRemoval(
                user_id=user_id,
                original_image_url=original_image_url,
                processed_image_url=processed_image_url,
                metadata=metadata,
                image_settings=image_settings,
                content_sources=content_sources
            )
            session.add(bg_removal)
            await session.commit()
            await session.refresh(bg_removal)
            return bg_removal

    @staticmethod
    async def get_background_removal(
        user_id: int,
        removal_id: int
    ) -> Optional[BackgroundRemoval]:
        """Get a background removal record by ID."""
        async with get_session_context_manager() as session:
            return await session.query(BackgroundRemoval).filter(
                and_(
                    BackgroundRemoval.id == removal_id,
                    BackgroundRemoval.user_id == user_id,
                    BackgroundRemoval.is_deleted == False
                )
            ).first()

    @staticmethod
    async def list_background_removals(
        user_id: int,
        limit: int = 100,
        offset: int = 0
    ) -> List[BackgroundRemoval]:
        """List background removal records for a user."""
        async with get_session_context_manager() as session:
            return await session.query(BackgroundRemoval).filter(
                and_(
                    BackgroundRemoval.user_id == user_id,
                    BackgroundRemoval.is_deleted == False
                )
            ).order_by(BackgroundRemoval.created_at.desc()).offset(offset).limit(limit).all()

    @staticmethod
    async def create_ads_analytics(
        user_id: int,
        ads_generation_id: int,
        metrics: Dict[str, Any],
        email_metrics: Optional[Dict[str, Any]] = None,
        email_settings: Optional[Dict[str, Any]] = None
    ) -> AdsAnalytics:
        """Create a new ads analytics record."""
        async with get_session_context_manager() as session:
            analytics = AdsAnalytics(
                user_id=user_id,
                ads_generation_id=ads_generation_id,
                metrics=metrics,
                email_metrics=email_metrics,
                email_settings=email_settings
            )
            session.add(analytics)
            await session.commit()
            await session.refresh(analytics)
            return analytics

    @staticmethod
    async def get_ads_analytics(
        user_id: int,
        analytics_id: int
    ) -> Optional[AdsAnalytics]:
        """Get an ads analytics record by ID."""
        async with get_session_context_manager() as session:
            return await session.query(AdsAnalytics).filter(
                and_(
                    AdsAnalytics.id == analytics_id,
                    AdsAnalytics.user_id == user_id
                )
            ).first()

    @staticmethod
    async def list_ads_analytics(
        user_id: int,
        ads_generation_id: Optional[int] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[AdsAnalytics]:
        """List ads analytics records for a user."""
        async with get_session_context_manager() as session:
            query = session.query(AdsAnalytics).filter(
                AdsAnalytics.user_id == user_id
            )
            if ads_generation_id:
                query = query.filter(AdsAnalytics.ads_generation_id == ads_generation_id)
            return await query.order_by(AdsAnalytics.created_at.desc()).offset(offset).limit(limit).all()

    @staticmethod
    async def soft_delete_ads_generation(
        user_id: int,
        ads_id: int
    ) -> bool:
        """Soft delete an ads generation record."""
        async with get_session_context_manager() as session:
            result = await session.query(AdsGeneration).filter(
                and_(
                    AdsGeneration.id == ads_id,
                    AdsGeneration.user_id == user_id,
                    AdsGeneration.is_deleted == False
                )
            ).update({"is_deleted": True})
            await session.commit()
            return result > 0

    @staticmethod
    async def soft_delete_background_removal(
        user_id: int,
        removal_id: int
    ) -> bool:
        """Soft delete a background removal record."""
        async with get_session_context_manager() as session:
            result = await session.query(BackgroundRemoval).filter(
                and_(
                    BackgroundRemoval.id == removal_id,
                    BackgroundRemoval.user_id == user_id,
                    BackgroundRemoval.is_deleted == False
                )
            ).update({"is_deleted": True})
            await session.commit()
            return result > 0 