"""
Repository implementations for the ads feature.

This module provides concrete implementations of the repository interfaces
defined in the database module, implementing the actual data access logic.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

from sqlalchemy import and_, desc, func, select, update, delete
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

try:
    from onyx.db.ads import AdsGeneration, BackgroundRemoval, AdsAnalytics  # type: ignore
except Exception:  # pragma: no cover - optionalize DB models for tests
    from typing import Any as _Any
    AdsGeneration = BackgroundRemoval = AdsAnalytics = _Any  # type: ignore
try:
    from onyx.utils.logger import setup_logger  # type: ignore
except Exception:  # pragma: no cover - fallback minimal logger for tests
    import logging as _logging

    def setup_logger(name: str | None = None):  # type: ignore[override]
        logger = _logging.getLogger(name or __name__)
        if not _logging.getLogger().handlers:
            _logging.basicConfig(level=_logging.INFO)
        return logger

from .database import (
    AdsRepository, CampaignRepository, GroupRepository, PerformanceRepository,
    AnalyticsRepository, OptimizationRepository, DatabaseManager
)

logger = setup_logger()

class AdsRepositoryImpl(AdsRepository):
    """Concrete implementation of AdsRepository."""
    
    def __init__(self, database_manager: DatabaseManager):
        self.database_manager = database_manager
    
    async def create(self, ads_data: Dict[str, Any]) -> AdsGeneration:
        """Create a new ads generation record."""
        async def _create_ads(session: AsyncSession, data: Dict[str, Any]) -> AdsGeneration:
            ads_generation = AdsGeneration(**data)
            session.add(ads_generation)
            await session.commit()
            await session.refresh(ads_generation)
            return ads_generation
        
        return await self.database_manager.execute_query(_create_ads, ads_data)
    
    async def get_by_id(self, ads_id: int, user_id: int) -> Optional[AdsGeneration]:
        """Get an ads generation record by ID."""
        async def _get_ads(session: AsyncSession, aid: int, uid: int) -> Optional[AdsGeneration]:
            result = await session.execute(
                select(AdsGeneration).filter(
                    and_(
                        AdsGeneration.id == aid,
                        AdsGeneration.user_id == uid,
                        AdsGeneration.is_deleted == False
                    )
                )
            )
            return result.scalar_one_or_none()
        
        return await self.database_manager.execute_query(_get_ads, ads_id, user_id)
    
    async def list_by_user(self, user_id: int, limit: int = 100, offset: int = 0) -> List[AdsGeneration]:
        """List ads generation records for a user."""
        async def _list_ads(session: AsyncSession, uid: int, lim: int, off: int) -> List[AdsGeneration]:
            result = await session.execute(
                select(AdsGeneration).filter(
                    and_(
                        AdsGeneration.user_id == uid,
                        AdsGeneration.is_deleted == False
                    )
                ).order_by(desc(AdsGeneration.created_at)).offset(off).limit(lim)
            )
            return result.scalars().all()
        
        return await self.database_manager.execute_query(_list_ads, user_id, limit, offset)
    
    async def update(self, ads_id: int, user_id: int, update_data: Dict[str, Any]) -> Optional[AdsGeneration]:
        """Update an ads generation record."""
        async def _update_ads(session: AsyncSession, aid: int, uid: int, data: Dict[str, Any]) -> Optional[AdsGeneration]:
            # Check if ads exists and belongs to user
            existing = await session.execute(
                select(AdsGeneration).filter(
                    and_(
                        AdsGeneration.id == aid,
                        AdsGeneration.user_id == uid,
                        AdsGeneration.is_deleted == False
                    )
                )
            )
            if not existing.scalar_one_or_none():
                return None
            
            # Update the record
            await session.execute(
                update(AdsGeneration)
                .where(AdsGeneration.id == aid)
                .values(**data, updated_at=datetime.now())
            )
            await session.commit()
            
            # Return updated record
            result = await session.execute(
                select(AdsGeneration).filter(AdsGeneration.id == aid)
            )
            return result.scalar_one_or_none()
        
        return await self.database_manager.execute_query(_update_ads, ads_id, user_id, update_data)
    
    async def delete(self, ads_id: int, user_id: int) -> bool:
        """Soft delete an ads generation record."""
        async def _delete_ads(session: AsyncSession, aid: int, uid: int) -> bool:
            result = await session.execute(
                update(AdsGeneration)
                .where(
                    and_(
                        AdsGeneration.id == aid,
                        AdsGeneration.user_id == uid,
                        AdsGeneration.is_deleted == False
                    )
                )
                .values(is_deleted=True, deleted_at=datetime.now())
            )
            await session.commit()
            return result.rowcount > 0
        
        return await self.database_manager.execute_query(_delete_ads, ads_id, user_id)
    
    async def get_user_stats(self, user_id: int) -> Dict[str, Any]:
        """Get user statistics for ads."""
        async def _get_stats(session: AsyncSession, uid: int) -> Dict[str, Any]:
            # Total ads count
            total_result = await session.execute(
                select(func.count(AdsGeneration.id)).filter(
                    and_(
                        AdsGeneration.user_id == uid,
                        AdsGeneration.is_deleted == False
                    )
                )
            )
            total_ads = total_result.scalar() or 0
            
            # Ads by type
            type_result = await session.execute(
                select(AdsGeneration.type, func.count(AdsGeneration.id))
                .filter(
                    and_(
                        AdsGeneration.user_id == uid,
                        AdsGeneration.is_deleted == False
                    )
                )
                .group_by(AdsGeneration.type)
            )
            ads_by_type = dict(type_result.all())
            
            # Recent activity
            recent_result = await session.execute(
                select(AdsGeneration.created_at)
                .filter(
                    and_(
                        AdsGeneration.user_id == uid,
                        AdsGeneration.is_deleted == False
                    )
                )
                .order_by(desc(AdsGeneration.created_at))
                .limit(5)
            )
            recent_activity = [row[0] for row in recent_result.all()]
            
            return {
                "total_ads": total_ads,
                "ads_by_type": ads_by_type,
                "recent_activity": recent_activity
            }
        
        return await self.database_manager.execute_query(_get_stats, user_id)

class CampaignRepositoryImpl(CampaignRepository):
    """Concrete implementation of CampaignRepository."""
    
    def __init__(self, database_manager: DatabaseManager):
        self.database_manager = database_manager
    
    async def create(self, campaign_data: Dict[str, Any]) -> Any:
        """Create a new campaign."""
        # TODO: Implement when Campaign model is available
        logger.warning("Campaign creation not yet implemented")
        return None
    
    async def get_by_id(self, campaign_id: int, user_id: int) -> Optional[Any]:
        """Get a campaign by ID."""
        # TODO: Implement when Campaign model is available
        logger.warning("Campaign retrieval not yet implemented")
        return None
    
    async def list_by_user(self, user_id: int, limit: int = 100, offset: int = 0) -> List[Any]:
        """List campaigns for a user."""
        # TODO: Implement when Campaign model is available
        logger.warning("Campaign listing not yet implemented")
        return []
    
    async def update(self, campaign_id: int, user_id: int, update_data: Dict[str, Any]) -> Optional[Any]:
        """Update a campaign."""
        # TODO: Implement when Campaign model is available
        logger.warning("Campaign update not yet implemented")
        return None
    
    async def delete(self, campaign_id: int, user_id: int) -> bool:
        """Delete a campaign."""
        # TODO: Implement when Campaign model is available
        logger.warning("Campaign deletion not yet implemented")
        return False

class GroupRepositoryImpl(GroupRepository):
    """Concrete implementation of GroupRepository."""
    
    def __init__(self, database_manager: DatabaseManager):
        self.database_manager = database_manager
    
    async def create(self, group_data: Dict[str, Any]) -> Any:
        """Create a new ad group."""
        # TODO: Implement when Group model is available
        logger.warning("Group creation not yet implemented")
        return None
    
    async def get_by_id(self, group_id: int, user_id: int) -> Optional[Any]:
        """Get an ad group by ID."""
        # TODO: Implement when Group model is available
        logger.warning("Group retrieval not yet implemented")
        return None
    
    async def list_by_campaign(self, campaign_id: int, user_id: int) -> List[Any]:
        """List ad groups for a campaign."""
        # TODO: Implement when Group model is available
        logger.warning("Group listing not yet implemented")
        return []
    
    async def update(self, group_id: int, user_id: int, update_data: Dict[str, Any]) -> Optional[Any]:
        """Update an ad group."""
        # TODO: Implement when Group model is available
        logger.warning("Group update not yet implemented")
        return None
    
    async def delete(self, group_id: int, user_id: int) -> bool:
        """Delete an ad group."""
        # TODO: Implement when Group model is available
        logger.warning("Group deletion not yet implemented")
        return False

class PerformanceRepositoryImpl(PerformanceRepository):
    """Concrete implementation of PerformanceRepository."""
    
    def __init__(self, database_manager: DatabaseManager):
        self.database_manager = database_manager
    
    async def create(self, performance_data: Dict[str, Any]) -> Any:
        """Create a new performance record."""
        # TODO: Implement when Performance model is available
        logger.warning("Performance creation not yet implemented")
        return None
    
    async def get_by_id(self, performance_id: int, user_id: int) -> Optional[Any]:
        """Get a performance record by ID."""
        # TODO: Implement when Performance model is available
        logger.warning("Performance retrieval not yet implemented")
        return None
    
    async def list_by_ads(self, ads_id: int, user_id: int) -> List[Any]:
        """List performance records for an ad."""
        # TODO: Implement when Performance model is available
        logger.warning("Performance listing not yet implemented")
        return []
    
    async def update(self, performance_id: int, user_id: int, update_data: Dict[str, Any]) -> Optional[Any]:
        """Update a performance record."""
        # TODO: Implement when Performance model is available
        logger.warning("Performance update not yet implemented")
        return None

class AnalyticsRepositoryImpl(AnalyticsRepository):
    """Concrete implementation of AnalyticsRepository."""
    
    def __init__(self, database_manager: DatabaseManager):
        self.database_manager = database_manager
    
    async def create(self, analytics_data: Dict[str, Any]) -> AdsAnalytics:
        """Create a new analytics record."""
        async def _create_analytics(session: AsyncSession, data: Dict[str, Any]) -> AdsAnalytics:
            analytics = AdsAnalytics(**data)
            session.add(analytics)
            await session.commit()
            await session.refresh(analytics)
            return analytics
        
        return await self.database_manager.execute_query(_create_analytics, analytics_data)
    
    async def get_by_id(self, analytics_id: int, user_id: int) -> Optional[AdsAnalytics]:
        """Get an analytics record by ID."""
        async def _get_analytics(session: AsyncSession, aid: int, uid: int) -> Optional[AdsAnalytics]:
            result = await session.execute(
                select(AdsAnalytics).filter(
                    and_(
                        AdsAnalytics.id == aid,
                        AdsAnalytics.user_id == uid
                    )
                )
            )
            return result.scalar_one_or_none()
        
        return await self.database_manager.execute_query(_get_analytics, analytics_id, user_id)
    
    async def list_by_ads(self, ads_id: int, user_id: int) -> List[AdsAnalytics]:
        """List analytics records for an ad."""
        async def _list_analytics(session: AsyncSession, aid: int, uid: int) -> List[AdsAnalytics]:
            result = await session.execute(
                select(AdsAnalytics).filter(
                    and_(
                        AdsAnalytics.ads_generation_id == aid,
                        AdsAnalytics.user_id == uid
                    )
                ).order_by(desc(AdsAnalytics.created_at))
            )
            return result.scalars().all()
        
        return await self.database_manager.execute_query(_list_analytics, ads_id, user_id)
    
    async def get_user_stats(self, user_id: int) -> Dict[str, Any]:
        """Get user statistics."""
        async def _get_stats(session: AsyncSession, uid: int) -> Dict[str, Any]:
            # Total analytics records
            total_result = await session.execute(
                select(func.count(AdsAnalytics.id)).filter(AdsAnalytics.user_id == uid)
            )
            total_analytics = total_result.scalar() or 0
            
            # Analytics by ads
            ads_result = await session.execute(
                select(AdsAnalytics.ads_generation_id, func.count(AdsAnalytics.id))
                .filter(AdsAnalytics.user_id == uid)
                .group_by(AdsAnalytics.ads_generation_id)
            )
            analytics_by_ads = dict(ads_result.all())
            
            # Recent analytics
            recent_result = await session.execute(
                select(AdsAnalytics.created_at)
                .filter(AdsAnalytics.user_id == uid)
                .order_by(desc(AdsAnalytics.created_at))
                .limit(10)
            )
            recent_analytics = [row[0] for row in recent_result.all()]
            
            return {
                "total_analytics": total_analytics,
                "analytics_by_ads": analytics_by_ads,
                "recent_analytics": recent_analytics
            }
        
        return await self.database_manager.execute_query(_get_stats, user_id)

class OptimizationRepositoryImpl(OptimizationRepository):
    """Concrete implementation of OptimizationRepository."""
    
    def __init__(self, database_manager: DatabaseManager):
        self.database_manager = database_manager
    
    async def create(self, optimization_data: Dict[str, Any]) -> Any:
        """Create a new optimization record."""
        # TODO: Implement when Optimization model is available
        logger.warning("Optimization creation not yet implemented")
        return None
    
    async def get_by_id(self, optimization_id: int, user_id: int) -> Optional[Any]:
        """Get an optimization record by ID."""
        # TODO: Implement when Optimization model is available
        logger.warning("Optimization retrieval not yet implemented")
        return None
    
    async def list_by_ads(self, ads_id: int, user_id: int) -> List[Any]:
        """List optimization records for an ad."""
        # TODO: Implement when Optimization model is available
        logger.warning("Optimization listing not yet implemented")
        return []
    
    async def get_optimization_history(self, ads_id: int, user_id: int) -> List[Any]:
        """Get optimization history for an ad."""
        # TODO: Implement when Optimization model is available
        logger.warning("Optimization history not yet implemented")
        return []

# Repository factory for easy instantiation
class RepositoryFactory:
    """Factory for creating repository instances."""
    
    def __init__(self, database_manager: DatabaseManager):
        self.database_manager = database_manager
    
    def create_ads_repository(self) -> AdsRepositoryImpl:
        """Create an ads repository instance."""
        return AdsRepositoryImpl(self.database_manager)
    
    def create_campaign_repository(self) -> CampaignRepositoryImpl:
        """Create a campaign repository instance."""
        return CampaignRepositoryImpl(self.database_manager)
    
    def create_group_repository(self) -> GroupRepositoryImpl:
        """Create a group repository instance."""
        return GroupRepositoryImpl(self.database_manager)
    
    def create_performance_repository(self) -> PerformanceRepositoryImpl:
        """Create a performance repository instance."""
        return PerformanceRepositoryImpl(self.database_manager)
    
    def create_analytics_repository(self) -> AnalyticsRepositoryImpl:
        """Create an analytics repository instance."""
        return AnalyticsRepositoryImpl(self.database_manager)
    
    def create_optimization_repository(self) -> OptimizationRepositoryImpl:
        """Create an optimization repository instance."""
        return OptimizationRepositoryImpl(self.database_manager)
    
    def create_all_repositories(self) -> Dict[str, Any]:
        """Create all repository instances."""
        return {
            "ads": self.create_ads_repository(),
            "campaign": self.create_campaign_repository(),
            "group": self.create_group_repository(),
            "performance": self.create_performance_repository(),
            "analytics": self.create_analytics_repository(),
            "optimization": self.create_optimization_repository()
        }
