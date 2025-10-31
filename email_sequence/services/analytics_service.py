"""
Email Analytics Service

This module provides comprehensive analytics and tracking functionality
for email sequences, campaigns, and subscriber behavior.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from uuid import UUID
import json

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_
import redis.asyncio as redis

from ..core.exceptions import AnalyticsServiceError, DatabaseError, CacheError

logger = logging.getLogger(__name__)


class EmailAnalyticsService:
    """
    Service for tracking and analyzing email sequence performance.
    """
    
    def __init__(
        self,
        database_url: str,
        redis_url: str,
        batch_size: int = 1000,
        flush_interval: int = 60
    ):
        """
        Initialize the analytics service.
        
        Args:
            database_url: Database connection URL
            redis_url: Redis connection URL
            batch_size: Batch size for analytics processing
            flush_interval: Interval for flushing analytics data
        """
        self.database_url = database_url
        self.redis_url = redis_url
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        
        # Analytics data cache
        self.analytics_cache: Dict[str, Any] = {}
        self.pending_events: List[Dict[str, Any]] = []
        
        # Background tasks
        self.background_tasks: List[asyncio.Task] = []
        self.is_running = False
        
        logger.info("Email Analytics Service initialized")
    
    async def start(self) -> None:
        """Start the analytics service"""
        try:
            self.is_running = True
            
            # Start background tasks
            flush_task = asyncio.create_task(self._flush_analytics_loop())
            self.background_tasks.append(flush_task)
            
            logger.info("Email Analytics Service started")
            
        except Exception as e:
            logger.error(f"Error starting analytics service: {e}")
            raise AnalyticsServiceError(f"Failed to start analytics service: {e}")
    
    async def stop(self) -> None:
        """Stop the analytics service"""
        try:
            self.is_running = False
            
            # Cancel background tasks
            for task in self.background_tasks:
                if not task.done():
                    task.cancel()
            
            # Wait for tasks to complete
            await asyncio.gather(*self.background_tasks, return_exceptions=True)
            
            # Flush remaining analytics data
            await self._flush_pending_analytics()
            
            logger.info("Email Analytics Service stopped")
            
        except Exception as e:
            logger.error(f"Error stopping analytics service: {e}")
            raise AnalyticsServiceError(f"Failed to stop analytics service: {e}")
    
    async def track_email_sent(
        self,
        sequence_id: UUID,
        step_id: UUID,
        subscriber_id: UUID,
        email_address: str,
        subject: str,
        template_id: Optional[UUID] = None
    ) -> None:
        """
        Track email sent event.
        
        Args:
            sequence_id: Sequence ID
            step_id: Step ID
            subscriber_id: Subscriber ID
            email_address: Email address
            subject: Email subject
            template_id: Template ID
        """
        try:
            event = {
                "event_type": "email_sent",
                "sequence_id": str(sequence_id),
                "step_id": str(step_id),
                "subscriber_id": str(subscriber_id),
                "email_address": email_address,
                "subject": subject,
                "template_id": str(template_id) if template_id else None,
                "timestamp": datetime.utcnow().isoformat(),
                "metadata": {}
            }
            
            await self._add_event(event)
            
        except Exception as e:
            logger.error(f"Error tracking email sent: {e}")
            raise AnalyticsServiceError(f"Failed to track email sent: {e}")
    
    async def track_email_opened(
        self,
        sequence_id: UUID,
        step_id: UUID,
        subscriber_id: UUID,
        email_address: str,
        opened_at: Optional[datetime] = None
    ) -> None:
        """
        Track email opened event.
        
        Args:
            sequence_id: Sequence ID
            step_id: Step ID
            subscriber_id: Subscriber ID
            email_address: Email address
            opened_at: When the email was opened
        """
        try:
            event = {
                "event_type": "email_opened",
                "sequence_id": str(sequence_id),
                "step_id": str(step_id),
                "subscriber_id": str(subscriber_id),
                "email_address": email_address,
                "timestamp": (opened_at or datetime.utcnow()).isoformat(),
                "metadata": {}
            }
            
            await self._add_event(event)
            
        except Exception as e:
            logger.error(f"Error tracking email opened: {e}")
            raise AnalyticsServiceError(f"Failed to track email opened: {e}")
    
    async def track_email_clicked(
        self,
        sequence_id: UUID,
        step_id: UUID,
        subscriber_id: UUID,
        email_address: str,
        link_url: str,
        clicked_at: Optional[datetime] = None
    ) -> None:
        """
        Track email link clicked event.
        
        Args:
            sequence_id: Sequence ID
            step_id: Step ID
            subscriber_id: Subscriber ID
            email_address: Email address
            link_url: Clicked link URL
            clicked_at: When the link was clicked
        """
        try:
            event = {
                "event_type": "email_clicked",
                "sequence_id": str(sequence_id),
                "step_id": str(step_id),
                "subscriber_id": str(subscriber_id),
                "email_address": email_address,
                "link_url": link_url,
                "timestamp": (clicked_at or datetime.utcnow()).isoformat(),
                "metadata": {}
            }
            
            await self._add_event(event)
            
        except Exception as e:
            logger.error(f"Error tracking email clicked: {e}")
            raise AnalyticsServiceError(f"Failed to track email clicked: {e}")
    
    async def track_email_bounced(
        self,
        sequence_id: UUID,
        step_id: UUID,
        subscriber_id: UUID,
        email_address: str,
        bounce_type: str,
        bounce_reason: Optional[str] = None
    ) -> None:
        """
        Track email bounced event.
        
        Args:
            sequence_id: Sequence ID
            step_id: Step ID
            subscriber_id: Subscriber ID
            email_address: Email address
            bounce_type: Type of bounce (hard, soft)
            bounce_reason: Reason for bounce
        """
        try:
            event = {
                "event_type": "email_bounced",
                "sequence_id": str(sequence_id),
                "step_id": str(step_id),
                "subscriber_id": str(subscriber_id),
                "email_address": email_address,
                "bounce_type": bounce_type,
                "bounce_reason": bounce_reason,
                "timestamp": datetime.utcnow().isoformat(),
                "metadata": {}
            }
            
            await self._add_event(event)
            
        except Exception as e:
            logger.error(f"Error tracking email bounced: {e}")
            raise AnalyticsServiceError(f"Failed to track email bounced: {e}")
    
    async def track_email_unsubscribed(
        self,
        sequence_id: UUID,
        step_id: UUID,
        subscriber_id: UUID,
        email_address: str,
        unsubscribe_reason: Optional[str] = None
    ) -> None:
        """
        Track email unsubscribed event.
        
        Args:
            sequence_id: Sequence ID
            step_id: Step ID
            subscriber_id: Subscriber ID
            email_address: Email address
            unsubscribe_reason: Reason for unsubscribing
        """
        try:
            event = {
                "event_type": "email_unsubscribed",
                "sequence_id": str(sequence_id),
                "step_id": str(step_id),
                "subscriber_id": str(subscriber_id),
                "email_address": email_address,
                "unsubscribe_reason": unsubscribe_reason,
                "timestamp": datetime.utcnow().isoformat(),
                "metadata": {}
            }
            
            await self._add_event(event)
            
        except Exception as e:
            logger.error(f"Error tracking email unsubscribed: {e}")
            raise AnalyticsServiceError(f"Failed to track email unsubscribed: {e}")
    
    async def track_sequence_completed(
        self,
        sequence_id: UUID,
        subscriber_id: UUID,
        completed_at: Optional[datetime] = None
    ) -> None:
        """
        Track sequence completed event.
        
        Args:
            sequence_id: Sequence ID
            subscriber_id: Subscriber ID
            completed_at: When the sequence was completed
        """
        try:
            event = {
                "event_type": "sequence_completed",
                "sequence_id": str(sequence_id),
                "subscriber_id": str(subscriber_id),
                "timestamp": (completed_at or datetime.utcnow()).isoformat(),
                "metadata": {}
            }
            
            await self._add_event(event)
            
        except Exception as e:
            logger.error(f"Error tracking sequence completed: {e}")
            raise AnalyticsServiceError(f"Failed to track sequence completed: {e}")
    
    async def get_sequence_analytics(
        self,
        sequence_id: UUID,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get analytics for a specific sequence.
        
        Args:
            sequence_id: Sequence ID
            start_date: Start date for analytics
            end_date: End date for analytics
            
        Returns:
            Analytics data
        """
        try:
            # Set default date range if not provided
            if not end_date:
                end_date = datetime.utcnow()
            if not start_date:
                start_date = end_date - timedelta(days=30)
            
            # Check cache first
            cache_key = f"analytics:sequence:{sequence_id}:{start_date.date()}:{end_date.date()}"
            cached_data = await self._get_cached_analytics(cache_key)
            if cached_data:
                return cached_data
            
            # Get analytics from database
            analytics = await self._fetch_sequence_analytics(sequence_id, start_date, end_date)
            
            # Cache the results
            await self._cache_analytics(cache_key, analytics, ttl=1800)  # 30 minutes
            
            return analytics
            
        except Exception as e:
            logger.error(f"Error getting sequence analytics: {e}")
            raise AnalyticsServiceError(f"Failed to get sequence analytics: {e}")
    
    async def get_campaign_analytics(
        self,
        campaign_id: UUID,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get analytics for a specific campaign.
        
        Args:
            campaign_id: Campaign ID
            start_date: Start date for analytics
            end_date: End date for analytics
            
        Returns:
            Campaign analytics data
        """
        try:
            # Set default date range if not provided
            if not end_date:
                end_date = datetime.utcnow()
            if not start_date:
                start_date = end_date - timedelta(days=30)
            
            # Check cache first
            cache_key = f"analytics:campaign:{campaign_id}:{start_date.date()}:{end_date.date()}"
            cached_data = await self._get_cached_analytics(cache_key)
            if cached_data:
                return cached_data
            
            # Get analytics from database
            analytics = await self._fetch_campaign_analytics(campaign_id, start_date, end_date)
            
            # Cache the results
            await self._cache_analytics(cache_key, analytics, ttl=1800)  # 30 minutes
            
            return analytics
            
        except Exception as e:
            logger.error(f"Error getting campaign analytics: {e}")
            raise AnalyticsServiceError(f"Failed to get campaign analytics: {e}")
    
    async def get_subscriber_analytics(
        self,
        subscriber_id: UUID,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get analytics for a specific subscriber.
        
        Args:
            subscriber_id: Subscriber ID
            start_date: Start date for analytics
            end_date: End date for analytics
            
        Returns:
            Subscriber analytics data
        """
        try:
            # Set default date range if not provided
            if not end_date:
                end_date = datetime.utcnow()
            if not start_date:
                start_date = end_date - timedelta(days=30)
            
            # Check cache first
            cache_key = f"analytics:subscriber:{subscriber_id}:{start_date.date()}:{end_date.date()}"
            cached_data = await self._get_cached_analytics(cache_key)
            if cached_data:
                return cached_data
            
            # Get analytics from database
            analytics = await self._fetch_subscriber_analytics(subscriber_id, start_date, end_date)
            
            # Cache the results
            await self._cache_analytics(cache_key, analytics, ttl=1800)  # 30 minutes
            
            return analytics
            
        except Exception as e:
            logger.error(f"Error getting subscriber analytics: {e}")
            raise AnalyticsServiceError(f"Failed to get subscriber analytics: {e}")
    
    async def process_pending_analytics(self) -> None:
        """Process pending analytics events"""
        try:
            if not self.pending_events:
                return
            
            # Process events in batches
            events_to_process = self.pending_events[:self.batch_size]
            self.pending_events = self.pending_events[self.batch_size:]
            
            # Store events in database
            await self._store_events(events_to_process)
            
            logger.info(f"Processed {len(events_to_process)} analytics events")
            
        except Exception as e:
            logger.error(f"Error processing pending analytics: {e}")
            raise AnalyticsServiceError(f"Failed to process pending analytics: {e}")
    
    async def _add_event(self, event: Dict[str, Any]) -> None:
        """Add event to pending events list"""
        self.pending_events.append(event)
        
        # Process if batch size reached
        if len(self.pending_events) >= self.batch_size:
            await self.process_pending_analytics()
    
    async def _flush_analytics_loop(self) -> None:
        """Background task to flush analytics data"""
        while self.is_running:
            try:
                await asyncio.sleep(self.flush_interval)
                await self._flush_pending_analytics()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in analytics flush loop: {e}")
    
    async def _flush_pending_analytics(self) -> None:
        """Flush all pending analytics data"""
        if self.pending_events:
            await self.process_pending_analytics()
    
    async def _store_events(self, events: List[Dict[str, Any]]) -> None:
        """Store events in database"""
        try:
            # In a real implementation, you would store these in the database
            # For now, we'll just log them
            for event in events:
                logger.debug(f"Storing analytics event: {event}")
            
        except Exception as e:
            logger.error(f"Error storing events: {e}")
            raise DatabaseError(f"Failed to store analytics events: {e}")
    
    async def _fetch_sequence_analytics(
        self,
        sequence_id: UUID,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Fetch sequence analytics from database"""
        try:
            # In a real implementation, you would query the database
            # For now, return mock data
            return {
                "sequence_id": str(sequence_id),
                "date_range": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat()
                },
                "metrics": {
                    "emails_sent": 1000,
                    "emails_opened": 250,
                    "emails_clicked": 50,
                    "emails_bounced": 25,
                    "emails_unsubscribed": 10,
                    "sequences_completed": 200,
                    "open_rate": 25.0,
                    "click_rate": 5.0,
                    "bounce_rate": 2.5,
                    "unsubscribe_rate": 1.0,
                    "completion_rate": 20.0
                },
                "trends": {
                    "daily_opens": [],
                    "daily_clicks": [],
                    "daily_bounces": []
                },
                "top_performing_steps": [],
                "subscriber_segments": {}
            }
            
        except Exception as e:
            logger.error(f"Error fetching sequence analytics: {e}")
            raise DatabaseError(f"Failed to fetch sequence analytics: {e}")
    
    async def _fetch_campaign_analytics(
        self,
        campaign_id: UUID,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Fetch campaign analytics from database"""
        try:
            # In a real implementation, you would query the database
            # For now, return mock data
            return {
                "campaign_id": str(campaign_id),
                "date_range": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat()
                },
                "metrics": {
                    "emails_sent": 5000,
                    "emails_opened": 1250,
                    "emails_clicked": 250,
                    "emails_bounced": 125,
                    "emails_unsubscribed": 50,
                    "open_rate": 25.0,
                    "click_rate": 5.0,
                    "bounce_rate": 2.5,
                    "unsubscribe_rate": 1.0
                },
                "trends": {
                    "daily_opens": [],
                    "daily_clicks": [],
                    "daily_bounces": []
                },
                "top_performing_sequences": [],
                "subscriber_segments": {}
            }
            
        except Exception as e:
            logger.error(f"Error fetching campaign analytics: {e}")
            raise DatabaseError(f"Failed to fetch campaign analytics: {e}")
    
    async def _fetch_subscriber_analytics(
        self,
        subscriber_id: UUID,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Fetch subscriber analytics from database"""
        try:
            # In a real implementation, you would query the database
            # For now, return mock data
            return {
                "subscriber_id": str(subscriber_id),
                "date_range": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat()
                },
                "metrics": {
                    "emails_received": 10,
                    "emails_opened": 7,
                    "emails_clicked": 3,
                    "sequences_started": 2,
                    "sequences_completed": 1,
                    "open_rate": 70.0,
                    "click_rate": 30.0,
                    "completion_rate": 50.0
                },
                "activity_timeline": [],
                "preferred_content_types": [],
                "engagement_score": 85.0
            }
            
        except Exception as e:
            logger.error(f"Error fetching subscriber analytics: {e}")
            raise DatabaseError(f"Failed to fetch subscriber analytics: {e}")
    
    async def _get_cached_analytics(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached analytics data"""
        try:
            # In a real implementation, you would use Redis
            return self.analytics_cache.get(cache_key)
        except Exception as e:
            logger.warning(f"Error getting cached analytics: {e}")
            return None
    
    async def _cache_analytics(self, cache_key: str, data: Dict[str, Any], ttl: int = 1800) -> None:
        """Cache analytics data"""
        try:
            # In a real implementation, you would use Redis
            self.analytics_cache[cache_key] = data
        except Exception as e:
            logger.warning(f"Error caching analytics: {e}")
    
    async def close(self) -> None:
        """Close the analytics service"""
        await self.stop()






























