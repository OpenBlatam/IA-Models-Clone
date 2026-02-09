"""
Scheduling & Publishing Automation System

Advanced scheduling and publishing automation for the Ultimate Opus Clip system including
social media scheduling, content calendar management, and automated publishing.
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Union, Callable
import asyncio
import uuid
import time
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from enum import Enum
import structlog
import json
import sqlite3
import threading
from pathlib import Path
import aiohttp
import aiofiles
from croniter import croniter
import pytz
from collections import defaultdict
import hashlib
import hmac
import base64
import urllib.parse

logger = structlog.get_logger("scheduling_automation")

class Platform(Enum):
    """Supported social media platforms."""
    YOUTUBE = "youtube"
    FACEBOOK = "facebook"
    INSTAGRAM = "instagram"
    TWITTER = "twitter"
    TIKTOK = "tiktok"
    LINKEDIN = "linkedin"
    TWITCH = "twitch"
    DISCORD = "discord"

class PostStatus(Enum):
    """Post status states."""
    DRAFT = "draft"
    SCHEDULED = "scheduled"
    PUBLISHING = "publishing"
    PUBLISHED = "published"
    FAILED = "failed"
    CANCELLED = "cancelled"

class ScheduleType(Enum):
    """Schedule types."""
    IMMEDIATE = "immediate"
    SCHEDULED = "scheduled"
    RECURRING = "recurring"
    OPTIMAL_TIME = "optimal_time"

@dataclass
class SocialMediaAccount:
    """Social media account information."""
    account_id: str
    platform: Platform
    username: str
    display_name: str
    access_token: str
    refresh_token: Optional[str] = None
    token_expires_at: Optional[datetime] = None
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    last_sync: Optional[datetime] = None

@dataclass
class ScheduledPost:
    """Scheduled post information."""
    post_id: str
    account_id: str
    platform: Platform
    content: str
    media_path: str
    scheduled_at: datetime
    status: PostStatus = PostStatus.DRAFT
    created_at: datetime = field(default_factory=datetime.now)
    published_at: Optional[datetime] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    hashtags: List[str] = field(default_factory=list)
    mentions: List[str] = field(default_factory=list)
    location: Optional[Dict[str, float]] = None

@dataclass
class ContentCalendar:
    """Content calendar information."""
    calendar_id: str
    name: str
    description: str
    owner_id: str
    timezone: str = "UTC"
    created_at: datetime = field(default_factory=datetime.now)
    settings: Dict[str, Any] = field(default_factory=dict)

@dataclass
class OptimalTimeSlot:
    """Optimal posting time slot."""
    platform: Platform
    day_of_week: int  # 0-6 (Monday-Sunday)
    hour: int  # 0-23
    engagement_score: float
    audience_size: int
    timezone: str

class SocialMediaAPI:
    """Base class for social media API integrations."""
    
    def __init__(self, platform: Platform):
        self.platform = platform
        self.logger = structlog.get_logger(f"social_media_api_{platform.value}")
    
    async def authenticate(self, access_token: str) -> bool:
        """Authenticate with the platform."""
        raise NotImplementedError
    
    async def publish_post(self, post: ScheduledPost) -> Dict[str, Any]:
        """Publish a post to the platform."""
        raise NotImplementedError
    
    async def get_account_info(self, access_token: str) -> Dict[str, Any]:
        """Get account information."""
        raise NotImplementedError
    
    async def validate_media(self, media_path: str) -> bool:
        """Validate media file for the platform."""
        raise NotImplementedError

class YouTubeAPI(SocialMediaAPI):
    """YouTube API integration."""
    
    def __init__(self):
        super().__init__(Platform.YOUTUBE)
        self.base_url = "https://www.googleapis.com/youtube/v3"
    
    async def authenticate(self, access_token: str) -> bool:
        """Authenticate with YouTube API."""
        try:
            async with aiohttp.ClientSession() as session:
                headers = {"Authorization": f"Bearer {access_token}"}
                async with session.get(f"{self.base_url}/channels?part=snippet&mine=true", headers=headers) as response:
                    return response.status == 200
        except Exception as e:
            self.logger.error(f"YouTube authentication failed: {e}")
            return False
    
    async def publish_post(self, post: ScheduledPost) -> Dict[str, Any]:
        """Publish video to YouTube."""
        try:
            # This would implement actual YouTube upload
            # For now, return mock response
            return {
                "success": True,
                "video_id": f"yt_{uuid.uuid4().hex[:11]}",
                "url": f"https://youtube.com/watch?v=yt_{uuid.uuid4().hex[:11]}",
                "published_at": datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"YouTube publish failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def get_account_info(self, access_token: str) -> Dict[str, Any]:
        """Get YouTube channel information."""
        try:
            async with aiohttp.ClientSession() as session:
                headers = {"Authorization": f"Bearer {access_token}"}
                async with session.get(f"{self.base_url}/channels?part=snippet&mine=true", headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get("items"):
                            channel = data["items"][0]
                            return {
                                "id": channel["id"],
                                "title": channel["snippet"]["title"],
                                "description": channel["snippet"]["description"],
                                "thumbnail": channel["snippet"]["thumbnails"]["default"]["url"]
                            }
            return {}
        except Exception as e:
            self.logger.error(f"Failed to get YouTube account info: {e}")
            return {}
    
    async def validate_media(self, media_path: str) -> bool:
        """Validate video file for YouTube."""
        try:
            # Check file size (YouTube limit: 128GB)
            file_size = Path(media_path).stat().st_size
            if file_size > 128 * 1024 * 1024 * 1024:  # 128GB
                return False
            
            # Check file format
            valid_formats = ['.mp4', '.mov', '.avi', '.wmv', '.flv', '.webm']
            return Path(media_path).suffix.lower() in valid_formats
        except Exception as e:
            self.logger.error(f"YouTube media validation failed: {e}")
            return False

class FacebookAPI(SocialMediaAPI):
    """Facebook API integration."""
    
    def __init__(self):
        super().__init__(Platform.FACEBOOK)
        self.base_url = "https://graph.facebook.com/v18.0"
    
    async def authenticate(self, access_token: str) -> bool:
        """Authenticate with Facebook API."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/me?access_token={access_token}") as response:
                    return response.status == 200
        except Exception as e:
            self.logger.error(f"Facebook authentication failed: {e}")
            return False
    
    async def publish_post(self, post: ScheduledPost) -> Dict[str, Any]:
        """Publish post to Facebook."""
        try:
            # This would implement actual Facebook posting
            # For now, return mock response
            return {
                "success": True,
                "post_id": f"fb_{uuid.uuid4().hex}",
                "url": f"https://facebook.com/posts/fb_{uuid.uuid4().hex}",
                "published_at": datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Facebook publish failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def get_account_info(self, access_token: str) -> Dict[str, Any]:
        """Get Facebook page information."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/me?fields=id,name&access_token={access_token}") as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            "id": data["id"],
                            "name": data["name"]
                        }
            return {}
        except Exception as e:
            self.logger.error(f"Failed to get Facebook account info: {e}")
            return {}
    
    async def validate_media(self, media_path: str) -> bool:
        """Validate media file for Facebook."""
        try:
            # Check file size (Facebook limit: 4GB)
            file_size = Path(media_path).stat().st_size
            if file_size > 4 * 1024 * 1024 * 1024:  # 4GB
                return False
            
            # Check file format
            valid_formats = ['.mp4', '.mov', '.avi', '.wmv', '.flv', '.webm', '.gif']
            return Path(media_path).suffix.lower() in valid_formats
        except Exception as e:
            self.logger.error(f"Facebook media validation failed: {e}")
            return False

class InstagramAPI(SocialMediaAPI):
    """Instagram API integration."""
    
    def __init__(self):
        super().__init__(Platform.INSTAGRAM)
        self.base_url = "https://graph.facebook.com/v18.0"
    
    async def authenticate(self, access_token: str) -> bool:
        """Authenticate with Instagram API."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/me?access_token={access_token}") as response:
                    return response.status == 200
        except Exception as e:
            self.logger.error(f"Instagram authentication failed: {e}")
            return False
    
    async def publish_post(self, post: ScheduledPost) -> Dict[str, Any]:
        """Publish post to Instagram."""
        try:
            # Instagram requires a two-step process: create media container, then publish
            # This is a simplified version
            return {
                "success": True,
                "post_id": f"ig_{uuid.uuid4().hex}",
                "url": f"https://instagram.com/p/ig_{uuid.uuid4().hex}",
                "published_at": datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Instagram publish failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def get_account_info(self, access_token: str) -> Dict[str, Any]:
        """Get Instagram account information."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/me?fields=id,username&access_token={access_token}") as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            "id": data["id"],
                            "username": data["username"]
                        }
            return {}
        except Exception as e:
            self.logger.error(f"Failed to get Instagram account info: {e}")
            return {}
    
    async def validate_media(self, media_path: str) -> bool:
        """Validate media file for Instagram."""
        try:
            # Check file size (Instagram limit: 100MB)
            file_size = Path(media_path).stat().st_size
            if file_size > 100 * 1024 * 1024:  # 100MB
                return False
            
            # Check file format
            valid_formats = ['.mp4', '.mov', '.avi']
            return Path(media_path).suffix.lower() in valid_formats
        except Exception as e:
            self.logger.error(f"Instagram media validation failed: {e}")
            return False

class TikTokAPI(SocialMediaAPI):
    """TikTok API integration."""
    
    def __init__(self):
        super().__init__(Platform.TIKTOK)
        self.base_url = "https://open-api.tiktok.com"
    
    async def authenticate(self, access_token: str) -> bool:
        """Authenticate with TikTok API."""
        try:
            async with aiohttp.ClientSession() as session:
                headers = {"Authorization": f"Bearer {access_token}"}
                async with session.get(f"{self.base_url}/user/info/", headers=headers) as response:
                    return response.status == 200
        except Exception as e:
            self.logger.error(f"TikTok authentication failed: {e}")
            return False
    
    async def publish_post(self, post: ScheduledPost) -> Dict[str, Any]:
        """Publish video to TikTok."""
        try:
            # TikTok API implementation would go here
            return {
                "success": True,
                "video_id": f"tt_{uuid.uuid4().hex}",
                "url": f"https://tiktok.com/@user/video/tt_{uuid.uuid4().hex}",
                "published_at": datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"TikTok publish failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def get_account_info(self, access_token: str) -> Dict[str, Any]:
        """Get TikTok account information."""
        try:
            async with aiohttp.ClientSession() as session:
                headers = {"Authorization": f"Bearer {access_token}"}
                async with session.get(f"{self.base_url}/user/info/", headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            "id": data.get("data", {}).get("open_id"),
                            "username": data.get("data", {}).get("display_name")
                        }
            return {}
        except Exception as e:
            self.logger.error(f"Failed to get TikTok account info: {e}")
            return {}
    
    async def validate_media(self, media_path: str) -> bool:
        """Validate media file for TikTok."""
        try:
            # Check file size (TikTok limit: 287MB)
            file_size = Path(media_path).stat().st_size
            if file_size > 287 * 1024 * 1024:  # 287MB
                return False
            
            # Check file format
            valid_formats = ['.mp4', '.mov', '.avi']
            return Path(media_path).suffix.lower() in valid_formats
        except Exception as e:
            self.logger.error(f"TikTok media validation failed: {e}")
            return False

class OptimalTimeAnalyzer:
    """Analyzes optimal posting times based on audience data."""
    
    def __init__(self):
        self.audience_data: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.optimal_times: Dict[Platform, List[OptimalTimeSlot]] = {}
        self.logger = structlog.get_logger("optimal_time_analyzer")
    
    async def analyze_audience_activity(self, platform: Platform, account_id: str, 
                                      audience_data: List[Dict[str, Any]]):
        """Analyze audience activity patterns."""
        try:
            self.audience_data[f"{platform.value}_{account_id}"] = audience_data
            
            # Analyze optimal times
            optimal_times = await self._calculate_optimal_times(platform, audience_data)
            self.optimal_times[platform] = optimal_times
            
            self.logger.info(f"Analyzed audience activity for {platform.value}")
            
        except Exception as e:
            self.logger.error(f"Failed to analyze audience activity: {e}")
    
    async def _calculate_optimal_times(self, platform: Platform, 
                                     audience_data: List[Dict[str, Any]]) -> List[OptimalTimeSlot]:
        """Calculate optimal posting times."""
        try:
            # Group data by day and hour
            activity_by_time = defaultdict(lambda: defaultdict(int))
            
            for data_point in audience_data:
                timestamp = datetime.fromisoformat(data_point["timestamp"])
                day_of_week = timestamp.weekday()
                hour = timestamp.hour
                activity_by_time[day_of_week][hour] += data_point.get("engagement", 1)
            
            # Calculate optimal times
            optimal_times = []
            
            for day_of_week in range(7):
                for hour in range(24):
                    engagement_score = activity_by_time[day_of_week][hour]
                    if engagement_score > 0:
                        optimal_times.append(OptimalTimeSlot(
                            platform=platform,
                            day_of_week=day_of_week,
                            hour=hour,
                            engagement_score=engagement_score,
                            audience_size=len(audience_data),
                            timezone="UTC"
                        ))
            
            # Sort by engagement score
            optimal_times.sort(key=lambda x: x.engagement_score, reverse=True)
            
            return optimal_times[:10]  # Top 10 optimal times
            
        except Exception as e:
            self.logger.error(f"Failed to calculate optimal times: {e}")
            return []
    
    async def get_optimal_time(self, platform: Platform, day_of_week: int = None) -> Optional[OptimalTimeSlot]:
        """Get optimal posting time for platform and day."""
        try:
            if platform not in self.optimal_times:
                return None
            
            optimal_times = self.optimal_times[platform]
            
            if day_of_week is not None:
                # Filter by specific day
                day_times = [t for t in optimal_times if t.day_of_week == day_of_week]
                return day_times[0] if day_times else None
            else:
                # Return best overall time
                return optimal_times[0] if optimal_times else None
                
        except Exception as e:
            self.logger.error(f"Failed to get optimal time: {e}")
            return None

class ContentScheduler:
    """Manages content scheduling and publishing."""
    
    def __init__(self, db_path: str = "scheduling.db"):
        self.db_path = db_path
        self.scheduled_posts: Dict[str, ScheduledPost] = {}
        self.social_media_apis: Dict[Platform, SocialMediaAPI] = {
            Platform.YOUTUBE: YouTubeAPI(),
            Platform.FACEBOOK: FacebookAPI(),
            Platform.INSTAGRAM: InstagramAPI(),
            Platform.TIKTOK: TikTokAPI()
        }
        self.optimal_time_analyzer = OptimalTimeAnalyzer()
        self.running = False
        self._scheduler_task: Optional[asyncio.Task] = None
        self._init_database()
        self.logger = structlog.get_logger("content_scheduler")
    
    def _init_database(self):
        """Initialize scheduling database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Scheduled posts table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS scheduled_posts (
                        post_id TEXT PRIMARY KEY,
                        account_id TEXT NOT NULL,
                        platform TEXT NOT NULL,
                        content TEXT NOT NULL,
                        media_path TEXT NOT NULL,
                        scheduled_at TIMESTAMP NOT NULL,
                        status TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        published_at TIMESTAMP,
                        error_message TEXT,
                        metadata TEXT,
                        hashtags TEXT,
                        mentions TEXT,
                        location TEXT
                    )
                """)
                
                # Social media accounts table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS social_media_accounts (
                        account_id TEXT PRIMARY KEY,
                        platform TEXT NOT NULL,
                        username TEXT NOT NULL,
                        display_name TEXT NOT NULL,
                        access_token TEXT NOT NULL,
                        refresh_token TEXT,
                        token_expires_at TIMESTAMP,
                        is_active BOOLEAN DEFAULT TRUE,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_sync TIMESTAMP
                    )
                """)
                
                conn.commit()
        except Exception as e:
            self.logger.error(f"Failed to initialize scheduling database: {e}")
    
    async def start(self):
        """Start the content scheduler."""
        try:
            self.running = True
            self._scheduler_task = asyncio.create_task(self._scheduler_loop())
            self.logger.info("Content scheduler started")
            
        except Exception as e:
            self.logger.error(f"Failed to start content scheduler: {e}")
            raise
    
    async def stop(self):
        """Stop the content scheduler."""
        try:
            self.running = False
            
            if self._scheduler_task:
                self._scheduler_task.cancel()
                try:
                    await self._scheduler_task
                except asyncio.CancelledError:
                    pass
            
            self.logger.info("Content scheduler stopped")
            
        except Exception as e:
            self.logger.error(f"Failed to stop content scheduler: {e}")
    
    async def _scheduler_loop(self):
        """Main scheduler loop."""
        while self.running:
            try:
                # Check for posts ready to publish
                await self._process_scheduled_posts()
                
                # Wait before next check
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error in scheduler loop: {e}")
                await asyncio.sleep(30)
    
    async def _process_scheduled_posts(self):
        """Process posts that are ready to publish."""
        try:
            current_time = datetime.now(timezone.utc)
            
            # Get posts ready to publish
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT * FROM scheduled_posts 
                    WHERE status = ? AND scheduled_at <= ?
                    ORDER BY scheduled_at
                """, (PostStatus.SCHEDULED.value, current_time.isoformat()))
                
                rows = cursor.fetchall()
                
                for row in rows:
                    post_id = row[0]
                    await self._publish_post(post_id)
                    
        except Exception as e:
            self.logger.error(f"Failed to process scheduled posts: {e}")
    
    async def _publish_post(self, post_id: str):
        """Publish a scheduled post."""
        try:
            # Get post data
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT * FROM scheduled_posts WHERE post_id = ?", (post_id,))
                row = cursor.fetchone()
                
                if not row:
                    return
                
                # Create ScheduledPost object
                post = ScheduledPost(
                    post_id=row[0],
                    account_id=row[1],
                    platform=Platform(row[2]),
                    content=row[3],
                    media_path=row[4],
                    scheduled_at=datetime.fromisoformat(row[5]),
                    status=PostStatus(row[6]),
                    created_at=datetime.fromisoformat(row[7]),
                    published_at=datetime.fromisoformat(row[8]) if row[8] else None,
                    error_message=row[9],
                    metadata=json.loads(row[10]) if row[10] else {},
                    hashtags=json.loads(row[11]) if row[11] else [],
                    mentions=json.loads(row[12]) if row[12] else [],
                    location=json.loads(row[13]) if row[13] else None
                )
                
                # Update status to publishing
                conn.execute(
                    "UPDATE scheduled_posts SET status = ? WHERE post_id = ?",
                    (PostStatus.PUBLISHING.value, post_id)
                )
                conn.commit()
            
            # Get API for platform
            api = self.social_media_apis.get(post.platform)
            if not api:
                raise Exception(f"No API available for platform: {post.platform}")
            
            # Validate media
            if not await api.validate_media(post.media_path):
                raise Exception(f"Invalid media file for platform: {post.platform}")
            
            # Publish post
            result = await api.publish_post(post)
            
            # Update post status
            with sqlite3.connect(self.db_path) as conn:
                if result.get("success"):
                    conn.execute("""
                        UPDATE scheduled_posts 
                        SET status = ?, published_at = ?, metadata = ?
                        WHERE post_id = ?
                    """, (
                        PostStatus.PUBLISHED.value,
                        datetime.now().isoformat(),
                        json.dumps(result),
                        post_id
                    ))
                else:
                    conn.execute("""
                        UPDATE scheduled_posts 
                        SET status = ?, error_message = ?
                        WHERE post_id = ?
                    """, (
                        PostStatus.FAILED.value,
                        result.get("error", "Unknown error"),
                        post_id
                    ))
                conn.commit()
            
            self.logger.info(f"Published post {post_id} to {post.platform.value}")
            
        except Exception as e:
            self.logger.error(f"Failed to publish post {post_id}: {e}")
            
            # Update post status to failed
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        UPDATE scheduled_posts 
                        SET status = ?, error_message = ?
                        WHERE post_id = ?
                    """, (PostStatus.FAILED.value, str(e), post_id))
                    conn.commit()
            except Exception as db_error:
                self.logger.error(f"Failed to update post status: {db_error}")
    
    async def schedule_post(self, account_id: str, platform: Platform, content: str,
                          media_path: str, scheduled_at: datetime, hashtags: List[str] = None,
                          mentions: List[str] = None, location: Dict[str, float] = None,
                          metadata: Dict[str, Any] = None) -> str:
        """Schedule a post for publishing."""
        try:
            post_id = str(uuid.uuid4())
            
            post = ScheduledPost(
                post_id=post_id,
                account_id=account_id,
                platform=platform,
                content=content,
                media_path=media_path,
                scheduled_at=scheduled_at,
                hashtags=hashtags or [],
                mentions=mentions or [],
                location=location,
                metadata=metadata or {}
            )
            
            # Save to database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO scheduled_posts (post_id, account_id, platform, content, 
                                              media_path, scheduled_at, status, hashtags, 
                                              mentions, location, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    post_id, account_id, platform.value, content, media_path,
                    scheduled_at.isoformat(), PostStatus.SCHEDULED.value,
                    json.dumps(hashtags or []), json.dumps(mentions or []),
                    json.dumps(location) if location else None,
                    json.dumps(metadata or {})
                ))
                conn.commit()
            
            self.scheduled_posts[post_id] = post
            self.logger.info(f"Scheduled post {post_id} for {scheduled_at}")
            return post_id
            
        except Exception as e:
            self.logger.error(f"Failed to schedule post: {e}")
            raise
    
    async def get_optimal_schedule_time(self, platform: Platform, 
                                      preferred_time: datetime = None) -> datetime:
        """Get optimal scheduling time for a platform."""
        try:
            if preferred_time:
                return preferred_time
            
            # Get optimal time from analyzer
            optimal_slot = await self.optimal_time_analyzer.get_optimal_time(platform)
            
            if optimal_slot:
                # Calculate next occurrence of optimal time
                now = datetime.now(timezone.utc)
                days_ahead = (optimal_slot.day_of_week - now.weekday()) % 7
                if days_ahead == 0 and now.hour >= optimal_slot.hour:
                    days_ahead = 7
                
                optimal_time = now + timedelta(days=days_ahead)
                optimal_time = optimal_time.replace(hour=optimal_slot.hour, minute=0, second=0, microsecond=0)
                
                return optimal_time
            
            # Fallback to next hour
            return datetime.now(timezone.utc) + timedelta(hours=1)
            
        except Exception as e:
            self.logger.error(f"Failed to get optimal schedule time: {e}")
            return datetime.now(timezone.utc) + timedelta(hours=1)
    
    async def get_scheduled_posts(self, account_id: str = None, 
                                status: PostStatus = None) -> List[ScheduledPost]:
        """Get scheduled posts."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = "SELECT * FROM scheduled_posts WHERE 1=1"
                params = []
                
                if account_id:
                    query += " AND account_id = ?"
                    params.append(account_id)
                
                if status:
                    query += " AND status = ?"
                    params.append(status.value)
                
                query += " ORDER BY scheduled_at"
                
                cursor = conn.execute(query, params)
                rows = cursor.fetchall()
                
                posts = []
                for row in rows:
                    post = ScheduledPost(
                        post_id=row[0],
                        account_id=row[1],
                        platform=Platform(row[2]),
                        content=row[3],
                        media_path=row[4],
                        scheduled_at=datetime.fromisoformat(row[5]),
                        status=PostStatus(row[6]),
                        created_at=datetime.fromisoformat(row[7]),
                        published_at=datetime.fromisoformat(row[8]) if row[8] else None,
                        error_message=row[9],
                        metadata=json.loads(row[10]) if row[10] else {},
                        hashtags=json.loads(row[11]) if row[11] else [],
                        mentions=json.loads(row[12]) if row[12] else [],
                        location=json.loads(row[13]) if row[13] else None
                    )
                    posts.append(post)
                
                return posts
                
        except Exception as e:
            self.logger.error(f"Failed to get scheduled posts: {e}")
            return []

class SchedulingAutomationSystem:
    """Main scheduling and publishing automation system."""
    
    def __init__(self, db_path: str = "scheduling.db"):
        self.content_scheduler = ContentScheduler(db_path)
        self.optimal_time_analyzer = OptimalTimeAnalyzer()
        self.logger = structlog.get_logger("scheduling_automation_system")
    
    async def start(self):
        """Start the scheduling automation system."""
        try:
            await self.content_scheduler.start()
            self.logger.info("Scheduling automation system started")
            
        except Exception as e:
            self.logger.error(f"Failed to start scheduling automation system: {e}")
            raise
    
    async def stop(self):
        """Stop the scheduling automation system."""
        try:
            await self.content_scheduler.stop()
            self.logger.info("Scheduling automation system stopped")
            
        except Exception as e:
            self.logger.error(f"Failed to stop scheduling automation system: {e}")
    
    async def schedule_video_post(self, video_path: str, platforms: List[Platform],
                                content: str, scheduled_at: datetime = None,
                                hashtags: List[str] = None, mentions: List[str] = None) -> List[str]:
        """Schedule a video post across multiple platforms."""
        try:
            post_ids = []
            
            for platform in platforms:
                # Get optimal time if not specified
                if scheduled_at is None:
                    scheduled_at = await self.content_scheduler.get_optimal_schedule_time(platform)
                
                # Schedule post
                post_id = await self.content_scheduler.schedule_post(
                    account_id=f"default_{platform.value}",  # Would use actual account ID
                    platform=platform,
                    content=content,
                    media_path=video_path,
                    scheduled_at=scheduled_at,
                    hashtags=hashtags,
                    mentions=mentions
                )
                
                post_ids.append(post_id)
            
            self.logger.info(f"Scheduled video post across {len(platforms)} platforms")
            return post_ids
            
        except Exception as e:
            self.logger.error(f"Failed to schedule video post: {e}")
            raise
    
    async def get_publishing_calendar(self, start_date: datetime, end_date: datetime) -> Dict[str, List[Dict[str, Any]]]:
        """Get publishing calendar for date range."""
        try:
            posts = await self.content_scheduler.get_scheduled_posts()
            
            calendar = defaultdict(list)
            
            for post in posts:
                if start_date <= post.scheduled_at <= end_date:
                    calendar[post.scheduled_at.date().isoformat()].append({
                        "post_id": post.post_id,
                        "platform": post.platform.value,
                        "content": post.content,
                        "scheduled_at": post.scheduled_at.isoformat(),
                        "status": post.status.value
                    })
            
            return dict(calendar)
            
        except Exception as e:
            self.logger.error(f"Failed to get publishing calendar: {e}")
            return {}
    
    async def analyze_optimal_times(self, platform: Platform, account_id: str,
                                  audience_data: List[Dict[str, Any]]):
        """Analyze optimal posting times for a platform."""
        try:
            await self.optimal_time_analyzer.analyze_audience_activity(
                platform, account_id, audience_data
            )
            
            self.logger.info(f"Analyzed optimal times for {platform.value}")
            
        except Exception as e:
            self.logger.error(f"Failed to analyze optimal times: {e}")

# Global scheduling automation system instance
scheduling_automation_system = SchedulingAutomationSystem()

# Export classes
__all__ = [
    "SchedulingAutomationSystem",
    "ContentScheduler",
    "OptimalTimeAnalyzer",
    "SocialMediaAPI",
    "YouTubeAPI",
    "FacebookAPI",
    "InstagramAPI",
    "TikTokAPI",
    "SocialMediaAccount",
    "ScheduledPost",
    "ContentCalendar",
    "OptimalTimeSlot",
    "Platform",
    "PostStatus",
    "ScheduleType",
    "scheduling_automation_system"
]


