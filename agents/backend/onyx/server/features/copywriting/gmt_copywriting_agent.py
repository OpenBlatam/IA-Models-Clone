from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import os
import sys
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import time
import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
    import orjson
    import json as orjson
    import pytz
    import redis.asyncio as aioredis
from .models import (
import structlog
            import pytz
                    import pytz
            from .final_main import get_service
    import uvicorn
from typing import Any, List, Dict, Optional
import logging
"""
GMT Copywriting Agent Platform.

Global time-aware copywriting agent for managing content generation across timezones:
- Multi-timezone content scheduling
- Global campaign coordination
- Time-sensitive content optimization
- Regional content adaptation
- Performance tracking across time zones
"""


# FastAPI and dependencies

# High-performance imports
try:
    JSON_AVAILABLE = True
except ImportError:
    JSON_AVAILABLE = False

try:
    PYTZ_AVAILABLE = True
except ImportError:
    PYTZ_AVAILABLE = False

try:
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

# Import copywriting models
    CopywritingInput, CopywritingOutput, Language, CopyTone, 
    UseCase, CopyVariant, WebsiteInfo, BrandVoice
)

# Logging
logger = structlog.get_logger(__name__)

# === TIME ZONE DEFINITIONS ===
class TimeZone(str, Enum):
    """Supported time zones for global operations."""
    GMT = "GMT"
    UTC = "UTC"
    EST = "America/New_York"
    PST = "America/Los_Angeles"
    CST = "America/Chicago"
    MST = "America/Denver"
    CET = "Europe/Paris"
    JST = "Asia/Tokyo"
    IST = "Asia/Kolkata"
    AEST = "Australia/Sydney"
    BST = "Europe/London"
    CAT = "Africa/Cairo"
    BRT = "America/Sao_Paulo"
    ART = "America/Argentina/Buenos_Aires"
    COT = "America/Bogota"
    PET = "America/Lima"
    CLT = "America/Santiago"
    VET = "America/Caracas"

class ContentPriority(str, Enum):
    """Content generation priority levels."""
    URGENT = "urgent"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"
    SCHEDULED = "scheduled"

class CampaignType(str, Enum):
    """Campaign types for time-based operations."""
    GLOBAL_LAUNCH = "global_launch"
    REGIONAL_PROMO = "regional_promo"
    TIME_SENSITIVE = "time_sensitive"
    EVERGREEN = "evergreen"
    SEASONAL = "seasonal"
    EVENT_BASED = "event_based"

# === MODELS ===
class TimeZoneInfo(BaseModel):
    """Time zone information model."""
    timezone: TimeZone = Field(..., description="Time zone identifier")
    current_time: datetime = Field(..., description="Current time in timezone")
    utc_offset: str = Field(..., description="UTC offset")
    is_dst: bool = Field(False, description="Is daylight saving time active")
    local_hour: int = Field(..., ge=0, le=23, description="Current hour in local time")
    business_hours: bool = Field(False, description="Is within business hours (9-17)")
    prime_time: bool = Field(False, description="Is prime time for content (18-22)")

class GlobalSchedule(BaseModel):
    """Global scheduling model."""
    schedule_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    campaign_type: CampaignType = Field(..., description="Type of campaign")
    target_timezones: List[TimeZone] = Field(..., description="Target time zones")
    scheduled_times: Dict[str, datetime] = Field(..., description="Scheduled times per timezone")
    content_variations: Dict[str, str] = Field(default_factory=dict, description="Content per timezone")
    priority: ContentPriority = Field(ContentPriority.NORMAL, description="Execution priority")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = Field(None, description="Expiration time")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class GMTCopywritingRequest(BaseModel):
    """GMT-aware copywriting request."""
    base_request: CopywritingInput = Field(..., description="Base copywriting request")
    target_timezone: TimeZone = Field(TimeZone.GMT, description="Target timezone")
    schedule_time: Optional[datetime] = Field(None, description="Scheduled execution time")
    priority: ContentPriority = Field(ContentPriority.NORMAL, description="Request priority")
    campaign_id: Optional[str] = Field(None, description="Campaign identifier")
    regional_adaptation: bool = Field(True, description="Enable regional content adaptation")
    time_sensitive: bool = Field(False, description="Is time-sensitive content")
    business_hours_only: bool = Field(False, description="Execute only during business hours")
    
class GMTCopywritingResponse(BaseModel):
    """GMT-aware copywriting response."""
    request_id: str = Field(..., description="Request identifier")
    output: CopywritingOutput = Field(..., description="Generated content")
    timezone_info: TimeZoneInfo = Field(..., description="Timezone information")
    execution_time: datetime = Field(..., description="Actual execution time")
    regional_adaptations: Dict[str, str] = Field(default_factory=dict, description="Regional adaptations applied")
    performance_metrics: Dict[str, Any] = Field(default_factory=dict, description="Performance metrics")

# === GMT AGENT CORE ===
class GMTTimeManager:
    """Advanced time management for global operations."""
    
    def __init__(self) -> Any:
        self.timezone_cache = {}
        self.business_hours = {
            "start": 9,  # 9 AM
            "end": 17    # 5 PM
        }
        self.prime_hours = {
            "start": 18,  # 6 PM
            "end": 22     # 10 PM
        }
    
    def get_timezone_info(self, tz: TimeZone) -> TimeZoneInfo:
        """Get comprehensive timezone information."""
        if not PYTZ_AVAILABLE:
            # Fallback without pytz
            now = datetime.now(timezone.utc)
            return TimeZoneInfo(
                timezone=tz,
                current_time=now,
                utc_offset="+00:00",
                is_dst=False,
                local_hour=now.hour,
                business_hours=9 <= now.hour <= 17,
                prime_time=18 <= now.hour <= 22
            )
        
        try:
            
            # Get timezone object
            if tz == TimeZone.GMT or tz == TimeZone.UTC:
                tz_obj = pytz.UTC
            else:
                tz_obj = pytz.timezone(tz.value)
            
            # Current time in timezone
            utc_now = datetime.now(timezone.utc)
            local_time = utc_now.astimezone(tz_obj)
            
            # Calculate offset
            offset = local_time.utcoffset()
            offset_str = f"{offset.total_seconds() / 3600:+03.0f}:00"
            
            # Check DST
            is_dst = bool(local_time.dst())
            
            # Business and prime time checks
            local_hour = local_time.hour
            business_hours = self.business_hours["start"] <= local_hour <= self.business_hours["end"]
            prime_time = self.prime_hours["start"] <= local_hour <= self.prime_hours["end"]
            
            return TimeZoneInfo(
                timezone=tz,
                current_time=local_time,
                utc_offset=offset_str,
                is_dst=is_dst,
                local_hour=local_hour,
                business_hours=business_hours,
                prime_time=prime_time
            )
            
        except Exception as e:
            logger.warning(f"Timezone calculation error: {e}")
            # Fallback
            now = datetime.now(timezone.utc)
            return TimeZoneInfo(
                timezone=tz,
                current_time=now,
                utc_offset="+00:00",
                is_dst=False,
                local_hour=now.hour,
                business_hours=9 <= now.hour <= 17,
                prime_time=18 <= now.hour <= 22
            )
    
    def get_optimal_posting_times(self, tz: TimeZone) -> List[int]:
        """Get optimal posting hours for a timezone."""
        timezone_optimal_hours = {
            TimeZone.EST: [9, 12, 15, 18, 20],  # US East Coast
            TimeZone.PST: [8, 11, 14, 17, 19],  # US West Coast
            TimeZone.CET: [10, 13, 16, 19, 21], # Central Europe
            TimeZone.JST: [9, 12, 15, 18, 20],  # Japan
            TimeZone.IST: [10, 13, 16, 19, 21], # India
            TimeZone.AEST: [9, 12, 15, 18, 20], # Australia
            TimeZone.GMT: [9, 12, 15, 18, 20],  # GMT/UTC
        }
        
        return timezone_optimal_hours.get(tz, [9, 12, 15, 18, 20])
    
    def calculate_global_launch_times(self, timezones: List[TimeZone], target_hour: int = 9) -> Dict[str, datetime]:
        """Calculate synchronized launch times across timezones."""
        launch_times = {}
        base_time = datetime.now(timezone.utc).replace(hour=target_hour, minute=0, second=0, microsecond=0)
        
        for tz in timezones:
            tz_info = self.get_timezone_info(tz)
            
            # Calculate local launch time
            if PYTZ_AVAILABLE:
                try:
                    if tz == TimeZone.GMT or tz == TimeZone.UTC:
                        tz_obj = pytz.UTC
                    else:
                        tz_obj = pytz.timezone(tz.value)
                    
                    local_launch = base_time.astimezone(tz_obj).replace(hour=target_hour)
                    launch_times[tz.value] = local_launch
                except:
                    launch_times[tz.value] = base_time
            else:
                launch_times[tz.value] = base_time
        
        return launch_times

class GMTContentAdapter:
    """Adapt content for different regions and timezones."""
    
    def __init__(self) -> Any:
        self.regional_preferences = {
            TimeZone.EST: {
                "greeting_style": "professional",
                "time_format": "12h",
                "currency": "USD",
                "cultural_context": "US_business"
            },
            TimeZone.PST: {
                "greeting_style": "casual",
                "time_format": "12h", 
                "currency": "USD",
                "cultural_context": "US_tech"
            },
            TimeZone.CET: {
                "greeting_style": "formal",
                "time_format": "24h",
                "currency": "EUR",
                "cultural_context": "EU_business"
            },
            TimeZone.JST: {
                "greeting_style": "respectful",
                "time_format": "24h",
                "currency": "JPY",
                "cultural_context": "JP_business"
            }
        }
    
    async def adapt_content_for_timezone(self, content: str, tz: TimeZone, tz_info: TimeZoneInfo) -> str:
        """Adapt content based on timezone and local context."""
        adapted_content = content
        
        # Get regional preferences
        prefs = self.regional_preferences.get(tz, {})
        
        # Time-based adaptations
        if tz_info.business_hours:
            adapted_content = self._add_business_context(adapted_content, prefs)
        elif tz_info.prime_time:
            adapted_content = self._add_prime_time_context(adapted_content, prefs)
        
        # Regional adaptations
        if prefs.get("greeting_style"):
            adapted_content = self._adapt_greeting_style(adapted_content, prefs["greeting_style"])
        
        # Currency adaptations
        if prefs.get("currency"):
            adapted_content = self._adapt_currency(adapted_content, prefs["currency"])
        
        return adapted_content
    
    def _add_business_context(self, content: str, prefs: Dict[str, str]) -> str:
        """Add business hours context."""
        if "professional" in prefs.get("greeting_style", ""):
            return f"Durante horario laboral: {content}"
        return content
    
    def _add_prime_time_context(self, content: str, prefs: Dict[str, str]) -> str:
        """Add prime time context."""
        return f"🌟 Horario prime: {content}"
    
    def _adapt_greeting_style(self, content: str, style: str) -> str:
        """Adapt greeting style based on cultural context."""
        style_adaptations = {
            "professional": content.replace("¡Hola!", "Buenos días"),
            "casual": content.replace("Buenos días", "¡Hey!"),
            "formal": content.replace("¡Hola!", "Estimado/a"),
            "respectful": content.replace("¡Hola!", "Saludos cordiales")
        }
        return style_adaptations.get(style, content)
    
    def _adapt_currency(self, content: str, currency: str) -> str:
        """Adapt currency references."""
        currency_symbols = {
            "USD": "$",
            "EUR": "€", 
            "JPY": "¥",
            "GBP": "£"
        }
        
        symbol = currency_symbols.get(currency, "$")
        # Simple currency adaptation
        content = content.replace("$", symbol)
        return content

class GMTScheduler:
    """Advanced scheduling system for global content operations."""
    
    def __init__(self) -> Any:
        self.scheduled_tasks = {}
        self.running_tasks = {}
        self.task_history = []
        self.max_concurrent_tasks = 50
    
    async def schedule_global_campaign(self, schedule: GlobalSchedule) -> str:
        """Schedule a global campaign across multiple timezones."""
        schedule_id = schedule.schedule_id
        
        # Store schedule
        self.scheduled_tasks[schedule_id] = schedule
        
        # Create individual timezone tasks
        tasks = []
        for tz_str, scheduled_time in schedule.scheduled_times.items():
            task_id = f"{schedule_id}_{tz_str}"
            
            # Calculate delay
            delay = (scheduled_time - datetime.now(timezone.utc)).total_seconds()
            
            if delay > 0:
                task = asyncio.create_task(
                    self._execute_scheduled_task(task_id, schedule, tz_str, delay)
                )
                tasks.append(task)
                self.running_tasks[task_id] = task
        
        logger.info(f"Scheduled global campaign {schedule_id} with {len(tasks)} timezone tasks")
        return schedule_id
    
    async def _execute_scheduled_task(self, task_id: str, schedule: GlobalSchedule, tz_str: str, delay: float):
        """Execute a scheduled task after delay."""
        try:
            # Wait for scheduled time
            if delay > 0:
                await asyncio.sleep(delay)
            
            # Execute task
            logger.info(f"Executing scheduled task {task_id} for timezone {tz_str}")
            
            # Here you would integrate with the actual copywriting service
            # For now, we'll simulate the execution
            await self._simulate_content_generation(schedule, tz_str)
            
            # Record completion
            self.task_history.append({
                "task_id": task_id,
                "schedule_id": schedule.schedule_id,
                "timezone": tz_str,
                "executed_at": datetime.now(timezone.utc),
                "status": "completed"
            })
            
        except Exception as e:
            logger.error(f"Task execution failed {task_id}: {e}")
            self.task_history.append({
                "task_id": task_id,
                "schedule_id": schedule.schedule_id,
                "timezone": tz_str,
                "executed_at": datetime.now(timezone.utc),
                "status": "failed",
                "error": str(e)
            })
        finally:
            # Clean up
            if task_id in self.running_tasks:
                del self.running_tasks[task_id]
    
    async def _simulate_content_generation(self, schedule: GlobalSchedule, tz_str: str):
        """Simulate content generation for a timezone."""
        # This would integrate with the actual copywriting service
        await asyncio.sleep(0.1)  # Simulate processing time
        logger.info(f"Generated content for {tz_str} in campaign {schedule.schedule_id}")
    
    def get_schedule_status(self, schedule_id: str) -> Dict[str, Any]:
        """Get status of a scheduled campaign."""
        if schedule_id not in self.scheduled_tasks:
            return {"error": "Schedule not found"}
        
        schedule = self.scheduled_tasks[schedule_id]
        
        # Count running tasks
        running_count = len([
            task_id for task_id in self.running_tasks.keys()
            if task_id.startswith(schedule_id)
        ])
        
        # Count completed tasks
        completed_tasks = [
            task for task in self.task_history
            if task["schedule_id"] == schedule_id
        ]
        
        return {
            "schedule_id": schedule_id,
            "campaign_type": schedule.campaign_type,
            "total_timezones": len(schedule.target_timezones),
            "running_tasks": running_count,
            "completed_tasks": len(completed_tasks),
            "created_at": schedule.created_at,
            "status": "running" if running_count > 0 else "completed"
        }

# === MAIN GMT AGENT ===
class GMTCopywritingAgent:
    """Main GMT-aware copywriting agent."""
    
    def __init__(self) -> Any:
        self.time_manager = GMTTimeManager()
        self.content_adapter = GMTContentAdapter()
        self.scheduler = GMTScheduler()
        self.request_queue = asyncio.Queue()
        self.processing_stats = {
            "requests_processed": 0,
            "timezones_served": set(),
            "average_response_time": 0.0
        }
        
        # Import the copywriting service
        self.copywriting_service = None
    
    async def initialize(self) -> Any:
        """Initialize the GMT agent."""
        try:
            # Import and initialize copywriting service
            self.copywriting_service = await get_service()
            logger.info("GMT Copywriting Agent initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize copywriting service: {e}")
            raise
    
    async async def process_gmt_request(self, request: GMTCopywritingRequest) -> GMTCopywritingResponse:
        """Process a GMT-aware copywriting request."""
        start_time = time.perf_counter()
        request_id = str(uuid.uuid4())
        
        try:
            # Get timezone information
            tz_info = self.time_manager.get_timezone_info(request.target_timezone)
            
            # Check if we should process now or schedule
            if request.schedule_time:
                # Schedule for later
                await self._schedule_request(request, request_id)
                # Return immediate response for scheduled request
                return GMTCopywritingResponse(
                    request_id=request_id,
                    output=CopywritingOutput(variants=[], model_used="gmt-agent-scheduled"),
                    timezone_info=tz_info,
                    execution_time=request.schedule_time,
                    regional_adaptations={"status": "scheduled"},
                    performance_metrics={"scheduled": True}
                )
            
            # Check business hours constraint
            if request.business_hours_only and not tz_info.business_hours:
                raise HTTPException(
                    status_code=400,
                    detail=f"Request requires business hours, but {request.target_timezone} is currently outside business hours"
                )
            
            # Process immediately
            if not self.copywriting_service:
                raise HTTPException(status_code=500, detail="Copywriting service not initialized")
            
            # Generate base content
            output = await self.copywriting_service.generate_copy(request.base_request)
            
            # Apply regional adaptations
            regional_adaptations = {}
            if request.regional_adaptation:
                for variant in output.variants:
                    adapted_headline = await self.content_adapter.adapt_content_for_timezone(
                        variant.headline, request.target_timezone, tz_info
                    )
                    adapted_text = await self.content_adapter.adapt_content_for_timezone(
                        variant.primary_text, request.target_timezone, tz_info
                    )
                    
                    variant.headline = adapted_headline
                    variant.primary_text = adapted_text
                    
                    regional_adaptations[variant.variant_id] = {
                        "timezone": request.target_timezone.value,
                        "business_hours": tz_info.business_hours,
                        "prime_time": tz_info.prime_time
                    }
            
            # Update stats
            processing_time = time.perf_counter() - start_time
            self.processing_stats["requests_processed"] += 1
            self.processing_stats["timezones_served"].add(request.target_timezone.value)
            
            # Calculate average response time
            current_avg = self.processing_stats["average_response_time"]
            new_avg = (current_avg * (self.processing_stats["requests_processed"] - 1) + processing_time) / self.processing_stats["requests_processed"]
            self.processing_stats["average_response_time"] = new_avg
            
            return GMTCopywritingResponse(
                request_id=request_id,
                output=output,
                timezone_info=tz_info,
                execution_time=datetime.now(timezone.utc),
                regional_adaptations=regional_adaptations,
                performance_metrics={
                    "processing_time_ms": processing_time * 1000,
                    "timezone": request.target_timezone.value,
                    "regional_adaptation": request.regional_adaptation,
                    "business_hours": tz_info.business_hours,
                    "prime_time": tz_info.prime_time
                }
            )
            
        except Exception as e:
            logger.error(f"GMT request processing failed: {e}")
            raise
    
    async def _schedule_request(self, request: GMTCopywritingRequest, request_id: str):
        """Schedule a request for later execution."""
        # Create a schedule for single timezone
        schedule = GlobalSchedule(
            schedule_id=request_id,
            campaign_type=CampaignType.TIME_SENSITIVE if request.time_sensitive else CampaignType.EVERGREEN,
            target_timezones=[request.target_timezone],
            scheduled_times={request.target_timezone.value: request.schedule_time},
            priority=request.priority,
            metadata={
                "original_request": request.model_dump(),
                "request_id": request_id
            }
        )
        
        await self.scheduler.schedule_global_campaign(schedule)
    
    async def create_global_campaign(self, 
                                   base_request: CopywritingInput,
                                   target_timezones: List[TimeZone],
                                   campaign_type: CampaignType = CampaignType.GLOBAL_LAUNCH,
                                   target_hour: int = 9) -> str:
        """Create a global campaign across multiple timezones."""
        
        # Calculate launch times for each timezone
        launch_times = self.time_manager.calculate_global_launch_times(target_timezones, target_hour)
        
        # Create schedule
        schedule = GlobalSchedule(
            campaign_type=campaign_type,
            target_timezones=target_timezones,
            scheduled_times=launch_times,
            priority=ContentPriority.HIGH,
            metadata={
                "base_request": base_request.model_dump(),
                "target_hour": target_hour
            }
        )
        
        # Schedule the campaign
        campaign_id = await self.scheduler.schedule_global_campaign(schedule)
        
        logger.info(f"Created global campaign {campaign_id} for {len(target_timezones)} timezones")
        return campaign_id
    
    def get_timezone_status(self, timezone: TimeZone) -> TimeZoneInfo:
        """Get current status of a timezone."""
        return self.time_manager.get_timezone_info(timezone)
    
    def get_all_timezones_status(self) -> Dict[str, TimeZoneInfo]:
        """Get status of all supported timezones."""
        return {
            tz.value: self.time_manager.get_timezone_info(tz)
            for tz in TimeZone
        }
    
    def get_agent_stats(self) -> Dict[str, Any]:
        """Get agent performance statistics."""
        return {
            "requests_processed": self.processing_stats["requests_processed"],
            "timezones_served": len(self.processing_stats["timezones_served"]),
            "average_response_time_ms": self.processing_stats["average_response_time"] * 1000,
            "supported_timezones": len(TimeZone),
            "running_campaigns": len(self.scheduler.running_tasks),
            "completed_tasks": len(self.scheduler.task_history)
        }

# === FASTAPI APPLICATION ===
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management."""
    # Startup
    logger.info("Starting GMT Copywriting Agent Platform")
    
    # Initialize agent
    agent = GMTCopywritingAgent()
    await agent.initialize()
    app.state.gmt_agent = agent
    
    yield
    
    # Shutdown
    logger.info("Shutting down GMT Copywriting Agent Platform")

def create_gmt_app() -> FastAPI:
    """Create GMT Agent FastAPI application."""
    
    app = FastAPI(
        title="GMT Copywriting Agent Platform",
        description="""
        **Global Time-Aware Copywriting Agent**
        
        🌍 **Global Operations**
        - Multi-timezone content generation
        - Regional content adaptation
        - Global campaign scheduling
        - Time-sensitive content optimization
        
        ⏰ **Time Management**
        - Business hours detection
        - Prime time optimization
        - Synchronized global launches
        - Timezone-aware scheduling
        
        🚀 **Features**
        - 17+ timezone support
        - Regional content adaptation
        - Global campaign coordination
        - Performance tracking
        - Intelligent scheduling
        """,
        version="1.0.0",
        lifespan=lifespan
    )
    
    # Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"]
    )
    
    # === ROUTES ===
    
    @app.get("/")
    async def root():
        """GMT Agent information."""
        return {
            "service": "GMT Copywriting Agent Platform",
            "version": "1.0.0",
            "status": "operational",
            "supported_timezones": len(TimeZone),
            "features": {
                "multi_timezone": True,
                "regional_adaptation": True,
                "global_campaigns": True,
                "intelligent_scheduling": True,
                "business_hours_detection": True,
                "prime_time_optimization": True
            },
            "endpoints": {
                "generate": "/gmt/generate",
                "global_campaign": "/gmt/campaign/global",
                "timezone_status": "/gmt/timezone/{timezone}",
                "all_timezones": "/gmt/timezones",
                "stats": "/gmt/stats"
            }
        }
    
    @app.post("/gmt/generate", response_model=GMTCopywritingResponse)
    async def generate_gmt_copy(request: GMTCopywritingRequest):
        """Generate timezone-aware copywriting content."""
        agent = app.state.gmt_agent
        return await agent.process_gmt_request(request)
    
    @app.post("/gmt/campaign/global")
    async def create_global_campaign(
        base_request: CopywritingInput,
        target_timezones: List[TimeZone],
        campaign_type: CampaignType = CampaignType.GLOBAL_LAUNCH,
        target_hour: int = 9
    ):
        """Create a global campaign across multiple timezones."""
        agent = app.state.gmt_agent
        campaign_id = await agent.create_global_campaign(
            base_request, target_timezones, campaign_type, target_hour
        )
        
        return {
            "campaign_id": campaign_id,
            "target_timezones": [tz.value for tz in target_timezones],
            "campaign_type": campaign_type,
            "target_hour": target_hour,
            "status": "scheduled"
        }
    
    @app.get("/gmt/timezone/{timezone}", response_model=TimeZoneInfo)
    async def get_timezone_status(timezone: TimeZone):
        """Get current status of a specific timezone."""
        agent = app.state.gmt_agent
        return agent.get_timezone_status(timezone)
    
    @app.get("/gmt/timezones")
    async def get_all_timezones():
        """Get status of all supported timezones."""
        agent = app.state.gmt_agent
        return agent.get_all_timezones_status()
    
    @app.get("/gmt/campaign/{campaign_id}")
    async def get_campaign_status(campaign_id: str):
        """Get status of a global campaign."""
        agent = app.state.gmt_agent
        return agent.scheduler.get_schedule_status(campaign_id)
    
    @app.get("/gmt/stats")
    async def get_agent_stats():
        """Get GMT agent performance statistics."""
        agent = app.state.gmt_agent
        return agent.get_agent_stats()
    
    @app.get("/gmt/optimal-times/{timezone}")
    async def get_optimal_times(timezone: TimeZone):
        """Get optimal posting times for a timezone."""
        agent = app.state.gmt_agent
        optimal_hours = agent.time_manager.get_optimal_posting_times(timezone)
        tz_info = agent.get_timezone_status(timezone)
        
        return {
            "timezone": timezone.value,
            "optimal_hours": optimal_hours,
            "current_hour": tz_info.local_hour,
            "is_optimal_now": tz_info.local_hour in optimal_hours,
            "business_hours": tz_info.business_hours,
            "prime_time": tz_info.prime_time
        }
    
    return app

# Create the GMT application
gmt_app = create_gmt_app()

# === MAIN ===
if __name__ == "__main__":
    
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting GMT Copywriting Agent Platform")
    
    uvicorn.run(
        "gmt_copywriting_agent:gmt_app",
        host="0.0.0.0",
        port=8001,  # Different port from main service
        reload=True,
        log_level="info"
    )

# Export
__all__ = [
    "GMTCopywritingAgent", "gmt_app", "create_gmt_app",
    "TimeZone", "CampaignType", "ContentPriority",
    "GMTCopywritingRequest", "GMTCopywritingResponse"
] 