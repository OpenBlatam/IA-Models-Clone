"""
Content Automation Engine - Advanced Content Automation and Orchestration
====================================================================

This module provides comprehensive content automation capabilities including:
- Intelligent content scheduling and publishing
- Automated content curation and aggregation
- Multi-channel content distribution
- Content lifecycle management
- Automated content optimization
- Smart content tagging and categorization
- Automated content translation
- Content performance monitoring and auto-adjustment
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import uuid
import schedule
import threading
import time
from collections import defaultdict, deque
import requests
import feedparser
from bs4 import BeautifulSoup
import openai
import anthropic
from googletrans import Translator
import pytz
from croniter import croniter
import redis
import sqlalchemy as sa
from sqlalchemy import create_engine, text
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import boto3
from google.cloud import storage
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import tweepy
import facebook
import linkedin
import instagram

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AutomationType(Enum):
    """Automation type enumeration"""
    SCHEDULING = "scheduling"
    CURATION = "curation"
    DISTRIBUTION = "distribution"
    OPTIMIZATION = "optimization"
    TRANSLATION = "translation"
    TAGGING = "tagging"
    MONITORING = "monitoring"
    LIFECYCLE = "lifecycle"

class ContentSource(Enum):
    """Content source enumeration"""
    RSS_FEED = "rss_feed"
    WEB_SCRAPING = "web_scraping"
    API = "api"
    USER_GENERATED = "user_generated"
    AI_GENERATED = "ai_generated"
    MANUAL = "manual"

class DistributionChannel(Enum):
    """Distribution channel enumeration"""
    WEBSITE = "website"
    BLOG = "blog"
    SOCIAL_MEDIA = "social_media"
    EMAIL = "email"
    NEWSLETTER = "newsletter"
    PODCAST = "podcast"
    VIDEO = "video"
    MOBILE_APP = "mobile_app"

class AutomationStatus(Enum):
    """Automation status enumeration"""
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    SCHEDULED = "scheduled"

@dataclass
class AutomationRule:
    """Automation rule data structure"""
    rule_id: str
    name: str
    description: str
    automation_type: AutomationType
    trigger_conditions: Dict[str, Any] = field(default_factory=dict)
    actions: List[Dict[str, Any]] = field(default_factory=list)
    schedule: str = ""  # Cron expression
    is_active: bool = True
    priority: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_executed: Optional[datetime] = None
    execution_count: int = 0
    success_count: int = 0
    failure_count: int = 0

@dataclass
class ContentSource:
    """Content source data structure"""
    source_id: str
    name: str
    source_type: ContentSource
    url: str = ""
    api_key: str = ""
    credentials: Dict[str, Any] = field(default_factory=dict)
    last_fetched: Optional[datetime] = None
    fetch_interval: int = 3600  # seconds
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ScheduledContent:
    """Scheduled content data structure"""
    schedule_id: str
    content_id: str
    title: str
    content: str
    scheduled_time: datetime
    distribution_channels: List[DistributionChannel] = field(default_factory=list)
    target_audience: str = ""
    status: AutomationStatus = AutomationStatus.SCHEDULED
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AutomationJob:
    """Automation job data structure"""
    job_id: str
    rule_id: str
    job_type: AutomationType
    status: AutomationStatus
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    result: Dict[str, Any] = field(default_factory=dict)
    error_message: str = ""
    execution_time: float = 0.0

class ContentAutomationEngine:
    """
    Advanced Content Automation Engine
    
    Provides comprehensive content automation and orchestration capabilities
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the Content Automation Engine"""
        self.config = config
        self.automation_rules = {}
        self.content_sources = {}
        self.scheduled_content = {}
        self.automation_jobs = {}
        self.redis_client = None
        self.database_engine = None
        self.scheduler_thread = None
        self.is_running = False
        
        # Initialize components
        self._initialize_database()
        self._initialize_redis()
        self._initialize_ai_models()
        self._initialize_social_media_clients()
        self._initialize_translation_service()
        
        # Start automation scheduler
        self._start_scheduler()
        
        logger.info("Content Automation Engine initialized successfully")
    
    def _initialize_database(self):
        """Initialize database connection"""
        try:
            if self.config.get("database_url"):
                self.database_engine = create_engine(self.config["database_url"])
                logger.info("Database connection established")
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
    
    def _initialize_redis(self):
        """Initialize Redis connection"""
        try:
            if self.config.get("redis_url"):
                self.redis_client = redis.Redis.from_url(self.config["redis_url"])
                logger.info("Redis connection established")
        except Exception as e:
            logger.error(f"Error initializing Redis: {e}")
    
    def _initialize_ai_models(self):
        """Initialize AI models for automation"""
        try:
            # OpenAI client
            if self.config.get("openai_api_key"):
                openai.api_key = self.config["openai_api_key"]
            
            # Anthropic client
            if self.config.get("anthropic_api_key"):
                self.anthropic_client = anthropic.Anthropic(api_key=self.config["anthropic_api_key"])
            
            # Translation service
            self.translator = Translator()
            
            # Content analysis models
            self.content_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            self.content_clusterer = KMeans(n_clusters=10, random_state=42)
            
            logger.info("AI models initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing AI models: {e}")
    
    def _initialize_social_media_clients(self):
        """Initialize social media API clients"""
        try:
            # Twitter API
            if self.config.get("twitter_api_key"):
                self.twitter_client = tweepy.Client(
                    bearer_token=self.config.get("twitter_bearer_token"),
                    consumer_key=self.config.get("twitter_api_key"),
                    consumer_secret=self.config.get("twitter_api_secret"),
                    access_token=self.config.get("twitter_access_token"),
                    access_token_secret=self.config.get("twitter_access_secret")
                )
            
            # Facebook API
            if self.config.get("facebook_access_token"):
                self.facebook_client = facebook.GraphAPI(access_token=self.config["facebook_access_token"])
            
            # LinkedIn API
            if self.config.get("linkedin_access_token"):
                self.linkedin_client = linkedin.LinkedIn(
                    access_token=self.config["linkedin_access_token"]
                )
            
            # Instagram API
            if self.config.get("instagram_access_token"):
                self.instagram_client = instagram.InstagramAPI(
                    access_token=self.config["instagram_access_token"]
                )
            
            logger.info("Social media clients initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing social media clients: {e}")
    
    def _initialize_translation_service(self):
        """Initialize translation service"""
        try:
            self.translator = Translator()
            logger.info("Translation service initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing translation service: {e}")
    
    def _start_scheduler(self):
        """Start the automation scheduler"""
        try:
            self.is_running = True
            self.scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
            self.scheduler_thread.start()
            logger.info("Automation scheduler started")
            
        except Exception as e:
            logger.error(f"Error starting scheduler: {e}")
    
    def _run_scheduler(self):
        """Run the automation scheduler"""
        while self.is_running:
            try:
                # Check for scheduled content
                self._check_scheduled_content()
                
                # Check for automation rules
                self._check_automation_rules()
                
                # Check for content sources
                self._check_content_sources()
                
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in scheduler: {e}")
                time.sleep(60)
    
    async def create_automation_rule(self, rule_data: Dict[str, Any]) -> AutomationRule:
        """Create a new automation rule"""
        try:
            rule_id = str(uuid.uuid4())
            
            rule = AutomationRule(
                rule_id=rule_id,
                name=rule_data["name"],
                description=rule_data["description"],
                automation_type=AutomationType(rule_data["automation_type"]),
                trigger_conditions=rule_data.get("trigger_conditions", {}),
                actions=rule_data.get("actions", []),
                schedule=rule_data.get("schedule", ""),
                priority=rule_data.get("priority", 0)
            )
            
            # Store rule
            self.automation_rules[rule_id] = rule
            
            # Schedule rule if it has a schedule
            if rule.schedule:
                self._schedule_rule(rule)
            
            logger.info(f"Automation rule {rule_id} created successfully")
            
            return rule
            
        except Exception as e:
            logger.error(f"Error creating automation rule: {e}")
            raise
    
    def _schedule_rule(self, rule: AutomationRule):
        """Schedule automation rule"""
        try:
            if rule.schedule:
                # Parse cron expression and schedule
                cron = croniter(rule.schedule, datetime.utcnow())
                next_run = cron.get_next(datetime)
                
                # Schedule the rule execution
                schedule.every().day.at(next_run.strftime("%H:%M")).do(
                    self._execute_automation_rule, rule.rule_id
                )
                
        except Exception as e:
            logger.error(f"Error scheduling rule: {e}")
    
    async def _execute_automation_rule(self, rule_id: str):
        """Execute automation rule"""
        try:
            if rule_id not in self.automation_rules:
                return
            
            rule = self.automation_rules[rule_id]
            
            # Create job
            job = AutomationJob(
                job_id=str(uuid.uuid4()),
                rule_id=rule_id,
                job_type=rule.automation_type,
                status=AutomationStatus.ACTIVE
            )
            
            self.automation_jobs[job.job_id] = job
            
            # Execute actions
            for action in rule.actions:
                await self._execute_action(action, job)
            
            # Update rule statistics
            rule.last_executed = datetime.utcnow()
            rule.execution_count += 1
            rule.success_count += 1
            
            job.status = AutomationStatus.COMPLETED
            job.completed_at = datetime.utcnow()
            job.execution_time = (job.completed_at - job.started_at).total_seconds()
            
            logger.info(f"Automation rule {rule_id} executed successfully")
            
        except Exception as e:
            logger.error(f"Error executing automation rule: {e}")
            
            # Update failure statistics
            if rule_id in self.automation_rules:
                rule = self.automation_rules[rule_id]
                rule.failure_count += 1
            
            if job:
                job.status = AutomationStatus.FAILED
                job.error_message = str(e)
                job.completed_at = datetime.utcnow()
    
    async def _execute_action(self, action: Dict[str, Any], job: AutomationJob):
        """Execute automation action"""
        try:
            action_type = action.get("type")
            
            if action_type == "schedule_content":
                await self._action_schedule_content(action, job)
            elif action_type == "curate_content":
                await self._action_curate_content(action, job)
            elif action_type == "distribute_content":
                await self._action_distribute_content(action, job)
            elif action_type == "optimize_content":
                await self._action_optimize_content(action, job)
            elif action_type == "translate_content":
                await self._action_translate_content(action, job)
            elif action_type == "tag_content":
                await self._action_tag_content(action, job)
            elif action_type == "monitor_content":
                await self._action_monitor_content(action, job)
            
        except Exception as e:
            logger.error(f"Error executing action: {e}")
            raise
    
    async def _action_schedule_content(self, action: Dict[str, Any], job: AutomationJob):
        """Schedule content action"""
        try:
            scheduled_time = datetime.fromisoformat(action["scheduled_time"])
            
            scheduled_content = ScheduledContent(
                schedule_id=str(uuid.uuid4()),
                content_id=action["content_id"],
                title=action["title"],
                content=action["content"],
                scheduled_time=scheduled_time,
                distribution_channels=[DistributionChannel(ch) for ch in action.get("channels", [])],
                target_audience=action.get("target_audience", "")
            )
            
            self.scheduled_content[scheduled_content.schedule_id] = scheduled_content
            
            job.result["scheduled_content_id"] = scheduled_content.schedule_id
            
        except Exception as e:
            logger.error(f"Error scheduling content: {e}")
            raise
    
    async def _action_curate_content(self, action: Dict[str, Any], job: AutomationJob):
        """Curate content action"""
        try:
            source_id = action["source_id"]
            
            if source_id in self.content_sources:
                source = self.content_sources[source_id]
                curated_content = await self._fetch_content_from_source(source)
                
                # Process and filter content
                filtered_content = await self._filter_content(curated_content, action.get("filters", {}))
                
                job.result["curated_content"] = filtered_content
                
        except Exception as e:
            logger.error(f"Error curating content: {e}")
            raise
    
    async def _action_distribute_content(self, action: Dict[str, Any], job: AutomationJob):
        """Distribute content action"""
        try:
            content_id = action["content_id"]
            channels = action.get("channels", [])
            
            distribution_results = {}
            
            for channel in channels:
                if channel == "twitter":
                    result = await self._distribute_to_twitter(content_id, action.get("twitter_config", {}))
                elif channel == "facebook":
                    result = await self._distribute_to_facebook(content_id, action.get("facebook_config", {}))
                elif channel == "linkedin":
                    result = await self._distribute_to_linkedin(content_id, action.get("linkedin_config", {}))
                elif channel == "instagram":
                    result = await self._distribute_to_instagram(content_id, action.get("instagram_config", {}))
                elif channel == "email":
                    result = await self._distribute_to_email(content_id, action.get("email_config", {}))
                else:
                    result = {"status": "unsupported_channel"}
                
                distribution_results[channel] = result
            
            job.result["distribution_results"] = distribution_results
            
        except Exception as e:
            logger.error(f"Error distributing content: {e}")
            raise
    
    async def _action_optimize_content(self, action: Dict[str, Any], job: AutomationJob):
        """Optimize content action"""
        try:
            content_id = action["content_id"]
            optimization_goals = action.get("goals", ["seo", "engagement"])
            
            # Get content
            content = action.get("content", "")
            
            optimized_content = content
            
            for goal in optimization_goals:
                if goal == "seo":
                    optimized_content = await self._optimize_for_seo(optimized_content)
                elif goal == "engagement":
                    optimized_content = await self._optimize_for_engagement(optimized_content)
                elif goal == "readability":
                    optimized_content = await self._optimize_for_readability(optimized_content)
            
            job.result["optimized_content"] = optimized_content
            
        except Exception as e:
            logger.error(f"Error optimizing content: {e}")
            raise
    
    async def _action_translate_content(self, action: Dict[str, Any], job: AutomationJob):
        """Translate content action"""
        try:
            content = action["content"]
            target_language = action["target_language"]
            
            # Translate content
            translated = self.translator.translate(content, dest=target_language)
            
            job.result["translated_content"] = translated.text
            job.result["source_language"] = translated.src
            job.result["target_language"] = target_language
            
        except Exception as e:
            logger.error(f"Error translating content: {e}")
            raise
    
    async def _action_tag_content(self, action: Dict[str, Any], job: AutomationJob):
        """Tag content action"""
        try:
            content = action["content"]
            
            # Generate tags using AI
            tags = await self._generate_content_tags(content)
            
            job.result["generated_tags"] = tags
            
        except Exception as e:
            logger.error(f"Error tagging content: {e}")
            raise
    
    async def _action_monitor_content(self, action: Dict[str, Any], job: AutomationJob):
        """Monitor content action"""
        try:
            content_id = action["content_id"]
            metrics = action.get("metrics", ["views", "engagement", "shares"])
            
            # Get content metrics
            content_metrics = await self._get_content_metrics(content_id, metrics)
            
            # Check thresholds
            alerts = []
            for metric, value in content_metrics.items():
                threshold = action.get("thresholds", {}).get(metric)
                if threshold and value < threshold:
                    alerts.append(f"{metric} below threshold: {value} < {threshold}")
            
            job.result["content_metrics"] = content_metrics
            job.result["alerts"] = alerts
            
        except Exception as e:
            logger.error(f"Error monitoring content: {e}")
            raise
    
    async def add_content_source(self, source_data: Dict[str, Any]) -> ContentSource:
        """Add content source for curation"""
        try:
            source_id = str(uuid.uuid4())
            
            source = ContentSource(
                source_id=source_id,
                name=source_data["name"],
                source_type=ContentSource(source_data["source_type"]),
                url=source_data.get("url", ""),
                api_key=source_data.get("api_key", ""),
                credentials=source_data.get("credentials", {}),
                fetch_interval=source_data.get("fetch_interval", 3600),
                metadata=source_data.get("metadata", {})
            )
            
            self.content_sources[source_id] = source
            
            logger.info(f"Content source {source_id} added successfully")
            
            return source
            
        except Exception as e:
            logger.error(f"Error adding content source: {e}")
            raise
    
    async def _fetch_content_from_source(self, source: ContentSource) -> List[Dict[str, Any]]:
        """Fetch content from source"""
        try:
            content_items = []
            
            if source.source_type == ContentSource.RSS_FEED:
                content_items = await self._fetch_rss_content(source.url)
            elif source.source_type == ContentSource.WEB_SCRAPING:
                content_items = await self._scrape_web_content(source.url)
            elif source.source_type == ContentSource.API:
                content_items = await self._fetch_api_content(source.url, source.api_key)
            
            # Update last fetched time
            source.last_fetched = datetime.utcnow()
            
            return content_items
            
        except Exception as e:
            logger.error(f"Error fetching content from source: {e}")
            return []
    
    async def _fetch_rss_content(self, url: str) -> List[Dict[str, Any]]:
        """Fetch content from RSS feed"""
        try:
            feed = feedparser.parse(url)
            content_items = []
            
            for entry in feed.entries[:10]:  # Limit to 10 items
                content_item = {
                    "title": entry.get("title", ""),
                    "content": entry.get("summary", ""),
                    "url": entry.get("link", ""),
                    "published": entry.get("published", ""),
                    "author": entry.get("author", ""),
                    "tags": [tag.term for tag in entry.get("tags", [])]
                }
                content_items.append(content_item)
            
            return content_items
            
        except Exception as e:
            logger.error(f"Error fetching RSS content: {e}")
            return []
    
    async def _scrape_web_content(self, url: str) -> List[Dict[str, Any]]:
        """Scrape content from web page"""
        try:
            response = requests.get(url, timeout=30)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract content based on common patterns
            title = soup.find('title')
            content = soup.find('article') or soup.find('div', class_='content')
            
            content_item = {
                "title": title.text if title else "",
                "content": content.text if content else "",
                "url": url,
                "scraped_at": datetime.utcnow().isoformat()
            }
            
            return [content_item]
            
        except Exception as e:
            logger.error(f"Error scraping web content: {e}")
            return []
    
    async def _fetch_api_content(self, url: str, api_key: str) -> List[Dict[str, Any]]:
        """Fetch content from API"""
        try:
            headers = {"Authorization": f"Bearer {api_key}"}
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Process API response (format depends on API)
            content_items = []
            if isinstance(data, list):
                for item in data:
                    content_item = {
                        "title": item.get("title", ""),
                        "content": item.get("content", ""),
                        "url": item.get("url", ""),
                        "published": item.get("published", ""),
                        "api_data": item
                    }
                    content_items.append(content_item)
            
            return content_items
            
        except Exception as e:
            logger.error(f"Error fetching API content: {e}")
            return []
    
    async def _filter_content(self, content_items: List[Dict[str, Any]], 
                            filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Filter content based on criteria"""
        try:
            filtered_items = []
            
            for item in content_items:
                # Apply filters
                if "min_length" in filters and len(item.get("content", "")) < filters["min_length"]:
                    continue
                
                if "max_length" in filters and len(item.get("content", "")) > filters["max_length"]:
                    continue
                
                if "required_keywords" in filters:
                    content_text = item.get("content", "").lower()
                    if not any(keyword.lower() in content_text for keyword in filters["required_keywords"]):
                        continue
                
                if "excluded_keywords" in filters:
                    content_text = item.get("content", "").lower()
                    if any(keyword.lower() in content_text for keyword in filters["excluded_keywords"]):
                        continue
                
                filtered_items.append(item)
            
            return filtered_items
            
        except Exception as e:
            logger.error(f"Error filtering content: {e}")
            return content_items
    
    async def _distribute_to_twitter(self, content_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Distribute content to Twitter"""
        try:
            if not hasattr(self, 'twitter_client'):
                return {"status": "error", "message": "Twitter client not configured"}
            
            # Get content
            content = config.get("content", "")
            
            # Truncate if too long
            if len(content) > 280:
                content = content[:277] + "..."
            
            # Post to Twitter
            response = self.twitter_client.create_tweet(text=content)
            
            return {
                "status": "success",
                "tweet_id": response.data["id"],
                "url": f"https://twitter.com/user/status/{response.data['id']}"
            }
            
        except Exception as e:
            logger.error(f"Error distributing to Twitter: {e}")
            return {"status": "error", "message": str(e)}
    
    async def _distribute_to_facebook(self, content_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Distribute content to Facebook"""
        try:
            if not hasattr(self, 'facebook_client'):
                return {"status": "error", "message": "Facebook client not configured"}
            
            # Get content
            content = config.get("content", "")
            page_id = config.get("page_id", "")
            
            # Post to Facebook
            response = self.facebook_client.put_object(
                parent_object=page_id,
                connection_name='feed',
                message=content
            )
            
            return {
                "status": "success",
                "post_id": response["id"]
            }
            
        except Exception as e:
            logger.error(f"Error distributing to Facebook: {e}")
            return {"status": "error", "message": str(e)}
    
    async def _distribute_to_linkedin(self, content_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Distribute content to LinkedIn"""
        try:
            if not hasattr(self, 'linkedin_client'):
                return {"status": "error", "message": "LinkedIn client not configured"}
            
            # Get content
            content = config.get("content", "")
            
            # Post to LinkedIn
            response = self.linkedin_client.submit_group_post(
                group_id=config.get("group_id", ""),
                title=config.get("title", ""),
                summary=content
            )
            
            return {
                "status": "success",
                "post_id": response["id"]
            }
            
        except Exception as e:
            logger.error(f"Error distributing to LinkedIn: {e}")
            return {"status": "error", "message": str(e)}
    
    async def _distribute_to_instagram(self, content_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Distribute content to Instagram"""
        try:
            if not hasattr(self, 'instagram_client'):
                return {"status": "error", "message": "Instagram client not configured"}
            
            # Get content
            content = config.get("content", "")
            image_url = config.get("image_url", "")
            
            # Post to Instagram
            response = self.instagram_client.post_photo(
                image_url=image_url,
                caption=content
            )
            
            return {
                "status": "success",
                "post_id": response["id"]
            }
            
        except Exception as e:
            logger.error(f"Error distributing to Instagram: {e}")
            return {"status": "error", "message": str(e)}
    
    async def _distribute_to_email(self, content_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Distribute content via email"""
        try:
            # Get email configuration
            smtp_server = self.config.get("smtp_server", "smtp.gmail.com")
            smtp_port = self.config.get("smtp_port", 587)
            username = self.config.get("email_username")
            password = self.config.get("email_password")
            
            if not username or not password:
                return {"status": "error", "message": "Email credentials not configured"}
            
            # Create email
            msg = MIMEMultipart()
            msg['From'] = username
            msg['To'] = config.get("recipients", "")
            msg['Subject'] = config.get("subject", "Content Update")
            
            body = config.get("content", "")
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.login(username, password)
            server.send_message(msg)
            server.quit()
            
            return {
                "status": "success",
                "recipients": config.get("recipients", "")
            }
            
        except Exception as e:
            logger.error(f"Error distributing via email: {e}")
            return {"status": "error", "message": str(e)}
    
    async def _optimize_for_seo(self, content: str) -> str:
        """Optimize content for SEO"""
        try:
            # Simple SEO optimization
            # In production, this would use more sophisticated techniques
            
            # Add meta keywords if not present
            if "keywords" not in content.lower():
                content += "\n\n<!-- SEO optimized -->"
            
            return content
            
        except Exception as e:
            logger.error(f"Error optimizing for SEO: {e}")
            return content
    
    async def _optimize_for_engagement(self, content: str) -> str:
        """Optimize content for engagement"""
        try:
            # Add engagement elements
            if "?" not in content:
                content += "\n\nWhat do you think about this?"
            
            return content
            
        except Exception as e:
            logger.error(f"Error optimizing for engagement: {e}")
            return content
    
    async def _optimize_for_readability(self, content: str) -> str:
        """Optimize content for readability"""
        try:
            # Simple readability optimization
            content = content.replace("  ", " ").replace("\n\n\n", "\n\n")
            
            return content
            
        except Exception as e:
            logger.error(f"Error optimizing for readability: {e}")
            return content
    
    async def _generate_content_tags(self, content: str) -> List[str]:
        """Generate content tags using AI"""
        try:
            # Use AI to generate tags
            if hasattr(self, 'anthropic_client'):
                response = self.anthropic_client.messages.create(
                    model="claude-3-sonnet-20240229",
                    max_tokens=100,
                    messages=[{
                        "role": "user",
                        "content": f"Generate 5 relevant tags for this content: {content[:500]}"
                    }]
                )
                
                tags_text = response.content[0].text
                tags = [tag.strip() for tag in tags_text.split(",")]
                return tags[:5]
            
            # Fallback to simple keyword extraction
            words = content.lower().split()
            word_counts = Counter(words)
            common_words = word_counts.most_common(10)
            return [word for word, count in common_words if len(word) > 3][:5]
            
        except Exception as e:
            logger.error(f"Error generating content tags: {e}")
            return []
    
    async def _get_content_metrics(self, content_id: str, metrics: List[str]) -> Dict[str, float]:
        """Get content performance metrics"""
        try:
            # Mock metrics - in production, this would fetch real data
            mock_metrics = {
                "views": np.random.randint(100, 10000),
                "engagement": np.random.random(),
                "shares": np.random.randint(10, 1000),
                "likes": np.random.randint(50, 5000),
                "comments": np.random.randint(5, 500)
            }
            
            return {metric: mock_metrics.get(metric, 0.0) for metric in metrics}
            
        except Exception as e:
            logger.error(f"Error getting content metrics: {e}")
            return {}
    
    def _check_scheduled_content(self):
        """Check for scheduled content to publish"""
        try:
            current_time = datetime.utcnow()
            
            for schedule_id, scheduled_content in self.scheduled_content.items():
                if (scheduled_content.status == AutomationStatus.SCHEDULED and 
                    scheduled_content.scheduled_time <= current_time):
                    
                    # Publish content
                    asyncio.create_task(self._publish_scheduled_content(scheduled_content))
                    
        except Exception as e:
            logger.error(f"Error checking scheduled content: {e}")
    
    async def _publish_scheduled_content(self, scheduled_content: ScheduledContent):
        """Publish scheduled content"""
        try:
            # Distribute to channels
            for channel in scheduled_content.distribution_channels:
                if channel == DistributionChannel.SOCIAL_MEDIA:
                    await self._distribute_to_social_media(scheduled_content)
                elif channel == DistributionChannel.EMAIL:
                    await self._distribute_to_email(scheduled_content.content_id, {
                        "content": scheduled_content.content,
                        "subject": scheduled_content.title
                    })
                elif channel == DistributionChannel.WEBSITE:
                    await self._publish_to_website(scheduled_content)
            
            # Update status
            scheduled_content.status = AutomationStatus.COMPLETED
            
            logger.info(f"Scheduled content {scheduled_content.schedule_id} published successfully")
            
        except Exception as e:
            logger.error(f"Error publishing scheduled content: {e}")
            scheduled_content.status = AutomationStatus.FAILED
    
    async def _distribute_to_social_media(self, scheduled_content: ScheduledContent):
        """Distribute scheduled content to social media"""
        try:
            # Twitter
            await self._distribute_to_twitter(scheduled_content.content_id, {
                "content": scheduled_content.content
            })
            
            # Facebook
            await self._distribute_to_facebook(scheduled_content.content_id, {
                "content": scheduled_content.content
            })
            
        except Exception as e:
            logger.error(f"Error distributing to social media: {e}")
    
    async def _publish_to_website(self, scheduled_content: ScheduledContent):
        """Publish scheduled content to website"""
        try:
            # In production, this would integrate with CMS
            logger.info(f"Publishing to website: {scheduled_content.title}")
            
        except Exception as e:
            logger.error(f"Error publishing to website: {e}")
    
    def _check_automation_rules(self):
        """Check for automation rules to execute"""
        try:
            current_time = datetime.utcnow()
            
            for rule in self.automation_rules.values():
                if rule.is_active and rule.schedule:
                    # Check if rule should execute
                    cron = croniter(rule.schedule, rule.last_executed or datetime.utcnow())
                    next_run = cron.get_next(datetime)
                    
                    if next_run <= current_time:
                        asyncio.create_task(self._execute_automation_rule(rule.rule_id))
                        
        except Exception as e:
            logger.error(f"Error checking automation rules: {e}")
    
    def _check_content_sources(self):
        """Check for content sources to fetch"""
        try:
            current_time = datetime.utcnow()
            
            for source in self.content_sources.values():
                if source.is_active:
                    last_fetch = source.last_fetched or datetime.min.replace(tzinfo=pytz.UTC)
                    time_since_fetch = (current_time - last_fetch).total_seconds()
                    
                    if time_since_fetch >= source.fetch_interval:
                        asyncio.create_task(self._fetch_content_from_source(source))
                        
        except Exception as e:
            logger.error(f"Error checking content sources: {e}")
    
    async def get_automation_analytics(self, time_period: str = "30d") -> Dict[str, Any]:
        """Get automation analytics and insights"""
        try:
            # Calculate time range
            end_date = datetime.utcnow()
            if time_period == "7d":
                start_date = end_date - timedelta(days=7)
            elif time_period == "30d":
                start_date = end_date - timedelta(days=30)
            elif time_period == "90d":
                start_date = end_date - timedelta(days=90)
            else:
                start_date = end_date - timedelta(days=30)
            
            # Get analytics data
            analytics = {
                "time_period": time_period,
                "total_rules": len(self.automation_rules),
                "active_rules": len([r for r in self.automation_rules.values() if r.is_active]),
                "total_jobs": len(self.automation_jobs),
                "successful_jobs": len([j for j in self.automation_jobs.values() if j.status == AutomationStatus.COMPLETED]),
                "failed_jobs": len([j for j in self.automation_jobs.values() if j.status == AutomationStatus.FAILED]),
                "total_sources": len(self.content_sources),
                "active_sources": len([s for s in self.content_sources.values() if s.is_active]),
                "scheduled_content": len(self.scheduled_content),
                "rule_performance": {},
                "source_performance": {}
            }
            
            # Rule performance
            for rule in self.automation_rules.values():
                if rule.execution_count > 0:
                    success_rate = rule.success_count / rule.execution_count
                    analytics["rule_performance"][rule.rule_id] = {
                        "name": rule.name,
                        "execution_count": rule.execution_count,
                        "success_rate": success_rate,
                        "last_executed": rule.last_executed.isoformat() if rule.last_executed else None
                    }
            
            # Source performance
            for source in self.content_sources.values():
                analytics["source_performance"][source.source_id] = {
                    "name": source.name,
                    "source_type": source.source_type.value,
                    "last_fetched": source.last_fetched.isoformat() if source.last_fetched else None,
                    "is_active": source.is_active
                }
            
            return analytics
            
        except Exception as e:
            logger.error(f"Error getting automation analytics: {e}")
            return {"error": str(e)}
    
    def stop_automation(self):
        """Stop automation engine"""
        try:
            self.is_running = False
            if self.scheduler_thread:
                self.scheduler_thread.join(timeout=5)
            
            logger.info("Automation engine stopped")
            
        except Exception as e:
            logger.error(f"Error stopping automation engine: {e}")

# Example usage and testing
async def main():
    """Example usage of the Content Automation Engine"""
    try:
        # Initialize engine
        config = {
            "database_url": "postgresql://user:password@localhost/automationdb",
            "redis_url": "redis://localhost:6379",
            "openai_api_key": "your-openai-api-key",
            "anthropic_api_key": "your-anthropic-api-key",
            "twitter_api_key": "your-twitter-api-key",
            "facebook_access_token": "your-facebook-access-token",
            "smtp_server": "smtp.gmail.com",
            "smtp_port": 587,
            "email_username": "your-email@gmail.com",
            "email_password": "your-password"
        }
        
        engine = ContentAutomationEngine(config)
        
        # Create automation rule
        print("Creating automation rule...")
        rule = await engine.create_automation_rule({
            "name": "Daily Content Curation",
            "description": "Automatically curate and publish content daily",
            "automation_type": "curation",
            "schedule": "0 9 * * *",  # Daily at 9 AM
            "actions": [
                {
                    "type": "curate_content",
                    "source_id": "tech_news_source",
                    "filters": {
                        "min_length": 100,
                        "required_keywords": ["AI", "technology"]
                    }
                },
                {
                    "type": "schedule_content",
                    "content_id": "curated_content_001",
                    "title": "Daily Tech News",
                    "content": "Curated tech news for today",
                    "scheduled_time": "2024-01-01T10:00:00",
                    "channels": ["twitter", "facebook", "email"]
                }
            ]
        })
        
        # Add content source
        print("Adding content source...")
        source = await engine.add_content_source({
            "name": "Tech News RSS",
            "source_type": "rss_feed",
            "url": "https://feeds.feedburner.com/oreilly/radar",
            "fetch_interval": 3600
        })
        
        # Schedule content
        print("Scheduling content...")
        scheduled_content = ScheduledContent(
            schedule_id=str(uuid.uuid4()),
            content_id="test_content_001",
            title="Test Scheduled Content",
            content="This is a test of scheduled content publishing.",
            scheduled_time=datetime.utcnow() + timedelta(minutes=1),
            distribution_channels=[DistributionChannel.SOCIAL_MEDIA, DistributionChannel.EMAIL]
        )
        engine.scheduled_content[scheduled_content.schedule_id] = scheduled_content
        
        # Wait for automation to run
        print("Waiting for automation to execute...")
        await asyncio.sleep(70)  # Wait for scheduled content
        
        # Get automation analytics
        print("Getting automation analytics...")
        analytics = await engine.get_automation_analytics("7d")
        print(f"Total rules: {analytics['total_rules']}")
        print(f"Active rules: {analytics['active_rules']}")
        print(f"Total jobs: {analytics['total_jobs']}")
        print(f"Successful jobs: {analytics['successful_jobs']}")
        
        # Stop automation
        print("Stopping automation engine...")
        engine.stop_automation()
        
        print("\nContent Automation Engine demo completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main demo: {e}")

if __name__ == "__main__":
    asyncio.run(main())
























