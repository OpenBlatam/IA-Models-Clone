"""
Content Analytics Engine - Advanced content analytics and business intelligence
Following FastAPI best practices: functional programming, RORO pattern, async operations
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple, AsyncGenerator, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter, defaultdict
import statistics

import aiofiles
import aiohttp
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

logger = logging.getLogger(__name__)


@dataclass
class ContentAnalytics:
    """Content analytics result"""
    content_id: str
    analytics_timestamp: datetime
    content_metrics: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    engagement_metrics: Dict[str, Any]
    quality_metrics: Dict[str, Any]
    seo_metrics: Dict[str, Any]
    audience_metrics: Dict[str, Any]
    competitive_metrics: Dict[str, Any]
    trend_metrics: Dict[str, Any]
    insights: List[str]
    recommendations: List[str]


@dataclass
class BusinessIntelligence:
    """Business intelligence analysis result"""
    analysis_id: str
    analysis_timestamp: datetime
    content_performance: Dict[str, Any]
    audience_insights: Dict[str, Any]
    market_trends: Dict[str, Any]
    competitive_analysis: Dict[str, Any]
    roi_metrics: Dict[str, Any]
    growth_metrics: Dict[str, Any]
    risk_assessment: Dict[str, Any]
    strategic_recommendations: List[str]


@dataclass
class ContentDashboard:
    """Content dashboard data"""
    dashboard_id: str
    dashboard_timestamp: datetime
    overview_metrics: Dict[str, Any]
    content_metrics: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    audience_metrics: Dict[str, Any]
    trend_metrics: Dict[str, Any]
    alerts: List[Dict[str, Any]]
    charts_data: Dict[str, Any]
    kpis: Dict[str, Any]


@dataclass
class ContentReport:
    """Content analytics report"""
    report_id: str
    report_timestamp: datetime
    report_type: str
    report_period: str
    executive_summary: str
    key_findings: List[str]
    detailed_analysis: Dict[str, Any]
    recommendations: List[str]
    appendices: Dict[str, Any]
    metadata: Dict[str, Any]


class ContentAnalyticsEngine:
    """Advanced content analytics and business intelligence engine"""
    
    def __init__(self):
        self.db_engine = None
        self.redis_client = None
        self.analytics_cache = {}
        self.report_templates = {}
        self.dashboard_configs = {}
        self.kpi_definitions = {}
        self.alert_rules = {}
        self.models_loaded = False
        
    async def initialize(self) -> None:
        """Initialize the analytics engine"""
        try:
            logger.info("Initializing Content Analytics Engine...")
            
            # Initialize database connection
            await self._initialize_database()
            
            # Initialize Redis connection
            await self._initialize_redis()
            
            # Load analytics configurations
            await self._load_analytics_configs()
            
            # Load KPI definitions
            await self._load_kpi_definitions()
            
            # Load alert rules
            await self._load_alert_rules()
            
            self.models_loaded = True
            logger.info("Content Analytics Engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing Content Analytics Engine: {e}")
            raise
    
    async def _initialize_database(self) -> None:
        """Initialize database connection"""
        try:
            # This would typically connect to a real database
            # For now, we'll use an in-memory approach
            self.db_engine = None
            logger.info("Database connection initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize database: {e}")
    
    async def _initialize_redis(self) -> None:
        """Initialize Redis connection"""
        try:
            # This would typically connect to Redis
            # For now, we'll use an in-memory cache
            self.redis_client = None
            logger.info("Redis connection initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Redis: {e}")
    
    async def _load_analytics_configs(self) -> None:
        """Load analytics configurations"""
        try:
            self.analytics_cache = {
                "content_metrics": {
                    "word_count": True,
                    "readability_score": True,
                    "sentiment_score": True,
                    "topic_classification": True,
                    "entity_extraction": True
                },
                "performance_metrics": {
                    "views": True,
                    "engagement_rate": True,
                    "conversion_rate": True,
                    "bounce_rate": True,
                    "time_on_page": True
                },
                "seo_metrics": {
                    "keyword_density": True,
                    "meta_tags": True,
                    "internal_links": True,
                    "external_links": True,
                    "page_speed": True
                }
            }
            logger.info("Analytics configurations loaded")
        except Exception as e:
            logger.warning(f"Failed to load analytics configs: {e}")
    
    async def _load_kpi_definitions(self) -> None:
        """Load KPI definitions"""
        try:
            self.kpi_definitions = {
                "content_quality": {
                    "readability_score": {"target": 70, "weight": 0.3},
                    "sentiment_score": {"target": 0.7, "weight": 0.2},
                    "uniqueness_score": {"target": 0.8, "weight": 0.3},
                    "completeness_score": {"target": 0.9, "weight": 0.2}
                },
                "performance": {
                    "engagement_rate": {"target": 0.15, "weight": 0.4},
                    "conversion_rate": {"target": 0.05, "weight": 0.3},
                    "bounce_rate": {"target": 0.3, "weight": 0.2},
                    "time_on_page": {"target": 180, "weight": 0.1}
                },
                "seo": {
                    "organic_traffic": {"target": 1000, "weight": 0.4},
                    "keyword_rankings": {"target": 10, "weight": 0.3},
                    "backlinks": {"target": 50, "weight": 0.2},
                    "page_speed": {"target": 2.0, "weight": 0.1}
                }
            }
            logger.info("KPI definitions loaded")
        except Exception as e:
            logger.warning(f"Failed to load KPI definitions: {e}")
    
    async def _load_alert_rules(self) -> None:
        """Load alert rules"""
        try:
            self.alert_rules = {
                "performance_alerts": {
                    "low_engagement": {"threshold": 0.05, "severity": "warning"},
                    "high_bounce_rate": {"threshold": 0.7, "severity": "critical"},
                    "low_conversion": {"threshold": 0.01, "severity": "warning"}
                },
                "quality_alerts": {
                    "low_readability": {"threshold": 50, "severity": "warning"},
                    "negative_sentiment": {"threshold": -0.5, "severity": "critical"},
                    "duplicate_content": {"threshold": 0.8, "severity": "warning"}
                },
                "seo_alerts": {
                    "traffic_drop": {"threshold": -0.2, "severity": "critical"},
                    "ranking_drop": {"threshold": -5, "severity": "warning"},
                    "slow_page_speed": {"threshold": 5.0, "severity": "warning"}
                }
            }
            logger.info("Alert rules loaded")
        except Exception as e:
            logger.warning(f"Failed to load alert rules: {e}")
    
    async def analyze_content_analytics(
        self,
        content: str,
        content_id: str = "",
        context: Dict[str, Any] = None
    ) -> ContentAnalytics:
        """Perform comprehensive content analytics analysis"""
        
        if not self.models_loaded:
            raise Exception("Analytics engine not loaded. Call initialize() first.")
        
        if context is None:
            context = {}
        
        try:
            # Run all analytics analyses in parallel
            results = await asyncio.gather(
                self._analyze_content_metrics(content),
                self._analyze_performance_metrics(content, context),
                self._analyze_engagement_metrics(content, context),
                self._analyze_quality_metrics(content),
                self._analyze_seo_metrics(content),
                self._analyze_audience_metrics(content, context),
                self._analyze_competitive_metrics(content, context),
                self._analyze_trend_metrics(content, context),
                return_exceptions=True
            )
            
            # Extract results
            content_metrics = results[0] if not isinstance(results[0], Exception) else {}
            performance_metrics = results[1] if not isinstance(results[1], Exception) else {}
            engagement_metrics = results[2] if not isinstance(results[2], Exception) else {}
            quality_metrics = results[3] if not isinstance(results[3], Exception) else {}
            seo_metrics = results[4] if not isinstance(results[4], Exception) else {}
            audience_metrics = results[5] if not isinstance(results[5], Exception) else {}
            competitive_metrics = results[6] if not isinstance(results[6], Exception) else {}
            trend_metrics = results[7] if not isinstance(results[7], Exception) else {}
            
            # Generate insights and recommendations
            insights = await self._generate_analytics_insights(
                content_metrics, performance_metrics, engagement_metrics,
                quality_metrics, seo_metrics, audience_metrics,
                competitive_metrics, trend_metrics
            )
            
            recommendations = await self._generate_analytics_recommendations(
                content_metrics, performance_metrics, engagement_metrics,
                quality_metrics, seo_metrics, audience_metrics,
                competitive_metrics, trend_metrics
            )
            
            return ContentAnalytics(
                content_id=content_id,
                analytics_timestamp=datetime.now(),
                content_metrics=content_metrics,
                performance_metrics=performance_metrics,
                engagement_metrics=engagement_metrics,
                quality_metrics=quality_metrics,
                seo_metrics=seo_metrics,
                audience_metrics=audience_metrics,
                competitive_metrics=competitive_metrics,
                trend_metrics=trend_metrics,
                insights=insights,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Error in content analytics analysis: {e}")
            raise
    
    async def _analyze_content_metrics(self, content: str) -> Dict[str, Any]:
        """Analyze content metrics"""
        try:
            words = content.split()
            sentences = content.split('.')
            
            # Basic content metrics
            word_count = len(words)
            sentence_count = len(sentences)
            avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
            
            # Readability metrics
            complex_words = [word for word in words if len(word) > 6]
            readability_score = max(0, 100 - (avg_sentence_length * 1.5) - (len(complex_words) / word_count * 100))
            
            # Content diversity
            unique_words = len(set(words))
            vocabulary_richness = unique_words / word_count if word_count > 0 else 0
            
            # Content structure
            paragraphs = content.split('\n\n')
            paragraph_count = len(paragraphs)
            avg_paragraph_length = word_count / paragraph_count if paragraph_count > 0 else 0
            
            return {
                "word_count": word_count,
                "sentence_count": sentence_count,
                "paragraph_count": paragraph_count,
                "avg_sentence_length": round(avg_sentence_length, 2),
                "avg_paragraph_length": round(avg_paragraph_length, 2),
                "readability_score": round(readability_score, 2),
                "vocabulary_richness": round(vocabulary_richness, 3),
                "complex_word_ratio": round(len(complex_words) / word_count, 3) if word_count > 0 else 0
            }
            
        except Exception as e:
            logger.warning(f"Content metrics analysis failed: {e}")
            return {}
    
    async def _analyze_performance_metrics(self, content: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze performance metrics"""
        try:
            # Simulate performance data (in real implementation, this would come from analytics)
            performance_data = context.get("performance_data", {})
            
            views = performance_data.get("views", 1000)
            clicks = performance_data.get("clicks", 150)
            conversions = performance_data.get("conversions", 25)
            time_on_page = performance_data.get("time_on_page", 120)
            
            # Calculate derived metrics
            click_through_rate = clicks / views if views > 0 else 0
            conversion_rate = conversions / views if views > 0 else 0
            engagement_rate = (clicks + conversions) / views if views > 0 else 0
            
            return {
                "views": views,
                "clicks": clicks,
                "conversions": conversions,
                "time_on_page": time_on_page,
                "click_through_rate": round(click_through_rate, 3),
                "conversion_rate": round(conversion_rate, 3),
                "engagement_rate": round(engagement_rate, 3),
                "bounce_rate": round(1 - engagement_rate, 3)
            }
            
        except Exception as e:
            logger.warning(f"Performance metrics analysis failed: {e}")
            return {}
    
    async def _analyze_engagement_metrics(self, content: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze engagement metrics"""
        try:
            # Simulate engagement data
            engagement_data = context.get("engagement_data", {})
            
            likes = engagement_data.get("likes", 50)
            shares = engagement_data.get("shares", 15)
            comments = engagement_data.get("comments", 8)
            bookmarks = engagement_data.get("bookmarks", 12)
            
            # Calculate engagement scores
            total_engagement = likes + shares + comments + bookmarks
            engagement_score = min(1.0, total_engagement / 100)  # Normalize to 0-1
            
            # Social signals
            social_signals = {
                "likes": likes,
                "shares": shares,
                "comments": comments,
                "bookmarks": bookmarks,
                "total_engagement": total_engagement,
                "engagement_score": round(engagement_score, 3)
            }
            
            return social_signals
            
        except Exception as e:
            logger.warning(f"Engagement metrics analysis failed: {e}")
            return {}
    
    async def _analyze_quality_metrics(self, content: str) -> Dict[str, Any]:
        """Analyze content quality metrics"""
        try:
            # Content completeness
            has_title = any(word in content.lower() for word in ['title', 'heading', 'h1', 'h2'])
            has_intro = len(content.split('\n')[0]) > 50
            has_conclusion = 'conclusion' in content.lower() or 'summary' in content.lower()
            
            completeness_score = sum([has_title, has_intro, has_conclusion]) / 3
            
            # Content uniqueness (simplified)
            words = content.split()
            unique_ratio = len(set(words)) / len(words) if words else 0
            
            # Content depth
            depth_score = min(1.0, len(content) / 2000)  # Normalize based on length
            
            return {
                "completeness_score": round(completeness_score, 3),
                "uniqueness_score": round(unique_ratio, 3),
                "depth_score": round(depth_score, 3),
                "has_title": has_title,
                "has_intro": has_intro,
                "has_conclusion": has_conclusion,
                "overall_quality": round((completeness_score + unique_ratio + depth_score) / 3, 3)
            }
            
        except Exception as e:
            logger.warning(f"Quality metrics analysis failed: {e}")
            return {}
    
    async def _analyze_seo_metrics(self, content: str) -> Dict[str, Any]:
        """Analyze SEO metrics"""
        try:
            content_lower = content.lower()
            words = content.split()
            
            # Keyword analysis
            word_freq = Counter(words)
            keywords = {word: count for word, count in word_freq.items() 
                       if len(word) > 3 and count > 1}
            
            # SEO elements
            has_meta_description = 'meta' in content_lower
            has_headers = any(tag in content_lower for tag in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
            has_links = 'http' in content_lower or 'www.' in content_lower
            has_images = 'img' in content_lower or 'image' in content_lower
            
            # SEO score calculation
            seo_elements = sum([has_meta_description, has_headers, has_links, has_images])
            seo_score = seo_elements / 4
            
            return {
                "keyword_count": len(keywords),
                "top_keywords": dict(list(keywords.items())[:5]),
                "has_meta_description": has_meta_description,
                "has_headers": has_headers,
                "has_links": has_links,
                "has_images": has_images,
                "seo_score": round(seo_score, 3),
                "keyword_density": round(len(keywords) / len(words), 3) if words else 0
            }
            
        except Exception as e:
            logger.warning(f"SEO metrics analysis failed: {e}")
            return {}
    
    async def _analyze_audience_metrics(self, content: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze audience metrics"""
        try:
            # Simulate audience data
            audience_data = context.get("audience_data", {})
            
            demographics = audience_data.get("demographics", {
                "age_groups": {"18-24": 0.2, "25-34": 0.3, "35-44": 0.25, "45-54": 0.15, "55+": 0.1},
                "genders": {"male": 0.55, "female": 0.45},
                "locations": {"US": 0.4, "UK": 0.2, "CA": 0.15, "AU": 0.1, "Other": 0.15}
            })
            
            # Content appeal analysis
            content_lower = content.lower()
            professional_terms = ['business', 'professional', 'corporate', 'enterprise']
            casual_terms = ['fun', 'awesome', 'cool', 'amazing']
            
            professional_score = sum(content_lower.count(term) for term in professional_terms)
            casual_score = sum(content_lower.count(term) for term in casual_terms)
            
            audience_type = "professional" if professional_score > casual_score else "casual"
            
            return {
                "demographics": demographics,
                "audience_type": audience_type,
                "professional_score": professional_score,
                "casual_score": casual_score,
                "target_audience_match": 0.8  # Simulated
            }
            
        except Exception as e:
            logger.warning(f"Audience metrics analysis failed: {e}")
            return {}
    
    async def _analyze_competitive_metrics(self, content: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze competitive metrics"""
        try:
            # Simulate competitive data
            competitive_data = context.get("competitive_data", {})
            
            market_position = competitive_data.get("market_position", "middle")
            competitor_count = competitive_data.get("competitor_count", 5)
            market_share = competitive_data.get("market_share", 0.15)
            
            # Content differentiation
            unique_elements = len(set(content.split()))
            differentiation_score = min(1.0, unique_elements / 1000)
            
            return {
                "market_position": market_position,
                "competitor_count": competitor_count,
                "market_share": market_share,
                "differentiation_score": round(differentiation_score, 3),
                "competitive_advantage": "content_quality" if differentiation_score > 0.7 else "standard"
            }
            
        except Exception as e:
            logger.warning(f"Competitive metrics analysis failed: {e}")
            return {}
    
    async def _analyze_trend_metrics(self, content: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze trend metrics"""
        try:
            # Simulate trend data
            trend_data = context.get("trend_data", {})
            
            trending_topics = trend_data.get("trending_topics", ["AI", "sustainability", "remote work"])
            content_topics = [word for word in content.lower().split() if len(word) > 4]
            
            # Trend alignment
            trend_alignment = sum(1 for topic in trending_topics if topic.lower() in content.lower()) / len(trending_topics)
            
            # Viral potential
            viral_keywords = ['viral', 'trending', 'breaking', 'exclusive', 'shocking']
            viral_score = sum(content.lower().count(keyword) for keyword in viral_keywords) / 10
            
            return {
                "trending_topics": trending_topics,
                "trend_alignment": round(trend_alignment, 3),
                "viral_score": round(viral_score, 3),
                "trend_potential": "high" if trend_alignment > 0.5 else "medium" if trend_alignment > 0.2 else "low"
            }
            
        except Exception as e:
            logger.warning(f"Trend metrics analysis failed: {e}")
            return {}
    
    async def _generate_analytics_insights(
        self,
        content_metrics: Dict[str, Any],
        performance_metrics: Dict[str, Any],
        engagement_metrics: Dict[str, Any],
        quality_metrics: Dict[str, Any],
        seo_metrics: Dict[str, Any],
        audience_metrics: Dict[str, Any],
        competitive_metrics: Dict[str, Any],
        trend_metrics: Dict[str, Any]
    ) -> List[str]:
        """Generate analytics insights"""
        try:
            insights = []
            
            # Content insights
            if content_metrics.get("readability_score", 0) < 60:
                insights.append("Content readability is below optimal level - consider simplifying language")
            
            if content_metrics.get("vocabulary_richness", 0) > 0.7:
                insights.append("High vocabulary richness indicates sophisticated content")
            
            # Performance insights
            if performance_metrics.get("engagement_rate", 0) > 0.2:
                insights.append("High engagement rate indicates strong audience connection")
            
            if performance_metrics.get("conversion_rate", 0) > 0.05:
                insights.append("Above-average conversion rate suggests effective call-to-action")
            
            # Quality insights
            if quality_metrics.get("overall_quality", 0) > 0.8:
                insights.append("High overall content quality score")
            
            # SEO insights
            if seo_metrics.get("seo_score", 0) > 0.7:
                insights.append("Good SEO optimization with proper structure")
            
            # Trend insights
            if trend_metrics.get("trend_alignment", 0) > 0.5:
                insights.append("Content aligns well with current trends")
            
            return insights
            
        except Exception as e:
            logger.warning(f"Insights generation failed: {e}")
            return []
    
    async def _generate_analytics_recommendations(
        self,
        content_metrics: Dict[str, Any],
        performance_metrics: Dict[str, Any],
        engagement_metrics: Dict[str, Any],
        quality_metrics: Dict[str, Any],
        seo_metrics: Dict[str, Any],
        audience_metrics: Dict[str, Any],
        competitive_metrics: Dict[str, Any],
        trend_metrics: Dict[str, Any]
    ) -> List[str]:
        """Generate analytics recommendations"""
        try:
            recommendations = []
            
            # Content recommendations
            if content_metrics.get("readability_score", 0) < 60:
                recommendations.append("Improve readability by using shorter sentences and simpler words")
            
            if content_metrics.get("avg_sentence_length", 0) > 20:
                recommendations.append("Reduce average sentence length for better readability")
            
            # Performance recommendations
            if performance_metrics.get("bounce_rate", 0) > 0.5:
                recommendations.append("Reduce bounce rate by improving content engagement")
            
            if performance_metrics.get("time_on_page", 0) < 60:
                recommendations.append("Increase time on page with more engaging content")
            
            # SEO recommendations
            if not seo_metrics.get("has_headers", False):
                recommendations.append("Add proper heading structure (H1, H2, H3) for better SEO")
            
            if seo_metrics.get("keyword_count", 0) < 3:
                recommendations.append("Include more relevant keywords for better SEO")
            
            # Quality recommendations
            if not quality_metrics.get("has_conclusion", False):
                recommendations.append("Add a conclusion or summary section")
            
            # Trend recommendations
            if trend_metrics.get("trend_alignment", 0) < 0.3:
                recommendations.append("Consider incorporating trending topics to increase relevance")
            
            return recommendations
            
        except Exception as e:
            logger.warning(f"Recommendations generation failed: {e}")
            return []
    
    async def generate_business_intelligence(
        self,
        content_list: List[str],
        time_period: str = "30d"
    ) -> BusinessIntelligence:
        """Generate business intelligence analysis"""
        
        try:
            # Analyze multiple content pieces
            analytics_results = []
            for content in content_list:
                analytics = await self.analyze_content_analytics(content)
                analytics_results.append(analytics)
            
            # Aggregate metrics
            content_performance = await self._aggregate_content_performance(analytics_results)
            audience_insights = await self._aggregate_audience_insights(analytics_results)
            market_trends = await self._analyze_market_trends(analytics_results)
            competitive_analysis = await self._analyze_competitive_position(analytics_results)
            roi_metrics = await self._calculate_roi_metrics(analytics_results)
            growth_metrics = await self._calculate_growth_metrics(analytics_results)
            risk_assessment = await self._assess_risks(analytics_results)
            
            # Generate strategic recommendations
            strategic_recommendations = await self._generate_strategic_recommendations(
                content_performance, audience_insights, market_trends,
                competitive_analysis, roi_metrics, growth_metrics, risk_assessment
            )
            
            return BusinessIntelligence(
                analysis_id=f"bi_{int(datetime.now().timestamp())}",
                analysis_timestamp=datetime.now(),
                content_performance=content_performance,
                audience_insights=audience_insights,
                market_trends=market_trends,
                competitive_analysis=competitive_analysis,
                roi_metrics=roi_metrics,
                growth_metrics=growth_metrics,
                risk_assessment=risk_assessment,
                strategic_recommendations=strategic_recommendations
            )
            
        except Exception as e:
            logger.error(f"Error generating business intelligence: {e}")
            raise
    
    async def _aggregate_content_performance(self, analytics_results: List[ContentAnalytics]) -> Dict[str, Any]:
        """Aggregate content performance metrics"""
        try:
            if not analytics_results:
                return {}
            
            # Aggregate performance metrics
            total_views = sum(r.performance_metrics.get("views", 0) for r in analytics_results)
            total_engagement = sum(r.engagement_metrics.get("total_engagement", 0) for r in analytics_results)
            avg_quality = statistics.mean(r.quality_metrics.get("overall_quality", 0) for r in analytics_results)
            avg_seo_score = statistics.mean(r.seo_metrics.get("seo_score", 0) for r in analytics_results)
            
            return {
                "total_content_pieces": len(analytics_results),
                "total_views": total_views,
                "total_engagement": total_engagement,
                "average_quality_score": round(avg_quality, 3),
                "average_seo_score": round(avg_seo_score, 3),
                "top_performing_content": len([r for r in analytics_results if r.performance_metrics.get("engagement_rate", 0) > 0.15])
            }
            
        except Exception as e:
            logger.warning(f"Content performance aggregation failed: {e}")
            return {}
    
    async def _aggregate_audience_insights(self, analytics_results: List[ContentAnalytics]) -> Dict[str, Any]:
        """Aggregate audience insights"""
        try:
            if not analytics_results:
                return {}
            
            # Aggregate audience data
            audience_types = [r.audience_metrics.get("audience_type", "unknown") for r in analytics_results]
            audience_type_distribution = Counter(audience_types)
            
            return {
                "audience_type_distribution": dict(audience_type_distribution),
                "primary_audience_type": audience_type_distribution.most_common(1)[0][0] if audience_type_distribution else "unknown",
                "audience_diversity": len(audience_type_distribution)
            }
            
        except Exception as e:
            logger.warning(f"Audience insights aggregation failed: {e}")
            return {}
    
    async def _analyze_market_trends(self, analytics_results: List[ContentAnalytics]) -> Dict[str, Any]:
        """Analyze market trends"""
        try:
            if not analytics_results:
                return {}
            
            # Aggregate trend data
            trend_alignments = [r.trend_metrics.get("trend_alignment", 0) for r in analytics_results]
            viral_scores = [r.trend_metrics.get("viral_score", 0) for r in analytics_results]
            
            return {
                "average_trend_alignment": round(statistics.mean(trend_alignments), 3),
                "average_viral_score": round(statistics.mean(viral_scores), 3),
                "trend_aware_content": len([r for r in analytics_results if r.trend_metrics.get("trend_alignment", 0) > 0.5])
            }
            
        except Exception as e:
            logger.warning(f"Market trends analysis failed: {e}")
            return {}
    
    async def _analyze_competitive_position(self, analytics_results: List[ContentAnalytics]) -> Dict[str, Any]:
        """Analyze competitive position"""
        try:
            if not analytics_results:
                return {}
            
            # Aggregate competitive data
            differentiation_scores = [r.competitive_metrics.get("differentiation_score", 0) for r in analytics_results]
            market_positions = [r.competitive_metrics.get("market_position", "unknown") for r in analytics_results]
            
            return {
                "average_differentiation_score": round(statistics.mean(differentiation_scores), 3),
                "market_position_distribution": dict(Counter(market_positions)),
                "competitive_advantage_content": len([r for r in analytics_results if r.competitive_metrics.get("differentiation_score", 0) > 0.7])
            }
            
        except Exception as e:
            logger.warning(f"Competitive position analysis failed: {e}")
            return {}
    
    async def _calculate_roi_metrics(self, analytics_results: List[ContentAnalytics]) -> Dict[str, Any]:
        """Calculate ROI metrics"""
        try:
            if not analytics_results:
                return {}
            
            # Calculate ROI metrics
            total_investment = len(analytics_results) * 100  # Simulated cost per content piece
            total_revenue = sum(r.performance_metrics.get("conversions", 0) * 50 for r in analytics_results)  # Simulated revenue per conversion
            
            roi = (total_revenue - total_investment) / total_investment if total_investment > 0 else 0
            
            return {
                "total_investment": total_investment,
                "total_revenue": total_revenue,
                "roi": round(roi, 3),
                "roi_percentage": round(roi * 100, 2),
                "cost_per_conversion": round(total_investment / sum(r.performance_metrics.get("conversions", 0) for r in analytics_results), 2) if any(r.performance_metrics.get("conversions", 0) for r in analytics_results) else 0
            }
            
        except Exception as e:
            logger.warning(f"ROI metrics calculation failed: {e}")
            return {}
    
    async def _calculate_growth_metrics(self, analytics_results: List[ContentAnalytics]) -> Dict[str, Any]:
        """Calculate growth metrics"""
        try:
            if not analytics_results:
                return {}
            
            # Calculate growth metrics (simplified)
            total_views = sum(r.performance_metrics.get("views", 0) for r in analytics_results)
            total_engagement = sum(r.engagement_metrics.get("total_engagement", 0) for r in analytics_results)
            
            return {
                "content_growth_rate": len(analytics_results),  # Number of content pieces
                "view_growth_rate": total_views / len(analytics_results) if analytics_results else 0,
                "engagement_growth_rate": total_engagement / len(analytics_results) if analytics_results else 0,
                "quality_improvement_rate": statistics.mean(r.quality_metrics.get("overall_quality", 0) for r in analytics_results)
            }
            
        except Exception as e:
            logger.warning(f"Growth metrics calculation failed: {e}")
            return {}
    
    async def _assess_risks(self, analytics_results: List[ContentAnalytics]) -> Dict[str, Any]:
        """Assess content risks"""
        try:
            if not analytics_results:
                return {}
            
            # Assess various risks
            low_quality_content = len([r for r in analytics_results if r.quality_metrics.get("overall_quality", 0) < 0.5])
            low_engagement_content = len([r for r in analytics_results if r.performance_metrics.get("engagement_rate", 0) < 0.05])
            seo_issues = len([r for r in analytics_results if r.seo_metrics.get("seo_score", 0) < 0.3])
            
            return {
                "low_quality_content_count": low_quality_content,
                "low_engagement_content_count": low_engagement_content,
                "seo_issues_count": seo_issues,
                "overall_risk_score": round((low_quality_content + low_engagement_content + seo_issues) / (len(analytics_results) * 3), 3),
                "risk_level": "high" if (low_quality_content + low_engagement_content + seo_issues) > len(analytics_results) * 0.3 else "medium" if (low_quality_content + low_engagement_content + seo_issues) > len(analytics_results) * 0.1 else "low"
            }
            
        except Exception as e:
            logger.warning(f"Risk assessment failed: {e}")
            return {}
    
    async def _generate_strategic_recommendations(
        self,
        content_performance: Dict[str, Any],
        audience_insights: Dict[str, Any],
        market_trends: Dict[str, Any],
        competitive_analysis: Dict[str, Any],
        roi_metrics: Dict[str, Any],
        growth_metrics: Dict[str, Any],
        risk_assessment: Dict[str, Any]
    ) -> List[str]:
        """Generate strategic recommendations"""
        try:
            recommendations = []
            
            # Performance-based recommendations
            if content_performance.get("average_quality_score", 0) < 0.7:
                recommendations.append("Focus on improving overall content quality to increase engagement")
            
            if content_performance.get("average_seo_score", 0) < 0.6:
                recommendations.append("Invest in SEO optimization to improve organic reach")
            
            # ROI-based recommendations
            if roi_metrics.get("roi", 0) < 0.2:
                recommendations.append("Optimize content ROI by focusing on high-converting content types")
            
            # Growth-based recommendations
            if growth_metrics.get("engagement_growth_rate", 0) < 10:
                recommendations.append("Implement engagement strategies to boost audience interaction")
            
            # Risk-based recommendations
            if risk_assessment.get("risk_level") == "high":
                recommendations.append("Address quality and engagement issues to reduce content risks")
            
            # Competitive recommendations
            if competitive_analysis.get("average_differentiation_score", 0) < 0.5:
                recommendations.append("Develop unique content angles to differentiate from competitors")
            
            return recommendations
            
        except Exception as e:
            logger.warning(f"Strategic recommendations generation failed: {e}")
            return []
    
    async def create_content_dashboard(
        self,
        content_list: List[str],
        dashboard_config: Dict[str, Any] = None
    ) -> ContentDashboard:
        """Create content analytics dashboard"""
        
        try:
            if dashboard_config is None:
                dashboard_config = {}
            
            # Generate analytics for all content
            analytics_results = []
            for content in content_list:
                analytics = await self.analyze_content_analytics(content)
                analytics_results.append(analytics)
            
            # Create dashboard data
            overview_metrics = await self._create_overview_metrics(analytics_results)
            content_metrics = await self._create_content_metrics(analytics_results)
            performance_metrics = await self._create_performance_metrics(analytics_results)
            audience_metrics = await self._create_audience_metrics(analytics_results)
            trend_metrics = await self._create_trend_metrics(analytics_results)
            alerts = await self._generate_dashboard_alerts(analytics_results)
            charts_data = await self._create_charts_data(analytics_results)
            kpis = await self._calculate_kpis(analytics_results)
            
            return ContentDashboard(
                dashboard_id=f"dashboard_{int(datetime.now().timestamp())}",
                dashboard_timestamp=datetime.now(),
                overview_metrics=overview_metrics,
                content_metrics=content_metrics,
                performance_metrics=performance_metrics,
                audience_metrics=audience_metrics,
                trend_metrics=trend_metrics,
                alerts=alerts,
                charts_data=charts_data,
                kpis=kpis
            )
            
        except Exception as e:
            logger.error(f"Error creating content dashboard: {e}")
            raise
    
    async def _create_overview_metrics(self, analytics_results: List[ContentAnalytics]) -> Dict[str, Any]:
        """Create overview metrics for dashboard"""
        try:
            if not analytics_results:
                return {}
            
            total_content = len(analytics_results)
            total_views = sum(r.performance_metrics.get("views", 0) for r in analytics_results)
            total_engagement = sum(r.engagement_metrics.get("total_engagement", 0) for r in analytics_results)
            avg_quality = statistics.mean(r.quality_metrics.get("overall_quality", 0) for r in analytics_results)
            
            return {
                "total_content_pieces": total_content,
                "total_views": total_views,
                "total_engagement": total_engagement,
                "average_quality_score": round(avg_quality, 3),
                "content_health_score": round(avg_quality * 100, 1)
            }
            
        except Exception as e:
            logger.warning(f"Overview metrics creation failed: {e}")
            return {}
    
    async def _create_content_metrics(self, analytics_results: List[ContentAnalytics]) -> Dict[str, Any]:
        """Create content metrics for dashboard"""
        try:
            if not analytics_results:
                return {}
            
            readability_scores = [r.content_metrics.get("readability_score", 0) for r in analytics_results]
            word_counts = [r.content_metrics.get("word_count", 0) for r in analytics_results]
            
            return {
                "average_readability": round(statistics.mean(readability_scores), 2),
                "average_word_count": round(statistics.mean(word_counts), 0),
                "readability_distribution": {
                    "excellent": len([s for s in readability_scores if s >= 80]),
                    "good": len([s for s in readability_scores if 60 <= s < 80]),
                    "fair": len([s for s in readability_scores if 40 <= s < 60]),
                    "poor": len([s for s in readability_scores if s < 40])
                }
            }
            
        except Exception as e:
            logger.warning(f"Content metrics creation failed: {e}")
            return {}
    
    async def _create_performance_metrics(self, analytics_results: List[ContentAnalytics]) -> Dict[str, Any]:
        """Create performance metrics for dashboard"""
        try:
            if not analytics_results:
                return {}
            
            engagement_rates = [r.performance_metrics.get("engagement_rate", 0) for r in analytics_results]
            conversion_rates = [r.performance_metrics.get("conversion_rate", 0) for r in analytics_results]
            
            return {
                "average_engagement_rate": round(statistics.mean(engagement_rates), 3),
                "average_conversion_rate": round(statistics.mean(conversion_rates), 3),
                "top_performers": len([r for r in analytics_results if r.performance_metrics.get("engagement_rate", 0) > 0.15]),
                "underperformers": len([r for r in analytics_results if r.performance_metrics.get("engagement_rate", 0) < 0.05])
            }
            
        except Exception as e:
            logger.warning(f"Performance metrics creation failed: {e}")
            return {}
    
    async def _create_audience_metrics(self, analytics_results: List[ContentAnalytics]) -> Dict[str, Any]:
        """Create audience metrics for dashboard"""
        try:
            if not analytics_results:
                return {}
            
            audience_types = [r.audience_metrics.get("audience_type", "unknown") for r in analytics_results]
            audience_distribution = Counter(audience_types)
            
            return {
                "audience_distribution": dict(audience_distribution),
                "primary_audience": audience_distribution.most_common(1)[0][0] if audience_distribution else "unknown",
                "audience_diversity": len(audience_distribution)
            }
            
        except Exception as e:
            logger.warning(f"Audience metrics creation failed: {e}")
            return {}
    
    async def _create_trend_metrics(self, analytics_results: List[ContentAnalytics]) -> Dict[str, Any]:
        """Create trend metrics for dashboard"""
        try:
            if not analytics_results:
                return {}
            
            trend_alignments = [r.trend_metrics.get("trend_alignment", 0) for r in analytics_results]
            viral_scores = [r.trend_metrics.get("viral_score", 0) for r in analytics_results]
            
            return {
                "average_trend_alignment": round(statistics.mean(trend_alignments), 3),
                "average_viral_score": round(statistics.mean(viral_scores), 3),
                "trend_aware_content": len([r for r in analytics_results if r.trend_metrics.get("trend_alignment", 0) > 0.5])
            }
            
        except Exception as e:
            logger.warning(f"Trend metrics creation failed: {e}")
            return {}
    
    async def _generate_dashboard_alerts(self, analytics_results: List[ContentAnalytics]) -> List[Dict[str, Any]]:
        """Generate dashboard alerts"""
        try:
            alerts = []
            
            # Check alert rules
            for analytics in analytics_results:
                # Performance alerts
                if analytics.performance_metrics.get("engagement_rate", 0) < 0.05:
                    alerts.append({
                        "type": "performance",
                        "severity": "warning",
                        "message": f"Low engagement rate for content {analytics.content_id}",
                        "metric": "engagement_rate",
                        "value": analytics.performance_metrics.get("engagement_rate", 0)
                    })
                
                # Quality alerts
                if analytics.quality_metrics.get("overall_quality", 0) < 0.5:
                    alerts.append({
                        "type": "quality",
                        "severity": "critical",
                        "message": f"Low quality score for content {analytics.content_id}",
                        "metric": "overall_quality",
                        "value": analytics.quality_metrics.get("overall_quality", 0)
                    })
                
                # SEO alerts
                if analytics.seo_metrics.get("seo_score", 0) < 0.3:
                    alerts.append({
                        "type": "seo",
                        "severity": "warning",
                        "message": f"Poor SEO optimization for content {analytics.content_id}",
                        "metric": "seo_score",
                        "value": analytics.seo_metrics.get("seo_score", 0)
                    })
            
            return alerts
            
        except Exception as e:
            logger.warning(f"Dashboard alerts generation failed: {e}")
            return []
    
    async def _create_charts_data(self, analytics_results: List[ContentAnalytics]) -> Dict[str, Any]:
        """Create charts data for dashboard"""
        try:
            if not analytics_results:
                return {}
            
            # Performance over time (simulated)
            performance_data = []
            for i, analytics in enumerate(analytics_results):
                performance_data.append({
                    "date": (datetime.now() - timedelta(days=len(analytics_results)-i)).isoformat(),
                    "views": analytics.performance_metrics.get("views", 0),
                    "engagement": analytics.engagement_metrics.get("total_engagement", 0),
                    "quality": analytics.quality_metrics.get("overall_quality", 0) * 100
                })
            
            # Quality distribution
            quality_scores = [r.quality_metrics.get("overall_quality", 0) * 100 for r in analytics_results]
            quality_distribution = {
                "excellent": len([s for s in quality_scores if s >= 80]),
                "good": len([s for s in quality_scores if 60 <= s < 80]),
                "fair": len([s for s in quality_scores if 40 <= s < 60]),
                "poor": len([s for s in quality_scores if s < 40])
            }
            
            return {
                "performance_over_time": performance_data,
                "quality_distribution": quality_distribution,
                "engagement_by_content": [
                    {"content_id": r.content_id, "engagement": r.engagement_metrics.get("total_engagement", 0)}
                    for r in analytics_results
                ]
            }
            
        except Exception as e:
            logger.warning(f"Charts data creation failed: {e}")
            return {}
    
    async def _calculate_kpis(self, analytics_results: List[ContentAnalytics]) -> Dict[str, Any]:
        """Calculate KPIs for dashboard"""
        try:
            if not analytics_results:
                return {}
            
            # Calculate KPIs based on definitions
            kpis = {}
            
            for category, metrics in self.kpi_definitions.items():
                kpis[category] = {}
                for metric, config in metrics.items():
                    if category == "content_quality":
                        if metric == "readability_score":
                            values = [r.content_metrics.get("readability_score", 0) for r in analytics_results]
                            kpis[category][metric] = {
                                "current": round(statistics.mean(values), 2),
                                "target": config["target"],
                                "status": "good" if statistics.mean(values) >= config["target"] else "needs_improvement"
                            }
                        elif metric == "sentiment_score":
                            # Simulated sentiment scores
                            sentiment_scores = [0.7, 0.8, 0.6, 0.9, 0.5]  # Would come from actual sentiment analysis
                            kpis[category][metric] = {
                                "current": round(statistics.mean(sentiment_scores), 2),
                                "target": config["target"],
                                "status": "good" if statistics.mean(sentiment_scores) >= config["target"] else "needs_improvement"
                            }
                    elif category == "performance":
                        if metric == "engagement_rate":
                            values = [r.performance_metrics.get("engagement_rate", 0) for r in analytics_results]
                            kpis[category][metric] = {
                                "current": round(statistics.mean(values), 3),
                                "target": config["target"],
                                "status": "good" if statistics.mean(values) >= config["target"] else "needs_improvement"
                            }
            
            return kpis
            
        except Exception as e:
            logger.warning(f"KPI calculation failed: {e}")
            return {}
    
    async def generate_content_report(
        self,
        content_list: List[str],
        report_type: str = "comprehensive",
        report_period: str = "30d"
    ) -> ContentReport:
        """Generate comprehensive content analytics report"""
        
        try:
            # Generate analytics for all content
            analytics_results = []
            for content in content_list:
                analytics = await self.analyze_content_analytics(content)
                analytics_results.append(analytics)
            
            # Generate business intelligence
            bi_analysis = await self.generate_business_intelligence(content_list, report_period)
            
            # Create executive summary
            executive_summary = await self._create_executive_summary(analytics_results, bi_analysis)
            
            # Extract key findings
            key_findings = await self._extract_key_findings(analytics_results, bi_analysis)
            
            # Create detailed analysis
            detailed_analysis = await self._create_detailed_analysis(analytics_results, bi_analysis)
            
            # Generate recommendations
            recommendations = await self._generate_report_recommendations(analytics_results, bi_analysis)
            
            # Create appendices
            appendices = await self._create_report_appendices(analytics_results)
            
            # Create metadata
            metadata = {
                "report_generated_at": datetime.now().isoformat(),
                "content_pieces_analyzed": len(content_list),
                "analysis_duration": "30d",
                "report_version": "1.0"
            }
            
            return ContentReport(
                report_id=f"report_{int(datetime.now().timestamp())}",
                report_timestamp=datetime.now(),
                report_type=report_type,
                report_period=report_period,
                executive_summary=executive_summary,
                key_findings=key_findings,
                detailed_analysis=detailed_analysis,
                recommendations=recommendations,
                appendices=appendices,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Error generating content report: {e}")
            raise
    
    async def _create_executive_summary(
        self,
        analytics_results: List[ContentAnalytics],
        bi_analysis: BusinessIntelligence
    ) -> str:
        """Create executive summary for report"""
        try:
            total_content = len(analytics_results)
            avg_quality = statistics.mean(r.quality_metrics.get("overall_quality", 0) for r in analytics_results)
            total_views = sum(r.performance_metrics.get("views", 0) for r in analytics_results)
            
            summary = f"""
            Executive Summary:
            
            This comprehensive content analytics report analyzes {total_content} content pieces over the specified period.
            
            Key Performance Indicators:
            - Average content quality score: {avg_quality:.2f}
            - Total content views: {total_views:,}
            - Overall ROI: {bi_analysis.roi_metrics.get('roi_percentage', 0):.1f}%
            
            The analysis reveals {len(bi_analysis.strategic_recommendations)} strategic recommendations for improving content performance and business outcomes.
            """
            
            return summary.strip()
            
        except Exception as e:
            logger.warning(f"Executive summary creation failed: {e}")
            return "Executive summary generation failed."
    
    async def _extract_key_findings(
        self,
        analytics_results: List[ContentAnalytics],
        bi_analysis: BusinessIntelligence
    ) -> List[str]:
        """Extract key findings from analysis"""
        try:
            findings = []
            
            # Content quality findings
            avg_quality = statistics.mean(r.quality_metrics.get("overall_quality", 0) for r in analytics_results)
            if avg_quality > 0.8:
                findings.append("Content quality is consistently high across all pieces")
            elif avg_quality < 0.5:
                findings.append("Content quality needs significant improvement")
            
            # Performance findings
            total_views = sum(r.performance_metrics.get("views", 0) for r in analytics_results)
            if total_views > 10000:
                findings.append("Content is achieving strong viewership")
            
            # ROI findings
            roi = bi_analysis.roi_metrics.get("roi_percentage", 0)
            if roi > 50:
                findings.append("Content is generating positive ROI")
            elif roi < 0:
                findings.append("Content ROI is negative and needs optimization")
            
            # Risk findings
            risk_level = bi_analysis.risk_assessment.get("risk_level", "unknown")
            if risk_level == "high":
                findings.append("High risk content identified requiring immediate attention")
            
            return findings
            
        except Exception as e:
            logger.warning(f"Key findings extraction failed: {e}")
            return []
    
    async def _create_detailed_analysis(
        self,
        analytics_results: List[ContentAnalytics],
        bi_analysis: BusinessIntelligence
    ) -> Dict[str, Any]:
        """Create detailed analysis section"""
        try:
            return {
                "content_performance": bi_analysis.content_performance,
                "audience_insights": bi_analysis.audience_insights,
                "market_trends": bi_analysis.market_trends,
                "competitive_analysis": bi_analysis.competitive_analysis,
                "roi_analysis": bi_analysis.roi_metrics,
                "growth_analysis": bi_analysis.growth_metrics,
                "risk_analysis": bi_analysis.risk_assessment,
                "content_quality_breakdown": {
                    "average_readability": statistics.mean(r.content_metrics.get("readability_score", 0) for r in analytics_results),
                    "average_word_count": statistics.mean(r.content_metrics.get("word_count", 0) for r in analytics_results),
                    "seo_optimization": statistics.mean(r.seo_metrics.get("seo_score", 0) for r in analytics_results)
                }
            }
            
        except Exception as e:
            logger.warning(f"Detailed analysis creation failed: {e}")
            return {}
    
    async def _generate_report_recommendations(
        self,
        analytics_results: List[ContentAnalytics],
        bi_analysis: BusinessIntelligence
    ) -> List[str]:
        """Generate recommendations for report"""
        try:
            recommendations = []
            
            # Add strategic recommendations
            recommendations.extend(bi_analysis.strategic_recommendations)
            
            # Add content-specific recommendations
            avg_quality = statistics.mean(r.quality_metrics.get("overall_quality", 0) for r in analytics_results)
            if avg_quality < 0.7:
                recommendations.append("Implement content quality guidelines and review processes")
            
            avg_seo = statistics.mean(r.seo_metrics.get("seo_score", 0) for r in analytics_results)
            if avg_seo < 0.6:
                recommendations.append("Invest in SEO training and optimization tools")
            
            return recommendations
            
        except Exception as e:
            logger.warning(f"Report recommendations generation failed: {e}")
            return []
    
    async def _create_report_appendices(self, analytics_results: List[ContentAnalytics]) -> Dict[str, Any]:
        """Create report appendices"""
        try:
            return {
                "content_analytics_details": [
                    {
                        "content_id": r.content_id,
                        "quality_score": r.quality_metrics.get("overall_quality", 0),
                        "engagement_rate": r.performance_metrics.get("engagement_rate", 0),
                        "seo_score": r.seo_metrics.get("seo_score", 0)
                    }
                    for r in analytics_results
                ],
                "methodology": {
                    "analytics_engine": "Content Analytics Engine v1.0",
                    "metrics_calculated": ["quality", "performance", "engagement", "seo", "audience", "competitive", "trends"],
                    "data_sources": ["content_analysis", "performance_data", "engagement_data"]
                }
            }
            
        except Exception as e:
            logger.warning(f"Report appendices creation failed: {e}")
            return {}
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of analytics engine"""
        return {
            "status": "healthy" if self.models_loaded else "unhealthy",
            "models_loaded": self.models_loaded,
            "analytics_cache_size": len(self.analytics_cache),
            "kpi_definitions_loaded": len(self.kpi_definitions),
            "alert_rules_loaded": len(self.alert_rules),
            "timestamp": datetime.now().isoformat()
        }


# Global analytics engine instance
content_analytics_engine = ContentAnalyticsEngine()


async def initialize_content_analytics_engine() -> None:
    """Initialize the global analytics engine"""
    await content_analytics_engine.initialize()


async def analyze_content_analytics(
    content: str,
    content_id: str = "",
    context: Dict[str, Any] = None
) -> ContentAnalytics:
    """Analyze content analytics"""
    return await content_analytics_engine.analyze_content_analytics(content, content_id, context)


async def generate_business_intelligence(
    content_list: List[str],
    time_period: str = "30d"
) -> BusinessIntelligence:
    """Generate business intelligence"""
    return await content_analytics_engine.generate_business_intelligence(content_list, time_period)


async def create_content_dashboard(
    content_list: List[str],
    dashboard_config: Dict[str, Any] = None
) -> ContentDashboard:
    """Create content dashboard"""
    return await content_analytics_engine.create_content_dashboard(content_list, dashboard_config)


async def generate_content_report(
    content_list: List[str],
    report_type: str = "comprehensive",
    report_period: str = "30d"
) -> ContentReport:
    """Generate content report"""
    return await content_analytics_engine.generate_content_report(content_list, report_type, report_period)


async def get_analytics_engine_health() -> Dict[str, Any]:
    """Get analytics engine health status"""
    return await content_analytics_engine.health_check()


