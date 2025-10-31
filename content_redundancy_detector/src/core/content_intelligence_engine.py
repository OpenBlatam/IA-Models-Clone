"""
Content Intelligence Engine - Advanced content intelligence and insights
Following FastAPI best practices: functional programming, RORO pattern, async operations
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple, AsyncGenerator
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json
import hashlib
import re
from collections import Counter, defaultdict

import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModel
import torch
from sentence_transformers import SentenceTransformer
import aiohttp
import aiofiles

logger = logging.getLogger(__name__)


@dataclass
class ContentIntelligence:
    """Content intelligence analysis result"""
    content_id: str
    intelligence_score: float
    content_type: str
    complexity_level: str
    target_audience: str
    engagement_potential: float
    viral_potential: float
    seo_potential: float
    conversion_potential: float
    brand_alignment: float
    content_gaps: List[str]
    improvement_opportunities: List[str]
    competitive_advantages: List[str]
    risk_factors: List[str]
    recommendations: List[str]
    analysis_timestamp: datetime


@dataclass
class ContentTrend:
    """Content trend analysis"""
    trend_type: str
    trend_score: float
    trend_direction: str  # "rising", "falling", "stable"
    trend_confidence: float
    related_keywords: List[str]
    trend_impact: str
    trend_timeline: str
    trend_source: str


@dataclass
class ContentInsight:
    """Content insight analysis"""
    insight_type: str
    insight_score: float
    insight_description: str
    insight_evidence: List[str]
    insight_impact: str
    insight_confidence: float
    insight_recommendations: List[str]


@dataclass
class ContentStrategy:
    """Content strategy recommendations"""
    strategy_type: str
    strategy_priority: int
    strategy_description: str
    strategy_goals: List[str]
    strategy_tactics: List[str]
    strategy_metrics: List[str]
    strategy_timeline: str
    strategy_resources: List[str]
    expected_impact: float


class ContentIntelligenceEngine:
    """Advanced content intelligence engine"""
    
    def __init__(self):
        self.trend_analyzer = None
        self.insight_generator = None
        self.strategy_planner = None
        self.competitive_analyzer = None
        self.audience_analyzer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models_loaded = False
        
    async def initialize(self) -> None:
        """Initialize content intelligence models"""
        try:
            logger.info("Initializing Content Intelligence Engine...")
            
            # Load models asynchronously
            await asyncio.gather(
                self._load_trend_analyzer(),
                self._load_insight_generator(),
                self._load_strategy_planner(),
                self._load_competitive_analyzer(),
                self._load_audience_analyzer()
            )
            
            self.models_loaded = True
            logger.info("Content Intelligence Engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing Content Intelligence Engine: {e}")
            raise
    
    async def _load_trend_analyzer(self) -> None:
        """Load trend analysis model"""
        try:
            self.trend_analyzer = pipeline(
                "text-classification",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=self.device
            )
        except Exception as e:
            logger.warning(f"Failed to load trend analyzer: {e}")
            self.trend_analyzer = None
    
    async def _load_insight_generator(self) -> None:
        """Load insight generation model"""
        try:
            self.insight_generator = pipeline(
                "text2text-generation",
                model="t5-base",
                device=self.device
            )
        except Exception as e:
            logger.warning(f"Failed to load insight generator: {e}")
            self.insight_generator = None
    
    async def _load_strategy_planner(self) -> None:
        """Load strategy planning model"""
        try:
            self.strategy_planner = pipeline(
                "text2text-generation",
                model="facebook/bart-base",
                device=self.device
            )
        except Exception as e:
            logger.warning(f"Failed to load strategy planner: {e}")
            self.strategy_planner = None
    
    async def _load_competitive_analyzer(self) -> None:
        """Load competitive analysis model"""
        try:
            self.competitive_analyzer = pipeline(
                "text-classification",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=self.device
            )
        except Exception as e:
            logger.warning(f"Failed to load competitive analyzer: {e}")
            self.competitive_analyzer = None
    
    async def _load_audience_analyzer(self) -> None:
        """Load audience analysis model"""
        try:
            self.audience_analyzer = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=self.device
            )
        except Exception as e:
            logger.warning(f"Failed to load audience analyzer: {e}")
            self.audience_analyzer = None
    
    async def analyze_content_intelligence(
        self, 
        content: str, 
        content_id: str = "",
        context: Dict[str, Any] = None
    ) -> ContentIntelligence:
        """Perform comprehensive content intelligence analysis"""
        
        if not self.models_loaded:
            raise Exception("Content intelligence models not loaded. Call initialize() first.")
        
        if context is None:
            context = {}
        
        try:
            # Run all intelligence analyses in parallel
            results = await asyncio.gather(
                self._analyze_content_type(content),
                self._analyze_complexity_level(content),
                self._analyze_target_audience(content),
                self._analyze_engagement_potential(content),
                self._analyze_viral_potential(content),
                self._analyze_seo_potential(content),
                self._analyze_conversion_potential(content),
                self._analyze_brand_alignment(content, context),
                self._identify_content_gaps(content, context),
                self._identify_improvement_opportunities(content),
                self._identify_competitive_advantages(content, context),
                self._identify_risk_factors(content),
                return_exceptions=True
            )
            
            # Extract results
            content_type = results[0] if not isinstance(results[0], Exception) else "general"
            complexity_level = results[1] if not isinstance(results[1], Exception) else "medium"
            target_audience = results[2] if not isinstance(results[2], Exception) else "general"
            engagement_potential = results[3] if not isinstance(results[3], Exception) else 0.5
            viral_potential = results[4] if not isinstance(results[4], Exception) else 0.3
            seo_potential = results[5] if not isinstance(results[5], Exception) else 0.5
            conversion_potential = results[6] if not isinstance(results[6], Exception) else 0.4
            brand_alignment = results[7] if not isinstance(results[7], Exception) else 0.6
            content_gaps = results[8] if not isinstance(results[8], Exception) else []
            improvement_opportunities = results[9] if not isinstance(results[9], Exception) else []
            competitive_advantages = results[10] if not isinstance(results[10], Exception) else []
            risk_factors = results[11] if not isinstance(results[11], Exception) else []
            
            # Calculate overall intelligence score
            intelligence_score = await self._calculate_intelligence_score(
                engagement_potential, viral_potential, seo_potential, 
                conversion_potential, brand_alignment
            )
            
            # Generate recommendations
            recommendations = await self._generate_recommendations(
                content_type, complexity_level, target_audience,
                engagement_potential, viral_potential, seo_potential,
                conversion_potential, brand_alignment, content_gaps,
                improvement_opportunities, competitive_advantages, risk_factors
            )
            
            return ContentIntelligence(
                content_id=content_id,
                intelligence_score=intelligence_score,
                content_type=content_type,
                complexity_level=complexity_level,
                target_audience=target_audience,
                engagement_potential=engagement_potential,
                viral_potential=viral_potential,
                seo_potential=seo_potential,
                conversion_potential=conversion_potential,
                brand_alignment=brand_alignment,
                content_gaps=content_gaps,
                improvement_opportunities=improvement_opportunities,
                competitive_advantages=competitive_advantages,
                risk_factors=risk_factors,
                recommendations=recommendations,
                analysis_timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error in content intelligence analysis: {e}")
            raise
    
    async def _analyze_content_type(self, content: str) -> str:
        """Analyze content type"""
        try:
            # Simple heuristic-based content type detection
            content_lower = content.lower()
            
            # Check for different content types
            if any(word in content_lower for word in ['tutorial', 'guide', 'how to', 'step by step']):
                return "tutorial"
            elif any(word in content_lower for word in ['news', 'breaking', 'update', 'announcement']):
                return "news"
            elif any(word in content_lower for word in ['review', 'opinion', 'thoughts', 'analysis']):
                return "review"
            elif any(word in content_lower for word in ['story', 'narrative', 'experience', 'journey']):
                return "story"
            elif any(word in content_lower for word in ['product', 'service', 'buy', 'purchase']):
                return "commercial"
            elif any(word in content_lower for word in ['question', 'help', 'support', 'faq']):
                return "support"
            else:
                return "general"
                
        except Exception as e:
            logger.warning(f"Content type analysis failed: {e}")
            return "general"
    
    async def _analyze_complexity_level(self, content: str) -> str:
        """Analyze content complexity level"""
        try:
            # Calculate complexity metrics
            sentences = re.split(r'[.!?]+', content)
            words = content.split()
            
            if not sentences or not words:
                return "low"
            
            # Average sentence length
            avg_sentence_length = len(words) / len(sentences)
            
            # Average word length
            avg_word_length = sum(len(word) for word in words) / len(words)
            
            # Complex word ratio (words > 6 characters)
            complex_words = [word for word in words if len(word) > 6]
            complex_word_ratio = len(complex_words) / len(words)
            
            # Determine complexity level
            if avg_sentence_length > 20 or avg_word_length > 6 or complex_word_ratio > 0.3:
                return "high"
            elif avg_sentence_length > 15 or avg_word_length > 5 or complex_word_ratio > 0.2:
                return "medium"
            else:
                return "low"
                
        except Exception as e:
            logger.warning(f"Complexity analysis failed: {e}")
            return "medium"
    
    async def _analyze_target_audience(self, content: str) -> str:
        """Analyze target audience"""
        try:
            if not self.audience_analyzer:
                return "general"
            
            # Define audience categories
            audience_categories = [
                "professionals", "students", "general public", "experts", 
                "beginners", "business owners", "consumers", "developers"
            ]
            
            result = self.audience_analyzer(content[:512], audience_categories)
            
            # Get top audience category
            top_audience = result['labels'][0]
            confidence = result['scores'][0]
            
            if confidence > 0.6:
                return top_audience
            else:
                return "general"
                
        except Exception as e:
            logger.warning(f"Target audience analysis failed: {e}")
            return "general"
    
    async def _analyze_engagement_potential(self, content: str) -> float:
        """Analyze engagement potential"""
        try:
            engagement_factors = []
            
            # Check for engagement elements
            content_lower = content.lower()
            
            # Questions
            question_count = content.count('?')
            if question_count > 0:
                engagement_factors.append(0.2)
            
            # Call-to-action elements
            cta_words = ['click', 'learn', 'discover', 'find out', 'get started', 'try']
            cta_count = sum(content_lower.count(word) for word in cta_words)
            if cta_count > 0:
                engagement_factors.append(0.3)
            
            # Emotional words
            emotional_words = ['amazing', 'incredible', 'shocking', 'surprising', 'excited', 'love']
            emotional_count = sum(content_lower.count(word) for word in emotional_words)
            if emotional_count > 0:
                engagement_factors.append(0.2)
            
            # Interactive elements
            interactive_words = ['you', 'your', 'imagine', 'think about', 'consider']
            interactive_count = sum(content_lower.count(word) for word in interactive_words)
            if interactive_count > 0:
                engagement_factors.append(0.2)
            
            # Storytelling elements
            story_words = ['story', 'experience', 'journey', 'adventure', 'discovery']
            story_count = sum(content_lower.count(word) for word in story_words)
            if story_count > 0:
                engagement_factors.append(0.1)
            
            # Calculate engagement potential
            engagement_potential = min(1.0, sum(engagement_factors))
            
            return engagement_potential
            
        except Exception as e:
            logger.warning(f"Engagement potential analysis failed: {e}")
            return 0.5
    
    async def _analyze_viral_potential(self, content: str) -> float:
        """Analyze viral potential"""
        try:
            viral_factors = []
            
            content_lower = content.lower()
            
            # Controversial or trending topics
            trending_words = ['trending', 'viral', 'breaking', 'shocking', 'controversial']
            trending_count = sum(content_lower.count(word) for word in trending_words)
            if trending_count > 0:
                viral_factors.append(0.3)
            
            # Emotional triggers
            emotional_triggers = ['love', 'hate', 'anger', 'joy', 'surprise', 'fear']
            emotional_count = sum(content_lower.count(word) for word in emotional_triggers)
            if emotional_count > 0:
                viral_factors.append(0.2)
            
            # Shareable content indicators
            shareable_words = ['share', 'tell friends', 'spread', 'recommend']
            shareable_count = sum(content_lower.count(word) for word in shareable_words)
            if shareable_count > 0:
                viral_factors.append(0.2)
            
            # List or numbered content
            list_indicators = ['top', 'best', 'worst', 'tips', 'secrets', 'ways']
            list_count = sum(content_lower.count(word) for word in list_indicators)
            if list_count > 0:
                viral_factors.append(0.2)
            
            # Personal stories or experiences
            personal_words = ['i', 'my', 'me', 'personal', 'experience', 'story']
            personal_count = sum(content_lower.count(word) for word in personal_words)
            if personal_count > 0:
                viral_factors.append(0.1)
            
            # Calculate viral potential
            viral_potential = min(1.0, sum(viral_factors))
            
            return viral_potential
            
        except Exception as e:
            logger.warning(f"Viral potential analysis failed: {e}")
            return 0.3
    
    async def _analyze_seo_potential(self, content: str) -> float:
        """Analyze SEO potential"""
        try:
            seo_factors = []
            
            # Content length
            word_count = len(content.split())
            if 300 <= word_count <= 2000:
                seo_factors.append(0.3)
            elif word_count > 2000:
                seo_factors.append(0.2)
            else:
                seo_factors.append(0.1)
            
            # Keyword density (simple heuristic)
            words = content.lower().split()
            word_freq = Counter(words)
            keywords = {word: count for word, count in word_freq.items() 
                       if len(word) > 3 and count > 1}
            
            if len(keywords) >= 3:
                seo_factors.append(0.3)
            elif len(keywords) >= 1:
                seo_factors.append(0.2)
            else:
                seo_factors.append(0.1)
            
            # Readability
            sentences = re.split(r'[.!?]+', content)
            if sentences and words:
                avg_sentence_length = len(words) / len(sentences)
                if avg_sentence_length <= 20:
                    seo_factors.append(0.2)
                else:
                    seo_factors.append(0.1)
            
            # Structure indicators
            structure_words = ['introduction', 'conclusion', 'summary', 'overview']
            structure_count = sum(content.lower().count(word) for word in structure_words)
            if structure_count > 0:
                seo_factors.append(0.2)
            
            # Calculate SEO potential
            seo_potential = min(1.0, sum(seo_factors))
            
            return seo_potential
            
        except Exception as e:
            logger.warning(f"SEO potential analysis failed: {e}")
            return 0.5
    
    async def _analyze_conversion_potential(self, content: str) -> float:
        """Analyze conversion potential"""
        try:
            conversion_factors = []
            
            content_lower = content.lower()
            
            # Call-to-action elements
            cta_words = ['buy', 'purchase', 'order', 'get', 'download', 'subscribe', 'sign up']
            cta_count = sum(content_lower.count(word) for word in cta_words)
            if cta_count > 0:
                conversion_factors.append(0.4)
            
            # Benefit-focused language
            benefit_words = ['benefit', 'advantage', 'solution', 'improve', 'save', 'gain']
            benefit_count = sum(content_lower.count(word) for word in benefit_words)
            if benefit_count > 0:
                conversion_factors.append(0.3)
            
            # Urgency indicators
            urgency_words = ['now', 'today', 'limited', 'exclusive', 'urgent', 'deadline']
            urgency_count = sum(content_lower.count(word) for word in urgency_words)
            if urgency_count > 0:
                conversion_factors.append(0.2)
            
            # Social proof indicators
            social_proof_words = ['testimonial', 'review', 'customer', 'user', 'recommend']
            social_proof_count = sum(content_lower.count(word) for word in social_proof_words)
            if social_proof_count > 0:
                conversion_factors.append(0.1)
            
            # Calculate conversion potential
            conversion_potential = min(1.0, sum(conversion_factors))
            
            return conversion_potential
            
        except Exception as e:
            logger.warning(f"Conversion potential analysis failed: {e}")
            return 0.4
    
    async def _analyze_brand_alignment(self, content: str, context: Dict[str, Any]) -> float:
        """Analyze brand alignment"""
        try:
            # This would typically compare against brand guidelines
            # For now, return a score based on content quality and consistency
            
            # Check for brand consistency indicators
            brand_factors = []
            
            # Professional tone
            professional_words = ['professional', 'expert', 'quality', 'reliable', 'trusted']
            professional_count = sum(content.lower().count(word) for word in professional_words)
            if professional_count > 0:
                brand_factors.append(0.3)
            
            # Consistent terminology
            # This would typically check against brand vocabulary
            brand_factors.append(0.2)
            
            # Appropriate content length
            word_count = len(content.split())
            if 200 <= word_count <= 1500:
                brand_factors.append(0.3)
            else:
                brand_factors.append(0.1)
            
            # Quality indicators
            quality_words = ['excellent', 'outstanding', 'premium', 'superior', 'exceptional']
            quality_count = sum(content.lower().count(word) for word in quality_words)
            if quality_count > 0:
                brand_factors.append(0.2)
            
            # Calculate brand alignment
            brand_alignment = min(1.0, sum(brand_factors))
            
            return brand_alignment
            
        except Exception as e:
            logger.warning(f"Brand alignment analysis failed: {e}")
            return 0.6
    
    async def _identify_content_gaps(self, content: str, context: Dict[str, Any]) -> List[str]:
        """Identify content gaps"""
        try:
            gaps = []
            
            # Check for missing elements
            content_lower = content.lower()
            
            # Missing introduction
            intro_words = ['introduction', 'overview', 'welcome', 'hello']
            if not any(word in content_lower for word in intro_words):
                gaps.append("Missing clear introduction or overview")
            
            # Missing conclusion
            conclusion_words = ['conclusion', 'summary', 'in conclusion', 'to summarize']
            if not any(word in content_lower for word in conclusion_words):
                gaps.append("Missing conclusion or summary")
            
            # Missing call-to-action
            cta_words = ['click', 'learn more', 'contact', 'subscribe', 'follow']
            if not any(word in content_lower for word in cta_words):
                gaps.append("Missing call-to-action")
            
            # Missing examples
            example_words = ['example', 'for instance', 'such as', 'like']
            if not any(word in content_lower for word in example_words):
                gaps.append("Missing examples or illustrations")
            
            # Missing data or statistics
            data_words = ['data', 'statistics', 'research', 'study', 'survey']
            if not any(word in content_lower for word in data_words):
                gaps.append("Missing data or statistics to support claims")
            
            return gaps
            
        except Exception as e:
            logger.warning(f"Content gaps identification failed: {e}")
            return []
    
    async def _identify_improvement_opportunities(self, content: str) -> List[str]:
        """Identify improvement opportunities"""
        try:
            opportunities = []
            
            # Check for improvement areas
            word_count = len(content.split())
            
            # Content length optimization
            if word_count < 300:
                opportunities.append("Expand content to improve SEO and provide more value")
            elif word_count > 2000:
                opportunities.append("Consider breaking content into smaller, more digestible sections")
            
            # Readability improvement
            sentences = re.split(r'[.!?]+', content)
            if sentences:
                avg_sentence_length = len(content.split()) / len(sentences)
                if avg_sentence_length > 25:
                    opportunities.append("Simplify sentence structure for better readability")
            
            # Engagement improvement
            if content.count('?') < 2:
                opportunities.append("Add more questions to increase reader engagement")
            
            # Visual content
            if not any(word in content.lower() for word in ['image', 'chart', 'graph', 'diagram']):
                opportunities.append("Consider adding visual elements to enhance content")
            
            # Personalization
            if not any(word in content.lower() for word in ['you', 'your', 'personal']):
                opportunities.append("Add more personalized language to connect with readers")
            
            return opportunities
            
        except Exception as e:
            logger.warning(f"Improvement opportunities identification failed: {e}")
            return []
    
    async def _identify_competitive_advantages(self, content: str, context: Dict[str, Any]) -> List[str]:
        """Identify competitive advantages"""
        try:
            advantages = []
            
            # Check for unique value propositions
            content_lower = content.lower()
            
            # Unique insights
            if any(word in content_lower for word in ['exclusive', 'unique', 'first', 'only']):
                advantages.append("Contains exclusive or unique insights")
            
            # Expert knowledge
            if any(word in content_lower for word in ['expert', 'professional', 'specialist', 'authority']):
                advantages.append("Demonstrates expert knowledge and authority")
            
            # Data-driven content
            if any(word in content_lower for word in ['data', 'research', 'study', 'analysis']):
                advantages.append("Backed by data and research")
            
            # Practical value
            if any(word in content_lower for word in ['how to', 'guide', 'tutorial', 'steps']):
                advantages.append("Provides practical, actionable value")
            
            # Comprehensive coverage
            if len(content.split()) > 1000:
                advantages.append("Comprehensive coverage of the topic")
            
            return advantages
            
        except Exception as e:
            logger.warning(f"Competitive advantages identification failed: {e}")
            return []
    
    async def _identify_risk_factors(self, content: str) -> List[str]:
        """Identify risk factors"""
        try:
            risks = []
            
            content_lower = content.lower()
            
            # Controversial content
            controversial_words = ['controversial', 'debate', 'dispute', 'conflict']
            if any(word in content_lower for word in controversial_words):
                risks.append("Contains potentially controversial content")
            
            # Legal issues
            legal_words = ['lawsuit', 'legal', 'court', 'illegal']
            if any(word in content_lower for word in legal_words):
                risks.append("May involve legal considerations")
            
            # Negative sentiment
            negative_words = ['hate', 'terrible', 'awful', 'disaster', 'failure']
            negative_count = sum(content_lower.count(word) for word in negative_words)
            if negative_count > 3:
                risks.append("High negative sentiment may impact brand perception")
            
            # Unverified claims
            if any(word in content_lower for word in ['guarantee', 'promise', 'certain']):
                risks.append("Contains unverified claims that may need substantiation")
            
            # Short content
            if len(content.split()) < 200:
                risks.append("Content may be too short to provide substantial value")
            
            return risks
            
        except Exception as e:
            logger.warning(f"Risk factors identification failed: {e}")
            return []
    
    async def _calculate_intelligence_score(
        self,
        engagement_potential: float,
        viral_potential: float,
        seo_potential: float,
        conversion_potential: float,
        brand_alignment: float
    ) -> float:
        """Calculate overall intelligence score"""
        try:
            # Weighted average of all factors
            intelligence_score = (
                engagement_potential * 0.25 +
                viral_potential * 0.15 +
                seo_potential * 0.25 +
                conversion_potential * 0.20 +
                brand_alignment * 0.15
            )
            
            return min(1.0, intelligence_score)
            
        except Exception as e:
            logger.warning(f"Intelligence score calculation failed: {e}")
            return 0.5
    
    async def _generate_recommendations(
        self,
        content_type: str,
        complexity_level: str,
        target_audience: str,
        engagement_potential: float,
        viral_potential: float,
        seo_potential: float,
        conversion_potential: float,
        brand_alignment: float,
        content_gaps: List[str],
        improvement_opportunities: List[str],
        competitive_advantages: List[str],
        risk_factors: List[str]
    ) -> List[str]:
        """Generate actionable recommendations"""
        try:
            recommendations = []
            
            # Engagement recommendations
            if engagement_potential < 0.5:
                recommendations.append("Add more interactive elements and questions to increase engagement")
            
            # Viral potential recommendations
            if viral_potential < 0.4:
                recommendations.append("Include more shareable content and emotional triggers")
            
            # SEO recommendations
            if seo_potential < 0.6:
                recommendations.append("Optimize content for search engines with better keywords and structure")
            
            # Conversion recommendations
            if conversion_potential < 0.5:
                recommendations.append("Add clear call-to-action elements and benefit-focused language")
            
            # Brand alignment recommendations
            if brand_alignment < 0.7:
                recommendations.append("Ensure content aligns with brand voice and guidelines")
            
            # Content gaps recommendations
            for gap in content_gaps[:3]:  # Top 3 gaps
                recommendations.append(f"Address content gap: {gap}")
            
            # Improvement opportunities
            for opportunity in improvement_opportunities[:3]:  # Top 3 opportunities
                recommendations.append(f"Improvement opportunity: {opportunity}")
            
            # Risk mitigation
            for risk in risk_factors[:2]:  # Top 2 risks
                recommendations.append(f"Risk mitigation: {risk}")
            
            return recommendations
            
        except Exception as e:
            logger.warning(f"Recommendations generation failed: {e}")
            return []
    
    async def analyze_content_trends(
        self, 
        content_list: List[str], 
        time_period: str = "30d"
    ) -> List[ContentTrend]:
        """Analyze content trends across multiple pieces of content"""
        
        try:
            trends = []
            
            # Analyze trending topics
            all_words = []
            for content in content_list:
                words = content.lower().split()
                all_words.extend(words)
            
            # Get most common words
            word_freq = Counter(all_words)
            common_words = word_freq.most_common(20)
            
            # Create trend for most common words
            for word, count in common_words[:5]:
                if len(word) > 3:  # Filter out short words
                    trend = ContentTrend(
                        trend_type="keyword",
                        trend_score=count / len(content_list),
                        trend_direction="rising",
                        trend_confidence=0.8,
                        related_keywords=[word],
                        trend_impact="medium",
                        trend_timeline=time_period,
                        trend_source="content_analysis"
                    )
                    trends.append(trend)
            
            return trends
            
        except Exception as e:
            logger.error(f"Error in content trends analysis: {e}")
            return []
    
    async def generate_content_insights(
        self, 
        content: str, 
        content_id: str = ""
    ) -> List[ContentInsight]:
        """Generate detailed content insights"""
        
        try:
            insights = []
            
            # Analyze content intelligence
            intelligence = await self.analyze_content_intelligence(content, content_id)
            
            # Create insights based on analysis
            if intelligence.engagement_potential > 0.7:
                insights.append(ContentInsight(
                    insight_type="engagement",
                    insight_score=intelligence.engagement_potential,
                    insight_description="Content has high engagement potential",
                    insight_evidence=["Interactive elements present", "Emotional triggers identified"],
                    insight_impact="high",
                    insight_confidence=0.8,
                    insight_recommendations=["Leverage engagement elements", "Monitor engagement metrics"]
                ))
            
            if intelligence.viral_potential > 0.6:
                insights.append(ContentInsight(
                    insight_type="viral",
                    insight_score=intelligence.viral_potential,
                    insight_description="Content has viral potential",
                    insight_evidence=["Shareable elements present", "Emotional triggers identified"],
                    insight_impact="high",
                    insight_confidence=0.7,
                    insight_recommendations=["Optimize for sharing", "Monitor viral metrics"]
                ))
            
            if intelligence.seo_potential > 0.7:
                insights.append(ContentInsight(
                    insight_type="seo",
                    insight_score=intelligence.seo_potential,
                    insight_description="Content is well-optimized for SEO",
                    insight_evidence=["Good keyword density", "Proper structure"],
                    insight_impact="medium",
                    insight_confidence=0.8,
                    insight_recommendations=["Monitor search rankings", "Track organic traffic"]
                ))
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating content insights: {e}")
            return []
    
    async def generate_content_strategy(
        self, 
        content: str, 
        goals: List[str] = None
    ) -> List[ContentStrategy]:
        """Generate content strategy recommendations"""
        
        if goals is None:
            goals = ["engagement", "seo", "conversion"]
        
        try:
            strategies = []
            
            # Analyze content intelligence
            intelligence = await self.analyze_content_intelligence(content)
            
            # Generate strategies based on goals
            for goal in goals:
                if goal == "engagement" and intelligence.engagement_potential < 0.6:
                    strategies.append(ContentStrategy(
                        strategy_type="engagement",
                        strategy_priority=1,
                        strategy_description="Improve content engagement through interactive elements",
                        strategy_goals=["Increase time on page", "Improve social shares"],
                        strategy_tactics=["Add questions", "Include polls", "Create interactive content"],
                        strategy_metrics=["Time on page", "Social shares", "Comments"],
                        strategy_timeline="2-4 weeks",
                        strategy_resources=["Content team", "Design team"],
                        expected_impact=0.3
                    ))
                
                elif goal == "seo" and intelligence.seo_potential < 0.6:
                    strategies.append(ContentStrategy(
                        strategy_type="seo",
                        strategy_priority=2,
                        strategy_description="Optimize content for search engines",
                        strategy_goals=["Improve search rankings", "Increase organic traffic"],
                        strategy_tactics=["Keyword optimization", "Content structure", "Internal linking"],
                        strategy_metrics=["Search rankings", "Organic traffic", "Click-through rate"],
                        strategy_timeline="4-8 weeks",
                        strategy_resources=["SEO team", "Content team"],
                        expected_impact=0.4
                    ))
                
                elif goal == "conversion" and intelligence.conversion_potential < 0.5:
                    strategies.append(ContentStrategy(
                        strategy_type="conversion",
                        strategy_priority=3,
                        strategy_description="Optimize content for conversions",
                        strategy_goals=["Increase conversion rate", "Improve lead generation"],
                        strategy_tactics=["Clear CTAs", "Benefit-focused language", "Social proof"],
                        strategy_metrics=["Conversion rate", "Lead generation", "Revenue"],
                        strategy_timeline="2-6 weeks",
                        strategy_resources=["Marketing team", "Design team"],
                        expected_impact=0.25
                    ))
            
            return strategies
            
        except Exception as e:
            logger.error(f"Error generating content strategy: {e}")
            return []
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of content intelligence engine"""
        return {
            "status": "healthy" if self.models_loaded else "unhealthy",
            "models_loaded": self.models_loaded,
            "device": self.device,
            "available_models": {
                "trend_analyzer": self.trend_analyzer is not None,
                "insight_generator": self.insight_generator is not None,
                "strategy_planner": self.strategy_planner is not None,
                "competitive_analyzer": self.competitive_analyzer is not None,
                "audience_analyzer": self.audience_analyzer is not None
            },
            "timestamp": datetime.now().isoformat()
        }


# Global content intelligence engine instance
content_intelligence_engine = ContentIntelligenceEngine()


async def initialize_content_intelligence_engine() -> None:
    """Initialize the global content intelligence engine"""
    await content_intelligence_engine.initialize()


async def analyze_content_intelligence(
    content: str, 
    content_id: str = "", 
    context: Dict[str, Any] = None
) -> ContentIntelligence:
    """Analyze content intelligence"""
    return await content_intelligence_engine.analyze_content_intelligence(content, content_id, context)


async def analyze_content_trends(
    content_list: List[str], 
    time_period: str = "30d"
) -> List[ContentTrend]:
    """Analyze content trends"""
    return await content_intelligence_engine.analyze_content_trends(content_list, time_period)


async def generate_content_insights(
    content: str, 
    content_id: str = ""
) -> List[ContentInsight]:
    """Generate content insights"""
    return await content_intelligence_engine.generate_content_insights(content, content_id)


async def generate_content_strategy(
    content: str, 
    goals: List[str] = None
) -> List[ContentStrategy]:
    """Generate content strategy"""
    return await content_intelligence_engine.generate_content_strategy(content, goals)


async def get_content_intelligence_health() -> Dict[str, Any]:
    """Get content intelligence engine health status"""
    return await content_intelligence_engine.health_check()




