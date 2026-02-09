from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema import HumanMessage
from ...shared.logging import get_logger
            import json
            import json
            import json
            import re
from typing import Any, List, Dict, Optional
import logging
"""
Engagement Analyzer for LinkedIn Posts
=====================================

AI-powered engagement analysis and prediction using LangChain.
"""



logger = get_logger(__name__)


class EngagementAnalyzer:
    """
    Engagement analyzer for LinkedIn posts using LangChain.
    
    Provides comprehensive engagement analysis, prediction,
    and optimization recommendations.
    """
    
    def __init__(self, llm: ChatOpenAI):
        """Initialize the engagement analyzer."""
        self.llm = llm
        self._setup_analysis_chains()
    
    def _setup_analysis_chains(self) -> Any:
        """Setup analysis chains."""
        
        # Engagement prediction
        self.prediction_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                input_variables=["content", "target_audience", "industry"],
                template="""Analyze this LinkedIn post and predict its engagement potential:

Content: {content}
Target Audience: {target_audience}
Industry: {industry}

Evaluate the following factors:
1. Content quality and relevance (0-100)
2. Emotional appeal and storytelling (0-100)
3. Call-to-action effectiveness (0-100)
4. Readability and formatting (0-100)
5. Trending topic relevance (0-100)
6. Industry-specific appeal (0-100)
7. Shareability potential (0-100)
8. Comment-worthy elements (0-100)

Provide:
- Overall engagement score (0-100)
- Detailed breakdown of each factor
- 3 specific recommendations for improvement
- Predicted likes, comments, and shares range

Format as JSON with keys: overall_score, factors, recommendations, predictions"""
            )
        )
        
        # Sentiment analysis
        self.sentiment_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                input_variables=["content"],
                template="""Analyze the sentiment and emotional tone of this LinkedIn post:

{content}

Provide:
- Overall sentiment (positive, negative, neutral)
- Sentiment score (-1 to 1, where -1 is very negative, 1 is very positive)
- Emotional tone (professional, casual, inspirational, authoritative, etc.)
- Emotional triggers identified
- Sentiment consistency throughout the post

Format as JSON with keys: sentiment, score, tone, triggers, consistency"""
            )
        )
        
        # Audience analysis
        self.audience_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                input_variables=["content", "target_audience"],
                template="""Analyze how well this LinkedIn post resonates with its target audience:

Content: {content}
Target Audience: {target_audience}

Evaluate:
- Audience relevance (0-100)
- Message alignment with audience needs
- Language appropriateness
- Value proposition clarity
- Pain points addressed
- Aspirational elements

Provide:
- Audience resonance score (0-100)
- Specific audience segments that would engage most
- Potential audience expansion opportunities
- Recommendations for better audience targeting

Format as JSON with keys: resonance_score, segments, opportunities, recommendations"""
            )
        )
    
    async def predict_engagement(
        self,
        content: str,
        target_audience: str,
        industry: str
    ) -> Dict[str, any]:
        """Predict engagement metrics for a LinkedIn post."""
        try:
            result = await self.prediction_chain.arun(
                content=content,
                target_audience=target_audience,
                industry=industry
            )
            
            # Parse JSON result
            try:
                prediction_data = json.loads(result)
                return prediction_data
            except json.JSONDecodeError:
                # Fallback parsing
                return self._parse_prediction_fallback(result)
                
        except Exception as e:
            logger.error(f"Error predicting engagement: {e}")
            return self._get_default_prediction()
    
    async def analyze_sentiment(self, content: str) -> Dict[str, any]:
        """Analyze sentiment and emotional tone of the content."""
        try:
            result = await self.sentiment_chain.arun(content=content)
            
            try:
                sentiment_data = json.loads(result)
                return sentiment_data
            except json.JSONDecodeError:
                return self._parse_sentiment_fallback(result)
                
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return self._get_default_sentiment()
    
    async def analyze_audience_resonance(
        self,
        content: str,
        target_audience: str
    ) -> Dict[str, any]:
        """Analyze how well content resonates with target audience."""
        try:
            result = await self.audience_chain.arun(
                content=content,
                target_audience=target_audience
            )
            
            try:
                audience_data = json.loads(result)
                return audience_data
            except json.JSONDecodeError:
                return self._parse_audience_fallback(result)
                
        except Exception as e:
            logger.error(f"Error analyzing audience resonance: {e}")
            return self._get_default_audience_analysis()
    
    async def comprehensive_analysis(
        self,
        content: str,
        target_audience: str,
        industry: str
    ) -> Dict[str, any]:
        """Perform comprehensive engagement analysis."""
        try:
            # Run all analyses in parallel
            prediction_task = self.predict_engagement(content, target_audience, industry)
            sentiment_task = self.analyze_sentiment(content)
            audience_task = self.analyze_audience_resonance(content, target_audience)
            
            # Wait for all analyses
            prediction, sentiment, audience = await asyncio.gather(
                prediction_task, sentiment_task, audience_task
            )
            
            # Calculate composite score
            composite_score = self._calculate_composite_score(prediction, sentiment, audience)
            
            # Generate optimization recommendations
            recommendations = await self._generate_optimization_recommendations(
                prediction, sentiment, audience
            )
            
            return {
                "prediction": prediction,
                "sentiment": sentiment,
                "audience": audience,
                "composite_score": composite_score,
                "recommendations": recommendations,
                "analysis_timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in comprehensive analysis: {e}")
            return self._get_default_comprehensive_analysis()
    
    def _calculate_composite_score(
        self,
        prediction: Dict,
        sentiment: Dict,
        audience: Dict
    ) -> float:
        """Calculate composite engagement score."""
        try:
            # Weighted average of different scores
            prediction_score = prediction.get("overall_score", 50)
            sentiment_score = (sentiment.get("score", 0) + 1) * 50  # Convert -1,1 to 0,100
            audience_score = audience.get("resonance_score", 50)
            
            # Weights: prediction 40%, sentiment 30%, audience 30%
            composite = (
                prediction_score * 0.4 +
                sentiment_score * 0.3 +
                audience_score * 0.3
            )
            
            return round(composite, 2)
            
        except Exception:
            return 50.0
    
    async def _generate_optimization_recommendations(
        self,
        prediction: Dict,
        sentiment: Dict,
        audience: Dict
    ) -> List[str]:
        """Generate optimization recommendations based on analysis."""
        try:
            recommendations = []
            
            # Prediction-based recommendations
            if prediction.get("overall_score", 50) < 70:
                recommendations.append("Consider improving content quality and relevance")
            
            factors = prediction.get("factors", {})
            if factors.get("call_to_action", 50) < 60:
                recommendations.append("Strengthen the call-to-action to drive engagement")
            
            if factors.get("readability", 50) < 60:
                recommendations.append("Improve formatting and readability with bullet points")
            
            # Sentiment-based recommendations
            sentiment_score = sentiment.get("score", 0)
            if sentiment_score < 0.2:
                recommendations.append("Consider making the tone more positive and inspiring")
            
            # Audience-based recommendations
            if audience.get("resonance_score", 50) < 70:
                recommendations.append("Better align content with target audience needs")
            
            # Add general recommendations
            recommendations.extend([
                "Include relevant hashtags for better discoverability",
                "Add questions to encourage comments and discussion",
                "Use storytelling elements to increase emotional connection"
            ])
            
            return recommendations[:5]  # Limit to 5 recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return ["Focus on creating valuable, authentic content"]
    
    def analyze_historical_performance(
        self,
        posts_data: List[Dict]
    ) -> Dict[str, any]:
        """Analyze historical post performance for insights."""
        try:
            if not posts_data:
                return {}
            
            # Calculate average metrics
            total_posts = len(posts_data)
            avg_likes = sum(p.get("likes_count", 0) for p in posts_data) / total_posts
            avg_comments = sum(p.get("comments_count", 0) for p in posts_data) / total_posts
            avg_shares = sum(p.get("shares_count", 0) for p in posts_data) / total_posts
            avg_views = sum(p.get("views_count", 0) for p in posts_data) / total_posts
            
            # Find best performing posts
            best_posts = sorted(
                posts_data,
                key=lambda x: x.get("likes_count", 0) + x.get("comments_count", 0) * 2 + x.get("shares_count", 0) * 3,
                reverse=True
            )[:3]
            
            # Analyze patterns
            patterns = self._analyze_performance_patterns(posts_data)
            
            return {
                "total_posts": total_posts,
                "average_metrics": {
                    "likes": round(avg_likes, 2),
                    "comments": round(avg_comments, 2),
                    "shares": round(avg_shares, 2),
                    "views": round(avg_views, 2)
                },
                "best_performing_posts": [
                    {
                        "id": p.get("id"),
                        "title": p.get("title"),
                        "engagement_score": p.get("likes_count", 0) + p.get("comments_count", 0) * 2 + p.get("shares_count", 0) * 3
                    }
                    for p in best_posts
                ],
                "performance_patterns": patterns
            }
            
        except Exception as e:
            logger.error(f"Error analyzing historical performance: {e}")
            return {}
    
    def _analyze_performance_patterns(self, posts_data: List[Dict]) -> Dict[str, any]:
        """Analyze patterns in post performance."""
        try:
            patterns = {}
            
            # Analyze by post type
            post_types = {}
            for post in posts_data:
                post_type = post.get("post_type", "unknown")
                if post_type not in post_types:
                    post_types[post_type] = []
                post_types[post_type].append(post)
            
            for post_type, posts in post_types.items():
                avg_engagement = sum(
                    p.get("likes_count", 0) + p.get("comments_count", 0) + p.get("shares_count", 0)
                    for p in posts
                ) / len(posts)
                patterns[f"avg_engagement_{post_type}"] = round(avg_engagement, 2)
            
            # Analyze by tone
            tones = {}
            for post in posts_data:
                tone = post.get("tone", "unknown")
                if tone not in tones:
                    tones[tone] = []
                tones[tone].append(post)
            
            for tone, posts in tones.items():
                avg_engagement = sum(
                    p.get("likes_count", 0) + p.get("comments_count", 0) + p.get("shares_count", 0)
                    for p in posts
                ) / len(posts)
                patterns[f"avg_engagement_{tone}"] = round(avg_engagement, 2)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error analyzing performance patterns: {e}")
            return {}
    
    def _parse_prediction_fallback(self, result: str) -> Dict[str, any]:
        """Parse prediction result when JSON parsing fails."""
        try:
            # Extract score from text
            score_match = re.search(r'(\d+)', result)
            score = int(score_match.group(1)) if score_match else 50
            
            return {
                "overall_score": score,
                "factors": {},
                "recommendations": ["Focus on creating valuable content"],
                "predictions": {"likes": "10-50", "comments": "2-10", "shares": "1-5"}
            }
        except Exception:
            return self._get_default_prediction()
    
    def _parse_sentiment_fallback(self, result: str) -> Dict[str, any]:
        """Parse sentiment result when JSON parsing fails."""
        try:
            return {
                "sentiment": "neutral",
                "score": 0.0,
                "tone": "professional",
                "triggers": [],
                "consistency": "consistent"
            }
        except Exception:
            return self._get_default_sentiment()
    
    def _parse_audience_fallback(self, result: str) -> Dict[str, any]:
        """Parse audience result when JSON parsing fails."""
        try:
            return {
                "resonance_score": 50,
                "segments": ["general professionals"],
                "opportunities": [],
                "recommendations": ["Better understand your target audience"]
            }
        except Exception:
            return self._get_default_audience_analysis()
    
    def _get_default_prediction(self) -> Dict[str, any]:
        """Get default prediction data."""
        return {
            "overall_score": 50,
            "factors": {},
            "recommendations": ["Focus on creating valuable content"],
            "predictions": {"likes": "10-50", "comments": "2-10", "shares": "1-5"}
        }
    
    def _get_default_sentiment(self) -> Dict[str, any]:
        """Get default sentiment data."""
        return {
            "sentiment": "neutral",
            "score": 0.0,
            "tone": "professional",
            "triggers": [],
            "consistency": "consistent"
        }
    
    def _get_default_audience_analysis(self) -> Dict[str, any]:
        """Get default audience analysis data."""
        return {
            "resonance_score": 50,
            "segments": ["general professionals"],
            "opportunities": [],
            "recommendations": ["Better understand your target audience"]
        }
    
    def _get_default_comprehensive_analysis(self) -> Dict[str, any]:
        """Get default comprehensive analysis data."""
        return {
            "prediction": self._get_default_prediction(),
            "sentiment": self._get_default_sentiment(),
            "audience": self._get_default_audience_analysis(),
            "composite_score": 50.0,
            "recommendations": ["Focus on creating valuable, authentic content"],
            "analysis_timestamp": datetime.utcnow().isoformat()
        } 