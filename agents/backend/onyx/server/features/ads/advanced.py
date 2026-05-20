from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from fastapi import HTTPException
import httpx
import json
import logging
from datetime import datetime
from onyx.core.config import settings
from onyx.core.functions import format_response, handle_error
from typing import Any, List, Dict, Optional
import asyncio
"""
Advanced Onyx functionalities for ads module.
"""

logger = logging.getLogger(__name__)

class AITrainingData(BaseModel):
    """AI training data model."""
    content_type: str
    content: str
    metadata: Dict[str, Any]
    performance_metrics: Optional[Dict[str, float]] = None
    feedback: Optional[List[Dict[str, Any]]] = None
    training_status: str = "pending"

class ContentOptimization(BaseModel):
    """Content optimization model."""
    content_id: str
    original_content: str
    optimized_content: str
    optimization_type: str
    metrics: Dict[str, float]
    a_b_test_results: Optional[Dict[str, Any]] = None
    recommendations: List[str]

class AudienceInsights(BaseModel):
    """Audience insights model."""
    segment_id: str
    demographics: Dict[str, Any]
    behavior_patterns: List[Dict[str, Any]]
    interests: List[str]
    engagement_metrics: Dict[str, float]
    conversion_funnel: Dict[str, float]
    recommendations: List[str]

class BrandVoiceAnalysis(BaseModel):
    """Brand voice analysis model."""
    content_samples: List[str]
    tone_analysis: Dict[str, float]
    consistency_score: float
    brand_alignment: float
    recommendations: List[str]
    improvement_areas: List[str]

class ContentPerformance(BaseModel):
    """Content performance model."""
    content_id: str
    metrics: Dict[str, float]
    audience_segments: List[Dict[str, Any]]
    channel_performance: Dict[str, Dict[str, float]]
    trends: List[Dict[str, Any]]
    recommendations: List[str]

class AdvancedAdsService:
    """Service for advanced Onyx functionalities."""
    
    def __init__(self, httpx_client: httpx.AsyncClient):
        
    """__init__ function."""
self.client = httpx_client
        self.base_url = settings.ONYX_API_URL
        self.headers = {
            "Authorization": f"Bearer {settings.ONYX_API_KEY}",
            "Content-Type": "application/json"
        }
    
    async def train_ai_model(self, training_data: List[AITrainingData]) -> Dict[str, Any]:
        """Train AI model with provided data."""
        try:
            response = await self.client.post(
                f"{self.base_url}/ai/train",
                json=[data.dict() for data in training_data],
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error training AI model: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def optimize_content(self, content: str, optimization_type: str) -> ContentOptimization:
        """Optimize content based on type."""
        try:
            response = await self.client.post(
                f"{self.base_url}/content/optimize",
                json={
                    "content": content,
                    "optimization_type": optimization_type
                },
                headers=self.headers
            )
            response.raise_for_status()
            return ContentOptimization(**response.json())
        except Exception as e:
            logger.error(f"Error optimizing content: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def analyze_audience(self, segment_id: str) -> AudienceInsights:
        """Analyze audience segment."""
        try:
            response = await self.client.get(
                f"{self.base_url}/audience/analyze/{segment_id}",
                headers=self.headers
            )
            response.raise_for_status()
            return AudienceInsights(**response.json())
        except Exception as e:
            logger.error(f"Error analyzing audience: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def analyze_brand_voice(self, content_samples: List[str]) -> BrandVoiceAnalysis:
        """Analyze brand voice from content samples."""
        try:
            response = await self.client.post(
                f"{self.base_url}/brand/analyze",
                json={"content_samples": content_samples},
                headers=self.headers
            )
            response.raise_for_status()
            return BrandVoiceAnalysis(**response.json())
        except Exception as e:
            logger.error(f"Error analyzing brand voice: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def track_content_performance(self, content_id: str) -> ContentPerformance:
        """Track content performance metrics."""
        try:
            response = await self.client.get(
                f"{self.base_url}/content/performance/{content_id}",
                headers=self.headers
            )
            response.raise_for_status()
            return ContentPerformance(**response.json())
        except Exception as e:
            logger.error(f"Error tracking content performance: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def generate_ai_recommendations(self, content: str, context: Dict[str, Any]) -> List[str]:
        """Generate AI-powered recommendations for content."""
        try:
            response = await self.client.post(
                f"{self.base_url}/ai/recommend",
                json={
                    "content": content,
                    "context": context
                },
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()["recommendations"]
        except Exception as e:
            logger.error(f"Error generating AI recommendations: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def analyze_content_impact(self, content_id: str) -> Dict[str, Any]:
        """Analyze content impact across channels."""
        try:
            response = await self.client.get(
                f"{self.base_url}/content/impact/{content_id}",
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error analyzing content impact: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def optimize_audience_targeting(self, segment_id: str) -> Dict[str, Any]:
        """Optimize audience targeting for a segment."""
        try:
            response = await self.client.post(
                f"{self.base_url}/audience/optimize/{segment_id}",
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error optimizing audience targeting: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def generate_content_variations(self, content: str, variations: int = 3) -> List[str]:
        """Generate variations of content for A/B testing."""
        try:
            response = await self.client.post(
                f"{self.base_url}/content/variations",
                json={
                    "content": content,
                    "variations": variations
                },
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()["variations"]
        except Exception as e:
            logger.error(f"Error generating content variations: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def analyze_competitor_content(self, competitor_urls: List[str]) -> Dict[str, Any]:
        """Analyze competitor content and strategies."""
        try:
            response = await self.client.post(
                f"{self.base_url}/competitor/analyze",
                json={"urls": competitor_urls},
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error analyzing competitor content: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e)) 