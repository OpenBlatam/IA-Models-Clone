"""
B-roll Integration System

AI-powered B-roll suggestion and insertion system for enhancing video content.
Automatically suggests and inserts relevant stock footage or generates visuals for abstract concepts.
"""

from __future__ import annotations
from typing import List, Dict, Optional, Any, Tuple, Union
import asyncio
import numpy as np
import cv2
import torch
import torch.nn as nn
from dataclasses import dataclass
from enum import Enum
import structlog
from pathlib import Path
import json
import time
from concurrent.futures import ThreadPoolExecutor
import requests
import aiohttp
from urllib.parse import urlencode

from ..models.video_models import VideoClipRequest, VideoClipResponse
from ..error_handling import ErrorHandler, ProcessingError, ValidationError

logger = structlog.get_logger("broll_integration_system")
error_handler = ErrorHandler()

class BrollType(Enum):
    """Types of B-roll content."""
    STOCK_FOOTAGE = "stock_footage"
    AI_GENERATED = "ai_generated"
    ANIMATION = "animation"
    GRAPHIC = "graphic"
    TEXT_OVERLAY = "text_overlay"

class ContentCategory(Enum):
    """Categories of content for B-roll matching."""
    BUSINESS = "business"
    TECHNOLOGY = "technology"
    LIFESTYLE = "lifestyle"
    EDUCATION = "education"
    ENTERTAINMENT = "entertainment"
    NEWS = "news"
    SPORTS = "sports"
    TRAVEL = "travel"
    FOOD = "food"
    HEALTH = "health"

@dataclass
class BrollSuggestion:
    """A B-roll suggestion for content enhancement."""
    content_id: str
    title: str
    description: str
    broll_type: BrollType
    category: ContentCategory
    duration: float
    start_time: float
    end_time: float
    confidence: float
    source_url: Optional[str] = None
    thumbnail_url: Optional[str] = None
    metadata: Dict[str, Any] = None

@dataclass
class BrollOpportunity:
    """An opportunity for B-roll insertion in content."""
    start_time: float
    end_time: float
    duration: float
    content_text: str
    keywords: List[str]
    sentiment: str
    category: ContentCategory
    confidence: float
    context: Dict[str, Any]

@dataclass
class BrollConfig:
    """Configuration for B-roll integration."""
    max_broll_duration: float = 5.0
    min_broll_duration: float = 1.0
    max_suggestions_per_opportunity: int = 3
    confidence_threshold: float = 0.7
    stock_footage_api_key: Optional[str] = None
    ai_generation_api_key: Optional[str] = None
    enable_ai_generation: bool = True
    enable_stock_footage: bool = True

class ContentAnalyzer:
    """Analyzes content to identify B-roll opportunities."""
    
    def __init__(self, config: BrollConfig = None):
        self.config = config or BrollConfig()
        self.nlp_model = None
        self.sentiment_analyzer = None
        self._load_models()
    
    def _load_models(self):
        """Load NLP models for content analysis."""
        try:
            # Load NLP model for content analysis
            self.nlp_model = self._load_nlp_model()
            
            # Load sentiment analyzer
            self.sentiment_analyzer = self._load_sentiment_analyzer()
            
            logger.info("Content analysis models loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load content analysis models: {e}")
            raise ProcessingError(f"Content analysis model loading failed: {e}")
    
    def _load_nlp_model(self):
        """Load NLP model for content analysis."""
        # Placeholder - would load actual NLP model
        return SimpleNLPModel()
    
    def _load_sentiment_analyzer(self):
        """Load sentiment analysis model."""
        # Placeholder - would load actual sentiment analyzer
        return SimpleSentimentAnalyzer()
    
    async def analyze_content(self, 
                            content: str, 
                            timestamps: List[float] = None) -> List[BrollOpportunity]:
        """Analyze content to identify B-roll opportunities."""
        try:
            # Split content into segments if timestamps provided
            if timestamps:
                segments = await self._split_content_by_timestamps(content, timestamps)
            else:
                segments = [{"text": content, "start_time": 0.0, "end_time": len(content) * 0.1}]
            
            opportunities = []
            
            for segment in segments:
                # Analyze segment for B-roll opportunities
                segment_opportunities = await self._analyze_segment(segment)
                opportunities.extend(segment_opportunities)
            
            # Filter and rank opportunities
            filtered_opportunities = await self._filter_opportunities(opportunities)
            
            return filtered_opportunities
            
        except Exception as e:
            logger.error(f"Content analysis failed: {e}")
            raise ProcessingError(f"Content analysis failed: {e}")
    
    async def _split_content_by_timestamps(self, 
                                         content: str, 
                                         timestamps: List[float]) -> List[Dict[str, Any]]:
        """Split content into segments based on timestamps."""
        try:
            # Simple word-based splitting (would use more sophisticated text segmentation)
            words = content.split()
            words_per_second = len(words) / max(timestamps) if timestamps else 1.0
            
            segments = []
            for i, timestamp in enumerate(timestamps):
                start_time = timestamp
                end_time = timestamps[i + 1] if i + 1 < len(timestamps) else timestamp + 10.0
                
                # Calculate word range for this segment
                start_word = int(start_time * words_per_second)
                end_word = int(end_time * words_per_second)
                
                segment_words = words[start_word:end_word]
                segment_text = " ".join(segment_words)
                
                segments.append({
                    "text": segment_text,
                    "start_time": start_time,
                    "end_time": end_time
                })
            
            return segments
            
        except Exception as e:
            logger.error(f"Content splitting failed: {e}")
            return [{"text": content, "start_time": 0.0, "end_time": 10.0}]
    
    async def _analyze_segment(self, segment: Dict[str, Any]) -> List[BrollOpportunity]:
        """Analyze a content segment for B-roll opportunities."""
        try:
            text = segment["text"]
            start_time = segment["start_time"]
            end_time = segment["end_time"]
            
            # Extract keywords
            keywords = await self._extract_keywords(text)
            
            # Analyze sentiment
            sentiment = await self._analyze_sentiment(text)
            
            # Determine content category
            category = await self._categorize_content(text, keywords)
            
            # Check if segment is suitable for B-roll
            confidence = await self._calculate_broll_confidence(text, keywords, sentiment)
            
            if confidence >= self.config.confidence_threshold:
                opportunity = BrollOpportunity(
                    start_time=start_time,
                    end_time=end_time,
                    duration=end_time - start_time,
                    content_text=text,
                    keywords=keywords,
                    sentiment=sentiment,
                    category=category,
                    confidence=confidence,
                    context={
                        "word_count": len(text.split()),
                        "has_numbers": any(char.isdigit() for char in text),
                        "has_questions": "?" in text,
                        "has_exclamations": "!" in text
                    }
                )
                
                return [opportunity]
            
            return []
            
        except Exception as e:
            logger.error(f"Segment analysis failed: {e}")
            return []
    
    async def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text."""
        try:
            # Use NLP model to extract keywords
            keywords = self.nlp_model.extract_keywords(text)
            
            # Filter and clean keywords
            cleaned_keywords = []
            for keyword in keywords:
                if len(keyword) > 2 and keyword.isalpha():
                    cleaned_keywords.append(keyword.lower())
            
            return cleaned_keywords[:10]  # Limit to top 10 keywords
            
        except Exception as e:
            logger.error(f"Keyword extraction failed: {e}")
            return []
    
    async def _analyze_sentiment(self, text: str) -> str:
        """Analyze sentiment of text."""
        try:
            sentiment = self.sentiment_analyzer.analyze(text)
            return sentiment
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return "neutral"
    
    async def _categorize_content(self, text: str, keywords: List[str]) -> ContentCategory:
        """Categorize content based on text and keywords."""
        try:
            # Simple keyword-based categorization
            text_lower = text.lower()
            
            # Business keywords
            business_keywords = ["business", "company", "profit", "revenue", "market", "investment"]
            if any(keyword in text_lower for keyword in business_keywords):
                return ContentCategory.BUSINESS
            
            # Technology keywords
            tech_keywords = ["technology", "software", "app", "digital", "ai", "machine learning"]
            if any(keyword in text_lower for keyword in tech_keywords):
                return ContentCategory.TECHNOLOGY
            
            # Education keywords
            education_keywords = ["learn", "education", "study", "course", "tutorial", "teach"]
            if any(keyword in text_lower for keyword in education_keywords):
                return ContentCategory.EDUCATION
            
            # Default to lifestyle
            return ContentCategory.LIFESTYLE
            
        except Exception as e:
            logger.error(f"Content categorization failed: {e}")
            return ContentCategory.LIFESTYLE
    
    async def _calculate_broll_confidence(self, 
                                        text: str, 
                                        keywords: List[str], 
                                        sentiment: str) -> float:
        """Calculate confidence that content is suitable for B-roll."""
        try:
            confidence = 0.0
            
            # Keyword density
            keyword_density = len(keywords) / max(len(text.split()), 1)
            confidence += keyword_density * 0.3
            
            # Sentiment intensity
            sentiment_scores = {"positive": 0.8, "negative": 0.7, "neutral": 0.3}
            confidence += sentiment_scores.get(sentiment, 0.3) * 0.2
            
            # Text length (prefer medium-length segments)
            word_count = len(text.split())
            if 10 <= word_count <= 50:
                confidence += 0.3
            elif 5 <= word_count <= 100:
                confidence += 0.2
            
            # Presence of visual concepts
            visual_keywords = ["see", "look", "show", "display", "image", "picture", "video"]
            visual_count = sum(1 for word in visual_keywords if word in text.lower())
            confidence += min(visual_count * 0.1, 0.2)
            
            return min(confidence, 1.0)
            
        except Exception as e:
            logger.error(f"B-roll confidence calculation failed: {e}")
            return 0.0
    
    async def _filter_opportunities(self, opportunities: List[BrollOpportunity]) -> List[BrollOpportunity]:
        """Filter and rank B-roll opportunities."""
        try:
            # Remove overlapping opportunities
            filtered = []
            used_times = []
            
            # Sort by confidence (highest first)
            sorted_opportunities = sorted(opportunities, key=lambda o: o.confidence, reverse=True)
            
            for opportunity in sorted_opportunities:
                # Check for overlap
                overlap = False
                for used_start, used_end in used_times:
                    if (opportunity.start_time < used_end and 
                        opportunity.end_time > used_start):
                        overlap = True
                        break
                
                if not overlap:
                    filtered.append(opportunity)
                    used_times.append((opportunity.start_time, opportunity.end_time))
            
            return filtered
            
        except Exception as e:
            logger.error(f"Opportunity filtering failed: {e}")
            return opportunities

class BrollSuggester:
    """Suggests relevant B-roll content for opportunities."""
    
    def __init__(self, config: BrollConfig = None):
        self.config = config or BrollConfig()
        self.stock_footage_api = None
        self.ai_generator = None
        self._initialize_apis()
    
    def _initialize_apis(self):
        """Initialize external APIs for B-roll content."""
        try:
            if self.config.enable_stock_footage and self.config.stock_footage_api_key:
                self.stock_footage_api = StockFootageAPI(self.config.stock_footage_api_key)
            
            if self.config.enable_ai_generation and self.config.ai_generation_api_key:
                self.ai_generator = AIVisualGenerator(self.config.ai_generation_api_key)
            
            logger.info("B-roll APIs initialized successfully")
        except Exception as e:
            logger.error(f"B-roll API initialization failed: {e}")
            raise ProcessingError(f"B-roll API initialization failed: {e}")
    
    async def suggest_broll(self, 
                           opportunity: BrollOpportunity) -> List[BrollSuggestion]:
        """Suggest B-roll content for an opportunity."""
        try:
            suggestions = []
            
            # Generate stock footage suggestions
            if self.stock_footage_api:
                stock_suggestions = await self._suggest_stock_footage(opportunity)
                suggestions.extend(stock_suggestions)
            
            # Generate AI-generated suggestions
            if self.ai_generator:
                ai_suggestions = await self._suggest_ai_generated(opportunity)
                suggestions.extend(ai_suggestions)
            
            # Generate graphic suggestions
            graphic_suggestions = await self._suggest_graphics(opportunity)
            suggestions.extend(graphic_suggestions)
            
            # Rank and filter suggestions
            ranked_suggestions = await self._rank_suggestions(suggestions, opportunity)
            
            # Return top suggestions
            return ranked_suggestions[:self.config.max_suggestions_per_opportunity]
            
        except Exception as e:
            logger.error(f"B-roll suggestion failed: {e}")
            raise ProcessingError(f"B-roll suggestion failed: {e}")
    
    async def _suggest_stock_footage(self, opportunity: BrollOpportunity) -> List[BrollSuggestion]:
        """Suggest stock footage for an opportunity."""
        try:
            if not self.stock_footage_api:
                return []
            
            # Search for relevant stock footage
            search_query = " ".join(opportunity.keywords[:5])  # Top 5 keywords
            search_results = await self.stock_footage_api.search(
                query=search_query,
                category=opportunity.category.value,
                duration=opportunity.duration,
                count=self.config.max_suggestions_per_opportunity
            )
            
            suggestions = []
            for result in search_results:
                suggestion = BrollSuggestion(
                    content_id=result["id"],
                    title=result["title"],
                    description=result["description"],
                    broll_type=BrollType.STOCK_FOOTAGE,
                    category=opportunity.category,
                    duration=min(result["duration"], self.config.max_broll_duration),
                    start_time=opportunity.start_time,
                    end_time=opportunity.start_time + min(result["duration"], self.config.max_broll_duration),
                    confidence=result["relevance_score"],
                    source_url=result["url"],
                    thumbnail_url=result["thumbnail"],
                    metadata=result.get("metadata", {})
                )
                suggestions.append(suggestion)
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Stock footage suggestion failed: {e}")
            return []
    
    async def _suggest_ai_generated(self, opportunity: BrollOpportunity) -> List[BrollSuggestion]:
        """Suggest AI-generated visuals for an opportunity."""
        try:
            if not self.ai_generator:
                return []
            
            # Generate AI visuals based on content
            prompt = await self._create_ai_prompt(opportunity)
            generated_content = await self.ai_generator.generate(
                prompt=prompt,
                duration=opportunity.duration,
                style=opportunity.category.value
            )
            
            suggestions = []
            for content in generated_content:
                suggestion = BrollSuggestion(
                    content_id=content["id"],
                    title=content["title"],
                    description=content["description"],
                    broll_type=BrollType.AI_GENERATED,
                    category=opportunity.category,
                    duration=min(content["duration"], self.config.max_broll_duration),
                    start_time=opportunity.start_time,
                    end_time=opportunity.start_time + min(content["duration"], self.config.max_broll_duration),
                    confidence=content["quality_score"],
                    source_url=content["url"],
                    thumbnail_url=content["thumbnail"],
                    metadata=content.get("metadata", {})
                )
                suggestions.append(suggestion)
            
            return suggestions
            
        except Exception as e:
            logger.error(f"AI-generated suggestion failed: {e}")
            return []
    
    async def _suggest_graphics(self, opportunity: BrollOpportunity) -> List[BrollSuggestion]:
        """Suggest graphic overlays for an opportunity."""
        try:
            suggestions = []
            
            # Create text overlay suggestion
            if opportunity.keywords:
                text_overlay = BrollSuggestion(
                    content_id=f"text_{int(time.time())}",
                    title="Text Overlay",
                    description=f"Text overlay with keywords: {', '.join(opportunity.keywords[:3])}",
                    broll_type=BrollType.TEXT_OVERLAY,
                    category=opportunity.category,
                    duration=min(opportunity.duration, 3.0),
                    start_time=opportunity.start_time,
                    end_time=opportunity.start_time + min(opportunity.duration, 3.0),
                    confidence=0.8,
                    metadata={
                        "text": " ".join(opportunity.keywords[:3]),
                        "style": "modern",
                        "animation": "fade_in_out"
                    }
                )
                suggestions.append(text_overlay)
            
            # Create graphic suggestion
            graphic = BrollSuggestion(
                content_id=f"graphic_{int(time.time())}",
                title="Info Graphic",
                description=f"Info graphic for {opportunity.category.value} content",
                broll_type=BrollType.GRAPHIC,
                category=opportunity.category,
                duration=min(opportunity.duration, 4.0),
                start_time=opportunity.start_time,
                end_time=opportunity.start_time + min(opportunity.duration, 4.0),
                confidence=0.7,
                metadata={
                    "type": "info_graphic",
                    "category": opportunity.category.value,
                    "style": "minimal"
                }
            )
            suggestions.append(graphic)
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Graphic suggestion failed: {e}")
            return []
    
    async def _create_ai_prompt(self, opportunity: BrollOpportunity) -> str:
        """Create AI prompt for visual generation."""
        try:
            prompt_parts = []
            
            # Add category context
            category_prompts = {
                ContentCategory.BUSINESS: "professional business environment",
                ContentCategory.TECHNOLOGY: "modern technology and innovation",
                ContentCategory.EDUCATION: "educational and learning environment",
                ContentCategory.LIFESTYLE: "lifestyle and daily activities",
                ContentCategory.ENTERTAINMENT: "entertainment and fun activities"
            }
            
            prompt_parts.append(category_prompts.get(opportunity.category, "general content"))
            
            # Add keywords
            if opportunity.keywords:
                prompt_parts.append("featuring " + ", ".join(opportunity.keywords[:3]))
            
            # Add sentiment
            if opportunity.sentiment == "positive":
                prompt_parts.append("with positive and uplifting mood")
            elif opportunity.sentiment == "negative":
                prompt_parts.append("with serious and thoughtful mood")
            
            # Add duration context
            if opportunity.duration < 3:
                prompt_parts.append("quick and dynamic")
            elif opportunity.duration > 8:
                prompt_parts.append("detailed and comprehensive")
            
            return ", ".join(prompt_parts)
            
        except Exception as e:
            logger.error(f"AI prompt creation failed: {e}")
            return "professional video content"
    
    async def _rank_suggestions(self, 
                               suggestions: List[BrollSuggestion], 
                               opportunity: BrollOpportunity) -> List[BrollSuggestion]:
        """Rank B-roll suggestions by relevance."""
        try:
            def calculate_relevance_score(suggestion: BrollSuggestion) -> float:
                score = suggestion.confidence
                
                # Boost score for matching category
                if suggestion.category == opportunity.category:
                    score += 0.1
                
                # Boost score for appropriate duration
                duration_diff = abs(suggestion.duration - opportunity.duration)
                if duration_diff < 1.0:
                    score += 0.1
                
                # Boost score for high-quality content
                if suggestion.broll_type == BrollType.STOCK_FOOTAGE:
                    score += 0.05  # Stock footage is generally higher quality
                
                return min(score, 1.0)
            
            # Sort by relevance score
            ranked = sorted(suggestions, key=calculate_relevance_score, reverse=True)
            
            return ranked
            
        except Exception as e:
            logger.error(f"Suggestion ranking failed: {e}")
            return suggestions

class BrollIntegrator:
    """Integrates B-roll content into video."""
    
    def __init__(self, config: BrollConfig = None):
        self.config = config or BrollConfig()
    
    async def integrate_broll(self, 
                            video_frames: List[np.ndarray],
                            suggestions: List[BrollSuggestion],
                            fps: float = 30.0) -> List[np.ndarray]:
        """Integrate B-roll content into video frames."""
        try:
            integrated_frames = video_frames.copy()
            
            for suggestion in suggestions:
                # Calculate frame range for B-roll insertion
                start_frame = int(suggestion.start_time * fps)
                end_frame = int(suggestion.end_time * fps)
                
                # Ensure frame range is within bounds
                start_frame = max(0, min(start_frame, len(integrated_frames) - 1))
                end_frame = max(start_frame + 1, min(end_frame, len(integrated_frames)))
                
                # Load B-roll content
                broll_frames = await self._load_broll_content(suggestion)
                
                if broll_frames:
                    # Integrate B-roll into video
                    integrated_frames = await self._insert_broll(
                        integrated_frames, broll_frames, start_frame, end_frame, suggestion
                    )
            
            return integrated_frames
            
        except Exception as e:
            logger.error(f"B-roll integration failed: {e}")
            raise ProcessingError(f"B-roll integration failed: {e}")
    
    async def _load_broll_content(self, suggestion: BrollSuggestion) -> List[np.ndarray]:
        """Load B-roll content from suggestion."""
        try:
            if suggestion.broll_type == BrollType.STOCK_FOOTAGE:
                return await self._load_stock_footage(suggestion)
            elif suggestion.broll_type == BrollType.AI_GENERATED:
                return await self._load_ai_generated(suggestion)
            elif suggestion.broll_type == BrollType.GRAPHIC:
                return await self._generate_graphic(suggestion)
            elif suggestion.broll_type == BrollType.TEXT_OVERLAY:
                return await self._generate_text_overlay(suggestion)
            else:
                return []
                
        except Exception as e:
            logger.error(f"B-roll content loading failed: {e}")
            return []
    
    async def _load_stock_footage(self, suggestion: BrollSuggestion) -> List[np.ndarray]:
        """Load stock footage content."""
        try:
            # Placeholder - would download and process stock footage
            # For now, return empty list
            return []
            
        except Exception as e:
            logger.error(f"Stock footage loading failed: {e}")
            return []
    
    async def _load_ai_generated(self, suggestion: BrollSuggestion) -> List[np.ndarray]:
        """Load AI-generated content."""
        try:
            # Placeholder - would load AI-generated content
            # For now, return empty list
            return []
            
        except Exception as e:
            logger.error(f"AI-generated content loading failed: {e}")
            return []
    
    async def _generate_graphic(self, suggestion: BrollSuggestion) -> List[np.ndarray]:
        """Generate graphic content."""
        try:
            # Create a simple graphic frame
            frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
            
            # Add some visual elements based on suggestion
            cv2.putText(frame, suggestion.title, (100, 540), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
            
            # Create multiple frames for animation
            frames = []
            for i in range(int(suggestion.duration * 30)):  # 30 FPS
                frames.append(frame.copy())
            
            return frames
            
        except Exception as e:
            logger.error(f"Graphic generation failed: {e}")
            return []
    
    async def _generate_text_overlay(self, suggestion: BrollSuggestion) -> List[np.ndarray]:
        """Generate text overlay content."""
        try:
            # Create transparent frame for text overlay
            frame = np.zeros((1080, 1920, 4), dtype=np.uint8)
            
            # Add text
            text = suggestion.metadata.get("text", suggestion.title)
            cv2.putText(frame, text, (100, 540), 
                       cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255, 255), 4)
            
            # Create multiple frames for animation
            frames = []
            for i in range(int(suggestion.duration * 30)):  # 30 FPS
                frames.append(frame.copy())
            
            return frames
            
        except Exception as e:
            logger.error(f"Text overlay generation failed: {e}")
            return []
    
    async def _insert_broll(self, 
                          video_frames: List[np.ndarray],
                          broll_frames: List[np.ndarray],
                          start_frame: int,
                          end_frame: int,
                          suggestion: BrollSuggestion) -> List[np.ndarray]:
        """Insert B-roll content into video frames."""
        try:
            # Calculate how many B-roll frames to use
            broll_duration = end_frame - start_frame
            broll_frame_count = min(len(broll_frames), broll_duration)
            
            # Resize B-roll frames to match video resolution
            if broll_frames:
                target_height, target_width = video_frames[0].shape[:2]
                resized_broll = []
                
                for frame in broll_frames[:broll_frame_count]:
                    resized = cv2.resize(frame, (target_width, target_height))
                    resized_broll.append(resized)
                
                # Insert B-roll frames
                for i, broll_frame in enumerate(resized_broll):
                    frame_idx = start_frame + i
                    if frame_idx < len(video_frames):
                        # Blend B-roll with original frame
                        if suggestion.broll_type == BrollType.TEXT_OVERLAY:
                            # Overlay text
                            video_frames[frame_idx] = self._overlay_text(
                                video_frames[frame_idx], broll_frame
                            )
                        else:
                            # Replace with B-roll
                            video_frames[frame_idx] = broll_frame
            
            return video_frames
            
        except Exception as e:
            logger.error(f"B-roll insertion failed: {e}")
            return video_frames
    
    def _overlay_text(self, video_frame: np.ndarray, text_frame: np.ndarray) -> np.ndarray:
        """Overlay text on video frame."""
        try:
            # Simple text overlay (would use more sophisticated blending)
            if text_frame.shape[2] == 4:  # Has alpha channel
                alpha = text_frame[:, :, 3] / 255.0
                for c in range(3):
                    video_frame[:, :, c] = (
                        alpha * text_frame[:, :, c] + 
                        (1 - alpha) * video_frame[:, :, c]
                    )
            else:
                # Simple overlay
                video_frame = cv2.addWeighted(video_frame, 0.7, text_frame, 0.3, 0)
            
            return video_frame
            
        except Exception as e:
            logger.error(f"Text overlay failed: {e}")
            return video_frame

class BrollIntegrationSystem:
    """Main B-roll integration system that orchestrates the entire process."""
    
    def __init__(self, config: BrollConfig = None):
        self.config = config or BrollConfig()
        self.content_analyzer = ContentAnalyzer(config)
        self.broll_suggester = BrollSuggester(config)
        self.broll_integrator = BrollIntegrator(config)
        self.executor = ThreadPoolExecutor(max_workers=3)
    
    async def process_video(self, 
                           video_path: str,
                           content_text: str,
                           output_path: str) -> Dict[str, Any]:
        """Process video with B-roll integration."""
        try:
            logger.info(f"Starting B-roll integration for video: {video_path}")
            start_time = time.time()
            
            # Extract frames
            frames = await self._extract_frames(video_path)
            
            # Analyze content for B-roll opportunities
            opportunities = await self.content_analyzer.analyze_content(content_text)
            
            # Generate B-roll suggestions
            all_suggestions = []
            for opportunity in opportunities:
                suggestions = await self.broll_suggester.suggest_broll(opportunity)
                all_suggestions.extend(suggestions)
            
            # Integrate B-roll into video
            integrated_frames = await self.broll_integrator.integrate_broll(
                frames, all_suggestions
            )
            
            # Create output video
            await self._create_output_video(integrated_frames, output_path)
            
            # Generate report
            report = await self._generate_integration_report(opportunities, all_suggestions)
            
            processing_time = time.time() - start_time
            
            return {
                "input_video": video_path,
                "output_video": output_path,
                "opportunities_found": len(opportunities),
                "suggestions_generated": len(all_suggestions),
                "integration_report": report,
                "processing_time": processing_time
            }
            
        except Exception as e:
            logger.error(f"B-roll integration processing failed: {e}")
            raise ProcessingError(f"B-roll integration processing failed: {e}")
    
    async def _extract_frames(self, video_path: str) -> List[np.ndarray]:
        """Extract frames from video."""
        try:
            cap = cv2.VideoCapture(video_path)
            frames = []
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            
            cap.release()
            return frames
            
        except Exception as e:
            logger.error(f"Frame extraction failed: {e}")
            raise ProcessingError(f"Frame extraction failed: {e}")
    
    async def _create_output_video(self, frames: List[np.ndarray], output_path: str):
        """Create output video from frames."""
        try:
            if not frames:
                return
            
            height, width = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))
            
            for frame in frames:
                out.write(frame)
            
            out.release()
            
        except Exception as e:
            logger.error(f"Output video creation failed: {e}")
            raise ProcessingError(f"Output video creation failed: {e}")
    
    async def _generate_integration_report(self, 
                                         opportunities: List[BrollOpportunity],
                                         suggestions: List[BrollSuggestion]) -> Dict[str, Any]:
        """Generate B-roll integration report."""
        try:
            # Count suggestions by type
            suggestion_counts = {}
            for suggestion in suggestions:
                broll_type = suggestion.broll_type.value
                suggestion_counts[broll_type] = suggestion_counts.get(broll_type, 0) + 1
            
            # Calculate average confidence
            avg_confidence = np.mean([s.confidence for s in suggestions]) if suggestions else 0.0
            
            # Count opportunities by category
            category_counts = {}
            for opportunity in opportunities:
                category = opportunity.category.value
                category_counts[category] = category_counts.get(category, 0) + 1
            
            return {
                "total_opportunities": len(opportunities),
                "total_suggestions": len(suggestions),
                "suggestion_counts": suggestion_counts,
                "category_counts": category_counts,
                "average_confidence": avg_confidence,
                "integration_quality": "high" if avg_confidence > 0.8 else "medium" if avg_confidence > 0.6 else "low"
            }
            
        except Exception as e:
            logger.error(f"Integration report generation failed: {e}")
            return {"error": str(e)}

# Placeholder classes for external APIs
class StockFootageAPI:
    """Placeholder for stock footage API."""
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    async def search(self, query: str, category: str, duration: float, count: int) -> List[Dict]:
        """Search for stock footage."""
        # Placeholder implementation
        return []

class AIVisualGenerator:
    """Placeholder for AI visual generation API."""
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    async def generate(self, prompt: str, duration: float, style: str) -> List[Dict]:
        """Generate AI visuals."""
        # Placeholder implementation
        return []

class SimpleNLPModel:
    """Simple NLP model placeholder."""
    def extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text."""
        # Simple keyword extraction
        words = text.lower().split()
        # Filter common words and return unique words
        common_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
        keywords = [word for word in words if word not in common_words and len(word) > 3]
        return list(set(keywords))[:10]

class SimpleSentimentAnalyzer:
    """Simple sentiment analyzer placeholder."""
    def analyze(self, text: str) -> str:
        """Analyze sentiment of text."""
        # Simple sentiment analysis
        positive_words = {"good", "great", "excellent", "amazing", "wonderful", "fantastic", "love", "like"}
        negative_words = {"bad", "terrible", "awful", "hate", "dislike", "horrible", "worst", "sad"}
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        else:
            return "neutral"

# Export the main class
__all__ = ["BrollIntegrationSystem", "ContentAnalyzer", "BrollSuggester", "BrollIntegrator", "BrollConfig"]


