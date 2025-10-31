"""
AI Service
==========

Advanced AI service for content generation, analysis, and optimization.
"""

from __future__ import annotations
import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
import json
import hashlib

from ...shared.container import DependencyInjectionContainer
from ...shared.events.event_bus import get_event_bus, DomainEvent, EventMetadata
from ...domain.value_objects.node_id import NodeId


logger = logging.getLogger(__name__)


@dataclass
class AIConfig:
    """AI service configuration"""
    provider: str = "openai"
    api_key: str = ""
    base_url: Optional[str] = None
    model: str = "gpt-4"
    max_tokens: int = 4000
    temperature: float = 0.7
    timeout: int = 30
    max_retries: int = 3
    rate_limit_per_minute: int = 60


@dataclass
class ContentAnalysis:
    """Content analysis result"""
    node_id: str
    overall_score: float
    readability_score: float
    sentiment_score: float
    seo_score: float
    grammar_score: float
    coherence_score: float
    topics: List[str]
    entities: List[Dict[str, Any]]
    keywords: List[str]
    summary: str
    suggestions: List[str]
    analyzed_at: datetime


@dataclass
class ContentGeneration:
    """Content generation result"""
    node_id: str
    generated_content: str
    prompt_used: str
    model_used: str
    tokens_used: int
    generation_time: float
    quality_score: float
    generated_at: datetime


class AIProvider(ABC):
    """Abstract AI provider interface"""
    
    @abstractmethod
    async def generate_content(self, prompt: str, **kwargs) -> str:
        """Generate content from prompt"""
        pass
    
    @abstractmethod
    async def analyze_content(self, content: str) -> ContentAnalysis:
        """Analyze content quality and characteristics"""
        pass
    
    @abstractmethod
    async def summarize_content(self, content: str) -> str:
        """Generate content summary"""
        pass
    
    @abstractmethod
    async def extract_topics(self, content: str) -> List[str]:
        """Extract topics from content"""
        pass
    
    @abstractmethod
    async def extract_entities(self, content: str) -> List[Dict[str, Any]]:
        """Extract entities from content"""
        pass


class OpenAIProvider(AIProvider):
    """OpenAI provider implementation"""
    
    def __init__(self, config: AIConfig):
        self.config = config
        self._client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize OpenAI client"""
        try:
            import openai
            self._client = openai.AsyncOpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                timeout=self.config.timeout
            )
        except ImportError:
            logger.error("OpenAI library not installed")
            raise
    
    async def generate_content(self, prompt: str, **kwargs) -> str:
        """Generate content using OpenAI"""
        try:
            response = await self._client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant that generates high-quality content."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
                temperature=kwargs.get("temperature", self.config.temperature)
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"OpenAI content generation failed: {e}")
            raise
    
    async def analyze_content(self, content: str) -> ContentAnalysis:
        """Analyze content using OpenAI"""
        try:
            analysis_prompt = f"""
            Analyze the following content and provide a comprehensive analysis:
            
            Content: {content}
            
            Please provide:
            1. Overall quality score (0-100)
            2. Readability score (0-100)
            3. Sentiment score (-1 to 1)
            4. SEO score (0-100)
            5. Grammar score (0-100)
            6. Coherence score (0-100)
            7. Main topics
            8. Key entities
            9. Important keywords
            10. Brief summary
            11. Improvement suggestions
            
            Format your response as JSON.
            """
            
            response = await self._client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": "You are an expert content analyst. Provide detailed analysis in JSON format."},
                    {"role": "user", "content": analysis_prompt}
                ],
                max_tokens=2000,
                temperature=0.3
            )
            
            analysis_data = json.loads(response.choices[0].message.content)
            
            return ContentAnalysis(
                node_id="",  # Will be set by caller
                overall_score=analysis_data.get("overall_score", 0),
                readability_score=analysis_data.get("readability_score", 0),
                sentiment_score=analysis_data.get("sentiment_score", 0),
                seo_score=analysis_data.get("seo_score", 0),
                grammar_score=analysis_data.get("grammar_score", 0),
                coherence_score=analysis_data.get("coherence_score", 0),
                topics=analysis_data.get("topics", []),
                entities=analysis_data.get("entities", []),
                keywords=analysis_data.get("keywords", []),
                summary=analysis_data.get("summary", ""),
                suggestions=analysis_data.get("suggestions", []),
                analyzed_at=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"OpenAI content analysis failed: {e}")
            raise
    
    async def summarize_content(self, content: str) -> str:
        """Generate content summary using OpenAI"""
        try:
            summary_prompt = f"""
            Provide a concise summary of the following content:
            
            {content}
            
            Summary:
            """
            
            response = await self._client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": "You are an expert at creating concise summaries."},
                    {"role": "user", "content": summary_prompt}
                ],
                max_tokens=500,
                temperature=0.3
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"OpenAI content summarization failed: {e}")
            raise
    
    async def extract_topics(self, content: str) -> List[str]:
        """Extract topics using OpenAI"""
        try:
            topics_prompt = f"""
            Extract the main topics from the following content:
            
            {content}
            
            Return only a list of topics, one per line.
            """
            
            response = await self._client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": "You are an expert at extracting topics from content."},
                    {"role": "user", "content": topics_prompt}
                ],
                max_tokens=200,
                temperature=0.3
            )
            
            topics_text = response.choices[0].message.content
            return [topic.strip() for topic in topics_text.split('\n') if topic.strip()]
            
        except Exception as e:
            logger.error(f"OpenAI topic extraction failed: {e}")
            raise
    
    async def extract_entities(self, content: str) -> List[Dict[str, Any]]:
        """Extract entities using OpenAI"""
        try:
            entities_prompt = f"""
            Extract named entities from the following content:
            
            {content}
            
            Return entities in JSON format with type and value.
            """
            
            response = await self._client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": "You are an expert at extracting named entities."},
                    {"role": "user", "content": entities_prompt}
                ],
                max_tokens=500,
                temperature=0.3
            )
            
            entities_text = response.choices[0].message.content
            return json.loads(entities_text)
            
        except Exception as e:
            logger.error(f"OpenAI entity extraction failed: {e}")
            raise


class AIService:
    """
    Advanced AI service for content operations
    
    Provides content generation, analysis, and optimization capabilities
    with multiple AI providers and caching.
    """
    
    def __init__(self, config: Optional[AIConfig] = None):
        self.config = config or AIConfig()
        self._providers: Dict[str, AIProvider] = {}
        self._cache: Dict[str, Any] = {}
        self._event_bus = get_event_bus()
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize AI providers"""
        if self.config.provider == "openai":
            self._providers["openai"] = OpenAIProvider(self.config)
        else:
            logger.warning(f"Unknown AI provider: {self.config.provider}")
    
    async def generate_content(
        self, 
        prompt: str, 
        node_id: str,
        provider: str = "openai",
        **kwargs
    ) -> ContentGeneration:
        """Generate content for a node"""
        try:
            start_time = datetime.utcnow()
            
            # Check cache
            cache_key = self._get_cache_key("generate", prompt, **kwargs)
            if cache_key in self._cache:
                cached_result = self._cache[cache_key]
                return ContentGeneration(
                    node_id=node_id,
                    generated_content=cached_result["content"],
                    prompt_used=prompt,
                    model_used=cached_result["model"],
                    tokens_used=cached_result["tokens"],
                    generation_time=0.1,  # Cached
                    quality_score=cached_result["quality_score"],
                    generated_at=start_time
                )
            
            # Generate content
            provider_instance = self._providers.get(provider)
            if not provider_instance:
                raise ValueError(f"Provider {provider} not available")
            
            generated_content = await provider_instance.generate_content(prompt, **kwargs)
            
            # Analyze quality
            quality_score = await self._analyze_quality(generated_content)
            
            end_time = datetime.utcnow()
            generation_time = (end_time - start_time).total_seconds()
            
            result = ContentGeneration(
                node_id=node_id,
                generated_content=generated_content,
                prompt_used=prompt,
                model_used=self.config.model,
                tokens_used=len(generated_content.split()) * 1.3,  # Approximate
                generation_time=generation_time,
                quality_score=quality_score,
                generated_at=end_time
            )
            
            # Cache result
            self._cache[cache_key] = {
                "content": generated_content,
                "model": self.config.model,
                "tokens": result.tokens_used,
                "quality_score": quality_score
            }
            
            # Publish event
            await self._publish_generation_event(result)
            
            logger.info(f"Generated content for node {node_id} in {generation_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Content generation failed for node {node_id}: {e}")
            raise
    
    async def analyze_node_content(self, node_id: str) -> ContentAnalysis:
        """Analyze content of a specific node"""
        try:
            # This would typically get the node content from the repository
            # For now, we'll simulate it
            content = "Sample content for analysis"
            
            # Check cache
            cache_key = self._get_cache_key("analyze", content)
            if cache_key in self._cache:
                cached_analysis = self._cache[cache_key]
                cached_analysis.node_id = node_id
                return cached_analysis
            
            # Analyze content
            provider = self._providers.get("openai")
            if not provider:
                raise ValueError("No AI provider available")
            
            analysis = await provider.analyze_content(content)
            analysis.node_id = node_id
            
            # Cache result
            self._cache[cache_key] = analysis
            
            # Publish event
            await self._publish_analysis_event(analysis)
            
            logger.info(f"Analyzed content for node {node_id}")
            return analysis
            
        except Exception as e:
            logger.error(f"Content analysis failed for node {node_id}: {e}")
            raise
    
    async def analyze_content_quality(self, node_id: str) -> float:
        """Analyze content quality and return score"""
        try:
            # This would typically get the node content from the repository
            content = "Sample content for quality analysis"
            
            # Check cache
            cache_key = self._get_cache_key("quality", content)
            if cache_key in self._cache:
                return self._cache[cache_key]
            
            # Analyze quality
            quality_score = await self._analyze_quality(content)
            
            # Cache result
            self._cache[cache_key] = quality_score
            
            logger.info(f"Analyzed quality for node {node_id}: {quality_score}")
            return quality_score
            
        except Exception as e:
            logger.error(f"Quality analysis failed for node {node_id}: {e}")
            raise
    
    async def optimize_content(self, node_id: str, content: str) -> str:
        """Optimize content for better quality"""
        try:
            optimization_prompt = f"""
            Optimize the following content for better quality, readability, and engagement:
            
            {content}
            
            Provide the optimized version.
            """
            
            provider = self._providers.get("openai")
            if not provider:
                raise ValueError("No AI provider available")
            
            optimized_content = await provider.generate_content(optimization_prompt)
            
            # Publish event
            await self._publish_optimization_event(node_id, content, optimized_content)
            
            logger.info(f"Optimized content for node {node_id}")
            return optimized_content
            
        except Exception as e:
            logger.error(f"Content optimization failed for node {node_id}: {e}")
            raise
    
    async def suggest_improvements(self, node_id: str, content: str) -> List[str]:
        """Suggest improvements for content"""
        try:
            suggestions_prompt = f"""
            Analyze the following content and suggest specific improvements:
            
            {content}
            
            Provide a list of actionable suggestions.
            """
            
            provider = self._providers.get("openai")
            if not provider:
                raise ValueError("No AI provider available")
            
            suggestions_text = await provider.generate_content(suggestions_prompt)
            suggestions = [s.strip() for s in suggestions_text.split('\n') if s.strip()]
            
            logger.info(f"Generated {len(suggestions)} suggestions for node {node_id}")
            return suggestions
            
        except Exception as e:
            logger.error(f"Improvement suggestions failed for node {node_id}: {e}")
            raise
    
    async def _analyze_quality(self, content: str) -> float:
        """Analyze content quality and return score"""
        try:
            # Simple quality analysis based on content characteristics
            word_count = len(content.split())
            sentence_count = content.count('.') + content.count('!') + content.count('?')
            paragraph_count = content.count('\n\n') + 1
            
            # Basic quality metrics
            avg_words_per_sentence = word_count / max(sentence_count, 1)
            avg_sentences_per_paragraph = sentence_count / max(paragraph_count, 1)
            
            # Quality score calculation (0-100)
            quality_score = 50  # Base score
            
            # Adjust based on metrics
            if 10 <= avg_words_per_sentence <= 20:
                quality_score += 20
            elif 5 <= avg_words_per_sentence <= 30:
                quality_score += 10
            
            if 3 <= avg_sentences_per_paragraph <= 7:
                quality_score += 20
            elif 2 <= avg_sentences_per_paragraph <= 10:
                quality_score += 10
            
            # Content length bonus
            if 100 <= word_count <= 1000:
                quality_score += 10
            
            return min(100, max(0, quality_score))
            
        except Exception as e:
            logger.error(f"Quality analysis failed: {e}")
            return 50.0  # Default score
    
    def _get_cache_key(self, operation: str, content: str, **kwargs) -> str:
        """Generate cache key for operation"""
        key_data = f"{operation}:{content}:{json.dumps(kwargs, sort_keys=True)}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    async def _publish_generation_event(self, result: ContentGeneration):
        """Publish content generation event"""
        event = DomainEvent(
            event_type="ai.content_generated",
            data={
                "node_id": result.node_id,
                "generated_content": result.generated_content,
                "prompt_used": result.prompt_used,
                "model_used": result.model_used,
                "tokens_used": result.tokens_used,
                "generation_time": result.generation_time,
                "quality_score": result.quality_score,
                "generated_at": result.generated_at.isoformat()
            },
            metadata=EventMetadata(
                source="ai_service",
                priority=3  # NORMAL
            )
        )
        
        await self._event_bus.publish(event)
    
    async def _publish_analysis_event(self, analysis: ContentAnalysis):
        """Publish content analysis event"""
        event = DomainEvent(
            event_type="ai.content_analyzed",
            data={
                "node_id": analysis.node_id,
                "overall_score": analysis.overall_score,
                "readability_score": analysis.readability_score,
                "sentiment_score": analysis.sentiment_score,
                "seo_score": analysis.seo_score,
                "grammar_score": analysis.grammar_score,
                "coherence_score": analysis.coherence_score,
                "topics": analysis.topics,
                "entities": analysis.entities,
                "keywords": analysis.keywords,
                "summary": analysis.summary,
                "suggestions": analysis.suggestions,
                "analyzed_at": analysis.analyzed_at.isoformat()
            },
            metadata=EventMetadata(
                source="ai_service",
                priority=3  # NORMAL
            )
        )
        
        await self._event_bus.publish(event)
    
    async def _publish_optimization_event(self, node_id: str, original_content: str, optimized_content: str):
        """Publish content optimization event"""
        event = DomainEvent(
            event_type="ai.content_optimized",
            data={
                "node_id": node_id,
                "original_content": original_content,
                "optimized_content": optimized_content,
                "optimized_at": datetime.utcnow().isoformat()
            },
            metadata=EventMetadata(
                source="ai_service",
                priority=3  # NORMAL
            )
        )
        
        await self._event_bus.publish(event)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get AI service statistics"""
        return {
            "providers": list(self._providers.keys()),
            "cache_size": len(self._cache),
            "config": {
                "provider": self.config.provider,
                "model": self.config.model,
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature
            }
        }




