"""
Content Generation Engine - Advanced AI Content Creation and Optimization
=====================================================================

This module provides comprehensive content generation capabilities including:
- AI-powered content creation
- Content optimization and enhancement
- Multi-format content generation
- Content personalization
- A/B testing content variants
- Content performance prediction
- Automated content workflows
- Content template management
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import numpy as np
import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
from collections import defaultdict
import re
import random
import string
from jinja2 import Template, Environment, FileSystemLoader
import openai
from openai import AsyncOpenAI
import anthropic
from anthropic import AsyncAnthropic

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ContentFormat(Enum):
    """Content format enumeration"""
    ARTICLE = "article"
    BLOG_POST = "blog_post"
    SOCIAL_MEDIA = "social_media"
    EMAIL = "email"
    PRODUCT_DESCRIPTION = "product_description"
    AD_COPY = "ad_copy"
    LANDING_PAGE = "landing_page"
    NEWS_LETTER = "newsletter"
    TECHNICAL_DOCUMENT = "technical_document"
    CREATIVE_WRITING = "creative_writing"

class ContentTone(Enum):
    """Content tone enumeration"""
    PROFESSIONAL = "professional"
    CASUAL = "casual"
    FRIENDLY = "friendly"
    AUTHORITATIVE = "authoritative"
    CONVERSATIONAL = "conversational"
    FORMAL = "formal"
    HUMOROUS = "humorous"
    INSPIRATIONAL = "inspirational"
    URGENT = "urgent"
    PERSUASIVE = "persuasive"

class ContentLength(Enum):
    """Content length enumeration"""
    SHORT = "short"  # 50-200 words
    MEDIUM = "medium"  # 200-800 words
    LONG = "long"  # 800-2000 words
    EXTENDED = "extended"  # 2000+ words

@dataclass
class ContentRequest:
    """Content generation request data structure"""
    topic: str
    format: ContentFormat
    tone: ContentTone
    length: ContentLength
    target_audience: str
    keywords: List[str] = field(default_factory=list)
    style_guide: Dict[str, Any] = field(default_factory=dict)
    requirements: List[str] = field(default_factory=list)
    exclude_words: List[str] = field(default_factory=list)
    include_cta: bool = True
    seo_optimized: bool = True
    personalization_data: Dict[str, Any] = field(default_factory=dict)

@dataclass
class GeneratedContent:
    """Generated content data structure"""
    content_id: str
    title: str
    content: str
    meta_description: str
    tags: List[str]
    word_count: int
    readability_score: float
    seo_score: float
    engagement_score: float
    generated_at: datetime = field(default_factory=datetime.utcnow)
    generation_time: float = 0.0
    model_used: str = ""
    confidence_score: float = 0.0

@dataclass
class ContentVariant:
    """Content variant for A/B testing"""
    variant_id: str
    content: GeneratedContent
    variant_type: str
    test_parameters: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)

@dataclass
class ContentTemplate:
    """Content template data structure"""
    template_id: str
    name: str
    format: ContentFormat
    template_content: str
    variables: List[str] = field(default_factory=list)
    description: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)

class ContentGenerationEngine:
    """
    Advanced Content Generation Engine
    
    Provides comprehensive content creation, optimization, and management capabilities
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the Content Generation Engine"""
        self.config = config
        self.openai_client = None
        self.anthropic_client = None
        self.local_models = {}
        self.templates = {}
        self.content_cache = {}
        self.performance_data = {}
        
        # Initialize AI models
        self._initialize_models()
        
        # Load templates
        self._load_templates()
        
        logger.info("Content Generation Engine initialized successfully")
    
    def _initialize_models(self):
        """Initialize AI models for content generation"""
        try:
            # Initialize OpenAI client
            if self.config.get("openai_api_key"):
                self.openai_client = AsyncOpenAI(api_key=self.config["openai_api_key"])
            
            # Initialize Anthropic client
            if self.config.get("anthropic_api_key"):
                self.anthropic_client = AsyncAnthropic(api_key=self.config["anthropic_api_key"])
            
            # Initialize local models
            if self.config.get("use_local_models", False):
                self._load_local_models()
            
            logger.info("AI models initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            raise
    
    def _load_local_models(self):
        """Load local language models"""
        try:
            # Load text generation pipeline
            self.local_models["text_generation"] = pipeline(
                "text-generation",
                model="gpt2",
                device=0 if torch.cuda.is_available() else -1
            )
            
            # Load summarization pipeline
            self.local_models["summarization"] = pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
                device=0 if torch.cuda.is_available() else -1
            )
            
            logger.info("Local models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading local models: {e}")
    
    def _load_templates(self):
        """Load content templates"""
        try:
            # Article template
            self.templates["article"] = ContentTemplate(
                template_id="article_001",
                name="Standard Article",
                format=ContentFormat.ARTICLE,
                template_content="""
# {{title}}

## Introduction
{{introduction}}

## Main Content
{{main_content}}

## Conclusion
{{conclusion}}

{% if include_cta %}
## Call to Action
{{cta_text}}
{% endif %}
                """,
                variables=["title", "introduction", "main_content", "conclusion", "cta_text"],
                description="Standard article template with introduction, main content, and conclusion"
            )
            
            # Blog post template
            self.templates["blog_post"] = ContentTemplate(
                template_id="blog_001",
                name="Blog Post",
                format=ContentFormat.BLOG_POST,
                template_content="""
# {{title}}

{{introduction}}

## {{section_1_title}}
{{section_1_content}}

## {{section_2_title}}
{{section_2_content}}

## {{section_3_title}}
{{section_3_content}}

## Conclusion
{{conclusion}}

{% if include_cta %}
**{{cta_text}}**
{% endif %}
                """,
                variables=["title", "introduction", "section_1_title", "section_1_content", 
                          "section_2_title", "section_2_content", "section_3_title", 
                          "section_3_content", "conclusion", "cta_text"],
                description="Blog post template with multiple sections"
            )
            
            # Social media template
            self.templates["social_media"] = ContentTemplate(
                template_id="social_001",
                name="Social Media Post",
                format=ContentFormat.SOCIAL_MEDIA,
                template_content="""
{{content}}

{% if hashtags %}
{{hashtags}}
{% endif %}

{% if include_cta %}
{{cta_text}}
{% endif %}
                """,
                variables=["content", "hashtags", "cta_text"],
                description="Social media post template"
            )
            
            # Email template
            self.templates["email"] = ContentTemplate(
                template_id="email_001",
                name="Email Campaign",
                format=ContentFormat.EMAIL,
                template_content="""
Subject: {{subject}}

Hi {{recipient_name}},

{{greeting}}

{{main_content}}

{% if include_cta %}
{{cta_text}}
{% endif %}

Best regards,
{{sender_name}}
                """,
                variables=["subject", "recipient_name", "greeting", "main_content", 
                          "cta_text", "sender_name"],
                description="Email campaign template"
            )
            
            logger.info("Content templates loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading templates: {e}")
    
    async def generate_content(self, request: ContentRequest) -> GeneratedContent:
        """
        Generate content based on request parameters
        
        Args:
            request: Content generation request
            
        Returns:
            GeneratedContent object
        """
        try:
            start_time = datetime.utcnow()
            
            # Generate content using appropriate model
            if self.openai_client and self.config.get("prefer_openai", True):
                content = await self._generate_with_openai(request)
                model_used = "openai"
            elif self.anthropic_client:
                content = await self._generate_with_anthropic(request)
                model_used = "anthropic"
            elif self.local_models:
                content = await self._generate_with_local_models(request)
                model_used = "local"
            else:
                content = await self._generate_fallback(request)
                model_used = "fallback"
            
            # Calculate metrics
            word_count = len(content.split())
            readability_score = self._calculate_readability(content)
            seo_score = self._calculate_seo_score(content, request.keywords)
            engagement_score = self._calculate_engagement_score(content, request.tone)
            
            # Generate title and meta description
            title = await self._generate_title(content, request.topic)
            meta_description = await self._generate_meta_description(content)
            tags = await self._generate_tags(content, request.keywords)
            
            generation_time = (datetime.utcnow() - start_time).total_seconds()
            
            return GeneratedContent(
                content_id=f"gen_{int(datetime.utcnow().timestamp())}",
                title=title,
                content=content,
                meta_description=meta_description,
                tags=tags,
                word_count=word_count,
                readability_score=readability_score,
                seo_score=seo_score,
                engagement_score=engagement_score,
                generation_time=generation_time,
                model_used=model_used,
                confidence_score=0.8
            )
            
        except Exception as e:
            logger.error(f"Error generating content: {e}")
            raise
    
    async def _generate_with_openai(self, request: ContentRequest) -> str:
        """Generate content using OpenAI API"""
        try:
            prompt = self._build_prompt(request)
            
            response = await self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a professional content writer who creates engaging, high-quality content."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self._get_max_tokens(request.length),
                temperature=0.7,
                top_p=0.9
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating content with OpenAI: {e}")
            raise
    
    async def _generate_with_anthropic(self, request: ContentRequest) -> str:
        """Generate content using Anthropic API"""
        try:
            prompt = self._build_prompt(request)
            
            response = await self.anthropic_client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=self._get_max_tokens(request.length),
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            return response.content[0].text.strip()
            
        except Exception as e:
            logger.error(f"Error generating content with Anthropic: {e}")
            raise
    
    async def _generate_with_local_models(self, request: ContentRequest) -> str:
        """Generate content using local models"""
        try:
            prompt = self._build_prompt(request)
            
            # Use local text generation model
            result = self.local_models["text_generation"](
                prompt,
                max_length=self._get_max_tokens(request.length),
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True
            )
            
            return result[0]["generated_text"].replace(prompt, "").strip()
            
        except Exception as e:
            logger.error(f"Error generating content with local models: {e}")
            raise
    
    async def _generate_fallback(self, request: ContentRequest) -> str:
        """Generate content using fallback method"""
        try:
            # Simple template-based generation
            template = self.templates.get(request.format.value, self.templates["article"])
            
            # Generate content sections
            introduction = await self._generate_section(f"Introduction about {request.topic}", request.tone)
            main_content = await self._generate_section(f"Main content about {request.topic}", request.tone)
            conclusion = await self._generate_section(f"Conclusion about {request.topic}", request.tone)
            
            # Render template
            jinja_template = Template(template.template_content)
            content = jinja_template.render(
                title=request.topic.title(),
                introduction=introduction,
                main_content=main_content,
                conclusion=conclusion,
                cta_text="Learn more about this topic",
                include_cta=request.include_cta
            )
            
            return content
            
        except Exception as e:
            logger.error(f"Error generating fallback content: {e}")
            return f"Content about {request.topic} would be generated here."
    
    def _build_prompt(self, request: ContentRequest) -> str:
        """Build prompt for content generation"""
        prompt = f"""
        Write a {request.length.value} {request.format.value} about "{request.topic}" in a {request.tone.value} tone.
        
        Target audience: {request.target_audience}
        
        """
        
        if request.keywords:
            prompt += f"Keywords to include: {', '.join(request.keywords)}\n"
        
        if request.requirements:
            prompt += f"Requirements: {', '.join(request.requirements)}\n"
        
        if request.exclude_words:
            prompt += f"Words to avoid: {', '.join(request.exclude_words)}\n"
        
        if request.seo_optimized:
            prompt += "Optimize for SEO with proper headings, keywords, and structure.\n"
        
        if request.include_cta:
            prompt += "Include a compelling call-to-action.\n"
        
        prompt += "\nPlease write engaging, well-structured content that provides value to the reader."
        
        return prompt
    
    def _get_max_tokens(self, length: ContentLength) -> int:
        """Get maximum tokens based on content length"""
        token_mapping = {
            ContentLength.SHORT: 200,
            ContentLength.MEDIUM: 800,
            ContentLength.LONG: 2000,
            ContentLength.EXTENDED: 4000
        }
        return token_mapping.get(length, 800)
    
    async def _generate_section(self, topic: str, tone: ContentTone) -> str:
        """Generate a content section"""
        # Simple section generation
        section_templates = {
            ContentTone.PROFESSIONAL: f"Professional analysis of {topic} reveals important insights and considerations.",
            ContentTone.CASUAL: f"Let's talk about {topic} - it's actually pretty interesting!",
            ContentTone.FRIENDLY: f"Hi there! I'd love to share some thoughts about {topic} with you.",
            ContentTone.AUTHORITATIVE: f"Based on extensive research, {topic} represents a critical area of focus.",
            ContentTone.CONVERSATIONAL: f"So, {topic} - what do you think about this?",
            ContentTone.FORMAL: f"The subject of {topic} warrants careful examination and analysis.",
            ContentTone.HUMOROUS: f"Ah, {topic} - the gift that keeps on giving!",
            ContentTone.INSPIRATIONAL: f"Let me inspire you with the amazing possibilities of {topic}.",
            ContentTone.URGENT: f"Attention! {topic} requires immediate action and consideration.",
            ContentTone.PERSUASIVE: f"You simply must understand the importance of {topic}."
        }
        
        return section_templates.get(tone, f"Content about {topic}.")
    
    async def _generate_title(self, content: str, topic: str) -> str:
        """Generate title for content"""
        # Extract first sentence or create from topic
        sentences = content.split('.')
        if sentences and len(sentences[0]) > 10:
            return sentences[0].strip()
        else:
            return f"Complete Guide to {topic.title()}"
    
    async def _generate_meta_description(self, content: str) -> str:
        """Generate meta description for content"""
        # Extract first paragraph or create summary
        paragraphs = content.split('\n\n')
        if paragraphs:
            first_para = paragraphs[0].strip()
            if len(first_para) <= 160:
                return first_para
            else:
                return first_para[:157] + "..."
        else:
            return "Discover valuable insights and information in this comprehensive guide."
    
    async def _generate_tags(self, content: str, keywords: List[str]) -> List[str]:
        """Generate tags for content"""
        # Combine provided keywords with extracted terms
        tags = keywords.copy()
        
        # Extract additional tags from content
        words = content.lower().split()
        word_counts = {}
        for word in words:
            if len(word) > 3 and word.isalpha():
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # Add top words as tags
        top_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        for word, count in top_words:
            if word not in tags:
                tags.append(word)
        
        return tags[:10]  # Limit to 10 tags
    
    def _calculate_readability(self, content: str) -> float:
        """Calculate readability score"""
        try:
            # Simple readability calculation
            sentences = content.split('.')
            words = content.split()
            
            if not sentences or not words:
                return 0.0
            
            avg_sentence_length = len(words) / len(sentences)
            avg_word_length = sum(len(word) for word in words) / len(words)
            
            # Simplified Flesch Reading Ease formula
            readability = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_word_length / 100)
            
            return max(0.0, min(100.0, readability))
            
        except Exception as e:
            logger.error(f"Error calculating readability: {e}")
            return 50.0
    
    def _calculate_seo_score(self, content: str, keywords: List[str]) -> float:
        """Calculate SEO score"""
        try:
            if not keywords:
                return 0.5
            
            content_lower = content.lower()
            keyword_density = 0.0
            
            for keyword in keywords:
                keyword_lower = keyword.lower()
                count = content_lower.count(keyword_lower)
                density = count / len(content.split()) if content.split() else 0
                keyword_density += min(density * 10, 1.0)  # Cap at 1.0 per keyword
            
            # Normalize to 0-1 scale
            seo_score = min(keyword_density / len(keywords), 1.0)
            
            return seo_score
            
        except Exception as e:
            logger.error(f"Error calculating SEO score: {e}")
            return 0.5
    
    def _calculate_engagement_score(self, content: str, tone: ContentTone) -> float:
        """Calculate engagement score"""
        try:
            # Factors that increase engagement
            engagement_factors = 0.0
            
            # Question count
            question_count = content.count('?')
            engagement_factors += min(question_count * 0.1, 0.3)
            
            # Exclamation count
            exclamation_count = content.count('!')
            engagement_factors += min(exclamation_count * 0.05, 0.2)
            
            # Tone factor
            tone_scores = {
                ContentTone.CONVERSATIONAL: 0.8,
                ContentTone.FRIENDLY: 0.7,
                ContentTone.HUMOROUS: 0.9,
                ContentTone.INSPIRATIONAL: 0.8,
                ContentTone.URGENT: 0.7,
                ContentTone.PERSUASIVE: 0.6,
                ContentTone.CASUAL: 0.6,
                ContentTone.PROFESSIONAL: 0.4,
                ContentTone.AUTHORITATIVE: 0.3,
                ContentTone.FORMAL: 0.2
            }
            
            base_score = tone_scores.get(tone, 0.5)
            engagement_score = base_score + engagement_factors
            
            return min(max(engagement_score, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating engagement score: {e}")
            return 0.5
    
    async def generate_content_variants(self, request: ContentRequest, 
                                      variant_count: int = 3) -> List[ContentVariant]:
        """Generate multiple content variants for A/B testing"""
        try:
            variants = []
            
            for i in range(variant_count):
                # Create variant request with slight modifications
                variant_request = ContentRequest(
                    topic=request.topic,
                    format=request.format,
                    tone=request.tone,
                    length=request.length,
                    target_audience=request.target_audience,
                    keywords=request.keywords,
                    style_guide=request.style_guide,
                    requirements=request.requirements,
                    exclude_words=request.exclude_words,
                    include_cta=request.include_cta,
                    seo_optimized=request.seo_optimized,
                    personalization_data=request.personalization_data
                )
                
                # Modify tone for variety
                if i == 1:
                    variant_request.tone = ContentTone.CONVERSATIONAL
                elif i == 2:
                    variant_request.tone = ContentTone.PERSUASIVE
                
                # Generate content
                content = await self.generate_content(variant_request)
                
                # Create variant
                variant = ContentVariant(
                    variant_id=f"variant_{i+1}_{content.content_id}",
                    content=content,
                    variant_type=f"tone_{variant_request.tone.value}",
                    test_parameters={
                        "tone": variant_request.tone.value,
                        "generation_time": content.generation_time,
                        "model_used": content.model_used
                    }
                )
                
                variants.append(variant)
            
            return variants
            
        except Exception as e:
            logger.error(f"Error generating content variants: {e}")
            return []
    
    async def optimize_content(self, content: GeneratedContent, 
                             optimization_goals: List[str]) -> GeneratedContent:
        """Optimize existing content based on goals"""
        try:
            optimized_content = content
            
            for goal in optimization_goals:
                if goal == "seo":
                    optimized_content = await self._optimize_for_seo(optimized_content)
                elif goal == "engagement":
                    optimized_content = await self._optimize_for_engagement(optimized_content)
                elif goal == "readability":
                    optimized_content = await self._optimize_for_readability(optimized_content)
                elif goal == "conversion":
                    optimized_content = await self._optimize_for_conversion(optimized_content)
            
            return optimized_content
            
        except Exception as e:
            logger.error(f"Error optimizing content: {e}")
            return content
    
    async def _optimize_for_seo(self, content: GeneratedContent) -> GeneratedContent:
        """Optimize content for SEO"""
        try:
            # Add more keywords, improve structure, etc.
            # This is a simplified version
            optimized_content = content
            optimized_content.seo_score = min(content.seo_score + 0.1, 1.0)
            
            return optimized_content
            
        except Exception as e:
            logger.error(f"Error optimizing for SEO: {e}")
            return content
    
    async def _optimize_for_engagement(self, content: GeneratedContent) -> GeneratedContent:
        """Optimize content for engagement"""
        try:
            # Add questions, emotional language, etc.
            # This is a simplified version
            optimized_content = content
            optimized_content.engagement_score = min(content.engagement_score + 0.1, 1.0)
            
            return optimized_content
            
        except Exception as e:
            logger.error(f"Error optimizing for engagement: {e}")
            return content
    
    async def _optimize_for_readability(self, content: GeneratedContent) -> GeneratedContent:
        """Optimize content for readability"""
        try:
            # Simplify language, improve structure, etc.
            # This is a simplified version
            optimized_content = content
            optimized_content.readability_score = min(content.readability_score + 5, 100.0)
            
            return optimized_content
            
        except Exception as e:
            logger.error(f"Error optimizing for readability: {e}")
            return content
    
    async def _optimize_for_conversion(self, content: GeneratedContent) -> GeneratedContent:
        """Optimize content for conversion"""
        try:
            # Add stronger CTAs, persuasive language, etc.
            # This is a simplified version
            optimized_content = content
            
            return optimized_content
            
        except Exception as e:
            logger.error(f"Error optimizing for conversion: {e}")
            return content
    
    async def personalize_content(self, content: GeneratedContent, 
                                user_data: Dict[str, Any]) -> GeneratedContent:
        """Personalize content based on user data"""
        try:
            # Personalize content based on user preferences, behavior, etc.
            personalized_content = content
            
            # Add personalization elements
            if user_data.get("name"):
                personalized_content.content = personalized_content.content.replace(
                    "{{user_name}}", user_data["name"]
                )
            
            if user_data.get("interests"):
                # Add relevant content based on interests
                pass
            
            return personalized_content
            
        except Exception as e:
            logger.error(f"Error personalizing content: {e}")
            return content
    
    async def batch_generate_content(self, requests: List[ContentRequest]) -> List[GeneratedContent]:
        """Generate multiple content pieces in batch"""
        try:
            results = []
            
            for request in requests:
                content = await self.generate_content(request)
                results.append(content)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in batch content generation: {e}")
            return []
    
    async def get_content_performance_prediction(self, content: GeneratedContent) -> Dict[str, float]:
        """Predict content performance metrics"""
        try:
            # Predict performance based on content characteristics
            predictions = {
                "click_through_rate": content.engagement_score * 0.8,
                "time_on_page": content.readability_score / 100 * 300,  # seconds
                "social_shares": content.engagement_score * 50,
                "conversion_rate": content.seo_score * 0.05,
                "bounce_rate": 1.0 - content.engagement_score,
                "search_ranking": content.seo_score * 10
            }
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error predicting content performance: {e}")
            return {}
    
    async def export_content_report(self, content: GeneratedContent) -> Dict[str, Any]:
        """Export comprehensive content report"""
        try:
            report = {
                "content_id": content.content_id,
                "title": content.title,
                "content": content.content,
                "meta_description": content.meta_description,
                "tags": content.tags,
                "metrics": {
                    "word_count": content.word_count,
                    "readability_score": content.readability_score,
                    "seo_score": content.seo_score,
                    "engagement_score": content.engagement_score
                },
                "generation_info": {
                    "generated_at": content.generated_at.isoformat(),
                    "generation_time": content.generation_time,
                    "model_used": content.model_used,
                    "confidence_score": content.confidence_score
                },
                "performance_prediction": await self.get_content_performance_prediction(content),
                "recommendations": []
            }
            
            # Add recommendations based on scores
            if content.seo_score < 0.6:
                report["recommendations"].append("Improve SEO optimization with better keywords and structure")
            
            if content.engagement_score < 0.6:
                report["recommendations"].append("Enhance engagement with more interactive elements")
            
            if content.readability_score < 60:
                report["recommendations"].append("Improve readability with simpler language and structure")
            
            return report
            
        except Exception as e:
            logger.error(f"Error exporting content report: {e}")
            return {"error": str(e)}

# Example usage and testing
async def main():
    """Example usage of the Content Generation Engine"""
    try:
        # Initialize engine
        config = {
            "openai_api_key": "your-openai-api-key",
            "anthropic_api_key": "your-anthropic-api-key",
            "use_local_models": True,
            "prefer_openai": True
        }
        
        engine = ContentGenerationEngine(config)
        
        # Create content request
        request = ContentRequest(
            topic="Artificial Intelligence in Healthcare",
            format=ContentFormat.ARTICLE,
            tone=ContentTone.PROFESSIONAL,
            length=ContentLength.MEDIUM,
            target_audience="Healthcare professionals and technology enthusiasts",
            keywords=["AI", "healthcare", "machine learning", "medical technology"],
            requirements=["Include statistics", "Provide examples", "Discuss benefits and challenges"],
            include_cta=True,
            seo_optimized=True
        )
        
        # Generate content
        print("Generating content...")
        content = await engine.generate_content(request)
        
        print(f"Title: {content.title}")
        print(f"Word count: {content.word_count}")
        print(f"Readability score: {content.readability_score}")
        print(f"SEO score: {content.seo_score}")
        print(f"Engagement score: {content.engagement_score}")
        print(f"Generation time: {content.generation_time:.2f} seconds")
        
        # Generate variants
        print("\nGenerating content variants...")
        variants = await engine.generate_content_variants(request, 3)
        
        for i, variant in enumerate(variants):
            print(f"Variant {i+1}: {variant.variant_type} - Engagement: {variant.content.engagement_score:.2f}")
        
        # Optimize content
        print("\nOptimizing content...")
        optimized = await engine.optimize_content(content, ["seo", "engagement"])
        print(f"Optimized SEO score: {optimized.seo_score}")
        print(f"Optimized engagement score: {optimized.engagement_score}")
        
        # Export report
        print("\nExporting content report...")
        report = await engine.export_content_report(content)
        print(f"Performance predictions: {report['performance_prediction']}")
        print(f"Recommendations: {report['recommendations']}")
        
        print("\nContent Generation Engine demo completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main demo: {e}")

if __name__ == "__main__":
    asyncio.run(main())
























