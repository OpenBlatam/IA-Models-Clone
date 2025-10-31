"""
AI Content Generation System
============================

Advanced AI-powered content generation system with multiple models,
style adaptation, and intelligent content optimization.
"""

import logging
import asyncio
import json
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import uuid
from pathlib import Path
import yaml
import re
import random

logger = logging.getLogger(__name__)

class ContentType(Enum):
    """Content types"""
    ARTICLE = "article"
    BLOG_POST = "blog_post"
    NEWS_STORY = "news_story"
    TECHNICAL_DOC = "technical_doc"
    CREATIVE_WRITING = "creative_writing"
    MARKETING_COPY = "marketing_copy"
    SOCIAL_MEDIA = "social_media"
    EMAIL = "email"
    PROPOSAL = "proposal"
    REPORT = "report"
    PRESENTATION = "presentation"
    SCRIPT = "script"
    ACADEMIC_PAPER = "academic_paper"
    LEGAL_DOCUMENT = "legal_document"
    BUSINESS_PLAN = "business_plan"

class WritingStyle(Enum):
    """Writing styles"""
    FORMAL = "formal"
    CASUAL = "casual"
    TECHNICAL = "technical"
    CREATIVE = "creative"
    PERSUASIVE = "persuasive"
    INFORMATIVE = "informative"
    CONVERSATIONAL = "conversational"
    ACADEMIC = "academic"
    BUSINESS = "business"
    JOURNALISTIC = "journalistic"

class ContentQuality(Enum):
    """Content quality levels"""
    DRAFT = "draft"
    GOOD = "good"
    EXCELLENT = "excellent"
    PUBLICATION_READY = "publication_ready"

@dataclass
class ContentRequest:
    """Content generation request"""
    id: str
    content_type: ContentType
    topic: str
    style: WritingStyle
    length: int  # words
    target_audience: str
    keywords: List[str] = field(default_factory=list)
    tone: str = "neutral"
    language: str = "en"
    requirements: Dict[str, Any] = field(default_factory=dict)
    template_id: Optional[str] = None
    reference_content: Optional[str] = None
    custom_instructions: Optional[str] = None

@dataclass
class ContentResult:
    """Generated content result"""
    id: str
    request_id: str
    content: str
    title: str
    summary: str
    quality_score: float
    quality_level: ContentQuality
    word_count: int
    reading_time: int  # minutes
    keywords_used: List[str]
    style_analysis: Dict[str, Any]
    seo_score: float
    readability_score: float
    generated_at: datetime
    processing_time: float
    model_used: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ContentTemplate:
    """Content template definition"""
    id: str
    name: str
    content_type: ContentType
    structure: List[Dict[str, Any]]
    style_guidelines: Dict[str, Any]
    word_count_ranges: Tuple[int, int]
    required_sections: List[str]
    optional_sections: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)

class AIContentGenerator:
    """
    Advanced AI content generation system
    """
    
    def __init__(self, templates_dir: Optional[str] = None):
        """
        Initialize AI content generator
        
        Args:
            templates_dir: Directory for content templates
        """
        self.templates_dir = Path(templates_dir) if templates_dir else Path(__file__).parent / "templates"
        self.templates_dir.mkdir(exist_ok=True)
        
        # Content templates
        self.templates: Dict[str, ContentTemplate] = {}
        self.generated_content: Dict[str, ContentResult] = {}
        
        # Content patterns and rules
        self.content_patterns = self._initialize_content_patterns()
        self.style_guidelines = self._initialize_style_guidelines()
        
        # Initialize templates
        self._initialize_templates()
        
        # Content optimization rules
        self.optimization_rules = self._initialize_optimization_rules()
    
    def _initialize_content_patterns(self) -> Dict[ContentType, Dict[str, Any]]:
        """Initialize content patterns for different types"""
        return {
            ContentType.ARTICLE: {
                "structure": ["introduction", "body", "conclusion"],
                "min_sections": 3,
                "max_sections": 10,
                "section_requirements": {
                    "introduction": {"min_words": 50, "max_words": 200},
                    "body": {"min_words": 300, "max_words": 2000},
                    "conclusion": {"min_words": 50, "max_words": 200}
                }
            },
            ContentType.BLOG_POST: {
                "structure": ["hook", "introduction", "main_content", "conclusion", "cta"],
                "min_sections": 4,
                "max_sections": 8,
                "section_requirements": {
                    "hook": {"min_words": 20, "max_words": 50},
                    "introduction": {"min_words": 100, "max_words": 300},
                    "main_content": {"min_words": 500, "max_words": 1500},
                    "conclusion": {"min_words": 50, "max_words": 150},
                    "cta": {"min_words": 20, "max_words": 100}
                }
            },
            ContentType.TECHNICAL_DOC: {
                "structure": ["overview", "requirements", "implementation", "testing", "conclusion"],
                "min_sections": 4,
                "max_sections": 12,
                "section_requirements": {
                    "overview": {"min_words": 100, "max_words": 300},
                    "requirements": {"min_words": 200, "max_words": 500},
                    "implementation": {"min_words": 400, "max_words": 1000},
                    "testing": {"min_words": 200, "max_words": 500},
                    "conclusion": {"min_words": 100, "max_words": 300}
                }
            },
            ContentType.MARKETING_COPY: {
                "structure": ["headline", "subheadline", "benefits", "features", "cta"],
                "min_sections": 3,
                "max_sections": 6,
                "section_requirements": {
                    "headline": {"min_words": 5, "max_words": 15},
                    "subheadline": {"min_words": 10, "max_words": 30},
                    "benefits": {"min_words": 100, "max_words": 300},
                    "features": {"min_words": 150, "max_words": 400},
                    "cta": {"min_words": 10, "max_words": 50}
                }
            },
            ContentType.ACADEMIC_PAPER: {
                "structure": ["abstract", "introduction", "literature_review", "methodology", "results", "discussion", "conclusion", "references"],
                "min_sections": 6,
                "max_sections": 10,
                "section_requirements": {
                    "abstract": {"min_words": 150, "max_words": 300},
                    "introduction": {"min_words": 300, "max_words": 600},
                    "literature_review": {"min_words": 500, "max_words": 1000},
                    "methodology": {"min_words": 400, "max_words": 800},
                    "results": {"min_words": 300, "max_words": 600},
                    "discussion": {"min_words": 400, "max_words": 800},
                    "conclusion": {"min_words": 200, "max_words": 400}
                }
            }
        }
    
    def _initialize_style_guidelines(self) -> Dict[WritingStyle, Dict[str, Any]]:
        """Initialize style guidelines for different writing styles"""
        return {
            WritingStyle.FORMAL: {
                "tone": "professional",
                "voice": "third_person",
                "sentence_structure": "complex",
                "vocabulary": "advanced",
                "contractions": False,
                "slang": False,
                "examples": ["utilize", "implement", "facilitate", "comprehensive"]
            },
            WritingStyle.CASUAL: {
                "tone": "friendly",
                "voice": "first_person",
                "sentence_structure": "simple",
                "vocabulary": "everyday",
                "contractions": True,
                "slang": True,
                "examples": ["use", "do", "help", "complete"]
            },
            WritingStyle.TECHNICAL: {
                "tone": "precise",
                "voice": "third_person",
                "sentence_structure": "complex",
                "vocabulary": "technical",
                "contractions": False,
                "slang": False,
                "examples": ["configure", "initialize", "execute", "optimize"]
            },
            WritingStyle.CREATIVE: {
                "tone": "engaging",
                "voice": "first_person",
                "sentence_structure": "varied",
                "vocabulary": "descriptive",
                "contractions": True,
                "slang": False,
                "examples": ["captivating", "mesmerizing", "breathtaking", "enchanting"]
            },
            WritingStyle.PERSUASIVE: {
                "tone": "convincing",
                "voice": "second_person",
                "sentence_structure": "varied",
                "vocabulary": "powerful",
                "contractions": True,
                "slang": False,
                "examples": ["transform", "revolutionize", "breakthrough", "essential"]
            },
            WritingStyle.ACADEMIC: {
                "tone": "scholarly",
                "voice": "third_person",
                "sentence_structure": "complex",
                "vocabulary": "academic",
                "contractions": False,
                "slang": False,
                "examples": ["analyze", "examine", "investigate", "demonstrate"]
            }
        }
    
    def _initialize_templates(self):
        """Initialize content templates"""
        # Article template
        self.templates["article_template"] = ContentTemplate(
            id="article_template",
            name="Standard Article Template",
            content_type=ContentType.ARTICLE,
            structure=[
                {"section": "introduction", "purpose": "Hook the reader and introduce the topic"},
                {"section": "body", "purpose": "Develop the main points with evidence"},
                {"section": "conclusion", "purpose": "Summarize and provide closure"}
            ],
            style_guidelines={
                "tone": "informative",
                "voice": "third_person",
                "sentence_length": "medium"
            },
            word_count_ranges=(500, 2000),
            required_sections=["introduction", "body", "conclusion"],
            examples=[
                "How to Start a Business in 2024",
                "The Future of Artificial Intelligence",
                "Sustainable Living: A Complete Guide"
            ]
        )
        
        # Blog post template
        self.templates["blog_template"] = ContentTemplate(
            id="blog_template",
            name="Blog Post Template",
            content_type=ContentType.BLOG_POST,
            structure=[
                {"section": "hook", "purpose": "Grab attention with an interesting opening"},
                {"section": "introduction", "purpose": "Set context and preview what's coming"},
                {"section": "main_content", "purpose": "Deliver value with actionable insights"},
                {"section": "conclusion", "purpose": "Wrap up and reinforce key points"},
                {"section": "cta", "purpose": "Encourage reader action"}
            ],
            style_guidelines={
                "tone": "conversational",
                "voice": "first_person",
                "sentence_length": "short"
            },
            word_count_ranges=(800, 1500),
            required_sections=["hook", "introduction", "main_content", "conclusion", "cta"],
            examples=[
                "5 Productivity Hacks That Changed My Life",
                "Why I Switched to Remote Work (And You Should Too)",
                "The Beginner's Guide to Digital Marketing"
            ]
        )
        
        # Technical documentation template
        self.templates["technical_template"] = ContentTemplate(
            id="technical_template",
            name="Technical Documentation Template",
            content_type=ContentType.TECHNICAL_DOC,
            structure=[
                {"section": "overview", "purpose": "Provide high-level understanding"},
                {"section": "requirements", "purpose": "List prerequisites and dependencies"},
                {"section": "implementation", "purpose": "Detail the technical implementation"},
                {"section": "testing", "purpose": "Explain testing procedures"},
                {"section": "conclusion", "purpose": "Summarize and provide next steps"}
            ],
            style_guidelines={
                "tone": "precise",
                "voice": "third_person",
                "sentence_length": "medium"
            },
            word_count_ranges=(1000, 3000),
            required_sections=["overview", "requirements", "implementation", "testing", "conclusion"],
            examples=[
                "API Integration Guide",
                "Database Migration Tutorial",
                "Security Best Practices"
            ]
        )
        
        # Marketing copy template
        self.templates["marketing_template"] = ContentTemplate(
            id="marketing_template",
            name="Marketing Copy Template",
            content_type=ContentType.MARKETING_COPY,
            structure=[
                {"section": "headline", "purpose": "Create compelling attention-grabber"},
                {"section": "subheadline", "purpose": "Provide additional context"},
                {"section": "benefits", "purpose": "Highlight value propositions"},
                {"section": "features", "purpose": "Detail product/service features"},
                {"section": "cta", "purpose": "Drive action with clear call-to-action"}
            ],
            style_guidelines={
                "tone": "persuasive",
                "voice": "second_person",
                "sentence_length": "short"
            },
            word_count_ranges=(200, 800),
            required_sections=["headline", "subheadline", "benefits", "features", "cta"],
            examples=[
                "Revolutionary Software That Saves 10 Hours Per Week",
                "Transform Your Business with AI-Powered Analytics",
                "The Ultimate Guide to Digital Transformation"
            ]
        )
    
    def _initialize_optimization_rules(self) -> Dict[str, List[Dict[str, Any]]]:
        """Initialize content optimization rules"""
        return {
            "seo": [
                {"rule": "keyword_density", "min": 0.5, "max": 2.0, "description": "Keyword density should be between 0.5% and 2%"},
                {"rule": "title_length", "min": 30, "max": 60, "description": "Title should be 30-60 characters"},
                {"rule": "meta_description", "min": 120, "max": 160, "description": "Meta description should be 120-160 characters"},
                {"rule": "heading_structure", "description": "Use proper H1, H2, H3 hierarchy"},
                {"rule": "internal_links", "min": 2, "description": "Include at least 2 internal links"}
            ],
            "readability": [
                {"rule": "sentence_length", "max": 20, "description": "Average sentence length should not exceed 20 words"},
                {"rule": "paragraph_length", "max": 150, "description": "Paragraphs should not exceed 150 words"},
                {"rule": "passive_voice", "max": 10, "description": "Passive voice should not exceed 10%"},
                {"rule": "complex_words", "max": 15, "description": "Complex words should not exceed 15%"},
                {"rule": "transition_words", "min": 3, "description": "Use at least 3 transition words"}
            ],
            "engagement": [
                {"rule": "hook_strength", "description": "Start with a compelling hook"},
                {"rule": "storytelling", "description": "Include storytelling elements"},
                {"rule": "questions", "min": 2, "description": "Ask at least 2 engaging questions"},
                {"rule": "examples", "min": 3, "description": "Provide at least 3 concrete examples"},
                {"rule": "visual_elements", "description": "Suggest visual elements for better engagement"}
            ]
        }
    
    async def generate_content(self, request: ContentRequest) -> ContentResult:
        """
        Generate content based on request
        
        Args:
            request: Content generation request
            
        Returns:
            Generated content result
        """
        start_time = datetime.now()
        result_id = str(uuid.uuid4())
        
        try:
            # Get template
            template = self._get_template(request)
            
            # Generate content structure
            content_structure = self._generate_content_structure(request, template)
            
            # Generate content for each section
            generated_sections = {}
            for section in content_structure:
                section_content = await self._generate_section_content(
                    section, request, template
                )
                generated_sections[section["name"]] = section_content
            
            # Combine sections into final content
            final_content = self._combine_sections(generated_sections, template)
            
            # Generate title and summary
            title = self._generate_title(request, final_content)
            summary = self._generate_summary(final_content)
            
            # Analyze content quality
            quality_analysis = self._analyze_content_quality(final_content, request)
            
            # Calculate metrics
            word_count = len(final_content.split())
            reading_time = max(1, word_count // 200)  # Average reading speed: 200 words/minute
            
            # Calculate SEO score
            seo_score = self._calculate_seo_score(final_content, request.keywords)
            
            # Calculate readability score
            readability_score = self._calculate_readability_score(final_content)
            
            # Create result
            result = ContentResult(
                id=result_id,
                request_id=request.id,
                content=final_content,
                title=title,
                summary=summary,
                quality_score=quality_analysis["overall_score"],
                quality_level=quality_analysis["quality_level"],
                word_count=word_count,
                reading_time=reading_time,
                keywords_used=self._extract_used_keywords(final_content, request.keywords),
                style_analysis=quality_analysis["style_analysis"],
                seo_score=seo_score,
                readability_score=readability_score,
                generated_at=datetime.now(),
                processing_time=(datetime.now() - start_time).total_seconds(),
                model_used="ai_content_generator_v1.0",
                metadata={
                    "template_used": template.id,
                    "sections_generated": len(generated_sections),
                    "optimization_suggestions": self._get_optimization_suggestions(final_content, request)
                }
            )
            
            # Store result
            self.generated_content[result_id] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating content: {e}")
            raise
    
    def _get_template(self, request: ContentRequest) -> ContentTemplate:
        """Get appropriate template for request"""
        if request.template_id and request.template_id in self.templates:
            return self.templates[request.template_id]
        
        # Find template by content type
        for template in self.templates.values():
            if template.content_type == request.content_type:
                return template
        
        # Default to article template
        return self.templates["article_template"]
    
    def _generate_content_structure(self, request: ContentRequest, template: ContentTemplate) -> List[Dict[str, Any]]:
        """Generate content structure based on template and requirements"""
        structure = []
        
        for section_def in template.structure:
            section = {
                "name": section_def["section"],
                "purpose": section_def["purpose"],
                "min_words": self.content_patterns[request.content_type]["section_requirements"].get(
                    section_def["section"], {}
                ).get("min_words", 50),
                "max_words": self.content_patterns[request.content_type]["section_requirements"].get(
                    section_def["section"], {}
                ).get("max_words", 300),
                "target_words": min(
                    request.length // len(template.structure),
                    self.content_patterns[request.content_type]["section_requirements"].get(
                        section_def["section"], {}
                    ).get("max_words", 300)
                )
            }
            structure.append(section)
        
        return structure
    
    async def _generate_section_content(
        self, 
        section: Dict[str, Any], 
        request: ContentRequest, 
        template: ContentTemplate
    ) -> str:
        """Generate content for a specific section"""
        # This is a simplified version - in practice, this would integrate with AI models
        
        section_templates = {
            "introduction": [
                f"In today's fast-paced world, {request.topic} has become increasingly important.",
                f"Understanding {request.topic} is crucial for {request.target_audience}.",
                f"Let's explore the fascinating world of {request.topic} and its implications."
            ],
            "body": [
                f"When it comes to {request.topic}, there are several key factors to consider.",
                f"Research shows that {request.topic} can significantly impact various aspects of life.",
                f"The implementation of {request.topic} requires careful planning and execution."
            ],
            "conclusion": [
                f"In conclusion, {request.topic} represents a significant opportunity for growth.",
                f"To summarize, {request.topic} offers numerous benefits when properly implemented.",
                f"Ultimately, {request.topic} is an essential component of modern success."
            ],
            "hook": [
                f"Did you know that {request.topic} can change everything?",
                f"Here's what nobody tells you about {request.topic}.",
                f"The secret to mastering {request.topic} lies in understanding these principles."
            ],
            "cta": [
                f"Ready to get started with {request.topic}? Take action today!",
                f"Don't miss out on the benefits of {request.topic}. Start now!",
                f"Transform your approach to {request.topic} with these proven strategies."
            ]
        }
        
        # Generate content based on section type
        base_content = random.choice(section_templates.get(section["name"], ["Content for this section."]))
        
        # Expand content to meet word count requirements
        expanded_content = self._expand_content(
            base_content, 
            section["target_words"], 
            request, 
            template
        )
        
        return expanded_content
    
    def _expand_content(
        self, 
        base_content: str, 
        target_words: int, 
        request: ContentRequest, 
        template: ContentTemplate
    ) -> str:
        """Expand content to meet word count requirements"""
        current_words = len(base_content.split())
        
        if current_words >= target_words:
            return base_content
        
        # Add more content based on style and requirements
        expansion_phrases = {
            WritingStyle.FORMAL: [
                "Furthermore, it is important to note that",
                "In addition to the aforementioned points",
                "It is worth considering that",
                "Moreover, research indicates that",
                "Additionally, studies have shown that"
            ],
            WritingStyle.CASUAL: [
                "What's more,",
                "On top of that,",
                "Another thing to consider is",
                "Plus, you should know that",
                "And here's the thing:"
            ],
            WritingStyle.TECHNICAL: [
                "From a technical perspective,",
                "The implementation requires",
                "The system architecture involves",
                "The configuration parameters include",
                "The performance metrics indicate"
            ],
            WritingStyle.CREATIVE: [
                "Imagine a world where",
                "Picture this scenario:",
                "In a realm of endless possibilities,",
                "The tapestry of life reveals",
                "Like a symphony of interconnected elements,"
            ]
        }
        
        style_phrases = expansion_phrases.get(request.style, expansion_phrases[WritingStyle.CASUAL])
        
        # Add content until we reach target word count
        expanded_content = base_content
        while len(expanded_content.split()) < target_words:
            phrase = random.choice(style_phrases)
            additional_content = f" {phrase} {request.topic} continues to evolve and adapt to changing circumstances."
            expanded_content += additional_content
        
        return expanded_content
    
    def _combine_sections(self, sections: Dict[str, str], template: ContentTemplate) -> str:
        """Combine sections into final content"""
        content_parts = []
        
        for section_def in template.structure:
            section_name = section_def["section"]
            if section_name in sections:
                # Add section heading
                heading = section_name.replace("_", " ").title()
                content_parts.append(f"## {heading}")
                content_parts.append(sections[section_name])
                content_parts.append("")  # Add spacing
        
        return "\n".join(content_parts)
    
    def _generate_title(self, request: ContentRequest, content: str) -> str:
        """Generate title for content"""
        title_templates = {
            ContentType.ARTICLE: [
                f"The Complete Guide to {request.topic}",
                f"Understanding {request.topic}: A Comprehensive Overview",
                f"{request.topic}: Everything You Need to Know"
            ],
            ContentType.BLOG_POST: [
                f"How {request.topic} Changed Everything",
                f"The Truth About {request.topic}",
                f"Why {request.topic} Matters More Than You Think"
            ],
            ContentType.TECHNICAL_DOC: [
                f"{request.topic}: Technical Implementation Guide",
                f"Building with {request.topic}: A Developer's Guide",
                f"{request.topic}: Architecture and Best Practices"
            ],
            ContentType.MARKETING_COPY: [
                f"Transform Your Business with {request.topic}",
                f"The {request.topic} Solution You've Been Waiting For",
                f"Revolutionary {request.topic}: Get Results Fast"
            ]
        }
        
        templates = title_templates.get(request.content_type, title_templates[ContentType.ARTICLE])
        return random.choice(templates)
    
    def _generate_summary(self, content: str) -> str:
        """Generate summary of content"""
        # Extract first few sentences as summary
        sentences = content.split('. ')
        summary_sentences = sentences[:2]  # Take first 2 sentences
        return '. '.join(summary_sentences) + '.'
    
    def _analyze_content_quality(self, content: str, request: ContentRequest) -> Dict[str, Any]:
        """Analyze content quality"""
        # Calculate various quality metrics
        word_count = len(content.split())
        sentence_count = len(content.split('.'))
        avg_sentence_length = word_count / max(sentence_count, 1)
        
        # Check style adherence
        style_guidelines = self.style_guidelines.get(request.style, {})
        style_score = self._calculate_style_score(content, style_guidelines)
        
        # Calculate overall quality score
        overall_score = min(100, (style_score + 70) / 2)  # Simplified calculation
        
        # Determine quality level
        if overall_score >= 90:
            quality_level = ContentQuality.PUBLICATION_READY
        elif overall_score >= 80:
            quality_level = ContentQuality.EXCELLENT
        elif overall_score >= 70:
            quality_level = ContentQuality.GOOD
        else:
            quality_level = ContentQuality.DRAFT
        
        return {
            "overall_score": overall_score,
            "quality_level": quality_level,
            "style_analysis": {
                "style_score": style_score,
                "avg_sentence_length": avg_sentence_length,
                "word_count": word_count,
                "sentence_count": sentence_count
            }
        }
    
    def _calculate_style_score(self, content: str, style_guidelines: Dict[str, Any]) -> float:
        """Calculate how well content adheres to style guidelines"""
        score = 50  # Base score
        
        # Check for contractions (if style doesn't allow them)
        if not style_guidelines.get("contractions", True):
            contractions = content.count("'")
            if contractions == 0:
                score += 10
        
        # Check sentence length
        sentences = content.split('.')
        avg_length = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
        
        if style_guidelines.get("sentence_structure") == "simple" and avg_length < 15:
            score += 10
        elif style_guidelines.get("sentence_structure") == "complex" and avg_length > 15:
            score += 10
        
        return min(100, score)
    
    def _calculate_seo_score(self, content: str, keywords: List[str]) -> float:
        """Calculate SEO score for content"""
        if not keywords:
            return 50  # Neutral score if no keywords
        
        score = 0
        content_lower = content.lower()
        
        # Check keyword density
        total_words = len(content.split())
        for keyword in keywords:
            keyword_count = content_lower.count(keyword.lower())
            density = (keyword_count / total_words) * 100
            if 0.5 <= density <= 2.0:  # Optimal density
                score += 20
            elif density > 0:
                score += 10
        
        # Check for title tag (simplified)
        if any(keyword.lower() in content_lower for keyword in keywords):
            score += 15
        
        # Check for headings (simplified)
        if '##' in content:
            score += 10
        
        return min(100, score)
    
    def _calculate_readability_score(self, content: str) -> float:
        """Calculate readability score (simplified Flesch Reading Ease)"""
        sentences = content.split('.')
        words = content.split()
        
        if not sentences or not words:
            return 50
        
        avg_sentence_length = len(words) / len(sentences)
        avg_syllables = sum(self._count_syllables(word) for word in words) / len(words)
        
        # Simplified Flesch Reading Ease formula
        score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables)
        return max(0, min(100, score))
    
    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word (simplified)"""
        word = word.lower()
        vowels = 'aeiouy'
        syllable_count = 0
        prev_was_vowel = False
        
        for char in word:
            if char in vowels:
                if not prev_was_vowel:
                    syllable_count += 1
                prev_was_vowel = True
            else:
                prev_was_vowel = False
        
        # Handle silent 'e'
        if word.endswith('e') and syllable_count > 1:
            syllable_count -= 1
        
        return max(1, syllable_count)
    
    def _extract_used_keywords(self, content: str, keywords: List[str]) -> List[str]:
        """Extract which keywords were used in content"""
        content_lower = content.lower()
        used_keywords = []
        
        for keyword in keywords:
            if keyword.lower() in content_lower:
                used_keywords.append(keyword)
        
        return used_keywords
    
    def _get_optimization_suggestions(self, content: str, request: ContentRequest) -> List[str]:
        """Get optimization suggestions for content"""
        suggestions = []
        
        # SEO suggestions
        if request.keywords:
            content_lower = content.lower()
            for keyword in request.keywords:
                if keyword.lower() not in content_lower:
                    suggestions.append(f"Consider including the keyword '{keyword}' in your content")
        
        # Readability suggestions
        sentences = content.split('.')
        avg_length = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
        if avg_length > 20:
            suggestions.append("Consider shortening some sentences for better readability")
        
        # Structure suggestions
        if '##' not in content:
            suggestions.append("Add subheadings to improve content structure")
        
        return suggestions
    
    def get_content_result(self, result_id: str) -> Optional[ContentResult]:
        """Get generated content result by ID"""
        return self.generated_content.get(result_id)
    
    def list_generated_content(self) -> List[Dict[str, Any]]:
        """List all generated content"""
        return [
            {
                "id": result.id,
                "title": result.title,
                "content_type": result.request_id,
                "quality_score": result.quality_score,
                "word_count": result.word_count,
                "generated_at": result.generated_at.isoformat()
            }
            for result in self.generated_content.values()
        ]
    
    def get_templates(self) -> List[Dict[str, Any]]:
        """Get available content templates"""
        return [
            {
                "id": template.id,
                "name": template.name,
                "content_type": template.content_type.value,
                "word_count_range": f"{template.word_count_ranges[0]}-{template.word_count_ranges[1]}",
                "required_sections": len(template.required_sections)
            }
            for template in self.templates.values()
        ]
    
    def get_generator_statistics(self) -> Dict[str, Any]:
        """Get content generator statistics"""
        total_content = len(self.generated_content)
        if total_content == 0:
            return {
                "total_content_generated": 0,
                "average_quality_score": 0,
                "average_word_count": 0,
                "content_types": {},
                "templates_available": len(self.templates)
            }
        
        quality_scores = [result.quality_score for result in self.generated_content.values()]
        word_counts = [result.word_count for result in self.generated_content.values()]
        
        # Count by content type
        content_types = {}
        for result in self.generated_content.values():
            # This would need to be stored in the result for accurate counting
            content_types["article"] = content_types.get("article", 0) + 1
        
        return {
            "total_content_generated": total_content,
            "average_quality_score": sum(quality_scores) / len(quality_scores),
            "average_word_count": sum(word_counts) / len(word_counts),
            "content_types": content_types,
            "templates_available": len(self.templates)
        }

# Example usage
if __name__ == "__main__":
    # Initialize content generator
    generator = AIContentGenerator()
    
    # Create content request
    request = ContentRequest(
        id="req_001",
        content_type=ContentType.ARTICLE,
        topic="Artificial Intelligence in Healthcare",
        style=WritingStyle.TECHNICAL,
        length=1000,
        target_audience="healthcare professionals",
        keywords=["AI", "healthcare", "machine learning", "diagnosis"],
        tone="informative"
    )
    
    # Generate content
    result = asyncio.run(generator.generate_content(request))
    
    print("Generated Content:")
    print(f"Title: {result.title}")
    print(f"Quality Score: {result.quality_score}")
    print(f"Word Count: {result.word_count}")
    print(f"SEO Score: {result.seo_score}")
    print(f"Readability Score: {result.readability_score}")
    print(f"Content Preview: {result.content[:200]}...")
    
    # Get statistics
    stats = generator.get_generator_statistics()
    print(f"\nGenerator Statistics:")
    print(f"Total content generated: {stats['total_content_generated']}")
    print(f"Templates available: {stats['templates_available']}")
    
    print("\nAI Content Generator initialized successfully")
