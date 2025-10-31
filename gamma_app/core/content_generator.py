"""
Gamma App - Core Content Generator
Advanced AI-powered content generation with multiple output formats
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import re
from pathlib import Path

import openai
import anthropic
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
from PIL import Image
import requests
from io import BytesIO

logger = logging.getLogger(__name__)

class ContentType(Enum):
    """Types of content that can be generated"""
    PRESENTATION = "presentation"
    DOCUMENT = "document"
    WEB_PAGE = "web_page"
    BLOG_POST = "blog_post"
    SOCIAL_MEDIA = "social_media"
    EMAIL = "email"
    REPORT = "report"
    PROPOSAL = "proposal"

class OutputFormat(Enum):
    """Output formats for generated content"""
    PDF = "pdf"
    PPTX = "pptx"
    HTML = "html"
    DOCX = "docx"
    MD = "markdown"
    JSON = "json"
    PNG = "png"
    JPG = "jpg"

class DesignStyle(Enum):
    """Design styles for content"""
    MODERN = "modern"
    MINIMALIST = "minimalist"
    CORPORATE = "corporate"
    CREATIVE = "creative"
    ACADEMIC = "academic"
    CASUAL = "casual"
    PROFESSIONAL = "professional"

@dataclass
class ContentRequest:
    """Request for content generation"""
    content_type: ContentType
    topic: str
    description: str
    target_audience: str
    length: str  # "short", "medium", "long"
    style: DesignStyle
    output_format: OutputFormat
    include_images: bool = True
    include_charts: bool = False
    language: str = "en"
    tone: str = "professional"
    keywords: List[str] = None
    custom_instructions: str = ""
    user_id: str = ""
    project_id: str = ""

@dataclass
class ContentResponse:
    """Response from content generation"""
    content_id: str
    content_type: ContentType
    title: str
    content: Dict[str, Any]
    metadata: Dict[str, Any]
    generated_at: datetime
    processing_time: float
    quality_score: float
    suggestions: List[str]
    export_urls: Dict[str, str]

@dataclass
class Slide:
    """Individual slide in a presentation"""
    slide_number: int
    title: str
    content: str
    slide_type: str  # "title", "content", "image", "chart", "conclusion"
    layout: str
    design_elements: Dict[str, Any]
    notes: str = ""

@dataclass
class Presentation:
    """Complete presentation structure"""
    title: str
    subtitle: str
    slides: List[Slide]
    theme: str
    color_scheme: Dict[str, str]
    fonts: Dict[str, str]
    metadata: Dict[str, Any]

class ContentGenerator:
    """
    Core content generator using multiple AI models
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the content generator"""
        self.config = config or {}
        self.openai_client = None
        self.anthropic_client = None
        self.local_models = {}
        self.templates = {}
        self.design_systems = {}
        
        # Initialize AI clients
        self._initialize_ai_clients()
        
        # Load templates and design systems
        self._load_templates()
        self._load_design_systems()
        
        logger.info("Content Generator initialized successfully")

    def _initialize_ai_clients(self):
        """Initialize AI model clients"""
        try:
            # OpenAI
            if self.config.get('openai_api_key'):
                self.openai_client = openai.OpenAI(
                    api_key=self.config['openai_api_key']
                )
                logger.info("OpenAI client initialized")
            
            # Anthropic
            if self.config.get('anthropic_api_key'):
                self.anthropic_client = anthropic.Anthropic(
                    api_key=self.config['anthropic_api_key']
                )
                logger.info("Anthropic client initialized")
            
            # Local models
            self._load_local_models()
            
        except Exception as e:
            logger.error(f"Error initializing AI clients: {e}")

    def _load_local_models(self):
        """Load local transformer models"""
        try:
            # Text generation model
            self.local_models['text_generator'] = pipeline(
                "text-generation",
                model="microsoft/DialoGPT-medium",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            # Summarization model
            self.local_models['summarizer'] = pipeline(
                "summarization",
                model="facebook/bart-large-cnn"
            )
            
            logger.info("Local models loaded successfully")
            
        except Exception as e:
            logger.warning(f"Could not load local models: {e}")

    def _load_templates(self):
        """Load content templates"""
        self.templates = {
            ContentType.PRESENTATION: {
                "business": {
                    "slide_count": 10,
                    "sections": ["title", "agenda", "problem", "solution", "benefits", "implementation", "timeline", "budget", "team", "conclusion"],
                    "layout": "corporate"
                },
                "academic": {
                    "slide_count": 15,
                    "sections": ["title", "abstract", "introduction", "literature", "methodology", "results", "analysis", "discussion", "conclusion", "references"],
                    "layout": "academic"
                },
                "creative": {
                    "slide_count": 8,
                    "sections": ["title", "concept", "inspiration", "process", "examples", "impact", "next_steps", "thank_you"],
                    "layout": "creative"
                }
            },
            ContentType.DOCUMENT: {
                "report": {
                    "sections": ["executive_summary", "introduction", "methodology", "findings", "analysis", "recommendations", "conclusion", "appendix"],
                    "length": "long"
                },
                "proposal": {
                    "sections": ["title", "executive_summary", "problem_statement", "proposed_solution", "methodology", "timeline", "budget", "team", "next_steps"],
                    "length": "medium"
                }
            }
        }

    def _load_design_systems(self):
        """Load design systems and themes"""
        self.design_systems = {
            DesignStyle.MODERN: {
                "colors": {
                    "primary": "#2563eb",
                    "secondary": "#64748b",
                    "accent": "#f59e0b",
                    "background": "#ffffff",
                    "text": "#1e293b"
                },
                "fonts": {
                    "heading": "Inter",
                    "body": "Inter",
                    "monospace": "JetBrains Mono"
                },
                "spacing": "generous",
                "border_radius": "8px"
            },
            DesignStyle.MINIMALIST: {
                "colors": {
                    "primary": "#000000",
                    "secondary": "#666666",
                    "accent": "#000000",
                    "background": "#ffffff",
                    "text": "#000000"
                },
                "fonts": {
                    "heading": "Helvetica",
                    "body": "Helvetica",
                    "monospace": "Monaco"
                },
                "spacing": "minimal",
                "border_radius": "0px"
            },
            DesignStyle.CORPORATE: {
                "colors": {
                    "primary": "#1e40af",
                    "secondary": "#374151",
                    "accent": "#dc2626",
                    "background": "#f9fafb",
                    "text": "#111827"
                },
                "fonts": {
                    "heading": "Arial",
                    "body": "Arial",
                    "monospace": "Courier New"
                },
                "spacing": "standard",
                "border_radius": "4px"
            }
        }

    async def generate_content(self, request: ContentRequest) -> ContentResponse:
        """Generate content based on the request"""
        start_time = datetime.now()
        
        try:
            # Generate content based on type
            if request.content_type == ContentType.PRESENTATION:
                content = await self._generate_presentation(request)
            elif request.content_type == ContentType.DOCUMENT:
                content = await self._generate_document(request)
            elif request.content_type == ContentType.WEB_PAGE:
                content = await self._generate_web_page(request)
            else:
                content = await self._generate_generic_content(request)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Generate content ID
            content_id = self._generate_content_id(request, content)
            
            # Calculate quality score
            quality_score = self._calculate_quality_score(content, request)
            
            # Generate suggestions
            suggestions = self._generate_suggestions(content, request)
            
            # Create export URLs
            export_urls = await self._create_export_urls(content, request)
            
            response = ContentResponse(
                content_id=content_id,
                content_type=request.content_type,
                title=content.get('title', 'Generated Content'),
                content=content,
                metadata={
                    'user_id': request.user_id,
                    'project_id': request.project_id,
                    'generation_model': self._get_used_model(),
                    'word_count': self._count_words(content),
                    'character_count': self._count_characters(content)
                },
                generated_at=datetime.now(),
                processing_time=processing_time,
                quality_score=quality_score,
                suggestions=suggestions,
                export_urls=export_urls
            )
            
            logger.info(f"Content generated successfully: {content_id}")
            return response
            
        except Exception as e:
            logger.error(f"Error generating content: {e}")
            raise

    async def _generate_presentation(self, request: ContentRequest) -> Dict[str, Any]:
        """Generate a presentation"""
        # Get template
        template = self.templates[ContentType.PRESENTATION].get(
            request.style.value, 
            self.templates[ContentType.PRESENTATION]['business']
        )
        
        # Generate content for each slide
        slides = []
        for i, section in enumerate(template['sections']):
            slide_content = await self._generate_slide_content(
                section, request, i + 1
            )
            
            slide = Slide(
                slide_number=i + 1,
                title=slide_content['title'],
                content=slide_content['content'],
                slide_type=slide_content['type'],
                layout=slide_content['layout'],
                design_elements=slide_content['design_elements'],
                notes=slide_content.get('notes', '')
            )
            slides.append(slide)
        
        # Create presentation
        presentation = Presentation(
            title=request.topic,
            subtitle=request.description,
            slides=slides,
            theme=request.style.value,
            color_scheme=self.design_systems[request.style]['colors'],
            fonts=self.design_systems[request.style]['fonts'],
            metadata={
                'target_audience': request.target_audience,
                'language': request.language,
                'tone': request.tone
            }
        )
        
        return asdict(presentation)

    async def _generate_document(self, request: ContentRequest) -> Dict[str, Any]:
        """Generate a document"""
        template = self.templates[ContentType.DOCUMENT].get(
            request.style.value,
            self.templates[ContentType.DOCUMENT]['report']
        )
        
        document = {
            'title': request.topic,
            'subtitle': request.description,
            'sections': [],
            'metadata': {
                'target_audience': request.target_audience,
                'language': request.language,
                'tone': request.tone,
                'style': request.style.value
            }
        }
        
        # Generate content for each section
        for section_name in template['sections']:
            section_content = await self._generate_section_content(
                section_name, request
            )
            document['sections'].append(section_content)
        
        return document

    async def _generate_web_page(self, request: ContentRequest) -> Dict[str, Any]:
        """Generate a web page"""
        web_page = {
            'title': request.topic,
            'description': request.description,
            'sections': [],
            'metadata': {
                'target_audience': request.target_audience,
                'language': request.language,
                'tone': request.tone,
                'style': request.style.value
            }
        }
        
        # Generate main sections
        sections = ['hero', 'about', 'features', 'testimonials', 'contact']
        for section_name in sections:
            section_content = await self._generate_web_section_content(
                section_name, request
            )
            web_page['sections'].append(section_content)
        
        return web_page

    async def _generate_generic_content(self, request: ContentRequest) -> Dict[str, Any]:
        """Generate generic content"""
        prompt = self._build_content_prompt(request)
        
        if self.openai_client:
            response = await self._generate_with_openai(prompt, request)
        elif self.anthropic_client:
            response = await self._generate_with_anthropic(prompt, request)
        else:
            response = await self._generate_with_local_model(prompt, request)
        
        return {
            'title': request.topic,
            'content': response,
            'metadata': {
                'target_audience': request.target_audience,
                'language': request.language,
                'tone': request.tone,
                'style': request.style.value
            }
        }

    async def _generate_slide_content(self, section: str, request: ContentRequest, slide_number: int) -> Dict[str, Any]:
        """Generate content for a specific slide"""
        prompt = f"""
        Generate content for slide {slide_number} of a {request.content_type.value} presentation.
        
        Section: {section}
        Topic: {request.topic}
        Description: {request.description}
        Target Audience: {request.target_audience}
        Style: {request.style.value}
        Tone: {request.tone}
        
        Provide:
        1. A compelling title (max 8 words)
        2. Main content (3-5 bullet points or 2-3 sentences)
        3. Slide type (title, content, image, chart, conclusion)
        4. Layout suggestion
        5. Design elements (colors, fonts, spacing)
        6. Speaker notes (optional)
        
        Format as JSON.
        """
        
        if self.openai_client:
            response = await self._generate_with_openai(prompt, request)
        else:
            response = await self._generate_with_local_model(prompt, request)
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            return {
                'title': f"{section.title()}",
                'content': response,
                'type': 'content',
                'layout': 'standard',
                'design_elements': {},
                'notes': ''
            }

    async def _generate_section_content(self, section: str, request: ContentRequest) -> Dict[str, Any]:
        """Generate content for a document section"""
        prompt = f"""
        Generate content for the "{section}" section of a {request.content_type.value}.
        
        Topic: {request.topic}
        Description: {request.description}
        Target Audience: {request.target_audience}
        Style: {request.style.value}
        Tone: {request.tone}
        Length: {request.length}
        
        Provide comprehensive, well-structured content for this section.
        """
        
        if self.openai_client:
            content = await self._generate_with_openai(prompt, request)
        else:
            content = await self._generate_with_local_model(prompt, request)
        
        return {
            'section_name': section,
            'content': content,
            'word_count': len(content.split())
        }

    async def _generate_web_section_content(self, section: str, request: ContentRequest) -> Dict[str, Any]:
        """Generate content for a web page section"""
        prompt = f"""
        Generate content for the "{section}" section of a web page.
        
        Topic: {request.topic}
        Description: {request.description}
        Target Audience: {request.target_audience}
        Style: {request.style.value}
        Tone: {request.tone}
        
        Make it engaging and web-optimized.
        """
        
        if self.openai_client:
            content = await self._generate_with_openai(prompt, request)
        else:
            content = await self._generate_with_local_model(prompt, request)
        
        return {
            'section_name': section,
            'content': content,
            'html_content': self._convert_to_html(content, section)
        }

    async def _generate_with_openai(self, prompt: str, request: ContentRequest) -> str:
        """Generate content using OpenAI"""
        try:
            response = self.openai_client.chat.completions.create(
                model=self.config.get('openai_model', 'gpt-4'),
                messages=[
                    {"role": "system", "content": "You are an expert content creator specializing in creating engaging, professional content."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self._get_max_tokens(request.length),
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI generation error: {e}")
            raise

    async def _generate_with_anthropic(self, prompt: str, request: ContentRequest) -> str:
        """Generate content using Anthropic"""
        try:
            response = self.anthropic_client.messages.create(
                model=self.config.get('anthropic_model', 'claude-3-sonnet-20240229'),
                max_tokens=self._get_max_tokens(request.length),
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Anthropic generation error: {e}")
            raise

    async def _generate_with_local_model(self, prompt: str, request: ContentRequest) -> str:
        """Generate content using local models"""
        try:
            if 'text_generator' in self.local_models:
                result = self.local_models['text_generator'](
                    prompt,
                    max_length=self._get_max_tokens(request.length),
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True
                )
                return result[0]['generated_text']
            else:
                return "Content generation not available. Please configure AI API keys."
        except Exception as e:
            logger.error(f"Local model generation error: {e}")
            return "Error generating content with local models."

    def _build_content_prompt(self, request: ContentRequest) -> str:
        """Build a comprehensive prompt for content generation"""
        return f"""
        Create a {request.content_type.value} about: {request.topic}
        
        Description: {request.description}
        Target Audience: {request.target_audience}
        Style: {request.style.value}
        Tone: {request.tone}
        Length: {request.length}
        Language: {request.language}
        
        Custom Instructions: {request.custom_instructions}
        
        Keywords: {', '.join(request.keywords) if request.keywords else 'None specified'}
        
        Please create engaging, well-structured content that meets these requirements.
        """

    def _get_max_tokens(self, length: str) -> int:
        """Get maximum tokens based on content length"""
        token_map = {
            'short': 500,
            'medium': 1000,
            'long': 2000
        }
        return token_map.get(length, 1000)

    def _get_used_model(self) -> str:
        """Get the model used for generation"""
        if self.openai_client:
            return 'openai'
        elif self.anthropic_client:
            return 'anthropic'
        else:
            return 'local'

    def _count_words(self, content: Dict[str, Any]) -> int:
        """Count words in generated content"""
        if isinstance(content, dict):
            text = ' '.join(str(v) for v in content.values() if isinstance(v, str))
        else:
            text = str(content)
        return len(text.split())

    def _count_characters(self, content: Dict[str, Any]) -> int:
        """Count characters in generated content"""
        if isinstance(content, dict):
            text = ' '.join(str(v) for v in content.values() if isinstance(v, str))
        else:
            text = str(content)
        return len(text)

    def _calculate_quality_score(self, content: Dict[str, Any], request: ContentRequest) -> float:
        """Calculate quality score for generated content"""
        # Simple quality scoring based on content length, structure, etc.
        word_count = self._count_words(content)
        
        # Base score
        score = 0.5
        
        # Length appropriateness
        if request.length == 'short' and 100 <= word_count <= 500:
            score += 0.2
        elif request.length == 'medium' and 500 <= word_count <= 1000:
            score += 0.2
        elif request.length == 'long' and word_count >= 1000:
            score += 0.2
        
        # Structure bonus
        if isinstance(content, dict) and len(content) > 3:
            score += 0.2
        
        # Content completeness
        if word_count > 50:
            score += 0.1
        
        return min(1.0, score)

    def _generate_suggestions(self, content: Dict[str, Any], request: ContentRequest) -> List[str]:
        """Generate improvement suggestions"""
        suggestions = []
        
        word_count = self._count_words(content)
        
        if word_count < 100:
            suggestions.append("Consider adding more detail to make the content more comprehensive")
        
        if request.include_images and not self._has_image_placeholders(content):
            suggestions.append("Add relevant images to enhance visual appeal")
        
        if request.include_charts and not self._has_chart_placeholders(content):
            suggestions.append("Include charts or graphs to support your data")
        
        if request.content_type == ContentType.PRESENTATION and word_count > 2000:
            suggestions.append("Consider reducing text per slide for better readability")
        
        return suggestions

    def _has_image_placeholders(self, content: Dict[str, Any]) -> bool:
        """Check if content has image placeholders"""
        content_str = str(content).lower()
        return any(keyword in content_str for keyword in ['image', 'photo', 'picture', 'visual'])

    def _has_chart_placeholders(self, content: Dict[str, Any]) -> bool:
        """Check if content has chart placeholders"""
        content_str = str(content).lower()
        return any(keyword in content_str for keyword in ['chart', 'graph', 'diagram', 'data visualization'])

    async def _create_export_urls(self, content: Dict[str, Any], request: ContentRequest) -> Dict[str, str]:
        """Create export URLs for different formats"""
        # This would integrate with the export engines
        return {
            'pdf': f"/api/export/{request.content_type.value}/pdf",
            'pptx': f"/api/export/{request.content_type.value}/pptx",
            'html': f"/api/export/{request.content_type.value}/html"
        }

    def _convert_to_html(self, content: str, section: str) -> str:
        """Convert content to HTML"""
        # Simple HTML conversion
        html_content = f"<div class='{section}'>"
        html_content += f"<h2>{section.title()}</h2>"
        html_content += f"<p>{content}</p>"
        html_content += "</div>"
        return html_content

    def _generate_content_id(self, request: ContentRequest, content: Dict[str, Any]) -> str:
        """Generate unique content ID"""
        content_str = f"{request.topic}_{request.content_type.value}_{datetime.now()}"
        return hashlib.md5(content_str.encode()).hexdigest()[:12]

    async def enhance_content(self, content_id: str, enhancement_type: str, 
                            instructions: str) -> ContentResponse:
        """Enhance existing content"""
        # This would load existing content and apply enhancements
        # Implementation would depend on storage system
        pass

    async def get_content_suggestions(self, content_id: str) -> List[str]:
        """Get suggestions for improving content"""
        # This would analyze existing content and provide suggestions
        # Implementation would depend on storage system
        pass

    def get_supported_formats(self, content_type: ContentType) -> List[OutputFormat]:
        """Get supported output formats for content type"""
        format_map = {
            ContentType.PRESENTATION: [OutputFormat.PPTX, OutputFormat.PDF, OutputFormat.HTML],
            ContentType.DOCUMENT: [OutputFormat.DOCX, OutputFormat.PDF, OutputFormat.HTML, OutputFormat.MD],
            ContentType.WEB_PAGE: [OutputFormat.HTML, OutputFormat.PDF],
            ContentType.BLOG_POST: [OutputFormat.HTML, OutputFormat.MD, OutputFormat.PDF],
            ContentType.SOCIAL_MEDIA: [OutputFormat.HTML, OutputFormat.PNG, OutputFormat.JPG],
            ContentType.EMAIL: [OutputFormat.HTML, OutputFormat.MD],
            ContentType.REPORT: [OutputFormat.DOCX, OutputFormat.PDF, OutputFormat.HTML],
            ContentType.PROPOSAL: [OutputFormat.DOCX, OutputFormat.PDF, OutputFormat.HTML]
        }
        return format_map.get(content_type, [OutputFormat.HTML])

    def get_available_templates(self, content_type: ContentType) -> List[str]:
        """Get available templates for content type"""
        if content_type in self.templates:
            return list(self.templates[content_type].keys())
        return []

    def get_available_styles(self) -> List[DesignStyle]:
        """Get available design styles"""
        return list(DesignStyle)

    def get_available_languages(self) -> List[str]:
        """Get available languages"""
        return ['en', 'es', 'fr', 'de', 'it', 'pt', 'zh', 'ja', 'ko', 'ar']

    def get_available_tones(self) -> List[str]:
        """Get available tones"""
        return [
            'professional', 'casual', 'friendly', 'formal', 'persuasive',
            'informative', 'conversational', 'authoritative', 'creative', 'technical'
        ]



























