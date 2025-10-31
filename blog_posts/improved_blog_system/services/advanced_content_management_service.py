"""
Advanced Content Management Service for intelligent content operations
"""

import asyncio
import json
import re
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_, desc, text
from dataclasses import dataclass
from enum import Enum
import hashlib
import uuid
from PIL import Image
import requests
from bs4 import BeautifulSoup
import markdown
from markdown.extensions import codehilite, fenced_code, tables, toc
import bleach
from slugify import slugify
import jieba
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textstat import flesch_reading_ease, flesch_kincaid_grade
from langdetect import detect, DetectorFactory
import spacy
from sentence_transformers import SentenceTransformer

from ..models.database import ContentVersion, ContentTemplate, ContentCategory, ContentTag, ContentAnalytics
from ..core.exceptions import DatabaseError, ValidationError


class ContentStatus(Enum):
    """Content status enumeration."""
    DRAFT = "draft"
    PUBLISHED = "published"
    ARCHIVED = "archived"
    SCHEDULED = "scheduled"
    REVIEW = "review"
    REJECTED = "rejected"


class ContentType(Enum):
    """Content type enumeration."""
    BLOG_POST = "blog_post"
    ARTICLE = "article"
    TUTORIAL = "tutorial"
    NEWS = "news"
    REVIEW = "review"
    INTERVIEW = "interview"
    CASE_STUDY = "case_study"
    WHITEPAPER = "whitepaper"


class ContentPriority(Enum):
    """Content priority enumeration."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


@dataclass
class ContentMetadata:
    """Content metadata structure."""
    title: str
    description: str
    keywords: List[str]
    category: str
    tags: List[str]
    author: str
    reading_time: int
    word_count: int
    language: str
    difficulty_level: str
    seo_score: float
    readability_score: float


@dataclass
class ContentTemplate:
    """Content template structure."""
    name: str
    type: ContentType
    structure: Dict[str, Any]
    fields: List[Dict[str, Any]]
    styling: Dict[str, Any]
    validation_rules: Dict[str, Any]


class AdvancedContentManagementService:
    """Service for advanced content management operations."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
        self.nlp = None
        self.sentence_model = None
        self.content_templates = {}
        self.content_cache = {}
        self._initialize_nlp()
        self._initialize_templates()
    
    def _initialize_nlp(self):
        """Initialize NLP models."""
        try:
            # Initialize spaCy for advanced NLP
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                # Fallback to basic model
                self.nlp = None
            
            # Initialize sentence transformer for content similarity
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Download NLTK data
            try:
                nltk.download('stopwords', quiet=True)
                nltk.download('punkt', quiet=True)
                nltk.download('averaged_perceptron_tagger', quiet=True)
            except:
                pass
                
        except Exception as e:
            print(f"Warning: Could not initialize NLP models: {e}")
    
    def _initialize_templates(self):
        """Initialize content templates."""
        self.content_templates = {
            "blog_post": {
                "name": "Blog Post",
                "type": ContentType.BLOG_POST,
                "structure": {
                    "title": {"required": True, "max_length": 200},
                    "excerpt": {"required": True, "max_length": 500},
                    "content": {"required": True, "min_length": 100},
                    "featured_image": {"required": False},
                    "tags": {"required": True, "min_count": 1},
                    "category": {"required": True}
                },
                "fields": [
                    {"name": "title", "type": "text", "label": "Title"},
                    {"name": "excerpt", "type": "textarea", "label": "Excerpt"},
                    {"name": "content", "type": "richtext", "label": "Content"},
                    {"name": "featured_image", "type": "image", "label": "Featured Image"},
                    {"name": "tags", "type": "tags", "label": "Tags"},
                    {"name": "category", "type": "select", "label": "Category"}
                ],
                "styling": {
                    "font_family": "Arial, sans-serif",
                    "font_size": "16px",
                    "line_height": "1.6",
                    "max_width": "800px"
                }
            },
            "tutorial": {
                "name": "Tutorial",
                "type": ContentType.TUTORIAL,
                "structure": {
                    "title": {"required": True, "max_length": 200},
                    "description": {"required": True, "max_length": 500},
                    "content": {"required": True, "min_length": 500},
                    "difficulty": {"required": True},
                    "estimated_time": {"required": True},
                    "prerequisites": {"required": False},
                    "tags": {"required": True, "min_count": 1}
                },
                "fields": [
                    {"name": "title", "type": "text", "label": "Title"},
                    {"name": "description", "type": "textarea", "label": "Description"},
                    {"name": "content", "type": "richtext", "label": "Content"},
                    {"name": "difficulty", "type": "select", "label": "Difficulty", "options": ["Beginner", "Intermediate", "Advanced"]},
                    {"name": "estimated_time", "type": "number", "label": "Estimated Time (minutes)"},
                    {"name": "prerequisites", "type": "textarea", "label": "Prerequisites"},
                    {"name": "tags", "type": "tags", "label": "Tags"}
                ],
                "styling": {
                    "font_family": "Georgia, serif",
                    "font_size": "18px",
                    "line_height": "1.8",
                    "max_width": "900px"
                }
            },
            "news_article": {
                "name": "News Article",
                "type": ContentType.NEWS,
                "structure": {
                    "headline": {"required": True, "max_length": 150},
                    "subheadline": {"required": False, "max_length": 300},
                    "content": {"required": True, "min_length": 200},
                    "source": {"required": True},
                    "publish_date": {"required": True},
                    "tags": {"required": True, "min_count": 1}
                },
                "fields": [
                    {"name": "headline", "type": "text", "label": "Headline"},
                    {"name": "subheadline", "type": "text", "label": "Subheadline"},
                    {"name": "content", "type": "richtext", "label": "Content"},
                    {"name": "source", "type": "text", "label": "Source"},
                    {"name": "publish_date", "type": "datetime", "label": "Publish Date"},
                    {"name": "tags", "type": "tags", "label": "Tags"}
                ],
                "styling": {
                    "font_family": "Times New Roman, serif",
                    "font_size": "17px",
                    "line_height": "1.7",
                    "max_width": "700px"
                }
            }
        }
    
    async def create_content(
        self,
        title: str,
        content: str,
        content_type: ContentType = ContentType.BLOG_POST,
        author_id: str = None,
        template_name: str = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create new content with advanced processing."""
        try:
            # Validate content
            validation_result = await self._validate_content(title, content, content_type, template_name)
            if not validation_result["valid"]:
                raise ValidationError(f"Content validation failed: {validation_result['errors']}")
            
            # Process content
            processed_content = await self._process_content(content)
            
            # Extract metadata
            content_metadata = await self._extract_metadata(title, processed_content)
            
            # Generate slug
            slug = self._generate_slug(title)
            
            # Calculate reading time
            reading_time = self._calculate_reading_time(processed_content)
            
            # Generate content ID
            content_id = str(uuid.uuid4())
            
            # Create content version
            content_version = ContentVersion(
                content_id=content_id,
                title=title,
                content=processed_content,
                slug=slug,
                content_type=content_type.value,
                author_id=author_id,
                status=ContentStatus.DRAFT.value,
                metadata=content_metadata.__dict__,
                reading_time=reading_time,
                word_count=content_metadata.word_count,
                language=content_metadata.language,
                seo_score=content_metadata.seo_score,
                readability_score=content_metadata.readability_score,
                created_at=datetime.utcnow()
            )
            
            self.session.add(content_version)
            await self.session.commit()
            
            # Cache content
            self.content_cache[content_id] = {
                "content": processed_content,
                "metadata": content_metadata.__dict__,
                "timestamp": datetime.utcnow()
            }
            
            return {
                "success": True,
                "content_id": content_id,
                "slug": slug,
                "metadata": content_metadata.__dict__,
                "reading_time": reading_time,
                "message": "Content created successfully"
            }
            
        except Exception as e:
            await self.session.rollback()
            raise DatabaseError(f"Failed to create content: {str(e)}")
    
    async def _validate_content(
        self,
        title: str,
        content: str,
        content_type: ContentType,
        template_name: str = None
    ) -> Dict[str, Any]:
        """Validate content against template rules."""
        try:
            errors = []
            
            # Get template
            template = None
            if template_name and template_name in self.content_templates:
                template = self.content_templates[template_name]
            else:
                # Use default template for content type
                template_key = content_type.value
                if template_key in self.content_templates:
                    template = self.content_templates[template_key]
            
            if template:
                structure = template["structure"]
                
                # Validate title
                if "title" in structure:
                    title_rules = structure["title"]
                    if title_rules.get("required", False) and not title:
                        errors.append("Title is required")
                    if title_rules.get("max_length") and len(title) > title_rules["max_length"]:
                        errors.append(f"Title exceeds maximum length of {title_rules['max_length']} characters")
                
                # Validate content
                if "content" in structure:
                    content_rules = structure["content"]
                    if content_rules.get("required", False) and not content:
                        errors.append("Content is required")
                    if content_rules.get("min_length") and len(content) < content_rules["min_length"]:
                        errors.append(f"Content must be at least {content_rules['min_length']} characters")
            
            # Basic validation
            if not title or len(title.strip()) == 0:
                errors.append("Title cannot be empty")
            
            if not content or len(content.strip()) == 0:
                errors.append("Content cannot be empty")
            
            if len(title) > 500:
                errors.append("Title is too long (maximum 500 characters)")
            
            if len(content) < 50:
                errors.append("Content is too short (minimum 50 characters)")
            
            return {
                "valid": len(errors) == 0,
                "errors": errors
            }
            
        except Exception as e:
            return {
                "valid": False,
                "errors": [f"Validation error: {str(e)}"]
            }
    
    async def _process_content(self, content: str) -> str:
        """Process content for optimization and formatting."""
        try:
            # Convert markdown to HTML if needed
            if self._is_markdown(content):
                content = self._markdown_to_html(content)
            
            # Clean HTML
            content = self._clean_html(content)
            
            # Optimize images
            content = await self._optimize_images(content)
            
            # Add table of contents
            content = self._add_table_of_contents(content)
            
            # Optimize for SEO
            content = self._optimize_for_seo(content)
            
            return content
            
        except Exception as e:
            return content  # Return original content if processing fails
    
    def _is_markdown(self, content: str) -> bool:
        """Check if content is in markdown format."""
        markdown_indicators = [
            "# ", "## ", "### ", "#### ", "##### ", "###### ",
            "**", "__", "*", "_", "`", "```", "> ", "- ", "* ",
            "[", "]", "(", ")", "|", "---", "==="
        ]
        
        return any(indicator in content for indicator in markdown_indicators)
    
    def _markdown_to_html(self, content: str) -> str:
        """Convert markdown to HTML."""
        try:
            md = markdown.Markdown(
                extensions=[
                    'codehilite',
                    'fenced_code',
                    'tables',
                    'toc',
                    'nl2br',
                    'attr_list'
                ]
            )
            return md.convert(content)
        except Exception as e:
            return content
    
    def _clean_html(self, content: str) -> str:
        """Clean and sanitize HTML content."""
        try:
            # Allowed tags and attributes
            allowed_tags = [
                'p', 'br', 'strong', 'em', 'u', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
                'ul', 'ol', 'li', 'blockquote', 'pre', 'code', 'a', 'img', 'table',
                'thead', 'tbody', 'tr', 'th', 'td', 'div', 'span', 'hr'
            ]
            
            allowed_attributes = {
                'a': ['href', 'title', 'target'],
                'img': ['src', 'alt', 'title', 'width', 'height'],
                'table': ['class'],
                'th': ['class'],
                'td': ['class'],
                'div': ['class', 'id'],
                'span': ['class', 'id']
            }
            
            return bleach.clean(content, tags=allowed_tags, attributes=allowed_attributes)
            
        except Exception as e:
            return content
    
    async def _optimize_images(self, content: str) -> str:
        """Optimize images in content."""
        try:
            # This would implement image optimization
            # For now, just return the content as-is
            return content
        except Exception as e:
            return content
    
    def _add_table_of_contents(self, content: str) -> str:
        """Add table of contents to content."""
        try:
            # Extract headings
            headings = re.findall(r'<h([1-6])[^>]*>(.*?)</h[1-6]>', content, re.IGNORECASE)
            
            if len(headings) > 2:  # Only add TOC if there are more than 2 headings
                toc_html = '<div class="table-of-contents"><h3>Table of Contents</h3><ul>'
                
                for level, heading in headings:
                    heading_text = re.sub(r'<[^>]+>', '', heading)  # Remove HTML tags
                    anchor = slugify(heading_text)
                    toc_html += f'<li class="toc-level-{level}"><a href="#{anchor}">{heading_text}</a></li>'
                
                toc_html += '</ul></div>'
                
                # Insert TOC after first heading
                first_heading = re.search(r'<h[1-6][^>]*>.*?</h[1-6]>', content, re.IGNORECASE)
                if first_heading:
                    insert_pos = first_heading.end()
                    content = content[:insert_pos] + toc_html + content[insert_pos:]
            
            return content
            
        except Exception as e:
            return content
    
    def _optimize_for_seo(self, content: str) -> str:
        """Optimize content for SEO."""
        try:
            # Add alt attributes to images without them
            content = re.sub(
                r'<img([^>]*?)(?<!alt="[^"]*")>',
                r'<img\1 alt="Image">',
                content
            )
            
            # Ensure proper heading hierarchy
            # This would implement more sophisticated SEO optimization
            
            return content
            
        except Exception as e:
            return content
    
    async def _extract_metadata(
        self,
        title: str,
        content: str
    ) -> ContentMetadata:
        """Extract metadata from content."""
        try:
            # Clean content for analysis
            clean_content = re.sub(r'<[^>]+>', '', content)  # Remove HTML tags
            
            # Extract keywords
            keywords = self._extract_keywords(title + " " + clean_content)
            
            # Detect language
            try:
                language = detect(clean_content)
            except:
                language = "en"
            
            # Calculate readability
            readability_score = self._calculate_readability(clean_content)
            
            # Calculate SEO score
            seo_score = self._calculate_seo_score(title, clean_content, keywords)
            
            # Determine difficulty level
            difficulty_level = self._determine_difficulty_level(readability_score)
            
            # Count words
            word_count = len(clean_content.split())
            
            # Calculate reading time
            reading_time = self._calculate_reading_time(clean_content)
            
            return ContentMetadata(
                title=title,
                description=clean_content[:200] + "..." if len(clean_content) > 200 else clean_content,
                keywords=keywords,
                category="",  # Would be determined by classification
                tags=[],  # Would be extracted or suggested
                author="",  # Would be provided
                reading_time=reading_time,
                word_count=word_count,
                language=language,
                difficulty_level=difficulty_level,
                seo_score=seo_score,
                readability_score=readability_score
            )
            
        except Exception as e:
            # Return default metadata if extraction fails
            return ContentMetadata(
                title=title,
                description="",
                keywords=[],
                category="",
                tags=[],
                author="",
                reading_time=1,
                word_count=len(content.split()),
                language="en",
                difficulty_level="intermediate",
                seo_score=0.0,
                readability_score=0.0
            )
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text."""
        try:
            if not self.nlp:
                # Fallback to simple keyword extraction
                words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
                word_freq = {}
                for word in words:
                    word_freq[word] = word_freq.get(word, 0) + 1
                return sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
            
            # Use spaCy for advanced keyword extraction
            doc = self.nlp(text)
            
            # Extract named entities
            entities = [ent.text for ent in doc.ents if ent.label_ in ["PERSON", "ORG", "GPE", "PRODUCT"]]
            
            # Extract noun phrases
            noun_phrases = [chunk.text for chunk in doc.noun_chunks]
            
            # Extract important words (nouns, adjectives)
            important_words = [token.text for token in doc if token.pos_ in ["NOUN", "ADJ"] and not token.is_stop]
            
            # Combine and deduplicate
            keywords = list(set(entities + noun_phrases + important_words))
            
            return keywords[:20]  # Return top 20 keywords
            
        except Exception as e:
            return []
    
    def _calculate_readability(self, text: str) -> float:
        """Calculate readability score."""
        try:
            # Use Flesch Reading Ease
            score = flesch_reading_ease(text)
            return max(0, min(100, score))  # Clamp between 0 and 100
        except Exception as e:
            return 50.0  # Default score
    
    def _calculate_seo_score(
        self,
        title: str,
        content: str,
        keywords: List[str]
    ) -> float:
        """Calculate SEO score."""
        try:
            score = 0.0
            
            # Title length (optimal: 50-60 characters)
            title_length = len(title)
            if 50 <= title_length <= 60:
                score += 20
            elif 40 <= title_length <= 70:
                score += 15
            else:
                score += 10
            
            # Content length (optimal: 300+ words)
            word_count = len(content.split())
            if word_count >= 300:
                score += 20
            elif word_count >= 150:
                score += 15
            else:
                score += 10
            
            # Keyword density (optimal: 1-3%)
            if keywords:
                keyword_density = sum(content.lower().count(keyword.lower()) for keyword in keywords[:5]) / word_count * 100
                if 1 <= keyword_density <= 3:
                    score += 20
                elif 0.5 <= keyword_density <= 5:
                    score += 15
                else:
                    score += 10
            
            # Heading structure
            headings = re.findall(r'<h[1-6][^>]*>', content, re.IGNORECASE)
            if len(headings) >= 2:
                score += 15
            
            # Internal links (would check for internal links)
            score += 10
            
            # Meta description (would check for meta description)
            score += 15
            
            return min(100, score)
            
        except Exception as e:
            return 50.0
    
    def _determine_difficulty_level(self, readability_score: float) -> str:
        """Determine content difficulty level based on readability score."""
        if readability_score >= 80:
            return "beginner"
        elif readability_score >= 60:
            return "intermediate"
        elif readability_score >= 40:
            return "advanced"
        else:
            return "expert"
    
    def _calculate_reading_time(self, content: str) -> int:
        """Calculate estimated reading time in minutes."""
        try:
            # Average reading speed: 200-250 words per minute
            word_count = len(content.split())
            reading_time = max(1, round(word_count / 225))  # 225 words per minute
            return reading_time
        except Exception as e:
            return 1
    
    def _generate_slug(self, title: str) -> str:
        """Generate URL-friendly slug from title."""
        try:
            return slugify(title)
        except Exception as e:
            # Fallback slug generation
            slug = re.sub(r'[^a-zA-Z0-9\s-]', '', title.lower())
            slug = re.sub(r'\s+', '-', slug.strip())
            return slug[:50]  # Limit length
    
    async def update_content(
        self,
        content_id: str,
        title: Optional[str] = None,
        content: Optional[str] = None,
        status: Optional[ContentStatus] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Update existing content."""
        try:
            # Get existing content
            content_query = select(ContentVersion).where(ContentVersion.content_id == content_id)
            content_result = await self.session.execute(content_query)
            existing_content = content_result.scalar_one_or_none()
            
            if not existing_content:
                raise ValidationError(f"Content with ID {content_id} not found")
            
            # Update fields
            if title is not None:
                existing_content.title = title
                existing_content.slug = self._generate_slug(title)
            
            if content is not None:
                processed_content = await self._process_content(content)
                existing_content.content = processed_content
                
                # Update metadata
                content_metadata = await self._extract_metadata(
                    existing_content.title,
                    processed_content
                )
                existing_content.metadata = content_metadata.__dict__
                existing_content.word_count = content_metadata.word_count
                existing_content.reading_time = content_metadata.reading_time
                existing_content.language = content_metadata.language
                existing_content.seo_score = content_metadata.seo_score
                existing_content.readability_score = content_metadata.readability_score
            
            if status is not None:
                existing_content.status = status.value
            
            if metadata is not None:
                existing_content.metadata.update(metadata)
            
            existing_content.updated_at = datetime.utcnow()
            
            await self.session.commit()
            
            # Update cache
            if content_id in self.content_cache:
                self.content_cache[content_id]["timestamp"] = datetime.utcnow()
            
            return {
                "success": True,
                "content_id": content_id,
                "message": "Content updated successfully"
            }
            
        except Exception as e:
            await self.session.rollback()
            raise DatabaseError(f"Failed to update content: {str(e)}")
    
    async def get_content(
        self,
        content_id: str,
        include_metadata: bool = True
    ) -> Dict[str, Any]:
        """Get content by ID."""
        try:
            # Check cache first
            if content_id in self.content_cache:
                cached_content = self.content_cache[content_id]
                if datetime.now() - cached_content["timestamp"] < timedelta(hours=1):
                    return {
                        "success": True,
                        "data": {
                            "content_id": content_id,
                            "content": cached_content["content"],
                            "metadata": cached_content["metadata"] if include_metadata else None
                        }
                    }
            
            # Get from database
            content_query = select(ContentVersion).where(ContentVersion.content_id == content_id)
            content_result = await self.session.execute(content_query)
            content = content_result.scalar_one_or_none()
            
            if not content:
                raise ValidationError(f"Content with ID {content_id} not found")
            
            result_data = {
                "content_id": content.content_id,
                "title": content.title,
                "content": content.content,
                "slug": content.slug,
                "content_type": content.content_type,
                "author_id": content.author_id,
                "status": content.status,
                "reading_time": content.reading_time,
                "word_count": content.word_count,
                "language": content.language,
                "seo_score": content.seo_score,
                "readability_score": content.readability_score,
                "created_at": content.created_at.isoformat(),
                "updated_at": content.updated_at.isoformat()
            }
            
            if include_metadata:
                result_data["metadata"] = content.metadata
            
            return {
                "success": True,
                "data": result_data
            }
            
        except Exception as e:
            raise DatabaseError(f"Failed to get content: {str(e)}")
    
    async def get_content_by_slug(
        self,
        slug: str,
        include_metadata: bool = True
    ) -> Dict[str, Any]:
        """Get content by slug."""
        try:
            content_query = select(ContentVersion).where(ContentVersion.slug == slug)
            content_result = await self.session.execute(content_query)
            content = content_result.scalar_one_or_none()
            
            if not content:
                raise ValidationError(f"Content with slug '{slug}' not found")
            
            return await self.get_content(content.content_id, include_metadata)
            
        except Exception as e:
            raise DatabaseError(f"Failed to get content by slug: {str(e)}")
    
    async def list_content(
        self,
        content_type: Optional[ContentType] = None,
        status: Optional[ContentStatus] = None,
        author_id: Optional[str] = None,
        page: int = 1,
        page_size: int = 20,
        sort_by: str = "created_at",
        sort_order: str = "desc"
    ) -> Dict[str, Any]:
        """List content with filtering and pagination."""
        try:
            # Build query
            query = select(ContentVersion)
            
            if content_type:
                query = query.where(ContentVersion.content_type == content_type.value)
            
            if status:
                query = query.where(ContentVersion.status == status.value)
            
            if author_id:
                query = query.where(ContentVersion.author_id == author_id)
            
            # Add sorting
            if sort_order.lower() == "desc":
                query = query.order_by(desc(getattr(ContentVersion, sort_by)))
            else:
                query = query.order_by(getattr(ContentVersion, sort_by))
            
            # Add pagination
            offset = (page - 1) * page_size
            query = query.offset(offset).limit(page_size)
            
            # Execute query
            result = await self.session.execute(query)
            content_list = result.scalars().all()
            
            # Get total count
            count_query = select(func.count(ContentVersion.id))
            if content_type:
                count_query = count_query.where(ContentVersion.content_type == content_type.value)
            if status:
                count_query = count_query.where(ContentVersion.status == status.value)
            if author_id:
                count_query = count_query.where(ContentVersion.author_id == author_id)
            
            count_result = await self.session.execute(count_query)
            total_count = count_result.scalar()
            
            # Format results
            formatted_content = []
            for content in content_list:
                formatted_content.append({
                    "content_id": content.content_id,
                    "title": content.title,
                    "slug": content.slug,
                    "content_type": content.content_type,
                    "author_id": content.author_id,
                    "status": content.status,
                    "reading_time": content.reading_time,
                    "word_count": content.word_count,
                    "language": content.language,
                    "seo_score": content.seo_score,
                    "readability_score": content.readability_score,
                    "created_at": content.created_at.isoformat(),
                    "updated_at": content.updated_at.isoformat()
                })
            
            return {
                "success": True,
                "data": {
                    "content": formatted_content,
                    "total": total_count,
                    "page": page,
                    "page_size": page_size,
                    "total_pages": (total_count + page_size - 1) // page_size
                }
            }
            
        except Exception as e:
            raise DatabaseError(f"Failed to list content: {str(e)}")
    
    async def delete_content(self, content_id: str) -> Dict[str, Any]:
        """Delete content."""
        try:
            # Get content
            content_query = select(ContentVersion).where(ContentVersion.content_id == content_id)
            content_result = await self.session.execute(content_query)
            content = content_result.scalar_one_or_none()
            
            if not content:
                raise ValidationError(f"Content with ID {content_id} not found")
            
            # Soft delete by changing status
            content.status = ContentStatus.ARCHIVED.value
            content.updated_at = datetime.utcnow()
            
            await self.session.commit()
            
            # Remove from cache
            if content_id in self.content_cache:
                del self.content_cache[content_id]
            
            return {
                "success": True,
                "content_id": content_id,
                "message": "Content deleted successfully"
            }
            
        except Exception as e:
            await self.session.rollback()
            raise DatabaseError(f"Failed to delete content: {str(e)}")
    
    async def get_content_templates(self) -> Dict[str, Any]:
        """Get available content templates."""
        try:
            return {
                "success": True,
                "data": {
                    "templates": self.content_templates,
                    "total": len(self.content_templates)
                }
            }
        except Exception as e:
            raise DatabaseError(f"Failed to get content templates: {str(e)}")
    
    async def get_content_analytics(
        self,
        content_id: Optional[str] = None,
        days: int = 30
    ) -> Dict[str, Any]:
        """Get content analytics."""
        try:
            # This would implement content analytics
            # For now, returning placeholder data
            return {
                "success": True,
                "data": {
                    "total_content": 0,
                    "published_content": 0,
                    "draft_content": 0,
                    "average_reading_time": 0,
                    "average_seo_score": 0,
                    "average_readability_score": 0,
                    "content_by_type": {},
                    "content_by_status": {},
                    "top_performing_content": []
                }
            }
        except Exception as e:
            raise DatabaseError(f"Failed to get content analytics: {str(e)}")
    
    async def get_content_stats(self) -> Dict[str, Any]:
        """Get content statistics."""
        try:
            # Get total content count
            total_query = select(func.count(ContentVersion.id))
            total_result = await self.session.execute(total_query)
            total_content = total_result.scalar()
            
            # Get content by status
            status_query = select(
                ContentVersion.status,
                func.count(ContentVersion.id).label('count')
            ).group_by(ContentVersion.status)
            
            status_result = await self.session.execute(status_query)
            content_by_status = {row[0]: row[1] for row in status_result}
            
            # Get content by type
            type_query = select(
                ContentVersion.content_type,
                func.count(ContentVersion.id).label('count')
            ).group_by(ContentVersion.content_type)
            
            type_result = await self.session.execute(type_query)
            content_by_type = {row[0]: row[1] for row in type_result}
            
            # Get average metrics
            avg_query = select(
                func.avg(ContentVersion.reading_time).label('avg_reading_time'),
                func.avg(ContentVersion.seo_score).label('avg_seo_score'),
                func.avg(ContentVersion.readability_score).label('avg_readability_score'),
                func.avg(ContentVersion.word_count).label('avg_word_count')
            )
            
            avg_result = await self.session.execute(avg_query)
            avg_metrics = avg_result.first()
            
            return {
                "success": True,
                "data": {
                    "total_content": total_content,
                    "content_by_status": content_by_status,
                    "content_by_type": content_by_type,
                    "average_reading_time": float(avg_metrics.avg_reading_time or 0),
                    "average_seo_score": float(avg_metrics.avg_seo_score or 0),
                    "average_readability_score": float(avg_metrics.avg_readability_score or 0),
                    "average_word_count": float(avg_metrics.avg_word_count or 0),
                    "cache_size": len(self.content_cache),
                    "templates_available": len(self.content_templates)
                }
            }
            
        except Exception as e:
            raise DatabaseError(f"Failed to get content stats: {str(e)}")
























