"""
Advanced Copywriting AI System

This module provides comprehensive AI-powered copywriting capabilities integrated
with the HeyGen AI system for intelligent content generation and optimization.
"""

import asyncio
import openai
import logging
import json
import re
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import Enum
import uuid
import sqlite3
from pathlib import Path
import requests
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import spacy
from textstat import flesch_reading_ease, flesch_kincaid_grade
import warnings
warnings.filterwarnings('ignore')


logger = logging.getLogger(__name__)


class ContentType(str, Enum):
    """Content types."""
    BLOG_POST = "blog_post"
    SOCIAL_MEDIA = "social_media"
    EMAIL = "email"
    AD_COPY = "ad_copy"
    PRODUCT_DESCRIPTION = "product_description"
    LANDING_PAGE = "landing_page"
    SALES_LETTER = "sales_letter"
    PRESS_RELEASE = "press_release"
    WHITEPAPER = "whitepaper"
    CASE_STUDY = "case_study"


class Tone(str, Enum):
    """Content tones."""
    PROFESSIONAL = "professional"
    CASUAL = "casual"
    FRIENDLY = "friendly"
    AUTHORITATIVE = "authoritative"
    CONVERSATIONAL = "conversational"
    URGENT = "urgent"
    INSPIRATIONAL = "inspirational"
    HUMOROUS = "humorous"
    TECHNICAL = "technical"
    PERSUASIVE = "persuasive"


class TargetAudience(str, Enum):
    """Target audiences."""
    B2B = "b2b"
    B2C = "b2c"
    TECHNICAL = "technical"
    GENERAL = "general"
    PROFESSIONAL = "professional"
    YOUTH = "youth"
    SENIOR = "senior"
    ENTREPRENEUR = "entrepreneur"
    INVESTOR = "investor"
    CONSUMER = "consumer"


@dataclass
class ContentRequest:
    """Content generation request."""
    content_type: ContentType
    topic: str
    tone: Tone
    target_audience: TargetAudience
    length: int  # words
    keywords: List[str] = field(default_factory=list)
    brand_voice: Optional[str] = None
    call_to_action: Optional[str] = None
    additional_requirements: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GeneratedContent:
    """Generated content structure."""
    content_id: str
    title: str
    content: str
    content_type: ContentType
    tone: Tone
    target_audience: TargetAudience
    word_count: int
    reading_level: str
    sentiment_score: float
    keywords_used: List[str]
    seo_score: float
    engagement_score: float
    created_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ContentOptimization:
    """Content optimization suggestions."""
    readability_score: float
    seo_score: float
    engagement_score: float
    suggestions: List[str]
    improvements: Dict[str, Any] = field(default_factory=dict)


class NLPAnalyzer:
    """Advanced NLP analysis for content."""
    
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        
        # Download required NLTK data
        try:
            nltk.download('vader_lexicon', quiet=True)
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
        except:
            pass
    
    def analyze_sentiment(self, text: str) -> float:
        """Analyze text sentiment."""
        scores = self.sia.polarity_scores(text)
        return scores['compound']  # -1 to 1
    
    def calculate_readability(self, text: str) -> Dict[str, float]:
        """Calculate readability metrics."""
        return {
            'flesch_reading_ease': flesch_reading_ease(text),
            'flesch_kincaid_grade': flesch_kincaid_grade(text)
        }
    
    def extract_keywords(self, text: str, num_keywords: int = 10) -> List[str]:
        """Extract keywords from text."""
        # Tokenize and clean text
        words = word_tokenize(text.lower())
        words = [word for word in words if word.isalpha() and word not in self.stop_words]
        
        # Calculate TF-IDF scores
        if len(words) > 0:
            tfidf_matrix = self.vectorizer.fit_transform([' '.join(words)])
            feature_names = self.vectorizer.get_feature_names_out()
            scores = tfidf_matrix.toarray()[0]
            
            # Get top keywords
            keyword_scores = list(zip(feature_names, scores))
            keyword_scores.sort(key=lambda x: x[1], reverse=True)
            
            return [keyword for keyword, score in keyword_scores[:num_keywords]]
        
        return []
    
    def calculate_engagement_score(self, text: str) -> float:
        """Calculate content engagement score."""
        # Factors that increase engagement
        engagement_factors = 0
        
        # Question marks (questions engage readers)
        engagement_factors += text.count('?') * 0.1
        
        # Exclamation marks (excitement)
        engagement_factors += text.count('!') * 0.05
        
        # Power words
        power_words = ['amazing', 'incredible', 'revolutionary', 'breakthrough', 'exclusive', 'limited', 'free', 'new', 'best', 'top']
        for word in power_words:
            engagement_factors += text.lower().count(word) * 0.2
        
        # Emotional words
        emotional_words = ['love', 'hate', 'excited', 'worried', 'thrilled', 'disappointed', 'surprised', 'shocked']
        for word in emotional_words:
            engagement_factors += text.lower().count(word) * 0.15
        
        # Normalize score
        return min(1.0, engagement_factors / 10)
    
    def calculate_seo_score(self, text: str, target_keywords: List[str]) -> float:
        """Calculate SEO score for content."""
        if not target_keywords:
            return 0.5
        
        text_lower = text.lower()
        seo_score = 0
        
        # Keyword density
        total_words = len(text.split())
        for keyword in target_keywords:
            keyword_count = text_lower.count(keyword.lower())
            density = keyword_count / total_words if total_words > 0 else 0
            seo_score += min(0.3, density * 10)  # Cap at 0.3 per keyword
        
        # Title optimization
        if target_keywords and any(keyword.lower() in text_lower[:100] for keyword in target_keywords):
            seo_score += 0.2
        
        # Meta description length (simplified)
        if 120 <= len(text) <= 160:
            seo_score += 0.1
        
        return min(1.0, seo_score)


class ContentGenerator:
    """Advanced AI content generator."""
    
    def __init__(self, openai_api_key: str = None):
        self.openai_api_key = openai_api_key
        if openai_api_key:
            openai.api_key = openai_api_key
        
        self.nlp_analyzer = NLPAnalyzer()
        self.content_templates = self._load_content_templates()
        self.brand_voices = self._load_brand_voices()
    
    def _load_content_templates(self) -> Dict[str, str]:
        """Load content generation templates."""
        return {
            ContentType.BLOG_POST: """
            Title: {title}
            
            Introduction:
            {introduction}
            
            Main Content:
            {main_content}
            
            Conclusion:
            {conclusion}
            
            Call to Action:
            {call_to_action}
            """,
            
            ContentType.SOCIAL_MEDIA: """
            {content}
            
            Hashtags: {hashtags}
            """,
            
            ContentType.EMAIL: """
            Subject: {subject}
            
            {greeting}
            
            {body}
            
            {closing}
            """,
            
            ContentType.AD_COPY: """
            {headline}
            
            {subheadline}
            
            {body_copy}
            
            {call_to_action}
            """,
            
            ContentType.PRODUCT_DESCRIPTION: """
            {product_name}
            
            {key_features}
            
            {benefits}
            
            {call_to_action}
            """
        }
    
    def _load_brand_voices(self) -> Dict[str, Dict[str, Any]]:
        """Load brand voice configurations."""
        return {
            'professional': {
                'tone': 'formal',
                'vocabulary': 'technical',
                'sentence_structure': 'complex',
                'emotion': 'neutral'
            },
            'casual': {
                'tone': 'informal',
                'vocabulary': 'simple',
                'sentence_structure': 'short',
                'emotion': 'friendly'
            },
            'authoritative': {
                'tone': 'confident',
                'vocabulary': 'expert',
                'sentence_structure': 'declarative',
                'emotion': 'strong'
            },
            'conversational': {
                'tone': 'chatty',
                'vocabulary': 'everyday',
                'sentence_structure': 'varied',
                'emotion': 'warm'
            }
        }
    
    async def generate_content(self, request: ContentRequest) -> GeneratedContent:
        """Generate content based on request."""
        try:
            # Generate content using AI
            if self.openai_api_key:
                content = await self._generate_with_openai(request)
            else:
                content = await self._generate_with_templates(request)
            
            # Analyze generated content
            sentiment_score = self.nlp_analyzer.analyze_sentiment(content)
            readability = self.nlp_analyzer.calculate_readability(content)
            keywords_used = self.nlp_analyzer.extract_keywords(content)
            seo_score = self.nlp_analyzer.calculate_seo_score(content, request.keywords)
            engagement_score = self.nlp_analyzer.calculate_engagement_score(content)
            
            # Determine reading level
            fk_grade = readability['flesch_kincaid_grade']
            if fk_grade <= 6:
                reading_level = "Elementary"
            elif fk_grade <= 9:
                reading_level = "Middle School"
            elif fk_grade <= 12:
                reading_level = "High School"
            else:
                reading_level = "College"
            
            # Create generated content object
            generated_content = GeneratedContent(
                content_id=str(uuid.uuid4()),
                title=self._extract_title(content),
                content=content,
                content_type=request.content_type,
                tone=request.tone,
                target_audience=request.target_audience,
                word_count=len(content.split()),
                reading_level=reading_level,
                sentiment_score=sentiment_score,
                keywords_used=keywords_used,
                seo_score=seo_score,
                engagement_score=engagement_score,
                created_at=datetime.now(timezone.utc),
                metadata={
                    'readability': readability,
                    'request_requirements': request.additional_requirements
                }
            )
            
            return generated_content
            
        except Exception as e:
            logger.error(f"Error generating content: {e}")
            raise
    
    async def _generate_with_openai(self, request: ContentRequest) -> str:
        """Generate content using OpenAI API."""
        try:
            prompt = self._build_prompt(request)
            
            response = await openai.ChatCompletion.acreate(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a professional copywriter with expertise in creating engaging, SEO-optimized content."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=request.length * 2,  # Approximate token count
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error with OpenAI API: {e}")
            return await self._generate_with_templates(request)
    
    async def _generate_with_templates(self, request: ContentRequest) -> str:
        """Generate content using templates."""
        template = self.content_templates.get(request.content_type, "")
        
        if not template:
            return f"Content about {request.topic} for {request.target_audience.value} audience."
        
        # Fill template with generated content
        content_parts = {
            'title': self._generate_title(request),
            'introduction': self._generate_introduction(request),
            'main_content': self._generate_main_content(request),
            'conclusion': self._generate_conclusion(request),
            'call_to_action': request.call_to_action or self._generate_cta(request),
            'content': self._generate_simple_content(request),
            'hashtags': self._generate_hashtags(request),
            'subject': self._generate_email_subject(request),
            'greeting': self._generate_greeting(request),
            'body': self._generate_email_body(request),
            'closing': self._generate_closing(request),
            'headline': self._generate_headline(request),
            'subheadline': self._generate_subheadline(request),
            'body_copy': self._generate_body_copy(request),
            'product_name': request.topic,
            'key_features': self._generate_key_features(request),
            'benefits': self._generate_benefits(request)
        }
        
        return template.format(**content_parts)
    
    def _build_prompt(self, request: ContentRequest) -> str:
        """Build prompt for AI content generation."""
        prompt = f"""
        Create {request.content_type.value} content with the following specifications:
        
        Topic: {request.topic}
        Tone: {request.tone.value}
        Target Audience: {request.target_audience.value}
        Length: {request.length} words
        Keywords: {', '.join(request.keywords)}
        
        Additional Requirements:
        {json.dumps(request.additional_requirements, indent=2)}
        
        Please create engaging, SEO-optimized content that resonates with the target audience.
        """
        
        if request.brand_voice:
            prompt += f"\nBrand Voice: {request.brand_voice}"
        
        if request.call_to_action:
            prompt += f"\nCall to Action: {request.call_to_action}"
        
        return prompt
    
    def _generate_title(self, request: ContentRequest) -> str:
        """Generate content title."""
        power_words = ['Ultimate', 'Complete', 'Essential', 'Proven', 'Secret', 'Best', 'Top', 'Amazing']
        power_word = np.random.choice(power_words)
        return f"{power_word} Guide to {request.topic}"
    
    def _generate_introduction(self, request: ContentRequest) -> str:
        """Generate content introduction."""
        return f"Welcome to our comprehensive guide on {request.topic}. In this article, we'll explore everything you need to know about this important topic and how it can benefit you."
    
    def _generate_main_content(self, request: ContentRequest) -> str:
        """Generate main content."""
        return f"Let's dive deep into {request.topic}. This topic is crucial for {request.target_audience.value} audiences because it offers numerous benefits and opportunities for growth and success."
    
    def _generate_conclusion(self, request: ContentRequest) -> str:
        """Generate content conclusion."""
        return f"In conclusion, {request.topic} is an essential topic that every {request.target_audience.value} professional should understand and implement in their strategy."
    
    def _generate_cta(self, request: ContentRequest) -> str:
        """Generate call to action."""
        ctas = [
            "Get started today!",
            "Learn more now!",
            "Contact us for more information!",
            "Download our free guide!",
            "Schedule a consultation!"
        ]
        return np.random.choice(ctas)
    
    def _generate_simple_content(self, request: ContentRequest) -> str:
        """Generate simple content."""
        return f"Discover the power of {request.topic} and how it can transform your {request.target_audience.value} strategy. Learn the secrets that top professionals use to achieve success."
    
    def _generate_hashtags(self, request: ContentRequest) -> str:
        """Generate hashtags."""
        base_hashtags = [request.topic.replace(' ', ''), request.target_audience.value]
        additional_hashtags = ['#success', '#tips', '#guide', '#professional']
        return ' '.join(base_hashtags + additional_hashtags)
    
    def _generate_email_subject(self, request: ContentRequest) -> str:
        """Generate email subject."""
        return f"Important: {request.topic} - Don't Miss Out!"
    
    def _generate_greeting(self, request: ContentRequest) -> str:
        """Generate email greeting."""
        return "Hi there,"
    
    def _generate_email_body(self, request: ContentRequest) -> str:
        """Generate email body."""
        return f"I wanted to share some valuable insights about {request.topic} that I think you'll find extremely helpful for your {request.target_audience.value} needs."
    
    def _generate_closing(self, request: ContentRequest) -> str:
        """Generate email closing."""
        return "Best regards,\nThe Team"
    
    def _generate_headline(self, request: ContentRequest) -> str:
        """Generate ad headline."""
        return f"Transform Your {request.topic} Strategy Today!"
    
    def _generate_subheadline(self, request: ContentRequest) -> str:
        """Generate ad subheadline."""
        return f"Join thousands of {request.target_audience.value} professionals who have already discovered the secret."
    
    def _generate_body_copy(self, request: ContentRequest) -> str:
        """Generate ad body copy."""
        return f"Discover how {request.topic} can revolutionize your approach and deliver amazing results in record time."
    
    def _generate_key_features(self, request: ContentRequest) -> str:
        """Generate product key features."""
        return f"• Advanced {request.topic} capabilities\n• Easy to use interface\n• Proven results\n• 24/7 support"
    
    def _generate_benefits(self, request: ContentRequest) -> str:
        """Generate product benefits."""
        return f"Experience the benefits of {request.topic} including increased efficiency, better results, and improved performance for your {request.target_audience.value} needs."
    
    def _extract_title(self, content: str) -> str:
        """Extract title from content."""
        lines = content.split('\n')
        for line in lines:
            if line.strip() and not line.startswith(' ') and len(line.strip()) < 100:
                return line.strip()
        return "Generated Content"


class ContentOptimizer:
    """Advanced content optimization engine."""
    
    def __init__(self):
        self.nlp_analyzer = NLPAnalyzer()
    
    def optimize_content(self, content: GeneratedContent) -> ContentOptimization:
        """Optimize content for better performance."""
        suggestions = []
        improvements = {}
        
        # Readability optimization
        readability = content.metadata.get('readability', {})
        if readability.get('flesch_reading_ease', 0) < 60:
            suggestions.append("Improve readability by using shorter sentences and simpler words")
            improvements['readability'] = "Use shorter sentences and simpler vocabulary"
        
        # SEO optimization
        if content.seo_score < 0.7:
            suggestions.append("Improve SEO by better keyword integration and optimization")
            improvements['seo'] = "Add more target keywords naturally throughout the content"
        
        # Engagement optimization
        if content.engagement_score < 0.6:
            suggestions.append("Increase engagement with more questions, power words, and emotional language")
            improvements['engagement'] = "Add questions, power words, and emotional triggers"
        
        # Length optimization
        if content.word_count < 300:
            suggestions.append("Consider adding more content for better SEO and value")
            improvements['length'] = "Expand content with more detailed information"
        elif content.word_count > 2000:
            suggestions.append("Consider breaking content into smaller sections for better readability")
            improvements['length'] = "Break content into smaller, digestible sections"
        
        # Sentiment optimization
        if abs(content.sentiment_score) < 0.1:
            suggestions.append("Add more emotional language to increase engagement")
            improvements['sentiment'] = "Include more emotional and persuasive language"
        
        return ContentOptimization(
            readability_score=readability.get('flesch_reading_ease', 0) / 100,
            seo_score=content.seo_score,
            engagement_score=content.engagement_score,
            suggestions=suggestions,
            improvements=improvements
        )


class AdvancedCopywritingAISystem:
    """
    Advanced copywriting AI system with comprehensive capabilities.
    
    Features:
    - AI-powered content generation
    - Multiple content types and tones
    - SEO optimization
    - Content analysis and optimization
    - Brand voice adaptation
    - Performance tracking
    - A/B testing capabilities
    - Content templates and libraries
    """
    
    def __init__(self, openai_api_key: str = None, db_path: str = "copywriting_ai.db"):
        """
        Initialize the advanced copywriting AI system.
        
        Args:
            openai_api_key: OpenAI API key for content generation
            db_path: SQLite database path for content storage
        """
        self.openai_api_key = openai_api_key
        self.db_path = db_path
        self.db_connection = None
        
        # Initialize components
        self.content_generator = ContentGenerator(openai_api_key)
        self.content_optimizer = ContentOptimizer()
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database."""
        self.db_connection = sqlite3.connect(self.db_path)
        cursor = self.db_connection.cursor()
        
        # Create tables
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS generated_content (
                content_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                content_type TEXT NOT NULL,
                tone TEXT NOT NULL,
                target_audience TEXT NOT NULL,
                word_count INTEGER,
                reading_level TEXT,
                sentiment_score REAL,
                keywords_used TEXT,
                seo_score REAL,
                engagement_score REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS content_optimizations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content_id TEXT NOT NULL,
                readability_score REAL,
                seo_score REAL,
                engagement_score REAL,
                suggestions TEXT,
                improvements TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (content_id) REFERENCES generated_content (content_id)
            )
        """)
        
        self.db_connection.commit()
    
    async def generate_content(self, request: ContentRequest) -> GeneratedContent:
        """Generate content based on request."""
        try:
            # Generate content
            content = await self.content_generator.generate_content(request)
            
            # Store in database
            await self._store_content(content)
            
            return content
            
        except Exception as e:
            logger.error(f"Error generating content: {e}")
            raise
    
    async def optimize_content(self, content_id: str) -> ContentOptimization:
        """Optimize existing content."""
        try:
            # Get content from database
            content = await self._get_content(content_id)
            if not content:
                raise ValueError(f"Content {content_id} not found")
            
            # Optimize content
            optimization = self.content_optimizer.optimize_content(content)
            
            # Store optimization
            await self._store_optimization(content_id, optimization)
            
            return optimization
            
        except Exception as e:
            logger.error(f"Error optimizing content: {e}")
            raise
    
    async def _store_content(self, content: GeneratedContent):
        """Store generated content in database."""
        try:
            cursor = self.db_connection.cursor()
            cursor.execute("""
                INSERT INTO generated_content 
                (content_id, title, content, content_type, tone, target_audience,
                 word_count, reading_level, sentiment_score, keywords_used,
                 seo_score, engagement_score, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                content.content_id,
                content.title,
                content.content,
                content.content_type.value,
                content.tone.value,
                content.target_audience.value,
                content.word_count,
                content.reading_level,
                content.sentiment_score,
                json.dumps(content.keywords_used),
                content.seo_score,
                content.engagement_score,
                json.dumps(content.metadata)
            ))
            self.db_connection.commit()
            
        except Exception as e:
            logger.error(f"Error storing content: {e}")
    
    async def _get_content(self, content_id: str) -> Optional[GeneratedContent]:
        """Get content from database."""
        try:
            cursor = self.db_connection.cursor()
            cursor.execute("""
                SELECT * FROM generated_content WHERE content_id = ?
            """, (content_id,))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            return GeneratedContent(
                content_id=row[0],
                title=row[1],
                content=row[2],
                content_type=ContentType(row[3]),
                tone=Tone(row[4]),
                target_audience=TargetAudience(row[5]),
                word_count=row[6],
                reading_level=row[7],
                sentiment_score=row[8],
                keywords_used=json.loads(row[9]),
                seo_score=row[10],
                engagement_score=row[11],
                created_at=datetime.fromisoformat(row[12]),
                metadata=json.loads(row[13]) if row[13] else {}
            )
            
        except Exception as e:
            logger.error(f"Error getting content: {e}")
            return None
    
    async def _store_optimization(self, content_id: str, optimization: ContentOptimization):
        """Store content optimization in database."""
        try:
            cursor = self.db_connection.cursor()
            cursor.execute("""
                INSERT INTO content_optimizations 
                (content_id, readability_score, seo_score, engagement_score,
                 suggestions, improvements)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                content_id,
                optimization.readability_score,
                optimization.seo_score,
                optimization.engagement_score,
                json.dumps(optimization.suggestions),
                json.dumps(optimization.improvements)
            ))
            self.db_connection.commit()
            
        except Exception as e:
            logger.error(f"Error storing optimization: {e}")
    
    def get_content_summary(self) -> Dict[str, Any]:
        """Get content generation summary."""
        try:
            cursor = self.db_connection.cursor()
            
            # Get total content count
            cursor.execute("SELECT COUNT(*) FROM generated_content")
            total_content = cursor.fetchone()[0]
            
            # Get content by type
            cursor.execute("""
                SELECT content_type, COUNT(*) 
                FROM generated_content 
                GROUP BY content_type
            """)
            content_by_type = dict(cursor.fetchall())
            
            # Get average scores
            cursor.execute("""
                SELECT AVG(seo_score), AVG(engagement_score), AVG(sentiment_score)
                FROM generated_content
            """)
            avg_scores = cursor.fetchone()
            
            return {
                'total_content': total_content,
                'content_by_type': content_by_type,
                'average_scores': {
                    'seo_score': avg_scores[0] or 0,
                    'engagement_score': avg_scores[1] or 0,
                    'sentiment_score': avg_scores[2] or 0
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting content summary: {e}")
            return {}
    
    def cleanup(self):
        """Cleanup resources."""
        if self.db_connection:
            self.db_connection.close()


# Example usage and demonstration
async def main():
    """Demonstrate the advanced copywriting AI system."""
    print("✍️ HeyGen AI - Advanced Copywriting AI System Demo")
    print("=" * 70)
    
    # Initialize system
    copywriting_ai = AdvancedCopywritingAISystem()
    
    try:
        # Create content requests
        requests = [
            ContentRequest(
                content_type=ContentType.BLOG_POST,
                topic="Artificial Intelligence in Business",
                tone=Tone.PROFESSIONAL,
                target_audience=TargetAudience.B2B,
                length=800,
                keywords=["AI", "business", "automation", "efficiency"],
                call_to_action="Learn more about AI implementation"
            ),
            ContentRequest(
                content_type=ContentType.SOCIAL_MEDIA,
                topic="Digital Transformation",
                tone=Tone.CASUAL,
                target_audience=TargetAudience.GENERAL,
                length=150,
                keywords=["digital", "transformation", "technology", "innovation"]
            ),
            ContentRequest(
                content_type=ContentType.EMAIL,
                topic="Product Launch Announcement",
                tone=Tone.URGENT,
                target_audience=TargetAudience.CONSUMER,
                length=300,
                keywords=["launch", "product", "exclusive", "limited"]
            )
        ]
        
        # Generate content
        print("\n📝 Generating Content...")
        generated_contents = []
        
        for i, request in enumerate(requests, 1):
            print(f"\n  {i}. Generating {request.content_type.value}...")
            content = await copywriting_ai.generate_content(request)
            generated_contents.append(content)
            
            print(f"    Title: {content.title}")
            print(f"    Word Count: {content.word_count}")
            print(f"    Reading Level: {content.reading_level}")
            print(f"    SEO Score: {content.seo_score:.2f}")
            print(f"    Engagement Score: {content.engagement_score:.2f}")
            print(f"    Sentiment Score: {content.sentiment_score:.2f}")
        
        # Optimize content
        print("\n🔧 Optimizing Content...")
        for content in generated_contents:
            print(f"\n  Optimizing: {content.title}")
            optimization = await copywriting_ai.optimize_content(content.content_id)
            
            print(f"    Readability Score: {optimization.readability_score:.2f}")
            print(f"    SEO Score: {optimization.seo_score:.2f}")
            print(f"    Engagement Score: {optimization.engagement_score:.2f}")
            
            if optimization.suggestions:
                print("    Suggestions:")
                for suggestion in optimization.suggestions[:3]:  # Show first 3
                    print(f"      - {suggestion}")
        
        # Get summary
        print("\n📊 Content Generation Summary:")
        summary = copywriting_ai.get_content_summary()
        print(f"  Total Content Generated: {summary.get('total_content', 0)}")
        print(f"  Content by Type: {summary.get('content_by_type', {})}")
        print(f"  Average SEO Score: {summary.get('average_scores', {}).get('seo_score', 0):.2f}")
        print(f"  Average Engagement Score: {summary.get('average_scores', {}).get('engagement_score', 0):.2f}")
        
        # Show sample content
        if generated_contents:
            print(f"\n📄 Sample Generated Content:")
            sample = generated_contents[0]
            print(f"  Type: {sample.content_type.value}")
            print(f"  Title: {sample.title}")
            print(f"  Content Preview: {sample.content[:200]}...")
        
    except Exception as e:
        print(f"❌ Demo Error: {e}")
        logger.error(f"Demo failed: {e}")
    
    finally:
        # Cleanup
        copywriting_ai.cleanup()
        print("\n✅ Demo completed")


if __name__ == "__main__":
    asyncio.run(main())
