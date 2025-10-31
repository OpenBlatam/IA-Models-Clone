"""
Content Analytics Engine for Export IA
======================================

Comprehensive content analytics and insights system that provides
detailed analysis of document content, quality metrics, and optimization
recommendations.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import uuid
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re
from collections import defaultdict, Counter
import networkx as nx
from wordcloud import WordCloud
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from textstat import flesch_reading_ease, flesch_kincaid_grade, gunning_fog
import spacy
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
import yake
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logger = logging.getLogger(__name__)

class AnalyticsLevel(Enum):
    """Levels of content analytics."""
    BASIC = "basic"
    STANDARD = "standard"
    ADVANCED = "advanced"
    ENTERPRISE = "enterprise"

class ContentType(Enum):
    """Types of content being analyzed."""
    BUSINESS_DOCUMENT = "business_document"
    TECHNICAL_REPORT = "technical_report"
    ACADEMIC_PAPER = "academic_paper"
    MARKETING_CONTENT = "marketing_content"
    LEGAL_DOCUMENT = "legal_document"
    CREATIVE_WRITING = "creative_writing"
    NEWS_ARTICLE = "news_article"
    BLOG_POST = "blog_post"

@dataclass
class ReadabilityMetrics:
    """Readability analysis metrics."""
    flesch_reading_ease: float
    flesch_kincaid_grade: float
    gunning_fog_index: float
    automated_readability_index: float
    coleman_liau_index: float
    smog_index: float
    average_sentence_length: float
    average_syllables_per_word: float
    complex_word_percentage: float

@dataclass
class SentimentAnalysis:
    """Sentiment analysis results."""
    overall_sentiment: str  # positive, negative, neutral
    sentiment_score: float  # -1 to 1
    positive_percentage: float
    negative_percentage: float
    neutral_percentage: float
    emotional_intensity: float
    sentiment_by_section: Dict[str, float]

@dataclass
class TopicAnalysis:
    """Topic modeling and analysis results."""
    main_topics: List[str]
    topic_distribution: Dict[str, float]
    topic_keywords: Dict[str, List[str]]
    topic_coherence: float
    topic_diversity: float
    dominant_topic: str

@dataclass
class ContentStructure:
    """Content structure analysis."""
    total_sections: int
    total_paragraphs: int
    total_sentences: int
    total_words: int
    average_paragraph_length: float
    average_sentence_length: float
    section_hierarchy: Dict[str, int]
    content_flow_score: float
    structural_balance: float

@dataclass
class KeywordAnalysis:
    """Keyword and phrase analysis."""
    top_keywords: List[Tuple[str, float]]
    key_phrases: List[Tuple[str, float]]
    keyword_density: Dict[str, float]
    keyword_distribution: Dict[str, int]
    semantic_keywords: List[str]
    trending_keywords: List[str]

@dataclass
class QualityMetrics:
    """Overall content quality metrics."""
    overall_quality_score: float
    content_relevance: float
    information_density: float
    clarity_score: float
    engagement_score: float
    credibility_score: float
    accessibility_score: float
    seo_score: float
    readability_score: float

@dataclass
class ContentAnalytics:
    """Comprehensive content analytics report."""
    id: str
    document_id: str
    content_type: ContentType
    analytics_level: AnalyticsLevel
    readability_metrics: ReadabilityMetrics
    sentiment_analysis: SentimentAnalysis
    topic_analysis: TopicAnalysis
    content_structure: ContentStructure
    keyword_analysis: KeywordAnalysis
    quality_metrics: QualityMetrics
    recommendations: List[str]
    insights: List[str]
    processing_time: float
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

class ContentAnalyticsEngine:
    """Advanced content analytics and insights engine."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize models
        self.nlp = None
        self.sentence_transformer = None
        self.sentiment_analyzer = None
        self.topic_model = None
        self.tfidf_vectorizer = None
        self.keyword_extractor = None
        
        # Analytics settings
        self.analytics_weights = {
            AnalyticsLevel.BASIC: {"readability": 1.0, "structure": 1.0},
            AnalyticsLevel.STANDARD: {"readability": 1.0, "structure": 1.0, "sentiment": 1.0, "keywords": 1.0},
            AnalyticsLevel.ADVANCED: {"readability": 1.0, "structure": 1.0, "sentiment": 1.0, "keywords": 1.0, "topics": 1.0, "quality": 1.0},
            AnalyticsLevel.ENTERPRISE: {"readability": 1.0, "structure": 1.0, "sentiment": 1.0, "keywords": 1.0, "topics": 1.0, "quality": 1.0, "seo": 1.0, "accessibility": 1.0}
        }
        
        # Initialize models
        self._initialize_models()
        
        logger.info("Content Analytics Engine initialized")
    
    def _initialize_models(self):
        """Initialize NLP and ML models for analytics."""
        try:
            # Initialize spaCy
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("spaCy model not found, using basic tokenization")
                self.nlp = None
            
            # Initialize sentence transformer
            self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Initialize sentiment analyzer
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
            
            # Initialize TF-IDF vectorizer
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 3)
            )
            
            # Initialize keyword extractor
            self.keyword_extractor = yake.KeywordExtractor(
                lan="en",
                n=3,
                dedupLim=0.7,
                top=20
            )
            
            logger.info("Analytics models initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize analytics models: {e}")
    
    async def analyze_content(
        self,
        content: str,
        document_id: str = None,
        content_type: ContentType = ContentType.BUSINESS_DOCUMENT,
        analytics_level: AnalyticsLevel = AnalyticsLevel.STANDARD
    ) -> ContentAnalytics:
        """Perform comprehensive content analysis."""
        
        if not document_id:
            document_id = str(uuid.uuid4())
        
        start_time = datetime.now()
        analytics_id = str(uuid.uuid4())
        
        logger.info(f"Starting content analysis: {content_type.value}, {analytics_level.value} level")
        
        try:
            # Basic analysis (always performed)
            readability_metrics = await self._analyze_readability(content)
            content_structure = await self._analyze_content_structure(content)
            
            # Standard analysis
            sentiment_analysis = None
            keyword_analysis = None
            
            if analytics_level in [AnalyticsLevel.STANDARD, AnalyticsLevel.ADVANCED, AnalyticsLevel.ENTERPRISE]:
                sentiment_analysis = await self._analyze_sentiment(content)
                keyword_analysis = await self._analyze_keywords(content)
            
            # Advanced analysis
            topic_analysis = None
            quality_metrics = None
            
            if analytics_level in [AnalyticsLevel.ADVANCED, AnalyticsLevel.ENTERPRISE]:
                topic_analysis = await self._analyze_topics(content)
                quality_metrics = await self._analyze_quality(content, content_type)
            
            # Enterprise analysis
            if analytics_level == AnalyticsLevel.ENTERPRISE:
                # Additional enterprise-level analysis
                pass
            
            # Generate recommendations and insights
            recommendations = await self._generate_recommendations(
                readability_metrics, content_structure, sentiment_analysis, 
                keyword_analysis, topic_analysis, quality_metrics, content_type
            )
            
            insights = await self._generate_insights(
                readability_metrics, content_structure, sentiment_analysis,
                keyword_analysis, topic_analysis, quality_metrics, content_type
            )
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Create analytics report
            analytics = ContentAnalytics(
                id=analytics_id,
                document_id=document_id,
                content_type=content_type,
                analytics_level=analytics_level,
                readability_metrics=readability_metrics,
                sentiment_analysis=sentiment_analysis,
                topic_analysis=topic_analysis,
                content_structure=content_structure,
                keyword_analysis=keyword_analysis,
                quality_metrics=quality_metrics,
                recommendations=recommendations,
                insights=insights,
                processing_time=processing_time,
                metadata={
                    "content_length": len(content),
                    "word_count": len(content.split()),
                    "character_count": len(content)
                }
            )
            
            logger.info(f"Content analysis completed in {processing_time:.2f} seconds")
            
            return analytics
            
        except Exception as e:
            logger.error(f"Content analysis failed: {e}")
            raise
    
    async def _analyze_readability(self, content: str) -> ReadabilityMetrics:
        """Analyze content readability."""
        
        # Calculate readability metrics
        flesch_ease = flesch_reading_ease(content)
        flesch_grade = flesch_kincaid_grade(content)
        gunning_fog = gunning_fog(content)
        
        # Calculate additional metrics
        sentences = self._split_into_sentences(content)
        words = content.split()
        
        avg_sentence_length = len(words) / len(sentences) if sentences else 0
        
        # Calculate syllables per word (simplified)
        syllables = sum(self._count_syllables(word) for word in words)
        avg_syllables_per_word = syllables / len(words) if words else 0
        
        # Calculate complex word percentage
        complex_words = sum(1 for word in words if self._count_syllables(word) > 2)
        complex_word_percentage = (complex_words / len(words)) * 100 if words else 0
        
        # Calculate other readability indices
        ari = self._calculate_ari(content)
        cli = self._calculate_cli(content)
        smog = self._calculate_smog(content)
        
        return ReadabilityMetrics(
            flesch_reading_ease=flesch_ease,
            flesch_kincaid_grade=flesch_grade,
            gunning_fog_index=gunning_fog,
            automated_readability_index=ari,
            coleman_liau_index=cli,
            smog_index=smog,
            average_sentence_length=avg_sentence_length,
            average_syllables_per_word=avg_syllables_per_word,
            complex_word_percentage=complex_word_percentage
        )
    
    async def _analyze_content_structure(self, content: str) -> ContentStructure:
        """Analyze content structure and organization."""
        
        # Split content into sections
        sections = content.split('\n\n')
        paragraphs = [p.strip() for p in sections if p.strip()]
        
        # Count sentences and words
        sentences = self._split_into_sentences(content)
        words = content.split()
        
        # Calculate averages
        avg_paragraph_length = len(words) / len(paragraphs) if paragraphs else 0
        avg_sentence_length = len(words) / len(sentences) if sentences else 0
        
        # Analyze section hierarchy
        section_hierarchy = self._analyze_section_hierarchy(content)
        
        # Calculate content flow score
        content_flow_score = self._calculate_content_flow_score(paragraphs)
        
        # Calculate structural balance
        structural_balance = self._calculate_structural_balance(paragraphs)
        
        return ContentStructure(
            total_sections=len(sections),
            total_paragraphs=len(paragraphs),
            total_sentences=len(sentences),
            total_words=len(words),
            average_paragraph_length=avg_paragraph_length,
            average_sentence_length=avg_sentence_length,
            section_hierarchy=section_hierarchy,
            content_flow_score=content_flow_score,
            structural_balance=structural_balance
        )
    
    async def _analyze_sentiment(self, content: str) -> SentimentAnalysis:
        """Analyze content sentiment."""
        
        # Overall sentiment
        sentiment_scores = self.sentiment_analyzer.polarity_scores(content)
        
        overall_sentiment = "neutral"
        if sentiment_scores['compound'] > 0.05:
            overall_sentiment = "positive"
        elif sentiment_scores['compound'] < -0.05:
            overall_sentiment = "negative"
        
        # Sentiment by section
        sections = content.split('\n\n')
        sentiment_by_section = {}
        
        for i, section in enumerate(sections):
            if section.strip():
                section_scores = self.sentiment_analyzer.polarity_scores(section)
                sentiment_by_section[f"section_{i}"] = section_scores['compound']
        
        # Calculate emotional intensity
        emotional_intensity = abs(sentiment_scores['compound'])
        
        return SentimentAnalysis(
            overall_sentiment=overall_sentiment,
            sentiment_score=sentiment_scores['compound'],
            positive_percentage=sentiment_scores['pos'] * 100,
            negative_percentage=sentiment_scores['neg'] * 100,
            neutral_percentage=sentiment_scores['neu'] * 100,
            emotional_intensity=emotional_intensity,
            sentiment_by_section=sentiment_by_section
        )
    
    async def _analyze_keywords(self, content: str) -> KeywordAnalysis:
        """Analyze keywords and key phrases."""
        
        # Extract keywords using YAKE
        keywords = self.keyword_extractor.extract_keywords(content)
        top_keywords = [(kw[1], kw[0]) for kw in keywords[:10]]  # (keyword, score)
        
        # Extract key phrases using TF-IDF
        if self.nlp:
            doc = self.nlp(content)
            phrases = [chunk.text for chunk in doc.noun_chunks if len(chunk.text.split()) >= 2]
        else:
            # Basic phrase extraction
            phrases = re.findall(r'\b[a-zA-Z]+(?:\s+[a-zA-Z]+){1,2}\b', content)
        
        # Calculate phrase scores
        phrase_scores = Counter(phrases)
        key_phrases = [(phrase, count) for phrase, count in phrase_scores.most_common(10)]
        
        # Calculate keyword density
        words = content.lower().split()
        word_count = len(words)
        keyword_density = {}
        
        for keyword, _ in top_keywords:
            keyword_lower = keyword.lower()
            count = words.count(keyword_lower)
            density = (count / word_count) * 100 if word_count > 0 else 0
            keyword_density[keyword] = density
        
        # Keyword distribution
        keyword_distribution = dict(phrase_scores.most_common(20))
        
        # Semantic keywords (simplified)
        semantic_keywords = [kw[0] for kw in top_keywords[:5]]
        
        return KeywordAnalysis(
            top_keywords=top_keywords,
            key_phrases=key_phrases,
            keyword_density=keyword_density,
            keyword_distribution=keyword_distribution,
            semantic_keywords=semantic_keywords,
            trending_keywords=[]  # Would require external data
        )
    
    async def _analyze_topics(self, content: str) -> TopicAnalysis:
        """Analyze topics using topic modeling."""
        
        try:
            # Prepare documents for topic modeling
            sentences = self._split_into_sentences(content)
            documents = [sent for sent in sentences if len(sent.split()) > 5]
            
            if len(documents) < 3:
                # Not enough content for topic modeling
                return TopicAnalysis(
                    main_topics=["General Content"],
                    topic_distribution={"General Content": 1.0},
                    topic_keywords={"General Content": ["content", "information"]},
                    topic_coherence=0.0,
                    topic_diversity=0.0,
                    dominant_topic="General Content"
                )
            
            # Create TF-IDF matrix
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(documents)
            
            # Apply LDA topic modeling
            n_topics = min(5, len(documents) // 3)  # Reasonable number of topics
            lda = LatentDirichletAllocation(
                n_components=n_topics,
                random_state=42,
                max_iter=10
            )
            lda.fit(tfidf_matrix)
            
            # Extract topics
            feature_names = self.tfidf_vectorizer.get_feature_names_out()
            main_topics = []
            topic_keywords = {}
            
            for topic_idx, topic in enumerate(lda.components_):
                top_words_idx = topic.argsort()[-10:][::-1]
                top_words = [feature_names[i] for i in top_words_idx]
                topic_name = f"Topic {topic_idx + 1}"
                main_topics.append(topic_name)
                topic_keywords[topic_name] = top_words
            
            # Calculate topic distribution
            doc_topic_probs = lda.transform(tfidf_matrix)
            topic_distribution = {}
            for i, topic in enumerate(main_topics):
                topic_distribution[topic] = float(np.mean(doc_topic_probs[:, i]))
            
            # Find dominant topic
            dominant_topic = max(topic_distribution, key=topic_distribution.get)
            
            # Calculate topic coherence and diversity
            topic_coherence = self._calculate_topic_coherence(topic_keywords)
            topic_diversity = self._calculate_topic_diversity(topic_distribution)
            
            return TopicAnalysis(
                main_topics=main_topics,
                topic_distribution=topic_distribution,
                topic_keywords=topic_keywords,
                topic_coherence=topic_coherence,
                topic_diversity=topic_diversity,
                dominant_topic=dominant_topic
            )
            
        except Exception as e:
            logger.error(f"Topic analysis failed: {e}")
            # Return default topic analysis
            return TopicAnalysis(
                main_topics=["General Content"],
                topic_distribution={"General Content": 1.0},
                topic_keywords={"General Content": ["content", "information"]},
                topic_coherence=0.0,
                topic_diversity=0.0,
                dominant_topic="General Content"
            )
    
    async def _analyze_quality(self, content: str, content_type: ContentType) -> QualityMetrics:
        """Analyze overall content quality."""
        
        # Content relevance (simplified)
        content_relevance = 0.8  # Would require domain-specific analysis
        
        # Information density
        words = content.split()
        sentences = self._split_into_sentences(content)
        information_density = len(words) / len(sentences) if sentences else 0
        information_density = min(1.0, information_density / 20)  # Normalize
        
        # Clarity score (based on readability)
        readability_score = flesch_reading_ease(content)
        clarity_score = min(1.0, readability_score / 100)
        
        # Engagement score (based on sentence variety and structure)
        engagement_score = self._calculate_engagement_score(content)
        
        # Credibility score (simplified)
        credibility_score = 0.7  # Would require fact-checking integration
        
        # Accessibility score
        accessibility_score = self._calculate_accessibility_score(content)
        
        # SEO score
        seo_score = self._calculate_seo_score(content)
        
        # Overall quality score
        overall_quality_score = (
            content_relevance * 0.2 +
            information_density * 0.15 +
            clarity_score * 0.2 +
            engagement_score * 0.15 +
            credibility_score * 0.1 +
            accessibility_score * 0.1 +
            seo_score * 0.1
        )
        
        return QualityMetrics(
            overall_quality_score=overall_quality_score,
            content_relevance=content_relevance,
            information_density=information_density,
            clarity_score=clarity_score,
            engagement_score=engagement_score,
            credibility_score=credibility_score,
            accessibility_score=accessibility_score,
            seo_score=seo_score,
            readability_score=clarity_score
        )
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        if self.nlp:
            doc = self.nlp(text)
            return [sent.text for sent in doc.sents]
        else:
            sentences = re.split(r'[.!?]+', text)
            return [s.strip() for s in sentences if s.strip()]
    
    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word (simplified)."""
        word = word.lower()
        vowels = "aeiouy"
        syllable_count = 0
        prev_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_was_vowel:
                syllable_count += 1
            prev_was_vowel = is_vowel
        
        # Handle silent 'e'
        if word.endswith('e') and syllable_count > 1:
            syllable_count -= 1
        
        return max(1, syllable_count)
    
    def _calculate_ari(self, content: str) -> float:
        """Calculate Automated Readability Index."""
        words = content.split()
        sentences = self._split_into_sentences(content)
        characters = len(content.replace(' ', ''))
        
        if sentences and words:
            return 4.71 * (characters / len(words)) + 0.5 * (len(words) / len(sentences)) - 21.43
        return 0
    
    def _calculate_cli(self, content: str) -> float:
        """Calculate Coleman-Liau Index."""
        words = content.split()
        sentences = self._split_into_sentences(content)
        characters = len(content.replace(' ', ''))
        
        if words and sentences:
            l = (characters / len(words)) * 100
            s = (len(sentences) / len(words)) * 100
            return 0.0588 * l - 0.296 * s - 15.8
        return 0
    
    def _calculate_smog(self, content: str) -> float:
        """Calculate SMOG Index."""
        sentences = self._split_into_sentences(content)
        complex_words = sum(1 for word in content.split() if self._count_syllables(word) > 2)
        
        if sentences:
            return 1.043 * (complex_words * (30 / len(sentences))) ** 0.5 + 3.1291
        return 0
    
    def _analyze_section_hierarchy(self, content: str) -> Dict[str, int]:
        """Analyze section hierarchy in content."""
        hierarchy = {}
        
        # Look for heading patterns
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('#'):
                level = len(line) - len(line.lstrip('#'))
                hierarchy[f"h{level}"] = hierarchy.get(f"h{level}", 0) + 1
            elif line.isupper() and len(line) > 3:
                hierarchy["uppercase_heading"] = hierarchy.get("uppercase_heading", 0) + 1
        
        return hierarchy
    
    def _calculate_content_flow_score(self, paragraphs: List[str]) -> float:
        """Calculate content flow score."""
        if len(paragraphs) < 2:
            return 1.0
        
        # Calculate transition words usage
        transition_words = ['however', 'therefore', 'furthermore', 'moreover', 'consequently', 'meanwhile', 'additionally']
        transition_count = 0
        
        for para in paragraphs:
            para_lower = para.lower()
            for word in transition_words:
                if word in para_lower:
                    transition_count += 1
        
        # Normalize score
        max_transitions = len(paragraphs) - 1
        flow_score = min(1.0, transition_count / max_transitions) if max_transitions > 0 else 0
        
        return flow_score
    
    def _calculate_structural_balance(self, paragraphs: List[str]) -> float:
        """Calculate structural balance score."""
        if not paragraphs:
            return 0.0
        
        # Calculate paragraph length variance
        lengths = [len(para.split()) for para in paragraphs]
        if len(lengths) < 2:
            return 1.0
        
        mean_length = np.mean(lengths)
        variance = np.var(lengths)
        
        # Lower variance = better balance
        balance_score = 1.0 / (1.0 + variance / (mean_length ** 2))
        
        return balance_score
    
    def _calculate_topic_coherence(self, topic_keywords: Dict[str, List[str]]) -> float:
        """Calculate topic coherence score."""
        # Simplified coherence calculation
        total_coherence = 0.0
        topic_count = 0
        
        for topic, keywords in topic_keywords.items():
            if len(keywords) > 1:
                # Calculate semantic similarity between keywords
                coherence = 0.5  # Simplified - would use actual semantic similarity
                total_coherence += coherence
                topic_count += 1
        
        return total_coherence / topic_count if topic_count > 0 else 0.0
    
    def _calculate_topic_diversity(self, topic_distribution: Dict[str, float]) -> float:
        """Calculate topic diversity score."""
        if not topic_distribution:
            return 0.0
        
        # Calculate entropy
        probabilities = list(topic_distribution.values())
        entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
        
        # Normalize by maximum possible entropy
        max_entropy = np.log2(len(probabilities))
        diversity = entropy / max_entropy if max_entropy > 0 else 0
        
        return diversity
    
    def _calculate_engagement_score(self, content: str) -> float:
        """Calculate engagement score."""
        sentences = self._split_into_sentences(content)
        words = content.split()
        
        # Sentence variety (length variation)
        sentence_lengths = [len(sent.split()) for sent in sentences]
        if len(sentence_lengths) > 1:
            length_variance = np.var(sentence_lengths)
            length_mean = np.mean(sentence_lengths)
            variety_score = min(1.0, length_variance / (length_mean ** 2))
        else:
            variety_score = 0.5
        
        # Question and exclamation usage
        questions = content.count('?')
        exclamations = content.count('!')
        total_sentences = len(sentences)
        
        if total_sentences > 0:
            punctuation_score = min(1.0, (questions + exclamations) / total_sentences * 10)
        else:
            punctuation_score = 0.0
        
        # Combine scores
        engagement_score = (variety_score * 0.7 + punctuation_score * 0.3)
        
        return engagement_score
    
    def _calculate_accessibility_score(self, content: str) -> float:
        """Calculate accessibility score."""
        score = 1.0
        
        # Check for alt text indicators (simplified)
        if '[image]' in content or '[figure]' in content:
            score -= 0.1
        
        # Check for heading structure
        if not any(line.startswith('#') for line in content.split('\n')):
            score -= 0.2
        
        # Check for list structures
        if not any(line.startswith(('-', '*', '1.')) for line in content.split('\n')):
            score -= 0.1
        
        return max(0.0, score)
    
    def _calculate_seo_score(self, content: str) -> float:
        """Calculate SEO score."""
        score = 0.0
        
        # Word count
        word_count = len(content.split())
        if 300 <= word_count <= 2000:
            score += 0.3
        elif word_count > 2000:
            score += 0.2
        
        # Heading structure
        lines = content.split('\n')
        has_h1 = any(line.startswith('# ') for line in lines)
        has_headings = any(line.startswith('#') for line in lines)
        
        if has_h1:
            score += 0.3
        elif has_headings:
            score += 0.2
        
        # Keyword density (simplified)
        words = content.lower().split()
        if words:
            unique_words = len(set(words))
            density = unique_words / len(words)
            if 0.3 <= density <= 0.7:
                score += 0.2
        
        # Meta information (would require HTML analysis)
        score += 0.2  # Default score
        
        return min(1.0, score)
    
    async def _generate_recommendations(
        self,
        readability_metrics: ReadabilityMetrics,
        content_structure: ContentStructure,
        sentiment_analysis: Optional[SentimentAnalysis],
        keyword_analysis: Optional[KeywordAnalysis],
        topic_analysis: Optional[TopicAnalysis],
        quality_metrics: Optional[QualityMetrics],
        content_type: ContentType
    ) -> List[str]:
        """Generate optimization recommendations."""
        
        recommendations = []
        
        # Readability recommendations
        if readability_metrics.flesch_reading_ease < 30:
            recommendations.append("Content is very difficult to read. Consider simplifying sentence structure and vocabulary.")
        elif readability_metrics.flesch_reading_ease > 80:
            recommendations.append("Content is very easy to read. Consider adding more sophisticated language for professional documents.")
        
        if readability_metrics.average_sentence_length > 25:
            recommendations.append("Average sentence length is too long. Break down complex sentences for better readability.")
        
        # Structure recommendations
        if content_structure.average_paragraph_length > 150:
            recommendations.append("Paragraphs are too long. Break them into smaller, more digestible chunks.")
        
        if content_structure.content_flow_score < 0.3:
            recommendations.append("Improve content flow by adding transition words and phrases between sections.")
        
        # Sentiment recommendations
        if sentiment_analysis:
            if sentiment_analysis.overall_sentiment == "negative" and content_type in [ContentType.MARKETING_CONTENT, ContentType.BLOG_POST]:
                recommendations.append("Content has negative sentiment. Consider adjusting tone for better engagement.")
        
        # Keyword recommendations
        if keyword_analysis:
            if len(keyword_analysis.top_keywords) < 5:
                recommendations.append("Add more relevant keywords to improve content discoverability.")
            
            high_density_keywords = [kw for kw, density in keyword_analysis.keyword_density.items() if density > 3]
            if high_density_keywords:
                recommendations.append(f"Reduce keyword density for: {', '.join(high_density_keywords[:3])}")
        
        # Quality recommendations
        if quality_metrics:
            if quality_metrics.engagement_score < 0.5:
                recommendations.append("Improve engagement by varying sentence structure and adding interactive elements.")
            
            if quality_metrics.accessibility_score < 0.7:
                recommendations.append("Improve accessibility by adding proper headings and structure.")
        
        return recommendations
    
    async def _generate_insights(
        self,
        readability_metrics: ReadabilityMetrics,
        content_structure: ContentStructure,
        sentiment_analysis: Optional[SentimentAnalysis],
        keyword_analysis: Optional[KeywordAnalysis],
        topic_analysis: Optional[TopicAnalysis],
        quality_metrics: Optional[QualityMetrics],
        content_type: ContentType
    ) -> List[str]:
        """Generate content insights."""
        
        insights = []
        
        # Readability insights
        if readability_metrics.flesch_reading_ease > 60:
            insights.append("Content is accessible to a general audience with basic reading skills.")
        elif readability_metrics.flesch_reading_ease < 30:
            insights.append("Content requires advanced reading skills and may limit audience reach.")
        
        # Structure insights
        if content_structure.total_sections > 5:
            insights.append("Well-structured content with clear section organization.")
        
        if content_structure.structural_balance > 0.7:
            insights.append("Content has good structural balance with consistent paragraph lengths.")
        
        # Sentiment insights
        if sentiment_analysis:
            if sentiment_analysis.emotional_intensity > 0.5:
                insights.append("Content has strong emotional impact and may be highly engaging.")
            elif sentiment_analysis.emotional_intensity < 0.2:
                insights.append("Content maintains neutral tone, suitable for professional contexts.")
        
        # Topic insights
        if topic_analysis:
            if topic_analysis.topic_diversity > 0.7:
                insights.append("Content covers diverse topics, providing comprehensive coverage.")
            elif topic_analysis.topic_diversity < 0.3:
                insights.append("Content is highly focused on specific topics.")
        
        # Quality insights
        if quality_metrics:
            if quality_metrics.overall_quality_score > 0.8:
                insights.append("High-quality content with excellent overall metrics.")
            elif quality_metrics.overall_quality_score < 0.5:
                insights.append("Content quality has room for improvement across multiple dimensions.")
        
        return insights

# Global content analytics engine instance
_global_content_analytics: Optional[ContentAnalyticsEngine] = None

def get_global_content_analytics() -> ContentAnalyticsEngine:
    """Get the global content analytics engine instance."""
    global _global_content_analytics
    if _global_content_analytics is None:
        _global_content_analytics = ContentAnalyticsEngine()
    return _global_content_analytics



























