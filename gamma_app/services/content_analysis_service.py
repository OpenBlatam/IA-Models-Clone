"""
Gamma App - Content Analysis Service
Advanced content analysis and insights generation
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import re
import json
import time
from collections import Counter, defaultdict
import numpy as np
from textstat import flesch_reading_ease, flesch_kincaid_grade, gunning_fog
import spacy
from transformers import pipeline, AutoTokenizer, AutoModel
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
import networkx as nx
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from pathlib import Path

logger = logging.getLogger(__name__)

class AnalysisType(Enum):
    """Analysis types"""
    SENTIMENT = "sentiment"
    TOPICS = "topics"
    KEYWORDS = "keywords"
    READABILITY = "readability"
    ENTITIES = "entities"
    RELATIONSHIPS = "relationships"
    QUALITY = "quality"
    ENGAGEMENT = "engagement"

class ContentType(Enum):
    """Content types"""
    TEXT = "text"
    PRESENTATION = "presentation"
    DOCUMENT = "document"
    WEB_PAGE = "web_page"
    EMAIL = "email"
    SOCIAL_MEDIA = "social_media"

@dataclass
class SentimentAnalysis:
    """Sentiment analysis result"""
    positive: float
    negative: float
    neutral: float
    compound: float
    label: str
    confidence: float

@dataclass
class TopicAnalysis:
    """Topic analysis result"""
    topics: List[Dict[str, Any]]
    dominant_topic: str
    topic_distribution: Dict[str, float]
    coherence_score: float

@dataclass
class KeywordAnalysis:
    """Keyword analysis result"""
    keywords: List[Dict[str, Any]]
    key_phrases: List[str]
    word_frequency: Dict[str, int]
    tfidf_scores: Dict[str, float]

@dataclass
class ReadabilityAnalysis:
    """Readability analysis result"""
    flesch_reading_ease: float
    flesch_kincaid_grade: float
    gunning_fog: float
    average_sentence_length: float
    average_word_length: float
    reading_level: str

@dataclass
class EntityAnalysis:
    """Entity analysis result"""
    entities: List[Dict[str, Any]]
    entity_types: Dict[str, int]
    entity_relationships: List[Dict[str, Any]]
    named_entities: List[str]

@dataclass
class QualityAnalysis:
    """Content quality analysis result"""
    overall_score: float
    grammar_score: float
    clarity_score: float
    structure_score: float
    engagement_score: float
    recommendations: List[str]

@dataclass
class ContentInsights:
    """Comprehensive content insights"""
    content_type: ContentType
    analysis_type: AnalysisType
    sentiment: Optional[SentimentAnalysis] = None
    topics: Optional[TopicAnalysis] = None
    keywords: Optional[KeywordAnalysis] = None
    readability: Optional[ReadabilityAnalysis] = None
    entities: Optional[EntityAnalysis] = None
    quality: Optional[QualityAnalysis] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class ContentAnalysisService:
    """Advanced content analysis service"""
    
    def __init__(self):
        self.nlp = None
        self.sentiment_analyzer = None
        self.topic_model = None
        self.vectorizer = None
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize NLP models"""
        try:
            # Load spaCy model
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("spaCy model not found, using basic tokenization")
                self.nlp = None
            
            # Load sentiment analysis model
            try:
                self.sentiment_analyzer = pipeline(
                    "sentiment-analysis",
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                    return_all_scores=True
                )
            except Exception as e:
                logger.warning(f"Could not load sentiment analyzer: {e}")
                self.sentiment_analyzer = None
            
            # Initialize TF-IDF vectorizer
            self.vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            logger.info("Content analysis models initialized")
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
    
    async def analyze_content(
        self,
        content: str,
        content_type: ContentType = ContentType.TEXT,
        analysis_types: List[AnalysisType] = None
    ) -> ContentInsights:
        """Analyze content comprehensively"""
        try:
            if analysis_types is None:
                analysis_types = list(AnalysisType)
            
            logger.info(f"Starting content analysis for {content_type.value}")
            
            insights = ContentInsights(
                content_type=content_type,
                analysis_type=AnalysisType.SENTIMENT,  # Default
                metadata={
                    'content_length': len(content),
                    'word_count': len(content.split()),
                    'sentence_count': len(re.split(r'[.!?]+', content)),
                    'analysis_timestamp': time.time()
                }
            )
            
            # Run analyses
            if AnalysisType.SENTIMENT in analysis_types:
                insights.sentiment = await self._analyze_sentiment(content)
            
            if AnalysisType.TOPICS in analysis_types:
                insights.topics = await self._analyze_topics(content)
            
            if AnalysisType.KEYWORDS in analysis_types:
                insights.keywords = await self._analyze_keywords(content)
            
            if AnalysisType.READABILITY in analysis_types:
                insights.readability = await self._analyze_readability(content)
            
            if AnalysisType.ENTITIES in analysis_types:
                insights.entities = await self._analyze_entities(content)
            
            if AnalysisType.QUALITY in analysis_types:
                insights.quality = await self._analyze_quality(content)
            
            logger.info("Content analysis completed")
            return insights
            
        except Exception as e:
            logger.error(f"Error analyzing content: {e}")
            raise
    
    async def _analyze_sentiment(self, content: str) -> SentimentAnalysis:
        """Analyze sentiment of content"""
        try:
            if self.sentiment_analyzer is None:
                # Fallback to simple sentiment analysis
                return self._simple_sentiment_analysis(content)
            
            # Use transformer model
            results = self.sentiment_analyzer(content)
            
            # Process results
            sentiment_scores = {}
            for result in results[0]:
                label = result['label'].lower()
                score = result['score']
                sentiment_scores[label] = score
            
            # Calculate compound score
            compound = sentiment_scores.get('positive', 0) - sentiment_scores.get('negative', 0)
            
            # Determine dominant label
            dominant_label = max(sentiment_scores.keys(), key=lambda k: sentiment_scores[k])
            confidence = sentiment_scores[dominant_label]
            
            return SentimentAnalysis(
                positive=sentiment_scores.get('positive', 0),
                negative=sentiment_scores.get('negative', 0),
                neutral=sentiment_scores.get('neutral', 0),
                compound=compound,
                label=dominant_label,
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return self._simple_sentiment_analysis(content)
    
    def _simple_sentiment_analysis(self, content: str) -> SentimentAnalysis:
        """Simple sentiment analysis fallback"""
        try:
            # Simple keyword-based sentiment
            positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love', 'like']
            negative_words = ['bad', 'terrible', 'awful', 'horrible', 'hate', 'dislike', 'poor', 'worst']
            
            words = content.lower().split()
            positive_count = sum(1 for word in words if word in positive_words)
            negative_count = sum(1 for word in words if word in negative_words)
            total_words = len(words)
            
            if total_words == 0:
                return SentimentAnalysis(0, 0, 1, 0, 'neutral', 0.5)
            
            positive_score = positive_count / total_words
            negative_score = negative_count / total_words
            neutral_score = 1 - positive_score - negative_score
            
            compound = positive_score - negative_score
            
            if compound > 0.1:
                label = 'positive'
                confidence = positive_score
            elif compound < -0.1:
                label = 'negative'
                confidence = negative_score
            else:
                label = 'neutral'
                confidence = neutral_score
            
            return SentimentAnalysis(
                positive=positive_score,
                negative=negative_score,
                neutral=neutral_score,
                compound=compound,
                label=label,
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"Error in simple sentiment analysis: {e}")
            return SentimentAnalysis(0, 0, 1, 0, 'neutral', 0.5)
    
    async def _analyze_topics(self, content: str) -> TopicAnalysis:
        """Analyze topics in content"""
        try:
            # Prepare text for topic modeling
            sentences = re.split(r'[.!?]+', content)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            if len(sentences) < 2:
                return TopicAnalysis(
                    topics=[],
                    dominant_topic="",
                    topic_distribution={},
                    coherence_score=0.0
                )
            
            # Vectorize text
            tfidf_matrix = self.vectorizer.fit_transform(sentences)
            
            # Apply LDA
            n_topics = min(5, len(sentences) // 2)
            if n_topics < 2:
                n_topics = 2
            
            lda = LatentDirichletAllocation(
                n_components=n_topics,
                random_state=42,
                max_iter=10
            )
            lda.fit(tfidf_matrix)
            
            # Extract topics
            feature_names = self.vectorizer.get_feature_names_out()
            topics = []
            
            for topic_idx, topic in enumerate(lda.components_):
                top_words_idx = topic.argsort()[-10:][::-1]
                top_words = [feature_names[i] for i in top_words_idx]
                top_weights = [topic[i] for i in top_words_idx]
                
                topics.append({
                    'topic_id': topic_idx,
                    'words': top_words,
                    'weights': top_weights,
                    'coherence': np.mean(top_weights)
                })
            
            # Find dominant topic
            doc_topic_probs = lda.transform(tfidf_matrix)
            topic_distribution = np.mean(doc_topic_probs, axis=0)
            dominant_topic_idx = np.argmax(topic_distribution)
            dominant_topic = topics[dominant_topic_idx]['words'][0] if topics else ""
            
            # Calculate coherence score
            coherence_score = np.mean([topic['coherence'] for topic in topics])
            
            return TopicAnalysis(
                topics=topics,
                dominant_topic=dominant_topic,
                topic_distribution={f"topic_{i}": float(prob) for i, prob in enumerate(topic_distribution)},
                coherence_score=float(coherence_score)
            )
            
        except Exception as e:
            logger.error(f"Error analyzing topics: {e}")
            return TopicAnalysis(
                topics=[],
                dominant_topic="",
                topic_distribution={},
                coherence_score=0.0
            )
    
    async def _analyze_keywords(self, content: str) -> KeywordAnalysis:
        """Analyze keywords and key phrases"""
        try:
            # Clean and tokenize text
            words = re.findall(r'\b[a-zA-Z]+\b', content.lower())
            
            # Remove stop words
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
            filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
            
            # Calculate word frequency
            word_frequency = Counter(filtered_words)
            
            # Get top keywords
            top_keywords = word_frequency.most_common(20)
            keywords = [
                {
                    'word': word,
                    'frequency': freq,
                    'importance': freq / len(filtered_words)
                }
                for word, freq in top_keywords
            ]
            
            # Extract key phrases (bigrams)
            bigrams = []
            words_list = content.lower().split()
            for i in range(len(words_list) - 1):
                if len(words_list[i]) > 2 and len(words_list[i + 1]) > 2:
                    bigram = f"{words_list[i]} {words_list[i + 1]}"
                    bigrams.append(bigram)
            
            key_phrases = [phrase for phrase, count in Counter(bigrams).most_common(10)]
            
            # Calculate TF-IDF scores
            tfidf_matrix = self.vectorizer.fit_transform([content])
            feature_names = self.vectorizer.get_feature_names_out()
            tfidf_scores = {}
            
            for i, word in enumerate(feature_names):
                tfidf_scores[word] = float(tfidf_matrix[0, i])
            
            return KeywordAnalysis(
                keywords=keywords,
                key_phrases=key_phrases,
                word_frequency=dict(word_frequency),
                tfidf_scores=tfidf_scores
            )
            
        except Exception as e:
            logger.error(f"Error analyzing keywords: {e}")
            return KeywordAnalysis(
                keywords=[],
                key_phrases=[],
                word_frequency={},
                tfidf_scores={}
            )
    
    async def _analyze_readability(self, content: str) -> ReadabilityAnalysis:
        """Analyze readability of content"""
        try:
            # Calculate readability metrics
            flesch_ease = flesch_reading_ease(content)
            flesch_grade = flesch_kincaid_grade(content)
            gunning_fog_score = gunning_fog(content)
            
            # Calculate average sentence length
            sentences = re.split(r'[.!?]+', content)
            sentences = [s.strip() for s in sentences if s.strip()]
            avg_sentence_length = len(content.split()) / len(sentences) if sentences else 0
            
            # Calculate average word length
            words = content.split()
            avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
            
            # Determine reading level
            if flesch_ease >= 90:
                reading_level = "Very Easy"
            elif flesch_ease >= 80:
                reading_level = "Easy"
            elif flesch_ease >= 70:
                reading_level = "Fairly Easy"
            elif flesch_ease >= 60:
                reading_level = "Standard"
            elif flesch_ease >= 50:
                reading_level = "Fairly Difficult"
            elif flesch_ease >= 30:
                reading_level = "Difficult"
            else:
                reading_level = "Very Difficult"
            
            return ReadabilityAnalysis(
                flesch_reading_ease=flesch_ease,
                flesch_kincaid_grade=flesch_grade,
                gunning_fog=gunning_fog_score,
                average_sentence_length=avg_sentence_length,
                average_word_length=avg_word_length,
                reading_level=reading_level
            )
            
        except Exception as e:
            logger.error(f"Error analyzing readability: {e}")
            return ReadabilityAnalysis(
                flesch_reading_ease=0,
                flesch_kincaid_grade=0,
                gunning_fog=0,
                average_sentence_length=0,
                average_word_length=0,
                reading_level="Unknown"
            )
    
    async def _analyze_entities(self, content: str) -> EntityAnalysis:
        """Analyze named entities in content"""
        try:
            if self.nlp is None:
                # Fallback to simple entity extraction
                return self._simple_entity_analysis(content)
            
            # Use spaCy for entity recognition
            doc = self.nlp(content)
            
            entities = []
            entity_types = defaultdict(int)
            named_entities = []
            
            for ent in doc.ents:
                entity_info = {
                    'text': ent.text,
                    'label': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'confidence': 1.0  # spaCy doesn't provide confidence scores
                }
                entities.append(entity_info)
                entity_types[ent.label_] += 1
                named_entities.append(ent.text)
            
            # Extract entity relationships (simplified)
            entity_relationships = []
            for i, ent1 in enumerate(entities):
                for j, ent2 in enumerate(entities[i+1:], i+1):
                    # Simple co-occurrence based relationship
                    if abs(ent1['start'] - ent2['start']) < 100:  # Within 100 characters
                        relationship = {
                            'entity1': ent1['text'],
                            'entity2': ent2['text'],
                            'relationship_type': 'co_occurrence',
                            'confidence': 0.5
                        }
                        entity_relationships.append(relationship)
            
            return EntityAnalysis(
                entities=entities,
                entity_types=dict(entity_types),
                entity_relationships=entity_relationships,
                named_entities=named_entities
            )
            
        except Exception as e:
            logger.error(f"Error analyzing entities: {e}")
            return self._simple_entity_analysis(content)
    
    def _simple_entity_analysis(self, content: str) -> EntityAnalysis:
        """Simple entity analysis fallback"""
        try:
            # Simple regex-based entity extraction
            entities = []
            entity_types = defaultdict(int)
            named_entities = []
            
            # Extract emails
            email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            emails = re.findall(email_pattern, content)
            for email in emails:
                entity_info = {
                    'text': email,
                    'label': 'EMAIL',
                    'start': content.find(email),
                    'end': content.find(email) + len(email),
                    'confidence': 0.9
                }
                entities.append(entity_info)
                entity_types['EMAIL'] += 1
                named_entities.append(email)
            
            # Extract URLs
            url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
            urls = re.findall(url_pattern, content)
            for url in urls:
                entity_info = {
                    'text': url,
                    'label': 'URL',
                    'start': content.find(url),
                    'end': content.find(url) + len(url),
                    'confidence': 0.9
                }
                entities.append(entity_info)
                entity_types['URL'] += 1
                named_entities.append(url)
            
            # Extract phone numbers
            phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
            phones = re.findall(phone_pattern, content)
            for phone in phones:
                entity_info = {
                    'text': phone,
                    'label': 'PHONE',
                    'start': content.find(phone),
                    'end': content.find(phone) + len(phone),
                    'confidence': 0.8
                }
                entities.append(entity_info)
                entity_types['PHONE'] += 1
                named_entities.append(phone)
            
            return EntityAnalysis(
                entities=entities,
                entity_types=dict(entity_types),
                entity_relationships=[],
                named_entities=named_entities
            )
            
        except Exception as e:
            logger.error(f"Error in simple entity analysis: {e}")
            return EntityAnalysis(
                entities=[],
                entity_types={},
                entity_relationships=[],
                named_entities=[]
            )
    
    async def _analyze_quality(self, content: str) -> QualityAnalysis:
        """Analyze content quality"""
        try:
            scores = {}
            recommendations = []
            
            # Grammar score (simplified)
            grammar_score = self._calculate_grammar_score(content)
            scores['grammar'] = grammar_score
            
            # Clarity score
            clarity_score = self._calculate_clarity_score(content)
            scores['clarity'] = clarity_score
            
            # Structure score
            structure_score = self._calculate_structure_score(content)
            scores['structure'] = structure_score
            
            # Engagement score
            engagement_score = self._calculate_engagement_score(content)
            scores['engagement'] = engagement_score
            
            # Overall score
            overall_score = np.mean(list(scores.values()))
            
            # Generate recommendations
            if grammar_score < 0.7:
                recommendations.append("Improve grammar and sentence structure")
            if clarity_score < 0.7:
                recommendations.append("Use simpler language and shorter sentences")
            if structure_score < 0.7:
                recommendations.append("Improve content organization and flow")
            if engagement_score < 0.7:
                recommendations.append("Add more engaging elements and active voice")
            
            return QualityAnalysis(
                overall_score=overall_score,
                grammar_score=grammar_score,
                clarity_score=clarity_score,
                structure_score=structure_score,
                engagement_score=engagement_score,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Error analyzing quality: {e}")
            return QualityAnalysis(
                overall_score=0.5,
                grammar_score=0.5,
                clarity_score=0.5,
                structure_score=0.5,
                engagement_score=0.5,
                recommendations=["Unable to analyze content quality"]
            )
    
    def _calculate_grammar_score(self, content: str) -> float:
        """Calculate grammar score"""
        try:
            # Simple grammar checks
            sentences = re.split(r'[.!?]+', content)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            if not sentences:
                return 0.0
            
            score = 1.0
            
            # Check for proper capitalization
            proper_caps = sum(1 for s in sentences if s and s[0].isupper())
            cap_score = proper_caps / len(sentences)
            score *= cap_score
            
            # Check for sentence length variety
            lengths = [len(s.split()) for s in sentences]
            if lengths:
                length_variance = np.var(lengths) / np.mean(lengths) if np.mean(lengths) > 0 else 0
                variety_score = min(1.0, length_variance / 10)  # Normalize
                score *= (0.5 + variety_score * 0.5)
            
            return min(1.0, max(0.0, score))
            
        except Exception as e:
            logger.error(f"Error calculating grammar score: {e}")
            return 0.5
    
    def _calculate_clarity_score(self, content: str) -> float:
        """Calculate clarity score"""
        try:
            # Simple clarity metrics
            words = content.split()
            sentences = re.split(r'[.!?]+', content)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            if not words or not sentences:
                return 0.0
            
            # Average sentence length
            avg_sentence_length = len(words) / len(sentences)
            
            # Average word length
            avg_word_length = sum(len(word) for word in words) / len(words)
            
            # Clarity score based on sentence and word length
            sentence_score = max(0, 1 - (avg_sentence_length - 15) / 15)  # Optimal around 15 words
            word_score = max(0, 1 - (avg_word_length - 5) / 5)  # Optimal around 5 characters
            
            return (sentence_score + word_score) / 2
            
        except Exception as e:
            logger.error(f"Error calculating clarity score: {e}")
            return 0.5
    
    def _calculate_structure_score(self, content: str) -> float:
        """Calculate structure score"""
        try:
            # Check for structure indicators
            paragraphs = content.split('\n\n')
            sentences = re.split(r'[.!?]+', content)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            score = 0.0
            
            # Paragraph structure
            if len(paragraphs) > 1:
                score += 0.3
            
            # Sentence variety
            if len(sentences) > 3:
                score += 0.3
            
            # Transition words
            transition_words = ['however', 'therefore', 'furthermore', 'moreover', 'additionally', 'consequently']
            transition_count = sum(1 for word in transition_words if word in content.lower())
            if transition_count > 0:
                score += 0.2
            
            # Lists or bullet points
            if any(char in content for char in ['•', '-', '*', '1.', '2.', '3.']):
                score += 0.2
            
            return min(1.0, score)
            
        except Exception as e:
            logger.error(f"Error calculating structure score: {e}")
            return 0.5
    
    def _calculate_engagement_score(self, content: str) -> float:
        """Calculate engagement score"""
        try:
            score = 0.0
            
            # Active voice
            passive_indicators = ['was', 'were', 'been', 'being', 'have been', 'has been', 'had been']
            passive_count = sum(1 for indicator in passive_indicators if indicator in content.lower())
            total_verbs = len(re.findall(r'\b(am|is|are|was|were|be|been|being|have|has|had|do|does|did|will|would|could|should|may|might|must|can)\b', content.lower()))
            
            if total_verbs > 0:
                active_ratio = 1 - (passive_count / total_verbs)
                score += active_ratio * 0.4
            
            # Questions
            question_count = content.count('?')
            if question_count > 0:
                score += 0.2
            
            # Exclamations
            exclamation_count = content.count('!')
            if exclamation_count > 0:
                score += 0.1
            
            # Personal pronouns
            personal_pronouns = ['i', 'you', 'we', 'us', 'our', 'your', 'my', 'me']
            pronoun_count = sum(1 for pronoun in personal_pronouns if pronoun in content.lower())
            if pronoun_count > 0:
                score += 0.2
            
            # Emotional words
            emotional_words = ['amazing', 'incredible', 'fantastic', 'wonderful', 'exciting', 'thrilling', 'inspiring']
            emotional_count = sum(1 for word in emotional_words if word in content.lower())
            if emotional_count > 0:
                score += 0.1
            
            return min(1.0, score)
            
        except Exception as e:
            logger.error(f"Error calculating engagement score: {e}")
            return 0.5
    
    async def generate_content_report(
        self,
        insights: ContentInsights,
        output_path: str
    ) -> str:
        """Generate comprehensive content analysis report"""
        try:
            report = {
                'content_analysis': {
                    'content_type': insights.content_type.value,
                    'metadata': insights.metadata
                }
            }
            
            # Add sentiment analysis
            if insights.sentiment:
                report['sentiment_analysis'] = {
                    'label': insights.sentiment.label,
                    'confidence': insights.sentiment.confidence,
                    'scores': {
                        'positive': insights.sentiment.positive,
                        'negative': insights.sentiment.negative,
                        'neutral': insights.sentiment.neutral,
                        'compound': insights.sentiment.compound
                    }
                }
            
            # Add topic analysis
            if insights.topics:
                report['topic_analysis'] = {
                    'dominant_topic': insights.topics.dominant_topic,
                    'coherence_score': insights.topics.coherence_score,
                    'topics': insights.topics.topics,
                    'topic_distribution': insights.topics.topic_distribution
                }
            
            # Add keyword analysis
            if insights.keywords:
                report['keyword_analysis'] = {
                    'top_keywords': insights.keywords.keywords[:10],
                    'key_phrases': insights.keywords.key_phrases,
                    'word_frequency': dict(list(insights.keywords.word_frequency.items())[:20])
                }
            
            # Add readability analysis
            if insights.readability:
                report['readability_analysis'] = {
                    'reading_level': insights.readability.reading_level,
                    'flesch_reading_ease': insights.readability.flesch_reading_ease,
                    'flesch_kincaid_grade': insights.readability.flesch_kincaid_grade,
                    'gunning_fog': insights.readability.gunning_fog,
                    'average_sentence_length': insights.readability.average_sentence_length,
                    'average_word_length': insights.readability.average_word_length
                }
            
            # Add entity analysis
            if insights.entities:
                report['entity_analysis'] = {
                    'entity_count': len(insights.entities.entities),
                    'entity_types': insights.entities.entity_types,
                    'top_entities': insights.entities.entities[:10],
                    'named_entities': insights.entities.named_entities[:20]
                }
            
            # Add quality analysis
            if insights.quality:
                report['quality_analysis'] = {
                    'overall_score': insights.quality.overall_score,
                    'scores': {
                        'grammar': insights.quality.grammar_score,
                        'clarity': insights.quality.clarity_score,
                        'structure': insights.quality.structure_score,
                        'engagement': insights.quality.engagement_score
                    },
                    'recommendations': insights.quality.recommendations
                }
            
            # Save report
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Content analysis report saved to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating content report: {e}")
            raise
    
    async def create_word_cloud(
        self,
        content: str,
        output_path: str,
        max_words: int = 100
    ) -> str:
        """Create word cloud from content"""
        try:
            # Generate word cloud
            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color='white',
                max_words=max_words,
                colormap='viridis'
            ).generate(content)
            
            # Save word cloud
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Word cloud saved to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error creating word cloud: {e}")
            raise
    
    def get_analysis_summary(self, insights: ContentInsights) -> Dict[str, Any]:
        """Get summary of content analysis"""
        try:
            summary = {
                'content_type': insights.content_type.value,
                'word_count': insights.metadata.get('word_count', 0),
                'sentence_count': insights.metadata.get('sentence_count', 0),
                'content_length': insights.metadata.get('content_length', 0)
            }
            
            # Add sentiment summary
            if insights.sentiment:
                summary['sentiment'] = {
                    'label': insights.sentiment.label,
                    'confidence': insights.sentiment.confidence
                }
            
            # Add readability summary
            if insights.readability:
                summary['readability'] = {
                    'level': insights.readability.reading_level,
                    'score': insights.readability.flesch_reading_ease
                }
            
            # Add quality summary
            if insights.quality:
                summary['quality'] = {
                    'overall_score': insights.quality.overall_score,
                    'recommendations_count': len(insights.quality.recommendations)
                }
            
            # Add topic summary
            if insights.topics:
                summary['topics'] = {
                    'dominant_topic': insights.topics.dominant_topic,
                    'topic_count': len(insights.topics.topics)
                }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting analysis summary: {e}")
            return {}

# Global content analysis service instance
content_analysis_service = ContentAnalysisService()

async def analyze_content_comprehensive(content: str, content_type: ContentType = ContentType.TEXT, analysis_types: List[AnalysisType] = None) -> ContentInsights:
    """Analyze content using global service"""
    return await content_analysis_service.analyze_content(content, content_type, analysis_types)

async def generate_content_analysis_report(insights: ContentInsights, output_path: str) -> str:
    """Generate content analysis report using global service"""
    return await content_analysis_service.generate_content_report(insights, output_path)

async def create_content_word_cloud(content: str, output_path: str, max_words: int = 100) -> str:
    """Create word cloud using global service"""
    return await content_analysis_service.create_word_cloud(content, output_path, max_words)

def get_content_analysis_summary(insights: ContentInsights) -> Dict[str, Any]:
    """Get content analysis summary using global service"""
    return content_analysis_service.get_analysis_summary(insights)

























