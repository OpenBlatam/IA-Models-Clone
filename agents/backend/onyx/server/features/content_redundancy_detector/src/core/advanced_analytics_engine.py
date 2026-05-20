"""
Advanced Analytics Engine - Enhanced analytics with advanced capabilities
"""

import asyncio
import logging
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import hashlib
from collections import defaultdict, Counter
import statistics
from scipy import stats
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA, LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import networkx as nx
from textstat import flesch_reading_ease, flesch_kincaid_grade, gunning_fog
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy
from transformers import pipeline, AutoTokenizer, AutoModel
import torch
from sentence_transformers import SentenceTransformer
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class AnalyticsConfig:
    """Analytics configuration"""
    enable_advanced_metrics: bool = True
    enable_sentiment_analysis: bool = True
    enable_topic_modeling: bool = True
    enable_network_analysis: bool = True
    enable_visualization: bool = True
    enable_anomaly_detection: bool = True
    enable_trend_analysis: bool = True
    enable_predictive_analytics: bool = True
    max_content_length: int = 10000
    min_content_length: int = 10
    similarity_threshold: float = 0.7
    clustering_eps: float = 0.5
    clustering_min_samples: int = 2
    n_topics: int = 10
    n_clusters: int = 5


@dataclass
class ContentMetrics:
    """Content metrics data class"""
    content_id: str
    timestamp: datetime
    word_count: int
    character_count: int
    sentence_count: int
    paragraph_count: int
    avg_word_length: float
    avg_sentence_length: float
    readability_score: float
    complexity_score: float
    diversity_score: float
    sentiment_score: float
    emotion_scores: Dict[str, float]
    topic_scores: Dict[str, float]
    entity_counts: Dict[str, int]
    keyword_density: Dict[str, float]
    language: str
    quality_score: float


@dataclass
class SimilarityResult:
    """Similarity analysis result"""
    content_id_1: str
    content_id_2: str
    similarity_score: float
    similarity_type: str
    common_words: List[str]
    common_entities: List[str]
    common_topics: List[str]
    difference_analysis: Dict[str, Any]


@dataclass
class ClusteringResult:
    """Clustering analysis result"""
    cluster_id: int
    content_ids: List[str]
    cluster_size: int
    cluster_centroid: List[float]
    cluster_quality: float
    dominant_topics: List[str]
    dominant_sentiments: List[str]
    representative_content: str


@dataclass
class TrendAnalysis:
    """Trend analysis result"""
    trend_type: str
    trend_direction: str
    trend_strength: float
    trend_period: str
    affected_content: List[str]
    trend_factors: List[str]
    prediction_confidence: float
    future_projection: Dict[str, Any]


@dataclass
class AnomalyDetection:
    """Anomaly detection result"""
    anomaly_type: str
    anomaly_score: float
    content_id: str
    anomaly_factors: List[str]
    severity: str
    recommendation: str
    confidence: float


class AdvancedTextProcessor:
    """Advanced text processing capabilities"""
    
    def __init__(self):
        self.nlp = None
        self.sentiment_pipeline = None
        self.sentence_transformer = None
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize NLP models"""
        try:
            # Load spaCy model
            self.nlp = spacy.load("en_core_web_sm")
            
            # Load sentiment analysis pipeline
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                return_all_scores=True
            )
            
            # Load sentence transformer
            self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
            
            logger.info("Advanced text processing models initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing text processing models: {e}")
    
    async def extract_advanced_metrics(self, content: str) -> ContentMetrics:
        """Extract advanced content metrics"""
        start_time = time.time()
        
        # Basic metrics
        word_count = len(content.split())
        character_count = len(content)
        sentence_count = len(re.split(r'[.!?]+', content))
        paragraph_count = len([p for p in content.split('\n\n') if p.strip()])
        
        # Advanced metrics
        avg_word_length = sum(len(word) for word in content.split()) / word_count if word_count > 0 else 0
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
        
        # Readability scores
        readability_score = flesch_reading_ease(content)
        complexity_score = flesch_kincaid_grade(content)
        
        # Diversity score (unique words / total words)
        unique_words = len(set(word.lower() for word in content.split()))
        diversity_score = unique_words / word_count if word_count > 0 else 0
        
        # Sentiment analysis
        sentiment_result = await self._analyze_sentiment(content)
        sentiment_score = sentiment_result.get('score', 0.0)
        emotion_scores = sentiment_result.get('emotions', {})
        
        # Topic analysis
        topic_scores = await self._analyze_topics(content)
        
        # Entity extraction
        entity_counts = await self._extract_entities(content)
        
        # Keyword density
        keyword_density = await self._calculate_keyword_density(content)
        
        # Language detection
        language = await self._detect_language(content)
        
        # Quality score (composite)
        quality_score = self._calculate_quality_score(
            readability_score, complexity_score, diversity_score, 
            sentiment_score, word_count
        )
        
        execution_time = (time.time() - start_time) * 1000
        
        return ContentMetrics(
            content_id=hashlib.md5(content.encode()).hexdigest(),
            timestamp=datetime.now(),
            word_count=word_count,
            character_count=character_count,
            sentence_count=sentence_count,
            paragraph_count=paragraph_count,
            avg_word_length=avg_word_length,
            avg_sentence_length=avg_sentence_length,
            readability_score=readability_score,
            complexity_score=complexity_score,
            diversity_score=diversity_score,
            sentiment_score=sentiment_score,
            emotion_scores=emotion_scores,
            topic_scores=topic_scores,
            entity_counts=entity_counts,
            keyword_density=keyword_density,
            language=language,
            quality_score=quality_score
        )
    
    async def _analyze_sentiment(self, content: str) -> Dict[str, Any]:
        """Analyze sentiment and emotions"""
        try:
            # Truncate content if too long
            truncated_content = content[:512] if len(content) > 512 else content
            
            # Get sentiment scores
            sentiment_results = self.sentiment_pipeline(truncated_content)
            
            # Extract scores
            scores = {}
            emotions = {}
            
            for result in sentiment_results[0]:
                label = result['label'].lower()
                score = result['score']
                scores[label] = score
                
                # Map to emotions
                if 'positive' in label:
                    emotions['joy'] = score
                elif 'negative' in label:
                    emotions['sadness'] = score
                elif 'neutral' in label:
                    emotions['neutral'] = score
            
            # Calculate overall sentiment score
            overall_score = scores.get('positive', 0) - scores.get('negative', 0)
            
            return {
                'score': overall_score,
                'scores': scores,
                'emotions': emotions
            }
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            return {'score': 0.0, 'scores': {}, 'emotions': {}}
    
    async def _analyze_topics(self, content: str) -> Dict[str, float]:
        """Analyze topics using LDA"""
        try:
            # Simple topic analysis based on keywords
            words = word_tokenize(content.lower())
            stop_words = set(stopwords.words('english'))
            filtered_words = [word for word in words if word.isalpha() and word not in stop_words]
            
            # Count word frequencies
            word_freq = Counter(filtered_words)
            
            # Define topic categories
            topics = {
                'technology': ['tech', 'software', 'computer', 'digital', 'ai', 'machine', 'data'],
                'business': ['business', 'company', 'market', 'sales', 'profit', 'revenue'],
                'health': ['health', 'medical', 'doctor', 'patient', 'treatment', 'medicine'],
                'education': ['education', 'school', 'student', 'teacher', 'learning', 'study'],
                'entertainment': ['movie', 'music', 'game', 'entertainment', 'fun', 'enjoy'],
                'sports': ['sport', 'game', 'team', 'player', 'match', 'competition'],
                'politics': ['government', 'political', 'policy', 'election', 'vote', 'democracy'],
                'science': ['science', 'research', 'study', 'experiment', 'discovery', 'theory']
            }
            
            topic_scores = {}
            total_words = len(filtered_words)
            
            for topic, keywords in topics.items():
                score = sum(word_freq.get(keyword, 0) for keyword in keywords) / total_words
                topic_scores[topic] = score
            
            return topic_scores
            
        except Exception as e:
            logger.error(f"Error in topic analysis: {e}")
            return {}
    
    async def _extract_entities(self, content: str) -> Dict[str, int]:
        """Extract named entities"""
        try:
            if not self.nlp:
                return {}
            
            doc = self.nlp(content)
            entities = {}
            
            for ent in doc.ents:
                entity_type = ent.label_
                if entity_type not in entities:
                    entities[entity_type] = 0
                entities[entity_type] += 1
            
            return entities
            
        except Exception as e:
            logger.error(f"Error in entity extraction: {e}")
            return {}
    
    async def _calculate_keyword_density(self, content: str) -> Dict[str, float]:
        """Calculate keyword density"""
        try:
            words = word_tokenize(content.lower())
            stop_words = set(stopwords.words('english'))
            filtered_words = [word for word in words if word.isalpha() and word not in stop_words]
            
            word_freq = Counter(filtered_words)
            total_words = len(filtered_words)
            
            keyword_density = {}
            for word, count in word_freq.most_common(20):  # Top 20 keywords
                density = (count / total_words) * 100
                keyword_density[word] = density
            
            return keyword_density
            
        except Exception as e:
            logger.error(f"Error calculating keyword density: {e}")
            return {}
    
    async def _detect_language(self, content: str) -> str:
        """Detect content language"""
        try:
            # Simple language detection based on common words
            english_words = ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']
            spanish_words = ['el', 'la', 'de', 'que', 'y', 'a', 'en', 'un', 'es', 'se', 'no', 'te']
            french_words = ['le', 'de', 'et', 'à', 'un', 'il', 'être', 'et', 'en', 'avoir', 'que', 'pour']
            
            words = content.lower().split()
            word_count = len(words)
            
            if word_count == 0:
                return 'unknown'
            
            english_score = sum(1 for word in words if word in english_words) / word_count
            spanish_score = sum(1 for word in words if word in spanish_words) / word_count
            french_score = sum(1 for word in words if word in french_words) / word_count
            
            if english_score > spanish_score and english_score > french_score:
                return 'english'
            elif spanish_score > french_score:
                return 'spanish'
            elif french_score > 0.1:
                return 'french'
            else:
                return 'unknown'
                
        except Exception as e:
            logger.error(f"Error detecting language: {e}")
            return 'unknown'
    
    def _calculate_quality_score(self, readability: float, complexity: float, 
                                diversity: float, sentiment: float, word_count: int) -> float:
        """Calculate composite quality score"""
        try:
            # Normalize scores
            readability_norm = max(0, min(1, readability / 100))
            complexity_norm = max(0, min(1, 1 - (complexity / 20)))
            diversity_norm = max(0, min(1, diversity))
            sentiment_norm = max(0, min(1, (sentiment + 1) / 2))
            length_norm = max(0, min(1, word_count / 1000))
            
            # Weighted average
            quality_score = (
                readability_norm * 0.25 +
                complexity_norm * 0.20 +
                diversity_norm * 0.20 +
                sentiment_norm * 0.20 +
                length_norm * 0.15
            )
            
            return round(quality_score * 100, 2)
            
        except Exception as e:
            logger.error(f"Error calculating quality score: {e}")
            return 0.0


class SimilarityAnalyzer:
    """Advanced similarity analysis"""
    
    def __init__(self, text_processor: AdvancedTextProcessor):
        self.text_processor = text_processor
    
    async def analyze_similarity(self, content1: str, content2: str) -> SimilarityResult:
        """Analyze similarity between two content pieces"""
        start_time = time.time()
        
        # Extract metrics for both contents
        metrics1 = await self.text_processor.extract_advanced_metrics(content1)
        metrics2 = await self.text_processor.extract_advanced_metrics(content2)
        
        # Calculate different types of similarity
        semantic_similarity = await self._calculate_semantic_similarity(content1, content2)
        structural_similarity = self._calculate_structural_similarity(metrics1, metrics2)
        topical_similarity = self._calculate_topical_similarity(metrics1, metrics2)
        
        # Overall similarity score
        overall_similarity = (semantic_similarity + structural_similarity + topical_similarity) / 3
        
        # Find common elements
        common_words = self._find_common_words(content1, content2)
        common_entities = self._find_common_entities(metrics1, metrics2)
        common_topics = self._find_common_topics(metrics1, metrics2)
        
        # Difference analysis
        difference_analysis = self._analyze_differences(metrics1, metrics2)
        
        execution_time = (time.time() - start_time) * 1000
        
        return SimilarityResult(
            content_id_1=metrics1.content_id,
            content_id_2=metrics2.content_id,
            similarity_score=overall_similarity,
            similarity_type="composite",
            common_words=common_words,
            common_entities=common_entities,
            common_topics=common_topics,
            difference_analysis=difference_analysis
        )
    
    async def _calculate_semantic_similarity(self, content1: str, content2: str) -> float:
        """Calculate semantic similarity using embeddings"""
        try:
            if not self.text_processor.sentence_transformer:
                return 0.0
            
            # Generate embeddings
            embeddings = self.text_processor.sentence_transformer.encode([content1, content2])
            
            # Calculate cosine similarity
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error calculating semantic similarity: {e}")
            return 0.0
    
    def _calculate_structural_similarity(self, metrics1: ContentMetrics, metrics2: ContentMetrics) -> float:
        """Calculate structural similarity"""
        try:
            # Compare structural features
            word_count_sim = 1 - abs(metrics1.word_count - metrics2.word_count) / max(metrics1.word_count, metrics2.word_count)
            sentence_count_sim = 1 - abs(metrics1.sentence_count - metrics2.sentence_count) / max(metrics1.sentence_count, metrics2.sentence_count)
            readability_sim = 1 - abs(metrics1.readability_score - metrics2.readability_score) / 100
            
            structural_similarity = (word_count_sim + sentence_count_sim + readability_sim) / 3
            return max(0, min(1, structural_similarity))
            
        except Exception as e:
            logger.error(f"Error calculating structural similarity: {e}")
            return 0.0
    
    def _calculate_topical_similarity(self, metrics1: ContentMetrics, metrics2: ContentMetrics) -> float:
        """Calculate topical similarity"""
        try:
            topics1 = set(metrics1.topic_scores.keys())
            topics2 = set(metrics2.topic_scores.keys())
            
            if not topics1 or not topics2:
                return 0.0
            
            # Jaccard similarity for topics
            intersection = topics1.intersection(topics2)
            union = topics1.union(topics2)
            
            jaccard_similarity = len(intersection) / len(union) if union else 0.0
            
            # Weighted similarity based on topic scores
            weighted_similarity = 0.0
            for topic in intersection:
                score1 = metrics1.topic_scores.get(topic, 0)
                score2 = metrics2.topic_scores.get(topic, 0)
                weighted_similarity += min(score1, score2)
            
            # Combine Jaccard and weighted similarity
            topical_similarity = (jaccard_similarity + weighted_similarity) / 2
            return max(0, min(1, topical_similarity))
            
        except Exception as e:
            logger.error(f"Error calculating topical similarity: {e}")
            return 0.0
    
    def _find_common_words(self, content1: str, content2: str) -> List[str]:
        """Find common words between contents"""
        try:
            words1 = set(word.lower() for word in content1.split() if word.isalpha())
            words2 = set(word.lower() for word in content2.split() if word.isalpha())
            
            common_words = list(words1.intersection(words2))
            return sorted(common_words)[:20]  # Top 20 common words
            
        except Exception as e:
            logger.error(f"Error finding common words: {e}")
            return []
    
    def _find_common_entities(self, metrics1: ContentMetrics, metrics2: ContentMetrics) -> List[str]:
        """Find common entities between contents"""
        try:
            entities1 = set(metrics1.entity_counts.keys())
            entities2 = set(metrics2.entity_counts.keys())
            
            common_entities = list(entities1.intersection(entities2))
            return sorted(common_entities)
            
        except Exception as e:
            logger.error(f"Error finding common entities: {e}")
            return []
    
    def _find_common_topics(self, metrics1: ContentMetrics, metrics2: ContentMetrics) -> List[str]:
        """Find common topics between contents"""
        try:
            topics1 = set(metrics1.topic_scores.keys())
            topics2 = set(metrics2.topic_scores.keys())
            
            common_topics = list(topics1.intersection(topics2))
            return sorted(common_topics)
            
        except Exception as e:
            logger.error(f"Error finding common topics: {e}")
            return []
    
    def _analyze_differences(self, metrics1: ContentMetrics, metrics2: ContentMetrics) -> Dict[str, Any]:
        """Analyze differences between contents"""
        try:
            differences = {
                'word_count_diff': metrics1.word_count - metrics2.word_count,
                'readability_diff': metrics1.readability_score - metrics2.readability_score,
                'sentiment_diff': metrics1.sentiment_score - metrics2.sentiment_score,
                'quality_diff': metrics1.quality_score - metrics2.quality_score,
                'language_match': metrics1.language == metrics2.language,
                'topic_differences': list(set(metrics1.topic_scores.keys()) - set(metrics2.topic_scores.keys())),
                'entity_differences': list(set(metrics1.entity_counts.keys()) - set(metrics2.entity_counts.keys()))
            }
            
            return differences
            
        except Exception as e:
            logger.error(f"Error analyzing differences: {e}")
            return {}


class ClusteringAnalyzer:
    """Advanced clustering analysis"""
    
    def __init__(self, text_processor: AdvancedTextProcessor):
        self.text_processor = text_processor
    
    async def cluster_content(self, contents: List[str], method: str = "dbscan") -> List[ClusteringResult]:
        """Cluster content based on similarity"""
        start_time = time.time()
        
        if len(contents) < 2:
            return []
        
        try:
            # Extract metrics for all contents
            metrics_list = []
            for content in contents:
                metrics = await self.text_processor.extract_advanced_metrics(content)
                metrics_list.append(metrics)
            
            # Create feature matrix
            feature_matrix = self._create_feature_matrix(metrics_list)
            
            # Apply clustering
            if method == "dbscan":
                cluster_labels = self._dbscan_clustering(feature_matrix)
            elif method == "kmeans":
                cluster_labels = self._kmeans_clustering(feature_matrix)
            else:
                cluster_labels = self._dbscan_clustering(feature_matrix)
            
            # Create clustering results
            clustering_results = self._create_clustering_results(
                contents, metrics_list, cluster_labels, feature_matrix
            )
            
            execution_time = (time.time() - start_time) * 1000
            logger.info(f"Clustering completed in {execution_time:.2f}ms")
            
            return clustering_results
            
        except Exception as e:
            logger.error(f"Error in clustering analysis: {e}")
            return []
    
    def _create_feature_matrix(self, metrics_list: List[ContentMetrics]) -> np.ndarray:
        """Create feature matrix from metrics"""
        try:
            features = []
            
            for metrics in metrics_list:
                feature_vector = [
                    metrics.word_count,
                    metrics.sentence_count,
                    metrics.readability_score,
                    metrics.complexity_score,
                    metrics.diversity_score,
                    metrics.sentiment_score,
                    metrics.quality_score,
                    metrics.avg_word_length,
                    metrics.avg_sentence_length
                ]
                
                # Add topic scores
                topic_scores = [metrics.topic_scores.get(topic, 0) for topic in 
                              ['technology', 'business', 'health', 'education', 'entertainment', 'sports', 'politics', 'science']]
                feature_vector.extend(topic_scores)
                
                features.append(feature_vector)
            
            return np.array(features)
            
        except Exception as e:
            logger.error(f"Error creating feature matrix: {e}")
            return np.array([])
    
    def _dbscan_clustering(self, feature_matrix: np.ndarray) -> np.ndarray:
        """Apply DBSCAN clustering"""
        try:
            # Normalize features
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            normalized_features = scaler.fit_transform(feature_matrix)
            
            # Apply DBSCAN
            dbscan = DBSCAN(eps=0.5, min_samples=2)
            cluster_labels = dbscan.fit_predict(normalized_features)
            
            return cluster_labels
            
        except Exception as e:
            logger.error(f"Error in DBSCAN clustering: {e}")
            return np.array([0] * len(feature_matrix))
    
    def _kmeans_clustering(self, feature_matrix: np.ndarray) -> np.ndarray:
        """Apply K-means clustering"""
        try:
            # Normalize features
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            normalized_features = scaler.fit_transform(feature_matrix)
            
            # Apply K-means
            n_clusters = min(5, len(feature_matrix) // 2) if len(feature_matrix) > 4 else 2
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(normalized_features)
            
            return cluster_labels
            
        except Exception as e:
            logger.error(f"Error in K-means clustering: {e}")
            return np.array([0] * len(feature_matrix))
    
    def _create_clustering_results(self, contents: List[str], metrics_list: List[ContentMetrics], 
                                 cluster_labels: np.ndarray, feature_matrix: np.ndarray) -> List[ClusteringResult]:
        """Create clustering results"""
        try:
            clustering_results = []
            unique_clusters = set(cluster_labels)
            
            for cluster_id in unique_clusters:
                if cluster_id == -1:  # Skip noise points in DBSCAN
                    continue
                
                # Get content indices for this cluster
                cluster_indices = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
                
                if len(cluster_indices) < 2:
                    continue
                
                # Get cluster contents and metrics
                cluster_contents = [contents[i] for i in cluster_indices]
                cluster_metrics = [metrics_list[i] for i in cluster_indices]
                
                # Calculate cluster centroid
                cluster_features = feature_matrix[cluster_indices]
                cluster_centroid = np.mean(cluster_features, axis=0).tolist()
                
                # Calculate cluster quality (cohesion)
                cluster_quality = self._calculate_cluster_quality(cluster_features, cluster_centroid)
                
                # Find dominant topics and sentiments
                dominant_topics = self._find_dominant_topics(cluster_metrics)
                dominant_sentiments = self._find_dominant_sentiments(cluster_metrics)
                
                # Find representative content (closest to centroid)
                representative_content = self._find_representative_content(
                    cluster_contents, cluster_metrics, cluster_centroid
                )
                
                clustering_result = ClusteringResult(
                    cluster_id=int(cluster_id),
                    content_ids=[metrics_list[i].content_id for i in cluster_indices],
                    cluster_size=len(cluster_indices),
                    cluster_centroid=cluster_centroid,
                    cluster_quality=cluster_quality,
                    dominant_topics=dominant_topics,
                    dominant_sentiments=dominant_sentiments,
                    representative_content=representative_content
                )
                
                clustering_results.append(clustering_result)
            
            return clustering_results
            
        except Exception as e:
            logger.error(f"Error creating clustering results: {e}")
            return []
    
    def _calculate_cluster_quality(self, cluster_features: np.ndarray, centroid: np.ndarray) -> float:
        """Calculate cluster quality (cohesion)"""
        try:
            distances = np.linalg.norm(cluster_features - centroid, axis=1)
            avg_distance = np.mean(distances)
            max_distance = np.max(distances)
            
            # Quality is inverse of average distance (normalized)
            quality = 1 - (avg_distance / (max_distance + 1e-8))
            return max(0, min(1, quality))
            
        except Exception as e:
            logger.error(f"Error calculating cluster quality: {e}")
            return 0.0
    
    def _find_dominant_topics(self, cluster_metrics: List[ContentMetrics]) -> List[str]:
        """Find dominant topics in cluster"""
        try:
            topic_scores = defaultdict(float)
            
            for metrics in cluster_metrics:
                for topic, score in metrics.topic_scores.items():
                    topic_scores[topic] += score
            
            # Sort by total score and return top 3
            dominant_topics = sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)[:3]
            return [topic for topic, score in dominant_topics]
            
        except Exception as e:
            logger.error(f"Error finding dominant topics: {e}")
            return []
    
    def _find_dominant_sentiments(self, cluster_metrics: List[ContentMetrics]) -> List[str]:
        """Find dominant sentiments in cluster"""
        try:
            sentiment_scores = defaultdict(float)
            
            for metrics in cluster_metrics:
                for emotion, score in metrics.emotion_scores.items():
                    sentiment_scores[emotion] += score
            
            # Sort by total score and return top 3
            dominant_sentiments = sorted(sentiment_scores.items(), key=lambda x: x[1], reverse=True)[:3]
            return [sentiment for sentiment, score in dominant_sentiments]
            
        except Exception as e:
            logger.error(f"Error finding dominant sentiments: {e}")
            return []
    
    def _find_representative_content(self, cluster_contents: List[str], 
                                   cluster_metrics: List[ContentMetrics], 
                                   centroid: np.ndarray) -> str:
        """Find representative content (closest to centroid)"""
        try:
            if not cluster_contents:
                return ""
            
            # For simplicity, return the content with highest quality score
            best_quality = 0
            representative_content = cluster_contents[0]
            
            for i, metrics in enumerate(cluster_metrics):
                if metrics.quality_score > best_quality:
                    best_quality = metrics.quality_score
                    representative_content = cluster_contents[i]
            
            return representative_content
            
        except Exception as e:
            logger.error(f"Error finding representative content: {e}")
            return cluster_contents[0] if cluster_contents else ""


class TrendAnalyzer:
    """Advanced trend analysis"""
    
    def __init__(self):
        self.trend_history = []
    
    async def analyze_trends(self, content_metrics: List[ContentMetrics], 
                           time_window: str = "daily") -> List[TrendAnalysis]:
        """Analyze trends in content metrics"""
        start_time = time.time()
        
        try:
            if len(content_metrics) < 3:
                return []
            
            # Group metrics by time window
            grouped_metrics = self._group_metrics_by_time(content_metrics, time_window)
            
            # Analyze different types of trends
            trends = []
            
            # Sentiment trends
            sentiment_trends = self._analyze_sentiment_trends(grouped_metrics)
            trends.extend(sentiment_trends)
            
            # Topic trends
            topic_trends = self._analyze_topic_trends(grouped_metrics)
            trends.extend(topic_trends)
            
            # Quality trends
            quality_trends = self._analyze_quality_trends(grouped_metrics)
            trends.extend(quality_trends)
            
            # Volume trends
            volume_trends = self._analyze_volume_trends(grouped_metrics)
            trends.extend(volume_trends)
            
            execution_time = (time.time() - start_time) * 1000
            logger.info(f"Trend analysis completed in {execution_time:.2f}ms")
            
            return trends
            
        except Exception as e:
            logger.error(f"Error in trend analysis: {e}")
            return []
    
    def _group_metrics_by_time(self, content_metrics: List[ContentMetrics], 
                              time_window: str) -> Dict[str, List[ContentMetrics]]:
        """Group metrics by time window"""
        try:
            grouped = defaultdict(list)
            
            for metrics in content_metrics:
                if time_window == "daily":
                    key = metrics.timestamp.date().isoformat()
                elif time_window == "hourly":
                    key = f"{metrics.timestamp.date()}_{metrics.timestamp.hour}"
                elif time_window == "weekly":
                    # Get week number
                    week = metrics.timestamp.isocalendar()[1]
                    key = f"{metrics.timestamp.year}_W{week}"
                else:
                    key = metrics.timestamp.date().isoformat()
                
                grouped[key].append(metrics)
            
            return dict(grouped)
            
        except Exception as e:
            logger.error(f"Error grouping metrics by time: {e}")
            return {}
    
    def _analyze_sentiment_trends(self, grouped_metrics: Dict[str, List[ContentMetrics]]) -> List[TrendAnalysis]:
        """Analyze sentiment trends"""
        try:
            trends = []
            time_points = sorted(grouped_metrics.keys())
            
            if len(time_points) < 2:
                return trends
            
            # Calculate average sentiment for each time point
            sentiment_values = []
            for time_point in time_points:
                metrics_list = grouped_metrics[time_point]
                avg_sentiment = sum(m.sentiment_score for m in metrics_list) / len(metrics_list)
                sentiment_values.append(avg_sentiment)
            
            # Detect trend direction
            trend_direction = self._detect_trend_direction(sentiment_values)
            trend_strength = self._calculate_trend_strength(sentiment_values)
            
            if trend_strength > 0.3:  # Significant trend
                trend = TrendAnalysis(
                    trend_type="sentiment",
                    trend_direction=trend_direction,
                    trend_strength=trend_strength,
                    trend_period=f"{len(time_points)} time points",
                    affected_content=[m.content_id for metrics_list in grouped_metrics.values() for m in metrics_list],
                    trend_factors=["content_sentiment", "emotional_tone"],
                    prediction_confidence=min(0.9, trend_strength * 1.5),
                    future_projection=self._project_future_trend(sentiment_values, trend_direction)
                )
                trends.append(trend)
            
            return trends
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment trends: {e}")
            return []
    
    def _analyze_topic_trends(self, grouped_metrics: Dict[str, List[ContentMetrics]]) -> List[TrendAnalysis]:
        """Analyze topic trends"""
        try:
            trends = []
            time_points = sorted(grouped_metrics.keys())
            
            if len(time_points) < 2:
                return trends
            
            # Analyze topic popularity over time
            topic_popularity = defaultdict(list)
            
            for time_point in time_points:
                metrics_list = grouped_metrics[time_point]
                topic_scores = defaultdict(float)
                
                for metrics in metrics_list:
                    for topic, score in metrics.topic_scores.items():
                        topic_scores[topic] += score
                
                # Normalize by number of content pieces
                for topic in topic_scores:
                    topic_scores[topic] /= len(metrics_list)
                    topic_popularity[topic].append(topic_scores[topic])
            
            # Detect trends for each topic
            for topic, popularity_values in topic_popularity.items():
                if len(popularity_values) < 2:
                    continue
                
                trend_direction = self._detect_trend_direction(popularity_values)
                trend_strength = self._calculate_trend_strength(popularity_values)
                
                if trend_strength > 0.3:  # Significant trend
                    trend = TrendAnalysis(
                        trend_type=f"topic_{topic}",
                        trend_direction=trend_direction,
                        trend_strength=trend_strength,
                        trend_period=f"{len(time_points)} time points",
                        affected_content=[m.content_id for metrics_list in grouped_metrics.values() for m in metrics_list],
                        trend_factors=[f"topic_{topic}", "content_themes"],
                        prediction_confidence=min(0.9, trend_strength * 1.5),
                        future_projection=self._project_future_trend(popularity_values, trend_direction)
                    )
                    trends.append(trend)
            
            return trends
            
        except Exception as e:
            logger.error(f"Error analyzing topic trends: {e}")
            return []
    
    def _analyze_quality_trends(self, grouped_metrics: Dict[str, List[ContentMetrics]]) -> List[TrendAnalysis]:
        """Analyze quality trends"""
        try:
            trends = []
            time_points = sorted(grouped_metrics.keys())
            
            if len(time_points) < 2:
                return trends
            
            # Calculate average quality for each time point
            quality_values = []
            for time_point in time_points:
                metrics_list = grouped_metrics[time_point]
                avg_quality = sum(m.quality_score for m in metrics_list) / len(metrics_list)
                quality_values.append(avg_quality)
            
            # Detect trend direction
            trend_direction = self._detect_trend_direction(quality_values)
            trend_strength = self._calculate_trend_strength(quality_values)
            
            if trend_strength > 0.3:  # Significant trend
                trend = TrendAnalysis(
                    trend_type="quality",
                    trend_direction=trend_direction,
                    trend_strength=trend_strength,
                    trend_period=f"{len(time_points)} time points",
                    affected_content=[m.content_id for metrics_list in grouped_metrics.values() for m in metrics_list],
                    trend_factors=["content_quality", "readability", "complexity"],
                    prediction_confidence=min(0.9, trend_strength * 1.5),
                    future_projection=self._project_future_trend(quality_values, trend_direction)
                )
                trends.append(trend)
            
            return trends
            
        except Exception as e:
            logger.error(f"Error analyzing quality trends: {e}")
            return []
    
    def _analyze_volume_trends(self, grouped_metrics: Dict[str, List[ContentMetrics]]) -> List[TrendAnalysis]:
        """Analyze volume trends"""
        try:
            trends = []
            time_points = sorted(grouped_metrics.keys())
            
            if len(time_points) < 2:
                return trends
            
            # Calculate content volume for each time point
            volume_values = [len(grouped_metrics[time_point]) for time_point in time_points]
            
            # Detect trend direction
            trend_direction = self._detect_trend_direction(volume_values)
            trend_strength = self._calculate_trend_strength(volume_values)
            
            if trend_strength > 0.3:  # Significant trend
                trend = TrendAnalysis(
                    trend_type="volume",
                    trend_direction=trend_direction,
                    trend_strength=trend_strength,
                    trend_period=f"{len(time_points)} time points",
                    affected_content=[m.content_id for metrics_list in grouped_metrics.values() for m in metrics_list],
                    trend_factors=["content_volume", "publishing_frequency"],
                    prediction_confidence=min(0.9, trend_strength * 1.5),
                    future_projection=self._project_future_trend(volume_values, trend_direction)
                )
                trends.append(trend)
            
            return trends
            
        except Exception as e:
            logger.error(f"Error analyzing volume trends: {e}")
            return []
    
    def _detect_trend_direction(self, values: List[float]) -> str:
        """Detect trend direction"""
        try:
            if len(values) < 2:
                return "stable"
            
            # Calculate slope using linear regression
            x = list(range(len(values)))
            slope, _, _, _, _ = stats.linregress(x, values)
            
            if slope > 0.1:
                return "increasing"
            elif slope < -0.1:
                return "decreasing"
            else:
                return "stable"
                
        except Exception as e:
            logger.error(f"Error detecting trend direction: {e}")
            return "stable"
    
    def _calculate_trend_strength(self, values: List[float]) -> float:
        """Calculate trend strength"""
        try:
            if len(values) < 2:
                return 0.0
            
            # Calculate R-squared for linear regression
            x = list(range(len(values)))
            slope, intercept, r_value, _, _ = stats.linregress(x, values)
            
            return abs(r_value)  # R-squared indicates trend strength
            
        except Exception as e:
            logger.error(f"Error calculating trend strength: {e}")
            return 0.0
    
    def _project_future_trend(self, values: List[float], direction: str) -> Dict[str, Any]:
        """Project future trend"""
        try:
            if len(values) < 2:
                return {"projection": "insufficient_data"}
            
            # Simple linear projection
            x = list(range(len(values)))
            slope, intercept, _, _, _ = stats.linregress(x, values)
            
            # Project next 3 time points
            future_values = []
            for i in range(1, 4):
                future_value = slope * (len(values) + i) + intercept
                future_values.append(future_value)
            
            return {
                "projection": "linear",
                "future_values": future_values,
                "confidence": min(0.8, abs(slope) * 10),
                "direction": direction
            }
            
        except Exception as e:
            logger.error(f"Error projecting future trend: {e}")
            return {"projection": "error"}


class AnomalyDetector:
    """Advanced anomaly detection"""
    
    def __init__(self):
        self.anomaly_threshold = 2.0  # Z-score threshold
    
    async def detect_anomalies(self, content_metrics: List[ContentMetrics]) -> List[AnomalyDetection]:
        """Detect anomalies in content metrics"""
        start_time = time.time()
        
        try:
            if len(content_metrics) < 5:
                return []
            
            anomalies = []
            
            # Detect different types of anomalies
            quality_anomalies = self._detect_quality_anomalies(content_metrics)
            anomalies.extend(quality_anomalies)
            
            sentiment_anomalies = self._detect_sentiment_anomalies(content_metrics)
            anomalies.extend(sentiment_anomalies)
            
            length_anomalies = self._detect_length_anomalies(content_metrics)
            anomalies.extend(length_anomalies)
            
            topic_anomalies = self._detect_topic_anomalies(content_metrics)
            anomalies.extend(topic_anomalies)
            
            execution_time = (time.time() - start_time) * 1000
            logger.info(f"Anomaly detection completed in {execution_time:.2f}ms")
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Error in anomaly detection: {e}")
            return []
    
    def _detect_quality_anomalies(self, content_metrics: List[ContentMetrics]) -> List[AnomalyDetection]:
        """Detect quality anomalies"""
        try:
            anomalies = []
            quality_scores = [m.quality_score for m in content_metrics]
            
            if len(quality_scores) < 3:
                return anomalies
            
            # Calculate Z-scores
            mean_quality = statistics.mean(quality_scores)
            std_quality = statistics.stdev(quality_scores) if len(quality_scores) > 1 else 0
            
            if std_quality == 0:
                return anomalies
            
            for i, metrics in enumerate(content_metrics):
                z_score = abs((metrics.quality_score - mean_quality) / std_quality)
                
                if z_score > self.anomaly_threshold:
                    severity = "high" if z_score > 3.0 else "medium"
                    recommendation = "Review content quality" if metrics.quality_score < mean_quality else "Use as quality benchmark"
                    
                    anomaly = AnomalyDetection(
                        anomaly_type="quality",
                        anomaly_score=z_score,
                        content_id=metrics.content_id,
                        anomaly_factors=["quality_score", "readability", "complexity"],
                        severity=severity,
                        recommendation=recommendation,
                        confidence=min(0.95, z_score / 4.0)
                    )
                    anomalies.append(anomaly)
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Error detecting quality anomalies: {e}")
            return []
    
    def _detect_sentiment_anomalies(self, content_metrics: List[ContentMetrics]) -> List[AnomalyDetection]:
        """Detect sentiment anomalies"""
        try:
            anomalies = []
            sentiment_scores = [m.sentiment_score for m in content_metrics]
            
            if len(sentiment_scores) < 3:
                return anomalies
            
            # Calculate Z-scores
            mean_sentiment = statistics.mean(sentiment_scores)
            std_sentiment = statistics.stdev(sentiment_scores) if len(sentiment_scores) > 1 else 0
            
            if std_sentiment == 0:
                return anomalies
            
            for i, metrics in enumerate(content_metrics):
                z_score = abs((metrics.sentiment_score - mean_sentiment) / std_sentiment)
                
                if z_score > self.anomaly_threshold:
                    severity = "high" if z_score > 3.0 else "medium"
                    recommendation = "Review sentiment tone" if abs(metrics.sentiment_score) > 0.8 else "Monitor sentiment consistency"
                    
                    anomaly = AnomalyDetection(
                        anomaly_type="sentiment",
                        anomaly_score=z_score,
                        content_id=metrics.content_id,
                        anomaly_factors=["sentiment_score", "emotional_tone"],
                        severity=severity,
                        recommendation=recommendation,
                        confidence=min(0.95, z_score / 4.0)
                    )
                    anomalies.append(anomaly)
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Error detecting sentiment anomalies: {e}")
            return []
    
    def _detect_length_anomalies(self, content_metrics: List[ContentMetrics]) -> List[AnomalyDetection]:
        """Detect length anomalies"""
        try:
            anomalies = []
            word_counts = [m.word_count for m in content_metrics]
            
            if len(word_counts) < 3:
                return anomalies
            
            # Calculate Z-scores
            mean_length = statistics.mean(word_counts)
            std_length = statistics.stdev(word_counts) if len(word_counts) > 1 else 0
            
            if std_length == 0:
                return anomalies
            
            for i, metrics in enumerate(content_metrics):
                z_score = abs((metrics.word_count - mean_length) / std_length)
                
                if z_score > self.anomaly_threshold:
                    severity = "high" if z_score > 3.0 else "medium"
                    recommendation = "Consider content length" if metrics.word_count < mean_length else "Review content structure"
                    
                    anomaly = AnomalyDetection(
                        anomaly_type="length",
                        anomaly_score=z_score,
                        content_id=metrics.content_id,
                        anomaly_factors=["word_count", "content_length"],
                        severity=severity,
                        recommendation=recommendation,
                        confidence=min(0.95, z_score / 4.0)
                    )
                    anomalies.append(anomaly)
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Error detecting length anomalies: {e}")
            return []
    
    def _detect_topic_anomalies(self, content_metrics: List[ContentMetrics]) -> List[AnomalyDetection]:
        """Detect topic anomalies"""
        try:
            anomalies = []
            
            # Collect all topics
            all_topics = set()
            for metrics in content_metrics:
                all_topics.update(metrics.topic_scores.keys())
            
            # Analyze each topic
            for topic in all_topics:
                topic_scores = [m.topic_scores.get(topic, 0) for m in content_metrics]
                
                if len(topic_scores) < 3:
                    continue
                
                # Calculate Z-scores
                mean_score = statistics.mean(topic_scores)
                std_score = statistics.stdev(topic_scores) if len(topic_scores) > 1 else 0
                
                if std_score == 0:
                    continue
                
                for i, metrics in enumerate(content_metrics):
                    topic_score = metrics.topic_scores.get(topic, 0)
                    z_score = abs((topic_score - mean_score) / std_score)
                    
                    if z_score > self.anomaly_threshold and topic_score > 0.5:  # Only if topic is significant
                        severity = "high" if z_score > 3.0 else "medium"
                        recommendation = f"Review {topic} content focus"
                        
                        anomaly = AnomalyDetection(
                            anomaly_type=f"topic_{topic}",
                            anomaly_score=z_score,
                            content_id=metrics.content_id,
                            anomaly_factors=[f"topic_{topic}", "content_themes"],
                            severity=severity,
                            recommendation=recommendation,
                            confidence=min(0.95, z_score / 4.0)
                        )
                        anomalies.append(anomaly)
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Error detecting topic anomalies: {e}")
            return []


class AdvancedAnalyticsEngine:
    """Main Advanced Analytics Engine"""
    
    def __init__(self, config: AnalyticsConfig):
        self.config = config
        self.text_processor = AdvancedTextProcessor()
        self.similarity_analyzer = SimilarityAnalyzer(self.text_processor)
        self.clustering_analyzer = ClusteringAnalyzer(self.text_processor)
        self.trend_analyzer = TrendAnalyzer()
        self.anomaly_detector = AnomalyDetector()
        
        self.analytics_cache = {}
        self.analytics_history = []
    
    async def analyze_content(self, content: str) -> ContentMetrics:
        """Analyze single content piece"""
        try:
            # Check cache first
            content_hash = hashlib.md5(content.encode()).hexdigest()
            if content_hash in self.analytics_cache:
                return self.analytics_cache[content_hash]
            
            # Perform analysis
            metrics = await self.text_processor.extract_advanced_metrics(content)
            
            # Cache result
            self.analytics_cache[content_hash] = metrics
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error analyzing content: {e}")
            raise
    
    async def analyze_similarity(self, content1: str, content2: str) -> SimilarityResult:
        """Analyze similarity between two content pieces"""
        try:
            return await self.similarity_analyzer.analyze_similarity(content1, content2)
        except Exception as e:
            logger.error(f"Error analyzing similarity: {e}")
            raise
    
    async def cluster_content(self, contents: List[str], method: str = "dbscan") -> List[ClusteringResult]:
        """Cluster multiple content pieces"""
        try:
            return await self.clustering_analyzer.cluster_content(contents, method)
        except Exception as e:
            logger.error(f"Error clustering content: {e}")
            raise
    
    async def analyze_trends(self, content_metrics: List[ContentMetrics], 
                           time_window: str = "daily") -> List[TrendAnalysis]:
        """Analyze trends in content metrics"""
        try:
            return await self.trend_analyzer.analyze_trends(content_metrics, time_window)
        except Exception as e:
            logger.error(f"Error analyzing trends: {e}")
            raise
    
    async def detect_anomalies(self, content_metrics: List[ContentMetrics]) -> List[AnomalyDetection]:
        """Detect anomalies in content metrics"""
        try:
            return await self.anomaly_detector.detect_anomalies(content_metrics)
        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
            raise
    
    async def comprehensive_analysis(self, contents: List[str]) -> Dict[str, Any]:
        """Perform comprehensive analysis of multiple content pieces"""
        start_time = time.time()
        
        try:
            # Analyze all content
            content_metrics = []
            for content in contents:
                metrics = await self.analyze_content(content)
                content_metrics.append(metrics)
            
            # Perform various analyses
            analyses = {
                "content_metrics": content_metrics,
                "similarity_matrix": await self._calculate_similarity_matrix(contents),
                "clustering_results": await self.cluster_content(contents),
                "trend_analysis": await self.analyze_trends(content_metrics),
                "anomaly_detection": await self.detect_anomalies(content_metrics),
                "summary_statistics": self._calculate_summary_statistics(content_metrics)
            }
            
            execution_time = (time.time() - start_time) * 1000
            
            # Store in history
            analysis_record = {
                "timestamp": datetime.now(),
                "content_count": len(contents),
                "execution_time_ms": execution_time,
                "analyses_performed": list(analyses.keys())
            }
            self.analytics_history.append(analysis_record)
            
            logger.info(f"Comprehensive analysis completed in {execution_time:.2f}ms")
            
            return analyses
            
        except Exception as e:
            logger.error(f"Error in comprehensive analysis: {e}")
            raise
    
    async def _calculate_similarity_matrix(self, contents: List[str]) -> List[List[float]]:
        """Calculate similarity matrix for all content pairs"""
        try:
            similarity_matrix = []
            
            for i, content1 in enumerate(contents):
                row = []
                for j, content2 in enumerate(contents):
                    if i == j:
                        row.append(1.0)
                    else:
                        similarity_result = await self.analyze_similarity(content1, content2)
                        row.append(similarity_result.similarity_score)
                similarity_matrix.append(row)
            
            return similarity_matrix
            
        except Exception as e:
            logger.error(f"Error calculating similarity matrix: {e}")
            return []
    
    def _calculate_summary_statistics(self, content_metrics: List[ContentMetrics]) -> Dict[str, Any]:
        """Calculate summary statistics"""
        try:
            if not content_metrics:
                return {}
            
            # Extract values
            word_counts = [m.word_count for m in content_metrics]
            quality_scores = [m.quality_score for m in content_metrics]
            sentiment_scores = [m.sentiment_score for m in content_metrics]
            readability_scores = [m.readability_score for m in content_metrics]
            
            # Calculate statistics
            summary = {
                "total_content_pieces": len(content_metrics),
                "word_count_stats": {
                    "mean": statistics.mean(word_counts),
                    "median": statistics.median(word_counts),
                    "std": statistics.stdev(word_counts) if len(word_counts) > 1 else 0,
                    "min": min(word_counts),
                    "max": max(word_counts)
                },
                "quality_stats": {
                    "mean": statistics.mean(quality_scores),
                    "median": statistics.median(quality_scores),
                    "std": statistics.stdev(quality_scores) if len(quality_scores) > 1 else 0,
                    "min": min(quality_scores),
                    "max": max(quality_scores)
                },
                "sentiment_stats": {
                    "mean": statistics.mean(sentiment_scores),
                    "median": statistics.median(sentiment_scores),
                    "std": statistics.stdev(sentiment_scores) if len(sentiment_scores) > 1 else 0,
                    "min": min(sentiment_scores),
                    "max": max(sentiment_scores)
                },
                "readability_stats": {
                    "mean": statistics.mean(readability_scores),
                    "median": statistics.median(readability_scores),
                    "std": statistics.stdev(readability_scores) if len(readability_scores) > 1 else 0,
                    "min": min(readability_scores),
                    "max": max(readability_scores)
                }
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error calculating summary statistics: {e}")
            return {}
    
    async def get_analytics_history(self) -> List[Dict[str, Any]]:
        """Get analytics history"""
        return self.analytics_history
    
    async def clear_cache(self) -> None:
        """Clear analytics cache"""
        self.analytics_cache.clear()
        logger.info("Analytics cache cleared")


# Global instance
advanced_analytics_engine: Optional[AdvancedAnalyticsEngine] = None


async def initialize_advanced_analytics_engine(config: Optional[AnalyticsConfig] = None) -> None:
    """Initialize advanced analytics engine"""
    global advanced_analytics_engine
    
    if config is None:
        config = AnalyticsConfig()
    
    advanced_analytics_engine = AdvancedAnalyticsEngine(config)
    logger.info("Advanced Analytics Engine initialized successfully")


async def get_advanced_analytics_engine() -> Optional[AdvancedAnalyticsEngine]:
    """Get advanced analytics engine instance"""
    return advanced_analytics_engine

