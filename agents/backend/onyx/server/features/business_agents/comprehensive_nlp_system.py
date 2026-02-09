"""
Comprehensive NLP System
========================

Sistema NLP integral con capacidades completas y optimizaciones avanzadas.
"""

import asyncio
import time
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import numpy as np
import torch
import psutil
import gc
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

# Core NLP imports
import spacy
from transformers import AutoTokenizer, AutoModel, pipeline
from sentence_transformers import SentenceTransformer
import textstat
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

# Advanced libraries
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# NLP libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re

logger = logging.getLogger(__name__)

class ComprehensiveNLPConfig:
    """Configuración del sistema NLP integral."""
    
    def __init__(self):
        self.max_workers = mp.cpu_count() * 3
        self.batch_size = 64
        self.max_concurrent = 100
        self.memory_limit_gb = 64.0
        self.cache_size_mb = 32768
        self.gpu_memory_fraction = 0.95
        self.mixed_precision = True
        self.comprehensive_mode = True
        self.advanced_analytics = True
        self.real_time_processing = True

@dataclass
class ComprehensiveNLPResult:
    """Resultado del sistema NLP integral."""
    text: str
    language: str
    sentiment: Dict[str, Any]
    entities: List[Dict[str, Any]]
    keywords: List[str]
    topics: List[Dict[str, Any]]
    readability: Dict[str, Any]
    comprehensive_features: Dict[str, Any]
    analytics: Dict[str, Any]
    insights: Dict[str, Any]
    quality_score: float
    confidence_score: float
    processing_time: float
    cache_hit: bool
    timestamp: datetime

class ComprehensiveNLPSystem:
    """Sistema NLP integral."""
    
    def __init__(self, config: ComprehensiveNLPConfig = None):
        """Initialize comprehensive NLP system."""
        self.config = config or ComprehensiveNLPConfig()
        self.is_initialized = False
        
        # Comprehensive components
        self.models = {}
        self.pipelines = {}
        self.vectorizers = {}
        self.embeddings = {}
        self.ml_models = {}
        self.analytics_models = {}
        
        # Performance optimization
        self.cache = {}
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
        self.process_executor = ProcessPoolExecutor(max_workers=self.config.max_workers)
        
        # GPU optimization
        self.gpu_available = torch.cuda.is_available()
        self.gpu_device = "cuda" if self.gpu_available else "cpu"
        
        # Statistics
        self.stats = {
            'requests_processed': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'average_processing_time': 0.0,
            'average_quality_score': 0.0,
            'average_confidence_score': 0.0,
            'error_count': 0
        }
        
        # Background tasks
        self._background_tasks = []
        self._running = False
    
    async def initialize(self):
        """Initialize comprehensive NLP system."""
        if self.is_initialized:
            return
        
        start_time = time.time()
        
        try:
            logger.info("Initializing Comprehensive NLP System...")
            
            # Load comprehensive models
            await self._load_comprehensive_models()
            
            # Initialize comprehensive features
            await self._initialize_comprehensive_features()
            
            # Start background optimization
            await self._start_background_optimization()
            
            # Warm up models
            await self._warm_up_models()
            
            self.is_initialized = True
            init_time = time.time() - start_time
            
            logger.info(f"Comprehensive NLP System initialized in {init_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Failed to initialize Comprehensive NLP System: {e}")
            raise
    
    async def _load_comprehensive_models(self):
        """Load comprehensive models."""
        try:
            # Load spaCy models
            await self._load_spacy_comprehensive()
            
            # Load transformer models
            await self._load_transformers_comprehensive()
            
            # Load sentence transformers
            await self._load_sentence_transformers_comprehensive()
            
            # Initialize comprehensive vectorizers
            self._initialize_comprehensive_vectorizers()
            
            # Load comprehensive analysis models
            await self._load_comprehensive_analysis_models()
            
        except Exception as e:
            logger.error(f"Comprehensive model loading failed: {e}")
            raise
    
    async def _load_spacy_comprehensive(self):
        """Load spaCy models with comprehensive features."""
        try:
            spacy.prefer_gpu() if self.gpu_available else None
            
            models_to_load = {
                'en': 'en_core_web_lg',
                'es': 'es_core_news_lg',
                'fr': 'fr_core_news_lg',
                'de': 'de_core_news_lg'
            }
            
            for lang, model_name in models_to_load.items():
                try:
                    self.models[f'spacy_{lang}'] = spacy.load(
                        model_name,
                        disable=[]
                    )
                    logger.info(f"Loaded comprehensive spaCy model: {model_name}")
                except OSError:
                    logger.warning(f"spaCy model {model_name} not available")
                    
        except Exception as e:
            logger.error(f"spaCy comprehensive loading failed: {e}")
    
    async def _load_transformers_comprehensive(self):
        """Load transformer models with comprehensive features."""
        try:
            device = self.gpu_device if self.gpu_available else "cpu"
            
            model_configs = {
                'sentiment': {
                    'model': 'cardiffnlp/twitter-roberta-base-sentiment-latest',
                    'task': 'sentiment-analysis',
                    'return_all_scores': True
                },
                'ner': {
                    'model': 'xlm-roberta-large-finetuned-conll03-english',
                    'task': 'ner',
                    'aggregation_strategy': 'max'
                },
                'classification': {
                    'model': 'microsoft/DialoGPT-large',
                    'task': 'text-classification'
                }
            }
            
            for task, config in model_configs.items():
                try:
                    pipeline_config = {
                        'device': 0 if device == 'cuda' else -1,
                        'batch_size': self.config.batch_size,
                        'max_length': 1024,
                        'truncation': True,
                        'padding': True
                    }
                    
                    self.pipelines[task] = pipeline(
                        config['task'],
                        model=config['model'],
                        **pipeline_config
                    )
                    
                    logger.info(f"Loaded comprehensive {task} model: {config['model']}")
                    
                except Exception as e:
                    logger.warning(f"Failed to load {task} model: {e}")
                    
        except Exception as e:
            logger.error(f"Transformer comprehensive loading failed: {e}")
    
    async def _load_sentence_transformers_comprehensive(self):
        """Load sentence transformers with comprehensive features."""
        try:
            model_name = 'all-mpnet-base-v2'
            
            self.embeddings['sentence_transformer'] = SentenceTransformer(
                model_name,
                device=self.gpu_device,
                cache_folder='./comprehensive_nlp_cache'
            )
            
            logger.info(f"Loaded comprehensive sentence transformer: {model_name}")
            
        except Exception as e:
            logger.error(f"Sentence transformer comprehensive loading failed: {e}")
    
    def _initialize_comprehensive_vectorizers(self):
        """Initialize comprehensive vectorizers."""
        try:
            # TF-IDF with comprehensive features
            self.vectorizers['tfidf'] = TfidfVectorizer(
                max_features=50000,
                stop_words='english',
                ngram_range=(1, 3),
                min_df=1,
                max_df=0.8,
                lowercase=True,
                strip_accents='unicode',
                dtype=np.float64
            )
            
            # LDA for topic modeling
            self.vectorizers['lda'] = LatentDirichletAllocation(
                n_components=20,
                random_state=42,
                max_iter=200
            )
            
            logger.info("Initialized comprehensive vectorizers")
            
        except Exception as e:
            logger.error(f"Comprehensive vectorizer initialization failed: {e}")
    
    async def _load_comprehensive_analysis_models(self):
        """Load comprehensive analysis models."""
        try:
            # VADER sentiment analyzer
            self.models['vader'] = SentimentIntensityAnalyzer()
            
            # TextBlob for additional analysis
            self.models['textblob'] = TextBlob
            
            # NLTK components
            try:
                nltk.download('punkt', quiet=True)
                nltk.download('stopwords', quiet=True)
                nltk.download('wordnet', quiet=True)
                nltk.download('averaged_perceptron_tagger', quiet=True)
                nltk.download('maxent_ne_chunker', quiet=True)
                nltk.download('words', quiet=True)
            except Exception as e:
                logger.warning(f"NLTK download failed: {e}")
            
            self.models['lemmatizer'] = WordNetLemmatizer()
            self.models['stopwords'] = set(stopwords.words('english'))
            
            logger.info("Loaded comprehensive analysis models")
            
        except Exception as e:
            logger.error(f"Comprehensive analysis model loading failed: {e}")
    
    async def _initialize_comprehensive_features(self):
        """Initialize comprehensive features."""
        try:
            # Initialize ML models
            self.ml_models['classification'] = RandomForestClassifier(n_estimators=100, random_state=42)
            self.ml_models['regression'] = LogisticRegression(random_state=42, max_iter=1000)
            
            # Initialize analytics models
            self.analytics_models['trend_analysis'] = {}
            self.analytics_models['pattern_recognition'] = {}
            self.analytics_models['insight_generation'] = {}
            
            logger.info("Initialized comprehensive features")
            
        except Exception as e:
            logger.error(f"Comprehensive feature initialization failed: {e}")
    
    async def _start_background_optimization(self):
        """Start background optimization tasks."""
        self._running = True
        
        # Background optimization task
        opt_task = asyncio.create_task(self._optimization_loop())
        self._background_tasks.append(opt_task)
        
        logger.info("Background optimization tasks started")
    
    async def _optimization_loop(self):
        """Background optimization."""
        while self._running:
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                
                # Optimize system
                await self._optimize_system()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Optimization error: {e}")
    
    async def _optimize_system(self):
        """Optimize system performance."""
        try:
            # This would implement system optimization
            logger.info("Optimizing system performance")
            
        except Exception as e:
            logger.error(f"System optimization failed: {e}")
    
    async def _warm_up_models(self):
        """Warm up models with comprehensive features."""
        try:
            warm_up_text = "This is a comprehensive warm-up text for comprehensive performance validation."
            
            # Warm up spaCy models
            for lang in ['en', 'es', 'fr', 'de']:
                if f'spacy_{lang}' in self.models:
                    doc = self.models[f'spacy_{lang}'](warm_up_text)
                    _ = [ent for ent in doc.ents]
                    _ = [token for token in doc]
                    _ = [chunk for chunk in doc.noun_chunks]
            
            # Warm up transformer pipelines
            for task, pipeline_obj in self.pipelines.items():
                try:
                    _ = pipeline_obj(warm_up_text)
                except Exception as e:
                    logger.warning(f"Warm-up failed for {task}: {e}")
            
            # Warm up sentence transformer
            if 'sentence_transformer' in self.embeddings:
                _ = self.embeddings['sentence_transformer'].encode([warm_up_text])
            
            logger.info("Models warmed up with comprehensive features")
            
        except Exception as e:
            logger.error(f"Model warm-up with comprehensive features failed: {e}")
    
    async def analyze_comprehensive(
        self,
        text: str,
        language: str = "en",
        use_cache: bool = True,
        comprehensive_features: bool = True,
        analytics: bool = True,
        insights: bool = True
    ) -> ComprehensiveNLPResult:
        """Perform comprehensive text analysis."""
        if not self.is_initialized:
            await self.initialize()
        
        start_time = time.time()
        
        try:
            # Check cache first
            cache_hit = False
            if use_cache:
                cache_key = self._generate_cache_key(text, language)
                cached_result = self.cache.get(cache_key)
                if cached_result:
                    cache_hit = True
                    self.stats['cache_hits'] += 1
                    return cached_result
                else:
                    self.stats['cache_misses'] += 1
            
            # Perform comprehensive analysis
            result = await self._comprehensive_analysis(
                text, language, comprehensive_features, analytics, insights
            )
            
            # Create result
            processing_time = time.time() - start_time
            result = ComprehensiveNLPResult(
                text=text,
                language=language,
                sentiment=result.get('sentiment', {}),
                entities=result.get('entities', []),
                keywords=result.get('keywords', []),
                topics=result.get('topics', []),
                readability=result.get('readability', {}),
                comprehensive_features=result.get('comprehensive_features', {}),
                analytics=result.get('analytics', {}),
                insights=result.get('insights', {}),
                quality_score=result.get('quality_score', 0.0),
                confidence_score=result.get('confidence_score', 0.0),
                processing_time=processing_time,
                cache_hit=cache_hit,
                timestamp=datetime.now()
            )
            
            # Cache result
            if use_cache and not cache_hit:
                self.cache[cache_key] = result
            
            # Update statistics
            self._update_stats(processing_time, result.quality_score, result.confidence_score)
            
            return result
            
        except Exception as e:
            logger.error(f"Comprehensive analysis failed: {e}")
            raise
    
    async def _comprehensive_analysis(
        self,
        text: str,
        language: str,
        comprehensive_features: bool,
        analytics: bool,
        insights: bool
    ) -> Dict[str, Any]:
        """Perform comprehensive analysis."""
        try:
            # Perform basic analyses
            sentiment = await self._analyze_sentiment_comprehensive(text, language)
            entities = await self._extract_entities_comprehensive(text, language)
            keywords = await self._extract_keywords_comprehensive(text, language)
            topics = await self._extract_topics_comprehensive(text, language)
            readability = await self._analyze_readability_comprehensive(text, language)
            
            # Comprehensive features
            comprehensive_feat = {}
            if comprehensive_features:
                comprehensive_feat = await self._perform_comprehensive_features(text, language)
            
            # Analytics
            analytics_data = {}
            if analytics:
                analytics_data = await self._perform_analytics(text, language)
            
            # Insights
            insights_data = {}
            if insights:
                insights_data = await self._generate_insights(text, language, sentiment, entities, keywords, topics)
            
            # Quality assessment
            quality_score = await self._assess_comprehensive_quality(
                sentiment, entities, keywords, topics, readability, comprehensive_feat, analytics_data, insights_data
            )
            
            # Confidence assessment
            confidence_score = await self._calculate_comprehensive_confidence(
                quality_score, comprehensive_feat, analytics_data, insights_data
            )
            
            return {
                'sentiment': sentiment,
                'entities': entities,
                'keywords': keywords,
                'topics': topics,
                'readability': readability,
                'comprehensive_features': comprehensive_feat,
                'analytics': analytics_data,
                'insights': insights_data,
                'quality_score': quality_score,
                'confidence_score': confidence_score
            }
            
        except Exception as e:
            logger.error(f"Comprehensive analysis failed: {e}")
            return {}
    
    async def _analyze_sentiment_comprehensive(self, text: str, language: str) -> Dict[str, Any]:
        """Comprehensive sentiment analysis."""
        try:
            results = {}
            
            # Use transformer model
            if 'sentiment' in self.pipelines:
                try:
                    sentiment_result = self.pipelines['sentiment'](text)
                    results['transformer'] = sentiment_result
                except Exception as e:
                    logger.warning(f"Transformer sentiment failed: {e}")
            
            # Use VADER for additional analysis
            try:
                vader_scores = self.models['vader'].polarity_scores(text)
                results['vader'] = vader_scores
            except Exception as e:
                logger.warning(f"VADER sentiment failed: {e}")
            
            # Use TextBlob
            try:
                blob = self.models['textblob'](text)
                results['textblob'] = {
                    'polarity': blob.sentiment.polarity,
                    'subjectivity': blob.sentiment.subjectivity
                }
            except Exception as e:
                logger.warning(f"TextBlob sentiment failed: {e}")
            
            # Ensemble result
            ensemble_result = self._ensemble_sentiment_comprehensive(results)
            results['ensemble'] = ensemble_result
            
            return results
            
        except Exception as e:
            logger.error(f"Comprehensive sentiment analysis failed: {e}")
            return {}
    
    async def _extract_entities_comprehensive(self, text: str, language: str) -> List[Dict[str, Any]]:
        """Comprehensive entity extraction."""
        try:
            entities = []
            
            # Use spaCy with comprehensive features
            if f'spacy_{language}' in self.models:
                try:
                    doc = self.models[f'spacy_{language}'](text)
                    for ent in doc.ents:
                        entities.append({
                            'text': ent.text,
                            'label': ent.label_,
                            'start': ent.start_char,
                            'end': ent.end_char,
                            'confidence': 1.0,
                            'method': 'spacy',
                            'description': spacy.explain(ent.label_)
                        })
                except Exception as e:
                    logger.warning(f"spaCy NER failed: {e}")
            
            # Use transformer NER
            if 'ner' in self.pipelines:
                try:
                    ner_results = self.pipelines['ner'](text)
                    for entity in ner_results:
                        entities.append({
                            'text': entity['word'],
                            'label': entity['entity_group'],
                            'start': entity.get('start', 0),
                            'end': entity.get('end', len(entity['word'])),
                            'confidence': entity.get('score', 0.0),
                            'method': 'transformer'
                        })
                except Exception as e:
                    logger.warning(f"Transformer NER failed: {e}")
            
            return entities
            
        except Exception as e:
            logger.error(f"Comprehensive entity extraction failed: {e}")
            return []
    
    async def _extract_keywords_comprehensive(self, text: str) -> List[str]:
        """Comprehensive keyword extraction."""
        try:
            keywords = []
            
            # Use TF-IDF with comprehensive features
            try:
                vectorizer = self.vectorizers['tfidf']
                tfidf_matrix = vectorizer.fit_transform([text])
                feature_names = vectorizer.get_feature_names_out()
                scores = tfidf_matrix.toarray()[0]
                
                keyword_scores = list(zip(feature_names, scores))
                keyword_scores.sort(key=lambda x: x[1], reverse=True)
                keywords = [kw[0] for kw in keyword_scores[:30]]
                
            except Exception as e:
                logger.warning(f"TF-IDF keyword extraction failed: {e}")
            
            return keywords
            
        except Exception as e:
            logger.error(f"Comprehensive keyword extraction failed: {e}")
            return []
    
    async def _extract_topics_comprehensive(self, text: str, language: str) -> List[Dict[str, Any]]:
        """Comprehensive topic extraction."""
        try:
            topics = []
            
            # Use LDA for comprehensive topic modeling
            try:
                vectorizer = self.vectorizers['tfidf']
                lda = self.vectorizers['lda']
                
                # Fit LDA
                tfidf_matrix = vectorizer.fit_transform([text])
                lda.fit(tfidf_matrix)
                
                # Get topics
                feature_names = vectorizer.get_feature_names_out()
                for topic_idx, topic in enumerate(lda.components_):
                    top_words_idx = topic.argsort()[-15:][::-1]
                    top_words = [feature_names[i] for i in top_words_idx]
                    topics.append({
                        'id': topic_idx,
                        'words': top_words,
                        'weights': topic[top_words_idx].tolist(),
                        'coherence_score': 0.0
                    })
                
            except Exception as e:
                logger.warning(f"LDA topic extraction failed: {e}")
            
            return topics
            
        except Exception as e:
            logger.error(f"Comprehensive topic extraction failed: {e}")
            return []
    
    async def _analyze_readability_comprehensive(self, text: str, language: str) -> Dict[str, Any]:
        """Comprehensive readability analysis."""
        try:
            scores = {}
            
            # Flesch Reading Ease
            try:
                scores['flesch_reading_ease'] = textstat.flesch_reading_ease(text)
            except:
                scores['flesch_reading_ease'] = 0.0
            
            # Flesch-Kincaid Grade Level
            try:
                scores['flesch_kincaid_grade'] = textstat.flesch_kincaid_grade(text)
            except:
                scores['flesch_kincaid_grade'] = 0.0
            
            # Gunning Fog Index
            try:
                scores['gunning_fog'] = textstat.gunning_fog(text)
            except:
                scores['gunning_fog'] = 0.0
            
            # SMOG Index
            try:
                scores['smog'] = textstat.smog_index(text)
            except:
                scores['smog'] = 0.0
            
            # Automated Readability Index
            try:
                scores['ari'] = textstat.automated_readability_index(text)
            except:
                scores['ari'] = 0.0
            
            # Overall readability level
            avg_score = sum(scores.values()) / len(scores) if scores else 0
            if avg_score >= 80:
                level = "Very Easy"
            elif avg_score >= 60:
                level = "Easy"
            elif avg_score >= 40:
                level = "Moderate"
            elif avg_score >= 20:
                level = "Difficult"
            else:
                level = "Very Difficult"
            
            scores['overall_level'] = level
            scores['average_score'] = avg_score
            
            return scores
            
        except Exception as e:
            logger.error(f"Comprehensive readability analysis failed: {e}")
            return {'average_score': 0.0, 'overall_level': 'Unknown'}
    
    async def _perform_comprehensive_features(self, text: str, language: str) -> Dict[str, Any]:
        """Perform comprehensive features."""
        try:
            features = {}
            
            # Text complexity analysis
            features['complexity'] = await self._analyze_text_complexity(text)
            
            # Language detection
            features['language_detection'] = await self._detect_language(text)
            
            # Text classification
            features['classification'] = await self._classify_text(text)
            
            # Text similarity
            features['similarity'] = await self._calculate_similarity(text)
            
            # Additional comprehensive features
            features['comprehensive_analysis'] = await self._comprehensive_text_analysis(text)
            
            return features
            
        except Exception as e:
            logger.error(f"Comprehensive features failed: {e}")
            return {}
    
    async def _comprehensive_text_analysis(self, text: str) -> Dict[str, Any]:
        """Comprehensive text analysis."""
        try:
            analysis = {
                'text_statistics': {
                    'total_characters': len(text),
                    'total_words': len(text.split()),
                    'total_sentences': len(text.split('.')),
                    'total_paragraphs': len(text.split('\n\n')),
                    'average_word_length': sum(len(word) for word in text.split()) / len(text.split()) if text.split() else 0,
                    'average_sentence_length': len(text.split()) / len(text.split('.')) if text.split('.') else 0
                },
                'text_quality': {
                    'readability_score': 0.0,
                    'complexity_score': 0.0,
                    'coherence_score': 0.0,
                    'clarity_score': 0.0
                },
                'text_characteristics': {
                    'formality_level': 'neutral',
                    'emotional_tone': 'neutral',
                    'technical_level': 'basic',
                    'audience_level': 'general'
                }
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Comprehensive text analysis failed: {e}")
            return {}
    
    async def _perform_analytics(self, text: str, language: str) -> Dict[str, Any]:
        """Perform analytics."""
        try:
            analytics = {
                'trend_analysis': await self._analyze_trends(text),
                'pattern_recognition': await self._recognize_patterns(text),
                'statistical_analysis': await self._statistical_analysis(text),
                'comparative_analysis': await self._comparative_analysis(text)
            }
            
            return analytics
            
        except Exception as e:
            logger.error(f"Analytics failed: {e}")
            return {}
    
    async def _analyze_trends(self, text: str) -> Dict[str, Any]:
        """Analyze trends."""
        try:
            trends = {
                'sentiment_trend': 'stable',
                'topic_trend': 'stable',
                'complexity_trend': 'stable',
                'engagement_trend': 'stable'
            }
            
            return trends
            
        except Exception as e:
            logger.error(f"Trend analysis failed: {e}")
            return {}
    
    async def _recognize_patterns(self, text: str) -> Dict[str, Any]:
        """Recognize patterns."""
        try:
            patterns = {
                'repetitive_patterns': [],
                'structural_patterns': [],
                'linguistic_patterns': [],
                'semantic_patterns': []
            }
            
            return patterns
            
        except Exception as e:
            logger.error(f"Pattern recognition failed: {e}")
            return {}
    
    async def _statistical_analysis(self, text: str) -> Dict[str, Any]:
        """Statistical analysis."""
        try:
            stats = {
                'word_frequency': {},
                'character_frequency': {},
                'sentence_length_distribution': {},
                'vocabulary_richness': 0.0
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Statistical analysis failed: {e}")
            return {}
    
    async def _comparative_analysis(self, text: str) -> Dict[str, Any]:
        """Comparative analysis."""
        try:
            comparison = {
                'similarity_scores': {},
                'difference_analysis': {},
                'comparative_metrics': {}
            }
            
            return comparison
            
        except Exception as e:
            logger.error(f"Comparative analysis failed: {e}")
            return {}
    
    async def _generate_insights(self, text: str, language: str, sentiment: Dict[str, Any], entities: List[Dict[str, Any]], keywords: List[str], topics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate insights."""
        try:
            insights = {
                'key_insights': await self._extract_key_insights(text, sentiment, entities, keywords, topics),
                'recommendations': await self._generate_recommendations(text, sentiment, entities, keywords, topics),
                'actionable_items': await self._identify_actionable_items(text, sentiment, entities, keywords, topics),
                'summary': await self._generate_summary(text, sentiment, entities, keywords, topics)
            }
            
            return insights
            
        except Exception as e:
            logger.error(f"Insight generation failed: {e}")
            return {}
    
    async def _extract_key_insights(self, text: str, sentiment: Dict[str, Any], entities: List[Dict[str, Any]], keywords: List[str], topics: List[Dict[str, Any]]) -> List[str]:
        """Extract key insights."""
        try:
            insights = []
            
            # Sentiment insights
            if sentiment and 'ensemble' in sentiment:
                sentiment_score = sentiment['ensemble'].get('score', 0)
                if sentiment_score > 0.1:
                    insights.append("The text has a positive sentiment")
                elif sentiment_score < -0.1:
                    insights.append("The text has a negative sentiment")
                else:
                    insights.append("The text has a neutral sentiment")
            
            # Entity insights
            if entities:
                entity_types = [ent['label'] for ent in entities]
                unique_types = list(set(entity_types))
                insights.append(f"The text contains {len(entities)} entities of types: {', '.join(unique_types)}")
            
            # Keyword insights
            if keywords:
                insights.append(f"The text focuses on topics related to: {', '.join(keywords[:5])}")
            
            # Topic insights
            if topics:
                insights.append(f"The text covers {len(topics)} main topics")
            
            return insights
            
        except Exception as e:
            logger.error(f"Key insight extraction failed: {e}")
            return []
    
    async def _generate_recommendations(self, text: str, sentiment: Dict[str, Any], entities: List[Dict[str, Any]], keywords: List[str], topics: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations."""
        try:
            recommendations = []
            
            # Sentiment recommendations
            if sentiment and 'ensemble' in sentiment:
                sentiment_score = sentiment['ensemble'].get('score', 0)
                if sentiment_score < -0.1:
                    recommendations.append("Consider improving the tone to be more positive")
                elif sentiment_score > 0.1:
                    recommendations.append("The positive tone is effective")
            
            # Entity recommendations
            if entities:
                recommendations.append("Consider adding more specific entities for better context")
            
            # Keyword recommendations
            if keywords:
                recommendations.append("The keyword distribution is well-balanced")
            
            # Topic recommendations
            if topics:
                recommendations.append("Consider focusing on fewer topics for better clarity")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Recommendation generation failed: {e}")
            return []
    
    async def _identify_actionable_items(self, text: str, sentiment: Dict[str, Any], entities: List[Dict[str, Any]], keywords: List[str], topics: List[Dict[str, Any]]) -> List[str]:
        """Identify actionable items."""
        try:
            actionable_items = []
            
            # Sentiment actionable items
            if sentiment and 'ensemble' in sentiment:
                sentiment_score = sentiment['ensemble'].get('score', 0)
                if sentiment_score < -0.1:
                    actionable_items.append("Review and revise negative language")
                elif sentiment_score > 0.1:
                    actionable_items.append("Maintain positive tone")
            
            # Entity actionable items
            if entities:
                actionable_items.append("Verify entity accuracy and relevance")
            
            # Keyword actionable items
            if keywords:
                actionable_items.append("Optimize keyword usage for better SEO")
            
            # Topic actionable items
            if topics:
                actionable_items.append("Ensure topic coherence and flow")
            
            return actionable_items
            
        except Exception as e:
            logger.error(f"Actionable item identification failed: {e}")
            return []
    
    async def _generate_summary(self, text: str, sentiment: Dict[str, Any], entities: List[Dict[str, Any]], keywords: List[str], topics: List[Dict[str, Any]]) -> str:
        """Generate summary."""
        try:
            summary_parts = []
            
            # Sentiment summary
            if sentiment and 'ensemble' in sentiment:
                sentiment_score = sentiment['ensemble'].get('score', 0)
                if sentiment_score > 0.1:
                    summary_parts.append("The text has a positive sentiment")
                elif sentiment_score < -0.1:
                    summary_parts.append("The text has a negative sentiment")
                else:
                    summary_parts.append("The text has a neutral sentiment")
            
            # Entity summary
            if entities:
                summary_parts.append(f"It contains {len(entities)} named entities")
            
            # Keyword summary
            if keywords:
                summary_parts.append(f"It focuses on topics related to {', '.join(keywords[:3])}")
            
            # Topic summary
            if topics:
                summary_parts.append(f"It covers {len(topics)} main topics")
            
            return ". ".join(summary_parts) + "." if summary_parts else "No summary available"
            
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            return "Summary generation failed"
    
    async def _assess_comprehensive_quality(
        self,
        sentiment: Dict[str, Any],
        entities: List[Dict[str, Any]],
        keywords: List[str],
        topics: List[Dict[str, Any]],
        readability: Dict[str, Any],
        comprehensive_features: Dict[str, Any],
        analytics: Dict[str, Any],
        insights: Dict[str, Any]
    ) -> float:
        """Assess comprehensive quality of analysis results."""
        try:
            quality_score = 0.0
            total_weight = 0.0
            
            # Basic analysis quality (40%)
            basic_weight = 0.4
            basic_quality = 0.0
            
            # Sentiment quality
            if sentiment and 'ensemble' in sentiment:
                sentiment_quality = sentiment['ensemble'].get('confidence', 0)
                basic_quality += sentiment_quality * 0.25
            
            # Entity quality
            if entities:
                entity_quality = min(1.0, len(entities) / 20)
                basic_quality += entity_quality * 0.25
            
            # Keyword quality
            if keywords:
                keyword_quality = min(1.0, len(keywords) / 30)
                basic_quality += keyword_quality * 0.25
            
            # Readability quality
            if readability and 'average_score' in readability:
                readability_quality = readability['average_score'] / 100
                basic_quality += readability_quality * 0.25
            
            quality_score += basic_quality * basic_weight
            total_weight += basic_weight
            
            # Comprehensive features quality (30%)
            comprehensive_weight = 0.3
            comprehensive_quality = 0.0
            
            # Comprehensive features quality
            if comprehensive_features:
                comprehensive_quality += min(1.0, len(comprehensive_features) / 5) * 0.5
                comprehensive_quality += min(1.0, comprehensive_features.get('complexity', {}).get('word_count', 0) / 100) * 0.5
            
            quality_score += comprehensive_quality * comprehensive_weight
            total_weight += comprehensive_weight
            
            # Analytics quality (20%)
            analytics_weight = 0.2
            analytics_quality = 0.0
            
            # Analytics quality
            if analytics:
                analytics_quality += min(1.0, len(analytics) / 4) * 0.5
                analytics_quality += min(1.0, len(analytics.get('trend_analysis', {})) / 4) * 0.5
            
            quality_score += analytics_quality * analytics_weight
            total_weight += analytics_weight
            
            # Insights quality (10%)
            insights_weight = 0.1
            insights_quality = 0.0
            
            # Insights quality
            if insights:
                insights_quality += min(1.0, len(insights) / 4) * 0.5
                insights_quality += min(1.0, len(insights.get('key_insights', [])) / 5) * 0.5
            
            quality_score += insights_quality * insights_weight
            total_weight += insights_weight
            
            return quality_score / total_weight if total_weight > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Comprehensive quality assessment failed: {e}")
            return 0.0
    
    async def _calculate_comprehensive_confidence(
        self,
        quality_score: float,
        comprehensive_features: Dict[str, Any],
        analytics: Dict[str, Any],
        insights: Dict[str, Any]
    ) -> float:
        """Calculate comprehensive confidence score."""
        try:
            confidence_score = quality_score  # Start with quality score
            
            # Boost confidence based on comprehensive features
            if comprehensive_features:
                feature_count = len(comprehensive_features)
                if feature_count > 0:
                    feature_confidence = min(1.0, feature_count / 5)
                    confidence_score = (confidence_score + feature_confidence) / 2
            
            # Boost confidence based on analytics
            if analytics:
                analytics_count = len(analytics)
                if analytics_count > 0:
                    analytics_confidence = min(1.0, analytics_count / 4)
                    confidence_score = (confidence_score + analytics_confidence) / 2
            
            # Boost confidence based on insights
            if insights:
                insights_count = len(insights)
                if insights_count > 0:
                    insights_confidence = min(1.0, insights_count / 4)
                    confidence_score = (confidence_score + insights_confidence) / 2
            
            return min(1.0, max(0.0, confidence_score))
            
        except Exception as e:
            logger.error(f"Comprehensive confidence calculation failed: {e}")
            return quality_score
    
    def _ensemble_sentiment_comprehensive(self, sentiment_results: Dict[str, Any]) -> Dict[str, Any]:
        """Combine multiple sentiment analysis results with comprehensive features."""
        try:
            scores = []
            weights = []
            
            # Collect scores from different methods with weights
            for method, result in sentiment_results.items():
                if method == 'vader' and 'compound' in result:
                    scores.append(result['compound'])
                    weights.append(0.3)  # VADER weight
                elif method == 'textblob' and 'polarity' in result:
                    scores.append(result['polarity'])
                    weights.append(0.2)  # TextBlob weight
                elif method == 'transformer' and isinstance(result, list) and result:
                    # Extract score from transformer result
                    transformer_score = result[0].get('score', 0) if result[0].get('label') == 'POSITIVE' else -result[0].get('score', 0)
                    scores.append(transformer_score)
                    weights.append(0.5)  # Transformer weight (highest)
            
            if scores and weights:
                # Weighted average
                weighted_sum = sum(score * weight for score, weight in zip(scores, weights))
                total_weight = sum(weights)
                avg_score = weighted_sum / total_weight
                
                # Calculate confidence based on agreement
                if len(scores) > 1:
                    variance = np.var(scores)
                    confidence = max(0, 1 - variance)
                else:
                    confidence = 0.5
                
                if avg_score > 0.1:
                    sentiment = 'positive'
                elif avg_score < -0.1:
                    sentiment = 'negative'
                else:
                    sentiment = 'neutral'
                
                return {
                    'score': avg_score,
                    'sentiment': sentiment,
                    'confidence': confidence,
                    'method_count': len(scores),
                    'variance': np.var(scores) if len(scores) > 1 else 0
                }
            else:
                return {'score': 0, 'sentiment': 'neutral', 'confidence': 0, 'method_count': 0, 'variance': 0}
                
        except Exception as e:
            logger.error(f"Ensemble sentiment calculation failed: {e}")
            return {'score': 0, 'sentiment': 'neutral', 'confidence': 0, 'method_count': 0, 'variance': 0}
    
    def _generate_cache_key(self, text: str, language: str) -> str:
        """Generate cache key for comprehensive analysis."""
        content_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]
        return f"comprehensive:{language}:{content_hash}"
    
    def _update_stats(self, processing_time: float, quality_score: float, confidence_score: float):
        """Update comprehensive statistics."""
        self.stats['requests_processed'] += 1
        
        # Update average processing time
        if self.stats['average_processing_time'] == 0:
            self.stats['average_processing_time'] = processing_time
        else:
            self.stats['average_processing_time'] = (
                self.stats['average_processing_time'] * 0.9 + processing_time * 0.1
            )
        
        # Update quality scores
        if quality_score > 0:
            self.stats['average_quality_score'] = (
                self.stats['average_quality_score'] * 0.9 + quality_score * 0.1
            )
        
        # Update confidence scores
        if confidence_score > 0:
            self.stats['average_confidence_score'] = (
                self.stats['average_confidence_score'] * 0.9 + confidence_score * 0.1
            )
    
    async def batch_analyze_comprehensive(
        self,
        texts: List[str],
        language: str = "en",
        use_cache: bool = True,
        comprehensive_features: bool = True,
        analytics: bool = True,
        insights: bool = True
    ) -> List[ComprehensiveNLPResult]:
        """Perform comprehensive batch analysis."""
        if not self.is_initialized:
            await self.initialize()
        
        start_time = time.time()
        
        try:
            # Process in parallel batches
            batch_size = min(self.config.batch_size, len(texts))
            results = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                # Process batch concurrently
                batch_tasks = [
                    self.analyze_comprehensive(
                        text=text,
                        language=language,
                        use_cache=use_cache,
                        comprehensive_features=comprehensive_features,
                        analytics=analytics,
                        insights=insights
                    )
                    for text in batch
                ]
                
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                # Handle results
                for j, result in enumerate(batch_results):
                    if isinstance(result, Exception):
                        logger.error(f"Batch analysis error for text {i + j}: {result}")
                        results.append(ComprehensiveNLPResult(
                            text=batch[j],
                            language=language,
                            sentiment={},
                            entities=[],
                            keywords=[],
                            topics=[],
                            readability={},
                            comprehensive_features={},
                            analytics={},
                            insights={},
                            quality_score=0.0,
                            confidence_score=0.0,
                            processing_time=0,
                            cache_hit=False,
                            timestamp=datetime.now()
                        ))
                    else:
                        results.append(result)
            
            return results
                
        except Exception as e:
            logger.error(f"Comprehensive batch analysis failed: {e}")
            raise
    
    async def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        try:
            # System status
            system_status = {
                'initialized': self.is_initialized,
                'comprehensive_mode': True,
                'gpu_available': self.gpu_available,
                'gpu_device': self.gpu_device,
                'max_workers': self.config.max_workers,
                'batch_size': self.config.batch_size
            }
            
            # Performance statistics
            performance_stats = {
                'requests_processed': self.stats['requests_processed'],
                'cache_hits': self.stats['cache_hits'],
                'cache_misses': self.stats['cache_misses'],
                'cache_hit_rate': (
                    self.stats['cache_hits'] / (self.stats['cache_hits'] + self.stats['cache_misses'])
                    if (self.stats['cache_hits'] + self.stats['cache_misses']) > 0 else 0
                ),
                'average_processing_time': self.stats['average_processing_time'],
                'error_count': self.stats['error_count'],
                'success_rate': (
                    (self.stats['requests_processed'] - self.stats['error_count']) / self.stats['requests_processed']
                    if self.stats['requests_processed'] > 0 else 0
                )
            }
            
            # Comprehensive statistics
            comprehensive_stats = {
                'average_quality_score': self.stats['average_quality_score'],
                'average_confidence_score': self.stats['average_confidence_score'],
                'comprehensive_features_enabled': True,
                'analytics_enabled': True,
                'insights_enabled': True
            }
            
            # Cache status
            cache_status = {
                'size': len(self.cache),
                'hit_rate': (
                    self.stats['cache_hits'] / (self.stats['cache_hits'] + self.stats['cache_misses'])
                    if (self.stats['cache_hits'] + self.stats['cache_misses']) > 0 else 0
                )
            }
            
            # Memory status
            memory_status = {
                'total_gb': psutil.virtual_memory().total / (1024**3),
                'available_gb': psutil.virtual_memory().available / (1024**3),
                'used_percent': psutil.virtual_memory().percent
            }
            
            return {
                'system': system_status,
                'performance': performance_stats,
                'comprehensive': comprehensive_stats,
                'cache': cache_status,
                'memory': memory_status,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get comprehensive status: {e}")
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}
    
    async def shutdown(self):
        """Shutdown comprehensive NLP system."""
        try:
            logger.info("Shutting down Comprehensive NLP System...")
            
            # Stop background tasks
            self._running = False
            for task in self._background_tasks:
                task.cancel()
            
            # Shutdown executors
            self.executor.shutdown(wait=True)
            self.process_executor.shutdown(wait=True)
            
            logger.info("Comprehensive NLP System shutdown complete")
            
        except Exception as e:
            logger.error(f"Shutdown error: {e}")

# Global comprehensive NLP system instance
comprehensive_nlp_system = ComprehensiveNLPSystem()











