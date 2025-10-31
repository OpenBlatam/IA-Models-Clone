"""
Advanced NLP Engine for AI History Comparison System
Motor NLP avanzado para el sistema de análisis de historial de IA
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import re
from collections import Counter, defaultdict
import pickle
import joblib

# NLP and ML imports
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize, wordpunct_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer, SnowballStemmer
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.collocations import BigramCollocationFinder, TrigramCollocationFinder
from nltk.metrics import BigramAssocMeasures, TrigramAssocMeasures
from textblob import TextBlob
import spacy
from spacy import displacy
from spacy.matcher import Matcher
from spacy.tokens import Span

# Text processing
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# Advanced text analysis
import gensim
from gensim import corpora, models
from gensim.models import Word2Vec, Doc2Vec, LdaModel, CoherenceModel
from gensim.models.phrases import Phrases, Phraser
from gensim.similarities import MatrixSimilarity

# Language detection and translation
from langdetect import detect, DetectorFactory
from googletrans import Translator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LanguageType(Enum):
    """Tipos de idiomas soportados"""
    SPANISH = "es"
    ENGLISH = "en"
    FRENCH = "fr"
    GERMAN = "de"
    ITALIAN = "it"
    PORTUGUESE = "pt"
    AUTO = "auto"

class AnalysisType(Enum):
    """Tipos de análisis NLP"""
    TOKENIZATION = "tokenization"
    POS_TAGGING = "pos_tagging"
    NER = "ner"
    SENTIMENT = "sentiment"
    TOPIC_MODELING = "topic_modeling"
    TEXT_SIMILARITY = "text_similarity"
    KEYWORD_EXTRACTION = "keyword_extraction"
    TEXT_CLASSIFICATION = "text_classification"
    SUMMARIZATION = "summarization"
    TRANSLATION = "translation"
    LANGUAGE_DETECTION = "language_detection"

class SentimentType(Enum):
    """Tipos de sentimiento"""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"

@dataclass
class TokenInfo:
    """Información de token"""
    text: str
    pos: str
    lemma: str
    is_stop: bool
    is_punct: bool
    is_alpha: bool
    is_digit: bool
    start: int
    end: int

@dataclass
class EntityInfo:
    """Información de entidad nombrada"""
    text: str
    label: str
    start: int
    end: int
    confidence: float

@dataclass
class SentimentInfo:
    """Información de sentimiento"""
    polarity: float
    subjectivity: float
    sentiment_type: SentimentType
    confidence: float
    emotional_tone: str
    intensity: float

@dataclass
class TopicInfo:
    """Información de tópico"""
    id: int
    name: str
    keywords: List[str]
    weight: float
    coherence: float
    documents: List[str]

@dataclass
class KeywordInfo:
    """Información de palabra clave"""
    text: str
    score: float
    frequency: int
    tfidf_score: float
    position: List[int]

@dataclass
class TextMetrics:
    """Métricas de texto"""
    word_count: int
    sentence_count: int
    character_count: int
    avg_word_length: float
    avg_sentence_length: float
    lexical_diversity: float
    readability_score: float
    complexity_score: float
    formality_score: float

@dataclass
class NLPAnalysis:
    """Análisis NLP completo"""
    document_id: str
    language: str
    tokens: List[TokenInfo]
    entities: List[EntityInfo]
    sentiment: SentimentInfo
    topics: List[TopicInfo]
    keywords: List[KeywordInfo]
    metrics: TextMetrics
    bigrams: List[Tuple[str, str]]
    trigrams: List[Tuple[str, str, str]]
    collocations: List[Tuple[str, str]]
    analyzed_at: datetime = field(default_factory=datetime.now)

class AdvancedNLPEngine:
    """
    Motor NLP avanzado para análisis de texto
    """
    
    def __init__(
        self,
        language: LanguageType = LanguageType.SPANISH,
        enable_spacy: bool = True,
        enable_gensim: bool = True,
        cache_directory: str = "cache/nlp/",
        models_directory: str = "models/nlp/"
    ):
        self.language = language
        self.enable_spacy = enable_spacy
        self.enable_gensim = enable_gensim
        self.cache_directory = cache_directory
        self.models_directory = models_directory
        
        # Crear directorios
        import os
        os.makedirs(cache_directory, exist_ok=True)
        os.makedirs(models_directory, exist_ok=True)
        
        # Inicializar componentes
        self._initialize_nltk()
        self._initialize_spacy()
        self._initialize_gensim()
        self._initialize_models()
        
        # Almacenamiento de análisis
        self.analyses: Dict[str, NLPAnalysis] = {}
        self.topic_models: Dict[str, Any] = {}
        self.word_vectors: Optional[gensim.models.Word2Vec] = None
        
        # Configuración
        self.config = {
            "max_tokens": 10000,
            "min_word_length": 2,
            "max_word_length": 50,
            "stop_words_threshold": 0.1,
            "sentiment_threshold": 0.1,
            "topic_modeling": {
                "num_topics": 10,
                "passes": 10,
                "alpha": "auto",
                "eta": "auto"
            }
        }
    
    def _initialize_nltk(self):
        """Inicializar componentes de NLTK"""
        try:
            # Descargar recursos necesarios
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
            nltk.download('maxent_ne_chunker', quiet=True)
            nltk.download('words', quiet=True)
            nltk.download('vader_lexicon', quiet=True)
            
            # Inicializar componentes
            self.lemmatizer = WordNetLemmatizer()
            self.stemmer = PorterStemmer()
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
            
            # Obtener stopwords
            if self.language == LanguageType.SPANISH:
                self.stop_words = set(stopwords.words('spanish'))
            elif self.language == LanguageType.ENGLISH:
                self.stop_words = set(stopwords.words('english'))
            else:
                self.stop_words = set(stopwords.words('english'))  # Fallback
            
            logger.info("NLTK components initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing NLTK: {e}")
            self.stop_words = set()
    
    def _initialize_spacy(self):
        """Inicializar componentes de spaCy"""
        if not self.enable_spacy:
            self.nlp = None
            return
        
        try:
            # Cargar modelo según idioma
            if self.language == LanguageType.SPANISH:
                try:
                    self.nlp = spacy.load("es_core_news_sm")
                except OSError:
                    logger.warning("Spanish spaCy model not found, using basic NLP")
                    self.nlp = None
            elif self.language == LanguageType.ENGLISH:
                try:
                    self.nlp = spacy.load("en_core_web_sm")
                except OSError:
                    logger.warning("English spaCy model not found, using basic NLP")
                    self.nlp = None
            else:
                self.nlp = None
            
            if self.nlp:
                logger.info(f"spaCy model loaded for {self.language.value}")
            
        except Exception as e:
            logger.error(f"Error initializing spaCy: {e}")
            self.nlp = None
    
    def _initialize_gensim(self):
        """Inicializar componentes de Gensim"""
        if not self.enable_gensim:
            return
        
        try:
            # Inicializar modelos de Gensim
            self.dictionary = None
            self.corpus = None
            self.lda_model = None
            
            logger.info("Gensim components initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing Gensim: {e}")
    
    def _initialize_models(self):
        """Inicializar modelos de ML"""
        try:
            # Inicializar vectorizadores
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words=list(self.stop_words),
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95
            )
            
            self.count_vectorizer = CountVectorizer(
                max_features=1000,
                stop_words=list(self.stop_words),
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95
            )
            
            # Inicializar scaler
            self.scaler = StandardScaler()
            
            logger.info("ML models initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing ML models: {e}")
    
    async def analyze_text(
        self,
        text: str,
        document_id: str,
        analysis_types: List[AnalysisType] = None
    ) -> NLPAnalysis:
        """
        Analizar texto completo
        
        Args:
            text: Texto a analizar
            document_id: ID del documento
            analysis_types: Tipos de análisis a realizar
            
        Returns:
            Análisis NLP completo
        """
        if analysis_types is None:
            analysis_types = [
                AnalysisType.TOKENIZATION,
                AnalysisType.POS_TAGGING,
                AnalysisType.NER,
                AnalysisType.SENTIMENT,
                AnalysisType.KEYWORD_EXTRACTION
            ]
        
        try:
            logger.info(f"Analyzing text for document {document_id}")
            
            # Detectar idioma si es necesario
            if self.language == LanguageType.AUTO:
                detected_language = self._detect_language(text)
            else:
                detected_language = self.language.value
            
            # Tokenización
            tokens = []
            if AnalysisType.TOKENIZATION in analysis_types:
                tokens = await self._tokenize_text(text)
            
            # POS Tagging
            if AnalysisType.POS_TAGGING in analysis_types:
                tokens = await self._pos_tag_tokens(tokens, text)
            
            # Named Entity Recognition
            entities = []
            if AnalysisType.NER in analysis_types:
                entities = await self._extract_entities(text)
            
            # Análisis de sentimiento
            sentiment = None
            if AnalysisType.SENTIMENT in analysis_types:
                sentiment = await self._analyze_sentiment(text)
            
            # Extracción de palabras clave
            keywords = []
            if AnalysisType.KEYWORD_EXTRACTION in analysis_types:
                keywords = await self._extract_keywords(text, tokens)
            
            # Análisis de tópicos
            topics = []
            if AnalysisType.TOPIC_MODELING in analysis_types:
                topics = await self._analyze_topics(text)
            
            # Métricas de texto
            metrics = await self._calculate_text_metrics(text, tokens)
            
            # N-gramas y colocaciones
            bigrams, trigrams, collocations = await self._extract_ngrams_and_collocations(text, tokens)
            
            # Crear análisis completo
            analysis = NLPAnalysis(
                document_id=document_id,
                language=detected_language,
                tokens=tokens,
                entities=entities,
                sentiment=sentiment,
                topics=topics,
                keywords=keywords,
                metrics=metrics,
                bigrams=bigrams,
                trigrams=trigrams,
                collocations=collocations
            )
            
            # Almacenar análisis
            self.analyses[document_id] = analysis
            
            logger.info(f"Analysis completed for document {document_id}")
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing text: {e}")
            raise
    
    async def _tokenize_text(self, text: str) -> List[TokenInfo]:
        """Tokenizar texto"""
        tokens = []
        
        if self.nlp:
            # Usar spaCy para tokenización avanzada
            doc = self.nlp(text)
            for token in doc:
                token_info = TokenInfo(
                    text=token.text,
                    pos=token.pos_,
                    lemma=token.lemma_,
                    is_stop=token.is_stop,
                    is_punct=token.is_punct,
                    is_alpha=token.is_alpha,
                    is_digit=token.is_digit,
                    start=token.idx,
                    end=token.idx + len(token.text)
                )
                tokens.append(token_info)
        else:
            # Usar NLTK para tokenización básica
            word_tokens = word_tokenize(text)
            pos_tags = pos_tag(word_tokens)
            
            current_pos = 0
            for i, (word, pos) in enumerate(pos_tags):
                start_pos = text.find(word, current_pos)
                end_pos = start_pos + len(word)
                
                token_info = TokenInfo(
                    text=word,
                    pos=pos,
                    lemma=self.lemmatizer.lemmatize(word),
                    is_stop=word.lower() in self.stop_words,
                    is_punct=not word.isalnum(),
                    is_alpha=word.isalpha(),
                    is_digit=word.isdigit(),
                    start=start_pos,
                    end=end_pos
                )
                tokens.append(token_info)
                current_pos = end_pos
        
        return tokens
    
    async def _pos_tag_tokens(self, tokens: List[TokenInfo], text: str) -> List[TokenInfo]:
        """Etiquetar tokens con POS"""
        if self.nlp:
            # spaCy ya incluye POS tagging
            return tokens
        
        # Usar NLTK para POS tagging
        word_list = [token.text for token in tokens]
        pos_tags = pos_tag(word_list)
        
        for i, (token, (word, pos)) in enumerate(zip(tokens, pos_tags)):
            token.pos = pos
        
        return tokens
    
    async def _extract_entities(self, text: str) -> List[EntityInfo]:
        """Extraer entidades nombradas"""
        entities = []
        
        if self.nlp:
            # Usar spaCy para NER
            doc = self.nlp(text)
            for ent in doc.ents:
                entity_info = EntityInfo(
                    text=ent.text,
                    label=ent.label_,
                    start=ent.start_char,
                    end=ent.end_char,
                    confidence=0.8  # spaCy no proporciona confianza por defecto
                )
                entities.append(entity_info)
        else:
            # Usar NLTK para NER básico
            tokens = word_tokenize(text)
            pos_tags = pos_tag(tokens)
            chunks = ne_chunk(pos_tags)
            
            current_pos = 0
            for chunk in chunks:
                if hasattr(chunk, 'label'):
                    entity_text = ' '.join([token for token, pos in chunk.leaves()])
                    start_pos = text.find(entity_text, current_pos)
                    end_pos = start_pos + len(entity_text)
                    
                    entity_info = EntityInfo(
                        text=entity_text,
                        label=chunk.label(),
                        start=start_pos,
                        end=end_pos,
                        confidence=0.6  # NLTK NER es menos preciso
                    )
                    entities.append(entity_info)
                    current_pos = end_pos
        
        return entities
    
    async def _analyze_sentiment(self, text: str) -> SentimentInfo:
        """Analizar sentimiento del texto"""
        # Análisis con VADER
        vader_scores = self.sentiment_analyzer.polarity_scores(text)
        
        # Análisis con TextBlob
        blob = TextBlob(text)
        textblob_polarity = blob.sentiment.polarity
        textblob_subjectivity = blob.sentiment.subjectivity
        
        # Determinar tipo de sentimiento
        compound_score = vader_scores['compound']
        if compound_score >= 0.05:
            sentiment_type = SentimentType.POSITIVE
        elif compound_score <= -0.05:
            sentiment_type = SentimentType.NEGATIVE
        else:
            sentiment_type = SentimentType.NEUTRAL
        
        # Determinar tono emocional
        emotional_tone = self._determine_emotional_tone(vader_scores)
        
        # Calcular intensidad
        intensity = abs(compound_score)
        
        return SentimentInfo(
            polarity=compound_score,
            subjectivity=textblob_subjectivity,
            sentiment_type=sentiment_type,
            confidence=abs(compound_score),
            emotional_tone=emotional_tone,
            intensity=intensity
        )
    
    def _determine_emotional_tone(self, vader_scores: Dict[str, float]) -> str:
        """Determinar tono emocional basado en scores de VADER"""
        if vader_scores['pos'] > 0.5:
            return "enthusiastic"
        elif vader_scores['neg'] > 0.5:
            return "critical"
        elif vader_scores['neu'] > 0.8:
            return "neutral"
        elif vader_scores['compound'] > 0.3:
            return "optimistic"
        elif vader_scores['compound'] < -0.3:
            return "pessimistic"
        else:
            return "balanced"
    
    async def _extract_keywords(self, text: str, tokens: List[TokenInfo]) -> List[KeywordInfo]:
        """Extraer palabras clave del texto"""
        keywords = []
        
        # Filtrar tokens relevantes
        relevant_tokens = [
            token for token in tokens
            if not token.is_stop and not token.is_punct and token.is_alpha
            and len(token.text) >= self.config["min_word_length"]
        ]
        
        # Calcular frecuencias
        word_freq = Counter([token.lemma.lower() for token in relevant_tokens])
        
        # Calcular TF-IDF si hay múltiples documentos
        if len(self.analyses) > 1:
            # Preparar corpus
            corpus = [analysis.tokens for analysis in self.analyses.values()]
            corpus_texts = []
            for doc_tokens in corpus:
                doc_text = ' '.join([token.lemma for token in doc_tokens if not token.is_stop])
                corpus_texts.append(doc_text)
            
            # Calcular TF-IDF
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(corpus_texts)
            feature_names = self.tfidf_vectorizer.get_feature_names_out()
            
            # Obtener scores TF-IDF para el documento actual
            current_doc_idx = len(corpus_texts) - 1
            tfidf_scores = tfidf_matrix[current_doc_idx].toarray()[0]
            
            # Crear diccionario de scores TF-IDF
            tfidf_dict = dict(zip(feature_names, tfidf_scores))
        else:
            tfidf_dict = {}
        
        # Crear keywords
        for word, freq in word_freq.most_common(20):  # Top 20 keywords
            tfidf_score = tfidf_dict.get(word, 0.0)
            
            # Calcular score combinado
            score = freq * 0.7 + tfidf_score * 0.3
            
            # Encontrar posiciones
            positions = []
            for i, token in enumerate(relevant_tokens):
                if token.lemma.lower() == word:
                    positions.append(i)
            
            keyword_info = KeywordInfo(
                text=word,
                score=score,
                frequency=freq,
                tfidf_score=tfidf_score,
                position=positions
            )
            keywords.append(keyword_info)
        
        return keywords
    
    async def _analyze_topics(self, text: str) -> List[TopicInfo]:
        """Analizar tópicos del texto"""
        topics = []
        
        if not self.enable_gensim or len(self.analyses) < 3:
            return topics
        
        try:
            # Preparar corpus
            texts = []
            for analysis in self.analyses.values():
                doc_tokens = [token.lemma for token in analysis.tokens if not token.is_stop and token.is_alpha]
                texts.append(doc_tokens)
            
            # Crear diccionario y corpus
            if self.dictionary is None:
                self.dictionary = corpora.Dictionary(texts)
                self.corpus = [self.dictionary.doc2bow(text) for text in texts]
            
            # Entrenar modelo LDA
            if self.lda_model is None:
                self.lda_model = LdaModel(
                    corpus=self.corpus,
                    id2word=self.dictionary,
                    num_topics=self.config["topic_modeling"]["num_topics"],
                    passes=self.config["topic_modeling"]["passes"],
                    alpha=self.config["topic_modeling"]["alpha"],
                    eta=self.config["topic_modeling"]["eta"],
                    random_state=42
                )
            
            # Obtener tópicos para el documento actual
            doc_bow = self.dictionary.doc2bow([token.lemma for token in self.analyses[list(self.analyses.keys())[-1]].tokens if not token.is_stop])
            doc_topics = self.lda_model[doc_bow]
            
            # Crear información de tópicos
            for topic_id, weight in doc_topics:
                if weight > 0.1:  # Solo tópicos con peso significativo
                    topic_words = self.lda_model.show_topic(topic_id, topn=5)
                    keywords = [word for word, _ in topic_words]
                    
                    topic_info = TopicInfo(
                        id=topic_id,
                        name=f"Topic {topic_id}",
                        keywords=keywords,
                        weight=weight,
                        coherence=0.5,  # Placeholder
                        documents=[list(self.analyses.keys())[-1]]
                    )
                    topics.append(topic_info)
            
        except Exception as e:
            logger.error(f"Error in topic analysis: {e}")
        
        return topics
    
    async def _calculate_text_metrics(self, text: str, tokens: List[TokenInfo]) -> TextMetrics:
        """Calcular métricas de texto"""
        # Métricas básicas
        word_count = len([token for token in tokens if not token.is_punct])
        sentence_count = len(sent_tokenize(text))
        character_count = len(text)
        
        # Métricas de longitud
        avg_word_length = sum(len(token.text) for token in tokens if not token.is_punct) / max(word_count, 1)
        avg_sentence_length = word_count / max(sentence_count, 1)
        
        # Diversidad léxica (Type-Token Ratio)
        unique_words = len(set(token.lemma.lower() for token in tokens if not token.is_punct))
        lexical_diversity = unique_words / max(word_count, 1)
        
        # Puntuación de legibilidad (Flesch Reading Ease simplificado)
        readability_score = self._calculate_readability_score(text, word_count, sentence_count)
        
        # Puntuación de complejidad
        complexity_score = self._calculate_complexity_score(tokens)
        
        # Puntuación de formalidad
        formality_score = self._calculate_formality_score(tokens)
        
        return TextMetrics(
            word_count=word_count,
            sentence_count=sentence_count,
            character_count=character_count,
            avg_word_length=avg_word_length,
            avg_sentence_length=avg_sentence_length,
            lexical_diversity=lexical_diversity,
            readability_score=readability_score,
            complexity_score=complexity_score,
            formality_score=formality_score
        )
    
    def _calculate_readability_score(self, text: str, word_count: int, sentence_count: int) -> float:
        """Calcular puntuación de legibilidad"""
        if sentence_count == 0:
            return 0.0
        
        # Flesch Reading Ease simplificado
        avg_sentence_length = word_count / sentence_count
        
        # Contar sílabas (aproximación)
        syllables = sum(self._count_syllables(word) for word in text.split())
        avg_syllables_per_word = syllables / max(word_count, 1)
        
        # Fórmula Flesch Reading Ease
        score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
        
        # Normalizar entre 0 y 1
        return max(0, min(1, score / 100))
    
    def _count_syllables(self, word: str) -> int:
        """Contar sílabas en una palabra (aproximación)"""
        word = word.lower()
        vowels = "aeiouy"
        syllable_count = 0
        prev_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_was_vowel:
                syllable_count += 1
            prev_was_vowel = is_vowel
        
        # Ajustar para palabras que terminan en 'e'
        if word.endswith('e') and syllable_count > 1:
            syllable_count -= 1
        
        return max(1, syllable_count)
    
    def _calculate_complexity_score(self, tokens: List[TokenInfo]) -> float:
        """Calcular puntuación de complejidad"""
        if not tokens:
            return 0.0
        
        # Factores de complejidad
        long_words = len([token for token in tokens if len(token.text) > 6])
        complex_pos = len([token for token in tokens if token.pos in ['JJ', 'RB', 'VB', 'NN']])
        punctuation_ratio = len([token for token in tokens if token.is_punct]) / len(tokens)
        
        # Calcular score
        complexity = (long_words / len(tokens)) * 0.4 + (complex_pos / len(tokens)) * 0.4 + punctuation_ratio * 0.2
        
        return min(1.0, complexity)
    
    def _calculate_formality_score(self, tokens: List[TokenInfo]) -> float:
        """Calcular puntuación de formalidad"""
        if not tokens:
            return 0.0
        
        # Indicadores de formalidad
        formal_indicators = 0
        total_words = len([token for token in tokens if not token.is_punct])
        
        for token in tokens:
            if token.is_punct:
                continue
            
            # Palabras formales comunes
            if token.lemma.lower() in ['therefore', 'however', 'furthermore', 'moreover', 'consequently']:
                formal_indicators += 1
            # Contracciones indican informalidad
            elif "'" in token.text:
                formal_indicators -= 0.5
            # Palabras largas indican formalidad
            elif len(token.text) > 8:
                formal_indicators += 0.3
        
        formality_score = (formal_indicators / max(total_words, 1)) + 0.5
        return max(0.0, min(1.0, formality_score))
    
    async def _extract_ngrams_and_collocations(self, text: str, tokens: List[TokenInfo]) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str, str]], List[Tuple[str, str]]]:
        """Extraer n-gramas y colocaciones"""
        # Filtrar tokens relevantes
        relevant_tokens = [token.text.lower() for token in tokens if not token.is_stop and not token.is_punct and token.is_alpha]
        
        # Bigramas
        bigrams = list(zip(relevant_tokens[:-1], relevant_tokens[1:]))
        
        # Trigramas
        trigrams = list(zip(relevant_tokens[:-2], relevant_tokens[1:-1], relevant_tokens[2:]))
        
        # Colocaciones usando NLTK
        try:
            finder = BigramCollocationFinder.from_words(relevant_tokens)
            finder.apply_freq_filter(2)  # Mínimo 2 ocurrencias
            collocations = finder.nbest(BigramAssocMeasures.pmi, 10)
        except:
            collocations = []
        
        return bigrams[:20], trigrams[:20], collocations
    
    def _detect_language(self, text: str) -> str:
        """Detectar idioma del texto"""
        try:
            DetectorFactory.seed = 0  # Para resultados consistentes
            detected = detect(text)
            return detected
        except:
            return "en"  # Fallback a inglés
    
    async def compare_texts(self, text1: str, text2: str) -> Dict[str, Any]:
        """Comparar dos textos"""
        try:
            # Analizar ambos textos
            analysis1 = await self.analyze_text(text1, "temp_1")
            analysis2 = await self.analyze_text(text2, "temp_2")
            
            # Calcular similitudes
            similarity_scores = {}
            
            # Similitud de sentimiento
            if analysis1.sentiment and analysis2.sentiment:
                sentiment_sim = 1 - abs(analysis1.sentiment.polarity - analysis2.sentiment.polarity)
                similarity_scores["sentiment"] = sentiment_sim
            
            # Similitud de palabras clave
            keywords1 = set([kw.text for kw in analysis1.keywords])
            keywords2 = set([kw.text for kw in analysis2.keywords])
            if keywords1 or keywords2:
                keyword_sim = len(keywords1.intersection(keywords2)) / len(keywords1.union(keywords2))
                similarity_scores["keywords"] = keyword_sim
            
            # Similitud de métricas
            metrics_sim = self._calculate_metrics_similarity(analysis1.metrics, analysis2.metrics)
            similarity_scores["metrics"] = metrics_sim
            
            # Similitud semántica usando TF-IDF
            semantic_sim = await self._calculate_semantic_similarity(text1, text2)
            similarity_scores["semantic"] = semantic_sim
            
            # Similitud general
            overall_similarity = np.mean(list(similarity_scores.values()))
            
            return {
                "overall_similarity": overall_similarity,
                "similarity_breakdown": similarity_scores,
                "analysis_1": self._analysis_to_dict(analysis1),
                "analysis_2": self._analysis_to_dict(analysis2),
                "comparison_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error comparing texts: {e}")
            return {"error": str(e)}
    
    def _calculate_metrics_similarity(self, metrics1: TextMetrics, metrics2: TextMetrics) -> float:
        """Calcular similitud de métricas"""
        # Normalizar métricas
        m1 = [
            metrics1.word_count / 1000,  # Normalizar
            metrics1.avg_word_length / 10,
            metrics1.avg_sentence_length / 20,
            metrics1.lexical_diversity,
            metrics1.readability_score,
            metrics1.complexity_score,
            metrics1.formality_score
        ]
        
        m2 = [
            metrics2.word_count / 1000,
            metrics2.avg_word_length / 10,
            metrics2.avg_sentence_length / 20,
            metrics2.lexical_diversity,
            metrics2.readability_score,
            metrics2.complexity_score,
            metrics2.formality_score
        ]
        
        # Calcular similitud coseno
        similarity = cosine_similarity([m1], [m2])[0][0]
        return float(similarity)
    
    async def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calcular similitud semántica"""
        try:
            # Usar TF-IDF para similitud semántica
            corpus = [text1, text2]
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(corpus)
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(similarity)
        except:
            return 0.0
    
    def _analysis_to_dict(self, analysis: NLPAnalysis) -> Dict[str, Any]:
        """Convertir análisis a diccionario"""
        return {
            "document_id": analysis.document_id,
            "language": analysis.language,
            "tokens": [
                {
                    "text": token.text,
                    "pos": token.pos,
                    "lemma": token.lemma,
                    "is_stop": token.is_stop,
                    "is_punct": token.is_punct
                }
                for token in analysis.tokens
            ],
            "entities": [
                {
                    "text": entity.text,
                    "label": entity.label,
                    "start": entity.start,
                    "end": entity.end,
                    "confidence": entity.confidence
                }
                for entity in analysis.entities
            ],
            "sentiment": {
                "polarity": analysis.sentiment.polarity,
                "subjectivity": analysis.sentiment.subjectivity,
                "sentiment_type": analysis.sentiment.sentiment_type.value,
                "confidence": analysis.sentiment.confidence,
                "emotional_tone": analysis.sentiment.emotional_tone,
                "intensity": analysis.sentiment.intensity
            } if analysis.sentiment else None,
            "keywords": [
                {
                    "text": kw.text,
                    "score": kw.score,
                    "frequency": kw.frequency,
                    "tfidf_score": kw.tfidf_score
                }
                for kw in analysis.keywords
            ],
            "metrics": {
                "word_count": analysis.metrics.word_count,
                "sentence_count": analysis.metrics.sentence_count,
                "character_count": analysis.metrics.character_count,
                "avg_word_length": analysis.metrics.avg_word_length,
                "avg_sentence_length": analysis.metrics.avg_sentence_length,
                "lexical_diversity": analysis.metrics.lexical_diversity,
                "readability_score": analysis.metrics.readability_score,
                "complexity_score": analysis.metrics.complexity_score,
                "formality_score": analysis.metrics.formality_score
            },
            "topics": [
                {
                    "id": topic.id,
                    "name": topic.name,
                    "keywords": topic.keywords,
                    "weight": topic.weight,
                    "coherence": topic.coherence
                }
                for topic in analysis.topics
            ],
            "analyzed_at": analysis.analyzed_at.isoformat()
        }
    
    async def get_analysis_summary(self) -> Dict[str, Any]:
        """Obtener resumen de análisis realizados"""
        if not self.analyses:
            return {"message": "No analyses available"}
        
        # Estadísticas generales
        total_documents = len(self.analyses)
        languages = Counter([analysis.language for analysis in self.analyses.values()])
        
        # Estadísticas de sentimiento
        sentiments = Counter([analysis.sentiment.sentiment_type.value for analysis in self.analyses.values() if analysis.sentiment])
        
        # Estadísticas de métricas
        avg_metrics = {
            "word_count": np.mean([analysis.metrics.word_count for analysis in self.analyses.values()]),
            "readability_score": np.mean([analysis.metrics.readability_score for analysis in self.analyses.values()]),
            "complexity_score": np.mean([analysis.metrics.complexity_score for analysis in self.analyses.values()]),
            "formality_score": np.mean([analysis.metrics.formality_score for analysis in self.analyses.values()])
        }
        
        return {
            "total_documents": total_documents,
            "languages": dict(languages),
            "sentiment_distribution": dict(sentiments),
            "average_metrics": {k: float(v) for k, v in avg_metrics.items()},
            "last_analysis": max([analysis.analyzed_at for analysis in self.analyses.values()]).isoformat()
        }
    
    async def save_analysis(self, document_id: str, filepath: str = None) -> str:
        """Guardar análisis en archivo"""
        if document_id not in self.analyses:
            raise ValueError(f"Analysis for document {document_id} not found")
        
        if filepath is None:
            filepath = f"{self.cache_directory}/analysis_{document_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        analysis_data = self._analysis_to_dict(self.analyses[document_id])
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(analysis_data, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Analysis saved to {filepath}")
        return filepath
    
    async def load_analysis(self, filepath: str) -> NLPAnalysis:
        """Cargar análisis desde archivo"""
        with open(filepath, 'r', encoding='utf-8') as f:
            analysis_data = json.load(f)
        
        # Reconstruir análisis (simplificado)
        # En una implementación completa, se reconstruirían todos los objetos
        logger.info(f"Analysis loaded from {filepath}")
        return None  # Placeholder



























