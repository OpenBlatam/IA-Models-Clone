from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import asyncio
import time
import re
import hashlib
import functools
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from collections import Counter, defaultdict, OrderedDict
import structlog
import json
from concurrent.futures import ThreadPoolExecutor
import threading
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF, TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from typing import Any, List, Dict, Optional
import logging
#!/usr/bin/env python3
"""
Modelador de Tópicos Ultra-Optimizado - NotebookLM AI
📊 Modelado avanzado de tópicos para producción con ML
"""


logger = structlog.get_logger()

# Cache LRU thread-safe
class LRUCache:
    """Cache LRU thread-safe para modelado de tópicos."""
    
    def __init__(self, maxsize: int = 100):
        
    """__init__ function."""
self.maxsize = maxsize
        self.cache = OrderedDict()
        self.lock = threading.Lock()
    
    def get(self, key: str) -> Optional[Any]:
        with self.lock:
            if key in self.cache:
                value = self.cache.pop(key)
                self.cache[key] = value
                return value
            return None
    
    def put(self, key: str, value: Any):
        
    """put function."""
with self.lock:
            if key in self.cache:
                self.cache.pop(key)
            elif len(self.cache) >= self.maxsize:
                self.cache.popitem(last=False)
            self.cache[key] = value
    
    def clear(self) -> Any:
        with self.lock:
            self.cache.clear()

@dataclass
class TopicConfig:
    """Configuración avanzada del modelador de tópicos."""
    # Modelado
    num_topics: int = 10
    min_topic_size: int = 5
    max_topic_size: int = 20
    
    # Algoritmos
    use_lda: bool = True
    use_nmf: bool = True
    use_hierarchical: bool = True
    use_kmeans: bool = True
    use_lsa: bool = False
    
    # Configuración avanzada
    enable_topic_evolution: bool = False
    enable_topic_similarity: bool = True
    enable_keyword_clustering: bool = True
    enable_coherence_analysis: bool = True
    
    # Filtros
    min_word_frequency: int = 2
    max_word_frequency: float = 0.8  # Porcentaje del corpus
    min_document_frequency: int = 1
    max_document_frequency: float = 0.95
    
    # Cache y rendimiento
    enable_caching: bool = True
    cache_maxsize: int = 100
    batch_size: int = 50
    max_workers: int = 4
    
    # ML Models
    use_sklearn_models: bool = True
    random_state: int = 42
    max_iter: int = 100
    learning_method: str = "batch"  # batch, online

class TopicModeler:
    """Modelador de tópicos ultra-optimizado."""
    
    def __init__(self, config: TopicConfig = None):
        
    """__init__ function."""
self.config = config or TopicConfig()
        self.stats = defaultdict(int)
        self.cache = LRUCache(self.config.cache_maxsize) if self.config.enable_caching else None
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
        
        # Stopwords multilingües
        self.stopwords = {
            "es": {
                "el", "la", "los", "las", "un", "una", "unos", "unas",
                "y", "o", "pero", "si", "no", "que", "cual", "quien",
                "donde", "cuando", "como", "por", "para", "con", "sin",
                "sobre", "entre", "detrás", "delante", "encima", "debajo",
                "es", "son", "está", "están", "era", "eran", "fue", "fueron",
                "ser", "estar", "tener", "haber", "hacer", "decir", "ver",
                "este", "esta", "estos", "estas", "ese", "esa", "esos", "esas",
                "aquel", "aquella", "aquellos", "aquellas", "mío", "mía", "míos", "mías",
                "tu", "tus", "su", "sus", "nuestro", "nuestra", "nuestros", "nuestras",
                "vuestro", "vuestra", "vuestros", "vuestras", "su", "sus"
            },
            "en": {
                "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
                "of", "with", "by", "is", "are", "was", "were", "be", "been", "being",
                "have", "has", "had", "do", "does", "did", "will", "would", "could",
                "should", "may", "might", "can", "this", "that", "these", "those",
                "i", "you", "he", "she", "it", "we", "they", "me", "him", "her",
                "us", "them", "my", "your", "his", "her", "its", "our", "their"
            }
        }
        
        # Tópicos predefinidos multilingües
        self.predefined_topics = {
            "es": {
                "tecnología": ["tecnología", "software", "programación", "desarrollo", "digital"],
                "negocios": ["negocios", "empresa", "comercial", "ventas", "marketing"],
                "educación": ["educación", "aprendizaje", "enseñanza", "estudio", "académico"],
                "salud": ["salud", "médico", "tratamiento", "enfermedad", "cura"],
                "deportes": ["deportes", "fútbol", "baloncesto", "atletismo", "competencia"],
                "política": ["política", "gobierno", "elecciones", "partido", "democracia"],
                "entretenimiento": ["entretenimiento", "película", "música", "arte", "cultura"],
                "ciencia": ["ciencia", "investigación", "descubrimiento", "experimento", "teoría"]
            },
            "en": {
                "technology": ["technology", "software", "programming", "development", "digital"],
                "business": ["business", "company", "commercial", "sales", "marketing"],
                "education": ["education", "learning", "teaching", "study", "academic"],
                "health": ["health", "medical", "treatment", "disease", "cure"],
                "sports": ["sports", "football", "basketball", "athletics", "competition"],
                "politics": ["politics", "government", "elections", "party", "democracy"],
                "entertainment": ["entertainment", "movie", "music", "art", "culture"],
                "science": ["science", "research", "discovery", "experiment", "theory"]
            }
        }
    
    def _generate_cache_key(self, texts: List[str], language: str) -> str:
        """Genera clave única para el cache."""
        # Usar hash del contenido de los textos
        content = f"{len(texts)}:{language}:{self.config.num_topics}:{self.config.min_word_frequency}"
        text_hash = hashlib.md5(' '.join(texts).encode()).hexdigest()[:16]
        return f"{content}:{text_hash}"
    
    def _detect_language(self, texts: List[str]) -> str:
        """Detecta el idioma de los textos."""
        all_words = []
        for text in texts:
            all_words.extend(text.lower().split())
        
        scores = defaultdict(int)
        for word in all_words:
            for lang, stopwords in self.stopwords.items():
                if word in stopwords:
                    scores[lang] += 1
        
        if scores:
            return max(scores.items(), key=lambda x: x[1])[0]
        
        return "es"  # Default
    
    async def model_topics(self, texts: List[str], language: str = "auto") -> Dict[str, Any]:
        """Modela tópicos a partir de una lista de textos con cache y optimizaciones."""
        start_time = time.time()
        
        try:
            # Detectar idioma si es necesario
            if language == "auto":
                language = self._detect_language(texts)
            
            # Verificar cache
            if self.cache:
                cache_key = self._generate_cache_key(texts, language)
                cached_result = self.cache.get(cache_key)
                if cached_result:
                    self.stats["cache_hits"] += 1
                    return cached_result
            
            # Preprocesamiento
            processed_texts = await self._preprocess_texts(texts, language)
            
            # Extraer vocabulario
            vocabulary = await self._extract_vocabulary(processed_texts)
            
            # Modelado de tópicos
            topics = {}
            
            if self.config.use_lda:
                topics["lda"] = await self._lda_modeling(processed_texts, vocabulary)
            
            if self.config.use_nmf:
                topics["nmf"] = await self._nmf_modeling(processed_texts, vocabulary)
            
            if self.config.use_hierarchical:
                topics["hierarchical"] = await self._hierarchical_modeling(processed_texts, vocabulary)
            
            if self.config.use_kmeans:
                topics["kmeans"] = await self._kmeans_modeling(processed_texts, vocabulary)
            
            if self.config.use_lsa:
                topics["lsa"] = await self._lsa_modeling(processed_texts, vocabulary)
            
            # Análisis de similitud
            topic_similarity = {}
            if self.config.enable_topic_similarity:
                topic_similarity = await self._calculate_topic_similarity(topics)
            
            # Clustering de palabras clave
            keyword_clusters = {}
            if self.config.enable_keyword_clustering:
                keyword_clusters = await self._cluster_keywords(vocabulary)
            
            # Análisis de coherencia
            coherence_analysis = {}
            if self.config.enable_coherence_analysis:
                coherence_analysis = await self._analyze_coherence(topics, processed_texts)
            
            duration = time.time() - start_time
            self.stats["total_modelings"] += 1
            self.stats["total_processing_time"] += duration
            
            result = {
                "texts_count": len(texts),
                "language": language,
                "topics": topics,
                "vocabulary_size": len(vocabulary),
                "topic_similarity": topic_similarity,
                "keyword_clusters": keyword_clusters,
                "coherence_analysis": coherence_analysis,
                "processing_time_ms": duration * 1000,
                "timestamp": time.time(),
                "config": {
                    "num_topics": self.config.num_topics,
                    "algorithms_used": list(topics.keys()),
                    "vocabulary_size": len(vocabulary)
                }
            }
            
            # Guardar en cache
            if self.cache:
                cache_key = self._generate_cache_key(texts, language)
                self.cache.put(cache_key, result)
            
            return result
            
        except Exception as e:
            self.stats["errors"] += 1
            logger.error("Error en modelado de tópicos", error=str(e))
            raise
    
    async def _preprocess_texts(self, texts: List[str], language: str) -> List[List[str]]:
        """Preprocesa los textos para modelado optimizado."""
        processed_texts = []
        
        for text in texts:
            # Limpiar texto
            cleaned = re.sub(r'[^\w\s]', ' ', text.lower())
            cleaned = re.sub(r'\s+', ' ', cleaned).strip()
            
            # Tokenizar
            words = cleaned.split()
            
            # Filtrar stopwords y palabras cortas
            stopwords = self.stopwords.get(language, self.stopwords["es"])
            filtered_words = [
                word for word in words 
                if word not in stopwords 
                and len(word) >= self.config.min_topic_size
                and not word.isdigit()
            ]
            
            processed_texts.append(filtered_words)
        
        return processed_texts
    
    async def _extract_vocabulary(self, processed_texts: List[List[str]]) -> Dict[str, int]:
        """Extrae el vocabulario del corpus optimizado."""
        word_freq = Counter()
        
        for text in processed_texts:
            word_freq.update(text)
        
        # Filtrar por frecuencia
        total_texts = len(processed_texts)
        filtered_vocab = {}
        
        for word, freq in word_freq.items():
            doc_freq = sum(1 for text in processed_texts if word in text)
            if (freq >= self.config.min_word_frequency and 
                freq / total_texts <= self.config.max_word_frequency and
                doc_freq >= self.config.min_document_frequency and
                doc_freq / total_texts <= self.config.max_document_frequency):
                filtered_vocab[word] = freq
        
        return filtered_vocab
    
    async def _lda_modeling(self, processed_texts: List[List[str]], vocabulary: Dict[str, int]) -> Dict[str, Any]:
        """Modelado LDA usando sklearn."""
        if not self.config.use_sklearn_models:
            return await self._lda_modeling_simple(processed_texts, vocabulary)
        
        try:
            # Preparar datos para sklearn
            texts_for_vectorizer = [' '.join(text) for text in processed_texts]
            
            # Vectorizar
            vectorizer = TfidfVectorizer(
                max_features=len(vocabulary),
                min_df=self.config.min_document_frequency,
                max_df=self.config.max_document_frequency,
                stop_words=list(self.stopwords.get("es", set()))
            )
            
            tfidf_matrix = vectorizer.fit_transform(texts_for_vectorizer)
            feature_names = vectorizer.get_feature_names_out()
            
            # LDA
            lda = LatentDirichletAllocation(
                n_components=self.config.num_topics,
                random_state=self.config.random_state,
                max_iter=self.config.max_iter,
                learning_method=self.config.learning_method
            )
            
            lda.fit(tfidf_matrix)
            
            # Extraer tópicos
            topics = []
            for topic_idx, topic in enumerate(lda.components_):
                top_words_idx = topic.argsort()[-self.config.max_topic_size:][::-1]
                topic_words = []
                
                for idx in top_words_idx:
                    word = feature_names[idx]
                    probability = topic[idx]
                    topic_words.append({
                        "word": word,
                        "probability": float(probability),
                        "frequency": vocabulary.get(word, 0)
                    })
                
                # Calcular coherencia
                coherence = self._calculate_topic_coherence(topic_words)
                
                topics.append({
                    "topic_id": topic_idx,
                    "name": f"LDA Tópico {topic_idx + 1}",
                    "words": topic_words,
                    "coherence": coherence,
                    "method": "lda_sklearn"
                })
            
            return {
                "method": "lda_sklearn",
                "num_topics": self.config.num_topics,
                "topics": topics,
                "doc_term_matrix_shape": tfidf_matrix.shape,
                "feature_names_count": len(feature_names)
            }
            
        except Exception as e:
            logger.warning(f"Error en LDA sklearn, usando implementación simple: {e}")
            return await self._lda_modeling_simple(processed_texts, vocabulary)
    
    async def _lda_modeling_simple(self, processed_texts: List[List[str]], vocabulary: Dict[str, int]) -> Dict[str, Any]:
        """Modelado LDA simplificado."""
        word_to_id = {word: i for i, word in enumerate(vocabulary.keys())}
        id_to_word = {i: word for word, i in word_to_id.items()}
        
        # Crear matriz de documentos
        doc_term_matrix = []
        for text in processed_texts:
            doc_vector = [0] * len(vocabulary)
            for word in text:
                if word in word_to_id:
                    doc_vector[word_to_id[word]] += 1
            doc_term_matrix.append(doc_vector)
        
        # Simulación de LDA
        topics = []
        for topic_id in range(self.config.num_topics):
            topic_words = []
            for word_id, word in id_to_word.items():
                prob = vocabulary[word] / sum(vocabulary.values())
                topic_words.append({
                    "word": word,
                    "probability": prob,
                    "frequency": vocabulary[word]
                })
            
            topic_words.sort(key=lambda x: x["probability"], reverse=True)
            
            topics.append({
                "topic_id": topic_id,
                "name": f"LDA Tópico {topic_id + 1}",
                "words": topic_words[:self.config.max_topic_size],
                "coherence": self._calculate_topic_coherence(topic_words[:10]),
                "method": "lda_simple"
            })
        
        return {
            "method": "lda_simple",
            "num_topics": self.config.num_topics,
            "topics": topics,
            "doc_term_matrix_shape": (len(processed_texts), len(vocabulary))
        }
    
    async def _nmf_modeling(self, processed_texts: List[List[str]], vocabulary: Dict[str, int]) -> Dict[str, Any]:
        """Modelado NMF usando sklearn."""
        if not self.config.use_sklearn_models:
            return await self._nmf_modeling_simple(processed_texts, vocabulary)
        
        try:
            # Preparar datos
            texts_for_vectorizer = [' '.join(text) for text in processed_texts]
            
            # Vectorizar
            vectorizer = TfidfVectorizer(
                max_features=len(vocabulary),
                min_df=self.config.min_document_frequency,
                max_df=self.config.max_document_frequency,
                stop_words=list(self.stopwords.get("es", set()))
            )
            
            tfidf_matrix = vectorizer.fit_transform(texts_for_vectorizer)
            feature_names = vectorizer.get_feature_names_out()
            
            # NMF
            nmf = NMF(
                n_components=self.config.num_topics,
                random_state=self.config.random_state,
                max_iter=self.config.max_iter
            )
            
            nmf.fit(tfidf_matrix)
            
            # Extraer tópicos
            topics = []
            for topic_idx, topic in enumerate(nmf.components_):
                top_words_idx = topic.argsort()[-self.config.max_topic_size:][::-1]
                topic_words = []
                
                for idx in top_words_idx:
                    word = feature_names[idx]
                    weight = topic[idx]
                    topic_words.append({
                        "word": word,
                        "weight": float(weight),
                        "frequency": vocabulary.get(word, 0)
                    })
                
                coherence = self._calculate_topic_coherence(topic_words)
                
                topics.append({
                    "topic_id": topic_idx,
                    "name": f"NMF Tópico {topic_idx + 1}",
                    "words": topic_words,
                    "coherence": coherence,
                    "method": "nmf_sklearn"
                })
            
            return {
                "method": "nmf_sklearn",
                "num_topics": self.config.num_topics,
                "topics": topics,
                "doc_term_matrix_shape": tfidf_matrix.shape
            }
            
        except Exception as e:
            logger.warning(f"Error en NMF sklearn, usando implementación simple: {e}")
            return await self._nmf_modeling_simple(processed_texts, vocabulary)
    
    async def _nmf_modeling_simple(self, processed_texts: List[List[str]], vocabulary: Dict[str, int]) -> Dict[str, Any]:
        """Modelado NMF simplificado."""
        word_to_id = {word: i for i, word in enumerate(vocabulary.keys())}
        id_to_word = {i: word for word, i in word_to_id.items()}
        
        # Crear matriz de documentos
        doc_term_matrix = []
        for text in processed_texts:
            doc_vector = [0] * len(vocabulary)
            for word in text:
                if word in word_to_id:
                    doc_vector[word_to_id[word]] += 1
            doc_term_matrix.append(doc_vector)
        
        # Simulación de NMF
        topics = []
        for topic_id in range(self.config.num_topics):
            topic_words = []
            for word_id, word in id_to_word.items():
                prob = max(0, vocabulary[word] / sum(vocabulary.values()) - topic_id * 0.01)
                topic_words.append({
                    "word": word,
                    "weight": prob,
                    "frequency": vocabulary[word]
                })
            
            topic_words.sort(key=lambda x: x["weight"], reverse=True)
            
            topics.append({
                "topic_id": topic_id,
                "name": f"NMF Tópico {topic_id + 1}",
                "words": topic_words[:self.config.max_topic_size],
                "coherence": self._calculate_topic_coherence(topic_words[:10]),
                "method": "nmf_simple"
            })
        
        return {
            "method": "nmf_simple",
            "num_topics": self.config.num_topics,
            "topics": topics,
            "doc_term_matrix_shape": (len(processed_texts), len(vocabulary))
        }
    
    async def _hierarchical_modeling(self, processed_texts: List[List[str]], vocabulary: Dict[str, int]) -> Dict[str, Any]:
        """Modelado jerárquico de tópicos."""
        word_to_id = {word: i for i, word in enumerate(vocabulary.keys())}
        id_to_word = {i: word for word, i in word_to_id.items()}
        
        # Simular clustering jerárquico basado en similitud de palabras
        word_clusters = self._cluster_words_by_similarity(list(vocabulary.keys()))
        
        topics = []
        for cluster_id, cluster_words in enumerate(word_clusters[:self.config.num_topics]):
            topic_words = []
            for word in cluster_words:
                topic_words.append({
                    "word": word,
                    "weight": vocabulary[word] / sum(vocabulary.values()),
                    "frequency": vocabulary[word],
                    "cluster_id": cluster_id
                })
            
            topic_words.sort(key=lambda x: x["weight"], reverse=True)
            
            topics.append({
                "topic_id": cluster_id,
                "name": f"Cluster {cluster_id + 1}",
                "words": topic_words[:self.config.max_topic_size],
                "cluster_size": len(cluster_words),
                "coherence": self._calculate_topic_coherence(topic_words[:10]),
                "method": "hierarchical"
            })
        
        return {
            "method": "hierarchical",
            "num_topics": len(topics),
            "topics": topics,
            "clustering_method": "word_similarity"
        }
    
    async def _kmeans_modeling(self, processed_texts: List[List[str]], vocabulary: Dict[str, int]) -> Dict[str, Any]:
        """Modelado usando K-means clustering."""
        if not self.config.use_sklearn_models:
            return await self._kmeans_modeling_simple(processed_texts, vocabulary)
        
        try:
            # Preparar datos
            texts_for_vectorizer = [' '.join(text) for text in processed_texts]
            
            # Vectorizar
            vectorizer = TfidfVectorizer(
                max_features=len(vocabulary),
                min_df=self.config.min_document_frequency,
                max_df=self.config.max_document_frequency,
                stop_words=list(self.stopwords.get("es", set()))
            )
            
            tfidf_matrix = vectorizer.fit_transform(texts_for_vectorizer)
            feature_names = vectorizer.get_feature_names_out()
            
            # K-means
            kmeans = KMeans(
                n_clusters=self.config.num_topics,
                random_state=self.config.random_state,
                max_iter=self.config.max_iter
            )
            
            kmeans.fit(tfidf_matrix)
            
            # Extraer tópicos basados en centroides
            topics = []
            for topic_idx, centroid in enumerate(kmeans.cluster_centers_):
                top_words_idx = centroid.argsort()[-self.config.max_topic_size:][::-1]
                topic_words = []
                
                for idx in top_words_idx:
                    word = feature_names[idx]
                    weight = centroid[idx]
                    topic_words.append({
                        "word": word,
                        "weight": float(weight),
                        "frequency": vocabulary.get(word, 0)
                    })
                
                coherence = self._calculate_topic_coherence(topic_words)
                
                topics.append({
                    "topic_id": topic_idx,
                    "name": f"K-means Cluster {topic_idx + 1}",
                    "words": topic_words,
                    "coherence": coherence,
                    "method": "kmeans_sklearn"
                })
            
            return {
                "method": "kmeans_sklearn",
                "num_topics": self.config.num_topics,
                "topics": topics,
                "doc_term_matrix_shape": tfidf_matrix.shape
            }
            
        except Exception as e:
            logger.warning(f"Error en K-means sklearn, usando implementación simple: {e}")
            return await self._kmeans_modeling_simple(processed_texts, vocabulary)
    
    async def _kmeans_modeling_simple(self, processed_texts: List[List[str]], vocabulary: Dict[str, int]) -> Dict[str, Any]:
        """Modelado K-means simplificado."""
        # Implementación simple basada en frecuencia de palabras
        word_freq = Counter()
        for text in processed_texts:
            word_freq.update(text)
        
        # Agrupar palabras por frecuencia
        freq_groups = defaultdict(list)
        for word, freq in word_freq.items():
            if word in vocabulary:
                freq_groups[freq // 10].append(word)
        
        topics = []
        for topic_idx, (freq_group, words) in enumerate(list(freq_groups.items())[:self.config.num_topics]):
            topic_words = []
            for word in words:
                topic_words.append({
                    "word": word,
                    "weight": vocabulary[word] / sum(vocabulary.values()),
                    "frequency": vocabulary[word],
                    "freq_group": freq_group
                })
            
            topic_words.sort(key=lambda x: x["weight"], reverse=True)
            
            topics.append({
                "topic_id": topic_idx,
                "name": f"Freq Group {topic_idx + 1}",
                "words": topic_words[:self.config.max_topic_size],
                "coherence": self._calculate_topic_coherence(topic_words[:10]),
                "method": "kmeans_simple"
            })
        
        return {
            "method": "kmeans_simple",
            "num_topics": len(topics),
            "topics": topics
        }
    
    async def _lsa_modeling(self, processed_texts: List[List[str]], vocabulary: Dict[str, int]) -> Dict[str, Any]:
        """Modelado LSA (Latent Semantic Analysis) usando SVD."""
        if not self.config.use_sklearn_models:
            return await self._lsa_modeling_simple(processed_texts, vocabulary)
        
        try:
            # Preparar datos
            texts_for_vectorizer = [' '.join(text) for text in processed_texts]
            
            # Vectorizar
            vectorizer = TfidfVectorizer(
                max_features=len(vocabulary),
                min_df=self.config.min_document_frequency,
                max_df=self.config.max_document_frequency,
                stop_words=list(self.stopwords.get("es", set()))
            )
            
            tfidf_matrix = vectorizer.fit_transform(texts_for_vectorizer)
            feature_names = vectorizer.get_feature_names_out()
            
            # LSA con SVD
            lsa = TruncatedSVD(
                n_components=self.config.num_topics,
                random_state=self.config.random_state
            )
            
            lsa.fit(tfidf_matrix)
            
            # Extraer tópicos
            topics = []
            for topic_idx, topic in enumerate(lsa.components_):
                top_words_idx = topic.argsort()[-self.config.max_topic_size:][::-1]
                topic_words = []
                
                for idx in top_words_idx:
                    word = feature_names[idx]
                    weight = topic[idx]
                    topic_words.append({
                        "word": word,
                        "weight": float(weight),
                        "frequency": vocabulary.get(word, 0)
                    })
                
                coherence = self._calculate_topic_coherence(topic_words)
                
                topics.append({
                    "topic_id": topic_idx,
                    "name": f"LSA Tópico {topic_idx + 1}",
                    "words": topic_words,
                    "coherence": coherence,
                    "method": "lsa_sklearn"
                })
            
            return {
                "method": "lsa_sklearn",
                "num_topics": self.config.num_topics,
                "topics": topics,
                "doc_term_matrix_shape": tfidf_matrix.shape,
                "explained_variance_ratio": lsa.explained_variance_ratio_.tolist()
            }
            
        except Exception as e:
            logger.warning(f"Error en LSA sklearn, usando implementación simple: {e}")
            return await self._lsa_modeling_simple(processed_texts, vocabulary)
    
    async def _lsa_modeling_simple(self, processed_texts: List[List[str]], vocabulary: Dict[str, int]) -> Dict[str, Any]:
        """Modelado LSA simplificado."""
        # Implementación simple basada en similitud de documentos
        topics = []
        for topic_id in range(self.config.num_topics):
            topic_words = []
            for word, freq in vocabulary.items():
                # Simular análisis semántico latente
                weight = freq / sum(vocabulary.values()) * (1 + topic_id * 0.1)
                topic_words.append({
                    "word": word,
                    "weight": weight,
                    "frequency": freq
                })
            
            topic_words.sort(key=lambda x: x["weight"], reverse=True)
            
            topics.append({
                "topic_id": topic_id,
                "name": f"LSA Tópico {topic_id + 1}",
                "words": topic_words[:self.config.max_topic_size],
                "coherence": self._calculate_topic_coherence(topic_words[:10]),
                "method": "lsa_simple"
            })
        
        return {
            "method": "lsa_simple",
            "num_topics": self.config.num_topics,
            "topics": topics
        }
    
    def _cluster_words_by_similarity(self, words: List[str]) -> List[List[str]]:
        """Agrupa palabras por similitud."""
        clusters = []
        used_words = set()
        
        for word in words:
            if word in used_words:
                continue
            
            cluster = [word]
            used_words.add(word)
            
            # Buscar palabras similares
            for other_word in words:
                if other_word in used_words:
                    continue
                
                # Similitud simple basada en caracteres comunes
                similarity = self._calculate_word_similarity(word, other_word)
                if similarity > 0.5:  # Umbral de similitud
                    cluster.append(other_word)
                    used_words.add(other_word)
            
            if len(cluster) >= self.config.min_topic_size:
                clusters.append(cluster)
        
        return clusters
    
    def _calculate_word_similarity(self, word1: str, word2: str) -> float:
        """Calcula similitud entre dos palabras."""
        # Similitud de Jaccard
        chars1 = set(word1)
        chars2 = set(word2)
        
        intersection = len(chars1.intersection(chars2))
        union = len(chars1.union(chars2))
        
        return intersection / union if union > 0 else 0
    
    def _calculate_topic_coherence(self, topic_words: List[Dict[str, Any]]) -> float:
        """Calcula la coherencia del tópico."""
        if len(topic_words) < 2:
            return 0.0
        
        # Coherencia simple basada en frecuencia promedio
        avg_freq = sum(word.get("frequency", 0) for word in topic_words) / len(topic_words)
        max_freq = max(word.get("frequency", 0) for word in topic_words)
        
        return avg_freq / max_freq if max_freq > 0 else 0.0
    
    async def _calculate_topic_similarity(self, topics: Dict[str, Any]) -> Dict[str, Any]:
        """Calcula similitud entre tópicos."""
        similarities = {}
        
        for method1, result1 in topics.items():
            for method2, result2 in topics.items():
                if method1 >= method2:
                    continue
                
                key = f"{method1}_vs_{method2}"
                similarities[key] = self._calculate_method_similarity(result1, result2)
        
        return similarities
    
    def _calculate_method_similarity(self, result1: Dict[str, Any], result2: Dict[str, Any]) -> float:
        """Calcula similitud entre dos métodos de modelado."""
        # Extraer palabras de ambos métodos
        words1 = set()
        words2 = set()
        
        for topic in result1.get("topics", []):
            for word_info in topic.get("words", []):
                words1.add(word_info["word"])
        
        for topic in result2.get("topics", []):
            for word_info in topic.get("words", []):
                words2.add(word_info["word"])
        
        # Similitud de Jaccard
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    async def _cluster_keywords(self, vocabulary: Dict[str, int]) -> Dict[str, Any]:
        """Agrupa palabras clave en clusters."""
        words = list(vocabulary.keys())
        clusters = self._cluster_words_by_similarity(words)
        
        cluster_info = []
        for cluster_id, cluster_words in enumerate(clusters):
            cluster_info.append({
                "cluster_id": cluster_id,
                "words": cluster_words,
                "size": len(cluster_words),
                "total_frequency": sum(vocabulary[word] for word in cluster_words),
                "avg_frequency": sum(vocabulary[word] for word in cluster_words) / len(cluster_words)
            })
        
        return {
            "num_clusters": len(clusters),
            "clusters": cluster_info,
            "total_words": len(words)
        }
    
    async def _analyze_coherence(self, topics: Dict[str, Any], processed_texts: List[List[str]]) -> Dict[str, Any]:
        """Analiza la coherencia de los tópicos."""
        coherence_analysis = {}
        
        for method, result in topics.items():
            method_coherence = []
            for topic in result.get("topics", []):
                topic_words = [word_info["word"] for word_info in topic.get("words", [])]
                
                # Calcular coherencia basada en co-ocurrencia
                coherence_score = self._calculate_coherence_score(topic_words, processed_texts)
                
                method_coherence.append({
                    "topic_id": topic["topic_id"],
                    "topic_name": topic["name"],
                    "coherence_score": coherence_score,
                    "word_count": len(topic_words)
                })
            
            coherence_analysis[method] = {
                "topics_coherence": method_coherence,
                "avg_coherence": sum(t["coherence_score"] for t in method_coherence) / len(method_coherence) if method_coherence else 0
            }
        
        return coherence_analysis
    
    def _calculate_coherence_score(self, topic_words: List[str], processed_texts: List[List[str]]) -> float:
        """Calcula score de coherencia basado en co-ocurrencia."""
        if len(topic_words) < 2:
            return 0.0
        
        # Contar documentos que contienen al menos dos palabras del tópico
        co_occurrence_count = 0
        total_docs = len(processed_texts)
        
        for text in processed_texts:
            text_words = set(text)
            topic_words_in_text = [word for word in topic_words if word in text_words]
            if len(topic_words_in_text) >= 2:
                co_occurrence_count += 1
        
        return co_occurrence_count / total_docs if total_docs > 0 else 0.0
    
    async def assign_topics_to_texts(self, texts: List[str], topics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Asigna tópicos a textos individuales."""
        processed_texts = await self._preprocess_texts(texts, "auto")
        assignments = []
        
        for i, text in enumerate(processed_texts):
            text_assignments = {}
            
            for method, result in topics.items():
                if "topics" not in result:
                    continue
                
                # Calcular similitud con cada tópico
                topic_scores = []
                for topic in result["topics"]:
                    topic_words = {word_info["word"] for word_info in topic["words"]}
                    text_words = set(text)
                    
                    # Similitud de Jaccard
                    intersection = len(topic_words.intersection(text_words))
                    union = len(topic_words.union(text_words))
                    similarity = intersection / union if union > 0 else 0
                    
                    topic_scores.append({
                        "topic_id": topic["topic_id"],
                        "topic_name": topic["name"],
                        "similarity": similarity
                    })
                
                # Ordenar por similitud
                topic_scores.sort(key=lambda x: x["similarity"], reverse=True)
                text_assignments[method] = topic_scores
            
            assignments.append({
                "text_id": i,
                "text_preview": " ".join(text[:10]) + "...",
                "topic_assignments": text_assignments
            })
        
        return assignments
    
    async def batch_model_topics(self, text_batches: List[List[str]], language: str = "auto") -> List[Dict[str, Any]]:
        """Modela tópicos de múltiples lotes de textos con procesamiento paralelo."""
        if not text_batches:
            return []
        
        # Procesar en lotes
        results = []
        for i in range(0, len(text_batches), self.config.batch_size):
            batch = text_batches[i:i + self.config.batch_size]
            
            # Crear tareas asíncronas para el lote
            tasks = [self.model_topics(texts, language) for texts in batch]
            
            # Ejecutar en paralelo
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Procesar resultados
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    logger.error("Error en modelado por lotes", error=str(result))
                    results.append({
                        "error": str(result),
                        "batch_index": i + j
                    })
                else:
                    results.append(result)
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas detalladas del modelador."""
        avg_processing_time = 0
        if self.stats["total_modelings"] > 0:
            avg_processing_time = self.stats["total_processing_time"] / self.stats["total_modelings"]
        
        cache_hit_rate = 0
        if self.cache and self.stats["total_modelings"] > 0:
            cache_hit_rate = self.stats["cache_hits"] / self.stats["total_modelings"]
        
        return {
            "total_modelings": self.stats["total_modelings"],
            "errors": self.stats["errors"],
            "cache_hits": self.stats.get("cache_hits", 0),
            "cache_hit_rate": cache_hit_rate,
            "avg_processing_time_ms": avg_processing_time * 1000,
            "error_rate": self.stats["errors"] / max(1, self.stats["total_modelings"]),
            "cache_enabled": self.cache is not None,
            "cache_size": len(self.cache.cache) if self.cache else 0
        }
    
    def clear_cache(self) -> Any:
        """Limpia el cache."""
        if self.cache:
            self.cache.clear()
            logger.info("Cache limpiado")
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check del modelador."""
        try:
            # Test básico
            test_texts = [
                "Este es un texto sobre tecnología y programación.",
                "Otro texto sobre negocios y marketing.",
                "Un texto más sobre educación y aprendizaje."
            ]
            test_result = await self.model_topics(test_texts, "es")
            
            return {
                "status": "healthy",
                "cache_working": self.cache is not None,
                "test_modeling": len(test_result["topics"]) > 0,
                "stats": self.get_stats()
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "stats": self.get_stats()
            }

# Instancia global
_topic_modeler = None

def get_topic_modeler(config: TopicConfig = None) -> TopicModeler:
    """Obtiene la instancia global del modelador de tópicos."""
    global _topic_modeler
    if _topic_modeler is None:
        _topic_modeler = TopicModeler(config)
    return _topic_modeler 