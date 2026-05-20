from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import time
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from collections import defaultdict
import structlog
from ..processors.text_processor import TextProcessor, TextProcessorConfig
from ..processors.tokenizer import AdvancedTokenizer, TokenizerConfig
from ..processors.embedder import EmbeddingEngine, EmbeddingConfig
from ..analyzers.sentiment_analyzer import SentimentAnalyzer, SentimentConfig
from ..analyzers.keyword_extractor import KeywordExtractor, KeywordConfig
from ..analyzers.topic_modeler import TopicModeler, TopicConfig
from ..analyzers.entity_recognizer import EntityRecognizer, EntityConfig
from ..analyzers.summarizer import TextSummarizer, SummaryConfig
from ..analyzers.classifier import TextClassifier, ClassificationConfig
from ..utils.nlp_utils import NLPUtils, TextMetrics, LanguageDetector
from typing import Any, List, Dict, Optional
import logging
#!/usr/bin/env python3
"""
Motor Principal NLP - NotebookLM AI
🧠 Coordina todos los componentes del sistema NLP
"""



logger = structlog.get_logger()

@dataclass
class NLPConfig:
    """Configuración del motor NLP."""
    # Configuraciones de componentes
    text_processor_config: TextProcessorConfig = field(default_factory=TextProcessorConfig)
    tokenizer_config: TokenizerConfig = field(default_factory=TokenizerConfig)
    embedding_config: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    sentiment_config: SentimentConfig = field(default_factory=SentimentConfig)
    keyword_config: KeywordConfig = field(default_factory=KeywordConfig)
    topic_config: TopicConfig = field(default_factory=TopicConfig)
    entity_config: EntityConfig = field(default_factory=EntityConfig)
    summary_config: SummaryConfig = field(default_factory=SummaryConfig)
    classification_config: ClassificationConfig = field(default_factory=ClassificationConfig)
    
    # Configuración general
    enable_caching: bool = True
    enable_parallel_processing: bool = True
    max_workers: int = 4
    cache_ttl: int = 3600
    
    # Idiomas soportados
    supported_languages: List[str] = field(default_factory=lambda: ["es", "en", "fr", "de", "it", "pt"])
    default_language: str = "es"

class NLPEngine:
    """Motor principal del sistema NLP."""
    
    def __init__(self, config: NLPConfig = None):
        
    """__init__ function."""
self.config = config or NLPConfig()
        self.stats = defaultdict(int)
        
        # Inicializar componentes
        self.text_processor = TextProcessor(self.config.text_processor_config)
        self.tokenizer = AdvancedTokenizer(self.config.tokenizer_config)
        self.embedding_engine = EmbeddingEngine(self.config.embedding_config)
        self.sentiment_analyzer = SentimentAnalyzer(self.config.sentiment_config)
        self.keyword_extractor = KeywordExtractor(self.config.keyword_config)
        self.topic_modeler = TopicModeler(self.config.topic_config)
        self.entity_recognizer = EntityRecognizer(self.config.entity_config)
        self.summarizer = TextSummarizer(self.config.summary_config)
        self.classifier = TextClassifier(self.config.classification_config)
        
        # Utilidades
        self.nlp_utils = NLPUtils()
        self.text_metrics = TextMetrics()
        self.language_detector = LanguageDetector()
        
        logger.info("Motor NLP inicializado", config=self.config)
    
    async def process_text(self, text: str, tasks: List[str] = None) -> Dict[str, Any]:
        """Procesa texto con múltiples tareas NLP."""
        start_time = time.time()
        
        if tasks is None:
            tasks = ["preprocess", "tokenize", "embed", "sentiment", "keywords", "entities"]
        
        try:
            # Detectar idioma
            language = await self.language_detector.detect(text)
            
            # Preprocesamiento
            processed_text = await self.text_processor.preprocess(text, language)
            
            # Ejecutar tareas en paralelo si está habilitado
            if self.config.enable_parallel_processing:
                results = await self._process_parallel(processed_text, tasks, language)
            else:
                results = await self._process_sequential(processed_text, tasks, language)
            
            # Métricas del texto
            text_metrics = await self.text_metrics.calculate(processed_text)
            
            duration = time.time() - start_time
            self.stats["total_requests"] += 1
            self.stats["total_processing_time"] += duration
            
            return {
                "original_text": text,
                "processed_text": processed_text,
                "language": language,
                "tasks_executed": tasks,
                "results": results,
                "metrics": text_metrics,
                "processing_time_ms": duration * 1000,
                "timestamp": time.time()
            }
            
        except Exception as e:
            self.stats["errors"] += 1
            logger.error("Error procesando texto", error=str(e), text=text[:100])
            raise
    
    async def _process_parallel(self, text: str, tasks: List[str], language: str) -> Dict[str, Any]:
        """Procesa tareas en paralelo."""
        task_functions = {
            "tokenize": lambda: self.tokenizer.tokenize(text, language),
            "embed": lambda: self.embedding_engine.embed(text, language),
            "sentiment": lambda: self.sentiment_analyzer.analyze(text, language),
            "keywords": lambda: self.keyword_extractor.extract(text, language),
            "entities": lambda: self.entity_recognizer.extract(text, language),
            "topics": lambda: self.topic_modeler.extract_topics(text, language),
            "summary": lambda: self.summarizer.summarize(text, language),
            "classify": lambda: self.classifier.classify(text, language)
        }
        
        # Filtrar tareas válidas
        valid_tasks = [task for task in tasks if task in task_functions]
        
        # Ejecutar en paralelo
        tasks_to_run = [task_functions[task]() for task in valid_tasks]
        results = await asyncio.gather(*tasks_to_run, return_exceptions=True)
        
        # Construir resultado
        result_dict = {}
        for task, result in zip(valid_tasks, results):
            if isinstance(result, Exception):
                logger.error(f"Error en tarea {task}", error=str(result))
                result_dict[task] = {"error": str(result)}
            else:
                result_dict[task] = result
        
        return result_dict
    
    async def _process_sequential(self, text: str, tasks: List[str], language: str) -> Dict[str, Any]:
        """Procesa tareas secuencialmente."""
        results = {}
        
        for task in tasks:
            try:
                if task == "tokenize":
                    results[task] = await self.tokenizer.tokenize(text, language)
                elif task == "embed":
                    results[task] = await self.embedding_engine.embed(text, language)
                elif task == "sentiment":
                    results[task] = await self.sentiment_analyzer.analyze(text, language)
                elif task == "keywords":
                    results[task] = await self.keyword_extractor.extract(text, language)
                elif task == "entities":
                    results[task] = await self.entity_recognizer.extract(text, language)
                elif task == "topics":
                    results[task] = await self.topic_modeler.extract_topics(text, language)
                elif task == "summary":
                    results[task] = await self.summarizer.summarize(text, language)
                elif task == "classify":
                    results[task] = await self.classifier.classify(text, language)
            except Exception as e:
                logger.error(f"Error en tarea {task}", error=str(e))
                results[task] = {"error": str(e)}
        
        return results
    
    async def analyze_document(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Analiza un documento completo."""
        text = document.get("content", "")
        title = document.get("title", "")
        
        # Análisis completo
        analysis = await self.process_text(
            text, 
            tasks=["preprocess", "tokenize", "embed", "sentiment", "keywords", 
                   "entities", "topics", "summary", "classify"]
        )
        
        # Análisis del título
        title_analysis = await self.process_text(
            title,
            tasks=["keywords", "sentiment"]
        )
        
        return {
            "document_id": document.get("id"),
            "title": title,
            "title_analysis": title_analysis,
            "content_analysis": analysis,
            "document_metrics": {
                "word_count": len(text.split()),
                "character_count": len(text),
                "sentence_count": len(text.split('.')),
                "paragraph_count": len(text.split('\n\n'))
            }
        }
    
    async def compare_texts(self, text1: str, text2: str) -> Dict[str, Any]:
        """Compara dos textos."""
        # Procesar ambos textos
        analysis1 = await self.process_text(text1, ["embed", "sentiment", "keywords"])
        analysis2 = await self.process_text(text2, ["embed", "sentiment", "keywords"])
        
        # Calcular similitud
        similarity = await self.embedding_engine.calculate_similarity(
            analysis1["results"]["embed"],
            analysis2["results"]["embed"]
        )
        
        return {
            "text1_analysis": analysis1,
            "text2_analysis": analysis2,
            "similarity_score": similarity,
            "comparison_metrics": {
                "sentiment_difference": abs(
                    analysis1["results"]["sentiment"]["score"] - 
                    analysis2["results"]["sentiment"]["score"]
                ),
                "common_keywords": len(
                    set(analysis1["results"]["keywords"]) & 
                    set(analysis2["results"]["keywords"])
                )
            }
        }
    
    async def batch_process(self, texts: List[str], tasks: List[str] = None) -> List[Dict[str, Any]]:
        """Procesa múltiples textos en lote."""
        if tasks is None:
            tasks = ["preprocess", "tokenize", "sentiment", "keywords"]
        
        results = []
        for text in texts:
            try:
                result = await self.process_text(text, tasks)
                results.append(result)
            except Exception as e:
                logger.error("Error en procesamiento por lotes", error=str(e))
                results.append({"error": str(e), "text": text[:100]})
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del motor."""
        avg_processing_time = 0
        if self.stats["total_requests"] > 0:
            avg_processing_time = self.stats["total_processing_time"] / self.stats["total_requests"]
        
        return {
            "total_requests": self.stats["total_requests"],
            "errors": self.stats["errors"],
            "avg_processing_time_ms": avg_processing_time * 1000,
            "total_processing_time": self.stats["total_processing_time"],
            "error_rate": self.stats["errors"] / max(1, self.stats["total_requests"])
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Verifica la salud del motor NLP."""
        test_text = "Este es un texto de prueba para verificar el funcionamiento del motor NLP."
        
        try:
            # Probar procesamiento básico
            result = await self.process_text(test_text, ["preprocess", "tokenize"])
            
            return {
                "status": "healthy",
                "components": {
                    "text_processor": "healthy",
                    "tokenizer": "healthy",
                    "embedding_engine": "healthy",
                    "analyzers": "healthy"
                },
                "test_result": result,
                "timestamp": time.time()
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": time.time()
            }
    
    async def cleanup(self) -> Any:
        """Limpia recursos del motor."""
        # Limpiar componentes
        await self.embedding_engine.cleanup()
        await self.topic_modeler.cleanup()
        
        logger.info("Motor NLP limpiado")

# Instancia global
_nlp_engine = None

def get_nlp_engine(config: NLPConfig = None) -> NLPEngine:
    """Obtiene la instancia global del motor NLP."""
    global _nlp_engine
    if _nlp_engine is None:
        _nlp_engine = NLPEngine(config)
    return _nlp_engine

async def cleanup_nlp_engine():
    """Limpia la instancia global del motor NLP."""
    global _nlp_engine
    if _nlp_engine:
        await _nlp_engine.cleanup()
        _nlp_engine = None 