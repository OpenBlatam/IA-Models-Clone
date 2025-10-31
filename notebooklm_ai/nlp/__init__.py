from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from .core.nlp_engine import (
from .processors.text_processor import (
from .processors.tokenizer import (
from .processors.embedder import (
from .analyzers.sentiment_analyzer import (
from .analyzers.keyword_extractor import (
from .analyzers.topic_modeler import (
from .analyzers.entity_recognizer import (
from .analyzers.summarizer import (
from .analyzers.classifier import (
from .utils.nlp_utils import (
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
NotebookLM AI - Sistema NLP Avanzado
🧠 Procesamiento de lenguaje natural con capacidades avanzadas
"""

    NLPEngine,
    NLPConfig,
    get_nlp_engine,
    cleanup_nlp_engine
)

    TextProcessor,
    TextProcessorConfig,
    get_text_processor
)

    AdvancedTokenizer,
    TokenizerConfig,
    get_tokenizer
)

    EmbeddingEngine,
    EmbeddingConfig,
    get_embedding_engine
)

    SentimentAnalyzer,
    SentimentConfig,
    get_sentiment_analyzer
)

    KeywordExtractor,
    KeywordConfig,
    get_keyword_extractor
)

    TopicModeler,
    TopicConfig,
    get_topic_modeler
)

    EntityRecognizer,
    EntityConfig,
    get_entity_recognizer
)

    TextSummarizer,
    SummaryConfig,
    get_summarizer
)

    TextClassifier,
    ClassificationConfig,
    get_classifier
)

    NLPUtils,
    TextMetrics,
    LanguageDetector
)

__all__ = [
    # Motor principal
    "NLPEngine",
    "NLPConfig", 
    "get_nlp_engine",
    "cleanup_nlp_engine",
    
    # Procesadores
    "TextProcessor",
    "TextProcessorConfig",
    "get_text_processor",
    
    "AdvancedTokenizer",
    "TokenizerConfig", 
    "get_tokenizer",
    
    "EmbeddingEngine",
    "EmbeddingConfig",
    "get_embedding_engine",
    
    # Analizadores
    "SentimentAnalyzer",
    "SentimentConfig",
    "get_sentiment_analyzer",
    
    "KeywordExtractor",
    "KeywordConfig",
    "get_keyword_extractor",
    
    "TopicModeler",
    "TopicConfig",
    "get_topic_modeler",
    
    "EntityRecognizer",
    "EntityConfig",
    "get_entity_recognizer",
    
    "TextSummarizer",
    "SummaryConfig",
    "get_summarizer",
    
    "TextClassifier",
    "ClassificationConfig",
    "get_classifier",
    
    # Utilidades
    "NLPUtils",
    "TextMetrics",
    "LanguageDetector"
]

__version__ = "2.0.0"
__author__ = "NotebookLM AI Team"
__description__ = "Sistema NLP avanzado para NotebookLM AI" 