"""
NLP Configuration
================

Configuration settings for the NLP system with optimized model selections
and performance settings.
"""

import os
from typing import Dict, Any, Optional, List
from pydantic import BaseSettings, Field
from enum import Enum

class NLPLanguage(str, Enum):
    """Supported languages for NLP processing."""
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    ITALIAN = "it"
    PORTUGUESE = "pt"
    CHINESE = "zh"
    JAPANESE = "ja"
    KOREAN = "ko"
    RUSSIAN = "ru"
    ARABIC = "ar"
    DUTCH = "nl"
    POLISH = "pl"
    TURKISH = "tr"
    HINDI = "hi"

class ModelProvider(str, Enum):
    """NLP model providers."""
    HUGGINGFACE = "huggingface"
    SPACY = "spacy"
    STANZA = "stanza"
    FLAIR = "flair"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"

class NLPTask(str, Enum):
    """NLP tasks supported."""
    SENTIMENT_ANALYSIS = "sentiment"
    NAMED_ENTITY_RECOGNITION = "ner"
    TEXT_CLASSIFICATION = "classification"
    SUMMARIZATION = "summarization"
    TRANSLATION = "translation"
    QUESTION_ANSWERING = "qa"
    TEXT_GENERATION = "generation"
    TOPIC_MODELING = "topic_modeling"
    KEYWORD_EXTRACTION = "keywords"
    LANGUAGE_DETECTION = "language_detection"

class NLPConfig(BaseSettings):
    """Configuration for NLP system."""
    
    # Model Configuration
    default_language: NLPLanguage = Field(default=NLPLanguage.ENGLISH, env="NLP_DEFAULT_LANGUAGE")
    model_provider: ModelProvider = Field(default=ModelProvider.HUGGINGFACE, env="NLP_MODEL_PROVIDER")
    
    # Performance Settings
    max_text_length: int = Field(default=512, env="NLP_MAX_TEXT_LENGTH")
    batch_size: int = Field(default=32, env="NLP_BATCH_SIZE")
    max_concurrent_requests: int = Field(default=10, env="NLP_MAX_CONCURRENT_REQUESTS")
    cache_models: bool = Field(default=True, env="NLP_CACHE_MODELS")
    model_cache_ttl: int = Field(default=3600, env="NLP_MODEL_CACHE_TTL")  # seconds
    
    # Model Selection
    sentiment_model: str = Field(
        default="cardiffnlp/twitter-roberta-base-sentiment-latest",
        env="NLP_SENTIMENT_MODEL"
    )
    ner_model: str = Field(
        default="dbmdz/bert-large-cased-finetuned-conll03-english",
        env="NLP_NER_MODEL"
    )
    classification_model: str = Field(
        default="microsoft/DialoGPT-medium",
        env="NLP_CLASSIFICATION_MODEL"
    )
    summarization_model: str = Field(
        default="facebook/bart-large-cnn",
        env="NLP_SUMMARIZATION_MODEL"
    )
    translation_model: str = Field(
        default="Helsinki-NLP/opus-mt-en-es",
        env="NLP_TRANSLATION_MODEL"
    )
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        env="NLP_EMBEDDING_MODEL"
    )
    
    # Advanced Model Settings
    use_gpu: bool = Field(default=True, env="NLP_USE_GPU")
    gpu_memory_fraction: float = Field(default=0.8, env="NLP_GPU_MEMORY_FRACTION")
    precision: str = Field(default="fp16", env="NLP_PRECISION")  # fp16, fp32, int8
    
    # Text Processing Settings
    min_text_length: int = Field(default=10, env="NLP_MIN_TEXT_LENGTH")
    max_sentences: int = Field(default=50, env="NLP_MAX_SENTENCES")
    remove_stopwords: bool = Field(default=True, env="NLP_REMOVE_STOPWORDS")
    lemmatize: bool = Field(default=True, env="NLP_LEMMATIZE")
    remove_punctuation: bool = Field(default=False, env="NLP_REMOVE_PUNCTUATION")
    
    # Sentiment Analysis Settings
    sentiment_threshold: float = Field(default=0.1, env="NLP_SENTIMENT_THRESHOLD")
    use_ensemble_sentiment: bool = Field(default=True, env="NLP_USE_ENSEMBLE_SENTIMENT")
    sentiment_models: List[str] = Field(
        default=[
            "cardiffnlp/twitter-roberta-base-sentiment-latest",
            "nlptown/bert-base-multilingual-uncased-sentiment",
            "distilbert-base-uncased-finetuned-sst-2-english"
        ],
        env="NLP_SENTIMENT_MODELS"
    )
    
    # Entity Recognition Settings
    entity_confidence_threshold: float = Field(default=0.5, env="NLP_ENTITY_CONFIDENCE_THRESHOLD")
    custom_entities: List[str] = Field(default=[], env="NLP_CUSTOM_ENTITIES")
    entity_types: List[str] = Field(
        default=["PERSON", "ORG", "GPE", "MONEY", "DATE", "TIME"],
        env="NLP_ENTITY_TYPES"
    )
    
    # Topic Modeling Settings
    num_topics: int = Field(default=10, env="NLP_NUM_TOPICS")
    topic_words: int = Field(default=10, env="NLP_TOPIC_WORDS")
    topic_iterations: int = Field(default=100, env="NLP_TOPIC_ITERATIONS")
    use_lda: bool = Field(default=True, env="NLP_USE_LDA")
    use_bertopic: bool = Field(default=False, env="NLP_USE_BERTOPIC")
    
    # Keyword Extraction Settings
    max_keywords: int = Field(default=20, env="NLP_MAX_KEYWORDS")
    keyword_algorithm: str = Field(default="tfidf", env="NLP_KEYWORD_ALGORITHM")  # tfidf, textrank, yake
    keyword_min_length: int = Field(default=3, env="NLP_KEYWORD_MIN_LENGTH")
    keyword_max_length: int = Field(default=20, env="NLP_KEYWORD_MAX_LENGTH")
    
    # Summarization Settings
    max_summary_length: int = Field(default=150, env="NLP_MAX_SUMMARY_LENGTH")
    min_summary_length: int = Field(default=30, env="NLP_MIN_SUMMARY_LENGTH")
    use_extractive: bool = Field(default=True, env="NLP_USE_EXTRACTIVE")
    use_abstractive: bool = Field(default=False, env="NLP_USE_ABSTRACTIVE")
    
    # Translation Settings
    supported_languages: List[str] = Field(
        default=["en", "es", "fr", "de", "it", "pt", "zh", "ja", "ko", "ru", "ar"],
        env="NLP_SUPPORTED_LANGUAGES"
    )
    translation_quality: str = Field(default="high", env="NLP_TRANSLATION_QUALITY")  # low, medium, high
    
    # Classification Settings
    classification_threshold: float = Field(default=0.5, env="NLP_CLASSIFICATION_THRESHOLD")
    use_multilabel: bool = Field(default=False, env="NLP_USE_MULTILABEL")
    max_categories: int = Field(default=10, env="NLP_MAX_CATEGORIES")
    
    # Performance Optimization
    enable_caching: bool = Field(default=True, env="NLP_ENABLE_CACHING")
    cache_size: int = Field(default=1000, env="NLP_CACHE_SIZE")
    cache_ttl: int = Field(default=3600, env="NLP_CACHE_TTL")
    
    # Monitoring & Logging
    enable_metrics: bool = Field(default=True, env="NLP_ENABLE_METRICS")
    log_predictions: bool = Field(default=False, env="NLP_LOG_PREDICTIONS")
    log_level: str = Field(default="INFO", env="NLP_LOG_LEVEL")
    
    # API Settings
    api_timeout: int = Field(default=30, env="NLP_API_TIMEOUT")
    rate_limit: int = Field(default=100, env="NLP_RATE_LIMIT")  # requests per minute
    max_request_size: int = Field(default=10485760, env="NLP_MAX_REQUEST_SIZE")  # 10MB
    
    # External API Keys
    huggingface_token: Optional[str] = Field(default=None, env="HUGGINGFACE_TOKEN")
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    google_api_key: Optional[str] = Field(default=None, env="GOOGLE_API_KEY")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

# Global NLP configuration
nlp_config = NLPConfig()

# Model configurations for different tasks
MODEL_CONFIGS = {
    "sentiment": {
        "models": [
            {
                "name": "cardiffnlp/twitter-roberta-base-sentiment-latest",
                "provider": "huggingface",
                "task": "sentiment-analysis",
                "languages": ["en"],
                "accuracy": 0.95,
                "speed": "fast"
            },
            {
                "name": "nlptown/bert-base-multilingual-uncased-sentiment",
                "provider": "huggingface",
                "task": "sentiment-analysis",
                "languages": ["en", "es", "fr", "de", "it", "pt"],
                "accuracy": 0.92,
                "speed": "medium"
            },
            {
                "name": "distilbert-base-uncased-finetuned-sst-2-english",
                "provider": "huggingface",
                "task": "sentiment-analysis",
                "languages": ["en"],
                "accuracy": 0.90,
                "speed": "fast"
            }
        ]
    },
    "ner": {
        "models": [
            {
                "name": "dbmdz/bert-large-cased-finetuned-conll03-english",
                "provider": "huggingface",
                "task": "ner",
                "languages": ["en"],
                "accuracy": 0.94,
                "speed": "medium"
            },
            {
                "name": "xlm-roberta-large-finetuned-conll03-english",
                "provider": "huggingface",
                "task": "ner",
                "languages": ["en"],
                "accuracy": 0.96,
                "speed": "slow"
            },
            {
                "name": "en_core_web_sm",
                "provider": "spacy",
                "task": "ner",
                "languages": ["en"],
                "accuracy": 0.85,
                "speed": "fast"
            }
        ]
    },
    "classification": {
        "models": [
            {
                "name": "microsoft/DialoGPT-medium",
                "provider": "huggingface",
                "task": "text-classification",
                "languages": ["en"],
                "accuracy": 0.88,
                "speed": "medium"
            },
            {
                "name": "distilbert-base-uncased",
                "provider": "huggingface",
                "task": "text-classification",
                "languages": ["en"],
                "accuracy": 0.85,
                "speed": "fast"
            }
        ]
    },
    "summarization": {
        "models": [
            {
                "name": "facebook/bart-large-cnn",
                "provider": "huggingface",
                "task": "summarization",
                "languages": ["en"],
                "accuracy": 0.92,
                "speed": "medium"
            },
            {
                "name": "google/pegasus-xsum",
                "provider": "huggingface",
                "task": "summarization",
                "languages": ["en"],
                "accuracy": 0.94,
                "speed": "slow"
            }
        ]
    },
    "translation": {
        "models": [
            {
                "name": "Helsinki-NLP/opus-mt-en-es",
                "provider": "huggingface",
                "task": "translation",
                "languages": ["en", "es"],
                "accuracy": 0.90,
                "speed": "fast"
            },
            {
                "name": "Helsinki-NLP/opus-mt-en-fr",
                "provider": "huggingface",
                "task": "translation",
                "languages": ["en", "fr"],
                "accuracy": 0.91,
                "speed": "fast"
            }
        ]
    }
}

# Language-specific configurations
LANGUAGE_CONFIGS = {
    "en": {
        "spacy_model": "en_core_web_sm",
        "stopwords": "english",
        "sentiment_model": "cardiffnlp/twitter-roberta-base-sentiment-latest",
        "ner_model": "dbmdz/bert-large-cased-finetuned-conll03-english"
    },
    "es": {
        "spacy_model": "es_core_news_sm",
        "stopwords": "spanish",
        "sentiment_model": "nlptown/bert-base-multilingual-uncased-sentiment",
        "ner_model": "xlm-roberta-large-finetuned-conll03-english"
    },
    "fr": {
        "spacy_model": "fr_core_news_sm",
        "stopwords": "french",
        "sentiment_model": "nlptown/bert-base-multilingual-uncased-sentiment",
        "ner_model": "xlm-roberta-large-finetuned-conll03-english"
    },
    "de": {
        "spacy_model": "de_core_news_sm",
        "stopwords": "german",
        "sentiment_model": "nlptown/bert-base-multilingual-uncased-sentiment",
        "ner_model": "xlm-roberta-large-finetuned-conll03-english"
    }
}

# Performance optimization settings
PERFORMANCE_CONFIGS = {
    "fast": {
        "batch_size": 64,
        "max_length": 256,
        "use_fp16": True,
        "cache_models": True
    },
    "balanced": {
        "batch_size": 32,
        "max_length": 512,
        "use_fp16": True,
        "cache_models": True
    },
    "accurate": {
        "batch_size": 16,
        "max_length": 1024,
        "use_fp16": False,
        "cache_models": True
    }
}

def get_model_config(task: str, language: str = "en", performance: str = "balanced") -> Dict[str, Any]:
    """Get model configuration for a specific task and language."""
    task_configs = MODEL_CONFIGS.get(task, {})
    models = task_configs.get("models", [])
    
    # Filter by language
    language_models = [m for m in models if language in m.get("languages", [])]
    
    if not language_models:
        # Fallback to English models
        language_models = [m for m in models if "en" in m.get("languages", [])]
    
    if not language_models:
        return {}
    
    # Select best model based on performance preference
    if performance == "fast":
        return min(language_models, key=lambda x: x.get("speed", "medium"))
    elif performance == "accurate":
        return max(language_models, key=lambda x: x.get("accuracy", 0))
    else:  # balanced
        return language_models[0]

def get_language_config(language: str) -> Dict[str, Any]:
    """Get language-specific configuration."""
    return LANGUAGE_CONFIGS.get(language, LANGUAGE_CONFIGS["en"])

def get_performance_config(performance: str) -> Dict[str, Any]:
    """Get performance configuration."""
    return PERFORMANCE_CONFIGS.get(performance, PERFORMANCE_CONFIGS["balanced"])

def is_language_supported(language: str) -> bool:
    """Check if language is supported."""
    return language in LANGUAGE_CONFIGS

def get_supported_languages() -> List[str]:
    """Get list of supported languages."""
    return list(LANGUAGE_CONFIGS.keys())

def get_available_tasks() -> List[str]:
    """Get list of available NLP tasks."""
    return list(MODEL_CONFIGS.keys())












