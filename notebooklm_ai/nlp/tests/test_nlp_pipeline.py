from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import asyncio
import pytest
from ..core.nlp_engine import NLPEngine, NLPConfig
from ..analyzers.sentiment_analyzer import SentimentAnalyzer, SentimentConfig
from ..analyzers.keyword_extractor import KeywordExtractor, KeywordConfig
from ..analyzers.topic_modeler import TopicModeler, TopicConfig
from ..analyzers.entity_recognizer import EntityRecognizer, EntityConfig

from typing import Any, List, Dict, Optional
import logging
@pytest.mark.asyncio
async def test_nlp_pipeline_end_to_end():
    
    """test_nlp_pipeline_end_to_end function."""
engine = NLPEngine(NLPConfig())
    text = "John Doe works at Acme Corp. He is very happy with the new AI-powered product launched in 2024. Contact: john@example.com, https://acme.com"
    result = await engine.process_text(text, tasks=["preprocess", "tokenize", "sentiment", "keywords", "entities", "topics"])
    assert "results" in result
    assert "sentiment" in result["results"]
    assert "keywords" in result["results"]
    assert "entities" in result["results"]
    assert "topics" in result["results"]
    assert result["language"] in ["en", "es"]
    assert result["metrics"]["char_count"] > 0

@pytest.mark.asyncio
async def test_sentiment_analyzer():
    
    """test_sentiment_analyzer function."""
analyzer = SentimentAnalyzer(SentimentConfig())
    text = "I love this product! It is fantastic."
    result = await analyzer.analyze(text, language="en")
    assert "score" in result
    assert result["score"] > 0
    assert result["label"] == "positive"

@pytest.mark.asyncio
async def test_keyword_extractor():
    
    """test_keyword_extractor function."""
extractor = KeywordExtractor(KeywordConfig())
    text = "AI, machine learning, and deep learning are revolutionizing technology."
    result = await extractor.extract(text, language="en")
    assert "keywords" in result
    assert len(result["keywords"]) > 0

@pytest.mark.asyncio
async def test_topic_modeler():
    
    """test_topic_modeler function."""
modeler = TopicModeler(TopicConfig())
    texts = [
        "AI is transforming healthcare and finance.",
        "Machine learning enables new business models.",
        "Deep learning powers autonomous vehicles."
    ]
    result = await modeler.model_topics(texts, language="en")
    assert "topics" in result
    assert len(result["topics"]) > 0

@pytest.mark.asyncio
async def test_entity_recognizer():
    
    """test_entity_recognizer function."""
recognizer = EntityRecognizer(EntityConfig())
    text = "Contact Jane Smith at jane@company.com or visit https://company.com."
    result = await recognizer.extract(text, language="en")
    assert "entities" in result
    assert "email" in result["entities"]
    assert "url" in result["entities"]

@pytest.mark.asyncio
async def test_health_checks():
    
    """test_health_checks function."""
engine = NLPEngine(NLPConfig())
    assert (await engine.health_check())["status"] == "healthy"
    assert (await SentimentAnalyzer().health_check())["status"] == "healthy"
    assert (await KeywordExtractor().health_check())["status"] == "healthy"
    assert (await TopicModeler().health_check())["status"] == "healthy"
    assert (await EntityRecognizer().health_check())["status"] == "healthy" 