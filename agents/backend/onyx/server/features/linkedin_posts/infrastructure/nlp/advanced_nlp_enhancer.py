from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import spacy
from transformers import pipeline
import textstat
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import language_tool_python
from keybert import KeyBERT
from typing import Dict, List, Any
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Advanced NLP Enhancer for LinkedIn Posts
========================================

Provides advanced NLP-based quality enhancement for LinkedIn posts using:
- spaCy (entities, grammar, similarity)
- transformers (rewriting, coherence)
- textstat (readability)
- vaderSentiment (sentiment)
- language_tool_python (grammar correction)
- keybert (keyword extraction)
"""


class AdvancedNLPEnhancer:
    def __init__(self) -> Any:
        self.nlp = spacy.load("en_core_web_sm")
        self.sentiment = SentimentIntensityAnalyzer()
        self.grammar_tool = language_tool_python.LanguageTool('en-US')
        self.keybert = KeyBERT('all-MiniLM-L6-v2')
        self.rewriter = pipeline("text2text-generation", model="google/flan-t5-base")

    def enhance_post(self, text: str) -> Dict[str, Any]:
        doc = self.nlp(text)
        sentiment = self.sentiment.polarity_scores(text)
        grammar_matches = self.grammar_tool.check(text)
        keywords = self.keybert.extract_keywords(text, top_n=5)
        readability = textstat.flesch_reading_ease(text)
        rewritten = self.rewriter(f"Improve this LinkedIn post: {text}", max_length=256)[0]['generated_text']

        return {
            "original": text,
            "rewritten": rewritten,
            "sentiment": sentiment,
            "entities": [(ent.text, ent.label_) for ent in doc.ents],
            "keywords": [kw[0] for kw in keywords],
            "readability": readability,
            "grammar_issues": len(grammar_matches),
            "grammar_suggestions": [m.message for m in grammar_matches[:3]],
        } 