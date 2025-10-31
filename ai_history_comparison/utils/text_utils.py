"""
Text Utilities

This module provides text processing and analysis utilities
used across the system.
"""

import re
import hashlib
import logging
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
import unicodedata

logger = logging.getLogger(__name__)


class TextUtils:
    """Utility class for text processing operations"""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text"""
        try:
            if not text:
                return ""
            
            # Normalize unicode
            text = unicodedata.normalize('NFKD', text)
            
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text)
            
            # Strip leading/trailing whitespace
            text = text.strip()
            
            return text
            
        except Exception as e:
            logger.error(f"Text cleaning failed: {e}")
            return text
    
    @staticmethod
    def extract_words(text: str) -> List[str]:
        """Extract words from text"""
        try:
            if not text:
                return []
            
            # Clean text first
            text = TextUtils.clean_text(text)
            
            # Extract words (alphanumeric sequences)
            words = re.findall(r'\b[a-zA-Z0-9]+\b', text.lower())
            
            return words
            
        except Exception as e:
            logger.error(f"Word extraction failed: {e}")
            return []
    
    @staticmethod
    def extract_sentences(text: str) -> List[str]:
        """Extract sentences from text"""
        try:
            if not text:
                return []
            
            # Clean text first
            text = TextUtils.clean_text(text)
            
            # Split by sentence endings
            sentences = re.split(r'[.!?]+', text)
            
            # Clean and filter sentences
            sentences = [s.strip() for s in sentences if s.strip()]
            
            return sentences
            
        except Exception as e:
            logger.error(f"Sentence extraction failed: {e}")
            return []
    
    @staticmethod
    def calculate_readability_score(text: str) -> float:
        """Calculate simple readability score"""
        try:
            if not text:
                return 0.0
            
            words = TextUtils.extract_words(text)
            sentences = TextUtils.extract_sentences(text)
            
            if not words or not sentences:
                return 0.0
            
            # Calculate average sentence length
            avg_sentence_length = len(words) / len(sentences)
            
            # Calculate average word length
            avg_word_length = sum(len(word) for word in words) / len(words)
            
            # Simple readability score (0-1, higher is more readable)
            readability = max(0.0, min(1.0, 1.0 - (avg_sentence_length / 30.0) - (avg_word_length / 10.0)))
            
            return readability
            
        except Exception as e:
            logger.error(f"Readability calculation failed: {e}")
            return 0.0
    
    @staticmethod
    def calculate_complexity_score(text: str) -> float:
        """Calculate text complexity score"""
        try:
            if not text:
                return 0.0
            
            words = TextUtils.extract_words(text)
            sentences = TextUtils.extract_sentences(text)
            
            if not words or not sentences:
                return 0.0
            
            # Calculate average sentence length
            avg_sentence_length = len(words) / len(sentences)
            
            # Calculate unique word ratio
            unique_words = len(set(words))
            unique_ratio = unique_words / len(words) if words else 0
            
            # Calculate complexity (0-1, higher is more complex)
            complexity = min(1.0, (avg_sentence_length / 20.0) + (unique_ratio * 0.5))
            
            return complexity
            
        except Exception as e:
            logger.error(f"Complexity calculation failed: {e}")
            return 0.0
    
    @staticmethod
    def calculate_sentiment_score(text: str) -> float:
        """Calculate simple sentiment score"""
        try:
            if not text:
                return 0.5  # Neutral
            
            # Simple sentiment word lists
            positive_words = {
                'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
                'awesome', 'brilliant', 'outstanding', 'perfect', 'love', 'best',
                'beautiful', 'incredible', 'superb', 'marvelous', 'fabulous'
            }
            
            negative_words = {
                'bad', 'terrible', 'awful', 'horrible', 'disappointing', 'poor',
                'worst', 'hate', 'disgusting', 'pathetic', 'useless', 'stupid',
                'annoying', 'frustrating', 'boring', 'dull', 'mediocre'
            }
            
            words = TextUtils.extract_words(text)
            
            if not words:
                return 0.5  # Neutral
            
            positive_count = sum(1 for word in words if word in positive_words)
            negative_count = sum(1 for word in words if word in negative_words)
            
            if positive_count + negative_count == 0:
                return 0.5  # Neutral
            
            # Calculate sentiment score (0-1, higher is more positive)
            sentiment = positive_count / (positive_count + negative_count)
            
            return sentiment
            
        except Exception as e:
            logger.error(f"Sentiment calculation failed: {e}")
            return 0.5  # Neutral
    
    @staticmethod
    def calculate_word_frequency(text: str, top_n: int = 10) -> List[Tuple[str, int]]:
        """Calculate word frequency in text"""
        try:
            if not text:
                return []
            
            words = TextUtils.extract_words(text)
            
            if not words:
                return []
            
            # Count word frequencies
            word_counts = Counter(words)
            
            # Get top N most frequent words
            top_words = word_counts.most_common(top_n)
            
            return top_words
            
        except Exception as e:
            logger.error(f"Word frequency calculation failed: {e}")
            return []
    
    @staticmethod
    def calculate_text_hash(text: str) -> str:
        """Calculate hash of text for comparison"""
        try:
            if not text:
                return ""
            
            # Clean text first
            clean_text = TextUtils.clean_text(text)
            
            # Calculate MD5 hash
            text_hash = hashlib.md5(clean_text.encode('utf-8')).hexdigest()
            
            return text_hash
            
        except Exception as e:
            logger.error(f"Text hash calculation failed: {e}")
            return ""
    
    @staticmethod
    def calculate_similarity(text1: str, text2: str) -> float:
        """Calculate similarity between two texts"""
        try:
            if not text1 or not text2:
                return 0.0
            
            # Extract words from both texts
            words1 = set(TextUtils.extract_words(text1))
            words2 = set(TextUtils.extract_words(text2))
            
            if not words1 or not words2:
                return 0.0
            
            # Calculate Jaccard similarity
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            
            similarity = intersection / union if union > 0 else 0.0
            
            return similarity
            
        except Exception as e:
            logger.error(f"Similarity calculation failed: {e}")
            return 0.0
    
    @staticmethod
    def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
        """Extract keywords from text"""
        try:
            if not text:
                return []
            
            # Get word frequencies
            word_freq = TextUtils.calculate_word_frequency(text, max_keywords * 2)
            
            # Filter out common stop words
            stop_words = {
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
                'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
                'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'
            }
            
            # Filter out stop words and get top keywords
            keywords = [
                word for word, count in word_freq 
                if word not in stop_words and len(word) > 2
            ][:max_keywords]
            
            return keywords
            
        except Exception as e:
            logger.error(f"Keyword extraction failed: {e}")
            return []
    
    @staticmethod
    def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
        """Truncate text to maximum length"""
        try:
            if not text:
                return ""
            
            if len(text) <= max_length:
                return text
            
            # Truncate and add suffix
            truncated = text[:max_length - len(suffix)] + suffix
            
            return truncated
            
        except Exception as e:
            logger.error(f"Text truncation failed: {e}")
            return text
    
    @staticmethod
    def count_characters(text: str) -> Dict[str, int]:
        """Count different types of characters in text"""
        try:
            if not text:
                return {"total": 0, "letters": 0, "digits": 0, "spaces": 0, "punctuation": 0}
            
            counts = {
                "total": len(text),
                "letters": len(re.findall(r'[a-zA-Z]', text)),
                "digits": len(re.findall(r'[0-9]', text)),
                "spaces": len(re.findall(r'\s', text)),
                "punctuation": len(re.findall(r'[^\w\s]', text))
            }
            
            return counts
            
        except Exception as e:
            logger.error(f"Character counting failed: {e}")
            return {"total": 0, "letters": 0, "digits": 0, "spaces": 0, "punctuation": 0}
    
    @staticmethod
    def detect_language(text: str) -> str:
        """Simple language detection"""
        try:
            if not text:
                return "unknown"
            
            # Simple language detection based on character patterns
            # This is a very basic implementation
            
            # Check for common English words
            english_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
            words = TextUtils.extract_words(text)
            
            if words:
                english_count = sum(1 for word in words if word in english_words)
                if english_count > len(words) * 0.1:  # 10% threshold
                    return "en"
            
            # Check for non-ASCII characters (basic detection)
            if any(ord(char) > 127 for char in text):
                return "non-ascii"
            
            return "en"  # Default to English
            
        except Exception as e:
            logger.error(f"Language detection failed: {e}")
            return "unknown"





















