"""
ML NLP Benchmark Utilities
Real, working utility functions for ML NLP Benchmark system
"""

import re
import string
import math
import hashlib
import base64
import json
import time
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class MLNLPBenchmarkUtils:
    """Utility functions for ML NLP Benchmark system"""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s.,!?;:()\-]', '', text)
        return text
    
    @staticmethod
    def extract_ngrams(text: str, n: int = 2) -> List[str]:
        """Extract n-grams from text"""
        words = text.lower().split()
        return [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]
    
    @staticmethod
    def calculate_text_statistics(text: str) -> Dict[str, Any]:
        """Calculate comprehensive text statistics"""
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        
        return {
            "character_count": len(text),
            "word_count": len(words),
            "sentence_count": len([s for s in sentences if s.strip()]),
            "paragraph_count": len([p for p in text.split('\n\n') if p.strip()]),
            "unique_words": len(set(word.lower() for word in words)),
            "average_word_length": np.mean([len(word) for word in words]) if words else 0,
            "average_sentence_length": len(words) / len(sentences) if sentences else 0,
            "vocabulary_richness": len(set(word.lower() for word in words)) / len(words) if words else 0,
            "punctuation_count": sum(1 for char in text if char in string.punctuation),
            "digit_count": sum(1 for char in text if char.isdigit()),
            "uppercase_count": sum(1 for char in text if char.isupper()),
            "lowercase_count": sum(1 for char in text if char.islower())
        }
    
    @staticmethod
    def detect_language_patterns(text: str) -> Dict[str, Any]:
        """Detect language patterns in text"""
        # Common English words
        english_words = {
            "the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it", "for", "not", "on", "with",
            "he", "as", "you", "do", "at", "this", "but", "his", "by", "from", "they", "we", "say", "her",
            "she", "or", "an", "will", "my", "one", "all", "would", "there", "their"
        }
        
        # Common Spanish words
        spanish_words = {
            "el", "la", "de", "que", "y", "a", "en", "un", "es", "se", "no", "te", "lo", "le", "da", "su",
            "por", "son", "con", "para", "al", "del", "los", "las", "una", "como", "más", "pero", "sus"
        }
        
        words = set(text.lower().split())
        
        english_score = len(words.intersection(english_words))
        spanish_score = len(words.intersection(spanish_words))
        
        if english_score > spanish_score:
            detected_language = "english"
            confidence = english_score / (english_score + spanish_score) if (english_score + spanish_score) > 0 else 0.5
        elif spanish_score > english_score:
            detected_language = "spanish"
            confidence = spanish_score / (english_score + spanish_score) if (english_score + spanish_score) > 0 else 0.5
        else:
            detected_language = "unknown"
            confidence = 0.5
        
        return {
            "detected_language": detected_language,
            "confidence": confidence,
            "english_score": english_score,
            "spanish_score": spanish_score
        }
    
    @staticmethod
    def extract_entities(text: str) -> Dict[str, List[str]]:
        """Extract named entities from text"""
        # Simple entity extraction using regex patterns
        entities = {
            "persons": re.findall(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', text),
            "organizations": re.findall(r'\b[A-Z][A-Za-z]+ (?:Inc|Corp|LLC|Ltd|Company|Corporation)\b', text),
            "locations": re.findall(r'\b[A-Z][a-z]+ (?:City|State|Country|Street|Avenue|Road)\b', text),
            "dates": re.findall(r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December) \d{1,2},? \d{4}\b', text),
            "emails": re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text),
            "urls": re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text),
            "phone_numbers": re.findall(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', text)
        }
        
        return entities
    
    @staticmethod
    def calculate_similarity(text1: str, text2: str, method: str = "jaccard") -> float:
        """Calculate similarity between two texts"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if method == "jaccard":
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            return intersection / union if union > 0 else 0.0
        
        elif method == "cosine":
            # Simple cosine similarity
            all_words = words1.union(words2)
            vec1 = [1 if word in words1 else 0 for word in all_words]
            vec2 = [1 if word in words2 else 0 for word in all_words]
            
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            magnitude1 = math.sqrt(sum(a * a for a in vec1))
            magnitude2 = math.sqrt(sum(a * a for a in vec2))
            
            return dot_product / (magnitude1 * magnitude2) if magnitude1 * magnitude2 > 0 else 0.0
        
        return 0.0
    
    @staticmethod
    def generate_text_hash(text: str, algorithm: str = "md5") -> str:
        """Generate hash for text"""
        if algorithm == "md5":
            return hashlib.md5(text.encode()).hexdigest()
        elif algorithm == "sha1":
            return hashlib.sha1(text.encode()).hexdigest()
        elif algorithm == "sha256":
            return hashlib.sha256(text.encode()).hexdigest()
        else:
            return hashlib.md5(text.encode()).hexdigest()
    
    @staticmethod
    def compress_text(text: str, method: str = "gzip") -> str:
        """Compress text using various methods"""
        if method == "gzip":
            import gzip
            compressed = gzip.compress(text.encode())
            return base64.b64encode(compressed).decode()
        elif method == "base64":
            return base64.b64encode(text.encode()).decode()
        else:
            return text
    
    @staticmethod
    def decompress_text(compressed_text: str, method: str = "gzip") -> str:
        """Decompress text"""
        if method == "gzip":
            import gzip
            decoded = base64.b64decode(compressed_text.encode())
            return gzip.decompress(decoded).decode()
        elif method == "base64":
            return base64.b64decode(compressed_text.encode()).decode()
        else:
            return compressed_text
    
    @staticmethod
    def create_text_summary(text: str, max_sentences: int = 3) -> str:
        """Create a simple text summary"""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) <= max_sentences:
            return text
        
        # Simple scoring based on word frequency
        word_freq = Counter(text.lower().split())
        sentence_scores = []
        
        for sentence in sentences:
            score = sum(word_freq.get(word.lower(), 0) for word in sentence.split())
            sentence_scores.append((score, sentence))
        
        # Sort by score and take top sentences
        sentence_scores.sort(reverse=True)
        top_sentences = [sentence for _, sentence in sentence_scores[:max_sentences]]
        
        return '. '.join(top_sentences) + '.'
    
    @staticmethod
    def extract_key_phrases(text: str, max_phrases: int = 10) -> List[str]:
        """Extract key phrases from text"""
        # Extract 2-grams and 3-grams
        bigrams = MLNLPBenchmarkUtils.extract_ngrams(text, 2)
        trigrams = MLNLPBenchmarkUtils.extract_ngrams(text, 3)
        
        # Count phrase frequency
        phrase_freq = Counter(bigrams + trigrams)
        
        # Filter out common stop phrases
        stop_phrases = {
            "the the", "and and", "of of", "in in", "to to", "a a", "is is", "it it",
            "that that", "for for", "with with", "as as", "this this", "but but"
        }
        
        filtered_phrases = {phrase: freq for phrase, freq in phrase_freq.items() 
                          if phrase not in stop_phrases and len(phrase.split()) > 1}
        
        # Return top phrases
        return [phrase for phrase, _ in Counter(filtered_phrases).most_common(max_phrases)]
    
    @staticmethod
    def analyze_text_quality(text: str) -> Dict[str, Any]:
        """Analyze text quality metrics"""
        stats = MLNLPBenchmarkUtils.calculate_text_statistics(text)
        
        # Quality metrics
        quality_score = 0.0
        
        # Length appropriateness (not too short, not too long)
        if 50 <= stats["word_count"] <= 1000:
            quality_score += 0.2
        
        # Vocabulary richness
        if stats["vocabulary_richness"] > 0.5:
            quality_score += 0.2
        
        # Sentence length variety
        if 10 <= stats["average_sentence_length"] <= 25:
            quality_score += 0.2
        
        # Punctuation usage
        if stats["punctuation_count"] > 0:
            quality_score += 0.1
        
        # Capitalization
        if stats["uppercase_count"] > 0:
            quality_score += 0.1
        
        # No excessive repetition
        words = text.lower().split()
        if len(set(words)) / len(words) > 0.7:
            quality_score += 0.2
        
        return {
            "quality_score": min(quality_score, 1.0),
            "word_count_appropriate": 50 <= stats["word_count"] <= 1000,
            "vocabulary_rich": stats["vocabulary_richness"] > 0.5,
            "sentence_length_appropriate": 10 <= stats["average_sentence_length"] <= 25,
            "has_punctuation": stats["punctuation_count"] > 0,
            "has_capitalization": stats["uppercase_count"] > 0,
            "low_repetition": len(set(words)) / len(words) > 0.7 if words else False
        }
    
    @staticmethod
    def format_analysis_result(result: Dict[str, Any], format_type: str = "json") -> str:
        """Format analysis result in different formats"""
        if format_type == "json":
            return json.dumps(result, indent=2, ensure_ascii=False)
        elif format_type == "csv":
            # Convert to CSV format
            rows = []
            for key, value in result.items():
                if isinstance(value, (dict, list)):
                    value = json.dumps(value)
                rows.append(f"{key},{value}")
            return "\n".join(rows)
        elif format_type == "text":
            # Convert to human-readable text
            lines = []
            for key, value in result.items():
                if isinstance(value, (dict, list)):
                    value = json.dumps(value, indent=2)
                lines.append(f"{key}: {value}")
            return "\n".join(lines)
        else:
            return str(result)
    
    @staticmethod
    def benchmark_performance(func, *args, **kwargs) -> Tuple[Any, float]:
        """Benchmark function performance"""
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        return result, execution_time
    
    @staticmethod
    def create_performance_report(results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create performance report from benchmark results"""
        if not results:
            return {}
        
        execution_times = [r.get("execution_time", 0) for r in results]
        
        return {
            "total_executions": len(results),
            "average_execution_time": np.mean(execution_times),
            "min_execution_time": np.min(execution_times),
            "max_execution_time": np.max(execution_times),
            "median_execution_time": np.median(execution_times),
            "std_execution_time": np.std(execution_times),
            "total_execution_time": np.sum(execution_times),
            "throughput_per_second": len(results) / np.sum(execution_times) if np.sum(execution_times) > 0 else 0
        }

# Global utility instance
ml_nlp_benchmark_utils = MLNLPBenchmarkUtils()











