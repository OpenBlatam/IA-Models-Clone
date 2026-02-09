"""
Rust Accelerator Service

High-performance operations using the Rust core module.
Provides fallback to pure Python if Rust module is not available.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Callable
from functools import lru_cache
import hashlib
import re
from collections import Counter
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

try:
    from ..rust_core.python.transcriber_core import (
        TextProcessor as RustTextProcessor,
        SearchEngine as RustSearchEngine,
        CacheService as RustCacheService,
        BatchProcessor as RustBatchProcessor,
        HashService as RustHashService,
        SimilarityEngine as RustSimilarityEngine,
        LanguageDetector as RustLanguageDetector,
        RUST_AVAILABLE,
    )
    logger.info("Rust core module loaded successfully")
except ImportError:
    RUST_AVAILABLE = False
    logger.warning("Rust core module not available, using Python fallback")


@dataclass
class AcceleratorStats:
    """Statistics for the accelerator service."""
    rust_available: bool = False
    operations_count: int = 0
    rust_operations: int = 0
    python_fallback_operations: int = 0
    total_time_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "rust_available": self.rust_available,
            "operations_count": self.operations_count,
            "rust_operations": self.rust_operations,
            "python_fallback_operations": self.python_fallback_operations,
            "total_time_ms": self.total_time_ms,
            "rust_percentage": (
                self.rust_operations / max(self.operations_count, 1) * 100
            )
        }


class RustAccelerator:
    """
    High-performance operations accelerator using Rust extensions.
    
    Provides unified interface with automatic fallback to Python implementations
    when Rust module is not available.
    """
    
    def __init__(
        self,
        cache_max_size: int = 10000,
        cache_ttl_seconds: Optional[int] = 3600,
        similarity_threshold: float = 0.8,
        min_word_length: int = 2,
    ):
        self.stats = AcceleratorStats(rust_available=RUST_AVAILABLE)
        
        if RUST_AVAILABLE:
            self._text_processor = RustTextProcessor(None, min_word_length)
            self._search_engine = RustSearchEngine()
            self._cache = RustCacheService(cache_max_size, cache_ttl_seconds, True)
            self._batch_processor = RustBatchProcessor(None)
            self._hash_service = RustHashService("blake3")
            self._similarity_engine = RustSimilarityEngine(similarity_threshold, False)
            self._language_detector = RustLanguageDetector(0.5, 20)
        else:
            self._text_processor = None
            self._search_engine = None
            self._cache = None
            self._batch_processor = None
            self._hash_service = None
            self._similarity_engine = None
            self._language_detector = None
            self._fallback_cache: Dict[str, str] = {}
            self._fallback_documents: Dict[str, Dict] = {}
        
        self._similarity_threshold = similarity_threshold
        self._min_word_length = min_word_length
    
    def _track_operation(self, used_rust: bool):
        """Track operation statistics."""
        self.stats.operations_count += 1
        if used_rust:
            self.stats.rust_operations += 1
        else:
            self.stats.python_fallback_operations += 1
    
    def segment_text(self, text: str, max_segment_chars: int = 500) -> List[Dict[str, Any]]:
        """Segment text into smaller chunks."""
        if self._text_processor:
            self._track_operation(True)
            segments = self._text_processor.segment_text(text, max_segment_chars)
            return [s.to_dict() for s in segments]
        
        self._track_operation(False)
        return self._fallback_segment(text, max_segment_chars)
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """Analyze text statistics."""
        if self._text_processor:
            self._track_operation(True)
            return self._text_processor.analyze_text(text).to_dict()
        
        self._track_operation(False)
        return self._fallback_analyze(text)
    
    def extract_keywords(self, text: str, max_keywords: int = 20) -> List[Dict[str, Any]]:
        """Extract keywords from text with TF-IDF scoring."""
        if self._text_processor:
            self._track_operation(True)
            keywords = self._text_processor.extract_keywords(text, max_keywords)
            return [k.to_dict() for k in keywords]
        
        self._track_operation(False)
        return self._fallback_keywords(text, max_keywords)
    
    def extract_keywords_batch(
        self,
        texts: List[str],
        max_keywords: int = 20
    ) -> List[List[Dict[str, Any]]]:
        """Extract keywords from multiple texts in parallel."""
        if self._text_processor:
            self._track_operation(True)
            results = self._text_processor.extract_keywords_parallel(texts, max_keywords)
            return [[k.to_dict() for k in kws] for kws in results]
        
        self._track_operation(False)
        return [self._fallback_keywords(t, max_keywords) for t in texts]
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        if self._text_processor:
            self._track_operation(True)
            return self._text_processor.clean_text(text)
        
        self._track_operation(False)
        return ' '.join(text.split())
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        if self._text_processor:
            self._track_operation(True)
            return self._text_processor.tokenize(text)
        
        self._track_operation(False)
        return text.split()
    
    def index_document(self, doc_id: str, content: str, metadata: Optional[Dict] = None):
        """Index a document for searching."""
        if self._search_engine:
            self._track_operation(True)
            from ..rust_core.python.transcriber_core import search
            doc = search.IndexedDocument(doc_id, content, metadata)
            self._search_engine.index_document(doc)
        else:
            self._track_operation(False)
            self._fallback_documents[doc_id] = {"content": content, "metadata": metadata or {}}
    
    def search(
        self,
        query: str,
        min_score: float = 0.0,
        max_results: int = 100,
    ) -> List[Dict[str, Any]]:
        """Search indexed documents."""
        if self._search_engine:
            self._track_operation(True)
            from ..rust_core.python.transcriber_core import search
            filter_obj = search.SearchFilter(False, False, False, min_score, max_results, 50)
            results = self._search_engine.search(query, filter_obj)
            return [r.to_dict() for r in results]
        
        self._track_operation(False)
        return self._fallback_search(query)
    
    def search_regex(
        self,
        pattern: str,
        max_results: int = 100,
    ) -> List[Dict[str, Any]]:
        """Search documents using regex pattern."""
        if self._search_engine:
            self._track_operation(True)
            from ..rust_core.python.transcriber_core import search
            filter_obj = search.SearchFilter(False, False, True, 0.0, max_results, 50)
            results = self._search_engine.search(pattern, filter_obj)
            return [r.to_dict() for r in results]
        
        self._track_operation(False)
        return []
    
    def multi_pattern_search(self, patterns: List[str]) -> List[Dict[str, Any]]:
        """Search for multiple patterns using Aho-Corasick algorithm."""
        if self._search_engine:
            self._track_operation(True)
            results = self._search_engine.multi_pattern_search(patterns)
            return [r.to_dict() for r in results]
        
        self._track_operation(False)
        return []
    
    def cache_get(self, key: str) -> Optional[str]:
        """Get value from cache."""
        if self._cache:
            self._track_operation(True)
            return self._cache.get(key)
        
        self._track_operation(False)
        return self._fallback_cache.get(key)
    
    def cache_set(self, key: str, value: str, ttl_seconds: Optional[int] = None):
        """Set value in cache."""
        if self._cache:
            self._track_operation(True)
            self._cache.set(key, value, ttl_seconds)
        else:
            self._track_operation(False)
            self._fallback_cache[key] = value
    
    def cache_delete(self, key: str) -> bool:
        """Delete key from cache."""
        if self._cache:
            self._track_operation(True)
            return self._cache.delete(key)
        
        self._track_operation(False)
        return self._fallback_cache.pop(key, None) is not None
    
    def cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if self._cache:
            self._track_operation(True)
            return self._cache.get_stats().to_dict()
        
        self._track_operation(False)
        return {
            "total_entries": len(self._fallback_cache),
            "total_size_bytes": sum(len(k) + len(v) for k, v in self._fallback_cache.items()),
            "hits": 0,
            "misses": 0,
            "hit_rate": 0.0
        }
    
    def hash_content(self, content: str, algorithm: str = "blake3") -> Dict[str, Any]:
        """Hash content using specified algorithm."""
        if self._hash_service:
            self._track_operation(True)
            return self._hash_service.hash_with_algorithm(content, algorithm).to_dict()
        
        self._track_operation(False)
        hash_hex = hashlib.sha256(content.encode()).hexdigest()
        return {
            "algorithm": "sha256",
            "hash_hex": hash_hex,
            "input_size": len(content),
            "computed_at": 0
        }
    
    def generate_content_id(self, content: str) -> str:
        """Generate unique content identifier."""
        if self._hash_service:
            self._track_operation(True)
            return self._hash_service.generate_content_id(content)
        
        self._track_operation(False)
        return f"cid_{hashlib.md5(content.encode()).hexdigest()[:16]}"
    
    def compare_similarity(
        self,
        text1: str,
        text2: str,
        algorithm: str = "jaro_winkler"
    ) -> Dict[str, Any]:
        """Compare two strings for similarity."""
        if self._similarity_engine:
            self._track_operation(True)
            return self._similarity_engine.compare(text1, text2, algorithm).to_dict()
        
        self._track_operation(False)
        return self._fallback_compare(text1, text2)
    
    def find_similar(
        self,
        query: str,
        candidates: List[str],
        threshold: Optional[float] = None,
        max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """Find similar strings from candidates."""
        if self._similarity_engine:
            self._track_operation(True)
            results = self._similarity_engine.find_similar(query, candidates, threshold, max_results)
            return [r.to_dict() for r in results]
        
        self._track_operation(False)
        return []
    
    def find_duplicates(
        self,
        texts: List[str],
        threshold: Optional[float] = None
    ) -> List[Tuple[str, str, float]]:
        """Find duplicate or near-duplicate texts."""
        if self._similarity_engine:
            self._track_operation(True)
            return self._similarity_engine.find_duplicates(texts, threshold)
        
        self._track_operation(False)
        return []
    
    def cluster_similar(
        self,
        texts: List[str],
        threshold: Optional[float] = None
    ) -> List[List[str]]:
        """Cluster similar texts together."""
        if self._similarity_engine:
            self._track_operation(True)
            return self._similarity_engine.cluster_similar(texts, threshold)
        
        self._track_operation(False)
        return [[t] for t in texts]
    
    def detect_language(self, text: str) -> Optional[Dict[str, Any]]:
        """Detect the language of text."""
        if self._language_detector:
            self._track_operation(True)
            result = self._language_detector.detect(text)
            return result.to_dict() if result else None
        
        self._track_operation(False)
        return {"language_code": "en", "language_name": "English", "confidence": 0.0, "is_reliable": False}
    
    def detect_languages_batch(self, texts: List[str]) -> List[Optional[Dict[str, Any]]]:
        """Detect languages for multiple texts."""
        if self._language_detector:
            self._track_operation(True)
            results = self._language_detector.detect_batch(texts)
            return [r.to_dict() if r else None for r in results]
        
        self._track_operation(False)
        return [self.detect_language(t) for t in texts]
    
    def stem_text(self, text: str, language_code: str = "en") -> str:
        """Stem all words in text."""
        if self._language_detector:
            self._track_operation(True)
            return self._language_detector.stem_text(text, language_code)
        
        self._track_operation(False)
        return text
    
    def process_batch(self, texts: List[str], operation: str = "identity") -> Dict[str, Any]:
        """Process multiple texts in parallel."""
        if self._batch_processor:
            self._track_operation(True)
            return self._batch_processor.process_texts(texts, operation).to_dict()
        
        self._track_operation(False)
        return self._fallback_batch_process(texts, operation)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get accelerator statistics."""
        stats = self.stats.to_dict()
        
        if self._cache:
            stats["cache_stats"] = self.cache_stats()
        
        if self._batch_processor:
            stats["batch_stats"] = self._batch_processor.get_stats().to_dict()
        
        return stats
    
    def _fallback_segment(self, content: str, max_chars: int) -> List[Dict]:
        segments = []
        sentences = content.split('. ')
        current = ""
        start = 0
        
        for sentence in sentences:
            if len(current) + len(sentence) > max_chars and current:
                segments.append({
                    "id": len(segments),
                    "text": current.strip(),
                    "start_char": start,
                    "end_char": start + len(current),
                    "word_count": len(current.split()),
                    "sentence_count": current.count('.') + 1
                })
                start += len(current)
                current = ""
            current += sentence + ". "
        
        if current.strip():
            segments.append({
                "id": len(segments),
                "text": current.strip(),
                "start_char": start,
                "end_char": start + len(current),
                "word_count": len(current.split()),
                "sentence_count": current.count('.') + 1
            })
        
        return segments
    
    def _fallback_analyze(self, content: str) -> Dict:
        words = content.split()
        return {
            "total_chars": len(content.replace(" ", "")),
            "total_words": len(words),
            "total_sentences": content.count('.') + content.count('!') + content.count('?') or 1,
            "total_paragraphs": content.count('\n\n') + 1,
            "avg_word_length": sum(len(w) for w in words) / max(len(words), 1),
            "avg_sentence_length": len(words) / max(content.count('.'), 1),
            "unique_words": len(set(w.lower() for w in words)),
            "reading_time_minutes": len(words) / 200
        }
    
    def _fallback_keywords(self, content: str, max_keywords: int) -> List[Dict]:
        words = [w.lower() for w in content.split() if len(w) >= self._min_word_length]
        counts = Counter(words)
        total = len(words)
        
        keywords = []
        for word, freq in counts.most_common(max_keywords):
            tf = freq / max(total, 1)
            idf = 1.0
            keywords.append({
                "word": word,
                "frequency": freq,
                "tf_idf": tf * idf,
                "relevance_score": tf * min(len(word) / 10, 1.0)
            })
        
        return keywords
    
    def _fallback_search(self, query: str) -> List[Dict]:
        results = []
        query_lower = query.lower()
        for doc_id, doc in self._fallback_documents.items():
            if query_lower in doc["content"].lower():
                results.append({
                    "document_id": doc_id,
                    "score": 1.0,
                    "matched_text": query,
                    "position": doc["content"].lower().find(query_lower),
                    "context": doc["content"][:100]
                })
        return results
    
    def _fallback_compare(self, text1: str, text2: str) -> Dict:
        s1, s2 = set(text1.lower()), set(text2.lower())
        intersection = len(s1 & s2)
        union = len(s1 | s2)
        score = intersection / union if union > 0 else 0.0
        return {
            "text1": text1,
            "text2": text2,
            "algorithm": "jaccard",
            "score": score,
            "distance": None,
            "is_similar": score >= self._similarity_threshold
        }
    
    def _fallback_batch_process(self, texts: List[str], operation: str) -> Dict:
        results = []
        for text in texts:
            if operation == "uppercase":
                result = text.upper()
            elif operation == "lowercase":
                result = text.lower()
            elif operation == "word_count":
                result = str(len(text.split()))
            else:
                result = text
            results.append({"data": text, "result": result, "status": "completed"})
        
        return {
            "batch_id": "fallback",
            "total_jobs": len(texts),
            "completed": len(texts),
            "failed": 0,
            "results": results,
            "total_time_ms": 0,
            "avg_time_per_job_ms": 0.0
        }


_accelerator_instance: Optional[RustAccelerator] = None


def get_rust_accelerator() -> RustAccelerator:
    """Get or create the singleton RustAccelerator instance."""
    global _accelerator_instance
    if _accelerator_instance is None:
        _accelerator_instance = RustAccelerator()
    return _accelerator_instance


def is_rust_available() -> bool:
    """Check if Rust extensions are available."""
    return RUST_AVAILABLE


__all__ = [
    "RustAccelerator",
    "AcceleratorStats",
    "get_rust_accelerator",
    "is_rust_available",
    "RUST_AVAILABLE",
]












