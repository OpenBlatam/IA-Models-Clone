"""
Transcriber Core - High Performance Rust Extensions for Social Video Transcriber AI

This module provides Python bindings to high-performance Rust implementations for:
- Text processing (segmentation, analysis, NLP)
- Search engine (regex, similarity, full-text)
- High-performance caching (LRU, TTL, concurrent)
- Batch processing (Rayon parallelization)
- Language detection and stemming
- Cryptography and hashing
- String similarity algorithms
"""

from typing import Any, Dict, List, Optional, Tuple, Callable
import json

try:
    from .transcriber_core import (
        text,
        search,
        cache,
        batch,
        crypto,
        similarity,
        language,
        utils,
        __version__,
        __author__,
    )
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    __version__ = "1.0.0"
    __author__ = "Social Video Transcriber AI Team"


class TextProcessor:
    """High-performance text processing with Rust backend."""
    
    def __init__(
        self,
        stop_words: Optional[List[str]] = None,
        min_word_length: int = 2
    ):
        if RUST_AVAILABLE:
            self._processor = text.TextProcessor(stop_words, min_word_length)
        else:
            self._processor = None
            self._stop_words = stop_words or []
            self._min_word_length = min_word_length
    
    def segment_text(self, content: str, max_segment_chars: int = 500) -> List[Dict[str, Any]]:
        """Segment text into smaller chunks."""
        if self._processor:
            segments = self._processor.segment_text(content, max_segment_chars)
            return [s.to_dict() for s in segments]
        return self._fallback_segment(content, max_segment_chars)
    
    def analyze_text(self, content: str) -> Dict[str, Any]:
        """Analyze text statistics."""
        if self._processor:
            return self._processor.analyze_text(content).to_dict()
        return self._fallback_analyze(content)
    
    def extract_keywords(self, content: str, max_keywords: int = 20) -> List[Dict[str, Any]]:
        """Extract keywords from text."""
        if self._processor:
            keywords = self._processor.extract_keywords(content, max_keywords)
            return [k.to_dict() for k in keywords]
        return self._fallback_keywords(content, max_keywords)
    
    def extract_keywords_parallel(
        self, 
        texts: List[str], 
        max_keywords: int = 20
    ) -> List[List[Dict[str, Any]]]:
        """Extract keywords from multiple texts in parallel."""
        if self._processor:
            results = self._processor.extract_keywords_parallel(texts, max_keywords)
            return [[k.to_dict() for k in keywords] for keywords in results]
        return [self._fallback_keywords(t, max_keywords) for t in texts]
    
    def clean_text(self, content: str) -> str:
        """Clean and normalize text."""
        if self._processor:
            return self._processor.clean_text(content)
        return ' '.join(content.split())
    
    def tokenize(self, content: str) -> List[str]:
        """Tokenize text into words."""
        if self._processor:
            return self._processor.tokenize(content)
        return content.split()
    
    def split_sentences(self, content: str) -> List[str]:
        """Split text into sentences."""
        if self._processor:
            return self._processor.split_sentences(content)
        import re
        return [s.strip() for s in re.split(r'[.!?]', content) if s.strip()]
    
    def normalize(self, content: str) -> str:
        """Normalize text (lowercase, alphanumeric only)."""
        if self._processor:
            return self._processor.normalize(content)
        return ''.join(c for c in content.lower() if c.isalnum() or c.isspace())
    
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
            "total_sentences": content.count('.') + content.count('!') + content.count('?'),
            "total_paragraphs": content.count('\n\n') + 1,
            "avg_word_length": sum(len(w) for w in words) / max(len(words), 1),
            "avg_sentence_length": len(words) / max(content.count('.'), 1),
            "unique_words": len(set(w.lower() for w in words)),
            "reading_time_minutes": len(words) / 200
        }
    
    def _fallback_keywords(self, content: str, max_keywords: int) -> List[Dict]:
        from collections import Counter
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


class SearchEngine:
    """High-performance search engine with Rust backend."""
    
    def __init__(self):
        if RUST_AVAILABLE:
            self._engine = search.SearchEngine()
        else:
            self._engine = None
            self._documents = {}
    
    def index_document(self, doc_id: str, content: str, metadata: Optional[Dict] = None):
        """Index a document for searching."""
        if self._engine:
            doc = search.IndexedDocument(doc_id, content, metadata)
            self._engine.index_document(doc)
        else:
            self._documents[doc_id] = {"content": content, "metadata": metadata or {}}
    
    def index_documents(self, documents: List[Dict[str, Any]]):
        """Index multiple documents."""
        for doc in documents:
            self.index_document(doc["id"], doc["content"], doc.get("metadata"))
    
    def search(
        self,
        query: str,
        case_sensitive: bool = False,
        whole_word: bool = False,
        regex_enabled: bool = False,
        min_score: float = 0.0,
        max_results: int = 100,
        context_size: int = 50
    ) -> List[Dict[str, Any]]:
        """Search indexed documents."""
        if self._engine:
            filter_obj = search.SearchFilter(
                case_sensitive, whole_word, regex_enabled,
                min_score, max_results, context_size
            )
            results = self._engine.search(query, filter_obj)
            return [r.to_dict() for r in results]
        return self._fallback_search(query)
    
    def multi_pattern_search(self, patterns: List[str]) -> List[Dict[str, Any]]:
        """Search for multiple patterns using Aho-Corasick algorithm."""
        if self._engine:
            results = self._engine.multi_pattern_search(patterns)
            return [r.to_dict() for r in results]
        return []
    
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get a document by ID."""
        if self._engine:
            doc = self._engine.get_document(doc_id)
            return {"id": doc.id, "content": doc.content, "metadata": doc.metadata} if doc else None
        return self._documents.get(doc_id)
    
    def remove_document(self, doc_id: str) -> bool:
        """Remove a document from the index."""
        if self._engine:
            return self._engine.remove_document(doc_id)
        return self._documents.pop(doc_id, None) is not None
    
    def document_count(self) -> int:
        """Get the number of indexed documents."""
        if self._engine:
            return self._engine.document_count()
        return len(self._documents)
    
    def clear(self):
        """Clear all indexed documents."""
        if self._engine:
            self._engine.clear()
        else:
            self._documents.clear()
    
    def _fallback_search(self, query: str) -> List[Dict]:
        results = []
        query_lower = query.lower()
        for doc_id, doc in self._documents.items():
            if query_lower in doc["content"].lower():
                results.append({
                    "document_id": doc_id,
                    "score": 1.0,
                    "matched_text": query,
                    "position": doc["content"].lower().find(query_lower),
                    "context": doc["content"][:100]
                })
        return results


class CacheService:
    """High-performance caching with Rust backend."""
    
    def __init__(
        self,
        max_size: int = 10000,
        default_ttl_seconds: Optional[int] = None,
        use_lru: bool = True
    ):
        if RUST_AVAILABLE:
            self._cache = cache.CacheService(max_size, default_ttl_seconds, use_lru)
        else:
            self._cache = None
            self._data = {}
            self._max_size = max_size
    
    def get(self, key: str) -> Optional[str]:
        """Get a value from cache."""
        if self._cache:
            return self._cache.get(key)
        return self._data.get(key)
    
    def set(self, key: str, value: str, ttl_seconds: Optional[int] = None):
        """Set a value in cache."""
        if self._cache:
            self._cache.set(key, value, ttl_seconds)
        else:
            self._data[key] = value
    
    def delete(self, key: str) -> bool:
        """Delete a key from cache."""
        if self._cache:
            return self._cache.delete(key)
        return self._data.pop(key, None) is not None
    
    def contains(self, key: str) -> bool:
        """Check if key exists in cache."""
        if self._cache:
            return self._cache.contains(key)
        return key in self._data
    
    def clear(self):
        """Clear all cache entries."""
        if self._cache:
            self._cache.clear()
        else:
            self._data.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if self._cache:
            return self._cache.get_stats().to_dict()
        return {
            "total_entries": len(self._data),
            "total_size_bytes": sum(len(k) + len(v) for k, v in self._data.items()),
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "hit_rate": 0.0
        }
    
    def cleanup_expired(self) -> int:
        """Remove expired entries."""
        if self._cache:
            return self._cache.cleanup_expired()
        return 0


class BatchProcessor:
    """High-performance batch processing with Rust backend."""
    
    def __init__(self, max_workers: Optional[int] = None):
        if RUST_AVAILABLE:
            self._processor = batch.BatchProcessor(max_workers)
        else:
            self._processor = None
    
    def process_texts(self, texts: List[str], operation: str = "identity") -> Dict[str, Any]:
        """Process multiple texts in parallel."""
        if self._processor:
            result = self._processor.process_texts(texts, operation)
            return result.to_dict()
        return self._fallback_process(texts, operation)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get batch processing statistics."""
        if self._processor:
            return self._processor.get_stats().to_dict()
        return {
            "total_batches": 0,
            "total_jobs_processed": 0,
            "total_jobs_failed": 0,
            "avg_batch_size": 0.0,
            "avg_processing_time_ms": 0.0,
            "throughput_per_second": 0.0
        }
    
    def reset_stats(self):
        """Reset batch processing statistics."""
        if self._processor:
            self._processor.reset_stats()
    
    def _fallback_process(self, texts: List[str], operation: str) -> Dict:
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


class HashService:
    """High-performance hashing with Rust backend."""
    
    def __init__(self, algorithm: str = "blake3"):
        if RUST_AVAILABLE:
            self._service = crypto.HashService(algorithm)
        else:
            self._service = None
            self._algorithm = algorithm
    
    def hash(self, data: str) -> Dict[str, Any]:
        """Hash data using the default algorithm."""
        if self._service:
            return self._service.hash(data).to_dict()
        return self._fallback_hash(data)
    
    def hash_with_algorithm(self, data: str, algorithm: str) -> Dict[str, Any]:
        """Hash data using a specific algorithm."""
        if self._service:
            return self._service.hash_with_algorithm(data, algorithm).to_dict()
        return self._fallback_hash(data)
    
    def verify(self, data: str, expected_hash: str) -> bool:
        """Verify data against an expected hash."""
        if self._service:
            return self._service.verify(data, expected_hash)
        return self._fallback_hash(data)["hash_hex"] == expected_hash
    
    def generate_content_id(self, content: str) -> str:
        """Generate a unique content ID."""
        if self._service:
            return self._service.generate_content_id(content)
        import hashlib
        return f"cid_{hashlib.md5(content.encode()).hexdigest()[:16]}"
    
    def generate_cache_key(self, parts: List[str]) -> str:
        """Generate a cache key from multiple parts."""
        if self._service:
            return self._service.generate_cache_key(parts)
        import hashlib
        return hashlib.md5("::".join(parts).encode()).hexdigest()
    
    def _fallback_hash(self, data: str) -> Dict:
        import hashlib
        hash_hex = hashlib.sha256(data.encode()).hexdigest()
        return {
            "algorithm": "sha256",
            "hash_hex": hash_hex,
            "input_size": len(data),
            "computed_at": 0
        }


class SimilarityEngine:
    """High-performance string similarity with Rust backend."""
    
    def __init__(self, default_threshold: float = 0.8, case_sensitive: bool = False):
        if RUST_AVAILABLE:
            self._engine = similarity.SimilarityEngine(default_threshold, case_sensitive)
        else:
            self._engine = None
            self._threshold = default_threshold
    
    def compare(self, text1: str, text2: str, algorithm: str = "jaro_winkler") -> Dict[str, Any]:
        """Compare two strings for similarity."""
        if self._engine:
            return self._engine.compare(text1, text2, algorithm).to_dict()
        return self._fallback_compare(text1, text2)
    
    def find_similar(
        self,
        query: str,
        candidates: List[str],
        threshold: Optional[float] = None,
        max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """Find similar strings from a list of candidates."""
        if self._engine:
            results = self._engine.find_similar(query, candidates, threshold, max_results)
            return [r.to_dict() for r in results]
        return []
    
    def find_duplicates(
        self,
        texts: List[str],
        threshold: Optional[float] = None
    ) -> List[Tuple[str, str, float]]:
        """Find duplicate or near-duplicate texts."""
        if self._engine:
            return self._engine.find_duplicates(texts, threshold)
        return []
    
    def cluster_similar(
        self,
        texts: List[str],
        threshold: Optional[float] = None
    ) -> List[List[str]]:
        """Cluster similar texts together."""
        if self._engine:
            return self._engine.cluster_similar(texts, threshold)
        return [[t] for t in texts]
    
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
            "is_similar": score >= self._threshold
        }


class LanguageDetector:
    """High-performance language detection with Rust backend."""
    
    def __init__(self, min_confidence: float = 0.5, min_text_length: int = 20):
        if RUST_AVAILABLE:
            self._detector = language.LanguageDetector(min_confidence, min_text_length)
        else:
            self._detector = None
    
    def detect(self, text: str) -> Optional[Dict[str, Any]]:
        """Detect the language of text."""
        if self._detector:
            result = self._detector.detect(text)
            return result.to_dict() if result else None
        return {"language_code": "en", "language_name": "English", "confidence": 0.0, "is_reliable": False}
    
    def detect_batch(self, texts: List[str]) -> List[Optional[Dict[str, Any]]]:
        """Detect languages for multiple texts."""
        if self._detector:
            results = self._detector.detect_batch(texts)
            return [r.to_dict() if r else None for r in results]
        return [self.detect(t) for t in texts]
    
    def is_language(self, text: str, language_code: str) -> bool:
        """Check if text is in a specific language."""
        if self._detector:
            return self._detector.is_language(text, language_code)
        return False
    
    def stem_word(self, word: str, language_code: str) -> str:
        """Stem a word in a specific language."""
        if self._detector:
            return self._detector.stem_word(word, language_code)
        return word
    
    def stem_text(self, text: str, language_code: str) -> str:
        """Stem all words in text."""
        if self._detector:
            return self._detector.stem_text(text, language_code)
        return text
    
    def get_supported_languages(self) -> List[Tuple[str, str]]:
        """Get list of supported languages."""
        if self._detector:
            return self._detector.get_supported_languages()
        return [("en", "English"), ("es", "Spanish"), ("fr", "French")]


class Timer:
    """High-precision timer with checkpoints."""
    
    def __init__(self):
        if RUST_AVAILABLE:
            self._timer = utils.Timer()
        else:
            self._timer = None
            import time
            self._start = time.perf_counter_ns()
            self._checkpoints = []
    
    def checkpoint(self, name: str):
        """Add a named checkpoint."""
        if self._timer:
            self._timer.checkpoint(name)
        else:
            import time
            self._checkpoints.append((name, time.perf_counter_ns() - self._start))
    
    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        if self._timer:
            return self._timer.elapsed_ms()
        import time
        return (time.perf_counter_ns() - self._start) / 1_000_000
    
    def elapsed_us(self) -> float:
        """Get elapsed time in microseconds."""
        if self._timer:
            return self._timer.elapsed_us()
        import time
        return (time.perf_counter_ns() - self._start) / 1_000
    
    def reset(self):
        """Reset the timer."""
        if self._timer:
            self._timer.reset()
        else:
            import time
            self._start = time.perf_counter_ns()
            self._checkpoints.clear()
    
    def get_checkpoints(self) -> List[Tuple[str, float]]:
        """Get all checkpoints with times in ms."""
        if self._timer:
            return self._timer.get_checkpoints()
        return [(name, ns / 1_000_000) for name, ns in self._checkpoints]


class DateUtils:
    """Date and time utilities."""
    
    def __init__(self):
        if RUST_AVAILABLE:
            self._utils = utils.DateUtils()
        else:
            self._utils = None
    
    def now_iso(self) -> str:
        """Get current time in ISO format."""
        if self._utils:
            return self._utils.now_iso()
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).isoformat()
    
    def now_unix(self) -> int:
        """Get current Unix timestamp."""
        if self._utils:
            return self._utils.now_unix()
        import time
        return int(time.time())
    
    def format_duration(self, seconds: int) -> str:
        """Format duration in human readable format."""
        if self._utils:
            return self._utils.format_duration(seconds)
        if seconds < 60:
            return f"{seconds}s"
        if seconds < 3600:
            return f"{seconds // 60}m {seconds % 60}s"
        if seconds < 86400:
            return f"{seconds // 3600}h {(seconds % 3600) // 60}m"
        return f"{seconds // 86400}d {(seconds % 86400) // 3600}h"
    
    def format_timestamp(self, seconds: float) -> str:
        """Format seconds to timestamp HH:MM:SS.mmm."""
        if self._utils:
            return self._utils.format_timestamp(seconds)
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        ms = int((seconds - int(seconds)) * 1000)
        if hours > 0:
            return f"{hours:02}:{mins:02}:{secs:02}.{ms:03}"
        return f"{mins:02}:{secs:02}.{ms:03}"


class StringUtils:
    """String manipulation utilities."""
    
    def __init__(self):
        if RUST_AVAILABLE:
            self._utils = utils.StringUtils()
        else:
            self._utils = None
    
    def truncate(self, text: str, max_length: int, suffix: str = "...") -> str:
        """Truncate text to max length."""
        if self._utils:
            return self._utils.truncate(text, max_length, suffix)
        if len(text) <= max_length:
            return text
        return text[:max_length - len(suffix)] + suffix
    
    def slugify(self, text: str) -> str:
        """Convert text to URL-safe slug."""
        if self._utils:
            return self._utils.slugify(text)
        import re
        slug = text.lower().replace(" ", "-")
        return re.sub(r'[^a-z0-9-]', '', slug)
    
    def to_camel_case(self, text: str) -> str:
        """Convert to camelCase."""
        if self._utils:
            return self._utils.to_camel_case(text)
        parts = text.replace("-", "_").split("_")
        return parts[0].lower() + "".join(p.capitalize() for p in parts[1:])
    
    def to_snake_case(self, text: str) -> str:
        """Convert to snake_case."""
        if self._utils:
            return self._utils.to_snake_case(text)
        import re
        return re.sub(r'(?<!^)(?=[A-Z])', '_', text).lower()
    
    def extract_hashtags(self, text: str) -> List[str]:
        """Extract hashtags from text."""
        if self._utils:
            return self._utils.extract_hashtags(text)
        return [w for w in text.split() if w.startswith('#') and len(w) > 1]
    
    def extract_mentions(self, text: str) -> List[str]:
        """Extract @mentions from text."""
        if self._utils:
            return self._utils.extract_mentions(text)
        return [w for w in text.split() if w.startswith('@') and len(w) > 1]
    
    def sanitize_filename(self, text: str) -> str:
        """Sanitize text for use as filename."""
        if self._utils:
            return self._utils.sanitize_filename(text)
        import re
        return re.sub(r'[^\w\-\.]', '_', text)


class SubtitleUtils:
    """Subtitle format utilities."""
    
    def __init__(self):
        if RUST_AVAILABLE:
            self._utils = utils.SubtitleUtils()
        else:
            self._utils = None
    
    def create_entry(self, index: int, start: float, end: float, text: str):
        """Create a subtitle entry."""
        if self._utils:
            return self._utils.create_entry(index, start, end, text)
        return {"index": index, "start": start, "end": end, "text": text}
    
    def entries_to_srt(self, entries: List[Dict]) -> str:
        """Convert entries to SRT format."""
        if self._utils and hasattr(entries[0], 'format_srt'):
            return self._utils.entries_to_srt(entries)
        
        def format_time(s):
            h, m = divmod(int(s), 3600)
            m, sec = divmod(m, 60)
            ms = int((s - int(s)) * 1000)
            return f"{h:02}:{m:02}:{sec:02},{ms:03}"
        
        lines = []
        for e in entries:
            lines.append(str(e['index']))
            lines.append(f"{format_time(e['start'])} --> {format_time(e['end'])}")
            lines.append(e['text'])
            lines.append("")
        return "\n".join(lines)
    
    def entries_to_vtt(self, entries: List[Dict]) -> str:
        """Convert entries to VTT format."""
        if self._utils and hasattr(entries[0], 'format_vtt'):
            return self._utils.entries_to_vtt(entries)
        
        def format_time(s):
            h, m = divmod(int(s), 3600)
            m, sec = divmod(m, 60)
            ms = int((s - int(s)) * 1000)
            return f"{h:02}:{m:02}:{sec:02}.{ms:03}"
        
        lines = ["WEBVTT", ""]
        for e in entries:
            lines.append(f"{format_time(e['start'])} --> {format_time(e['end'])}")
            lines.append(e['text'])
            lines.append("")
        return "\n".join(lines)


def is_rust_available() -> bool:
    """Check if Rust extensions are available."""
    return RUST_AVAILABLE


def get_version() -> str:
    """Get the library version."""
    return __version__


def get_system_info() -> Dict[str, str]:
    """Get system information."""
    if RUST_AVAILABLE:
        return utils.get_system_info()
    return {
        "rust_version": "not available",
        "module_name": "transcriber_core (Python fallback)"
    }


def create_timer() -> Timer:
    """Create a new high-precision timer."""
    return Timer()


__all__ = [
    "TextProcessor",
    "SearchEngine",
    "CacheService",
    "BatchProcessor",
    "HashService",
    "SimilarityEngine",
    "LanguageDetector",
    "Timer",
    "DateUtils",
    "StringUtils",
    "SubtitleUtils",
    "is_rust_available",
    "get_version",
    "get_system_info",
    "create_timer",
    "RUST_AVAILABLE",
    "__version__",
    "__author__",
]

