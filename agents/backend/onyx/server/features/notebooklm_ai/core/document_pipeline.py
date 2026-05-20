from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import asyncio
import time
import logging
import json
import hashlib
from typing import Dict, Any, Optional, List, Union, AsyncGenerator
from dataclasses import dataclass, field
from pathlib import Path
import aiofiles
import aiohttp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
from collections import defaultdict, deque
    import pypdf2
    from docx import Document
    import markdown
    from bs4 import BeautifulSoup
    from prometheus_client import Counter, Histogram, Gauge, Summary
        import re
        import re
        from collections import Counter
        import re
from typing import Any, List, Dict, Optional
#!/usr/bin/env python3
"""
Ultra-Optimized Document Pipeline v4.0
📄 Advanced document processing with enhanced speed and quality
"""


# Document processing libraries
try:
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False

try:
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

try:
    MARKDOWN_AVAILABLE = True
except ImportError:
    MARKDOWN_AVAILABLE = False

try:
    BEAUTIFULSOUP_AVAILABLE = True
except ImportError:
    BEAUTIFULSOUP_AVAILABLE = False

# Performance monitoring
try:
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
if PROMETHEUS_AVAILABLE:
    PIPELINE_REQUESTS = Counter('pipeline_requests_total', 'Document pipeline requests')
    PIPELINE_LATENCY = Histogram('pipeline_latency_seconds', 'Document pipeline latency')
    PIPELINE_DOCUMENTS_PROCESSED = Counter('pipeline_documents_total', 'Documents processed')
    PIPELINE_ERRORS = Counter('pipeline_errors_total', 'Pipeline errors')

@dataclass
class PipelineConfig:
    """Ultra-optimized pipeline configuration."""
    # Processing stages
    enable_document_intelligence: bool = True
    enable_citation_management: bool = True
    enable_nlp_analysis: bool = True
    enable_ml_integration: bool = True
    enable_performance_optimization: bool = True
    enable_ocr: bool = True
    enable_sentiment_analysis: bool = True
    enable_keyword_extraction: bool = True
    enable_topic_modeling: bool = True
    enable_entity_recognition: bool = True
    enable_summarization: bool = True
    enable_citation_generation: bool = True
    enable_insight_generation: bool = True
    
    # Performance settings
    batch_size: int = 64
    max_workers: int = 16
    max_processes: int = 8
    enable_parallel_processing: bool = True
    enable_streaming: bool = True
    
    # Output settings
    output_format: str = "json"
    include_metadata: bool = True
    include_metrics: bool = True
    include_insights: bool = True
    
    # Caching
    enable_caching: bool = True
    cache_ttl: int = 7200
    cache_size: int = 10000
    
    # Quality settings
    enable_quality_checks: bool = True
    min_content_length: int = 10
    enable_duplicate_detection: bool = True
    enable_content_validation: bool = True

@dataclass
class DocumentMetadata:
    """Enhanced document metadata."""
    file_path: str
    file_size: int
    file_type: str
    created_time: float
    modified_time: float
    processing_time: float
    content_length: int
    word_count: int
    sentence_count: int
    paragraph_count: int
    language: str = "en"
    confidence: float = 1.0
    quality_score: float = 1.0

@dataclass
class ProcessingResult:
    """Enhanced processing result."""
    success: bool
    content: str
    metadata: DocumentMetadata
    insights: Dict[str, Any]
    citations: List[Dict[str, Any]]
    nlp_analysis: Dict[str, Any]
    quality_metrics: Dict[str, Any]
    processing_time: float
    cache_hit: bool = False
    error: Optional[str] = None

class UltraDocumentProcessor:
    """Ultra-fast document processor with enhanced capabilities."""
    
    def __init__(self, config: PipelineConfig):
        
    """__init__ function."""
self.config = config
        self.stats = defaultdict(int)
        self._cache = {}
        self._lock = threading.RLock()
        self._thread_pool = ThreadPoolExecutor(max_workers=config.max_workers)
        self._process_pool = ProcessPoolExecutor(max_workers=config.max_processes)
    
    async def process_document(self, file_path: str) -> ProcessingResult:
        """Process single document with ultra optimization."""
        start_time = time.perf_counter()
        
        try:
            # Check cache
            cache_key = self._generate_cache_key(file_path)
            if self.config.enable_caching and cache_key in self._cache:
                cached_result = self._cache[cache_key]
                if time.time() - cached_result["timestamp"] < self.config.cache_ttl:
                    self.stats["cache_hits"] += 1
                    return cached_result["result"]
            
            # Get file metadata
            metadata = await self._get_file_metadata(file_path)
            
            # Extract content
            content = await self._extract_content(file_path, metadata.file_type)
            
            # Validate content
            if not self._validate_content(content):
                raise ValueError("Content validation failed")
            
            # Process content
            insights = await self._generate_insights(content)
            citations = await self._extract_citations(content)
            nlp_analysis = await self._analyze_nlp(content)
            quality_metrics = await self._calculate_quality_metrics(content, metadata)
            
            # Create result
            processing_time = time.perf_counter() - start_time
            metadata.processing_time = processing_time
            
            result = ProcessingResult(
                success=True,
                content=content,
                metadata=metadata,
                insights=insights,
                citations=citations,
                nlp_analysis=nlp_analysis,
                quality_metrics=quality_metrics,
                processing_time=processing_time
            )
            
            # Cache result
            if self.config.enable_caching:
                with self._lock:
                    if len(self._cache) >= self.config.cache_size:
                        self._cleanup_cache()
                    self._cache[cache_key] = {
                        "result": result,
                        "timestamp": time.time()
                    }
            
            self.stats["documents_processed"] += 1
            if PROMETHEUS_AVAILABLE:
                PIPELINE_DOCUMENTS_PROCESSED.inc()
                PIPELINE_LATENCY.observe(processing_time)
            
            return result
            
        except Exception as e:
            processing_time = time.perf_counter() - start_time
            logger.error(f"Document processing failed: {e}", file_path=file_path)
            
            self.stats["errors"] += 1
            if PROMETHEUS_AVAILABLE:
                PIPELINE_ERRORS.inc()
            
            return ProcessingResult(
                success=False,
                content="",
                metadata=DocumentMetadata(file_path=file_path, file_size=0, file_type="unknown",
                                        created_time=0, modified_time=0, processing_time=processing_time,
                                        content_length=0, word_count=0, sentence_count=0, paragraph_count=0),
                insights={},
                citations=[],
                nlp_analysis={},
                quality_metrics={},
                processing_time=processing_time,
                error=str(e)
            )
    
    async def process_documents_batch(self, file_paths: List[str]) -> List[ProcessingResult]:
        """Process multiple documents in parallel."""
        if not file_paths:
            return []
        
        # Process in batches
        batch_size = min(self.config.batch_size, len(file_paths))
        results = []
        
        for i in range(0, len(file_paths), batch_size):
            batch = file_paths[i:i + batch_size]
            
            if self.config.enable_parallel_processing:
                # Process batch in parallel
                tasks = [self.process_document(path) for path in batch]
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for result in batch_results:
                    if isinstance(result, Exception):
                        logger.error(f"Batch processing error: {result}")
                        results.append(ProcessingResult(
                            success=False,
                            content="",
                            metadata=DocumentMetadata(file_path="", file_size=0, file_type="unknown",
                                                    created_time=0, modified_time=0, processing_time=0,
                                                    content_length=0, word_count=0, sentence_count=0, paragraph_count=0),
                            insights={},
                            citations=[],
                            nlp_analysis={},
                            quality_metrics={},
                            processing_time=0,
                            error=str(result)
                        ))
                    else:
                        results.append(result)
            else:
                # Process sequentially
                for path in batch:
                    result = await self.process_document(path)
                    results.append(result)
        
        return results
    
    async def _get_file_metadata(self, file_path: str) -> DocumentMetadata:
        """Get enhanced file metadata."""
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        stat = path.stat()
        
        return DocumentMetadata(
            file_path=str(path),
            file_size=stat.st_size,
            file_type=path.suffix.lower(),
            created_time=stat.st_ctime,
            modified_time=stat.st_mtime,
            processing_time=0,
            content_length=0,
            word_count=0,
            sentence_count=0,
            paragraph_count=0
        )
    
    async def _extract_content(self, file_path: str, file_type: str) -> str:
        """Extract content from various file types."""
        try:
            if file_type == ".pdf" and PYPDF2_AVAILABLE:
                return await self._extract_pdf_content(file_path)
            elif file_type == ".docx" and DOCX_AVAILABLE:
                return await self._extract_docx_content(file_path)
            elif file_type == ".md" and MARKDOWN_AVAILABLE:
                return await self._extract_markdown_content(file_path)
            elif file_type in [".html", ".htm"] and BEAUTIFULSOUP_AVAILABLE:
                return await self._extract_html_content(file_path)
            else:
                # Fallback to text extraction
                return await self._extract_text_content(file_path)
                
        except Exception as e:
            logger.error(f"Content extraction failed: {e}", file_path=file_path, file_type=file_type)
            raise
    
    async def _extract_pdf_content(self, file_path: str) -> str:
        """Extract content from PDF with enhanced processing."""
        loop = asyncio.get_event_loop()
        
        def extract_pdf():
            
    """extract_pdf function."""
content = []
            with open(file_path, 'rb') as file:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                reader = pypdf2.PdfReader(file)
                for page in reader.pages:
                    text = page.extract_text()
                    if text.strip():
                        content.append(text)
            return '\n'.join(content)
        
        return await loop.run_in_executor(self._thread_pool, extract_pdf)
    
    async def _extract_docx_content(self, file_path: str) -> str:
        """Extract content from DOCX with enhanced processing."""
        loop = asyncio.get_event_loop()
        
        def extract_docx():
            
    """extract_docx function."""
doc = Document(file_path)
            content = []
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    content.append(paragraph.text)
            return '\n'.join(content)
        
        return await loop.run_in_executor(self._thread_pool, extract_docx)
    
    async def _extract_markdown_content(self, file_path: str) -> str:
        """Extract content from Markdown with enhanced processing."""
        async with aiofiles.open(file_path, 'r', encoding='utf-8') as file:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            content = await file.read()
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        
        # Convert markdown to HTML then extract text
        html = markdown.markdown(content)
        
        if BEAUTIFULSOUP_AVAILABLE:
            soup = BeautifulSoup(html, 'html.parser')
            return soup.get_text()
        else:
            return content
    
    async def _extract_html_content(self, file_path: str) -> str:
        """Extract content from HTML with enhanced processing."""
        async with aiofiles.open(file_path, 'r', encoding='utf-8') as file:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            content = await file.read()
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        
        soup = BeautifulSoup(content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        return soup.get_text()
    
    async def _extract_text_content(self, file_path: str) -> str:
        """Extract content from text files."""
        async with aiofiles.open(file_path, 'r', encoding='utf-8') as file:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            return await file.read()
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
    
    def _validate_content(self, content: str) -> bool:
        """Validate extracted content."""
        if not content or len(content.strip()) < self.config.min_content_length:
            return False
        
        if self.config.enable_content_validation:
            # Check for common issues
            if content.count('\n') > len(content) * 0.8:  # Too many newlines
                return False
            
            if len(set(content)) < 10:  # Too few unique characters
                return False
        
        return True
    
    async def _generate_insights(self, content: str) -> Dict[str, Any]:
        """Generate insights from content."""
        if not self.config.enable_insight_generation:
            return {}
        
        insights = {
            "readability_score": self._calculate_readability(content),
            "complexity_score": self._calculate_complexity(content),
            "topic_diversity": self._calculate_topic_diversity(content),
            "content_structure": self._analyze_structure(content)
        }
        
        return insights
    
    async def _extract_citations(self, content: str) -> List[Dict[str, Any]]:
        """Extract citations from content."""
        if not self.config.enable_citation_generation:
            return []
        
        # Simple citation extraction (would be enhanced with ML)
        citations = []
        
        # Look for common citation patterns
        patterns = [
            r'\(([^)]+)\)',  # Parenthetical citations
            r'\[([^\]]+)\]',  # Bracket citations
            r'([A-Z][a-z]+ et al\.)',  # Author et al.
            r'([A-Z][a-z]+ \d{4})',  # Author Year
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                citations.append({
                    "text": match,
                    "type": "extracted",
                    "confidence": 0.7
                })
        
        return citations[:50]  # Limit to 50 citations
    
    async def _analyze_nlp(self, content: str) -> Dict[str, Any]:
        """Analyze content with NLP."""
        if not self.config.enable_nlp_analysis:
            return {}
        
        analysis = {
            "sentiment": self._analyze_sentiment(content),
            "keywords": self._extract_keywords(content),
            "entities": self._extract_entities(content),
            "topics": self._extract_topics(content),
            "summary": self._generate_summary(content)
        }
        
        return analysis
    
    def _analyze_sentiment(self, content: str) -> Dict[str, Any]:
        """Analyze sentiment of content."""
        if not self.config.enable_sentiment_analysis:
            return {"sentiment": "neutral", "confidence": 0.5}
        
        # Simple sentiment analysis (would be enhanced with ML)
        positive_words = ["good", "great", "excellent", "amazing", "wonderful", "best"]
        negative_words = ["bad", "terrible", "awful", "worst", "horrible", "disappointing"]
        
        words = content.lower().split()
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        
        total_words = len(words)
        if total_words == 0:
            return {"sentiment": "neutral", "confidence": 0.5}
        
        positive_ratio = positive_count / total_words
        negative_ratio = negative_count / total_words
        
        if positive_ratio > negative_ratio:
            sentiment = "positive"
            confidence = positive_ratio
        elif negative_ratio > positive_ratio:
            sentiment = "negative"
            confidence = negative_ratio
        else:
            sentiment = "neutral"
            confidence = 0.5
        
        return {
            "sentiment": sentiment,
            "confidence": min(confidence * 10, 1.0),
            "positive_ratio": positive_ratio,
            "negative_ratio": negative_ratio
        }
    
    def _extract_keywords(self, content: str) -> List[str]:
        """Extract keywords from content."""
        if not self.config.enable_keyword_extraction:
            return []
        
        # Simple keyword extraction (would be enhanced with ML)
        
        # Remove common words
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
        
        words = re.findall(r'\b[a-zA-Z]{3,}\b', content.lower())
        words = [word for word in words if word not in stop_words]
        
        # Get most common words
        word_counts = Counter(words)
        keywords = [word for word, count in word_counts.most_common(20)]
        
        return keywords
    
    def _extract_entities(self, content: str) -> List[Dict[str, Any]]:
        """Extract entities from content."""
        if not self.config.enable_entity_recognition:
            return []
        
        # Simple entity extraction (would be enhanced with ML)
        
        entities = []
        
        # Extract names (capitalized words)
        names = re.findall(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', content)
        for name in names[:10]:  # Limit to 10 names
            entities.append({
                "text": name,
                "type": "PERSON",
                "confidence": 0.8
            })
        
        # Extract organizations
        org_patterns = [r'\b[A-Z][a-z]+ (Inc|Corp|LLC|Ltd|Company)\b']
        for pattern in org_patterns:
            orgs = re.findall(pattern, content)
            for org in orgs[:5]:  # Limit to 5 organizations
                entities.append({
                    "text": org,
                    "type": "ORGANIZATION",
                    "confidence": 0.7
                })
        
        return entities
    
    def _extract_topics(self, content: str) -> List[Dict[str, Any]]:
        """Extract topics from content."""
        if not self.config.enable_topic_modeling:
            return []
        
        # Simple topic extraction (would be enhanced with ML)
        topics = []
        
        # Common topic keywords
        topic_keywords = {
            "technology": ["computer", "software", "hardware", "digital", "tech"],
            "science": ["research", "study", "experiment", "analysis", "data"],
            "business": ["company", "market", "industry", "profit", "strategy"],
            "health": ["medical", "health", "treatment", "patient", "disease"],
            "education": ["learning", "teaching", "student", "education", "school"]
        }
        
        content_lower = content.lower()
        for topic, keywords in topic_keywords.items():
            score = sum(1 for keyword in keywords if keyword in content_lower)
            if score > 0:
                topics.append({
                    "topic": topic,
                    "score": score / len(keywords),
                    "keywords": [k for k in keywords if k in content_lower]
                })
        
        return sorted(topics, key=lambda x: x["score"], reverse=True)[:5]
    
    def _generate_summary(self, content: str) -> str:
        """Generate summary of content."""
        if not self.config.enable_summarization:
            return ""
        
        # Simple summarization (would be enhanced with ML)
        sentences = content.split('.')
        if len(sentences) <= 3:
            return content
        
        # Take first few sentences as summary
        summary_sentences = sentences[:3]
        return '. '.join(summary_sentences) + '.'
    
    async def _calculate_quality_metrics(self, content: str, metadata: DocumentMetadata) -> Dict[str, Any]:
        """Calculate quality metrics for content."""
        metrics = {
            "content_length": len(content),
            "word_count": len(content.split()),
            "sentence_count": len(content.split('.')),
            "paragraph_count": len(content.split('\n\n')),
            "avg_sentence_length": len(content.split()) / max(1, len(content.split('.'))),
            "avg_word_length": sum(len(word) for word in content.split()) / max(1, len(content.split())),
            "unique_word_ratio": len(set(content.lower().split())) / max(1, len(content.split())),
            "readability_score": self._calculate_readability(content),
            "complexity_score": self._calculate_complexity(content)
        }
        
        # Update metadata
        metadata.content_length = metrics["content_length"]
        metadata.word_count = metrics["word_count"]
        metadata.sentence_count = metrics["sentence_count"]
        metadata.paragraph_count = metrics["paragraph_count"]
        
        return metrics
    
    def _calculate_readability(self, content: str) -> float:
        """Calculate readability score."""
        sentences = content.split('.')
        words = content.split()
        syllables = sum(self._count_syllables(word) for word in words)
        
        if len(sentences) == 0 or len(words) == 0:
            return 0.0
        
        # Flesch Reading Ease
        avg_sentence_length = len(words) / len(sentences)
        avg_syllables_per_word = syllables / len(words)
        
        readability = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
        return max(0.0, min(100.0, readability))
    
    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word."""
        word = word.lower()
        count = 0
        vowels = "aeiouy"
        on_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not on_vowel:
                count += 1
            on_vowel = is_vowel
        
        if word.endswith('e'):
            count -= 1
        
        return max(1, count)
    
    def _calculate_complexity(self, content: str) -> float:
        """Calculate complexity score."""
        words = content.split()
        if not words:
            return 0.0
        
        # Calculate average word length
        avg_word_length = sum(len(word) for word in words) / len(words)
        
        # Calculate unique word ratio
        unique_ratio = len(set(words)) / len(words)
        
        # Calculate sentence complexity
        sentences = content.split('.')
        avg_sentence_length = len(words) / max(1, len(sentences))
        
        # Combine metrics
        complexity = (avg_word_length * 0.4 + unique_ratio * 0.3 + avg_sentence_length * 0.3) / 10
        return min(1.0, complexity)
    
    def _calculate_topic_diversity(self, content: str) -> float:
        """Calculate topic diversity score."""
        topics = self._extract_topics(content)
        if not topics:
            return 0.0
        
        # Calculate diversity based on number of topics and their distribution
        topic_count = len(topics)
        max_score = max(topic["score"] for topic in topics) if topics else 0
        
        diversity = (topic_count * 0.6 + max_score * 0.4) / 10
        return min(1.0, diversity)
    
    def _analyze_structure(self, content: str) -> Dict[str, Any]:
        """Analyze content structure."""
        paragraphs = content.split('\n\n')
        sentences = content.split('.')
        
        return {
            "paragraph_count": len(paragraphs),
            "sentence_count": len(sentences),
            "avg_paragraph_length": len(sentences) / max(1, len(paragraphs)),
            "has_headings": any(line.strip().isupper() for line in content.split('\n')),
            "has_lists": any(line.strip().startswith(('-', '*', '1.', '2.')) for line in content.split('\n'))
        }
    
    def _generate_cache_key(self, file_path: str) -> str:
        """Generate cache key for file."""
        return hashlib.md5(f"{file_path}:{Path(file_path).stat().st_mtime}".encode()).hexdigest()
    
    def _cleanup_cache(self) -> Any:
        """Cleanup old cache entries."""
        current_time = time.time()
        keys_to_remove = []
        
        for key, value in self._cache.items():
            if current_time - value["timestamp"] > self.config.cache_ttl:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self._cache[key]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        return {
            "documents_processed": self.stats["documents_processed"],
            "cache_hits": self.stats["cache_hits"],
            "errors": self.stats["errors"],
            "cache_size": len(self._cache),
            "config": {
                "batch_size": self.config.batch_size,
                "max_workers": self.config.max_workers,
                "enable_caching": self.config.enable_caching
            }
        }
    
    async def startup(self) -> Any:
        """Start the document pipeline."""
        logger.info("🚀 Starting Ultra Document Pipeline v4.0")
        logger.info(f"✅ Pipeline configured with {self.config.max_workers} workers")
    
    async def shutdown(self) -> Any:
        """Shutdown the document pipeline."""
        logger.info("🛑 Shutting down Document Pipeline")
        
        if self._thread_pool:
            self._thread_pool.shutdown(wait=True)
        
        if self._process_pool:
            self._process_pool.shutdown(wait=True)
        
        logger.info("✅ Document Pipeline shutdown complete")
    
    async def health_check(self) -> Dict[str, Any]:
        """Check pipeline health."""
        return {
            "status": "healthy",
            "workers_available": self.config.max_workers,
            "cache_size": len(self._cache),
            "stats": dict(self.stats)
        }


# Example usage
async def main():
    """Example usage of Document Pipeline"""
    
    # Initialize pipeline
    config = PipelineConfig(
        enable_document_intelligence=True,
        enable_citation_management=True,
        enable_nlp_analysis=True,
        enable_ml_integration=True,
        enable_performance_optimization=True,
        enable_ocr=True,
        enable_sentiment_analysis=True,
        enable_keyword_extraction=True,
        enable_topic_modeling=True,
        enable_entity_recognition=True,
        enable_summarization=True,
        enable_citation_generation=True,
        enable_insight_generation=True,
        batch_size=5,
        max_workers=4,
        output_format="json",
        include_metadata=True,
        include_metrics=True,
        include_insights=True
    )
    
    async with UltraDocumentProcessor(config) as processor:
        # Process single document
        result = await processor.process_document("path/to/document.pdf")
        print(f"Pipeline completed in {result.processing_time:.2f}s")
        print(f"Stages completed: {result.stages_completed}")
        print(f"Errors: {result.errors}")
        
        # Process multiple documents
        documents = ["doc1.pdf", "doc2.docx", "doc3.txt"]
        results = await processor.process_documents_batch(documents)
        print(f"Processed {len(results)} documents")
        
        # Get metrics
        metrics = await processor.get_stats()
        print(f"Metrics: {metrics}")
        
        # Health check
        health = await processor.health_check()
        print(f"Health: {health}")


match __name__:
    case "__main__":
    asyncio.run(main()) 