from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
TIMEOUT_SECONDS = 60

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json
import hashlib
from concurrent.futures import ThreadPoolExecutor
import threading
from contextlib import asynccontextmanager
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
import redis.asyncio as redis
from sqlalchemy.ext.asyncio import AsyncSession
import aiofiles
import aiohttp
from .entities import Document, Citation, Analysis, Insight, ProcessingResult
from ..nlp import NLPEngine
from ..ml_integration import MLModelManager
from ..optimization import UltraPerformanceBoost
from ..api.enhanced_api import EnhancedAPI
            import PyPDF2
            import io
            import pytesseract
            from PIL import Image
            import io
        import mimetypes
from typing import Any, List, Dict, Optional
"""
Advanced Document Intelligence Engine
====================================

A comprehensive engine that integrates all AI capabilities for document processing,
analysis, citation, and intelligent insights generation.
"""



# Core imports

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ProcessingConfig:
    """Configuration for document processing"""
    enable_ocr: bool = True
    enable_sentiment_analysis: bool = True
    enable_keyword_extraction: bool = True
    enable_topic_modeling: bool = True
    enable_entity_recognition: bool = True
    enable_summarization: bool = True
    enable_citation_generation: bool = True
    enable_insight_generation: bool = True
    batch_size: int = 10
    max_workers: int = 4
    cache_ttl: int = 3600
    enable_gpu: bool = True
    enable_quantization: bool = True


class DocumentIntelligenceEngine:
    """
    Advanced Document Intelligence Engine
    
    Integrates all AI capabilities for comprehensive document processing:
    - Document parsing and OCR
    - NLP analysis (sentiment, keywords, topics, entities)
    - Citation generation and validation
    - Insight generation
    - Performance optimization
    - Caching and persistence
    """
    
    def __init__(
        self,
        config: ProcessingConfig = None,
        redis_url: str = "redis://localhost:6379",
        db_session: AsyncSession = None
    ):
        
    """__init__ function."""
self.config = config or ProcessingConfig()
        self.redis_url = redis_url
        self.db_session = db_session
        
        # Initialize components
        self.nlp_engine = NLPEngine()
        self.ml_manager = MLModelManager()
        self.performance_boost = UltraPerformanceBoost()
        self.api = EnhancedAPI()
        
        # Threading and async
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
        self.lock = threading.Lock()
        
        # Cache and state
        self._cache = {}
        self._processing_queue = asyncio.Queue()
        self._results_cache = {}
        
        # Performance metrics
        self.metrics = {
            'documents_processed': 0,
            'processing_time_avg': 0.0,
            'cache_hits': 0,
            'cache_misses': 0,
            'errors': 0
        }
        
        logger.info("Document Intelligence Engine initialized")
    
    async def __aenter__(self) -> Any:
        """Async context manager entry"""
        await self.startup()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> Any:
        """Async context manager exit"""
        await self.shutdown()
    
    async def startup(self) -> Any:
        """Initialize all components"""
        try:
            # Initialize Redis connection
            self.redis = redis.from_url(self.redis_url)
            await self.redis.ping()
            
            # Initialize performance boost
            await self.performance_boost.initialize()
            
            # Initialize ML models
            await self.ml_manager.initialize()
            
            # Start background workers
            asyncio.create_task(self._background_processor())
            
            logger.info("Document Intelligence Engine started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start Document Intelligence Engine: {e}")
            raise
    
    async def shutdown(self) -> Any:
        """Cleanup and shutdown"""
        try:
            # Stop background workers
            self._processing_queue.put_nowait(None)
            
            # Close connections
            if hasattr(self, 'redis'):
                await self.redis.close()
            
            # Shutdown executor
            self.executor.shutdown(wait=True)
            
            logger.info("Document Intelligence Engine shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
    
    async def process_document(
        self,
        document_path: Union[str, Path],
        config: ProcessingConfig = None
    ) -> ProcessingResult:
        """
        Process a single document with comprehensive analysis
        
        Args:
            document_path: Path to the document
            config: Processing configuration override
            
        Returns:
            ProcessingResult with all analysis data
        """
        start_time = time.time()
        config = config or self.config
        
        try:
            # Check cache first
            cache_key = self._generate_cache_key(document_path, config)
            cached_result = await self._get_cached_result(cache_key)
            if cached_result:
                self.metrics['cache_hits'] += 1
                return cached_result
            
            self.metrics['cache_misses'] += 1
            
            # Load and parse document
            document = await self._load_document(document_path)
            
            # Extract text with OCR if needed
            text_content = await self._extract_text(document, config)
            
            # Perform comprehensive analysis
            analysis = await self._analyze_content(text_content, config)
            
            # Generate citations
            citations = await self._generate_citations(document, analysis, config)
            
            # Generate insights
            insights = await self._generate_insights(document, analysis, citations, config)
            
            # Create processing result
            result = ProcessingResult(
                document=document,
                analysis=analysis,
                citations=citations,
                insights=insights,
                processing_time=time.time() - start_time,
                cache_key=cache_key
            )
            
            # Cache result
            await self._cache_result(cache_key, result)
            
            # Update metrics
            self.metrics['documents_processed'] += 1
            self.metrics['processing_time_avg'] = (
                (self.metrics['processing_time_avg'] * (self.metrics['documents_processed'] - 1) + 
                 result.processing_time) / self.metrics['documents_processed']
            )
            
            return result
            
        except Exception as e:
            self.metrics['errors'] += 1
            logger.error(f"Error processing document {document_path}: {e}")
            raise
    
    async def process_documents_batch(
        self,
        document_paths: List[Union[str, Path]],
        config: ProcessingConfig = None
    ) -> List[ProcessingResult]:
        """
        Process multiple documents in parallel
        
        Args:
            document_paths: List of document paths
            config: Processing configuration override
            
        Returns:
            List of ProcessingResult objects
        """
        config = config or self.config
        
        # Create tasks for parallel processing
        tasks = [
            self.process_document(path, config)
            for path in document_paths
        ]
        
        # Process in batches
        results = []
        for i in range(0, len(tasks), config.batch_size):
            batch = tasks[i:i + config.batch_size]
            batch_results = await asyncio.gather(*batch, return_exceptions=True)
            
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Batch processing error: {result}")
                    self.metrics['errors'] += 1
                else:
                    results.append(result)
        
        return results
    
    async def _load_document(self, document_path: Union[str, Path]) -> Document:
        """Load and parse document"""
        document_path = Path(document_path)
        
        if not document_path.exists():
            raise FileNotFoundError(f"Document not found: {document_path}")
        
        # Read file content
        async with aiofiles.open(document_path, 'rb') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            content = await f.read()
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        
        # Create document object
        document = Document(
            id=str(document_path),
            name=document_path.name,
            path=str(document_path),
            size=len(content),
            content=content,
            mime_type=self._detect_mime_type(document_path),
            created_at=time.time(),
            metadata={
                'extension': document_path.suffix,
                'parent_dir': str(document_path.parent)
            }
        )
        
        return document
    
    async def _extract_text(self, document: Document, config: ProcessingConfig) -> str:
        """Extract text content from document"""
        if document.mime_type.startswith('text/'):
            # Text-based documents
            return document.content.decode('utf-8')
        
        elif document.mime_type == 'application/pdf':
            # PDF documents
            return await self._extract_pdf_text(document)
        
        elif document.mime_type.startswith('image/'):
            # Image documents with OCR
            if config.enable_ocr:
                return await self._extract_ocr_text(document)
            else:
                return ""
        
        else:
            # Other document types
            return await self._extract_generic_text(document)
    
    async def _analyze_content(self, text: str, config: ProcessingConfig) -> Analysis:
        """Perform comprehensive content analysis"""
        analysis = Analysis()
        
        if not text.strip():
            return analysis
        
        # Parallel analysis tasks
        tasks = []
        
        if config.enable_sentiment_analysis:
            tasks.append(self.nlp_engine.analyze_sentiment(text))
        
        if config.enable_keyword_extraction:
            tasks.append(self.nlp_engine.extract_keywords(text))
        
        if config.enable_topic_modeling:
            tasks.append(self.nlp_engine.model_topics(text))
        
        if config.enable_entity_recognition:
            tasks.append(self.nlp_engine.recognize_entities(text))
        
        if config.enable_summarization:
            tasks.append(self.nlp_engine.summarize_text(text))
        
        # Execute all analysis tasks
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Analysis error: {result}")
                    continue
                
                if i == 0 and config.enable_sentiment_analysis:
                    analysis.sentiment = result
                elif i == 1 and config.enable_keyword_extraction:
                    analysis.keywords = result
                elif i == 2 and config.enable_topic_modeling:
                    analysis.topics = result
                elif i == 3 and config.enable_entity_recognition:
                    analysis.entities = result
                elif i == 4 and config.enable_summarization:
                    analysis.summary = result
        
        return analysis
    
    async def _generate_citations(
        self,
        document: Document,
        analysis: Analysis,
        config: ProcessingConfig
    ) -> List[Citation]:
        """Generate citations for document content"""
        if not config.enable_citation_generation:
            return []
        
        citations = []
        
        # Extract potential citations from text
        text_content = document.content.decode('utf-8') if isinstance(document.content, bytes) else str(document.content)
        
        # Use NLP to identify citation patterns
        citation_patterns = await self.nlp_engine.extract_citation_patterns(text_content)
        
        for pattern in citation_patterns:
            citation = Citation(
                id=f"citation_{len(citations)}",
                source=pattern.get('source', ''),
                title=pattern.get('title', ''),
                authors=pattern.get('authors', []),
                year=pattern.get('year', ''),
                url=pattern.get('url', ''),
                confidence=pattern.get('confidence', 0.0),
                context=pattern.get('context', '')
            )
            citations.append(citation)
        
        return citations
    
    async def _generate_insights(
        self,
        document: Document,
        analysis: Analysis,
        citations: List[Citation],
        config: ProcessingConfig
    ) -> List[Insight]:
        """Generate intelligent insights from analysis"""
        if not config.enable_insight_generation:
            return []
        
        insights = []
        
        # Generate insights based on analysis results
        if analysis.sentiment:
            sentiment_insight = Insight(
                id=f"insight_sentiment_{len(insights)}",
                type="sentiment",
                title="Document Sentiment Analysis",
                description=f"Document shows {analysis.sentiment.polarity} sentiment with {analysis.sentiment.confidence:.2f} confidence",
                confidence=analysis.sentiment.confidence,
                data=analysis.sentiment.dict()
            )
            insights.append(sentiment_insight)
        
        if analysis.keywords:
            keyword_insight = Insight(
                id=f"insight_keywords_{len(insights)}",
                type="keywords",
                title="Key Topics Identified",
                description=f"Found {len(analysis.keywords)} key topics in the document",
                confidence=0.8,
                data={'keywords': analysis.keywords}
            )
            insights.append(keyword_insight)
        
        if analysis.topics:
            topic_insight = Insight(
                id=f"insight_topics_{len(insights)}",
                type="topics",
                title="Document Topics",
                description=f"Document covers {len(analysis.topics)} main topics",
                confidence=0.7,
                data={'topics': analysis.topics}
            )
            insights.append(topic_insight)
        
        if citations:
            citation_insight = Insight(
                id=f"insight_citations_{len(insights)}",
                type="citations",
                title="References and Citations",
                description=f"Document references {len(citations)} external sources",
                confidence=0.9,
                data={'citations': [c.dict() for c in citations]}
            )
            insights.append(citation_insight)
        
        return insights
    
    async def _extract_pdf_text(self, document: Document) -> str:
        """Extract text from PDF document"""
        try:
            
            pdf_file = io.BytesIO(document.content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            
            return text
            
        except ImportError:
            logger.warning("PyPDF2 not available, using fallback PDF extraction")
            return ""
        except Exception as e:
            logger.error(f"PDF extraction error: {e}")
            return ""
    
    async def _extract_ocr_text(self, document: Document) -> str:
        """Extract text from image using OCR"""
        try:
            
            image = Image.open(io.BytesIO(document.content))
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            text = pytesseract.image_to_string(image)
            
            return text
            
        except ImportError:
            logger.warning("pytesseract not available, OCR disabled")
            return ""
        except Exception as e:
            logger.error(f"OCR extraction error: {e}")
            return ""
    
    async def _extract_generic_text(self, document: Document) -> str:
        """Extract text from generic document types"""
        try:
            # Try to decode as text
            return document.content.decode('utf-8')
        except UnicodeDecodeError:
            logger.warning(f"Could not decode content as text for {document.name}")
            return ""
    
    def _detect_mime_type(self, file_path: Path) -> str:
        """Detect MIME type of file"""
        
        mime_type, _ = mimetypes.guess_type(str(file_path))
        return mime_type or 'application/octet-stream'
    
    def _generate_cache_key(self, document_path: Union[str, Path], config: ProcessingConfig) -> str:
        """Generate cache key for document processing"""
        document_path = Path(document_path)
        
        # Create hash from file path and config
        content = f"{document_path}_{config.__dict__}"
        return hashlib.md5(content.encode()).hexdigest()
    
    async def _get_cached_result(self, cache_key: str) -> Optional[ProcessingResult]:
        """Get cached processing result"""
        try:
            # Check memory cache first
            if cache_key in self._cache:
                return self._cache[cache_key]
            
            # Check Redis cache
            if hasattr(self, 'redis'):
                cached_data = await self.redis.get(f"doc_intel:{cache_key}")
                if cached_data:
                    result_dict = json.loads(cached_data)
                    return ProcessingResult(**result_dict)
            
            return None
            
        except Exception as e:
            logger.error(f"Cache retrieval error: {e}")
            return None
    
    async def _cache_result(self, cache_key: str, result: ProcessingResult):
        """Cache processing result"""
        try:
            # Cache in memory
            self._cache[cache_key] = result
            
            # Cache in Redis
            if hasattr(self, 'redis'):
                result_dict = result.dict()
                await self.redis.setex(
                    f"doc_intel:{cache_key}",
                    self.config.cache_ttl,
                    json.dumps(result_dict)
                )
            
        except Exception as e:
            logger.error(f"Cache storage error: {e}")
    
    async def _background_processor(self) -> Any:
        """Background processor for queued documents"""
        while True:
            try:
                item = await self._processing_queue.get()
                if item is None:
                    break
                
                document_path, config, future = item
                
                try:
                    result = await self.process_document(document_path, config)
                    future.set_result(result)
                except Exception as e:
                    future.set_exception(e)
                finally:
                    self._processing_queue.task_done()
                    
            except Exception as e:
                logger.error(f"Background processor error: {e}")
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return {
            **self.metrics,
            'cache_size': len(self._cache),
            'queue_size': self._processing_queue.qsize(),
            'uptime': time.time() - getattr(self, '_start_time', time.time())
        }
    
    async def clear_cache(self) -> Any:
        """Clear all caches"""
        self._cache.clear()
        if hasattr(self, 'redis'):
            await self.redis.flushdb()
        logger.info("Cache cleared")
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        try:
            # Check Redis connection
            redis_ok = False
            if hasattr(self, 'redis'):
                await self.redis.ping()
                redis_ok = True
            
            # Check NLP engine
            nlp_ok = self.nlp_engine is not None
            
            # Check ML manager
            ml_ok = self.ml_manager is not None
            
            return {
                'status': 'healthy' if all([redis_ok, nlp_ok, ml_ok]) else 'degraded',
                'components': {
                    'redis': redis_ok,
                    'nlp_engine': nlp_ok,
                    'ml_manager': ml_ok
                },
                'metrics': await self.get_metrics()
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                'status': 'unhealthy',
                'error': str(e)
            }


# Example usage
async def main():
    """Example usage of Document Intelligence Engine"""
    
    # Initialize engine
    config = ProcessingConfig(
        enable_ocr=True,
        enable_sentiment_analysis=True,
        enable_keyword_extraction=True,
        enable_topic_modeling=True,
        enable_entity_recognition=True,
        enable_summarization=True,
        enable_citation_generation=True,
        enable_insight_generation=True,
        batch_size=5,
        max_workers=4
    )
    
    async with DocumentIntelligenceEngine(config) as engine:
        # Process single document
        result = await engine.process_document("path/to/document.pdf")
        print(f"Processing completed in {result.processing_time:.2f}s")
        
        # Process multiple documents
        documents = ["doc1.pdf", "doc2.docx", "doc3.txt"]
        results = await engine.process_documents_batch(documents)
        print(f"Processed {len(results)} documents")
        
        # Get metrics
        metrics = await engine.get_metrics()
        print(f"Metrics: {metrics}")
        
        # Health check
        health = await engine.health_check()
        print(f"Health: {health}")


match __name__:
    case "__main__":
    asyncio.run(main()) 