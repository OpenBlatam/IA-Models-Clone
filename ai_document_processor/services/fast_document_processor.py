"""
Fast Document Processor - Ultra High Performance Version
======================================================

Optimized document processor with maximum speed and efficiency.
Implements parallel processing, streaming, and memory optimization.
"""

import asyncio
import logging
import time
import hashlib
from typing import Optional, Dict, Any, List, AsyncGenerator, Tuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from pathlib import Path
import multiprocessing as mp
import gc
import weakref
from dataclasses import dataclass
import json

from services.enhanced_cache_service import get_cache_service
from services.performance_monitor import get_performance_monitor
from utils.file_handlers import FileHandlerFactory
from services.ai_classifier import AIClassifier
from services.professional_transformer import ProfessionalTransformer
from models.document_models import (
    DocumentAnalysis, ProfessionalDocument, ProfessionalFormat,
    DocumentProcessingRequest, DocumentProcessingResponse
)

logger = logging.getLogger(__name__)

@dataclass
class ProcessingChunk:
    """Chunk of document for parallel processing"""
    content: str
    chunk_id: int
    total_chunks: int
    metadata: Dict[str, Any]

class FastDocumentProcessor:
    """Ultra-fast document processor with parallel processing"""
    
    def __init__(self, 
                 max_workers: Optional[int] = None,
                 chunk_size: int = 8192,
                 enable_streaming: bool = True,
                 enable_parallel_ai: bool = True):
        """
        Initialize fast document processor
        
        Args:
            max_workers: Maximum number of worker threads/processes
            chunk_size: Size of processing chunks
            enable_streaming: Enable streaming processing
            enable_parallel_ai: Enable parallel AI processing
        """
        self.max_workers = max_workers or min(32, (mp.cpu_count() or 1) + 4)
        self.chunk_size = chunk_size
        self.enable_streaming = enable_streaming
        self.enable_parallel_ai = enable_parallel_ai
        
        # Initialize components
        self.file_handler_factory = FileHandlerFactory()
        self.ai_classifier = AIClassifier()
        self.professional_transformer = ProfessionalTransformer()
        
        # Thread/Process pools
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        self.process_pool = ProcessPoolExecutor(max_workers=min(4, self.max_workers))
        
        # Cache and monitoring
        self.cache_service = None
        self.performance_monitor = None
        
        # Processing statistics
        self.stats = {
            'documents_processed': 0,
            'total_processing_time': 0.0,
            'avg_processing_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0,
            'parallel_operations': 0
        }
        
        # Memory optimization
        self._processed_documents = weakref.WeakValueDictionary()
        
    async def initialize(self):
        """Initialize the fast processor"""
        logger.info("Initializing fast document processor...")
        
        # Initialize services
        self.cache_service = await get_cache_service()
        self.performance_monitor = await get_performance_monitor()
        
        # Initialize AI components
        await self.ai_classifier.initialize()
        await self.professional_transformer.initialize()
        
        # Register health checks
        self.performance_monitor.register_health_check(
            "fast_processor", 
            self._health_check
        )
        
        logger.info(f"Fast processor initialized with {self.max_workers} workers")
    
    async def _health_check(self) -> Any:
        """Health check for fast processor"""
        from services.performance_monitor import HealthCheck
        
        try:
            # Check thread pool health
            thread_pool_healthy = self.thread_pool._threads and len(self.thread_pool._threads) > 0
            
            # Check memory usage
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            status = "healthy"
            message = f"Fast processor running with {self.max_workers} workers"
            
            if memory_mb > 1000:  # 1GB
                status = "warning"
                message = f"High memory usage: {memory_mb:.1f}MB"
            
            if not thread_pool_healthy:
                status = "critical"
                message = "Thread pool not healthy"
            
            return HealthCheck(
                name="fast_processor",
                status=status,
                message=message,
                timestamp=time.time(),
                details={
                    'workers': self.max_workers,
                    'memory_mb': memory_mb,
                    'thread_pool_healthy': thread_pool_healthy
                }
            )
        except Exception as e:
            return HealthCheck(
                name="fast_processor",
                status="critical",
                message=f"Health check failed: {str(e)}",
                timestamp=time.time()
            )
    
    def _generate_cache_key(self, file_path: str, processing_options: Dict[str, Any]) -> str:
        """Generate cache key for document processing"""
        # Create hash of file path and options
        key_data = {
            'file_path': file_path,
            'file_mtime': Path(file_path).stat().st_mtime if Path(file_path).exists() else 0,
            'options': processing_options
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    async def process_document_fast(self, 
                                  file_path: str, 
                                  filename: str,
                                  processing_options: Optional[Dict[str, Any]] = None) -> DocumentProcessingResponse:
        """Process document with maximum speed optimization"""
        start_time = time.time()
        timer_id = self.performance_monitor.start_operation_timer("fast_document_processing")
        
        try:
            processing_options = processing_options or {}
            cache_key = self._generate_cache_key(file_path, processing_options)
            
            # Check cache first
            cached_result = await self.cache_service.get(cache_key)
            if cached_result:
                self.stats['cache_hits'] += 1
                logger.info(f"Cache hit for {filename}")
                return cached_result
            
            self.stats['cache_misses'] += 1
            
            # Process document with parallel optimization
            if self.enable_streaming and Path(file_path).stat().st_size > 1024 * 1024:  # > 1MB
                result = await self._process_large_document_streaming(file_path, filename, processing_options)
            else:
                result = await self._process_document_parallel(file_path, filename, processing_options)
            
            # Cache result
            await self.cache_service.set(cache_key, result, ttl=3600)
            
            # Update statistics
            processing_time = time.time() - start_time
            self.stats['documents_processed'] += 1
            self.stats['total_processing_time'] += processing_time
            self.stats['avg_processing_time'] = (
                self.stats['total_processing_time'] / self.stats['documents_processed']
            )
            
            # Record metrics
            await self.performance_monitor.record_metric(
                "fast_processor.document_processing_time_ms",
                processing_time * 1000,
                unit="ms"
            )
            
            await self.performance_monitor.record_metric(
                "fast_processor.documents_processed_total",
                self.stats['documents_processed'],
                unit="count"
            )
            
            self.performance_monitor.end_operation_timer(timer_id, success=True)
            
            logger.info(f"Fast processing completed for {filename} in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            self.performance_monitor.end_operation_timer(timer_id, success=False)
            self.performance_monitor.record_error("fast_document_processing", e)
            logger.error(f"Fast processing failed for {filename}: {e}")
            raise
    
    async def _process_document_parallel(self, 
                                       file_path: str, 
                                       filename: str,
                                       processing_options: Dict[str, Any]) -> DocumentProcessingResponse:
        """Process document using parallel techniques"""
        
        # Step 1: Extract text in parallel with classification
        text_extraction_task = asyncio.create_task(
            self._extract_text_async(file_path, filename)
        )
        
        # Step 2: Start AI classification in parallel
        classification_task = None
        if processing_options.get('enable_ai_classification', True):
            classification_task = asyncio.create_task(
                self._classify_document_async(file_path, filename)
            )
        
        # Wait for text extraction
        extracted_text = await text_extraction_task
        
        # Step 3: Process text chunks in parallel
        if len(extracted_text) > self.chunk_size:
            chunks = self._split_into_chunks(extracted_text)
            processed_chunks = await self._process_chunks_parallel(chunks)
            processed_text = self._merge_chunks(processed_chunks)
        else:
            processed_text = extracted_text
        
        # Step 4: Get classification result
        classification = None
        if classification_task:
            classification = await classification_task
        
        # Step 5: Transform to professional format
        professional_doc = await self._transform_to_professional_async(
            processed_text, classification, processing_options
        )
        
        return DocumentProcessingResponse(
            success=True,
            document_analysis=classification,
            professional_document=professional_doc,
            processing_time=time.time(),
            metadata={
                'processor': 'fast_parallel',
                'chunks_processed': len(chunks) if len(extracted_text) > self.chunk_size else 1,
                'parallel_operations': self.stats['parallel_operations']
            }
        )
    
    async def _process_large_document_streaming(self, 
                                              file_path: str, 
                                              filename: str,
                                              processing_options: Dict[str, Any]) -> DocumentProcessingResponse:
        """Process large documents using streaming"""
        
        # Stream processing for large files
        processed_chunks = []
        classification = None
        
        async for chunk in self._stream_document_chunks(file_path):
            # Process chunk in parallel
            processed_chunk = await self._process_chunk_async(chunk)
            processed_chunks.append(processed_chunk)
            
            # Get classification from first chunk
            if classification is None and chunk.chunk_id == 0:
                classification = await self._classify_chunk_async(chunk)
        
        # Merge processed chunks
        processed_text = self._merge_chunks(processed_chunks)
        
        # Transform to professional format
        professional_doc = await self._transform_to_professional_async(
            processed_text, classification, processing_options
        )
        
        return DocumentProcessingResponse(
            success=True,
            document_analysis=classification,
            professional_document=professional_doc,
            processing_time=time.time(),
            metadata={
                'processor': 'fast_streaming',
                'chunks_processed': len(processed_chunks),
                'streaming_enabled': True
            }
        )
    
    async def _extract_text_async(self, file_path: str, filename: str) -> str:
        """Extract text asynchronously"""
        loop = asyncio.get_event_loop()
        
        def extract_text_sync():
            handler = self.file_handler_factory.get_handler(file_path)
            return handler.extract_text(file_path)
        
        return await loop.run_in_executor(self.thread_pool, extract_text_sync)
    
    async def _classify_document_async(self, file_path: str, filename: str) -> DocumentAnalysis:
        """Classify document asynchronously"""
        loop = asyncio.get_event_loop()
        
        def classify_sync():
            return self.ai_classifier.classify_document(file_path, filename)
        
        return await loop.run_in_executor(self.thread_pool, classify_sync)
    
    async def _classify_chunk_async(self, chunk: ProcessingChunk) -> DocumentAnalysis:
        """Classify document chunk asynchronously"""
        loop = asyncio.get_event_loop()
        
        def classify_chunk_sync():
            return self.ai_classifier.classify_text(chunk.content)
        
        return await loop.run_in_executor(self.thread_pool, classify_chunk_sync)
    
    async def _transform_to_professional_async(self, 
                                             text: str, 
                                             classification: Optional[DocumentAnalysis],
                                             processing_options: Dict[str, Any]) -> ProfessionalDocument:
        """Transform to professional format asynchronously"""
        loop = asyncio.get_event_loop()
        
        def transform_sync():
            return self.professional_transformer.transform_to_professional(
                text, classification, processing_options
            )
        
        return await loop.run_in_executor(self.thread_pool, transform_sync)
    
    def _split_into_chunks(self, text: str) -> List[ProcessingChunk]:
        """Split text into processing chunks"""
        chunks = []
        text_length = len(text)
        chunk_count = (text_length + self.chunk_size - 1) // self.chunk_size
        
        for i in range(chunk_count):
            start = i * self.chunk_size
            end = min(start + self.chunk_size, text_length)
            
            chunk = ProcessingChunk(
                content=text[start:end],
                chunk_id=i,
                total_chunks=chunk_count,
                metadata={'start': start, 'end': end}
            )
            chunks.append(chunk)
        
        return chunks
    
    async def _process_chunks_parallel(self, chunks: List[ProcessingChunk]) -> List[str]:
        """Process chunks in parallel"""
        tasks = [self._process_chunk_async(chunk) for chunk in chunks]
        self.stats['parallel_operations'] += len(tasks)
        return await asyncio.gather(*tasks)
    
    async def _process_chunk_async(self, chunk: ProcessingChunk) -> str:
        """Process individual chunk asynchronously"""
        loop = asyncio.get_event_loop()
        
        def process_chunk_sync():
            # Basic text processing (can be enhanced with specific processing logic)
            return chunk.content.strip()
        
        return await loop.run_in_executor(self.thread_pool, process_chunk_sync)
    
    async def _stream_document_chunks(self, file_path: str) -> AsyncGenerator[ProcessingChunk, None]:
        """Stream document in chunks for large files"""
        handler = self.file_handler_factory.get_handler(file_path)
        
        # For now, read entire file and chunk it
        # In a real implementation, this would stream from disk
        full_text = handler.extract_text(file_path)
        chunks = self._split_into_chunks(full_text)
        
        for chunk in chunks:
            yield chunk
            await asyncio.sleep(0)  # Yield control
    
    def _merge_chunks(self, processed_chunks: List[str]) -> str:
        """Merge processed chunks back into full text"""
        return '\n'.join(processed_chunks)
    
    async def process_batch_fast(self, 
                               file_paths: List[str],
                               processing_options: Optional[Dict[str, Any]] = None) -> List[DocumentProcessingResponse]:
        """Process multiple documents in parallel"""
        processing_options = processing_options or {}
        
        # Create processing tasks
        tasks = [
            self.process_document_fast(file_path, Path(file_path).name, processing_options)
            for file_path in file_paths
        ]
        
        # Process in parallel with semaphore to limit concurrent operations
        semaphore = asyncio.Semaphore(min(len(file_paths), self.max_workers))
        
        async def process_with_semaphore(task):
            async with semaphore:
                return await task
        
        # Execute all tasks
        results = await asyncio.gather(
            *[process_with_semaphore(task) for task in tasks],
            return_exceptions=True
        )
        
        # Handle results and exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch processing failed for {file_paths[i]}: {result}")
                processed_results.append(DocumentProcessingResponse(
                    success=False,
                    error=str(result),
                    processing_time=time.time()
                ))
            else:
                processed_results.append(result)
        
        return processed_results
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            **self.stats,
            'cache_hit_rate': (
                self.stats['cache_hits'] / (self.stats['cache_hits'] + self.stats['cache_misses']) * 100
                if (self.stats['cache_hits'] + self.stats['cache_misses']) > 0 else 0
            ),
            'max_workers': self.max_workers,
            'chunk_size': self.chunk_size,
            'streaming_enabled': self.enable_streaming,
            'parallel_ai_enabled': self.enable_parallel_ai
        }
    
    async def close(self):
        """Close processor and cleanup resources"""
        logger.info("Closing fast document processor...")
        
        # Shutdown thread pools
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)
        
        # Clear caches
        if self.cache_service:
            await self.cache_service.clear()
        
        # Force garbage collection
        gc.collect()
        
        logger.info("Fast document processor closed")

# Global fast processor instance
_fast_processor: Optional[FastDocumentProcessor] = None

async def get_fast_processor() -> FastDocumentProcessor:
    """Get global fast processor instance"""
    global _fast_processor
    if _fast_processor is None:
        _fast_processor = FastDocumentProcessor()
        await _fast_processor.initialize()
    return _fast_processor

async def close_fast_processor():
    """Close global fast processor"""
    global _fast_processor
    if _fast_processor:
        await _fast_processor.close()
        _fast_processor = None

















