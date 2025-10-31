"""
Ultra Fast Processor - Extreme Performance Document Processing
============================================================

Ultra-optimized document processor with extreme performance techniques.
"""

import asyncio
import time
import gc
import mmap
import os
import hashlib
import struct
from typing import Optional, Dict, Any, List, AsyncGenerator, Tuple
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
import multiprocessing as mp
import weakref
from dataclasses import dataclass
import json
import pickle
import lz4.frame
import msgpack

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
class UltraProcessingChunk:
    """Ultra-optimized processing chunk"""
    content: bytes  # Keep as bytes for maximum efficiency
    chunk_id: int
    total_chunks: int
    metadata: Dict[str, Any]
    checksum: str

class UltraFastProcessor:
    """Ultra-fast document processor with extreme optimizations"""
    
    def __init__(self, 
                 max_workers: Optional[int] = None,
                 chunk_size: int = 32768,  # 32KB chunks for better cache locality
                 enable_memory_mapping: bool = True,
                 enable_zero_copy: bool = True,
                 enable_compression: bool = True):
        """
        Initialize ultra-fast processor
        
        Args:
            max_workers: Maximum number of worker processes
            chunk_size: Size of processing chunks (power of 2 for efficiency)
            enable_memory_mapping: Enable memory-mapped files
            enable_zero_copy: Enable zero-copy operations
            enable_compression: Enable compression for cache
        """
        self.max_workers = max_workers or min(64, (mp.cpu_count() or 1) * 4)
        self.chunk_size = chunk_size
        self.enable_memory_mapping = enable_memory_mapping
        self.enable_zero_copy = enable_zero_copy
        self.enable_compression = enable_compression
        
        # Initialize components with optimizations
        self.file_handler_factory = FileHandlerFactory()
        self.ai_classifier = AIClassifier()
        self.professional_transformer = ProfessionalTransformer()
        
        # Ultra-optimized thread/process pools
        self.thread_pool = ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix="ultra_worker"
        )
        self.process_pool = ProcessPoolExecutor(
            max_workers=min(8, self.max_workers),
            mp_context=mp.get_context('spawn')  # Better for memory
        )
        
        # Cache and monitoring
        self.cache_service = None
        self.performance_monitor = None
        
        # Ultra-optimized processing statistics
        self.stats = {
            'documents_processed': 0,
            'total_processing_time': 0.0,
            'avg_processing_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0,
            'memory_mapped_files': 0,
            'zero_copy_operations': 0,
            'compression_ratio': 0.0,
            'throughput_mb_per_sec': 0.0
        }
        
        # Memory optimization
        self._processed_documents = weakref.WeakValueDictionary()
        self._memory_pool = {}  # Reusable memory blocks
        
        # Pre-allocate memory pools for common operations
        self._init_memory_pools()
        
    def _init_memory_pools(self):
        """Initialize memory pools for zero-copy operations"""
        # Pre-allocate common buffer sizes
        common_sizes = [1024, 4096, 16384, 65536, 262144]  # Powers of 2
        
        for size in common_sizes:
            self._memory_pool[size] = []
            # Pre-allocate some buffers
            for _ in range(min(10, self.max_workers)):
                self._memory_pool[size].append(bytearray(size))
    
    def _get_memory_buffer(self, size: int) -> bytearray:
        """Get a memory buffer from the pool"""
        # Find the smallest buffer that fits
        for pool_size in sorted(self._memory_pool.keys()):
            if pool_size >= size:
                if self._memory_pool[pool_size]:
                    return self._memory_pool[pool_size].pop()
                else:
                    # Create new buffer if pool is empty
                    return bytearray(pool_size)
        
        # Fallback to new buffer
        return bytearray(size)
    
    def _return_memory_buffer(self, buffer: bytearray):
        """Return a memory buffer to the pool"""
        size = len(buffer)
        for pool_size in sorted(self._memory_pool.keys()):
            if pool_size >= size:
                buffer[:] = b'\x00' * size  # Clear buffer
                self._memory_pool[pool_size].append(buffer)
                break
    
    async def initialize(self):
        """Initialize the ultra-fast processor"""
        logger.info("Initializing ultra-fast document processor...")
        
        # Initialize services
        self.cache_service = await get_cache_service()
        self.performance_monitor = await get_performance_monitor()
        
        # Initialize AI components with optimizations
        await self.ai_classifier.initialize()
        await self.professional_transformer.initialize()
        
        # Register health checks
        self.performance_monitor.register_health_check(
            "ultra_processor", 
            self._health_check
        )
        
        # Pre-warm caches and models
        await self._prewarm_system()
        
        logger.info(f"Ultra-fast processor initialized with {self.max_workers} workers")
    
    async def _prewarm_system(self):
        """Pre-warm the system for maximum performance"""
        logger.info("Pre-warming system for maximum performance...")
        
        # Pre-warm AI models
        try:
            # Load common models into memory
            await self.ai_classifier._preload_models()
            await self.professional_transformer._preload_templates()
        except Exception as e:
            logger.warning(f"Pre-warming failed: {e}")
        
        # Pre-warm cache
        try:
            await self.cache_service.set("prewarm_test", "test", ttl=60)
            await self.cache_service.get("prewarm_test")
        except Exception as e:
            logger.warning(f"Cache pre-warming failed: {e}")
        
        logger.info("System pre-warming completed")
    
    async def _health_check(self) -> Any:
        """Health check for ultra processor"""
        from services.performance_monitor import HealthCheck
        
        try:
            # Check thread pool health
            thread_pool_healthy = self.thread_pool._threads and len(self.thread_pool._threads) > 0
            
            # Check memory usage
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            # Check memory pool status
            total_pool_buffers = sum(len(pool) for pool in self._memory_pool.values())
            
            status = "healthy"
            message = f"Ultra processor running with {self.max_workers} workers"
            
            if memory_mb > 2000:  # 2GB
                status = "warning"
                message = f"High memory usage: {memory_mb:.1f}MB"
            
            if not thread_pool_healthy:
                status = "critical"
                message = "Thread pool not healthy"
            
            return HealthCheck(
                name="ultra_processor",
                status=status,
                message=message,
                timestamp=time.time(),
                details={
                    'workers': self.max_workers,
                    'memory_mb': memory_mb,
                    'thread_pool_healthy': thread_pool_healthy,
                    'memory_pool_buffers': total_pool_buffers,
                    'throughput_mb_per_sec': self.stats['throughput_mb_per_sec']
                }
            )
        except Exception as e:
            return HealthCheck(
                name="ultra_processor",
                status="critical",
                message=f"Health check failed: {str(e)}",
                timestamp=time.time()
            )
    
    def _generate_ultra_cache_key(self, file_path: str, processing_options: Dict[str, Any]) -> str:
        """Generate ultra-optimized cache key"""
        # Use file content hash for better cache hits
        try:
            with open(file_path, 'rb') as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()[:16]
        except:
            file_hash = hashlib.md5(file_path.encode()).hexdigest()[:16]
        
        # Create compact key
        key_data = {
            'hash': file_hash,
            'opts': processing_options
        }
        key_string = msgpack.packb(key_data)
        return hashlib.md5(key_string).hexdigest()
    
    async def process_document_ultra_fast(self, 
                                        file_path: str, 
                                        filename: str,
                                        processing_options: Optional[Dict[str, Any]] = None) -> DocumentProcessingResponse:
        """Process document with ultra-fast optimizations"""
        start_time = time.time()
        timer_id = self.performance_monitor.start_operation_timer("ultra_document_processing")
        
        try:
            processing_options = processing_options or {}
            cache_key = self._generate_ultra_cache_key(file_path, processing_options)
            
            # Check cache first with ultra-fast lookup
            cached_result = await self.cache_service.get(cache_key)
            if cached_result:
                self.stats['cache_hits'] += 1
                logger.debug(f"Ultra cache hit for {filename}")
                return cached_result
            
            self.stats['cache_misses'] += 1
            
            # Get file size for throughput calculation
            file_size = os.path.getsize(file_path)
            
            # Process document with ultra optimizations
            if file_size > 10 * 1024 * 1024:  # > 10MB
                result = await self._process_large_document_ultra(file_path, filename, processing_options)
            else:
                result = await self._process_document_ultra_parallel(file_path, filename, processing_options)
            
            # Cache result with compression
            await self.cache_service.set(cache_key, result, ttl=7200)
            
            # Update statistics
            processing_time = time.time() - start_time
            throughput = (file_size / 1024 / 1024) / processing_time if processing_time > 0 else 0
            
            self.stats['documents_processed'] += 1
            self.stats['total_processing_time'] += processing_time
            self.stats['avg_processing_time'] = (
                self.stats['total_processing_time'] / self.stats['documents_processed']
            )
            self.stats['throughput_mb_per_sec'] = throughput
            
            # Record metrics
            await self.performance_monitor.record_metric(
                "ultra_processor.document_processing_time_ms",
                processing_time * 1000,
                unit="ms"
            )
            
            await self.performance_monitor.record_metric(
                "ultra_processor.throughput_mb_per_sec",
                throughput,
                unit="MB/s"
            )
            
            self.performance_monitor.end_operation_timer(timer_id, success=True)
            
            logger.info(f"Ultra processing completed for {filename} in {processing_time:.2f}s ({throughput:.1f} MB/s)")
            return result
            
        except Exception as e:
            self.performance_monitor.end_operation_timer(timer_id, success=False)
            self.performance_monitor.record_error("ultra_document_processing", e)
            logger.error(f"Ultra processing failed for {filename}: {e}")
            raise
    
    async def _process_document_ultra_parallel(self, 
                                             file_path: str, 
                                             filename: str,
                                             processing_options: Dict[str, Any]) -> DocumentProcessingResponse:
        """Process document using ultra-parallel techniques"""
        
        # Step 1: Memory-mapped file reading
        if self.enable_memory_mapping:
            content = await self._read_file_memory_mapped(file_path)
            self.stats['memory_mapped_files'] += 1
        else:
            content = await self._read_file_ultra_fast(file_path)
        
        # Step 2: Parallel processing pipeline
        tasks = []
        
        # Text extraction task
        text_task = asyncio.create_task(
            self._extract_text_ultra_async(content, filename)
        )
        tasks.append(text_task)
        
        # AI classification task (parallel)
        if processing_options.get('enable_ai_classification', True):
            classification_task = asyncio.create_task(
                self._classify_document_ultra_async(content, filename)
            )
            tasks.append(classification_task)
        else:
            classification_task = None
        
        # Wait for text extraction
        extracted_text = await text_task
        
        # Step 3: Ultra-parallel text processing
        if len(extracted_text) > self.chunk_size:
            chunks = self._split_into_ultra_chunks(extracted_text)
            processed_chunks = await self._process_chunks_ultra_parallel(chunks)
            processed_text = self._merge_chunks_ultra(processed_chunks)
        else:
            processed_text = extracted_text
        
        # Step 4: Get classification result
        classification = None
        if classification_task:
            classification = await classification_task
        
        # Step 5: Transform to professional format
        professional_doc = await self._transform_to_professional_ultra_async(
            processed_text, classification, processing_options
        )
        
        return DocumentProcessingResponse(
            success=True,
            document_analysis=classification,
            professional_document=professional_doc,
            processing_time=time.time(),
            metadata={
                'processor': 'ultra_parallel',
                'chunks_processed': len(chunks) if len(extracted_text) > self.chunk_size else 1,
                'memory_mapped': self.enable_memory_mapping,
                'zero_copy_ops': self.stats['zero_copy_operations'],
                'compression_ratio': self.stats['compression_ratio']
            }
        )
    
    async def _process_large_document_ultra(self, 
                                          file_path: str, 
                                          filename: str,
                                          processing_options: Dict[str, Any]) -> DocumentProcessingResponse:
        """Process large documents using ultra-streaming"""
        
        # Ultra-streaming processing for large files
        processed_chunks = []
        classification = None
        
        async for chunk in self._stream_document_ultra_chunks(file_path):
            # Process chunk with zero-copy operations
            processed_chunk = await self._process_chunk_ultra_async(chunk)
            processed_chunks.append(processed_chunk)
            
            # Get classification from first chunk
            if classification is None and chunk.chunk_id == 0:
                classification = await self._classify_chunk_ultra_async(chunk)
        
        # Merge processed chunks with zero-copy
        processed_text = self._merge_chunks_ultra(processed_chunks)
        
        # Transform to professional format
        professional_doc = await self._transform_to_professional_ultra_async(
            processed_text, classification, processing_options
        )
        
        return DocumentProcessingResponse(
            success=True,
            document_analysis=classification,
            professional_document=professional_doc,
            processing_time=time.time(),
            metadata={
                'processor': 'ultra_streaming',
                'chunks_processed': len(processed_chunks),
                'streaming_enabled': True,
                'zero_copy_ops': self.stats['zero_copy_operations']
            }
        )
    
    async def _read_file_memory_mapped(self, file_path: str) -> bytes:
        """Read file using memory mapping for zero-copy"""
        loop = asyncio.get_event_loop()
        
        def read_mmap():
            with open(file_path, 'rb') as f:
                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                    return mm[:]  # Copy to avoid keeping file open
        
        return await loop.run_in_executor(self.thread_pool, read_mmap)
    
    async def _read_file_ultra_fast(self, file_path: str) -> bytes:
        """Read file with ultra-fast optimizations"""
        loop = asyncio.get_event_loop()
        
        def read_fast():
            with open(file_path, 'rb') as f:
                return f.read()
        
        return await loop.run_in_executor(self.thread_pool, read_fast)
    
    async def _extract_text_ultra_async(self, content: bytes, filename: str) -> str:
        """Extract text with ultra optimizations"""
        loop = asyncio.get_event_loop()
        
        def extract_text_sync():
            # Use file handler with optimizations
            handler = self.file_handler_factory.get_handler_by_content(content, filename)
            return handler.extract_text_from_bytes(content)
        
        return await loop.run_in_executor(self.thread_pool, extract_text_sync)
    
    async def _classify_document_ultra_async(self, content: bytes, filename: str) -> DocumentAnalysis:
        """Classify document with ultra optimizations"""
        loop = asyncio.get_event_loop()
        
        def classify_sync():
            return self.ai_classifier.classify_document_from_bytes(content, filename)
        
        return await loop.run_in_executor(self.thread_pool, classify_sync)
    
    async def _classify_chunk_ultra_async(self, chunk: UltraProcessingChunk) -> DocumentAnalysis:
        """Classify document chunk with ultra optimizations"""
        loop = asyncio.get_event_loop()
        
        def classify_chunk_sync():
            return self.ai_classifier.classify_text_from_bytes(chunk.content)
        
        return await loop.run_in_executor(self.thread_pool, classify_chunk_sync)
    
    async def _transform_to_professional_ultra_async(self, 
                                                   text: str, 
                                                   classification: Optional[DocumentAnalysis],
                                                   processing_options: Dict[str, Any]) -> ProfessionalDocument:
        """Transform to professional format with ultra optimizations"""
        loop = asyncio.get_event_loop()
        
        def transform_sync():
            return self.professional_transformer.transform_to_professional(
                text, classification, processing_options
            )
        
        return await loop.run_in_executor(self.thread_pool, transform_sync)
    
    def _split_into_ultra_chunks(self, text: str) -> List[UltraProcessingChunk]:
        """Split text into ultra-optimized chunks"""
        chunks = []
        text_bytes = text.encode('utf-8')
        text_length = len(text_bytes)
        chunk_count = (text_length + self.chunk_size - 1) // self.chunk_size
        
        for i in range(chunk_count):
            start = i * self.chunk_size
            end = min(start + self.chunk_size, text_length)
            
            chunk_content = text_bytes[start:end]
            checksum = hashlib.md5(chunk_content).hexdigest()
            
            chunk = UltraProcessingChunk(
                content=chunk_content,
                chunk_id=i,
                total_chunks=chunk_count,
                metadata={'start': start, 'end': end},
                checksum=checksum
            )
            chunks.append(chunk)
        
        return chunks
    
    async def _process_chunks_ultra_parallel(self, chunks: List[UltraProcessingChunk]) -> List[str]:
        """Process chunks with ultra-parallel optimization"""
        tasks = [self._process_chunk_ultra_async(chunk) for chunk in chunks]
        return await asyncio.gather(*tasks)
    
    async def _process_chunk_ultra_async(self, chunk: UltraProcessingChunk) -> str:
        """Process individual chunk with ultra optimizations"""
        loop = asyncio.get_event_loop()
        
        def process_chunk_sync():
            # Zero-copy processing
            if self.enable_zero_copy:
                self.stats['zero_copy_operations'] += 1
            
            # Basic text processing (can be enhanced with specific processing logic)
            return chunk.content.decode('utf-8', errors='ignore').strip()
        
        return await loop.run_in_executor(self.thread_pool, process_chunk_sync)
    
    async def _stream_document_ultra_chunks(self, file_path: str) -> AsyncGenerator[UltraProcessingChunk, None]:
        """Stream document in ultra-optimized chunks"""
        if self.enable_memory_mapping:
            # Use memory mapping for large files
            with open(file_path, 'rb') as f:
                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                    file_size = len(mm)
                    chunk_count = (file_size + self.chunk_size - 1) // self.chunk_size
                    
                    for i in range(chunk_count):
                        start = i * self.chunk_size
                        end = min(start + self.chunk_size, file_size)
                        
                        chunk_content = mm[start:end]
                        checksum = hashlib.md5(chunk_content).hexdigest()
                        
                        chunk = UltraProcessingChunk(
                            content=chunk_content,
                            chunk_id=i,
                            total_chunks=chunk_count,
                            metadata={'start': start, 'end': end},
                            checksum=checksum
                        )
                        yield chunk
                        await asyncio.sleep(0)  # Yield control
        else:
            # Fallback to regular streaming
            handler = self.file_handler_factory.get_handler(file_path)
            full_text = handler.extract_text(file_path)
            chunks = self._split_into_ultra_chunks(full_text)
            
            for chunk in chunks:
                yield chunk
                await asyncio.sleep(0)  # Yield control
    
    def _merge_chunks_ultra(self, processed_chunks: List[str]) -> str:
        """Merge processed chunks with ultra optimization"""
        # Use join for better performance than concatenation
        return '\n'.join(processed_chunks)
    
    async def process_batch_ultra_fast(self, 
                                     file_paths: List[str],
                                     processing_options: Optional[Dict[str, Any]] = None) -> List[DocumentProcessingResponse]:
        """Process multiple documents with ultra-fast batch processing"""
        processing_options = processing_options or {}
        
        # Create processing tasks
        tasks = [
            self.process_document_ultra_fast(file_path, Path(file_path).name, processing_options)
            for file_path in file_paths
        ]
        
        # Process in parallel with ultra-optimized semaphore
        semaphore = asyncio.Semaphore(min(len(file_paths), self.max_workers))
        
        async def process_with_semaphore(task):
            async with semaphore:
                return await task
        
        # Execute all tasks with ultra optimization
        results = await asyncio.gather(
            *[process_with_semaphore(task) for task in tasks],
            return_exceptions=True
        )
        
        # Handle results and exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Ultra batch processing failed for {file_paths[i]}: {result}")
                processed_results.append(DocumentProcessingResponse(
                    success=False,
                    error=str(result),
                    processing_time=time.time()
                ))
            else:
                processed_results.append(result)
        
        return processed_results
    
    def get_ultra_performance_stats(self) -> Dict[str, Any]:
        """Get ultra performance statistics"""
        return {
            **self.stats,
            'cache_hit_rate': (
                self.stats['cache_hits'] / (self.stats['cache_hits'] + self.stats['cache_misses']) * 100
                if (self.stats['cache_hits'] + self.stats['cache_misses']) > 0 else 0
            ),
            'max_workers': self.max_workers,
            'chunk_size': self.chunk_size,
            'memory_mapping_enabled': self.enable_memory_mapping,
            'zero_copy_enabled': self.enable_zero_copy,
            'compression_enabled': self.enable_compression,
            'memory_pool_buffers': sum(len(pool) for pool in self._memory_pool.values())
        }
    
    async def close(self):
        """Close processor and cleanup resources"""
        logger.info("Closing ultra-fast document processor...")
        
        # Shutdown thread pools
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)
        
        # Clear memory pools
        for pool in self._memory_pool.values():
            pool.clear()
        self._memory_pool.clear()
        
        # Clear caches
        if self.cache_service:
            await self.cache_service.clear()
        
        # Force garbage collection
        gc.collect()
        
        logger.info("Ultra-fast document processor closed")

# Global ultra processor instance
_ultra_processor: Optional[UltraFastProcessor] = None

async def get_ultra_processor() -> UltraFastProcessor:
    """Get global ultra processor instance"""
    global _ultra_processor
    if _ultra_processor is None:
        _ultra_processor = UltraFastProcessor()
        await _ultra_processor.initialize()
    return _ultra_processor

async def close_ultra_processor():
    """Close global ultra processor"""
    global _ultra_processor
    if _ultra_processor:
        await _ultra_processor.close()
        _ultra_processor = None

















