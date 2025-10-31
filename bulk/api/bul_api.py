"""
BUL API - Perfect Integration with TruthGPT Bulk Processing
Seamless integration with existing bulk processing architecture
"""

import asyncio
import logging
import time
import json
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from dataclasses import dataclass
import threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np

# Import TruthGPT Bulk API components
from .truthgpt_bulk_api import (
    TruthGPTBulkAPI,
    TruthGPTBulkAPIConfig,
    create_truthgpt_bulk_api,
    create_truthgpt_bulk_api_config,
    create_truthgpt_api,
    create_high_performance_truthgpt_api,
    create_memory_efficient_truthgpt_api
)

logger = logging.getLogger(__name__)

@dataclass
class BULAPIConfig:
    """BUL API configuration."""
    
    # API settings
    api_version: str = "v1"
    base_url: str = "/api/bul"
    timeout: int = 300  # 5 minutes
    
    # TruthGPT settings
    model_name: str = "truthgpt-base"
    model_size: str = "medium"
    max_sequence_length: int = 4096
    
    # Bulk processing settings
    batch_size: int = 8
    max_batch_size: int = 32
    processing_timeout: int = 300
    
    # K/V Cache settings
    use_kv_cache: bool = True
    cache_size: int = 16384
    cache_strategy: str = "adaptive"
    compression_ratio: float = 0.5
    quantization_bits: int = 4
    
    # Performance settings
    use_mixed_precision: bool = True
    use_parallel_processing: bool = True
    num_workers: int = 8
    memory_strategy: str = "aggressive"
    
    # Monitoring
    enable_metrics: bool = True
    enable_profiling: bool = True
    log_level: str = "INFO"

class BULAPI:
    """
    BUL API for TruthGPT Bulk Processing.
    
    Features:
    - Perfect integration with existing bulk processing architecture
    - Ultra-adaptive K/V cache optimization
    - Bulk processing with automatic scaling
    - Real-time performance monitoring
    - Seamless API integration
    """
    
    def __init__(self, config: BULAPIConfig):
        self.config = config
        
        # Initialize TruthGPT Bulk API
        self._setup_truthgpt_api()
        
        # Initialize BUL API components
        self._setup_bul_api_components()
        
        # Initialize monitoring
        self._setup_monitoring()
        
        # State management
        self.active_sessions = {}
        self.request_queue = asyncio.Queue()
        self.response_cache = {}
        
        logger.info("BUL API initialized")
    
    def _setup_truthgpt_api(self):
        """Setup TruthGPT Bulk API."""
        # Create TruthGPT API configuration
        truthgpt_config = create_truthgpt_bulk_api_config(
            api_version=self.config.api_version,
            base_url=self.config.base_url,
            timeout=self.config.timeout,
            model_name=self.config.model_name,
            model_size=self.config.model_size,
            max_sequence_length=self.config.max_sequence_length,
            batch_size=self.config.batch_size,
            max_batch_size=self.config.max_batch_size,
            processing_timeout=self.config.processing_timeout,
            use_kv_cache=self.config.use_kv_cache,
            cache_size=self.config.cache_size,
            cache_strategy=self.config.cache_strategy,
            compression_ratio=self.config.compression_ratio,
            quantization_bits=self.config.quantization_bits,
            memory_strategy=self.config.memory_strategy,
            use_mixed_precision=self.config.use_mixed_precision,
            use_parallel_processing=self.config.use_parallel_processing,
            num_workers=self.config.num_workers,
            enable_metrics=self.config.enable_metrics,
            enable_profiling=self.config.enable_profiling
        )
        
        # Create TruthGPT API
        self.truthgpt_api = create_truthgpt_bulk_api(truthgpt_config)
        
        logger.info("TruthGPT Bulk API initialized for BUL API")
    
    def _setup_bul_api_components(self):
        """Setup BUL API components."""
        self.request_processor = BULRequestProcessor(
            timeout=self.config.timeout
        )
        
        self.response_formatter = BULResponseFormatter(
            api_version=self.config.api_version
        )
        
        self.session_manager = BULSessionManager()
        
        logger.info("BUL API components initialized")
    
    def _setup_monitoring(self):
        """Setup monitoring and metrics."""
        self.metrics_collector = BULMetricsCollector(
            enable_metrics=self.config.enable_metrics,
            enable_profiling=self.config.enable_profiling
        )
        
        logger.info("Monitoring components initialized")
    
    async def process_bulk_requests(self, requests: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process bulk requests through BUL API.
        
        Args:
            requests: List of request data
            
        Returns:
            API response with results
        """
        logger.info(f"Processing {len(requests)} bulk requests through BUL API")
        
        start_time = time.time()
        
        try:
            # Validate requests
            validated_requests = self._validate_requests(requests)
            
            if not validated_requests:
                return self._create_error_response("No valid requests provided")
            
            # Process requests through TruthGPT API
            results = await self.truthgpt_api.process_bulk_requests(validated_requests)
            
            # Format response
            response = self._format_bulk_response(results, start_time)
            
            # Update metrics
            self._update_bul_metrics(start_time, len(requests), results)
            
            logger.info(f"BUL API processing completed in {time.time() - start_time:.2f}s")
            return response
            
        except Exception as e:
            logger.error(f"Error in BUL API processing: {e}")
            return self._create_error_response(str(e))
    
    def _validate_requests(self, requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate and clean requests."""
        validated_requests = []
        
        for request in requests:
            if self._is_valid_request(request):
                validated_requests.append(request)
            else:
                logger.warning(f"Invalid request: {request}")
        
        return validated_requests
    
    def _is_valid_request(self, request: Dict[str, Any]) -> bool:
        """Check if request is valid."""
        required_fields = ['text']
        
        for field in required_fields:
            if field not in request:
                return False
        
        # Check text length
        text = request.get('text', '')
        if len(text) > self.config.max_sequence_length:
            return False
        
        return True
    
    def _format_bulk_response(self, results: Dict[str, Any], 
                            start_time: float) -> Dict[str, Any]:
        """Format bulk response."""
        processing_time = time.time() - start_time
        
        # Extract results from TruthGPT API response
        truthgpt_results = results.get('results', [])
        success_rate = results.get('success_rate', 0.0)
        
        # Format individual results
        formatted_results = []
        for result in truthgpt_results:
            if result.get('success', False):
                formatted_results.append({
                    'success': True,
                    'text': result.get('text', ''),
                    'session_id': result.get('session_id', ''),
                    'cached': result.get('cached', False),
                    'processing_time': result.get('processing_time', 0.0)
                })
            else:
                formatted_results.append({
                    'success': False,
                    'error': result.get('error', 'Unknown error'),
                    'processing_time': result.get('processing_time', 0.0)
                })
        
        return {
            'api_version': self.config.api_version,
            'success': True,
            'total_requests': len(truthgpt_results),
            'successful_requests': len([r for r in truthgpt_results if r.get('success', False)]),
            'success_rate': success_rate,
            'processing_time': processing_time,
            'results': formatted_results,
            'metadata': {
                'model_name': self.config.model_name,
                'model_size': self.config.model_size,
                'cache_enabled': self.config.use_kv_cache,
                'batch_size': self.config.batch_size,
                'bul_api': True
            }
        }
    
    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """Create error response."""
        return {
            'api_version': self.config.api_version,
            'success': False,
            'error': error_message,
            'results': []
        }
    
    def _update_bul_metrics(self, start_time: float, num_requests: int, 
                          results: Dict[str, Any]):
        """Update BUL API metrics."""
        if not self.config.enable_metrics:
            return
        
        processing_time = time.time() - start_time
        
        # Calculate metrics
        truthgpt_results = results.get('results', [])
        successful_results = [r for r in truthgpt_results if r.get('success', False)]
        success_rate = len(successful_results) / len(truthgpt_results) if truthgpt_results else 0.0
        throughput = num_requests / processing_time if processing_time > 0 else 0.0
        
        # Update metrics
        self.metrics_collector.update_bul_metrics({
            'processing_time': processing_time,
            'num_requests': num_requests,
            'success_rate': success_rate,
            'throughput': throughput,
            'avg_response_time': processing_time / num_requests if num_requests > 0 else 0.0
        })
    
    async def process_single_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single request through BUL API.
        
        Args:
            request: Request data
            
        Returns:
            API response
        """
        logger.info("Processing single request through BUL API")
        
        start_time = time.time()
        
        try:
            # Validate request
            if not self._is_valid_request(request):
                return self._create_error_response("Invalid request")
            
            # Process through TruthGPT API
            results = await self.truthgpt_api.process_single_request(request)
            
            if not results.get('success', False):
                return self._create_error_response(results.get('error', 'Unknown error'))
            
            # Format single response
            response = {
                'api_version': self.config.api_version,
                'success': True,
                'text': results.get('text', ''),
                'session_id': results.get('session_id', ''),
                'cached': results.get('cached', False),
                'processing_time': results.get('processing_time', 0.0),
                'metadata': {
                    'model_name': self.config.model_name,
                    'model_size': self.config.model_size,
                    'cache_enabled': self.config.use_kv_cache,
                    'bul_api': True
                }
            }
            
            # Update metrics
            self._update_bul_metrics(start_time, 1, results)
            
            return response
            
        except Exception as e:
            logger.error(f"Error in single request processing: {e}")
            return self._create_error_response(str(e))
    
    def get_bul_api_stats(self) -> Dict[str, Any]:
        """Get comprehensive BUL API statistics."""
        stats = {
            'bul_api_stats': self.metrics_collector.get_stats(),
            'truthgpt_api_stats': self.truthgpt_api.get_api_stats(),
            'active_sessions': len(self.active_sessions),
            'request_queue_size': self.request_queue.qsize(),
            'config': self.config.__dict__
        }
        
        return stats
    
    def clear_cache(self):
        """Clear all caches."""
        self.truthgpt_api.clear_cache()
        self.response_cache.clear()
        
        logger.info("All caches cleared")
    
    def shutdown(self):
        """Shutdown the BUL API gracefully."""
        logger.info("Shutting down BUL API")
        
        # Clear caches
        self.clear_cache()
        
        # Shutdown TruthGPT API
        self.truthgpt_api.shutdown()
        
        logger.info("BUL API shutdown complete")

class BULRequestProcessor:
    """BUL request processor for API requests."""
    
    def __init__(self, timeout: int):
        self.timeout = timeout
    
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single request."""
        # Implementation for request processing
        pass

class BULResponseFormatter:
    """BUL response formatter for API responses."""
    
    def __init__(self, api_version: str):
        self.api_version = api_version
    
    def format_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Format API response."""
        # Implementation for response formatting
        pass

class BULSessionManager:
    """BUL session manager for API sessions."""
    
    def __init__(self):
        self.sessions = {}
    
    def create_session(self, session_id: str) -> Dict[str, Any]:
        """Create a new session."""
        session = {
            'id': session_id,
            'created_at': time.time(),
            'last_used': time.time(),
            'request_count': 0
        }
        self.sessions[session_id] = session
        return session
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session by ID."""
        return self.sessions.get(session_id)
    
    def update_session(self, session_id: str, updates: Dict[str, Any]):
        """Update session."""
        if session_id in self.sessions:
            self.sessions[session_id].update(updates)

class BULMetricsCollector:
    """BUL metrics collector."""
    
    def __init__(self, enable_metrics: bool, enable_profiling: bool):
        self.enable_metrics = enable_metrics
        self.enable_profiling = enable_profiling
        self.metrics = {}
    
    def update_bul_metrics(self, metrics: Dict[str, Any]):
        """Update BUL API metrics."""
        if not self.enable_metrics:
            return
        
        for key, value in metrics.items():
            self.metrics[key] = value
    
    def get_stats(self) -> Dict[str, Any]:
        """Get collected statistics."""
        return self.metrics.copy()

# Factory functions
def create_bul_api(config: BULAPIConfig = None) -> BULAPI:
    """Create BUL API."""
    if config is None:
        config = BULAPIConfig()
    return BULAPI(config)

def create_bul_api_config(**kwargs) -> BULAPIConfig:
    """Create BUL API configuration."""
    return BULAPIConfig(**kwargs)

# TruthGPT integration helpers
def create_truthgpt_bul_api(model_name: str = "truthgpt-base", 
                           model_size: str = "medium") -> BULAPI:
    """Create BUL API optimized for TruthGPT."""
    config = create_bul_api_config(
        model_name=model_name,
        model_size=model_size,
        batch_size=8,
        max_batch_size=32,
        use_kv_cache=True,
        cache_size=16384,
        cache_strategy="adaptive",
        compression_ratio=0.5,
        quantization_bits=4,
        memory_strategy="aggressive",
        use_mixed_precision=True,
        use_parallel_processing=True,
        num_workers=8,
        enable_metrics=True,
        enable_profiling=True
    )
    
    return create_bul_api(config)

def create_high_performance_bul_api() -> BULAPI:
    """Create high-performance BUL API."""
    config = create_bul_api_config(
        model_name="truthgpt-large",
        model_size="large",
        batch_size=16,
        max_batch_size=64,
        use_kv_cache=True,
        cache_size=32768,
        cache_strategy="adaptive",
        compression_ratio=0.3,
        quantization_bits=8,
        memory_strategy="balanced",
        use_mixed_precision=True,
        use_parallel_processing=True,
        num_workers=16,
        enable_metrics=True,
        enable_profiling=True
    )
    
    return create_bul_api(config)

def create_memory_efficient_bul_api() -> BULAPI:
    """Create memory-efficient BUL API."""
    config = create_bul_api_config(
        model_name="truthgpt-base",
        model_size="medium",
        batch_size=4,
        max_batch_size=16,
        use_kv_cache=True,
        cache_size=8192,
        cache_strategy="compressed",
        compression_ratio=0.7,
        quantization_bits=4,
        memory_strategy="aggressive",
        use_mixed_precision=True,
        use_parallel_processing=True,
        num_workers=4,
        enable_metrics=True,
        enable_profiling=True
    )
    
    return create_bul_api(config)