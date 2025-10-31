"""
Batch Processing and Analytics System
=====================================

Advanced batch processing system for document classification with analytics,
caching, and performance optimization.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time
from datetime import datetime, timedelta
from pathlib import Path
import hashlib
import sqlite3
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from queue import Queue, Empty
import threading

logger = logging.getLogger(__name__)

class ProcessingStatus(Enum):
    """Processing status for batch operations"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class BatchJob:
    """Individual batch job"""
    id: str
    query: str
    status: ProcessingStatus = ProcessingStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    processing_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BatchResult:
    """Batch processing result"""
    batch_id: str
    total_jobs: int
    completed_jobs: int
    failed_jobs: int
    processing_time: float
    results: List[Dict[str, Any]]
    errors: List[Dict[str, Any]]
    analytics: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

class BatchProcessor:
    """
    Advanced batch processor for document classification with analytics
    """
    
    def __init__(
        self, 
        classifier_engine,
        cache_dir: Optional[str] = None,
        max_workers: int = None,
        use_multiprocessing: bool = False
    ):
        """
        Initialize batch processor
        
        Args:
            classifier_engine: Document classifier engine instance
            cache_dir: Directory for caching results
            max_workers: Maximum number of worker threads/processes
            use_multiprocessing: Whether to use multiprocessing instead of threading
        """
        self.classifier_engine = classifier_engine
        self.cache_dir = Path(cache_dir) if cache_dir else Path(__file__).parent / "cache"
        self.cache_dir.mkdir(exist_ok=True)
        
        # Processing configuration
        self.max_workers = max_workers or min(32, (mp.cpu_count() or 1) + 4)
        self.use_multiprocessing = use_multiprocessing
        
        # Initialize database for job tracking
        self.db_path = self.cache_dir / "batch_jobs.db"
        self._init_database()
        
        # Cache for results
        self.result_cache = {}
        self.cache_ttl = timedelta(hours=24)
        
        # Analytics
        self.analytics = {
            "total_processed": 0,
            "total_failed": 0,
            "avg_processing_time": 0.0,
            "document_type_distribution": {},
            "confidence_distribution": [],
            "processing_methods": {},
            "cache_hit_rate": 0.0
        }
        
        # Performance monitoring
        self.performance_metrics = {
            "throughput": [],  # jobs per second
            "latency": [],     # average processing time
            "error_rate": [],  # error percentage
            "cache_hits": 0,
            "cache_misses": 0
        }
    
    def _init_database(self):
        """Initialize SQLite database for job tracking"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS batch_jobs (
                    id TEXT PRIMARY KEY,
                    query TEXT NOT NULL,
                    status TEXT NOT NULL,
                    result TEXT,
                    error TEXT,
                    created_at TIMESTAMP,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    processing_time REAL,
                    metadata TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS batch_results (
                    batch_id TEXT PRIMARY KEY,
                    total_jobs INTEGER,
                    completed_jobs INTEGER,
                    failed_jobs INTEGER,
                    processing_time REAL,
                    results TEXT,
                    errors TEXT,
                    analytics TEXT,
                    created_at TIMESTAMP
                )
            """)
    
    def process_batch(
        self, 
        queries: List[str], 
        batch_id: Optional[str] = None,
        use_cache: bool = True,
        progress_callback: Optional[callable] = None
    ) -> BatchResult:
        """
        Process a batch of document classification queries
        
        Args:
            queries: List of text queries to classify
            batch_id: Optional batch identifier
            use_cache: Whether to use cached results
            progress_callback: Optional callback for progress updates
            
        Returns:
            BatchResult with all processing results
        """
        if not batch_id:
            batch_id = f"batch_{int(time.time())}_{len(queries)}"
        
        logger.info(f"Starting batch processing: {batch_id} with {len(queries)} queries")
        
        start_time = time.time()
        
        # Create batch jobs
        jobs = []
        for i, query in enumerate(queries):
            job_id = f"{batch_id}_job_{i}"
            job = BatchJob(
                id=job_id,
                query=query,
                metadata={"batch_id": batch_id, "index": i}
            )
            jobs.append(job)
            self._save_job(job)
        
        # Process jobs
        results = []
        errors = []
        
        if self.use_multiprocessing and len(queries) > 10:
            # Use multiprocessing for large batches
            results, errors = self._process_with_multiprocessing(jobs, progress_callback)
        else:
            # Use threading for smaller batches
            results, errors = self._process_with_threading(jobs, use_cache, progress_callback)
        
        processing_time = time.time() - start_time
        
        # Generate analytics
        analytics = self._generate_analytics(results, errors, processing_time)
        
        # Create batch result
        batch_result = BatchResult(
            batch_id=batch_id,
            total_jobs=len(queries),
            completed_jobs=len(results),
            failed_jobs=len(errors),
            processing_time=processing_time,
            results=results,
            errors=errors,
            analytics=analytics
        )
        
        # Save batch result
        self._save_batch_result(batch_result)
        
        # Update analytics
        self._update_analytics(batch_result)
        
        logger.info(f"Batch processing completed: {batch_id} in {processing_time:.2f}s")
        
        return batch_result
    
    def _process_with_threading(
        self, 
        jobs: List[BatchJob], 
        use_cache: bool,
        progress_callback: Optional[callable]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Process jobs using threading"""
        results = []
        errors = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all jobs
            future_to_job = {
                executor.submit(self._process_single_job, job, use_cache): job 
                for job in jobs
            }
            
            # Collect results
            for future in future_to_job:
                job = future_to_job[future]
                try:
                    result = future.result()
                    if result.get("error"):
                        errors.append(result)
                    else:
                        results.append(result)
                    
                    # Update progress
                    if progress_callback:
                        progress = (len(results) + len(errors)) / len(jobs)
                        progress_callback(progress, job.id)
                        
                except Exception as e:
                    error_result = {
                        "job_id": job.id,
                        "query": job.query,
                        "error": str(e),
                        "timestamp": datetime.now().isoformat()
                    }
                    errors.append(error_result)
                    logger.error(f"Job {job.id} failed: {e}")
        
        return results, errors
    
    def _process_with_multiprocessing(
        self, 
        jobs: List[BatchJob], 
        progress_callback: Optional[callable]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Process jobs using multiprocessing"""
        results = []
        errors = []
        
        # Prepare job data for multiprocessing
        job_data = [(job.id, job.query) for job in jobs]
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all jobs
            future_to_job = {
                executor.submit(self._process_job_worker, job_id, query): job_id 
                for job_id, query in job_data
            }
            
            # Collect results
            for future in future_to_job:
                job_id = future_to_job[future]
                try:
                    result = future.result()
                    if result.get("error"):
                        errors.append(result)
                    else:
                        results.append(result)
                    
                    # Update progress
                    if progress_callback:
                        progress = (len(results) + len(errors)) / len(jobs)
                        progress_callback(progress, job_id)
                        
                except Exception as e:
                    error_result = {
                        "job_id": job_id,
                        "query": jobs[0].query,  # Fallback
                        "error": str(e),
                        "timestamp": datetime.now().isoformat()
                    }
                    errors.append(error_result)
                    logger.error(f"Job {job_id} failed: {e}")
        
        return results, errors
    
    def _process_single_job(self, job: BatchJob, use_cache: bool) -> Dict[str, Any]:
        """Process a single classification job"""
        job.started_at = datetime.now()
        job.status = ProcessingStatus.PROCESSING
        self._save_job(job)
        
        try:
            # Check cache first
            if use_cache:
                cached_result = self._get_cached_result(job.query)
                if cached_result:
                    self.performance_metrics["cache_hits"] += 1
                    result = {
                        "job_id": job.id,
                        "query": job.query,
                        "result": cached_result,
                        "cached": True,
                        "timestamp": datetime.now().isoformat()
                    }
                    job.status = ProcessingStatus.COMPLETED
                    job.result = json.dumps(result)
                    job.completed_at = datetime.now()
                    job.processing_time = 0.0
                    self._save_job(job)
                    return result
            
            self.performance_metrics["cache_misses"] += 1
            
            # Process with classifier
            start_time = time.time()
            classification_result = self.classifier_engine.classify_document(job.query)
            processing_time = time.time() - start_time
            
            # Prepare result
            result_data = {
                "document_type": classification_result.document_type.value,
                "confidence": classification_result.confidence,
                "keywords": classification_result.keywords,
                "reasoning": classification_result.reasoning,
                "template_suggestions": classification_result.template_suggestions
            }
            
            result = {
                "job_id": job.id,
                "query": job.query,
                "result": result_data,
                "cached": False,
                "processing_time": processing_time,
                "timestamp": datetime.now().isoformat()
            }
            
            # Cache result
            if use_cache:
                self._cache_result(job.query, result_data)
            
            # Update job
            job.status = ProcessingStatus.COMPLETED
            job.result = json.dumps(result)
            job.completed_at = datetime.now()
            job.processing_time = processing_time
            self._save_job(job)
            
            return result
            
        except Exception as e:
            error_result = {
                "job_id": job.id,
                "query": job.query,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            
            job.status = ProcessingStatus.FAILED
            job.error = str(e)
            job.completed_at = datetime.now()
            self._save_job(job)
            
            return error_result
    
    @staticmethod
    def _process_job_worker(job_id: str, query: str) -> Dict[str, Any]:
        """Worker function for multiprocessing"""
        try:
            # Note: In a real implementation, you'd need to pass the classifier
            # or recreate it in the worker process
            # For now, this is a placeholder
            return {
                "job_id": job_id,
                "query": query,
                "error": "Multiprocessing not fully implemented",
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "job_id": job_id,
                "query": query,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _get_cached_result(self, query: str) -> Optional[Dict[str, Any]]:
        """Get cached result for query"""
        query_hash = hashlib.md5(query.encode()).hexdigest()
        cache_file = self.cache_dir / f"cache_{query_hash}.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                
                # Check TTL
                cached_time = datetime.fromisoformat(cached_data['timestamp'])
                if datetime.now() - cached_time < self.cache_ttl:
                    return cached_data['result']
                else:
                    # Remove expired cache
                    cache_file.unlink()
            except Exception as e:
                logger.warning(f"Failed to read cache file: {e}")
        
        return None
    
    def _cache_result(self, query: str, result: Dict[str, Any]):
        """Cache classification result"""
        query_hash = hashlib.md5(query.encode()).hexdigest()
        cache_file = self.cache_dir / f"cache_{query_hash}.json"
        
        try:
            cache_data = {
                "query": query,
                "result": result,
                "timestamp": datetime.now().isoformat()
            }
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to cache result: {e}")
    
    def _generate_analytics(
        self, 
        results: List[Dict[str, Any]], 
        errors: List[Dict[str, Any]], 
        processing_time: float
    ) -> Dict[str, Any]:
        """Generate analytics from batch results"""
        analytics = {
            "total_processed": len(results),
            "total_failed": len(errors),
            "success_rate": len(results) / (len(results) + len(errors)) if (results or errors) else 0,
            "avg_processing_time": sum(r.get("processing_time", 0) for r in results) / len(results) if results else 0,
            "total_processing_time": processing_time,
            "throughput": len(results) / processing_time if processing_time > 0 else 0,
            "document_type_distribution": {},
            "confidence_distribution": [],
            "error_types": {}
        }
        
        # Document type distribution
        for result in results:
            if "result" in result:
                doc_type = result["result"].get("document_type", "unknown")
                analytics["document_type_distribution"][doc_type] = \
                    analytics["document_type_distribution"].get(doc_type, 0) + 1
        
        # Confidence distribution
        for result in results:
            if "result" in result:
                confidence = result["result"].get("confidence", 0)
                analytics["confidence_distribution"].append(confidence)
        
        # Error types
        for error in errors:
            error_type = error.get("error", "unknown")
            analytics["error_types"][error_type] = \
                analytics["error_types"].get(error_type, 0) + 1
        
        return analytics
    
    def _update_analytics(self, batch_result: BatchResult):
        """Update global analytics with batch result"""
        self.analytics["total_processed"] += batch_result.completed_jobs
        self.analytics["total_failed"] += batch_result.failed_jobs
        
        # Update document type distribution
        for doc_type, count in batch_result.analytics.get("document_type_distribution", {}).items():
            self.analytics["document_type_distribution"][doc_type] = \
                self.analytics["document_type_distribution"].get(doc_type, 0) + count
        
        # Update confidence distribution
        self.analytics["confidence_distribution"].extend(
            batch_result.analytics.get("confidence_distribution", [])
        )
        
        # Update cache hit rate
        total_requests = self.performance_metrics["cache_hits"] + self.performance_metrics["cache_misses"]
        if total_requests > 0:
            self.analytics["cache_hit_rate"] = self.performance_metrics["cache_hits"] / total_requests
    
    def _save_job(self, job: BatchJob):
        """Save job to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO batch_jobs 
                    (id, query, status, result, error, created_at, started_at, completed_at, processing_time, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    job.id, job.query, job.status.value, job.result, job.error,
                    job.created_at.isoformat(), 
                    job.started_at.isoformat() if job.started_at else None,
                    job.completed_at.isoformat() if job.completed_at else None,
                    job.processing_time, json.dumps(job.metadata)
                ))
        except Exception as e:
            logger.error(f"Failed to save job {job.id}: {e}")
    
    def _save_batch_result(self, batch_result: BatchResult):
        """Save batch result to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO batch_results 
                    (batch_id, total_jobs, completed_jobs, failed_jobs, processing_time, 
                     results, errors, analytics, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    batch_result.batch_id, batch_result.total_jobs, batch_result.completed_jobs,
                    batch_result.failed_jobs, batch_result.processing_time,
                    json.dumps(batch_result.results), json.dumps(batch_result.errors),
                    json.dumps(batch_result.analytics), batch_result.created_at.isoformat()
                ))
        except Exception as e:
            logger.error(f"Failed to save batch result {batch_result.batch_id}: {e}")
    
    def get_batch_status(self, batch_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a batch job"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT * FROM batch_results WHERE batch_id = ?
                """, (batch_id,))
                
                row = cursor.fetchone()
                if row:
                    return {
                        "batch_id": row[0],
                        "total_jobs": row[1],
                        "completed_jobs": row[2],
                        "failed_jobs": row[3],
                        "processing_time": row[4],
                        "created_at": row[8]
                    }
        except Exception as e:
            logger.error(f"Failed to get batch status: {e}")
        
        return None
    
    def get_analytics(self) -> Dict[str, Any]:
        """Get comprehensive analytics"""
        analytics = self.analytics.copy()
        
        # Add performance metrics
        analytics["performance"] = {
            "cache_hit_rate": self.analytics["cache_hit_rate"],
            "total_cache_hits": self.performance_metrics["cache_hits"],
            "total_cache_misses": self.performance_metrics["cache_misses"]
        }
        
        # Calculate additional metrics
        if self.analytics["confidence_distribution"]:
            confidences = self.analytics["confidence_distribution"]
            analytics["confidence_stats"] = {
                "mean": sum(confidences) / len(confidences),
                "min": min(confidences),
                "max": max(confidences),
                "median": sorted(confidences)[len(confidences) // 2]
            }
        
        return analytics
    
    def clear_cache(self):
        """Clear all cached results"""
        try:
            for cache_file in self.cache_dir.glob("cache_*.json"):
                cache_file.unlink()
            
            self.performance_metrics["cache_hits"] = 0
            self.performance_metrics["cache_misses"] = 0
            
            logger.info("Cache cleared successfully")
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
    
    def export_analytics(self, format: str = "json") -> str:
        """Export analytics in specified format"""
        analytics = self.get_analytics()
        
        if format == "json":
            return json.dumps(analytics, indent=2)
        elif format == "csv":
            # Convert to CSV format
            csv_lines = ["metric,value"]
            for key, value in analytics.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        csv_lines.append(f"{key}.{sub_key},{sub_value}")
                else:
                    csv_lines.append(f"{key},{value}")
            return "\n".join(csv_lines)
        else:
            return str(analytics)

# Example usage
if __name__ == "__main__":
    # This would be used with the actual classifier engine
    # from document_classifier_engine import DocumentClassifierEngine
    
    # classifier = DocumentClassifierEngine()
    # processor = BatchProcessor(classifier)
    
    # Test queries
    test_queries = [
        "I want to write a science fiction novel",
        "Create a service agreement contract",
        "Design a mobile app interface",
        "Write a business plan for startup",
        "Research paper on machine learning"
    ]
    
    # Process batch
    # result = processor.process_batch(test_queries)
    # print(f"Processed {result.completed_jobs} jobs in {result.processing_time:.2f}s")
    # print(f"Success rate: {result.analytics['success_rate']:.2%}")
    
    print("Batch processor initialized successfully")



























