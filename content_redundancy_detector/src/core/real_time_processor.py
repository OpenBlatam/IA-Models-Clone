"""
Real-time Content Processor - Functional approach for streaming content analysis
Following FastAPI best practices: functional programming, RORO pattern, async operations
"""

import asyncio
import logging
from collections import deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, AsyncGenerator, Callable, Set
from dataclasses import dataclass, asdict
import json
import uuid

import redis.asyncio as redis
from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)


@dataclass
class ProcessingJob:
    """Real-time processing job"""
    job_id: str
    content_id: str
    content: str
    priority: int = 1
    created_at: datetime
    status: str = "pending"
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@dataclass
class ProcessingMetrics:
    """Real-time processing metrics"""
    total_jobs: int
    completed_jobs: int
    failed_jobs: int
    average_processing_time: float
    queue_size: int
    active_connections: int
    last_updated: datetime


class ConnectionManager:
    """Manage WebSocket connections for real-time updates"""
    
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.connection_jobs: Dict[WebSocket, Set[str]] = {}
    
    async def connect(self, websocket: WebSocket) -> None:
        """Accept new WebSocket connection"""
        await websocket.accept()
        self.active_connections.add(websocket)
        self.connection_jobs[websocket] = set()
        logger.info(f"WebSocket connection established. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket) -> None:
        """Remove WebSocket connection"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            if websocket in self.connection_jobs:
                del self.connection_jobs[websocket]
            logger.info(f"WebSocket connection closed. Total connections: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: Dict[str, Any], websocket: WebSocket) -> None:
        """Send message to specific WebSocket connection"""
        try:
            await websocket.send_text(json.dumps(message, default=str))
        except Exception as e:
            logger.error(f"Error sending message to WebSocket: {e}")
            self.disconnect(websocket)
    
    async def broadcast(self, message: Dict[str, Any]) -> None:
        """Broadcast message to all active connections"""
        if not self.active_connections:
            return
        
        disconnected = set()
        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(message, default=str))
            except Exception as e:
                logger.error(f"Error broadcasting to WebSocket: {e}")
                disconnected.add(connection)
        
        # Remove disconnected connections
        for connection in disconnected:
            self.disconnect(connection)
    
    def add_job_to_connection(self, websocket: WebSocket, job_id: str) -> None:
        """Add job to connection tracking"""
        if websocket in self.connection_jobs:
            self.connection_jobs[websocket].add(job_id)
    
    def remove_job_from_connection(self, websocket: WebSocket, job_id: str) -> None:
        """Remove job from connection tracking"""
        if websocket in self.connection_jobs:
            self.connection_jobs[websocket].discard(job_id)


# Global connection manager
connection_manager = ConnectionManager()


class RealTimeProcessor:
    """Real-time content processing engine"""
    
    def __init__(self, max_queue_size: int = 1000, max_workers: int = 10):
        self.max_queue_size = max_queue_size
        self.max_workers = max_workers
        self.job_queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue_size)
        self.processing_jobs: Dict[str, ProcessingJob] = {}
        self.completed_jobs: deque = deque(maxlen=1000)  # Keep last 1000 completed jobs
        self.metrics = ProcessingMetrics(
            total_jobs=0,
            completed_jobs=0,
            failed_jobs=0,
            average_processing_time=0.0,
            queue_size=0,
            active_connections=0,
            last_updated=datetime.now()
        )
        self.redis_client: Optional[redis.Redis] = None
        self.is_running = False
        self.workers: List[asyncio.Task] = []
    
    async def initialize(self) -> None:
        """Initialize the real-time processor"""
        try:
            # Initialize Redis connection
            self.redis_client = redis.Redis(
                host="localhost", 
                port=6379, 
                db=1, 
                decode_responses=True
            )
            await self.redis_client.ping()
            logger.info("Real-time processor initialized with Redis")
        except Exception as e:
            logger.warning(f"Redis not available, using in-memory processing: {e}")
            self.redis_client = None
    
    async def start(self) -> None:
        """Start the real-time processor"""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start worker tasks
        for i in range(self.max_workers):
            worker = asyncio.create_task(self._worker(f"worker-{i}"))
            self.workers.append(worker)
        
        # Start metrics updater
        metrics_task = asyncio.create_task(self._update_metrics())
        self.workers.append(metrics_task)
        
        logger.info(f"Real-time processor started with {self.max_workers} workers")
    
    async def stop(self) -> None:
        """Stop the real-time processor"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Cancel all workers
        for worker in self.workers:
            worker.cancel()
        
        # Wait for workers to finish
        await asyncio.gather(*self.workers, return_exceptions=True)
        self.workers.clear()
        
        # Close Redis connection
        if self.redis_client:
            await self.redis_client.close()
        
        logger.info("Real-time processor stopped")
    
    async def submit_job(
        self, 
        content_id: str, 
        content: str, 
        priority: int = 1,
        websocket: Optional[WebSocket] = None
    ) -> str:
        """Submit a new processing job"""
        
        if self.job_queue.full():
            raise Exception("Processing queue is full")
        
        job_id = str(uuid.uuid4())
        job = ProcessingJob(
            job_id=job_id,
            content_id=content_id,
            content=content,
            priority=priority,
            created_at=datetime.now()
        )
        
        # Store job
        self.processing_jobs[job_id] = job
        
        # Add to queue
        await self.job_queue.put((priority, job))
        
        # Track job for WebSocket connection
        if websocket:
            connection_manager.add_job_to_connection(websocket, job_id)
        
        # Update metrics
        self.metrics.total_jobs += 1
        
        logger.info(f"Job {job_id} submitted for content {content_id}")
        return job_id
    
    async def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job status and result"""
        
        # Check active jobs
        if job_id in self.processing_jobs:
            job = self.processing_jobs[job_id]
            return {
                "job_id": job_id,
                "status": job.status,
                "created_at": job.created_at.isoformat(),
                "result": job.result,
                "error": job.error
            }
        
        # Check completed jobs
        for job in self.completed_jobs:
            if job.job_id == job_id:
                return {
                    "job_id": job_id,
                    "status": job.status,
                    "created_at": job.created_at.isoformat(),
                    "result": job.result,
                    "error": job.error
                }
        
        return None
    
    async def _worker(self, worker_name: str) -> None:
        """Worker task for processing jobs"""
        logger.info(f"Worker {worker_name} started")
        
        while self.is_running:
            try:
                # Get job from queue with timeout
                priority, job = await asyncio.wait_for(
                    self.job_queue.get(), 
                    timeout=1.0
                )
                
                # Process job
                await self._process_job(job)
                
                # Mark task as done
                self.job_queue.task_done()
                
            except asyncio.TimeoutError:
                # No job available, continue
                continue
            except Exception as e:
                logger.error(f"Worker {worker_name} error: {e}")
                await asyncio.sleep(1)
        
        logger.info(f"Worker {worker_name} stopped")
    
    async def _process_job(self, job: ProcessingJob) -> None:
        """Process a single job"""
        start_time = datetime.now()
        
        try:
            # Update job status
            job.status = "processing"
            
            # Perform content analysis
            result = await self._analyze_content(job.content)
            
            # Update job with result
            job.result = result
            job.status = "completed"
            
            # Move to completed jobs
            self.completed_jobs.append(job)
            
            # Remove from active jobs
            if job.job_id in self.processing_jobs:
                del self.processing_jobs[job.job_id]
            
            # Update metrics
            processing_time = (datetime.now() - start_time).total_seconds()
            self.metrics.completed_jobs += 1
            
            # Broadcast completion
            await self._broadcast_job_completion(job)
            
            logger.info(f"Job {job.job_id} completed in {processing_time:.2f}s")
            
        except Exception as e:
            # Handle job failure
            job.status = "failed"
            job.error = str(e)
            self.metrics.failed_jobs += 1
            
            # Broadcast failure
            await self._broadcast_job_failure(job)
            
            logger.error(f"Job {job.job_id} failed: {e}")
    
    async def _analyze_content(self, content: str) -> Dict[str, Any]:
        """Analyze content (placeholder for actual analysis)"""
        # Simulate processing time
        await asyncio.sleep(0.1)
        
        # Basic content analysis
        words = content.split()
        sentences = content.split('.')
        
        return {
            "word_count": len(words),
            "character_count": len(content),
            "sentence_count": len([s for s in sentences if s.strip()]),
            "unique_words": len(set(words)),
            "average_word_length": sum(len(word) for word in words) / len(words) if words else 0,
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    async def _broadcast_job_completion(self, job: ProcessingJob) -> None:
        """Broadcast job completion to relevant connections"""
        message = {
            "type": "job_completed",
            "job_id": job.job_id,
            "content_id": job.content_id,
            "result": job.result,
            "timestamp": datetime.now().isoformat()
        }
        
        await connection_manager.broadcast(message)
    
    async def _broadcast_job_failure(self, job: ProcessingJob) -> None:
        """Broadcast job failure to relevant connections"""
        message = {
            "type": "job_failed",
            "job_id": job.job_id,
            "content_id": job.content_id,
            "error": job.error,
            "timestamp": datetime.now().isoformat()
        }
        
        await connection_manager.broadcast(message)
    
    async def _update_metrics(self) -> None:
        """Update processing metrics periodically"""
        while self.is_running:
            try:
                # Calculate average processing time
                if self.completed_jobs:
                    total_time = sum(
                        (job.created_at - datetime.now()).total_seconds() 
                        for job in self.completed_jobs[-100:]  # Last 100 jobs
                    )
                    self.metrics.average_processing_time = abs(total_time) / min(100, len(self.completed_jobs))
                
                # Update queue size
                self.metrics.queue_size = self.job_queue.qsize()
                
                # Update active connections
                self.metrics.active_connections = len(connection_manager.active_connections)
                
                # Update timestamp
                self.metrics.last_updated = datetime.now()
                
                # Broadcast metrics update
                await connection_manager.broadcast({
                    "type": "metrics_update",
                    "metrics": asdict(self.metrics),
                    "timestamp": datetime.now().isoformat()
                })
                
                await asyncio.sleep(5)  # Update every 5 seconds
                
            except Exception as e:
                logger.error(f"Error updating metrics: {e}")
                await asyncio.sleep(5)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current processing metrics"""
        return asdict(self.metrics)


# Global processor instance
processor = RealTimeProcessor()


async def initialize_processor() -> None:
    """Initialize the global processor"""
    await processor.initialize()
    await processor.start()


async def shutdown_processor() -> None:
    """Shutdown the global processor"""
    await processor.stop()


async def submit_content_for_processing(
    content_id: str, 
    content: str, 
    priority: int = 1,
    websocket: Optional[WebSocket] = None
) -> str:
    """Submit content for real-time processing"""
    return await processor.submit_job(content_id, content, priority, websocket)


async def get_processing_status(job_id: str) -> Optional[Dict[str, Any]]:
    """Get processing status for a job"""
    return await processor.get_job_status(job_id)


async def get_processor_metrics() -> Dict[str, Any]:
    """Get processor metrics"""
    return processor.get_metrics()


async def handle_websocket_connection(websocket: WebSocket) -> None:
    """Handle WebSocket connection for real-time updates"""
    await connection_manager.connect(websocket)
    
    try:
        while True:
            # Keep connection alive and handle incoming messages
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Handle different message types
            if message.get("type") == "subscribe_job":
                job_id = message.get("job_id")
                if job_id:
                    connection_manager.add_job_to_connection(websocket, job_id)
                    await connection_manager.send_personal_message({
                        "type": "subscription_confirmed",
                        "job_id": job_id,
                        "timestamp": datetime.now().isoformat()
                    }, websocket)
            
            elif message.get("type") == "unsubscribe_job":
                job_id = message.get("job_id")
                if job_id:
                    connection_manager.remove_job_from_connection(websocket, job_id)
                    await connection_manager.send_personal_message({
                        "type": "unsubscription_confirmed",
                        "job_id": job_id,
                        "timestamp": datetime.now().isoformat()
                    }, websocket)
            
            elif message.get("type") == "get_metrics":
                metrics = await get_processor_metrics()
                await connection_manager.send_personal_message({
                    "type": "metrics_response",
                    "metrics": metrics,
                    "timestamp": datetime.now().isoformat()
                }, websocket)
    
    except WebSocketDisconnect:
        connection_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        connection_manager.disconnect(websocket)




