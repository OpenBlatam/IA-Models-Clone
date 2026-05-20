"""
WebSocket Routes - Real-time content processing and updates
Following FastAPI best practices: functional programming, RORO pattern, async operations
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, Query
from fastapi.responses import HTMLResponse
import json

from ..core.real_time_processor import (
    handle_websocket_connection,
    submit_content_for_processing,
    get_processing_status,
    get_processor_metrics,
    initialize_processor,
    shutdown_processor
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/websocket", tags=["WebSocket"])


# WebSocket endpoint for real-time updates
@router.websocket("/realtime")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time content processing updates
    
    Message types:
    - subscribe_job: Subscribe to updates for a specific job
    - unsubscribe_job: Unsubscribe from job updates
    - get_metrics: Request current processor metrics
    - submit_content: Submit content for processing
    """
    
    await handle_websocket_connection(websocket)


# HTTP endpoints for WebSocket-related operations
@router.post("/submit")
async def submit_content_via_http(
    content_id: str,
    content: str,
    priority: int = Query(default=1, ge=1, le=10),
    websocket_connection: Optional[str] = Query(default=None, description="WebSocket connection ID")
) -> Dict[str, Any]:
    """
    Submit content for real-time processing via HTTP
    
    - **content_id**: Unique identifier for the content
    - **content**: Content text to process
    - **priority**: Processing priority (1-10, higher is more urgent)
    - **websocket_connection**: Optional WebSocket connection ID for real-time updates
    """
    
    try:
        # Validate input
        if not content_id.strip():
            return {"error": "content_id cannot be empty"}
        
        if not content.strip():
            return {"error": "content cannot be empty"}
        
        if len(content) > 100000:  # 100KB limit
            return {"error": "content too large (max 100KB)"}
        
        # Submit for processing
        job_id = await submit_content_for_processing(
            content_id=content_id,
            content=content,
            priority=priority
        )
        
        return {
            "job_id": job_id,
            "content_id": content_id,
            "status": "submitted",
            "priority": priority,
            "timestamp": datetime.now().isoformat(),
            "websocket_url": f"/api/v1/websocket/realtime?job_id={job_id}" if websocket_connection else None
        }
        
    except Exception as e:
        logger.error(f"Error submitting content: {e}")
        return {
            "error": "Failed to submit content for processing",
            "detail": str(e),
            "timestamp": datetime.now().isoformat()
        }


@router.get("/status/{job_id}")
async def get_job_status(job_id: str) -> Dict[str, Any]:
    """
    Get processing status for a specific job
    
    - **job_id**: Unique identifier for the processing job
    """
    
    try:
        status = await get_processing_status(job_id)
        
        if status is None:
            return {
                "error": "Job not found",
                "job_id": job_id,
                "timestamp": datetime.now().isoformat()
            }
        
        return {
            "job_id": job_id,
            "status": status["status"],
            "created_at": status["created_at"],
            "result": status.get("result"),
            "error": status.get("error"),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting job status: {e}")
        return {
            "error": "Failed to get job status",
            "detail": str(e),
            "timestamp": datetime.now().isoformat()
        }


@router.get("/metrics")
async def get_metrics() -> Dict[str, Any]:
    """
    Get real-time processor metrics
    
    Returns current processing statistics including:
    - Total jobs processed
    - Queue size
    - Active connections
    - Average processing time
    """
    
    try:
        metrics = await get_processor_metrics()
        
        return {
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        return {
            "error": "Failed to get metrics",
            "detail": str(e),
            "timestamp": datetime.now().isoformat()
        }


@router.get("/health")
async def websocket_health_check() -> Dict[str, Any]:
    """
    Health check for WebSocket service
    
    Verifies that the real-time processor is running and healthy
    """
    
    try:
        metrics = await get_processor_metrics()
        
        # Check if processor is healthy
        is_healthy = (
            metrics["active_connections"] >= 0 and
            metrics["queue_size"] >= 0 and
            metrics["total_jobs"] >= 0
        )
        
        return {
            "status": "healthy" if is_healthy else "unhealthy",
            "service": "websocket-realtime",
            "processor_running": True,
            "active_connections": metrics["active_connections"],
            "queue_size": metrics["queue_size"],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"WebSocket health check failed: {e}")
        return {
            "status": "unhealthy",
            "service": "websocket-realtime",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


@router.get("/demo")
async def websocket_demo_page() -> HTMLResponse:
    """
    Demo page for testing WebSocket functionality
    
    Returns an HTML page with JavaScript for testing real-time processing
    """
    
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Content Redundancy Detector - WebSocket Demo</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .container { max-width: 800px; margin: 0 auto; }
            .section { margin: 20px 0; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }
            .status { padding: 10px; margin: 10px 0; border-radius: 3px; }
            .success { background-color: #d4edda; color: #155724; }
            .error { background-color: #f8d7da; color: #721c24; }
            .info { background-color: #d1ecf1; color: #0c5460; }
            button { padding: 10px 20px; margin: 5px; cursor: pointer; }
            input, textarea { width: 100%; padding: 8px; margin: 5px 0; }
            #messages { height: 300px; overflow-y: scroll; border: 1px solid #ccc; padding: 10px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Content Redundancy Detector - WebSocket Demo</h1>
            
            <div class="section">
                <h2>Connection Status</h2>
                <div id="connection-status" class="status info">Disconnected</div>
                <button onclick="connect()">Connect</button>
                <button onclick="disconnect()">Disconnect</button>
            </div>
            
            <div class="section">
                <h2>Submit Content for Processing</h2>
                <input type="text" id="content-id" placeholder="Content ID" value="demo-content-1">
                <textarea id="content" placeholder="Enter content to analyze..." rows="4">This is a sample content for testing the real-time processing system. It contains multiple sentences and should be analyzed for redundancy and similarity.</textarea>
                <input type="number" id="priority" placeholder="Priority (1-10)" value="5" min="1" max="10">
                <button onclick="submitContent()">Submit for Processing</button>
            </div>
            
            <div class="section">
                <h2>Job Status</h2>
                <input type="text" id="job-id" placeholder="Job ID">
                <button onclick="getJobStatus()">Get Status</button>
                <div id="job-status" class="status info">No job selected</div>
            </div>
            
            <div class="section">
                <h2>Real-time Messages</h2>
                <button onclick="clearMessages()">Clear Messages</button>
                <div id="messages"></div>
            </div>
            
            <div class="section">
                <h2>Processor Metrics</h2>
                <button onclick="getMetrics()">Get Metrics</button>
                <div id="metrics" class="status info">Click to get metrics</div>
            </div>
        </div>

        <script>
            let ws = null;
            let currentJobId = null;

            function connect() {
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                const wsUrl = `${protocol}//${window.location.host}/api/v1/websocket/realtime`;
                
                ws = new WebSocket(wsUrl);
                
                ws.onopen = function(event) {
                    updateConnectionStatus('Connected', 'success');
                    addMessage('Connected to WebSocket server', 'success');
                };
                
                ws.onmessage = function(event) {
                    const data = JSON.parse(event.data);
                    addMessage(`Received: ${JSON.stringify(data, null, 2)}`, 'info');
                    
                    if (data.type === 'job_completed' && data.job_id === currentJobId) {
                        updateJobStatus(`Job completed: ${JSON.stringify(data.result, null, 2)}`, 'success');
                    } else if (data.type === 'job_failed' && data.job_id === currentJobId) {
                        updateJobStatus(`Job failed: ${data.error}`, 'error');
                    } else if (data.type === 'metrics_response') {
                        updateMetrics(JSON.stringify(data.metrics, null, 2));
                    }
                };
                
                ws.onclose = function(event) {
                    updateConnectionStatus('Disconnected', 'error');
                    addMessage('WebSocket connection closed', 'error');
                };
                
                ws.onerror = function(error) {
                    updateConnectionStatus('Error', 'error');
                    addMessage(`WebSocket error: ${error}`, 'error');
                };
            }

            function disconnect() {
                if (ws) {
                    ws.close();
                    ws = null;
                }
            }

            function submitContent() {
                const contentId = document.getElementById('content-id').value;
                const content = document.getElementById('content').value;
                const priority = parseInt(document.getElementById('priority').value);
                
                if (!contentId || !content) {
                    alert('Please provide both content ID and content');
                    return;
                }
                
                fetch('/api/v1/websocket/submit', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/x-www-form-urlencoded',
                    },
                    body: `content_id=${encodeURIComponent(contentId)}&content=${encodeURIComponent(content)}&priority=${priority}`
                })
                .then(response => response.json())
                .then(data => {
                    if (data.error) {
                        addMessage(`Error: ${data.error}`, 'error');
                    } else {
                        currentJobId = data.job_id;
                        document.getElementById('job-id').value = data.job_id;
                        addMessage(`Content submitted: ${JSON.stringify(data, null, 2)}`, 'success');
                        
                        // Subscribe to job updates
                        if (ws && ws.readyState === WebSocket.OPEN) {
                            ws.send(JSON.stringify({
                                type: 'subscribe_job',
                                job_id: data.job_id
                            }));
                        }
                    }
                })
                .catch(error => {
                    addMessage(`Error submitting content: ${error}`, 'error');
                });
            }

            function getJobStatus() {
                const jobId = document.getElementById('job-id').value;
                
                if (!jobId) {
                    alert('Please provide a job ID');
                    return;
                }
                
                fetch(`/api/v1/websocket/status/${jobId}`)
                .then(response => response.json())
                .then(data => {
                    if (data.error) {
                        updateJobStatus(`Error: ${data.error}`, 'error');
                    } else {
                        updateJobStatus(`Status: ${JSON.stringify(data, null, 2)}`, 'info');
                    }
                })
                .catch(error => {
                    updateJobStatus(`Error: ${error}`, 'error');
                });
            }

            function getMetrics() {
                fetch('/api/v1/websocket/metrics')
                .then(response => response.json())
                .then(data => {
                    if (data.error) {
                        updateMetrics(`Error: ${data.error}`);
                    } else {
                        updateMetrics(JSON.stringify(data.metrics, null, 2));
                    }
                })
                .catch(error => {
                    updateMetrics(`Error: ${error}`);
                });
            }

            function updateConnectionStatus(message, type) {
                const status = document.getElementById('connection-status');
                status.textContent = message;
                status.className = `status ${type}`;
            }

            function updateJobStatus(message, type) {
                const status = document.getElementById('job-status');
                status.textContent = message;
                status.className = `status ${type}`;
            }

            function updateMetrics(message) {
                const metrics = document.getElementById('metrics');
                metrics.textContent = message;
                metrics.className = 'status info';
            }

            function addMessage(message, type) {
                const messages = document.getElementById('messages');
                const messageDiv = document.createElement('div');
                messageDiv.className = `status ${type}`;
                messageDiv.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
                messages.appendChild(messageDiv);
                messages.scrollTop = messages.scrollHeight;
            }

            function clearMessages() {
                document.getElementById('messages').innerHTML = '';
            }

            // Auto-connect on page load
            window.onload = function() {
                connect();
            };
        </script>
    </body>
    </html>
    """
    
    return HTMLResponse(content=html_content)


# Startup and shutdown handlers
@router.on_event("startup")
async def startup_websocket_service():
    """Initialize WebSocket service on startup"""
    try:
        await initialize_processor()
        logger.info("WebSocket service initialized")
    except Exception as e:
        logger.error(f"Failed to initialize WebSocket service: {e}")


@router.on_event("shutdown")
async def shutdown_websocket_service():
    """Shutdown WebSocket service on shutdown"""
    try:
        await shutdown_processor()
        logger.info("WebSocket service shutdown")
    except Exception as e:
        logger.error(f"Failed to shutdown WebSocket service: {e}")