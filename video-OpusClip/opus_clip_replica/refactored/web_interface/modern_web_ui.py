"""
Modern Web Interface for Refactored Opus Clip

Advanced web interface with:
- Real-time job monitoring
- Interactive video analysis
- Performance dashboards
- User management
- File upload and management
- Real-time notifications
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List
import asyncio
import json
import uuid
from datetime import datetime
from pathlib import Path
import structlog
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Form, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel
import aiofiles

logger = structlog.get_logger("web_interface")

# Initialize FastAPI app
app = FastAPI(
    title="Opus Clip Web Interface",
    description="Modern web interface for Opus Clip video processing",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.user_connections: Dict[str, List[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, user_id: str = None):
        await websocket.accept()
        self.active_connections.append(websocket)
        if user_id:
            if user_id not in self.user_connections:
                self.user_connections[user_id] = []
            self.user_connections[user_id].append(websocket)
    
    def disconnect(self, websocket: WebSocket, user_id: str = None):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        if user_id and user_id in self.user_connections:
            if websocket in self.user_connections[user_id]:
                self.user_connections[user_id].remove(websocket)
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)
    
    async def send_to_user(self, message: str, user_id: str):
        if user_id in self.user_connections:
            for connection in self.user_connections[user_id]:
                try:
                    await connection.send_text(message)
                except:
                    self.disconnect(connection, user_id)
    
    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                self.active_connections.remove(connection)

manager = ConnectionManager()

# Pydantic models
class JobStatusUpdate(BaseModel):
    job_id: str
    status: str
    progress: float
    message: str
    timestamp: datetime

class VideoUploadRequest(BaseModel):
    filename: str
    size: int
    content_type: str

class AnalysisRequest(BaseModel):
    video_id: str
    max_clips: int = 10
    min_duration: float = 3.0
    max_duration: float = 30.0
    priority: str = "normal"

# Global state
uploaded_videos: Dict[str, Dict[str, Any]] = {}
active_jobs: Dict[str, Dict[str, Any]] = {}

@app.get("/", response_class=HTMLResponse)
async def get_homepage(request: Request):
    """Serve the main web interface."""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "title": "Opus Clip - AI Video Processing",
        "version": "1.0.0"
    })

@app.get("/dashboard", response_class=HTMLResponse)
async def get_dashboard(request: Request):
    """Serve the dashboard page."""
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "title": "Dashboard - Opus Clip",
        "active_jobs": len(active_jobs),
        "uploaded_videos": len(uploaded_videos)
    })

@app.get("/analytics", response_class=HTMLResponse)
async def get_analytics(request: Request):
    """Serve the analytics page."""
    return templates.TemplateResponse("analytics.html", {
        "request": request,
        "title": "Analytics - Opus Clip"
    })

@app.get("/api/status")
async def get_system_status():
    """Get system status."""
    return {
        "status": "online",
        "timestamp": datetime.now().isoformat(),
        "active_connections": len(manager.active_connections),
        "active_jobs": len(active_jobs),
        "uploaded_videos": len(uploaded_videos)
    }

@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    """Upload video file."""
    try:
        # Generate unique video ID
        video_id = str(uuid.uuid4())
        
        # Create upload directory
        upload_dir = Path("uploads")
        upload_dir.mkdir(exist_ok=True)
        
        # Save file
        file_path = upload_dir / f"{video_id}_{file.filename}"
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        # Store video metadata
        uploaded_videos[video_id] = {
            "id": video_id,
            "filename": file.filename,
            "size": len(content),
            "content_type": file.content_type,
            "path": str(file_path),
            "uploaded_at": datetime.now().isoformat(),
            "status": "uploaded"
        }
        
        # Broadcast upload notification
        await manager.broadcast(json.dumps({
            "type": "video_uploaded",
            "video_id": video_id,
            "filename": file.filename,
            "size": len(content)
        }))
        
        return {
            "success": True,
            "video_id": video_id,
            "filename": file.filename,
            "size": len(content),
            "message": "Video uploaded successfully"
        }
        
    except Exception as e:
        logger.error(f"Video upload failed: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.post("/api/analyze")
async def analyze_video(request: AnalysisRequest):
    """Start video analysis."""
    try:
        if request.video_id not in uploaded_videos:
            return JSONResponse(
                status_code=404,
                content={"success": False, "error": "Video not found"}
            )
        
        # Generate job ID
        job_id = str(uuid.uuid4())
        
        # Create analysis job
        active_jobs[job_id] = {
            "id": job_id,
            "video_id": request.video_id,
            "type": "analysis",
            "status": "pending",
            "progress": 0.0,
            "created_at": datetime.now().isoformat(),
            "max_clips": request.max_clips,
            "min_duration": request.min_duration,
            "max_duration": request.max_duration,
            "priority": request.priority
        }
        
        # Simulate analysis process (in real implementation, this would call the refactored API)
        asyncio.create_task(simulate_analysis_process(job_id))
        
        # Broadcast job creation
        await manager.broadcast(json.dumps({
            "type": "job_created",
            "job_id": job_id,
            "video_id": request.video_id,
            "status": "pending"
        }))
        
        return {
            "success": True,
            "job_id": job_id,
            "message": "Analysis job created successfully"
        }
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

async def simulate_analysis_process(job_id: str):
    """Simulate video analysis process."""
    try:
        job = active_jobs[job_id]
        
        # Update status to running
        job["status"] = "running"
        await manager.broadcast(json.dumps({
            "type": "job_update",
            "job_id": job_id,
            "status": "running",
            "progress": 0.0
        }))
        
        # Simulate processing steps
        steps = [
            ("Loading video", 10),
            ("Extracting frames", 25),
            ("Analyzing content", 50),
            ("Detecting faces", 70),
            ("Calculating engagement", 85),
            ("Generating segments", 95),
            ("Finalizing results", 100)
        ]
        
        for step_name, progress in steps:
            await asyncio.sleep(1)  # Simulate processing time
            
            job["progress"] = progress
            await manager.broadcast(json.dumps({
                "type": "job_update",
                "job_id": job_id,
                "status": "running",
                "progress": progress,
                "current_step": step_name
            }))
        
        # Mark as completed
        job["status"] = "completed"
        job["completed_at"] = datetime.now().isoformat()
        job["result"] = {
            "segments": [
                {
                    "id": f"segment_{i}",
                    "start_time": i * 5.0,
                    "end_time": (i + 1) * 5.0,
                    "duration": 5.0,
                    "engagement_score": 0.8 - (i * 0.1),
                    "title": f"Engaging Segment {i+1}"
                }
                for i in range(5)
            ],
            "total_segments": 5,
            "video_duration": 30.0
        }
        
        await manager.broadcast(json.dumps({
            "type": "job_completed",
            "job_id": job_id,
            "status": "completed",
            "progress": 100.0,
            "result": job["result"]
        }))
        
    except Exception as e:
        logger.error(f"Analysis simulation failed: {e}")
        job["status"] = "failed"
        job["error"] = str(e)
        await manager.broadcast(json.dumps({
            "type": "job_failed",
            "job_id": job_id,
            "status": "failed",
            "error": str(e)
        }))

@app.get("/api/jobs")
async def get_jobs():
    """Get all jobs."""
    return {
        "success": True,
        "jobs": list(active_jobs.values()),
        "total": len(active_jobs)
    }

@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    """Get specific job."""
    if job_id not in active_jobs:
        return JSONResponse(
            status_code=404,
            content={"success": False, "error": "Job not found"}
        )
    
    return {
        "success": True,
        "job": active_jobs[job_id]
    }

@app.delete("/api/jobs/{job_id}")
async def cancel_job(job_id: str):
    """Cancel a job."""
    if job_id not in active_jobs:
        return JSONResponse(
            status_code=404,
            content={"success": False, "error": "Job not found"}
        )
    
    job = active_jobs[job_id]
    if job["status"] in ["completed", "failed", "cancelled"]:
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": "Job cannot be cancelled"}
        )
    
    job["status"] = "cancelled"
    job["cancelled_at"] = datetime.now().isoformat()
    
    await manager.broadcast(json.dumps({
        "type": "job_cancelled",
        "job_id": job_id,
        "status": "cancelled"
    }))
    
    return {
        "success": True,
        "message": "Job cancelled successfully"
    }

@app.get("/api/videos")
async def get_videos():
    """Get all uploaded videos."""
    return {
        "success": True,
        "videos": list(uploaded_videos.values()),
        "total": len(uploaded_videos)
    }

@app.get("/api/videos/{video_id}")
async def get_video(video_id: str):
    """Get specific video."""
    if video_id not in uploaded_videos:
        return JSONResponse(
            status_code=404,
            content={"success": False, "error": "Video not found"}
        )
    
    return {
        "success": True,
        "video": uploaded_videos[video_id]
    }

@app.delete("/api/videos/{video_id}")
async def delete_video(video_id: str):
    """Delete video."""
    if video_id not in uploaded_videos:
        return JSONResponse(
            status_code=404,
            content={"success": False, "error": "Video not found"}
        )
    
    video = uploaded_videos[video_id]
    
    # Delete file
    try:
        Path(video["path"]).unlink()
    except:
        pass
    
    # Remove from storage
    del uploaded_videos[video_id]
    
    await manager.broadcast(json.dumps({
        "type": "video_deleted",
        "video_id": video_id
    }))
    
    return {
        "success": True,
        "message": "Video deleted successfully"
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time communication."""
    user_id = str(uuid.uuid4())
    await manager.connect(websocket, user_id)
    
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket, user_id)

@app.get("/api/analytics/performance")
async def get_performance_analytics():
    """Get performance analytics."""
    # Simulate performance data
    return {
        "success": True,
        "data": {
            "cpu_usage": 45.2,
            "memory_usage": 67.8,
            "disk_usage": 23.4,
            "active_jobs": len(active_jobs),
            "completed_jobs": len([j for j in active_jobs.values() if j["status"] == "completed"]),
            "failed_jobs": len([j for j in active_jobs.values() if j["status"] == "failed"]),
            "average_processing_time": 12.5,
            "throughput_per_hour": 24.3
        }
    }

@app.get("/api/analytics/jobs")
async def get_job_analytics():
    """Get job analytics."""
    jobs = list(active_jobs.values())
    
    # Calculate analytics
    total_jobs = len(jobs)
    completed_jobs = len([j for j in jobs if j["status"] == "completed"])
    failed_jobs = len([j for j in jobs if j["status"] == "failed"])
    running_jobs = len([j for j in jobs if j["status"] == "running"])
    
    success_rate = (completed_jobs / total_jobs * 100) if total_jobs > 0 else 0
    
    return {
        "success": True,
        "data": {
            "total_jobs": total_jobs,
            "completed_jobs": completed_jobs,
            "failed_jobs": failed_jobs,
            "running_jobs": running_jobs,
            "success_rate": success_rate,
            "jobs_by_status": {
                "completed": completed_jobs,
                "failed": failed_jobs,
                "running": running_jobs,
                "pending": len([j for j in jobs if j["status"] == "pending"])
            }
        }
    }

# Create templates directory and files
def create_templates():
    """Create HTML templates."""
    templates_dir = Path("templates")
    templates_dir.mkdir(exist_ok=True)
    
    # Create static directory
    static_dir = Path("static")
    static_dir.mkdir(exist_ok=True)
    
    # Create CSS directory
    css_dir = static_dir / "css"
    css_dir.mkdir(exist_ok=True)
    
    # Create JS directory
    js_dir = static_dir / "js"
    js_dir.mkdir(exist_ok=True)

if __name__ == "__main__":
    create_templates()
    uvicorn.run(
        "modern_web_ui:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info"
    )


