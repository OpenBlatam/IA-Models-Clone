"""
Gamma App - Real Improvement API
REST API for managing real improvements that actually work
"""

import asyncio
import logging
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import uuid
from pathlib import Path
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

logger = logging.getLogger(__name__)

# Pydantic models for API
class ImprovementCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=255)
    description: str = Field(..., min_length=1)
    category: str = Field(..., regex="^(performance|security|user_experience|reliability|maintainability|cost_optimization)$")
    priority: str = Field(..., regex="^(low|medium|high|critical)$")
    effort_hours: float = Field(..., gt=0, le=100)
    impact_score: int = Field(..., ge=1, le=10)
    implementation_steps: List[str] = Field(..., min_items=1)
    code_examples: List[str] = Field(..., min_items=1)
    testing_notes: str = Field(..., min_length=1)

class ImprovementUpdate(BaseModel):
    title: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = Field(None, min_length=1)
    status: Optional[str] = Field(None, regex="^(pending|in_progress|completed|testing|deployed)$")
    notes: Optional[str] = Field(None)

class TaskCreate(BaseModel):
    improvement_id: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1, max_length=255)
    description: str = Field(..., min_length=1)
    improvement_type: str = Field(..., regex="^(code_optimization|security_enhancement|performance_boost|bug_fix|feature_addition)$")
    priority: int = Field(..., ge=1, le=10)
    estimated_hours: float = Field(..., gt=0, le=100)
    assigned_to: Optional[str] = Field(None, max_length=100)

class PlanCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=255)
    description: str = Field(..., min_length=1)
    priority: str = Field(..., regex="^(low|medium|high|critical)$")
    estimated_duration: float = Field(..., gt=0, le=1000)
    improvements: List[str] = Field(..., min_items=1)
    dependencies: Optional[List[str]] = Field(None, default=[])

class ExecutionRequest(BaseModel):
    improvement_id: str = Field(..., min_length=1)
    execution_type: str = Field(..., regex="^(automated|manual|hybrid)$")
    dry_run: bool = Field(False)

class AnalysisRequest(BaseModel):
    analysis_type: Optional[str] = Field(None, regex="^(code_quality|performance|security|maintainability|testing|documentation)$")
    include_recommendations: bool = Field(True)

class DashboardResponse(BaseModel):
    timestamp: str
    metrics: Dict[str, Any]
    alerts: List[Dict[str, Any]]
    system_status: Dict[str, Any]

class RealImprovementAPI:
    """
    REST API for managing real improvements
    """
    
    def __init__(self, project_root: str = "."):
        """Initialize improvement API"""
        self.project_root = Path(project_root)
        self.app = FastAPI(
            title="Gamma App - Real Improvements API",
            description="API for managing real improvements that actually work",
            version="1.0.0"
        )
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Setup routes
        self._setup_routes()
        
        logger.info(f"Real Improvement API initialized for {self.project_root}")
    
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/")
        async def root():
            return {"message": "Gamma App - Real Improvements API", "version": "1.0.0"}
        
        @self.app.get("/health")
        async def health_check():
            return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}
        
        # Improvement routes
        @self.app.post("/improvements/")
        async def create_improvement(improvement: ImprovementCreate):
            """Create new improvement"""
            try:
                # Import here to avoid circular imports
                from real_improvements_engine import get_real_improvements_engine
                
                engine = get_real_improvements_engine()
                improvement_id = engine.create_improvement(
                    title=improvement.title,
                    description=improvement.description,
                    category=improvement.category,
                    priority=improvement.priority,
                    effort_hours=improvement.effort_hours,
                    impact_score=improvement.impact_score,
                    implementation_steps=improvement.implementation_steps,
                    code_examples=improvement.code_examples,
                    testing_notes=improvement.testing_notes
                )
                
                return {"improvement_id": improvement_id, "status": "created"}
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/improvements/")
        async def get_improvements():
            """Get all improvements"""
            try:
                from real_improvements_engine import get_real_improvements_engine
                
                engine = get_real_improvements_engine()
                improvements = engine.export_improvements()
                
                return improvements
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/improvements/{improvement_id}")
        async def get_improvement(improvement_id: str):
            """Get specific improvement"""
            try:
                from real_improvements_engine import get_real_improvements_engine
                
                engine = get_real_improvements_engine()
                if improvement_id in engine.improvements:
                    improvement = engine.improvements[improvement_id]
                    return {
                        "improvement_id": improvement.improvement_id,
                        "title": improvement.title,
                        "description": improvement.description,
                        "category": improvement.category.value,
                        "priority": improvement.priority.value,
                        "effort_hours": improvement.effort_hours,
                        "impact_score": improvement.impact_score,
                        "status": improvement.status,
                        "created_at": improvement.created_at.isoformat(),
                        "completed_at": improvement.completed_at.isoformat() if improvement.completed_at else None
                    }
                else:
                    raise HTTPException(status_code=404, detail="Improvement not found")
                
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.put("/improvements/{improvement_id}")
        async def update_improvement(improvement_id: str, improvement: ImprovementUpdate):
            """Update improvement"""
            try:
                from real_improvements_engine import get_real_improvements_engine
                
                engine = get_real_improvements_engine()
                if improvement_id in engine.improvements:
                    imp = engine.improvements[improvement_id]
                    
                    if improvement.title:
                        imp.title = improvement.title
                    if improvement.description:
                        imp.description = improvement.description
                    if improvement.status:
                        imp.status = improvement.status
                    if improvement.notes:
                        imp.notes = improvement.notes
                    
                    return {"status": "updated"}
                else:
                    raise HTTPException(status_code=404, detail="Improvement not found")
                
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.delete("/improvements/{improvement_id}")
        async def delete_improvement(improvement_id: str):
            """Delete improvement"""
            try:
                from real_improvements_engine import get_real_improvements_engine
                
                engine = get_real_improvements_engine()
                if improvement_id in engine.improvements:
                    del engine.improvements[improvement_id]
                    return {"status": "deleted"}
                else:
                    raise HTTPException(status_code=404, detail="Improvement not found")
                
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        # Task routes
        @self.app.post("/tasks/")
        async def create_task(task: TaskCreate):
            """Create implementation task"""
            try:
                from improvement_implementer import get_improvement_implementer
                
                implementer = get_improvement_implementer()
                task_id = implementer.create_implementation_task(
                    improvement_id=task.improvement_id,
                    title=task.title,
                    description=task.description,
                    improvement_type=task.improvement_type,
                    priority=task.priority,
                    estimated_hours=task.estimated_hours
                )
                
                return {"task_id": task_id, "status": "created"}
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/tasks/{task_id}")
        async def get_task_status(task_id: str):
            """Get task status"""
            try:
                from improvement_implementer import get_improvement_implementer
                
                implementer = get_improvement_implementer()
                status = implementer.get_task_status(task_id)
                
                if status:
                    return status
                else:
                    raise HTTPException(status_code=404, detail="Task not found")
                
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/tasks/{task_id}/start")
        async def start_task(task_id: str, assigned_to: str = "developer"):
            """Start task execution"""
            try:
                from improvement_implementer import get_improvement_implementer
                
                implementer = get_improvement_implementer()
                success = await implementer.start_implementation(task_id, assigned_to)
                
                if success:
                    return {"status": "started"}
                else:
                    raise HTTPException(status_code=400, detail="Failed to start task")
                
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        # Plan routes
        @self.app.post("/plans/")
        async def create_plan(plan: PlanCreate):
            """Create improvement plan"""
            try:
                from improvement_coordinator import get_improvement_coordinator
                
                coordinator = get_improvement_coordinator()
                plan_id = coordinator.create_improvement_plan(
                    title=plan.title,
                    description=plan.description,
                    priority=plan.priority,
                    estimated_duration=plan.estimated_duration,
                    improvements=plan.improvements,
                    dependencies=plan.dependencies
                )
                
                return {"plan_id": plan_id, "status": "created"}
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/plans/{plan_id}")
        async def get_plan_status(plan_id: str):
            """Get plan status"""
            try:
                from improvement_coordinator import get_improvement_coordinator
                
                coordinator = get_improvement_coordinator()
                status = await coordinator.get_plan_status(plan_id)
                
                if status:
                    return status
                else:
                    raise HTTPException(status_code=404, detail="Plan not found")
                
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/plans/{plan_id}/execute")
        async def execute_plan(plan_id: str, dry_run: bool = False):
            """Execute improvement plan"""
            try:
                from improvement_coordinator import get_improvement_coordinator
                
                coordinator = get_improvement_coordinator()
                result = await coordinator.execute_plan(plan_id, dry_run)
                
                return result
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        # Execution routes
        @self.app.post("/execute/")
        async def execute_improvement(request: ExecutionRequest):
            """Execute improvement"""
            try:
                from improvement_executor import get_improvement_executor
                
                executor = get_improvement_executor()
                result = await executor.execute_improvement(
                    improvement_id=request.improvement_id,
                    execution_type=request.execution_type,
                    dry_run=request.dry_run
                )
                
                return result
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/execute/{task_id}/rollback")
        async def rollback_execution(task_id: str):
            """Rollback execution"""
            try:
                from improvement_executor import get_improvement_executor
                
                executor = get_improvement_executor()
                result = await executor.rollback_execution(task_id)
                
                return result
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        # Analysis routes
        @self.app.post("/analyze/")
        async def analyze_project(request: AnalysisRequest):
            """Analyze project"""
            try:
                from improvement_analyzer import get_improvement_analyzer
                
                analyzer = get_improvement_analyzer()
                result = await analyzer.analyze_project()
                
                return result
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/analyze/summary")
        async def get_analysis_summary():
            """Get analysis summary"""
            try:
                from improvement_analyzer import get_improvement_analyzer
                
                analyzer = get_improvement_analyzer()
                summary = analyzer.get_analysis_summary()
                
                return summary
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        # Dashboard routes
        @self.app.get("/dashboard/")
        async def get_dashboard():
            """Get dashboard data"""
            try:
                from improvement_dashboard import get_improvement_dashboard
                
                dashboard = get_improvement_dashboard()
                data = await dashboard.update_dashboard()
                
                return data
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/dashboard/summary")
        async def get_dashboard_summary():
            """Get dashboard summary"""
            try:
                from improvement_dashboard import get_improvement_dashboard
                
                dashboard = get_improvement_dashboard()
                summary = await dashboard.get_dashboard_summary()
                
                return summary
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/dashboard/progress")
        async def get_improvement_progress():
            """Get improvement progress"""
            try:
                from improvement_dashboard import get_improvement_dashboard
                
                dashboard = get_improvement_dashboard()
                progress = await dashboard.get_improvement_progress()
                
                return progress
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/dashboard/trends")
        async def get_improvement_trends():
            """Get improvement trends"""
            try:
                from improvement_dashboard import get_improvement_dashboard
                
                dashboard = get_improvement_dashboard()
                trends = await dashboard.get_improvement_trends()
                
                return trends
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/dashboard/activities")
        async def get_recent_activities():
            """Get recent activities"""
            try:
                from improvement_dashboard import get_improvement_dashboard
                
                dashboard = get_improvement_dashboard()
                activities = await dashboard.get_recent_activities()
                
                return activities
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        # Automation routes
        @self.app.post("/automate/execute/{improvement_id}")
        async def execute_automated_improvement(improvement_id: str, dry_run: bool = False):
            """Execute automated improvement"""
            try:
                from improvement_automator import get_improvement_automator
                
                automator = get_improvement_automator()
                result = await automator.execute_improvement(improvement_id, dry_run)
                
                return result
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/automate/execute-all")
        async def execute_all_improvements(category: str = None, max_concurrent: int = 3):
            """Execute all improvements"""
            try:
                from improvement_automator import get_improvement_automator
                
                automator = get_improvement_automator()
                result = await automator.execute_all_improvements(category, max_concurrent)
                
                return result
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/automate/rollback/{improvement_id}")
        async def rollback_automated_improvement(improvement_id: str):
            """Rollback automated improvement"""
            try:
                from improvement_automator import get_improvement_automator
                
                automator = get_improvement_automator()
                result = await automator.rollback_improvement(improvement_id)
                
                return result
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        # Statistics routes
        @self.app.get("/stats/improvements")
        async def get_improvement_stats():
            """Get improvement statistics"""
            try:
                from real_improvements_engine import get_real_improvements_engine
                
                engine = get_real_improvements_engine()
                stats = engine.get_improvement_stats()
                
                return stats
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/stats/execution")
        async def get_execution_stats():
            """Get execution statistics"""
            try:
                from improvement_executor import get_improvement_executor
                
                executor = get_improvement_executor()
                stats = executor.get_execution_summary()
                
                return stats
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/stats/coordination")
        async def get_coordination_stats():
            """Get coordination statistics"""
            try:
                from improvement_coordinator import get_improvement_coordinator
                
                coordinator = get_improvement_coordinator()
                stats = coordinator.get_coordination_summary()
                
                return stats
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
    
    def run(self, host: str = "0.0.0.0", port: int = 8000, debug: bool = False):
        """Run the API server"""
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            log_level="debug" if debug else "info"
        )

# Global API instance
improvement_api = None

def get_improvement_api() -> RealImprovementAPI:
    """Get improvement API instance"""
    global improvement_api
    if not improvement_api:
        improvement_api = RealImprovementAPI()
    return improvement_api

if __name__ == "__main__":
    api = get_improvement_api()
    api.run(debug=True)













