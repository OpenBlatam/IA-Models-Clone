"""
BUL - Business Universal Language (Enterprise Integration)
=========================================================

Advanced enterprise integration and business features for BUL.
"""

import asyncio
import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
import sqlite3
from pathlib import Path
import requests
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
from sqlalchemy import create_engine, Column, String, DateTime, Text, Integer, Boolean, Float, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import redis
from prometheus_client import Counter, Histogram, Gauge, generate_latest

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database setup
Base = declarative_base()
engine = create_engine('sqlite:///bul_enterprise.db', echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Prometheus Metrics
ENTERPRISE_REQUESTS = Counter('bul_enterprise_requests_total', 'Total enterprise requests', ['method', 'endpoint'])
ENTERPRISE_DURATION = Histogram('bul_enterprise_request_duration_seconds', 'Request duration')
ENTERPRISE_USERS = Gauge('bul_enterprise_active_users', 'Number of active users')
ENTERPRISE_PROJECTS = Gauge('bul_enterprise_active_projects', 'Number of active projects')
ENTERPRISE_REVENUE = Gauge('bul_enterprise_revenue', 'Total revenue')

class ProjectStatus(str, Enum):
    """Project status enumeration."""
    PLANNING = "planning"
    ACTIVE = "active"
    COMPLETED = "completed"
    ON_HOLD = "on_hold"
    CANCELLED = "cancelled"

class UserRole(str, Enum):
    """User role enumeration."""
    ADMIN = "admin"
    MANAGER = "manager"
    DEVELOPER = "developer"
    ANALYST = "analyst"
    VIEWER = "viewer"

# Database Models
class EnterpriseUser(Base):
    __tablename__ = "enterprise_users"
    
    id = Column(String, primary_key=True)
    username = Column(String, unique=True, nullable=False)
    email = Column(String, unique=True, nullable=False)
    full_name = Column(String, nullable=False)
    role = Column(String, default=UserRole.VIEWER)
    department = Column(String)
    manager_id = Column(String, ForeignKey("enterprise_users.id"))
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime)
    permissions = Column(Text, default="{}")
    
    # Relationships
    managed_users = relationship("EnterpriseUser", backref="manager", remote_side=[id])
    projects = relationship("EnterpriseProject", back_populates="owner")

class EnterpriseProject(Base):
    __tablename__ = "enterprise_projects"
    
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(Text)
    status = Column(String, default=ProjectStatus.PLANNING)
    owner_id = Column(String, ForeignKey("enterprise_users.id"))
    start_date = Column(DateTime)
    end_date = Column(DateTime)
    budget = Column(Float, default=0.0)
    actual_cost = Column(Float, default=0.0)
    progress = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    owner = relationship("EnterpriseUser", back_populates="projects")
    tasks = relationship("EnterpriseTask", back_populates="project")

class EnterpriseTask(Base):
    __tablename__ = "enterprise_tasks"
    
    id = Column(String, primary_key=True)
    title = Column(String, nullable=False)
    description = Column(Text)
    project_id = Column(String, ForeignKey("enterprise_projects.id"))
    assignee_id = Column(String, ForeignKey("enterprise_users.id"))
    status = Column(String, default="pending")
    priority = Column(String, default="medium")
    due_date = Column(DateTime)
    completed_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    project = relationship("EnterpriseProject", back_populates="tasks")
    assignee = relationship("EnterpriseUser")

class EnterpriseAnalytics(Base):
    __tablename__ = "enterprise_analytics"
    
    id = Column(String, primary_key=True)
    metric_name = Column(String, nullable=False)
    metric_value = Column(Float, nullable=False)
    metric_unit = Column(String)
    category = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)
    metadata = Column(Text, default="{}")

# Create tables
Base.metadata.create_all(bind=engine)

# Pydantic Models
class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., regex=r'^[\w\.-]+@[\w\.-]+\.\w+$')
    full_name: str = Field(..., min_length=2, max_length=100)
    role: UserRole = Field(default=UserRole.VIEWER)
    department: Optional[str] = Field(None, max_length=100)
    manager_id: Optional[str] = Field(None)

class ProjectCreate(BaseModel):
    name: str = Field(..., min_length=3, max_length=100)
    description: Optional[str] = Field(None, max_length=1000)
    owner_id: str = Field(...)
    start_date: Optional[datetime] = Field(None)
    end_date: Optional[datetime] = Field(None)
    budget: Optional[float] = Field(None, ge=0)

class TaskCreate(BaseModel):
    title: str = Field(..., min_length=3, max_length=200)
    description: Optional[str] = Field(None, max_length=1000)
    project_id: str = Field(...)
    assignee_id: Optional[str] = Field(None)
    priority: str = Field(default="medium")
    due_date: Optional[datetime] = Field(None)

class AnalyticsRequest(BaseModel):
    metric_name: str = Field(..., min_length=3, max_length=100)
    metric_value: float = Field(...)
    metric_unit: Optional[str] = Field(None, max_length=20)
    category: Optional[str] = Field(None, max_length=50)
    metadata: Optional[Dict[str, Any]] = Field(default={})

class EnterpriseBULSystem:
    """Enterprise BUL system with advanced business features."""
    
    def __init__(self):
        self.app = FastAPI(
            title="BUL - Business Universal Language (Enterprise)",
            description="Enterprise-grade BUL system with advanced business features",
            version="23.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # Database session
        self.db = SessionLocal()
        
        # Initialize components
        self.setup_middleware()
        self.setup_routes()
        self.setup_default_data()
        
        logger.info("Enterprise BUL System initialized")
    
    def setup_middleware(self):
        """Setup enterprise middleware."""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def setup_routes(self):
        """Setup enterprise API routes."""
        
        @self.app.get("/", tags=["System"])
        async def root():
            """Root endpoint with enterprise system information."""
            return {
                "message": "BUL - Business Universal Language (Enterprise)",
                "version": "23.0.0",
                "status": "operational",
                "features": [
                    "User Management",
                    "Project Management",
                    "Task Management",
                    "Analytics Dashboard",
                    "Enterprise Integration",
                    "Advanced Reporting",
                    "Team Collaboration",
                    "Resource Planning"
                ],
                "timestamp": datetime.now().isoformat()
            }
        
        @self.app.post("/users", tags=["Users"])
        async def create_user(user: UserCreate):
            """Create a new enterprise user."""
            try:
                db_user = EnterpriseUser(
                    id=f"user_{int(time.time())}",
                    username=user.username,
                    email=user.email,
                    full_name=user.full_name,
                    role=user.role,
                    department=user.department,
                    manager_id=user.manager_id
                )
                
                self.db.add(db_user)
                self.db.commit()
                
                ENTERPRISE_USERS.inc()
                
                return {
                    "message": "User created successfully",
                    "user_id": db_user.id,
                    "username": db_user.username
                }
            except Exception as e:
                self.db.rollback()
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/users", tags=["Users"])
        async def get_users():
            """Get all enterprise users."""
            try:
                users = self.db.query(EnterpriseUser).all()
                return {
                    "users": [
                        {
                            "id": user.id,
                            "username": user.username,
                            "email": user.email,
                            "full_name": user.full_name,
                            "role": user.role,
                            "department": user.department,
                            "is_active": user.is_active,
                            "created_at": user.created_at.isoformat()
                        }
                        for user in users
                    ],
                    "total": len(users)
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/projects", tags=["Projects"])
        async def create_project(project: ProjectCreate):
            """Create a new enterprise project."""
            try:
                db_project = EnterpriseProject(
                    id=f"project_{int(time.time())}",
                    name=project.name,
                    description=project.description,
                    owner_id=project.owner_id,
                    start_date=project.start_date,
                    end_date=project.end_date,
                    budget=project.budget or 0.0
                )
                
                self.db.add(db_project)
                self.db.commit()
                
                ENTERPRISE_PROJECTS.inc()
                
                return {
                    "message": "Project created successfully",
                    "project_id": db_project.id,
                    "name": db_project.name
                }
            except Exception as e:
                self.db.rollback()
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/projects", tags=["Projects"])
        async def get_projects():
            """Get all enterprise projects."""
            try:
                projects = self.db.query(EnterpriseProject).all()
                return {
                    "projects": [
                        {
                            "id": project.id,
                            "name": project.name,
                            "description": project.description,
                            "status": project.status,
                            "owner_id": project.owner_id,
                            "start_date": project.start_date.isoformat() if project.start_date else None,
                            "end_date": project.end_date.isoformat() if project.end_date else None,
                            "budget": project.budget,
                            "actual_cost": project.actual_cost,
                            "progress": project.progress,
                            "created_at": project.created_at.isoformat()
                        }
                        for project in projects
                    ],
                    "total": len(projects)
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/tasks", tags=["Tasks"])
        async def create_task(task: TaskCreate):
            """Create a new enterprise task."""
            try:
                db_task = EnterpriseTask(
                    id=f"task_{int(time.time())}",
                    title=task.title,
                    description=task.description,
                    project_id=task.project_id,
                    assignee_id=task.assignee_id,
                    priority=task.priority,
                    due_date=task.due_date
                )
                
                self.db.add(db_task)
                self.db.commit()
                
                return {
                    "message": "Task created successfully",
                    "task_id": db_task.id,
                    "title": db_task.title
                }
            except Exception as e:
                self.db.rollback()
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/tasks", tags=["Tasks"])
        async def get_tasks():
            """Get all enterprise tasks."""
            try:
                tasks = self.db.query(EnterpriseTask).all()
                return {
                    "tasks": [
                        {
                            "id": task.id,
                            "title": task.title,
                            "description": task.description,
                            "project_id": task.project_id,
                            "assignee_id": task.assignee_id,
                            "status": task.status,
                            "priority": task.priority,
                            "due_date": task.due_date.isoformat() if task.due_date else None,
                            "completed_at": task.completed_at.isoformat() if task.completed_at else None,
                            "created_at": task.created_at.isoformat()
                        }
                        for task in tasks
                    ],
                    "total": len(tasks)
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/analytics", tags=["Analytics"])
        async def record_analytics(analytics: AnalyticsRequest):
            """Record enterprise analytics data."""
            try:
                db_analytics = EnterpriseAnalytics(
                    id=f"analytics_{int(time.time())}",
                    metric_name=analytics.metric_name,
                    metric_value=analytics.metric_value,
                    metric_unit=analytics.metric_unit,
                    category=analytics.category,
                    metadata=json.dumps(analytics.metadata)
                )
                
                self.db.add(db_analytics)
                self.db.commit()
                
                return {
                    "message": "Analytics recorded successfully",
                    "analytics_id": db_analytics.id
                }
            except Exception as e:
                self.db.rollback()
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/analytics/dashboard", tags=["Analytics"])
        async def get_analytics_dashboard():
            """Get analytics dashboard data."""
            try:
                # Get basic counts
                user_count = self.db.query(EnterpriseUser).count()
                project_count = self.db.query(EnterpriseProject).count()
                task_count = self.db.query(EnterpriseTask).count()
                
                # Get project status distribution
                project_statuses = self.db.query(EnterpriseProject.status).all()
                status_counts = {}
                for status in project_statuses:
                    status_counts[status[0]] = status_counts.get(status[0], 0) + 1
                
                # Get task status distribution
                task_statuses = self.db.query(EnterpriseTask.status).all()
                task_status_counts = {}
                for status in task_statuses:
                    task_status_counts[status[0]] = task_status_counts.get(status[0], 0) + 1
                
                # Get recent analytics
                recent_analytics = self.db.query(EnterpriseAnalytics).order_by(EnterpriseAnalytics.timestamp.desc()).limit(10).all()
                
                return {
                    "summary": {
                        "total_users": user_count,
                        "total_projects": project_count,
                        "total_tasks": task_count,
                        "active_projects": status_counts.get("active", 0),
                        "completed_projects": status_counts.get("completed", 0)
                    },
                    "project_status_distribution": status_counts,
                    "task_status_distribution": task_status_counts,
                    "recent_analytics": [
                        {
                            "metric_name": analytics.metric_name,
                            "metric_value": analytics.metric_value,
                            "metric_unit": analytics.metric_unit,
                            "category": analytics.category,
                            "timestamp": analytics.timestamp.isoformat()
                        }
                        for analytics in recent_analytics
                    ]
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/reports/project-performance", tags=["Reports"])
        async def get_project_performance_report():
            """Get project performance report."""
            try:
                projects = self.db.query(EnterpriseProject).all()
                
                report_data = []
                for project in projects:
                    # Get tasks for this project
                    tasks = self.db.query(EnterpriseTask).filter(EnterpriseTask.project_id == project.id).all()
                    
                    completed_tasks = len([t for t in tasks if t.status == "completed"])
                    total_tasks = len(tasks)
                    completion_rate = (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0
                    
                    report_data.append({
                        "project_id": project.id,
                        "project_name": project.name,
                        "status": project.status,
                        "budget": project.budget,
                        "actual_cost": project.actual_cost,
                        "budget_utilization": (project.actual_cost / project.budget * 100) if project.budget > 0 else 0,
                        "total_tasks": total_tasks,
                        "completed_tasks": completed_tasks,
                        "completion_rate": completion_rate,
                        "start_date": project.start_date.isoformat() if project.start_date else None,
                        "end_date": project.end_date.isoformat() if project.end_date else None
                    })
                
                return {
                    "report_type": "project_performance",
                    "generated_at": datetime.now().isoformat(),
                    "projects": report_data,
                    "summary": {
                        "total_projects": len(projects),
                        "active_projects": len([p for p in projects if p.status == "active"]),
                        "completed_projects": len([p for p in projects if p.status == "completed"]),
                        "total_budget": sum(p.budget for p in projects),
                        "total_actual_cost": sum(p.actual_cost for p in projects)
                    }
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
    
    def setup_default_data(self):
        """Setup default enterprise data."""
        try:
            # Create default admin user
            admin_user = EnterpriseUser(
                id="admin",
                username="admin",
                email="admin@bul-enterprise.com",
                full_name="System Administrator",
                role=UserRole.ADMIN,
                department="IT",
                is_active=True
            )
            
            # Create sample project
            sample_project = EnterpriseProject(
                id="project_1",
                name="BUL System Implementation",
                description="Implementation of BUL enterprise system",
                status=ProjectStatus.ACTIVE,
                owner_id="admin",
                budget=100000.0,
                progress=75.0
            )
            
            # Create sample tasks
            sample_tasks = [
                EnterpriseTask(
                    id="task_1",
                    title="System Design",
                    description="Design the enterprise system architecture",
                    project_id="project_1",
                    assignee_id="admin",
                    status="completed",
                    priority="high"
                ),
                EnterpriseTask(
                    id="task_2",
                    title="API Development",
                    description="Develop REST API endpoints",
                    project_id="project_1",
                    assignee_id="admin",
                    status="active",
                    priority="high"
                ),
                EnterpriseTask(
                    id="task_3",
                    title="Testing",
                    description="Comprehensive system testing",
                    project_id="project_1",
                    assignee_id="admin",
                    status="pending",
                    priority="medium"
                )
            ]
            
            # Add to database
            self.db.add(admin_user)
            self.db.add(sample_project)
            for task in sample_tasks:
                self.db.add(task)
            
            self.db.commit()
            logger.info("Default enterprise data created")
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error creating default data: {e}")
    
    def run(self, host: str = "0.0.0.0", port: int = 8002, debug: bool = False):
        """Run the enterprise BUL system."""
        logger.info(f"Starting Enterprise BUL system on {host}:{port}")
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            log_level="debug" if debug else "info"
        )

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="BUL - Business Universal Language (Enterprise)")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8002, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    # Create and run enterprise system
    system = EnterpriseBULSystem()
    system.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == "__main__":
    main()
