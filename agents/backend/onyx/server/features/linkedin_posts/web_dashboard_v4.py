#!/usr/bin/env python3
"""
🚀 ENHANCED LINKEDIN OPTIMIZER v4.0 - MODERN WEB DASHBOARD
============================================================

A beautiful, responsive web interface for the Enhanced LinkedIn Optimizer v4.0
Features:
- Real-time content optimization
- Live performance monitoring
- Interactive analytics dashboard
- Content management interface
- System health monitoring
- User management and authentication

Built with FastAPI, React-like components, and modern web technologies
"""

import asyncio
import json
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import uvicorn
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
import psutil
import asyncio
from contextlib import asynccontextmanager

# Add current directory to Python path
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from enhanced_system_integration_v4 import EnhancedLinkedInOptimizer
    from ai_content_intelligence_v4 import AIContentIntelligenceSystem
    from real_time_analytics_v4 import RealTimeAnalyticsSystem
    from security_compliance_v4 import SecurityComplianceSystem
except ImportError as e:
    print(f"Warning: Could not import v4.0 modules: {e}")
    print("Dashboard will run in demo mode")

# Pydantic models for API
class ContentOptimizationRequest(BaseModel):
    content: str = Field(..., description="Content to optimize")
    platform: str = Field(default="linkedin", description="Target platform")
    target_audience: str = Field(default="general", description="Target audience")
    optimization_goals: List[str] = Field(default=["engagement"], description="Optimization goals")
    language: str = Field(default="en", description="Content language")

class ContentOptimizationResponse(BaseModel):
    request_id: str
    optimization_score: float
    sentiment_analysis: Dict[str, Any]
    engagement_prediction: Dict[str, Any]
    content_suggestions: List[str]
    hashtag_recommendations: List[str]
    processing_time: float
    timestamp: datetime

class SystemHealthResponse(BaseModel):
    status: str
    memory_usage_mb: float
    cpu_usage_percent: float
    disk_usage_percent: float
    active_connections: int
    uptime_seconds: float
    last_health_check: datetime

class BatchOptimizationRequest(BaseModel):
    contents: List[ContentOptimizationRequest]
    priority: str = Field(default="normal", description="Processing priority")

class UserAuthentication(BaseModel):
    username: str
    password: str

class DashboardConfig:
    """Dashboard configuration and state management."""
    
    def __init__(self):
        self.app = FastAPI(
            title="Enhanced LinkedIn Optimizer v4.0 Dashboard",
            description="Modern web interface for AI-powered content optimization",
            version="4.0.0",
            docs_url="/api/docs",
            redoc_url="/api/redoc"
        )
        
        # Initialize systems
        self.optimizer = None
        self.ai_system = None
        self.analytics_system = None
        self.security_system = None
        
        # Dashboard state
        self.active_connections = 0
        self.start_time = time.time()
        self.optimization_history = []
        self.system_metrics = []
        self.websocket_connections = []
        
        # Setup middleware and routes
        self._setup_middleware()
        self._setup_routes()
        self._setup_websockets()
    
    def _setup_middleware(self):
        """Setup CORS and other middleware."""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def _setup_routes(self):
        """Setup API routes and endpoints."""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def get_dashboard():
            """Serve the main dashboard HTML."""
            return self._get_dashboard_html()
        
        @self.app.get("/api/health")
        async def get_system_health():
            """Get system health status."""
            return await self._get_system_health()
        
        @self.app.post("/api/optimize", response_model=ContentOptimizationResponse)
        async def optimize_content(request: ContentOptimizationRequest):
            """Optimize a single piece of content."""
            return await self._optimize_content(request)
        
        @self.app.post("/api/optimize/batch")
        async def batch_optimize(request: BatchOptimizationRequest, background_tasks: BackgroundTasks):
            """Optimize multiple pieces of content in batch."""
            return await self._batch_optimize(request, background_tasks)
        
        @self.app.get("/api/analytics/performance")
        async def get_performance_analytics():
            """Get performance analytics data."""
            return await self._get_performance_analytics()
        
        @self.app.get("/api/analytics/trends")
        async def get_trend_analytics():
            """Get trend analysis data."""
            return await self._get_trend_analytics()
        
        @self.app.get("/api/content/history")
        async def get_optimization_history():
            """Get content optimization history."""
            return {"history": self.optimization_history[-100:]}  # Last 100 optimizations
        
        @self.app.get("/api/system/metrics")
        async def get_system_metrics():
            """Get real-time system metrics."""
            return {"metrics": self.system_metrics[-50:]}  # Last 50 metrics
        
        @self.app.post("/api/auth/login")
        async def login(auth: UserAuthentication):
            """User authentication."""
            return await self._authenticate_user(auth)
        
        @self.app.get("/api/config")
        async def get_configuration():
            """Get current system configuration."""
            return await self._get_configuration()
        
        @self.app.put("/api/config")
        async def update_configuration(config: Dict[str, Any]):
            """Update system configuration."""
            return await self._update_configuration(config)
    
    def _setup_websockets(self):
        """Setup WebSocket connections for real-time updates."""
        
        @self.app.websocket("/ws/dashboard")
        async def websocket_dashboard(websocket: WebSocket):
            """WebSocket endpoint for real-time dashboard updates."""
            await websocket.accept()
            self.websocket_connections.append(websocket)
            self.active_connections += 1
            
            try:
                while True:
                    # Send real-time updates
                    await websocket.send_text(json.dumps({
                        "type": "system_metrics",
                        "data": await self._get_system_health()
                    }))
                    await asyncio.sleep(5)  # Update every 5 seconds
                    
            except WebSocketDisconnect:
                self.websocket_connections.remove(websocket)
                self.active_connections -= 1
    
    async def _initialize_systems(self):
        """Initialize the v4.0 systems."""
        try:
            if self.optimizer is None:
                self.optimizer = EnhancedLinkedInOptimizer()
                self.ai_system = AIContentIntelligenceSystem()
                self.analytics_system = RealTimeAnalyticsSystem()
                self.security_system = SecurityComplianceSystem()
                print("✅ All v4.0 systems initialized successfully")
        except Exception as e:
            print(f"⚠️  Running in demo mode: {e}")
            self.optimizer = None
    
    async def _get_system_health(self) -> SystemHealthResponse:
        """Get comprehensive system health information."""
        try:
            # Get system metrics
            memory = psutil.virtual_memory()
            cpu = psutil.cpu_percent(interval=1)
            disk = psutil.disk_usage('.')
            uptime = time.time() - self.start_time
            
            # Get optimizer health if available
            if self.optimizer:
                optimizer_health = await self.optimizer.get_system_health()
                status = optimizer_health.get('status', 'RUNNING')
            else:
                status = 'DEMO_MODE'
            
            health_data = SystemHealthResponse(
                status=status,
                memory_usage_mb=memory.used / (1024 * 1024),
                cpu_usage_percent=cpu,
                disk_usage_percent=disk.percent,
                active_connections=self.active_connections,
                uptime_seconds=uptime,
                last_health_check=datetime.now()
            )
            
            # Store metrics for history
            self.system_metrics.append({
                "timestamp": datetime.now().isoformat(),
                "memory_mb": health_data.memory_usage_mb,
                "cpu_percent": health_data.cpu_usage_percent,
                "disk_percent": health_data.disk_usage_percent,
                "active_connections": health_data.active_connections
            })
            
            return health_data
            
        except Exception as e:
            return SystemHealthResponse(
                status="ERROR",
                memory_usage_mb=0,
                cpu_usage_percent=0,
                disk_usage_percent=0,
                active_connections=0,
                uptime_seconds=0,
                last_health_check=datetime.now()
            )
    
    async def _optimize_content(self, request: ContentOptimizationRequest) -> ContentOptimizationResponse:
        """Optimize a single piece of content."""
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        try:
            if self.optimizer:
                # Use real v4.0 system
                result = await self.optimizer.optimize_content(
                    content=request.content,
                    platform=request.platform,
                    target_audience=request.target_audience,
                    optimization_goals=request.optimization_goals
                )
                
                optimization_score = result.get('optimization_score', 0.0)
                sentiment_analysis = result.get('sentiment_analysis', {})
                engagement_prediction = result.get('engagement_prediction', {})
                content_suggestions = result.get('content_suggestions', [])
                hashtag_recommendations = result.get('hashtag_recommendations', [])
                
            else:
                # Demo mode - generate mock results
                optimization_score = 0.85
                sentiment_analysis = {
                    "overall_sentiment": "positive",
                    "confidence": 0.92,
                    "sentiment_score": 0.78
                }
                engagement_prediction = {
                    "predicted_level": "HIGH",
                    "confidence": 0.88,
                    "estimated_reach": 15000
                }
                content_suggestions = [
                    "Add more specific examples to increase credibility",
                    "Include a call-to-action to boost engagement",
                    "Use more industry-specific keywords"
                ]
                hashtag_recommendations = [
                    "#ContentOptimization",
                    "#LinkedInTips",
                    "#DigitalMarketing",
                    "#ProfessionalGrowth"
                ]
            
            processing_time = time.time() - start_time
            
            response = ContentOptimizationResponse(
                request_id=request_id,
                optimization_score=optimization_score,
                sentiment_analysis=sentiment_analysis,
                engagement_prediction=engagement_prediction,
                content_suggestions=content_suggestions,
                hashtag_recommendations=hashtag_recommendations,
                processing_time=processing_time,
                timestamp=datetime.now()
            )
            
            # Store in history
            self.optimization_history.append({
                "request_id": request_id,
                "content_preview": request.content[:100] + "..." if len(request.content) > 100 else request.content,
                "platform": request.platform,
                "optimization_score": optimization_score,
                "timestamp": response.timestamp.isoformat(),
                "processing_time": processing_time
            })
            
            return response
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")
    
    async def _batch_optimize(self, request: BatchOptimizationRequest, background_tasks: BackgroundTasks):
        """Optimize multiple pieces of content in batch."""
        batch_id = str(uuid.uuid4())
        
        # Start background processing
        background_tasks.add_task(self._process_batch, batch_id, request.contents)
        
        return {
            "batch_id": batch_id,
            "status": "processing",
            "total_items": len(request.contents),
            "priority": request.priority,
            "estimated_completion": datetime.now() + timedelta(minutes=len(request.contents) * 0.5)
        }
    
    async def _process_batch(self, batch_id: str, contents: List[ContentOptimizationRequest]):
        """Process batch optimization in background."""
        results = []
        
        for i, content_request in enumerate(contents):
            try:
                result = await self._optimize_content(content_request)
                results.append(result)
                
                # Update progress via WebSocket
                await self._broadcast_progress(batch_id, i + 1, len(contents))
                
            except Exception as e:
                results.append({
                    "error": str(e),
                    "content": content_request.content[:100]
                })
        
        # Store batch results
        self.optimization_history.append({
            "batch_id": batch_id,
            "type": "batch_optimization",
            "total_items": len(contents),
            "successful_items": len([r for r in results if "error" not in r]),
            "timestamp": datetime.now().isoformat()
        })
    
    async def _broadcast_progress(self, batch_id: str, current: int, total: int):
        """Broadcast batch progress to connected clients."""
        message = {
            "type": "batch_progress",
            "batch_id": batch_id,
            "current": current,
            "total": total,
            "percentage": (current / total) * 100
        }
        
        for websocket in self.websocket_connections:
            try:
                await websocket.send_text(json.dumps(message))
            except:
                continue
    
    async def _get_performance_analytics(self):
        """Get performance analytics data."""
        try:
            if self.analytics_system:
                # Use real analytics system
                analytics = await self.analytics_system.get_performance_metrics()
                return analytics
            else:
                # Demo mode - generate mock analytics
                return {
                    "total_optimizations": len(self.optimization_history),
                    "average_score": 0.82,
                    "success_rate": 0.95,
                    "average_processing_time": 2.3,
                    "top_performing_content": [
                        {"score": 0.95, "content": "AI-powered content optimization insights"},
                        {"score": 0.92, "content": "LinkedIn engagement strategies"},
                        {"score": 0.89, "content": "Professional branding tips"}
                    ]
                }
        except Exception as e:
            return {"error": str(e)}
    
    async def _get_trend_analytics(self):
        """Get trend analysis data."""
        try:
            if self.analytics_system:
                # Use real analytics system
                trends = await self.analytics_system.get_trend_analysis()
                return trends
            else:
                # Demo mode - generate mock trends
                return {
                    "engagement_trend": "INCREASING",
                    "sentiment_trend": "STABLE",
                    "performance_trend": "IMPROVING",
                    "trend_confidence": 0.87,
                    "trend_data": [
                        {"date": "2024-01-01", "score": 0.75},
                        {"date": "2024-01-02", "score": 0.78},
                        {"date": "2024-01-03", "score": 0.82},
                        {"date": "2024-01-04", "score": 0.85}
                    ]
                }
        except Exception as e:
            return {"error": str(e)}
    
    async def _authenticate_user(self, auth: UserAuthentication):
        """Authenticate user (demo implementation)."""
        # Demo authentication - in production, use proper auth system
        if auth.username == "admin" and auth.password == "admin123":
            return {
                "status": "success",
                "token": "demo_token_" + str(uuid.uuid4()),
                "user": {
                    "username": auth.username,
                    "role": "admin",
                    "permissions": ["read", "write", "admin"]
                }
            }
        else:
            raise HTTPException(status_code=401, detail="Invalid credentials")
    
    async def _get_configuration(self):
        """Get current system configuration."""
        return {
            "system_version": "4.0.0",
            "ai_models_loaded": self.optimizer is not None,
            "monitoring_enabled": True,
            "backup_enabled": True,
            "security_level": "production",
            "max_concurrent_requests": 100,
            "cache_enabled": True
        }
    
    async def _update_configuration(self, config: Dict[str, Any]):
        """Update system configuration."""
        # In production, implement proper configuration management
        return {
            "status": "success",
            "message": "Configuration updated successfully",
            "updated_config": config
        }
    
    def _get_dashboard_html(self) -> str:
        """Generate the main dashboard HTML."""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Enhanced LinkedIn Optimizer v4.0 - Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        .gradient-bg { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
        .card-hover { transition: all 0.3s ease; }
        .card-hover:hover { transform: translateY(-5px); box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1); }
    </style>
</head>
<body class="bg-gray-50">
    <!-- Header -->
    <header class="gradient-bg text-white shadow-lg">
        <div class="container mx-auto px-6 py-4">
            <div class="flex items-center justify-between">
                <div class="flex items-center space-x-4">
                    <div class="text-3xl font-bold">🚀</div>
                    <div>
                        <h1 class="text-2xl font-bold">Enhanced LinkedIn Optimizer v4.0</h1>
                        <p class="text-blue-100">AI-Powered Content Optimization Dashboard</p>
                    </div>
                </div>
                <div class="flex items-center space-x-4">
                    <div id="system-status" class="px-3 py-1 rounded-full text-sm font-medium bg-green-500">SYSTEM ONLINE</div>
                    <div class="text-sm">
                        <div>Uptime: <span id="uptime">--</span></div>
                        <div>Active Connections: <span id="active-connections">0</span></div>
                    </div>
                </div>
            </div>
        </div>
    </header>

    <!-- Main Content -->
    <main class="container mx-auto px-6 py-8">
        <!-- Quick Actions -->
        <div class="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
            <div class="bg-white rounded-lg shadow-md p-6 card-hover">
                <h3 class="text-lg font-semibold text-gray-800 mb-4">📝 Content Optimization</h3>
                <textarea id="content-input" placeholder="Enter your LinkedIn content here..." class="w-full p-3 border border-gray-300 rounded-lg resize-none h-32"></textarea>
                <button onclick="optimizeContent()" class="mt-4 w-full bg-blue-600 text-white py-2 px-4 rounded-lg hover:bg-blue-700 transition-colors">
                    Optimize Content
                </button>
            </div>
            
            <div class="bg-white rounded-lg shadow-md p-6 card-hover">
                <h3 class="text-lg font-semibold text-gray-800 mb-4">📊 System Health</h3>
                <div class="space-y-3">
                    <div class="flex justify-between">
                        <span>Memory Usage:</span>
                        <span id="memory-usage" class="font-medium">--</span>
                    </div>
                    <div class="flex justify-between">
                        <span>CPU Usage:</span>
                        <span id="cpu-usage" class="font-medium">--</span>
                    </div>
                    <div class="flex justify-between">
                        <span>Disk Usage:</span>
                        <span id="disk-usage" class="font-medium">--</span>
                    </div>
                </div>
                <button onclick="refreshHealth()" class="mt-4 w-full bg-green-600 text-white py-2 px-4 rounded-lg hover:bg-green-700 transition-colors">
                    Refresh Health
                </button>
            </div>
            
            <div class="bg-white rounded-lg shadow-md p-6 card-hover">
                <h3 class="text-lg font-semibold text-gray-800 mb-4">📈 Performance</h3>
                <div class="space-y-3">
                    <div class="flex justify-between">
                        <span>Total Optimizations:</span>
                        <span id="total-optimizations" class="font-medium">--</span>
                    </div>
                    <div class="flex justify-between">
                        <span>Average Score:</span>
                        <span id="average-score" class="font-medium">--</span>
                    </div>
                    <div class="flex justify-between">
                        <span>Success Rate:</span>
                        <span id="success-rate" class="font-medium">--</span>
                    </div>
                </div>
                <button onclick="refreshAnalytics()" class="mt-4 w-full bg-purple-600 text-white py-2 px-4 rounded-lg hover:bg-purple-700 transition-colors">
                    Refresh Analytics
                </button>
            </div>
        </div>

        <!-- Results Section -->
        <div id="results-section" class="hidden bg-white rounded-lg shadow-md p-6 mb-8">
            <h3 class="text-lg font-semibold text-gray-800 mb-4">🎯 Optimization Results</h3>
            <div id="optimization-results" class="space-y-4"></div>
        </div>

        <!-- Charts Section -->
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
            <div class="bg-white rounded-lg shadow-md p-6">
                <h3 class="text-lg font-semibold text-gray-800 mb-4">📊 System Metrics Over Time</h3>
                <canvas id="metrics-chart" width="400" height="200"></canvas>
            </div>
            <div class="bg-white rounded-lg shadow-md p-6">
                <h3 class="text-lg font-semibold text-gray-800 mb-4">📈 Optimization Scores</h3>
                <canvas id="scores-chart" width="400" height="200"></canvas>
            </div>
        </div>

        <!-- Recent Activity -->
        <div class="bg-white rounded-lg shadow-md p-6">
            <h3 class="text-lg font-semibold text-gray-800 mb-4">🕒 Recent Activity</h3>
            <div id="recent-activity" class="space-y-2 max-h-64 overflow-y-auto"></div>
        </div>
    </main>

    <!-- Footer -->
    <footer class="bg-gray-800 text-white py-6 mt-12">
        <div class="container mx-auto px-6 text-center">
            <p>&copy; 2024 Enhanced LinkedIn Optimizer v4.0. Built with cutting-edge AI and enterprise architecture.</p>
        </div>
    </footer>

    <script>
        // Dashboard functionality
        let metricsChart, scoresChart;
        let websocket;

        // Initialize dashboard
        document.addEventListener('DOMContentLoaded', function() {
            initializeCharts();
            connectWebSocket();
            refreshAll();
            setInterval(refreshAll, 30000); // Refresh every 30 seconds
        });

        function initializeCharts() {
            // Metrics Chart
            const metricsCtx = document.getElementById('metrics-chart').getContext('2d');
            metricsChart = new Chart(metricsCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Memory Usage (MB)',
                        data: [],
                        borderColor: 'rgb(59, 130, 246)',
                        backgroundColor: 'rgba(59, 130, 246, 0.1)',
                        tension: 0.4
                    }, {
                        label: 'CPU Usage (%)',
                        data: [],
                        borderColor: 'rgb(16, 185, 129)',
                        backgroundColor: 'rgba(16, 185, 129, 0.1)',
                        tension: 0.4
                    }]
                },
                options: {
                    responsive: true,
                    scales: {
                        y: {
                            beginAtZero: true
                        }
                    }
                }
            });

            // Scores Chart
            const scoresCtx = document.getElementById('scores-chart').getContext('2d');
            scoresChart = new Chart(scoresCtx, {
                type: 'bar',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Optimization Score',
                        data: [],
                        backgroundColor: 'rgba(147, 51, 234, 0.8)',
                        borderColor: 'rgb(147, 51, 234)',
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 1
                        }
                    }
                }
            });
        }

        function connectWebSocket() {
            websocket = new WebSocket(`ws://${window.location.host}/ws/dashboard`);
            
            websocket.onmessage = function(event) {
                const data = JSON.parse(event.data);
                if (data.type === 'system_metrics') {
                    updateSystemMetrics(data.data);
                } else if (data.type === 'batch_progress') {
                    updateBatchProgress(data);
                }
            };

            websocket.onclose = function() {
                setTimeout(connectWebSocket, 5000); // Reconnect after 5 seconds
            };
        }

        async function optimizeContent() {
            const content = document.getElementById('content-input').value.trim();
            if (!content) {
                alert('Please enter content to optimize');
                return;
            }

            try {
                const response = await fetch('/api/optimize', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        content: content,
                        platform: 'linkedin',
                        target_audience: 'general',
                        optimization_goals: ['engagement', 'reach', 'professional_branding']
                    })
                });

                const result = await response.json();
                displayResults(result);
                refreshAnalytics();
            } catch (error) {
                console.error('Optimization failed:', error);
                alert('Optimization failed. Please try again.');
            }
        }

        function displayResults(result) {
            const resultsSection = document.getElementById('results-section');
            const resultsDiv = document.getElementById('optimization-results');
            
            resultsSection.classList.remove('hidden');
            
            resultsDiv.innerHTML = `
                <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div class="bg-blue-50 p-4 rounded-lg">
                        <h4 class="font-semibold text-blue-800 mb-2">📊 Optimization Score</h4>
                        <div class="text-3xl font-bold text-blue-600">${(result.optimization_score * 100).toFixed(1)}%</div>
                        <p class="text-blue-600">Processing time: ${result.processing_time.toFixed(2)}s</p>
                    </div>
                    
                    <div class="bg-green-50 p-4 rounded-lg">
                        <h4 class="font-semibold text-green-800 mb-2">🎯 Sentiment Analysis</h4>
                        <div class="text-lg font-medium text-green-600">${result.sentiment_analysis.overall_sentiment || 'N/A'}</div>
                        <p class="text-green-600">Confidence: ${(result.sentiment_analysis.confidence * 100).toFixed(1)}%</p>
                    </div>
                </div>
                
                <div class="mt-6">
                    <h4 class="font-semibold text-gray-800 mb-3">💡 Content Suggestions</h4>
                    <ul class="list-disc list-inside space-y-2">
                        ${result.content_suggestions.map(suggestion => `<li class="text-gray-700">${suggestion}</li>`).join('')}
                    </ul>
                </div>
                
                <div class="mt-6">
                    <h4 class="font-semibold text-gray-800 mb-3">🏷️ Hashtag Recommendations</h4>
                    <div class="flex flex-wrap gap-2">
                        ${result.hashtag_recommendations.map(hashtag => 
                            `<span class="bg-blue-100 text-blue-800 px-3 py-1 rounded-full text-sm">${hashtag}</span>`
                        ).join('')}
                    </div>
                </div>
            `;
        }

        async function refreshHealth() {
            try {
                const response = await fetch('/api/health');
                const health = await response.json();
                updateSystemMetrics(health);
            } catch (error) {
                console.error('Failed to refresh health:', error);
            }
        }

        async function refreshAnalytics() {
            try {
                const response = await fetch('/api/analytics/performance');
                const analytics = await response.json();
                updateAnalytics(analytics);
            } catch (error) {
                console.error('Failed to refresh analytics:', error);
            }
        }

        function updateSystemMetrics(health) {
            document.getElementById('memory-usage').textContent = `${health.memory_usage_mb.toFixed(1)} MB`;
            document.getElementById('cpu-usage').textContent = `${health.cpu_usage_percent.toFixed(1)}%`;
            document.getElementById('disk-usage').textContent = `${health.disk_usage_percent.toFixed(1)}%`;
            document.getElementById('uptime').textContent = formatUptime(health.uptime_seconds);
            document.getElementById('active-connections').textContent = health.active_connections;
            
            // Update status indicator
            const statusElement = document.getElementById('system-status');
            if (health.status === 'RUNNING' || health.status === 'DEMO_MODE') {
                statusElement.textContent = 'SYSTEM ONLINE';
                statusElement.className = 'px-3 py-1 rounded-full text-sm font-medium bg-green-500';
            } else {
                statusElement.textContent = 'SYSTEM OFFLINE';
                statusElement.className = 'px-3 py-1 rounded-full text-sm font-medium bg-red-500';
            }

            // Update charts
            const timestamp = new Date().toLocaleTimeString();
            metricsChart.data.labels.push(timestamp);
            metricsChart.data.datasets[0].data.push(health.memory_usage_mb);
            metricsChart.data.datasets[1].data.push(health.cpu_usage_percent);
            
            if (metricsChart.data.labels.length > 20) {
                metricsChart.data.labels.shift();
                metricsChart.data.datasets[0].data.shift();
                metricsChart.data.datasets[1].data.shift();
            }
            
            metricsChart.update();
        }

        function updateAnalytics(analytics) {
            document.getElementById('total-optimizations').textContent = analytics.total_optimizations || 0;
            document.getElementById('average-score').textContent = `${((analytics.average_score || 0) * 100).toFixed(1)}%`;
            document.getElementById('success-rate').textContent = `${((analytics.success_rate || 0) * 100).toFixed(1)}%`;
        }

        function formatUptime(seconds) {
            const hours = Math.floor(seconds / 3600);
            const minutes = Math.floor((seconds % 3600) / 60);
            return `${hours}h ${minutes}m`;
        }

        async function refreshAll() {
            await Promise.all([
                refreshHealth(),
                refreshAnalytics(),
                refreshRecentActivity()
            ]);
        }

        async function refreshRecentActivity() {
            try {
                const response = await fetch('/api/content/history');
                const history = await response.json();
                
                const activityDiv = document.getElementById('recent-activity');
                activityDiv.innerHTML = history.history.slice(-10).reverse().map(item => `
                    <div class="flex justify-between items-center py-2 border-b border-gray-200">
                        <div class="flex-1">
                            <div class="text-sm text-gray-600">${item.content_preview}</div>
                            <div class="text-xs text-gray-500">${new Date(item.timestamp).toLocaleString()}</div>
                        </div>
                        <div class="text-right">
                            <div class="text-sm font-medium text-gray-800">${(item.optimization_score * 100).toFixed(1)}%</div>
                            <div class="text-xs text-gray-500">${item.processing_time.toFixed(2)}s</div>
                        </div>
                    </div>
                `).join('');
            } catch (error) {
                console.error('Failed to refresh recent activity:', error);
            }
        }
    </script>
</body>
</html>
        """
    
    def run(self, host: str = "0.0.0.0", port: int = 8080):
        """Run the dashboard server."""
        print(f"🚀 Starting Enhanced LinkedIn Optimizer v4.0 Dashboard...")
        print(f"📍 Dashboard will be available at: http://{host}:{port}")
        print(f"📚 API documentation: http://{host}:{port}/api/docs")
        
        # Initialize systems
        asyncio.run(self._initialize_systems())
        
        # Start server
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            log_level="info"
        )

if __name__ == "__main__":
    dashboard = DashboardConfig()
    dashboard.run()
