"""
Dashboard for Document Workflow Chain Monitoring
===============================================

This module provides a web-based dashboard for monitoring and managing
document workflow chains, including real-time statistics and visualizations.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import aiofiles
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

# Create router for dashboard
dashboard_router = APIRouter(prefix="/dashboard", tags=["Dashboard"])

# Templates directory
templates_dir = Path(__file__).parent / "templates"
templates_dir.mkdir(exist_ok=True)

# Static files directory
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)

# Initialize templates
templates = Jinja2Templates(directory=str(templates_dir))

# Mount static files
dashboard_router.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

class DashboardManager:
    """Manager for dashboard data and operations"""
    
    def __init__(self, workflow_engine=None, database_manager=None):
        self.workflow_engine = workflow_engine
        self.database_manager = database_manager
        self.dashboard_data = {
            "total_chains": 0,
            "active_chains": 0,
            "completed_chains": 0,
            "total_documents": 0,
            "total_tokens": 0,
            "avg_quality_score": 0.0,
            "recent_activity": [],
            "ai_client_stats": {},
            "performance_metrics": {}
        }
    
    async def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data"""
        try:
            # Get basic statistics
            if self.workflow_engine:
                active_chains = self.workflow_engine.get_all_active_chains()
                self.dashboard_data["active_chains"] = len(active_chains)
                self.dashboard_data["total_documents"] = sum(
                    len(chain.nodes) for chain in active_chains
                )
            
            # Get database statistics if available
            if self.database_manager:
                # This would require additional database queries
                # For now, we'll use the workflow engine data
                pass
            
            # Get AI client statistics
            if self.workflow_engine and self.workflow_engine.ai_client:
                self.dashboard_data["ai_client_stats"] = self.workflow_engine.ai_client.get_stats()
            
            # Calculate performance metrics
            self.dashboard_data["performance_metrics"] = await self._calculate_performance_metrics()
            
            # Get recent activity
            self.dashboard_data["recent_activity"] = await self._get_recent_activity()
            
            return self.dashboard_data
            
        except Exception as e:
            logger.error(f"Error getting dashboard data: {str(e)}")
            return self.dashboard_data
    
    async def _calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate performance metrics"""
        try:
            metrics = {
                "avg_generation_time": 0.0,
                "success_rate": 100.0,
                "tokens_per_hour": 0,
                "documents_per_hour": 0,
                "quality_trend": []
            }
            
            if self.workflow_engine:
                active_chains = self.workflow_engine.get_all_active_chains()
                
                if active_chains:
                    total_generation_time = 0
                    total_documents = 0
                    total_quality_scores = []
                    
                    for chain in active_chains:
                        for node in chain.nodes.values():
                            if "generation_time" in node.metadata:
                                total_generation_time += node.metadata["generation_time"]
                            if "quality_score" in node.metadata:
                                total_quality_scores.append(node.metadata["quality_score"])
                            total_documents += 1
                    
                    if total_documents > 0:
                        metrics["avg_generation_time"] = total_generation_time / total_documents
                        metrics["documents_per_hour"] = total_documents  # Simplified
                        
                        if total_quality_scores:
                            metrics["avg_quality_score"] = sum(total_quality_scores) / len(total_quality_scores)
                            metrics["quality_trend"] = total_quality_scores[-10:]  # Last 10 scores
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {str(e)}")
            return {}
    
    async def _get_recent_activity(self) -> List[Dict[str, Any]]:
        """Get recent activity data"""
        try:
            activity = []
            
            if self.workflow_engine:
                active_chains = self.workflow_engine.get_all_active_chains()
                
                for chain in active_chains:
                    for node in chain.nodes.values():
                        activity.append({
                            "type": "document_generated",
                            "chain_id": chain.id,
                            "chain_name": chain.name,
                            "document_title": node.title,
                            "timestamp": node.generated_at.isoformat(),
                            "quality_score": node.metadata.get("quality_score", 0.0)
                        })
                
                # Sort by timestamp (most recent first)
                activity.sort(key=lambda x: x["timestamp"], reverse=True)
                return activity[:20]  # Last 20 activities
            
            return activity
            
        except Exception as e:
            logger.error(f"Error getting recent activity: {str(e)}")
            return []

# Global dashboard manager
dashboard_manager = DashboardManager()

@dashboard_router.get("/", response_class=HTMLResponse)
async def dashboard_home(request: Request):
    """Main dashboard page"""
    try:
        # Create basic HTML template if it doesn't exist
        template_path = templates_dir / "dashboard.html"
        if not template_path.exists():
            await create_dashboard_template()
        
        dashboard_data = await dashboard_manager.get_dashboard_data()
        
        return templates.TemplateResponse("dashboard.html", {
            "request": request,
            "data": dashboard_data,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error rendering dashboard: {str(e)}")
        return HTMLResponse(f"<h1>Dashboard Error</h1><p>{str(e)}</p>", status_code=500)

@dashboard_router.get("/api/data")
async def get_dashboard_data():
    """API endpoint for dashboard data"""
    try:
        data = await dashboard_manager.get_dashboard_data()
        return JSONResponse(content=data)
        
    except Exception as e:
        logger.error(f"Error getting dashboard data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@dashboard_router.get("/api/chains")
async def get_chains_data():
    """API endpoint for chains data"""
    try:
        if dashboard_manager.workflow_engine:
            chains = dashboard_manager.workflow_engine.get_all_active_chains()
            chains_data = []
            
            for chain in chains:
                chain_data = {
                    "id": chain.id,
                    "name": chain.name,
                    "description": chain.description,
                    "status": chain.status,
                    "created_at": chain.created_at.isoformat(),
                    "updated_at": chain.updated_at.isoformat(),
                    "document_count": len(chain.nodes),
                    "settings": chain.settings
                }
                chains_data.append(chain_data)
            
            return JSONResponse(content={"chains": chains_data})
        
        return JSONResponse(content={"chains": []})
        
    except Exception as e:
        logger.error(f"Error getting chains data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@dashboard_router.get("/api/chain/{chain_id}")
async def get_chain_details(chain_id: str):
    """API endpoint for specific chain details"""
    try:
        if dashboard_manager.workflow_engine:
            chain = dashboard_manager.workflow_engine.get_workflow_chain(chain_id)
            if not chain:
                raise HTTPException(status_code=404, detail="Chain not found")
            
            # Get chain history
            history = dashboard_manager.workflow_engine.get_chain_history(chain_id)
            
            chain_data = {
                "id": chain.id,
                "name": chain.name,
                "description": chain.description,
                "status": chain.status,
                "created_at": chain.created_at.isoformat(),
                "updated_at": chain.updated_at.isoformat(),
                "settings": chain.settings,
                "documents": []
            }
            
            for doc in history:
                doc_data = {
                    "id": doc.id,
                    "title": doc.title,
                    "content": doc.content[:500] + "..." if len(doc.content) > 500 else doc.content,
                    "generated_at": doc.generated_at.isoformat(),
                    "parent_id": doc.parent_id,
                    "children_count": len(doc.children_ids),
                    "metadata": doc.metadata
                }
                chain_data["documents"].append(doc_data)
            
            return JSONResponse(content=chain_data)
        
        raise HTTPException(status_code=404, detail="Chain not found")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting chain details: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@dashboard_router.post("/api/chain/{chain_id}/action")
async def perform_chain_action(
    chain_id: str,
    action: str = Form(...)
):
    """API endpoint for performing chain actions"""
    try:
        if not dashboard_manager.workflow_engine:
            raise HTTPException(status_code=500, detail="Workflow engine not available")
        
        chain = dashboard_manager.workflow_engine.get_workflow_chain(chain_id)
        if not chain:
            raise HTTPException(status_code=404, detail="Chain not found")
        
        result = {"success": False, "message": ""}
        
        if action == "pause":
            success = dashboard_manager.workflow_engine.pause_workflow_chain(chain_id)
            result = {"success": success, "message": "Chain paused" if success else "Failed to pause chain"}
        
        elif action == "resume":
            success = dashboard_manager.workflow_engine.resume_workflow_chain(chain_id)
            result = {"success": success, "message": "Chain resumed" if success else "Failed to resume chain"}
        
        elif action == "complete":
            success = dashboard_manager.workflow_engine.complete_workflow_chain(chain_id)
            result = {"success": success, "message": "Chain completed" if success else "Failed to complete chain"}
        
        else:
            raise HTTPException(status_code=400, detail=f"Unknown action: {action}")
        
        return JSONResponse(content=result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error performing chain action: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def create_dashboard_template():
    """Create the dashboard HTML template"""
    template_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Document Workflow Chain Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .stat-card {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .stat-value {
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }
        .stat-label {
            color: #666;
            margin-top: 5px;
        }
        .charts-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 30px;
        }
        .chart-container {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .chains-table {
            background: white;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        table {
            width: 100%;
            border-collapse: collapse;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }
        th {
            background-color: #f8f9fa;
            font-weight: 600;
        }
        .status-badge {
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.8em;
            font-weight: 500;
        }
        .status-active { background-color: #d4edda; color: #155724; }
        .status-paused { background-color: #fff3cd; color: #856404; }
        .status-completed { background-color: #d1ecf1; color: #0c5460; }
        .btn {
            padding: 6px 12px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.8em;
            margin: 2px;
        }
        .btn-primary { background-color: #007bff; color: white; }
        .btn-success { background-color: #28a745; color: white; }
        .btn-warning { background-color: #ffc107; color: black; }
        .btn-danger { background-color: #dc3545; color: white; }
        .loading {
            text-align: center;
            padding: 20px;
            color: #666;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🚀 Document Workflow Chain Dashboard</h1>
        <p>Monitor and manage your AI-powered document generation workflows</p>
        <p>Last updated: <span id="lastUpdated">{{ timestamp }}</span></p>
    </div>

    <div class="stats-grid">
        <div class="stat-card">
            <div class="stat-value" id="totalChains">{{ data.total_chains }}</div>
            <div class="stat-label">Total Workflow Chains</div>
        </div>
        <div class="stat-card">
            <div class="stat-value" id="activeChains">{{ data.active_chains }}</div>
            <div class="stat-label">Active Chains</div>
        </div>
        <div class="stat-card">
            <div class="stat-value" id="totalDocuments">{{ data.total_documents }}</div>
            <div class="stat-label">Total Documents</div>
        </div>
        <div class="stat-card">
            <div class="stat-value" id="totalTokens">{{ data.total_tokens }}</div>
            <div class="stat-label">Total Tokens Used</div>
        </div>
        <div class="stat-card">
            <div class="stat-value" id="avgQuality">{{ data.avg_quality_score.toFixed(2) if data.avg_quality_score else '0.00' }}</div>
            <div class="stat-label">Average Quality Score</div>
        </div>
    </div>

    <div class="charts-grid">
        <div class="chart-container">
            <h3>Quality Score Trend</h3>
            <canvas id="qualityChart"></canvas>
        </div>
        <div class="chart-container">
            <h3>Document Generation Rate</h3>
            <canvas id="generationChart"></canvas>
        </div>
    </div>

    <div class="chains-table">
        <h3 style="padding: 20px; margin: 0;">Workflow Chains</h3>
        <div id="chainsTable">
            <div class="loading">Loading chains...</div>
        </div>
    </div>

    <script>
        // Initialize charts
        const qualityCtx = document.getElementById('qualityChart').getContext('2d');
        const qualityChart = new Chart(qualityCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Quality Score',
                    data: [],
                    borderColor: '#667eea',
                    backgroundColor: 'rgba(102, 126, 234, 0.1)',
                    tension: 0.4
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

        const generationCtx = document.getElementById('generationChart').getContext('2d');
        const generationChart = new Chart(generationCtx, {
            type: 'bar',
            data: {
                labels: [],
                datasets: [{
                    label: 'Documents Generated',
                    data: [],
                    backgroundColor: '#764ba2'
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

        // Load chains data
        async function loadChains() {
            try {
                const response = await fetch('/dashboard/api/chains');
                const data = await response.json();
                
                if (data.chains && data.chains.length > 0) {
                    const tableHTML = `
                        <table>
                            <thead>
                                <tr>
                                    <th>Name</th>
                                    <th>Status</th>
                                    <th>Documents</th>
                                    <th>Created</th>
                                    <th>Actions</th>
                                </tr>
                            </thead>
                            <tbody>
                                ${data.chains.map(chain => `
                                    <tr>
                                        <td>
                                            <strong>${chain.name}</strong><br>
                                            <small>${chain.description}</small>
                                        </td>
                                        <td>
                                            <span class="status-badge status-${chain.status}">
                                                ${chain.status}
                                            </span>
                                        </td>
                                        <td>${chain.document_count}</td>
                                        <td>${new Date(chain.created_at).toLocaleDateString()}</td>
                                        <td>
                                            <button class="btn btn-primary" onclick="viewChain('${chain.id}')">View</button>
                                            ${chain.status === 'active' ? 
                                                `<button class="btn btn-warning" onclick="performAction('${chain.id}', 'pause')">Pause</button>` :
                                                `<button class="btn btn-success" onclick="performAction('${chain.id}', 'resume')">Resume</button>`
                                            }
                                            <button class="btn btn-danger" onclick="performAction('${chain.id}', 'complete')">Complete</button>
                                        </td>
                                    </tr>
                                `).join('')}
                            </tbody>
                        </table>
                    `;
                    document.getElementById('chainsTable').innerHTML = tableHTML;
                } else {
                    document.getElementById('chainsTable').innerHTML = '<div class="loading">No workflow chains found</div>';
                }
            } catch (error) {
                console.error('Error loading chains:', error);
                document.getElementById('chainsTable').innerHTML = '<div class="loading">Error loading chains</div>';
            }
        }

        // Perform chain action
        async function performAction(chainId, action) {
            try {
                const formData = new FormData();
                formData.append('action', action);
                
                const response = await fetch(`/dashboard/api/chain/${chainId}/action`, {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                
                if (result.success) {
                    alert(result.message);
                    loadChains(); // Reload chains
                } else {
                    alert('Error: ' + result.message);
                }
            } catch (error) {
                console.error('Error performing action:', error);
                alert('Error performing action');
            }
        }

        // View chain details
        function viewChain(chainId) {
            window.open(`/dashboard/api/chain/${chainId}`, '_blank');
        }

        // Auto-refresh data every 30 seconds
        setInterval(() => {
            loadChains();
            document.getElementById('lastUpdated').textContent = new Date().toLocaleString();
        }, 30000);

        // Initial load
        loadChains();
    </script>
</body>
</html>
    """
    
    async with aiofiles.open(templates_dir / "dashboard.html", "w", encoding="utf-8") as f:
        await f.write(template_content)
    
    logger.info("Created dashboard template")

def initialize_dashboard(workflow_engine, database_manager=None):
    """Initialize dashboard with workflow engine and database manager"""
    global dashboard_manager
    dashboard_manager = DashboardManager(workflow_engine, database_manager)
    logger.info("Dashboard initialized")

# Example usage
if __name__ == "__main__":
    print("Dashboard module loaded successfully")
    print("Use initialize_dashboard() to set up the dashboard with your workflow engine")


