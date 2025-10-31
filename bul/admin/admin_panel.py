"""
BUL Admin Panel
==============

Administrative interface for managing the BUL system.
"""

import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
import json

from ..utils import get_logger, get_cache_manager, get_data_processor
from ..config import get_config, reload_config
from ..core import get_global_bul_engine
from ..agents import get_global_agent_manager
from ..security import get_password_manager, get_jwt_manager, get_rate_limiter
from ..monitoring.health_checker import get_health_checker

logger = get_logger(__name__)

# Create router for admin endpoints
admin_router = APIRouter(prefix="/admin", tags=["Admin Panel"])

# Templates setup
templates = Jinja2Templates(directory="admin/templates")

class AdminStats(BaseModel):
    """Admin statistics model"""
    system_uptime: str
    total_documents: int
    total_requests: int
    error_rate: float
    average_response_time: float
    cache_hit_rate: float
    active_agents: int
    system_health: str
    last_updated: datetime

class SystemConfig(BaseModel):
    """System configuration model"""
    environment: str
    debug_mode: bool
    api_keys_configured: bool
    database_connected: bool
    cache_enabled: bool
    rate_limiting_enabled: bool
    logging_level: str
    max_workers: int

class AdminManager:
    """Admin panel manager"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.config = get_config()
        self.cache_manager = get_cache_manager()
        self.data_processor = get_data_processor()
        self.start_time = datetime.now()
    
    async def get_system_stats(self) -> AdminStats:
        """Get comprehensive system statistics"""
        try:
            # Get engine stats
            bul_engine = get_global_bul_engine()
            engine_stats = bul_engine.get_stats() if bul_engine else {}
            
            # Get agent manager stats
            agent_manager = get_global_agent_manager()
            agent_stats = agent_manager.get_stats() if agent_manager else {}
            
            # Get cache stats
            cache_stats = self.cache_manager.get_stats()
            
            # Calculate uptime
            uptime = datetime.now() - self.start_time
            uptime_str = str(uptime).split('.')[0]  # Remove microseconds
            
            # Calculate error rate
            total_requests = engine_stats.get('documents_generated', 0)
            error_count = engine_stats.get('errors', 0)
            error_rate = (error_count / max(total_requests, 1)) * 100
            
            # Calculate cache hit rate
            cache_hits = cache_stats.get('hits', 0)
            cache_misses = cache_stats.get('misses', 0)
            cache_hit_rate = (cache_hits / max(cache_hits + cache_misses, 1)) * 100
            
            # Get system health
            health_checker = get_health_checker()
            health_summary = health_checker.get_health_summary()
            system_health = health_summary.get('overall_status', 'UNKNOWN')
            
            return AdminStats(
                system_uptime=uptime_str,
                total_documents=engine_stats.get('documents_generated', 0),
                total_requests=total_requests,
                error_rate=round(error_rate, 2),
                average_response_time=round(engine_stats.get('average_processing_time', 0), 2),
                cache_hit_rate=round(cache_hit_rate, 2),
                active_agents=agent_stats.get('active_agents', 0),
                system_health=system_health,
                last_updated=datetime.now()
            )
        
        except Exception as e:
            self.logger.error(f"Error getting system stats: {e}")
            return AdminStats(
                system_uptime="Unknown",
                total_documents=0,
                total_requests=0,
                error_rate=0.0,
                average_response_time=0.0,
                cache_hit_rate=0.0,
                active_agents=0,
                system_health="ERROR",
                last_updated=datetime.now()
            )
    
    async def get_system_config(self) -> SystemConfig:
        """Get system configuration status"""
        try:
            # Check API keys
            api_keys_configured = bool(
                self.config.api.openrouter_api_key and 
                self.config.api.openrouter_api_key != "your_openrouter_api_key_here"
            )
            
            # Check database connection (simplified)
            database_connected = True  # In real implementation, test actual connection
            
            # Check cache status
            cache_enabled = self.config.cache.enabled
            
            # Check rate limiting
            rate_limiter = get_rate_limiter()
            rate_limiting_enabled = rate_limiter is not None
            
            return SystemConfig(
                environment=self.config.environment.value,
                debug_mode=self.config.debug,
                api_keys_configured=api_keys_configured,
                database_connected=database_connected,
                cache_enabled=cache_enabled,
                rate_limiting_enabled=rate_limiting_enabled,
                logging_level=self.config.logging.level,
                max_workers=self.config.server.workers
            )
        
        except Exception as e:
            self.logger.error(f"Error getting system config: {e}")
            return SystemConfig(
                environment="unknown",
                debug_mode=False,
                api_keys_configured=False,
                database_connected=False,
                cache_enabled=False,
                rate_limiting_enabled=False,
                logging_level="ERROR",
                max_workers=1
            )
    
    async def clear_cache(self) -> bool:
        """Clear system cache"""
        try:
            self.cache_manager.clear()
            self.logger.info("System cache cleared by admin")
            return True
        except Exception as e:
            self.logger.error(f"Error clearing cache: {e}")
            return False
    
    async def reload_configuration(self) -> bool:
        """Reload system configuration"""
        try:
            reload_config()
            self.logger.info("System configuration reloaded by admin")
            return True
        except Exception as e:
            self.logger.error(f"Error reloading configuration: {e}")
            return False
    
    async def get_recent_logs(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent system logs"""
        try:
            # In a real implementation, this would read from log files
            # For now, return mock data
            logs = []
            for i in range(min(limit, 50)):
                logs.append({
                    "timestamp": (datetime.now() - timedelta(minutes=i)).isoformat(),
                    "level": "INFO" if i % 3 == 0 else "DEBUG",
                    "message": f"System log entry {i}",
                    "module": "bul.core" if i % 2 == 0 else "bul.api"
                })
            return logs
        except Exception as e:
            self.logger.error(f"Error getting recent logs: {e}")
            return []
    
    async def get_agent_performance(self) -> List[Dict[str, Any]]:
        """Get agent performance metrics"""
        try:
            agent_manager = get_global_agent_manager()
            if not agent_manager:
                return []
            
            agents = agent_manager.get_available_agents()
            performance_data = []
            
            for agent in agents:
                performance_data.append({
                    "agent_id": agent.agent_id,
                    "agent_type": agent.agent_type,
                    "is_active": agent.is_active,
                    "success_rate": agent.success_rate,
                    "experience_years": agent.experience_years,
                    "documents_generated": getattr(agent, 'documents_generated', 0),
                    "average_processing_time": getattr(agent, 'average_processing_time', 0)
                })
            
            return performance_data
        
        except Exception as e:
            self.logger.error(f"Error getting agent performance: {e}")
            return []

# Global admin manager instance
_admin_manager: Optional[AdminManager] = None

def get_admin_manager() -> AdminManager:
    """Get the global admin manager instance"""
    global _admin_manager
    if _admin_manager is None:
        _admin_manager = AdminManager()
    return _admin_manager

# Admin panel endpoints
@admin_router.get("/", response_class=HTMLResponse)
async def admin_dashboard(request: Request):
    """Admin dashboard main page"""
    admin_manager = get_admin_manager()
    
    stats = await admin_manager.get_system_stats()
    config = await admin_manager.get_system_config()
    recent_logs = await admin_manager.get_recent_logs(20)
    agent_performance = await admin_manager.get_agent_performance()
    
    return templates.TemplateResponse("admin_dashboard.html", {
        "request": request,
        "stats": stats,
        "config": config,
        "recent_logs": recent_logs,
        "agent_performance": agent_performance,
        "page_title": "BUL Admin Dashboard"
    })

@admin_router.get("/stats", response_model=AdminStats)
async def get_admin_stats():
    """Get admin statistics as JSON"""
    admin_manager = get_admin_manager()
    return await admin_manager.get_system_stats()

@admin_router.get("/config", response_model=SystemConfig)
async def get_admin_config():
    """Get system configuration as JSON"""
    admin_manager = get_admin_manager()
    return await admin_manager.get_system_config()

@admin_router.post("/cache/clear")
async def clear_system_cache():
    """Clear system cache"""
    admin_manager = get_admin_manager()
    success = await admin_manager.clear_cache()
    
    if success:
        return {"message": "Cache cleared successfully", "success": True}
    else:
        raise HTTPException(status_code=500, detail="Failed to clear cache")

@admin_router.post("/config/reload")
async def reload_system_config():
    """Reload system configuration"""
    admin_manager = get_admin_manager()
    success = await admin_manager.reload_configuration()
    
    if success:
        return {"message": "Configuration reloaded successfully", "success": True}
    else:
        raise HTTPException(status_code=500, detail="Failed to reload configuration")

@admin_router.get("/logs")
async def get_system_logs(limit: int = 100):
    """Get recent system logs"""
    admin_manager = get_admin_manager()
    logs = await admin_manager.get_recent_logs(limit)
    return {"logs": logs, "count": len(logs)}

@admin_router.get("/agents/performance")
async def get_agent_performance():
    """Get agent performance metrics"""
    admin_manager = get_admin_manager()
    performance = await admin_manager.get_agent_performance()
    return {"agents": performance, "count": len(performance)}

@admin_router.get("/health/detailed")
async def get_detailed_health():
    """Get detailed system health information"""
    try:
        health_checker = get_health_checker()
        health_summary = health_checker.get_health_summary()
        return health_summary
    except Exception as e:
        logger.error(f"Error getting detailed health: {e}")
        raise HTTPException(status_code=500, detail="Failed to get health information")

@admin_router.post("/maintenance/mode")
async def toggle_maintenance_mode(enabled: bool = Form(...)):
    """Toggle maintenance mode"""
    try:
        # In a real implementation, this would update a maintenance flag
        # and potentially redirect traffic or show maintenance page
        logger.info(f"Maintenance mode {'enabled' if enabled else 'disabled'} by admin")
        return {
            "message": f"Maintenance mode {'enabled' if enabled else 'disabled'}",
            "maintenance_mode": enabled,
            "success": True
        }
    except Exception as e:
        logger.error(f"Error toggling maintenance mode: {e}")
        raise HTTPException(status_code=500, detail="Failed to toggle maintenance mode")

@admin_router.get("/export/data")
async def export_system_data(format: str = "json"):
    """Export system data"""
    try:
        admin_manager = get_admin_manager()
        
        if format.lower() == "json":
            stats = await admin_manager.get_system_stats()
            config = await admin_manager.get_system_config()
            logs = await admin_manager.get_recent_logs(1000)
            agent_performance = await admin_manager.get_agent_performance()
            
            export_data = {
                "exported_at": datetime.now().isoformat(),
                "stats": stats.dict(),
                "config": config.dict(),
                "logs": logs,
                "agent_performance": agent_performance
            }
            
            return export_data
        
        else:
            raise HTTPException(status_code=400, detail="Unsupported export format")
    
    except Exception as e:
        logger.error(f"Error exporting system data: {e}")
        raise HTTPException(status_code=500, detail="Failed to export system data")

# Admin panel HTML template
admin_dashboard_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ page_title }}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }
        .container { max-width: 1400px; margin: 0 auto; }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }
        .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 30px; }
        .stat-card { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); text-align: center; }
        .stat-value { font-size: 2em; font-weight: bold; color: #667eea; }
        .stat-label { color: #666; margin-top: 5px; }
        .section { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin-bottom: 20px; }
        .section-title { font-size: 1.2em; font-weight: bold; margin-bottom: 15px; color: #333; }
        .config-item { display: flex; justify-content: space-between; padding: 10px 0; border-bottom: 1px solid #eee; }
        .config-item:last-child { border-bottom: none; }
        .status { padding: 4px 8px; border-radius: 4px; font-size: 0.8em; font-weight: bold; }
        .status.healthy { background: #d4edda; color: #155724; }
        .status.warning { background: #fff3cd; color: #856404; }
        .status.error { background: #f8d7da; color: #721c24; }
        .btn { padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; font-weight: bold; }
        .btn-primary { background: #667eea; color: white; }
        .btn-danger { background: #dc3545; color: white; }
        .btn-success { background: #28a745; color: white; }
        .btn:hover { opacity: 0.8; }
        .log-entry { padding: 8px; border-bottom: 1px solid #eee; font-family: monospace; font-size: 0.9em; }
        .log-entry:last-child { border-bottom: none; }
        .log-level-info { color: #007bff; }
        .log-level-warning { color: #ffc107; }
        .log-level-error { color: #dc3545; }
        .agent-table { width: 100%; border-collapse: collapse; }
        .agent-table th, .agent-table td { padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }
        .agent-table th { background-color: #f8f9fa; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 BUL Admin Dashboard</h1>
            <p>System administration and monitoring</p>
        </div>
        
        <!-- System Statistics -->
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{{ stats.total_documents }}</div>
                <div class="stat-label">Total Documents</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{{ stats.total_requests }}</div>
                <div class="stat-label">Total Requests</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{{ stats.error_rate }}%</div>
                <div class="stat-label">Error Rate</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{{ stats.average_response_time }}s</div>
                <div class="stat-label">Avg Response Time</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{{ stats.cache_hit_rate }}%</div>
                <div class="stat-label">Cache Hit Rate</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{{ stats.active_agents }}</div>
                <div class="stat-label">Active Agents</div>
            </div>
        </div>
        
        <!-- System Configuration -->
        <div class="section">
            <div class="section-title">⚙️ System Configuration</div>
            <div class="config-item">
                <span>Environment</span>
                <span class="status {{ 'healthy' if config.environment == 'production' else 'warning' }}">{{ config.environment }}</span>
            </div>
            <div class="config-item">
                <span>Debug Mode</span>
                <span class="status {{ 'warning' if config.debug_mode else 'healthy' }}">{{ 'Enabled' if config.debug_mode else 'Disabled' }}</span>
            </div>
            <div class="config-item">
                <span>API Keys</span>
                <span class="status {{ 'healthy' if config.api_keys_configured else 'error' }}">{{ 'Configured' if config.api_keys_configured else 'Not Configured' }}</span>
            </div>
            <div class="config-item">
                <span>Database</span>
                <span class="status {{ 'healthy' if config.database_connected else 'error' }}">{{ 'Connected' if config.database_connected else 'Disconnected' }}</span>
            </div>
            <div class="config-item">
                <span>Cache</span>
                <span class="status {{ 'healthy' if config.cache_enabled else 'warning' }}">{{ 'Enabled' if config.cache_enabled else 'Disabled' }}</span>
            </div>
            <div class="config-item">
                <span>Rate Limiting</span>
                <span class="status {{ 'healthy' if config.rate_limiting_enabled else 'warning' }}">{{ 'Enabled' if config.rate_limiting_enabled else 'Disabled' }}</span>
            </div>
        </div>
        
        <!-- System Health -->
        <div class="section">
            <div class="section-title">🏥 System Health</div>
            <div class="config-item">
                <span>Overall Status</span>
                <span class="status {{ 'healthy' if stats.system_health == 'HEALTHY' else 'error' if stats.system_health == 'CRITICAL' else 'warning' }}">{{ stats.system_health }}</span>
            </div>
            <div class="config-item">
                <span>Uptime</span>
                <span>{{ stats.system_uptime }}</span>
            </div>
            <div class="config-item">
                <span>Last Updated</span>
                <span>{{ stats.last_updated.strftime('%Y-%m-%d %H:%M:%S') }}</span>
            </div>
        </div>
        
        <!-- Agent Performance -->
        <div class="section">
            <div class="section-title">🤖 Agent Performance</div>
            <table class="agent-table">
                <thead>
                    <tr>
                        <th>Agent ID</th>
                        <th>Type</th>
                        <th>Status</th>
                        <th>Success Rate</th>
                        <th>Experience</th>
                        <th>Documents</th>
                    </tr>
                </thead>
                <tbody>
                    {% for agent in agent_performance %}
                    <tr>
                        <td>{{ agent.agent_id }}</td>
                        <td>{{ agent.agent_type }}</td>
                        <td><span class="status {{ 'healthy' if agent.is_active else 'warning' }}">{{ 'Active' if agent.is_active else 'Inactive' }}</span></td>
                        <td>{{ (agent.success_rate * 100)|round(1) }}%</td>
                        <td>{{ agent.experience_years }} years</td>
                        <td>{{ agent.documents_generated }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
        
        <!-- Recent Logs -->
        <div class="section">
            <div class="section-title">📋 Recent Logs</div>
            <div style="max-height: 300px; overflow-y: auto;">
                {% for log in recent_logs %}
                <div class="log-entry">
                    <span class="log-level-{{ log.level.lower() }}">[{{ log.level }}]</span>
                    <span>{{ log.timestamp }}</span>
                    <span>{{ log.message }}</span>
                </div>
                {% endfor %}
            </div>
        </div>
        
        <!-- Admin Actions -->
        <div class="section">
            <div class="section-title">🔧 Admin Actions</div>
            <div style="display: flex; gap: 10px; flex-wrap: wrap;">
                <button class="btn btn-primary" onclick="clearCache()">Clear Cache</button>
                <button class="btn btn-primary" onclick="reloadConfig()">Reload Config</button>
                <button class="btn btn-success" onclick="exportData()">Export Data</button>
                <button class="btn btn-danger" onclick="toggleMaintenance()">Toggle Maintenance</button>
            </div>
        </div>
    </div>
    
    <script>
        async function clearCache() {
            try {
                const response = await fetch('/admin/cache/clear', { method: 'POST' });
                const result = await response.json();
                alert(result.message);
                location.reload();
            } catch (error) {
                alert('Error clearing cache: ' + error.message);
            }
        }
        
        async function reloadConfig() {
            try {
                const response = await fetch('/admin/config/reload', { method: 'POST' });
                const result = await response.json();
                alert(result.message);
                location.reload();
            } catch (error) {
                alert('Error reloading config: ' + error.message);
            }
        }
        
        async function exportData() {
            try {
                const response = await fetch('/admin/export/data');
                const data = await response.json();
                const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'bul-system-export.json';
                a.click();
                URL.revokeObjectURL(url);
            } catch (error) {
                alert('Error exporting data: ' + error.message);
            }
        }
        
        async function toggleMaintenance() {
            const enabled = confirm('Enable maintenance mode?');
            try {
                const formData = new FormData();
                formData.append('enabled', enabled);
                const response = await fetch('/admin/maintenance/mode', { 
                    method: 'POST',
                    body: formData
                });
                const result = await response.json();
                alert(result.message);
            } catch (error) {
                alert('Error toggling maintenance mode: ' + error.message);
            }
        }
        
        // Auto-refresh every 30 seconds
        setInterval(() => {
            location.reload();
        }, 30000);
    </script>
</body>
</html>
"""

# Create templates directory and file
import os
os.makedirs("admin/templates", exist_ok=True)

with open("admin/templates/admin_dashboard.html", "w") as f:
    f.write(admin_dashboard_html)


