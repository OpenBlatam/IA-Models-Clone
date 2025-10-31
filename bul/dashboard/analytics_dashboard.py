"""
BUL Analytics Dashboard
======================

Advanced analytics and metrics dashboard for the BUL system.
"""

import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import json
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np

from ..utils import get_data_processor, get_logger, get_cache_manager
from ..config import get_config
from ..core import BusinessArea, DocumentType

logger = get_logger(__name__)

# Create router for dashboard endpoints
dashboard_router = APIRouter(prefix="/dashboard", tags=["Analytics Dashboard"])

@dataclass
class DashboardMetrics:
    """Dashboard metrics data structure"""
    total_documents: int
    total_words: int
    average_processing_time: float
    success_rate: float
    popular_business_areas: List[Dict[str, Any]]
    popular_document_types: List[Dict[str, Any]]
    agent_performance: List[Dict[str, Any]]
    daily_trends: List[Dict[str, Any]]
    system_health: Dict[str, Any]
    cache_performance: Dict[str, Any]

class AnalyticsRequest(BaseModel):
    """Request model for analytics queries"""
    start_date: Optional[datetime] = Field(None, description="Start date for analytics")
    end_date: Optional[datetime] = Field(None, description="End date for analytics")
    business_area: Optional[BusinessArea] = Field(None, description="Filter by business area")
    document_type: Optional[DocumentType] = Field(None, description="Filter by document type")
    agent_id: Optional[str] = Field(None, description="Filter by agent ID")

class AnalyticsResponse(BaseModel):
    """Response model for analytics data"""
    metrics: DashboardMetrics
    generated_at: datetime
    query_params: Dict[str, Any]

class AdvancedAnalytics:
    """Advanced analytics engine for BUL system"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.data_processor = get_data_processor()
        self.cache_manager = get_cache_manager()
        self.config = get_config()
    
    async def get_dashboard_metrics(self, request: AnalyticsRequest) -> DashboardMetrics:
        """Get comprehensive dashboard metrics"""
        try:
            # Get cached data or generate new
            cache_key = f"dashboard_metrics:{hash(str(request.dict()))}"
            cached_metrics = self.cache_manager.get(cache_key)
            
            if cached_metrics:
                self.logger.debug("Returning cached dashboard metrics")
                return DashboardMetrics(**cached_metrics)
            
            # Generate metrics
            metrics = await self._generate_metrics(request)
            
            # Cache for 5 minutes
            self.cache_manager.set(cache_key, asdict(metrics), ttl=300)
            
            return metrics
        
        except Exception as e:
            self.logger.error(f"Error generating dashboard metrics: {e}")
            raise HTTPException(status_code=500, detail="Failed to generate analytics")
    
    async def _generate_metrics(self, request: AnalyticsRequest) -> DashboardMetrics:
        """Generate comprehensive metrics"""
        # Simulate data processing (in real implementation, this would query the database)
        mock_data = await self._get_mock_analytics_data(request)
        
        # Process data with pandas
        df = pd.DataFrame(mock_data)
        
        if df.empty:
            return self._get_empty_metrics()
        
        # Calculate metrics
        total_documents = len(df)
        total_words = df['word_count'].sum()
        average_processing_time = df['processing_time'].mean()
        success_rate = (df['success'] == True).mean()
        
        # Popular business areas
        popular_business_areas = df.groupby('business_area').agg({
            'id': 'count',
            'word_count': 'mean',
            'processing_time': 'mean',
            'success': 'mean'
        }).round(2).to_dict('index')
        
        # Popular document types
        popular_document_types = df.groupby('document_type').agg({
            'id': 'count',
            'word_count': 'mean',
            'processing_time': 'mean',
            'success': 'mean'
        }).round(2).to_dict('index')
        
        # Agent performance
        agent_performance = df.groupby('agent_used').agg({
            'id': 'count',
            'processing_time': 'mean',
            'success': 'mean',
            'word_count': 'mean'
        }).round(2).to_dict('index')
        
        # Daily trends
        df['date'] = pd.to_datetime(df['created_at']).dt.date
        daily_trends = df.groupby('date').agg({
            'id': 'count',
            'word_count': 'sum',
            'processing_time': 'mean',
            'success': 'mean'
        }).round(2).to_dict('index')
        
        # System health (mock data)
        system_health = {
            "api_status": "healthy",
            "database_status": "healthy",
            "cache_status": "healthy",
            "memory_usage": 65.2,
            "cpu_usage": 23.8,
            "disk_usage": 45.1
        }
        
        # Cache performance
        cache_stats = self.cache_manager.get_stats()
        cache_performance = {
            "hit_rate": cache_stats["hits"] / max(cache_stats["hits"] + cache_stats["misses"], 1),
            "total_hits": cache_stats["hits"],
            "total_misses": cache_stats["misses"],
            "cache_size": cache_stats["size"],
            "evictions": cache_stats["evictions"]
        }
        
        return DashboardMetrics(
            total_documents=total_documents,
            total_words=int(total_words),
            average_processing_time=round(average_processing_time, 2),
            success_rate=round(success_rate, 3),
            popular_business_areas=list(popular_business_areas.items()),
            popular_document_types=list(popular_document_types.items()),
            agent_performance=list(agent_performance.items()),
            daily_trends=list(daily_trends.items()),
            system_health=system_health,
            cache_performance=cache_performance
        )
    
    async def _get_mock_analytics_data(self, request: AnalyticsRequest) -> List[Dict[str, Any]]:
        """Get mock analytics data (replace with real database queries)"""
        # Generate mock data for demonstration
        import random
        from datetime import datetime, timedelta
        
        data = []
        start_date = request.start_date or datetime.now() - timedelta(days=30)
        end_date = request.end_date or datetime.now()
        
        business_areas = [area.value for area in BusinessArea]
        document_types = [doc_type.value for doc_type in DocumentType]
        agents = ["marketing_agent", "sales_agent", "finance_agent", "hr_agent", "legal_agent"]
        
        for i in range(100):  # Generate 100 mock records
            created_at = start_date + timedelta(
                days=random.randint(0, (end_date - start_date).days),
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59)
            )
            
            data.append({
                "id": f"doc_{i:04d}",
                "business_area": random.choice(business_areas),
                "document_type": random.choice(document_types),
                "agent_used": random.choice(agents),
                "word_count": random.randint(100, 5000),
                "processing_time": round(random.uniform(2.0, 15.0), 2),
                "success": random.choice([True, True, True, False]),  # 75% success rate
                "created_at": created_at.isoformat()
            })
        
        return data
    
    def _get_empty_metrics(self) -> DashboardMetrics:
        """Return empty metrics when no data is available"""
        return DashboardMetrics(
            total_documents=0,
            total_words=0,
            average_processing_time=0.0,
            success_rate=0.0,
            popular_business_areas=[],
            popular_document_types=[],
            agent_performance=[],
            daily_trends=[],
            system_health={},
            cache_performance={}
        )

# Global analytics instance
_analytics: Optional[AdvancedAnalytics] = None

def get_analytics() -> AdvancedAnalytics:
    """Get the global analytics instance"""
    global _analytics
    if _analytics is None:
        _analytics = AdvancedAnalytics()
    return _analytics

# Dashboard endpoints
@dashboard_router.get("/", response_class=HTMLResponse)
async def dashboard_home():
    """Serve the main dashboard HTML page"""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>BUL Analytics Dashboard</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }
            .container { max-width: 1200px; margin: 0 auto; }
            .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }
            .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 30px; }
            .metric-card { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            .metric-value { font-size: 2em; font-weight: bold; color: #667eea; }
            .metric-label { color: #666; margin-top: 5px; }
            .chart-container { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin-bottom: 20px; }
            .chart-title { font-size: 1.2em; font-weight: bold; margin-bottom: 15px; color: #333; }
            .loading { text-align: center; padding: 40px; color: #666; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🚀 BUL Analytics Dashboard</h1>
                <p>Real-time insights into your document generation system</p>
            </div>
            
            <div id="loading" class="loading">Loading analytics data...</div>
            
            <div id="dashboard" style="display: none;">
                <div class="metrics-grid" id="metrics-grid"></div>
                
                <div class="chart-container">
                    <div class="chart-title">📊 Daily Document Generation Trends</div>
                    <canvas id="trendsChart" width="400" height="200"></canvas>
                </div>
                
                <div class="chart-container">
                    <div class="chart-title">🎯 Business Area Distribution</div>
                    <canvas id="businessAreaChart" width="400" height="200"></canvas>
                </div>
                
                <div class="chart-container">
                    <div class="chart-title">⚡ Agent Performance</div>
                    <canvas id="agentChart" width="400" height="200"></canvas>
                </div>
            </div>
        </div>
        
        <script>
            async function loadDashboard() {
                try {
                    const response = await fetch('/dashboard/api/metrics');
                    const data = await response.json();
                    
                    displayMetrics(data.metrics);
                    createCharts(data.metrics);
                    
                    document.getElementById('loading').style.display = 'none';
                    document.getElementById('dashboard').style.display = 'block';
                } catch (error) {
                    console.error('Error loading dashboard:', error);
                    document.getElementById('loading').innerHTML = 'Error loading dashboard data';
                }
            }
            
            function displayMetrics(metrics) {
                const metricsGrid = document.getElementById('metrics-grid');
                metricsGrid.innerHTML = `
                    <div class="metric-card">
                        <div class="metric-value">${metrics.total_documents}</div>
                        <div class="metric-label">Total Documents</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">${metrics.total_words.toLocaleString()}</div>
                        <div class="metric-label">Total Words</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">${metrics.average_processing_time}s</div>
                        <div class="metric-label">Avg Processing Time</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">${(metrics.success_rate * 100).toFixed(1)}%</div>
                        <div class="metric-label">Success Rate</div>
                    </div>
                `;
            }
            
            function createCharts(metrics) {
                // Trends Chart
                const trendsCtx = document.getElementById('trendsChart').getContext('2d');
                const trendsData = Object.entries(metrics.daily_trends).slice(-7);
                new Chart(trendsCtx, {
                    type: 'line',
                    data: {
                        labels: trendsData.map(([date]) => new Date(date).toLocaleDateString()),
                        datasets: [{
                            label: 'Documents Generated',
                            data: trendsData.map(([, data]) => data.id),
                            borderColor: '#667eea',
                            backgroundColor: 'rgba(102, 126, 234, 0.1)',
                            tension: 0.4
                        }]
                    },
                    options: {
                        responsive: true,
                        scales: {
                            y: { beginAtZero: true }
                        }
                    }
                });
                
                // Business Area Chart
                const businessCtx = document.getElementById('businessAreaChart').getContext('2d');
                const businessData = Object.entries(metrics.popular_business_areas);
                new Chart(businessCtx, {
                    type: 'doughnut',
                    data: {
                        labels: businessData.map(([area]) => area),
                        datasets: [{
                            data: businessData.map(([, data]) => data.id),
                            backgroundColor: [
                                '#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe'
                            ]
                        }]
                    },
                    options: {
                        responsive: true,
                        plugins: {
                            legend: { position: 'bottom' }
                        }
                    }
                });
                
                // Agent Performance Chart
                const agentCtx = document.getElementById('agentChart').getContext('2d');
                const agentData = Object.entries(metrics.agent_performance);
                new Chart(agentCtx, {
                    type: 'bar',
                    data: {
                        labels: agentData.map(([agent]) => agent.replace('_agent', '')),
                        datasets: [{
                            label: 'Success Rate (%)',
                            data: agentData.map(([, data]) => (data.success * 100).toFixed(1)),
                            backgroundColor: '#667eea'
                        }]
                    },
                    options: {
                        responsive: true,
                        scales: {
                            y: { beginAtZero: true, max: 100 }
                        }
                    }
                });
            }
            
            // Load dashboard on page load
            loadDashboard();
            
            // Refresh every 30 seconds
            setInterval(loadDashboard, 30000);
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@dashboard_router.get("/api/metrics", response_model=AnalyticsResponse)
async def get_analytics_metrics(
    start_date: Optional[datetime] = Query(None, description="Start date for analytics"),
    end_date: Optional[datetime] = Query(None, description="End date for analytics"),
    business_area: Optional[BusinessArea] = Query(None, description="Filter by business area"),
    document_type: Optional[DocumentType] = Query(None, description="Filter by document type"),
    agent_id: Optional[str] = Query(None, description="Filter by agent ID")
):
    """Get analytics metrics for the dashboard"""
    try:
        analytics = get_analytics()
        
        request = AnalyticsRequest(
            start_date=start_date,
            end_date=end_date,
            business_area=business_area,
            document_type=document_type,
            agent_id=agent_id
        )
        
        metrics = await analytics.get_dashboard_metrics(request)
        
        return AnalyticsResponse(
            metrics=metrics,
            generated_at=datetime.now(),
            query_params=request.dict()
        )
    
    except Exception as e:
        logger.error(f"Error getting analytics metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get analytics metrics")

@dashboard_router.get("/api/export")
async def export_analytics_data(
    format: str = Query("json", description="Export format (json, csv)"),
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None)
):
    """Export analytics data in various formats"""
    try:
        analytics = get_analytics()
        
        request = AnalyticsRequest(start_date=start_date, end_date=end_date)
        metrics = await analytics.get_dashboard_metrics(request)
        
        if format.lower() == "csv":
            # Convert to CSV format
            import io
            import csv
            
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Write metrics data
            writer.writerow(["Metric", "Value"])
            writer.writerow(["Total Documents", metrics.total_documents])
            writer.writerow(["Total Words", metrics.total_words])
            writer.writerow(["Average Processing Time", metrics.average_processing_time])
            writer.writerow(["Success Rate", metrics.success_rate])
            
            csv_content = output.getvalue()
            output.close()
            
            return {"data": csv_content, "format": "csv"}
        
        else:  # JSON format
            return {
                "data": asdict(metrics),
                "format": "json",
                "exported_at": datetime.now().isoformat()
            }
    
    except Exception as e:
        logger.error(f"Error exporting analytics data: {e}")
        raise HTTPException(status_code=500, detail="Failed to export analytics data")




