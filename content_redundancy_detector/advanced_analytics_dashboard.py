"""
Advanced Analytics Dashboard and Reporting System
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import uuid

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import redis.asyncio as redis
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

from config import settings

logger = logging.getLogger(__name__)


class AnalyticsQuery(BaseModel):
    """Model for analytics queries"""
    query_type: str = Field(..., description="Type of analytics query")
    filters: Dict[str, Any] = Field(default_factory=dict, description="Query filters")
    date_range: Optional[Tuple[str, str]] = Field(default=None, description="Date range")
    group_by: Optional[str] = Field(default=None, description="Group by field")
    aggregation: str = Field(default="count", description="Aggregation type")
    limit: int = Field(default=100, description="Result limit")


class DashboardWidget(BaseModel):
    """Model for dashboard widgets"""
    widget_id: str = Field(..., description="Widget ID")
    widget_type: str = Field(..., description="Widget type")
    title: str = Field(..., description="Widget title")
    query: AnalyticsQuery = Field(..., description="Widget query")
    position: Dict[str, int] = Field(..., description="Widget position")
    size: Dict[str, int] = Field(..., description="Widget size")
    refresh_interval: int = Field(default=300, description="Refresh interval in seconds")
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())


class Dashboard(BaseModel):
    """Model for analytics dashboard"""
    dashboard_id: str = Field(..., description="Dashboard ID")
    name: str = Field(..., description="Dashboard name")
    description: str = Field(..., description="Dashboard description")
    widgets: List[DashboardWidget] = Field(..., description="Dashboard widgets")
    is_public: bool = Field(default=False, description="Is dashboard public")
    created_by: str = Field(..., description="Creator user ID")
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())


class ReportConfig(BaseModel):
    """Model for report configuration"""
    report_id: str = Field(..., description="Report ID")
    name: str = Field(..., description="Report name")
    description: str = Field(..., description="Report description")
    query: AnalyticsQuery = Field(..., description="Report query")
    format: str = Field(default="pdf", description="Report format")
    schedule: Optional[str] = Field(default=None, description="Cron schedule")
    recipients: List[str] = Field(default_factory=list, description="Email recipients")
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())


class AnalyticsResult(BaseModel):
    """Model for analytics results"""
    query_id: str = Field(..., description="Query ID")
    data: List[Dict[str, Any]] = Field(..., description="Query results")
    metadata: Dict[str, Any] = Field(..., description="Result metadata")
    execution_time: float = Field(..., description="Execution time in seconds")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class AdvancedAnalyticsDashboard:
    """Advanced analytics dashboard and reporting system"""
    
    def __init__(self):
        self.redis_client: Optional[redis.Redis] = None
        self.db_engine = None
        self.dashboards: Dict[str, Dashboard] = {}
        self.reports: Dict[str, ReportConfig] = {}
        self.widget_cache: Dict[str, Any] = {}
        self.initialized = False
    
    async def initialize(self):
        """Initialize the analytics dashboard system"""
        try:
            # Initialize Redis
            self.redis_client = redis.from_url(settings.redis_url)
            await self.redis_client.ping()
            
            # Initialize database connection
            if settings.database_url:
                self.db_engine = create_async_engine(settings.database_url)
            
            # Create tables if they don't exist
            await self._create_tables()
            
            # Load existing dashboards and reports
            await self._load_dashboards()
            await self._load_reports()
            
            self.initialized = True
            logger.info("Advanced analytics dashboard initialized")
            
        except Exception as e:
            logger.error(f"Error initializing analytics dashboard: {e}")
            raise
    
    async def _create_tables(self):
        """Create database tables for analytics"""
        if not self.db_engine:
            return
        
        try:
            async with self.db_engine.begin() as conn:
                # Analytics events table
                await conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS analytics_events (
                        id SERIAL PRIMARY KEY,
                        event_type VARCHAR(100) NOT NULL,
                        event_data JSONB,
                        user_id VARCHAR(100),
                        session_id VARCHAR(100),
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """))
                
                # Analysis results table
                await conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS analysis_results (
                        id SERIAL PRIMARY KEY,
                        analysis_type VARCHAR(100) NOT NULL,
                        content_hash VARCHAR(64),
                        result_data JSONB,
                        processing_time FLOAT,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """))
                
                # User interactions table
                await conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS user_interactions (
                        id SERIAL PRIMARY KEY,
                        user_id VARCHAR(100),
                        action VARCHAR(100),
                        endpoint VARCHAR(200),
                        request_data JSONB,
                        response_time FLOAT,
                        status_code INTEGER,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """))
                
                # System metrics table
                await conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS system_metrics (
                        id SERIAL PRIMARY KEY,
                        metric_name VARCHAR(100) NOT NULL,
                        metric_value FLOAT,
                        metric_unit VARCHAR(50),
                        tags JSONB,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """))
                
        except Exception as e:
            logger.error(f"Error creating analytics tables: {e}")
    
    async def _load_dashboards(self):
        """Load dashboards from database"""
        if not self.db_engine:
            return
        
        try:
            async with self.db_engine.begin() as conn:
                result = await conn.execute(text("SELECT * FROM dashboards"))
                rows = result.fetchall()
                
                for row in rows:
                    dashboard_data = {
                        "dashboard_id": row[0],
                        "name": row[1],
                        "description": row[2],
                        "widgets": json.loads(row[3]) if row[3] else [],
                        "is_public": row[4],
                        "created_by": row[5],
                        "created_at": row[6],
                        "updated_at": row[7]
                    }
                    
                    dashboard = Dashboard(**dashboard_data)
                    self.dashboards[dashboard.dashboard_id] = dashboard
                    
        except Exception as e:
            logger.error(f"Error loading dashboards: {e}")
    
    async def _load_reports(self):
        """Load reports from database"""
        if not self.db_engine:
            return
        
        try:
            async with self.db_engine.begin() as conn:
                result = await conn.execute(text("SELECT * FROM reports"))
                rows = result.fetchall()
                
                for row in rows:
                    report_data = {
                        "report_id": row[0],
                        "name": row[1],
                        "description": row[2],
                        "query": json.loads(row[3]) if row[3] else {},
                        "format": row[4],
                        "schedule": row[5],
                        "recipients": json.loads(row[6]) if row[6] else [],
                        "created_at": row[7]
                    }
                    
                    report = ReportConfig(**report_data)
                    self.reports[report.report_id] = report
                    
        except Exception as e:
            logger.error(f"Error loading reports: {e}")
    
    async def create_dashboard(self, name: str, description: str, created_by: str, is_public: bool = False) -> str:
        """Create a new analytics dashboard"""
        dashboard_id = str(uuid.uuid4())
        
        dashboard = Dashboard(
            dashboard_id=dashboard_id,
            name=name,
            description=description,
            widgets=[],
            is_public=is_public,
            created_by=created_by
        )
        
        self.dashboards[dashboard_id] = dashboard
        await self._save_dashboard(dashboard)
        
        logger.info(f"Created dashboard: {dashboard_id}")
        return dashboard_id
    
    async def add_widget_to_dashboard(self, dashboard_id: str, widget: DashboardWidget) -> bool:
        """Add widget to dashboard"""
        if dashboard_id not in self.dashboards:
            return False
        
        dashboard = self.dashboards[dashboard_id]
        dashboard.widgets.append(widget)
        dashboard.updated_at = datetime.now().isoformat()
        
        await self._save_dashboard(dashboard)
        
        logger.info(f"Added widget {widget.widget_id} to dashboard {dashboard_id}")
        return True
    
    async def execute_analytics_query(self, query: AnalyticsQuery) -> AnalyticsResult:
        """Execute analytics query"""
        query_id = str(uuid.uuid4())
        start_time = datetime.now()
        
        try:
            # Execute query based on type
            if query.query_type == "usage_stats":
                data = await self._get_usage_stats(query)
            elif query.query_type == "performance_metrics":
                data = await self._get_performance_metrics(query)
            elif query.query_type == "user_activity":
                data = await self._get_user_activity(query)
            elif query.query_type == "content_analysis":
                data = await self._get_content_analysis_stats(query)
            elif query.query_type == "error_analysis":
                data = await self._get_error_analysis(query)
            else:
                data = []
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            result = AnalyticsResult(
                query_id=query_id,
                data=data,
                metadata={
                    "query_type": query.query_type,
                    "filters": query.filters,
                    "result_count": len(data)
                },
                execution_time=execution_time
            )
            
            # Cache result
            if self.redis_client:
                await self.redis_client.setex(
                    f"analytics_result:{query_id}",
                    3600,  # 1 hour TTL
                    json.dumps(result.model_dump())
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing analytics query: {e}")
            raise
    
    async def _get_usage_stats(self, query: AnalyticsQuery) -> List[Dict[str, Any]]:
        """Get usage statistics"""
        if not self.db_engine:
            return []
        
        try:
            async with self.db_engine.begin() as conn:
                sql = """
                    SELECT 
                        DATE(timestamp) as date,
                        COUNT(*) as total_requests,
                        COUNT(DISTINCT user_id) as unique_users,
                        AVG(response_time) as avg_response_time
                    FROM user_interactions
                    WHERE timestamp >= :start_date AND timestamp <= :end_date
                    GROUP BY DATE(timestamp)
                    ORDER BY date DESC
                    LIMIT :limit
                """
                
                start_date = query.date_range[0] if query.date_range else (datetime.now() - timedelta(days=30)).isoformat()
                end_date = query.date_range[1] if query.date_range else datetime.now().isoformat()
                
                result = await conn.execute(text(sql), {
                    "start_date": start_date,
                    "end_date": end_date,
                    "limit": query.limit
                })
                
                rows = result.fetchall()
                return [
                    {
                        "date": row[0].isoformat(),
                        "total_requests": row[1],
                        "unique_users": row[2],
                        "avg_response_time": float(row[3]) if row[3] else 0
                    }
                    for row in rows
                ]
                
        except Exception as e:
            logger.error(f"Error getting usage stats: {e}")
            return []
    
    async def _get_performance_metrics(self, query: AnalyticsQuery) -> List[Dict[str, Any]]:
        """Get performance metrics"""
        if not self.db_engine:
            return []
        
        try:
            async with self.db_engine.begin() as conn:
                sql = """
                    SELECT 
                        endpoint,
                        COUNT(*) as request_count,
                        AVG(response_time) as avg_response_time,
                        MAX(response_time) as max_response_time,
                        MIN(response_time) as min_response_time,
                        COUNT(CASE WHEN status_code >= 400 THEN 1 END) as error_count
                    FROM user_interactions
                    WHERE timestamp >= :start_date AND timestamp <= :end_date
                    GROUP BY endpoint
                    ORDER BY request_count DESC
                    LIMIT :limit
                """
                
                start_date = query.date_range[0] if query.date_range else (datetime.now() - timedelta(days=7)).isoformat()
                end_date = query.date_range[1] if query.date_range else datetime.now().isoformat()
                
                result = await conn.execute(text(sql), {
                    "start_date": start_date,
                    "end_date": end_date,
                    "limit": query.limit
                })
                
                rows = result.fetchall()
                return [
                    {
                        "endpoint": row[0],
                        "request_count": row[1],
                        "avg_response_time": float(row[2]) if row[2] else 0,
                        "max_response_time": float(row[3]) if row[3] else 0,
                        "min_response_time": float(row[4]) if row[4] else 0,
                        "error_count": row[5],
                        "error_rate": float(row[5]) / row[1] if row[1] > 0 else 0
                    }
                    for row in rows
                ]
                
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return []
    
    async def _get_user_activity(self, query: AnalyticsQuery) -> List[Dict[str, Any]]:
        """Get user activity data"""
        if not self.db_engine:
            return []
        
        try:
            async with self.db_engine.begin() as conn:
                sql = """
                    SELECT 
                        user_id,
                        COUNT(*) as activity_count,
                        COUNT(DISTINCT DATE(timestamp)) as active_days,
                        AVG(response_time) as avg_response_time,
                        MAX(timestamp) as last_activity
                    FROM user_interactions
                    WHERE timestamp >= :start_date AND timestamp <= :end_date
                    GROUP BY user_id
                    ORDER BY activity_count DESC
                    LIMIT :limit
                """
                
                start_date = query.date_range[0] if query.date_range else (datetime.now() - timedelta(days=30)).isoformat()
                end_date = query.date_range[1] if query.date_range else datetime.now().isoformat()
                
                result = await conn.execute(text(sql), {
                    "start_date": start_date,
                    "end_date": end_date,
                    "limit": query.limit
                })
                
                rows = result.fetchall()
                return [
                    {
                        "user_id": row[0],
                        "activity_count": row[1],
                        "active_days": row[2],
                        "avg_response_time": float(row[3]) if row[3] else 0,
                        "last_activity": row[4].isoformat() if row[4] else None
                    }
                    for row in rows
                ]
                
        except Exception as e:
            logger.error(f"Error getting user activity: {e}")
            return []
    
    async def _get_content_analysis_stats(self, query: AnalyticsQuery) -> List[Dict[str, Any]]:
        """Get content analysis statistics"""
        if not self.db_engine:
            return []
        
        try:
            async with self.db_engine.begin() as conn:
                sql = """
                    SELECT 
                        analysis_type,
                        COUNT(*) as analysis_count,
                        AVG(processing_time) as avg_processing_time,
                        COUNT(DISTINCT content_hash) as unique_content_count
                    FROM analysis_results
                    WHERE timestamp >= :start_date AND timestamp <= :end_date
                    GROUP BY analysis_type
                    ORDER BY analysis_count DESC
                """
                
                start_date = query.date_range[0] if query.date_range else (datetime.now() - timedelta(days=30)).isoformat()
                end_date = query.date_range[1] if query.date_range else datetime.now().isoformat()
                
                result = await conn.execute(text(sql), {
                    "start_date": start_date,
                    "end_date": end_date
                })
                
                rows = result.fetchall()
                return [
                    {
                        "analysis_type": row[0],
                        "analysis_count": row[1],
                        "avg_processing_time": float(row[2]) if row[2] else 0,
                        "unique_content_count": row[3]
                    }
                    for row in rows
                ]
                
        except Exception as e:
            logger.error(f"Error getting content analysis stats: {e}")
            return []
    
    async def _get_error_analysis(self, query: AnalyticsQuery) -> List[Dict[str, Any]]:
        """Get error analysis data"""
        if not self.db_engine:
            return []
        
        try:
            async with self.db_engine.begin() as conn:
                sql = """
                    SELECT 
                        status_code,
                        endpoint,
                        COUNT(*) as error_count,
                        AVG(response_time) as avg_response_time
                    FROM user_interactions
                    WHERE status_code >= 400 
                    AND timestamp >= :start_date AND timestamp <= :end_date
                    GROUP BY status_code, endpoint
                    ORDER BY error_count DESC
                    LIMIT :limit
                """
                
                start_date = query.date_range[0] if query.date_range else (datetime.now() - timedelta(days=7)).isoformat()
                end_date = query.date_range[1] if query.date_range else datetime.now().isoformat()
                
                result = await conn.execute(text(sql), {
                    "start_date": start_date,
                    "end_date": end_date,
                    "limit": query.limit
                })
                
                rows = result.fetchall()
                return [
                    {
                        "status_code": row[0],
                        "endpoint": row[1],
                        "error_count": row[2],
                        "avg_response_time": float(row[3]) if row[3] else 0
                    }
                    for row in rows
                ]
                
        except Exception as e:
            logger.error(f"Error getting error analysis: {e}")
            return []
    
    async def generate_chart(self, data: List[Dict[str, Any]], chart_type: str, title: str) -> str:
        """Generate chart from data"""
        try:
            if not data:
                return ""
            
            df = pd.DataFrame(data)
            
            if chart_type == "line":
                fig = px.line(df, x=df.columns[0], y=df.columns[1], title=title)
            elif chart_type == "bar":
                fig = px.bar(df, x=df.columns[0], y=df.columns[1], title=title)
            elif chart_type == "pie":
                fig = px.pie(df, names=df.columns[0], values=df.columns[1], title=title)
            elif chart_type == "scatter":
                fig = px.scatter(df, x=df.columns[0], y=df.columns[1], title=title)
            else:
                fig = px.bar(df, x=df.columns[0], y=df.columns[1], title=title)
            
            # Convert to HTML
            chart_html = fig.to_html(include_plotlyjs=False)
            return chart_html
            
        except Exception as e:
            logger.error(f"Error generating chart: {e}")
            return ""
    
    async def generate_dashboard_html(self, dashboard_id: str) -> str:
        """Generate HTML for dashboard"""
        if dashboard_id not in self.dashboards:
            return ""
        
        dashboard = self.dashboards[dashboard_id]
        
        html_parts = [
            f"<h1>{dashboard.name}</h1>",
            f"<p>{dashboard.description}</p>",
            "<div class='dashboard-grid'>"
        ]
        
        for widget in dashboard.widgets:
            # Execute widget query
            result = await self.execute_analytics_query(widget.query)
            
            # Generate chart
            chart_html = await self.generate_chart(
                result.data,
                widget.widget_type,
                widget.title
            )
            
            html_parts.append(f"""
                <div class='widget' style='grid-column: {widget.position.get("col", 1)}; grid-row: {widget.position.get("row", 1)}'>
                    <h3>{widget.title}</h3>
                    {chart_html}
                </div>
            """)
        
        html_parts.append("</div>")
        
        return "".join(html_parts)
    
    async def create_report(self, name: str, description: str, query: AnalyticsQuery, 
                          format: str = "pdf", schedule: str = None, recipients: List[str] = None) -> str:
        """Create a new report configuration"""
        report_id = str(uuid.uuid4())
        
        report = ReportConfig(
            report_id=report_id,
            name=name,
            description=description,
            query=query,
            format=format,
            schedule=schedule,
            recipients=recipients or []
        )
        
        self.reports[report_id] = report
        await self._save_report(report)
        
        logger.info(f"Created report: {report_id}")
        return report_id
    
    async def generate_report(self, report_id: str) -> bytes:
        """Generate report in specified format"""
        if report_id not in self.reports:
            raise ValueError(f"Report {report_id} not found")
        
        report = self.reports[report_id]
        
        # Execute query
        result = await self.execute_analytics_query(report.query)
        
        if report.format == "pdf":
            return await self._generate_pdf_report(report, result)
        elif report.format == "excel":
            return await self._generate_excel_report(report, result)
        elif report.format == "csv":
            return await self._generate_csv_report(report, result)
        else:
            raise ValueError(f"Unsupported report format: {report.format}")
    
    async def _generate_pdf_report(self, report: ReportConfig, result: AnalyticsResult) -> bytes:
        """Generate PDF report"""
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
            from reportlab.lib.styles import getSampleStyleSheet
            from reportlab.lib import colors
            
            buffer = BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=letter)
            styles = getSampleStyleSheet()
            story = []
            
            # Title
            title = Paragraph(report.name, styles['Title'])
            story.append(title)
            story.append(Spacer(1, 12))
            
            # Description
            desc = Paragraph(report.description, styles['Normal'])
            story.append(desc)
            story.append(Spacer(1, 12))
            
            # Data table
            if result.data:
                df = pd.DataFrame(result.data)
                
                # Create table
                table_data = [df.columns.tolist()]
                for _, row in df.iterrows():
                    table_data.append(row.tolist())
                
                table = Table(table_data)
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 14),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                
                story.append(table)
            
            # Metadata
            story.append(Spacer(1, 12))
            metadata = Paragraph(f"Generated on: {result.timestamp}<br/>Execution time: {result.execution_time:.2f}s", styles['Normal'])
            story.append(metadata)
            
            doc.build(story)
            buffer.seek(0)
            return buffer.getvalue()
            
        except Exception as e:
            logger.error(f"Error generating PDF report: {e}")
            raise
    
    async def _generate_excel_report(self, report: ReportConfig, result: AnalyticsResult) -> bytes:
        """Generate Excel report"""
        try:
            buffer = BytesIO()
            
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                if result.data:
                    df = pd.DataFrame(result.data)
                    df.to_excel(writer, sheet_name='Data', index=False)
                
                # Metadata sheet
                metadata_df = pd.DataFrame([
                    {"Metric": "Report Name", "Value": report.name},
                    {"Metric": "Generated On", "Value": result.timestamp},
                    {"Metric": "Execution Time", "Value": f"{result.execution_time:.2f}s"},
                    {"Metric": "Result Count", "Value": len(result.data)}
                ])
                metadata_df.to_excel(writer, sheet_name='Metadata', index=False)
            
            buffer.seek(0)
            return buffer.getvalue()
            
        except Exception as e:
            logger.error(f"Error generating Excel report: {e}")
            raise
    
    async def _generate_csv_report(self, report: ReportConfig, result: AnalyticsResult) -> bytes:
        """Generate CSV report"""
        try:
            if result.data:
                df = pd.DataFrame(result.data)
                csv_content = df.to_csv(index=False)
                return csv_content.encode('utf-8')
            else:
                return b""
                
        except Exception as e:
            logger.error(f"Error generating CSV report: {e}")
            raise
    
    async def _save_dashboard(self, dashboard: Dashboard):
        """Save dashboard to database"""
        if not self.db_engine:
            return
        
        try:
            async with self.db_engine.begin() as conn:
                await conn.execute(text("""
                    INSERT INTO dashboards (dashboard_id, name, description, widgets, is_public, created_by, created_at, updated_at)
                    VALUES (:dashboard_id, :name, :description, :widgets, :is_public, :created_by, :created_at, :updated_at)
                    ON CONFLICT (dashboard_id) DO UPDATE SET
                    name = :name,
                    description = :description,
                    widgets = :widgets,
                    is_public = :is_public,
                    updated_at = :updated_at
                """), {
                    "dashboard_id": dashboard.dashboard_id,
                    "name": dashboard.name,
                    "description": dashboard.description,
                    "widgets": json.dumps([widget.model_dump() for widget in dashboard.widgets]),
                    "is_public": dashboard.is_public,
                    "created_by": dashboard.created_by,
                    "created_at": dashboard.created_at,
                    "updated_at": dashboard.updated_at
                })
                
        except Exception as e:
            logger.error(f"Error saving dashboard: {e}")
    
    async def _save_report(self, report: ReportConfig):
        """Save report to database"""
        if not self.db_engine:
            return
        
        try:
            async with self.db_engine.begin() as conn:
                await conn.execute(text("""
                    INSERT INTO reports (report_id, name, description, query, format, schedule, recipients, created_at)
                    VALUES (:report_id, :name, :description, :query, :format, :schedule, :recipients, :created_at)
                    ON CONFLICT (report_id) DO UPDATE SET
                    name = :name,
                    description = :description,
                    query = :query,
                    format = :format,
                    schedule = :schedule,
                    recipients = :recipients
                """), {
                    "report_id": report.report_id,
                    "name": report.name,
                    "description": report.description,
                    "query": json.dumps(report.query.model_dump()),
                    "format": report.format,
                    "schedule": report.schedule,
                    "recipients": json.dumps(report.recipients),
                    "created_at": report.created_at
                })
                
        except Exception as e:
            logger.error(f"Error saving report: {e}")
    
    async def get_dashboard(self, dashboard_id: str) -> Optional[Dashboard]:
        """Get dashboard by ID"""
        return self.dashboards.get(dashboard_id)
    
    async def list_dashboards(self) -> List[Dashboard]:
        """List all dashboards"""
        return list(self.dashboards.values())
    
    async def get_report(self, report_id: str) -> Optional[ReportConfig]:
        """Get report by ID"""
        return self.reports.get(report_id)
    
    async def list_reports(self) -> List[ReportConfig]:
        """List all reports"""
        return list(self.reports.values())


# Global analytics dashboard
analytics_dashboard = AdvancedAnalyticsDashboard()


async def initialize_analytics_dashboard():
    """Initialize the analytics dashboard system"""
    await analytics_dashboard.initialize()
















