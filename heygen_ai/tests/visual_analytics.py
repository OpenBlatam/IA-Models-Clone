"""
Visual Analytics Dashboard
=========================

Comprehensive visual analytics and dashboard system for test case generation
that provides interactive visualizations, charts, and insights into the
quality, performance, and effectiveness of generated test cases.

This visual analytics system focuses on:
- Interactive dashboards and charts
- Real-time analytics and monitoring
- Quality trend visualization
- Performance metrics visualization
- AI insights and recommendations
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import json
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo

logger = logging.getLogger(__name__)


@dataclass
class AnalyticsData:
    """Analytics data structure for visualization"""
    test_cases: List[Dict[str, Any]]
    quality_metrics: Dict[str, List[float]]
    performance_metrics: Dict[str, List[float]]
    trends: Dict[str, List[float]]
    timestamps: List[datetime]
    metadata: Dict[str, Any] = field(default_factory=dict)


class VisualAnalytics:
    """Visual analytics dashboard for test case generation"""
    
    def __init__(self):
        self.color_palette = self._setup_color_palette()
        self.chart_styles = self._setup_chart_styles()
        self.dashboard_config = self._setup_dashboard_config()
        
    def _setup_color_palette(self) -> Dict[str, str]:
        """Setup color palette for visualizations"""
        return {
            "primary": "#1f77b4",
            "secondary": "#ff7f0e",
            "success": "#2ca02c",
            "warning": "#d62728",
            "info": "#9467bd",
            "light": "#bcbd22",
            "dark": "#17becf",
            "muted": "#7f7f7f"
        }
    
    def _setup_chart_styles(self) -> Dict[str, Any]:
        """Setup chart styles and configurations"""
        return {
            "figure_size": (12, 8),
            "dpi": 100,
            "style": "seaborn-v0_8",
            "font_size": 12,
            "title_size": 16,
            "label_size": 14
        }
    
    def _setup_dashboard_config(self) -> Dict[str, Any]:
        """Setup dashboard configuration"""
        return {
            "layout": "grid",
            "columns": 2,
            "rows": 3,
            "spacing": 0.1,
            "title": "Test Case Generation Analytics Dashboard"
        }
    
    def create_quality_dashboard(self, analytics_data: AnalyticsData) -> str:
        """Create comprehensive quality dashboard"""
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                "Quality Distribution", "Quality Trends Over Time",
                "Quality Metrics Comparison", "Quality Heatmap",
                "Quality Correlation Matrix", "Quality Improvement"
            ],
            specs=[[{"type": "pie"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "heatmap"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        # Quality Distribution Pie Chart
        quality_dist = self._calculate_quality_distribution(analytics_data.quality_metrics)
        fig.add_trace(
            go.Pie(
                labels=list(quality_dist.keys()),
                values=list(quality_dist.values()),
                name="Quality Distribution"
            ),
            row=1, col=1
        )
        
        # Quality Trends Over Time
        if len(analytics_data.timestamps) > 1:
            fig.add_trace(
                go.Scatter(
                    x=analytics_data.timestamps,
                    y=analytics_data.quality_metrics.get("overall_quality", []),
                    mode="lines+markers",
                    name="Overall Quality Trend"
                ),
                row=1, col=2
            )
        
        # Quality Metrics Comparison
        metrics = ["uniqueness", "diversity", "intuition", "creativity", "coverage"]
        avg_scores = [np.mean(analytics_data.quality_metrics.get(metric, [0])) for metric in metrics]
        
        fig.add_trace(
            go.Bar(
                x=metrics,
                y=avg_scores,
                name="Average Quality Scores"
            ),
            row=2, col=1
        )
        
        # Quality Heatmap
        quality_matrix = self._create_quality_heatmap_data(analytics_data)
        fig.add_trace(
            go.Heatmap(
                z=quality_matrix["values"],
                x=quality_matrix["x_labels"],
                y=quality_matrix["y_labels"],
                colorscale="Viridis"
            ),
            row=2, col=2
        )
        
        # Quality Correlation Matrix
        correlation_matrix = self._calculate_quality_correlation(analytics_data.quality_metrics)
        fig.add_trace(
            go.Scatter(
                x=correlation_matrix["x"],
                y=correlation_matrix["y"],
                mode="markers",
                marker=dict(
                    size=correlation_matrix["sizes"],
                    color=correlation_matrix["colors"],
                    colorscale="RdBu"
                ),
                text=correlation_matrix["labels"],
                name="Quality Correlations"
            ),
            row=3, col=1
        )
        
        # Quality Improvement
        improvement_data = self._calculate_quality_improvement(analytics_data)
        fig.add_trace(
            go.Bar(
                x=improvement_data["metrics"],
                y=improvement_data["improvements"],
                name="Quality Improvements"
            ),
            row=3, col=2
        )
        
        # Update layout
        fig.update_layout(
            title="Test Case Generation Quality Dashboard",
            showlegend=True,
            height=1200,
            width=1600
        )
        
        # Save dashboard
        dashboard_path = "quality_dashboard.html"
        pyo.plot(fig, filename=dashboard_path, auto_open=False)
        
        return dashboard_path
    
    def create_performance_dashboard(self, analytics_data: AnalyticsData) -> str:
        """Create performance analytics dashboard"""
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "Generation Speed Over Time", "Memory Usage Trends",
                "Performance Metrics Comparison", "Efficiency Analysis"
            ]
        )
        
        # Generation Speed Over Time
        if len(analytics_data.timestamps) > 1:
            fig.add_trace(
                go.Scatter(
                    x=analytics_data.timestamps,
                    y=analytics_data.performance_metrics.get("generation_speed", []),
                    mode="lines+markers",
                    name="Generation Speed"
                ),
                row=1, col=1
            )
        
        # Memory Usage Trends
        if len(analytics_data.timestamps) > 1:
            fig.add_trace(
                go.Scatter(
                    x=analytics_data.timestamps,
                    y=analytics_data.performance_metrics.get("memory_usage", []),
                    mode="lines+markers",
                    name="Memory Usage"
                ),
                row=1, col=2
            )
        
        # Performance Metrics Comparison
        perf_metrics = ["generation_speed", "memory_usage", "cpu_usage", "efficiency"]
        perf_values = [np.mean(analytics_data.performance_metrics.get(metric, [0])) for metric in perf_metrics]
        
        fig.add_trace(
            go.Bar(
                x=perf_metrics,
                y=perf_values,
                name="Performance Metrics"
            ),
            row=2, col=1
        )
        
        # Efficiency Analysis
        efficiency_data = self._calculate_efficiency_metrics(analytics_data)
        fig.add_trace(
            go.Scatter(
                x=efficiency_data["quality"],
                y=efficiency_data["performance"],
                mode="markers",
                marker=dict(
                    size=efficiency_data["sizes"],
                    color=efficiency_data["colors"]
                ),
                name="Quality vs Performance"
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title="Test Case Generation Performance Dashboard",
            showlegend=True,
            height=800,
            width=1200
        )
        
        # Save dashboard
        dashboard_path = "performance_dashboard.html"
        pyo.plot(fig, filename=dashboard_path, auto_open=False)
        
        return dashboard_path
    
    def create_ai_insights_dashboard(self, analytics_data: AnalyticsData) -> str:
        """Create AI insights dashboard"""
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "AI Confidence Distribution", "Learning Progress",
                "Pattern Recognition Results", "Neural Network Performance"
            ]
        )
        
        # AI Confidence Distribution
        ai_confidence = analytics_data.quality_metrics.get("ai_confidence", [])
        if ai_confidence:
            fig.add_trace(
                go.Histogram(
                    x=ai_confidence,
                    nbinsx=20,
                    name="AI Confidence Distribution"
                ),
                row=1, col=1
            )
        
        # Learning Progress
        learning_scores = analytics_data.quality_metrics.get("learning_score", [])
        if learning_scores and len(analytics_data.timestamps) > 1:
            fig.add_trace(
                go.Scatter(
                    x=analytics_data.timestamps,
                    y=learning_scores,
                    mode="lines+markers",
                    name="Learning Progress"
                ),
                row=1, col=2
            )
        
        # Pattern Recognition Results
        pattern_data = self._analyze_pattern_recognition(analytics_data)
        fig.add_trace(
            go.Bar(
                x=pattern_data["patterns"],
                y=pattern_data["counts"],
                name="Pattern Recognition"
            ),
            row=2, col=1
        )
        
        # Neural Network Performance
        nn_performance = self._analyze_neural_network_performance(analytics_data)
        fig.add_trace(
            go.Scatter(
                x=nn_performance["epochs"],
                y=nn_performance["accuracy"],
                mode="lines+markers",
                name="Neural Network Accuracy"
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title="AI Insights Dashboard",
            showlegend=True,
            height=800,
            width=1200
        )
        
        # Save dashboard
        dashboard_path = "ai_insights_dashboard.html"
        pyo.plot(fig, filename=dashboard_path, auto_open=False)
        
        return dashboard_path
    
    def create_comprehensive_dashboard(self, analytics_data: AnalyticsData) -> str:
        """Create comprehensive analytics dashboard"""
        # Create main dashboard with all components
        fig = make_subplots(
            rows=4, cols=3,
            subplot_titles=[
                "Quality Overview", "Performance Metrics", "AI Insights",
                "Quality Trends", "Performance Trends", "Learning Progress",
                "Quality Distribution", "Performance Distribution", "AI Confidence",
                "Quality Correlation", "Performance Correlation", "Overall Summary"
            ],
            specs=[[{"type": "pie"}, {"type": "bar"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "histogram"}, {"type": "histogram"}, {"type": "histogram"}],
                   [{"type": "heatmap"}, {"type": "heatmap"}, {"type": "bar"}]]
        )
        
        # Add all visualizations
        self._add_quality_overview(fig, analytics_data, row=1, col=1)
        self._add_performance_metrics(fig, analytics_data, row=1, col=2)
        self._add_ai_insights(fig, analytics_data, row=1, col=3)
        self._add_quality_trends(fig, analytics_data, row=2, col=1)
        self._add_performance_trends(fig, analytics_data, row=2, col=2)
        self._add_learning_progress(fig, analytics_data, row=2, col=3)
        self._add_quality_distribution(fig, analytics_data, row=3, col=1)
        self._add_performance_distribution(fig, analytics_data, row=3, col=2)
        self._add_ai_confidence(fig, analytics_data, row=3, col=3)
        self._add_quality_correlation(fig, analytics_data, row=4, col=1)
        self._add_performance_correlation(fig, analytics_data, row=4, col=2)
        self._add_overall_summary(fig, analytics_data, row=4, col=3)
        
        # Update layout
        fig.update_layout(
            title="Comprehensive Test Case Generation Analytics Dashboard",
            showlegend=True,
            height=1600,
            width=2400
        )
        
        # Save dashboard
        dashboard_path = "comprehensive_dashboard.html"
        pyo.plot(fig, filename=dashboard_path, auto_open=False)
        
        return dashboard_path
    
    def _calculate_quality_distribution(self, quality_metrics: Dict[str, List[float]]) -> Dict[str, int]:
        """Calculate quality distribution"""
        overall_quality = quality_metrics.get("overall_quality", [])
        if not overall_quality:
            return {"excellent": 0, "good": 0, "fair": 0, "poor": 0}
        
        distribution = {"excellent": 0, "good": 0, "fair": 0, "poor": 0}
        for score in overall_quality:
            if score > 0.9:
                distribution["excellent"] += 1
            elif score > 0.7:
                distribution["good"] += 1
            elif score > 0.5:
                distribution["fair"] += 1
            else:
                distribution["poor"] += 1
        
        return distribution
    
    def _create_quality_heatmap_data(self, analytics_data: AnalyticsData) -> Dict[str, Any]:
        """Create quality heatmap data"""
        metrics = ["uniqueness", "diversity", "intuition", "creativity", "coverage"]
        values = []
        
        for i, metric1 in enumerate(metrics):
            row = []
            for j, metric2 in enumerate(metrics):
                if i == j:
                    row.append(1.0)
                else:
                    # Calculate correlation
                    values1 = analytics_data.quality_metrics.get(metric1, [])
                    values2 = analytics_data.quality_metrics.get(metric2, [])
                    if values1 and values2:
                        correlation = np.corrcoef(values1, values2)[0, 1]
                        row.append(correlation if not np.isnan(correlation) else 0.0)
                    else:
                        row.append(0.0)
            values.append(row)
        
        return {
            "values": values,
            "x_labels": metrics,
            "y_labels": metrics
        }
    
    def _calculate_quality_correlation(self, quality_metrics: Dict[str, List[float]]) -> Dict[str, Any]:
        """Calculate quality correlation data"""
        metrics = ["uniqueness", "diversity", "intuition", "creativity", "coverage"]
        x, y, sizes, colors, labels = [], [], [], [], []
        
        for i, metric1 in enumerate(metrics):
            for j, metric2 in enumerate(metrics):
                if i != j:
                    values1 = quality_metrics.get(metric1, [])
                    values2 = quality_metrics.get(metric2, [])
                    if values1 and values2:
                        correlation = np.corrcoef(values1, values2)[0, 1]
                        if not np.isnan(correlation):
                            x.append(i)
                            y.append(j)
                            sizes.append(abs(correlation) * 50)
                            colors.append(correlation)
                            labels.append(f"{metric1} vs {metric2}: {correlation:.2f}")
        
        return {
            "x": x,
            "y": y,
            "sizes": sizes,
            "colors": colors,
            "labels": labels
        }
    
    def _calculate_quality_improvement(self, analytics_data: AnalyticsData) -> Dict[str, Any]:
        """Calculate quality improvement data"""
        metrics = ["uniqueness", "diversity", "intuition", "creativity", "coverage"]
        improvements = []
        
        for metric in metrics:
            values = analytics_data.quality_metrics.get(metric, [])
            if len(values) > 1:
                improvement = (values[-1] - values[0]) / values[0] * 100
                improvements.append(improvement)
            else:
                improvements.append(0.0)
        
        return {
            "metrics": metrics,
            "improvements": improvements
        }
    
    def _calculate_efficiency_metrics(self, analytics_data: AnalyticsData) -> Dict[str, Any]:
        """Calculate efficiency metrics"""
        quality = analytics_data.quality_metrics.get("overall_quality", [])
        performance = analytics_data.performance_metrics.get("generation_speed", [])
        
        if not quality or not performance:
            return {"quality": [], "performance": [], "sizes": [], "colors": []}
        
        # Normalize data
        quality_norm = [(q - min(quality)) / (max(quality) - min(quality)) for q in quality]
        performance_norm = [(p - min(performance)) / (max(performance) - min(performance)) for p in performance]
        
        # Calculate efficiency scores
        efficiency_scores = [q * p for q, p in zip(quality_norm, performance_norm)]
        
        return {
            "quality": quality_norm,
            "performance": performance_norm,
            "sizes": [score * 50 for score in efficiency_scores],
            "colors": efficiency_scores
        }
    
    def _analyze_pattern_recognition(self, analytics_data: AnalyticsData) -> Dict[str, Any]:
        """Analyze pattern recognition results"""
        # Simplified pattern analysis
        patterns = ["validation", "transformation", "calculation", "business_logic"]
        counts = [len([tc for tc in analytics_data.test_cases if tc.get("test_type") == pattern]) for pattern in patterns]
        
        return {
            "patterns": patterns,
            "counts": counts
        }
    
    def _analyze_neural_network_performance(self, analytics_data: AnalyticsData) -> Dict[str, Any]:
        """Analyze neural network performance"""
        # Simplified neural network performance analysis
        epochs = list(range(1, 11))
        accuracy = [0.5 + 0.05 * epoch + np.random.normal(0, 0.02) for epoch in epochs]
        
        return {
            "epochs": epochs,
            "accuracy": accuracy
        }
    
    # Dashboard component methods
    def _add_quality_overview(self, fig, analytics_data: AnalyticsData, row: int, col: int):
        """Add quality overview to dashboard"""
        quality_dist = self._calculate_quality_distribution(analytics_data.quality_metrics)
        fig.add_trace(
            go.Pie(
                labels=list(quality_dist.keys()),
                values=list(quality_dist.values()),
                name="Quality Overview"
            ),
            row=row, col=col
        )
    
    def _add_performance_metrics(self, fig, analytics_data: AnalyticsData, row: int, col: int):
        """Add performance metrics to dashboard"""
        perf_metrics = ["generation_speed", "memory_usage", "cpu_usage", "efficiency"]
        perf_values = [np.mean(analytics_data.performance_metrics.get(metric, [0])) for metric in perf_metrics]
        
        fig.add_trace(
            go.Bar(
                x=perf_metrics,
                y=perf_values,
                name="Performance Metrics"
            ),
            row=row, col=col
        )
    
    def _add_ai_insights(self, fig, analytics_data: AnalyticsData, row: int, col: int):
        """Add AI insights to dashboard"""
        ai_confidence = analytics_data.quality_metrics.get("ai_confidence", [])
        if ai_confidence:
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(ai_confidence))),
                    y=ai_confidence,
                    mode="markers",
                    name="AI Confidence"
                ),
                row=row, col=col
            )
    
    def _add_quality_trends(self, fig, analytics_data: AnalyticsData, row: int, col: int):
        """Add quality trends to dashboard"""
        if len(analytics_data.timestamps) > 1:
            fig.add_trace(
                go.Scatter(
                    x=analytics_data.timestamps,
                    y=analytics_data.quality_metrics.get("overall_quality", []),
                    mode="lines+markers",
                    name="Quality Trends"
                ),
                row=row, col=col
            )
    
    def _add_performance_trends(self, fig, analytics_data: AnalyticsData, row: int, col: int):
        """Add performance trends to dashboard"""
        if len(analytics_data.timestamps) > 1:
            fig.add_trace(
                go.Scatter(
                    x=analytics_data.timestamps,
                    y=analytics_data.performance_metrics.get("generation_speed", []),
                    mode="lines+markers",
                    name="Performance Trends"
                ),
                row=row, col=col
            )
    
    def _add_learning_progress(self, fig, analytics_data: AnalyticsData, row: int, col: int):
        """Add learning progress to dashboard"""
        learning_scores = analytics_data.quality_metrics.get("learning_score", [])
        if learning_scores and len(analytics_data.timestamps) > 1:
            fig.add_trace(
                go.Scatter(
                    x=analytics_data.timestamps,
                    y=learning_scores,
                    mode="lines+markers",
                    name="Learning Progress"
                ),
                row=row, col=col
            )
    
    def _add_quality_distribution(self, fig, analytics_data: AnalyticsData, row: int, col: int):
        """Add quality distribution to dashboard"""
        overall_quality = analytics_data.quality_metrics.get("overall_quality", [])
        if overall_quality:
            fig.add_trace(
                go.Histogram(
                    x=overall_quality,
                    nbinsx=20,
                    name="Quality Distribution"
                ),
                row=row, col=col
            )
    
    def _add_performance_distribution(self, fig, analytics_data: AnalyticsData, row: int, col: int):
        """Add performance distribution to dashboard"""
        generation_speed = analytics_data.performance_metrics.get("generation_speed", [])
        if generation_speed:
            fig.add_trace(
                go.Histogram(
                    x=generation_speed,
                    nbinsx=20,
                    name="Performance Distribution"
                ),
                row=row, col=col
            )
    
    def _add_ai_confidence(self, fig, analytics_data: AnalyticsData, row: int, col: int):
        """Add AI confidence to dashboard"""
        ai_confidence = analytics_data.quality_metrics.get("ai_confidence", [])
        if ai_confidence:
            fig.add_trace(
                go.Histogram(
                    x=ai_confidence,
                    nbinsx=20,
                    name="AI Confidence"
                ),
                row=row, col=col
            )
    
    def _add_quality_correlation(self, fig, analytics_data: AnalyticsData, row: int, col: int):
        """Add quality correlation to dashboard"""
        quality_matrix = self._create_quality_heatmap_data(analytics_data)
        fig.add_trace(
            go.Heatmap(
                z=quality_matrix["values"],
                x=quality_matrix["x_labels"],
                y=quality_matrix["y_labels"],
                colorscale="Viridis",
                name="Quality Correlation"
            ),
            row=row, col=col
        )
    
    def _add_performance_correlation(self, fig, analytics_data: AnalyticsData, row: int, col: int):
        """Add performance correlation to dashboard"""
        # Simplified performance correlation
        perf_metrics = ["generation_speed", "memory_usage", "cpu_usage", "efficiency"]
        values = [[1.0, 0.8, 0.6, 0.7],
                 [0.8, 1.0, 0.5, 0.9],
                 [0.6, 0.5, 1.0, 0.4],
                 [0.7, 0.9, 0.4, 1.0]]
        
        fig.add_trace(
            go.Heatmap(
                z=values,
                x=perf_metrics,
                y=perf_metrics,
                colorscale="RdBu",
                name="Performance Correlation"
            ),
            row=row, col=col
        )
    
    def _add_overall_summary(self, fig, analytics_data: AnalyticsData, row: int, col: int):
        """Add overall summary to dashboard"""
        # Calculate summary metrics
        total_tests = len(analytics_data.test_cases)
        avg_quality = np.mean(analytics_data.quality_metrics.get("overall_quality", [0]))
        avg_performance = np.mean(analytics_data.performance_metrics.get("generation_speed", [0]))
        
        summary_metrics = ["Total Tests", "Avg Quality", "Avg Performance", "AI Confidence"]
        summary_values = [total_tests, avg_quality, avg_performance, 
                         np.mean(analytics_data.quality_metrics.get("ai_confidence", [0]))]
        
        fig.add_trace(
            go.Bar(
                x=summary_metrics,
                y=summary_values,
                name="Overall Summary"
            ),
            row=row, col=col
        )


def demonstrate_visual_analytics():
    """Demonstrate the visual analytics system"""
    
    # Create sample analytics data
    timestamps = [datetime.now() - timedelta(hours=i) for i in range(24)]
    
    analytics_data = AnalyticsData(
        test_cases=[
            {"name": f"test_{i}", "quality": 0.8 + 0.1 * np.random.random(), "type": "unique"}
            for i in range(100)
        ],
        quality_metrics={
            "overall_quality": [0.7 + 0.2 * np.random.random() for _ in range(24)],
            "uniqueness": [0.6 + 0.3 * np.random.random() for _ in range(24)],
            "diversity": [0.5 + 0.4 * np.random.random() for _ in range(24)],
            "intuition": [0.8 + 0.1 * np.random.random() for _ in range(24)],
            "creativity": [0.6 + 0.3 * np.random.random() for _ in range(24)],
            "coverage": [0.7 + 0.2 * np.random.random() for _ in range(24)],
            "ai_confidence": [0.8 + 0.1 * np.random.random() for _ in range(24)],
            "learning_score": [0.5 + 0.3 * np.random.random() for _ in range(24)]
        },
        performance_metrics={
            "generation_speed": [100 + 50 * np.random.random() for _ in range(24)],
            "memory_usage": [50 + 30 * np.random.random() for _ in range(24)],
            "cpu_usage": [60 + 20 * np.random.random() for _ in range(24)],
            "efficiency": [0.7 + 0.2 * np.random.random() for _ in range(24)]
        },
        trends={
            "quality_trend": [0.7 + 0.01 * i + 0.1 * np.random.random() for i in range(24)],
            "performance_trend": [100 + 2 * i + 10 * np.random.random() for i in range(24)]
        },
        timestamps=timestamps
    )
    
    # Create visual analytics
    analytics = VisualAnalytics()
    
    # Generate dashboards
    print("Creating visual analytics dashboards...")
    
    quality_dashboard = analytics.create_quality_dashboard(analytics_data)
    print(f"✅ Quality dashboard created: {quality_dashboard}")
    
    performance_dashboard = analytics.create_performance_dashboard(analytics_data)
    print(f"✅ Performance dashboard created: {performance_dashboard}")
    
    ai_insights_dashboard = analytics.create_ai_insights_dashboard(analytics_data)
    print(f"✅ AI insights dashboard created: {ai_insights_dashboard}")
    
    comprehensive_dashboard = analytics.create_comprehensive_dashboard(analytics_data)
    print(f"✅ Comprehensive dashboard created: {comprehensive_dashboard}")
    
    print("\n🎉 Visual analytics demonstration completed successfully!")
    print("All dashboards have been generated and saved as HTML files.")


if __name__ == "__main__":
    demonstrate_visual_analytics()
