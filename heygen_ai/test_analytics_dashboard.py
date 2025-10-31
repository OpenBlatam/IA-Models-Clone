#!/usr/bin/env python3
"""
Advanced Test Analytics Dashboard
================================

This module provides comprehensive analytics and visualization
for test metrics, performance trends, and quality insights.
"""

import sys
import json
import time
import sqlite3
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import statistics

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("⚠️  Visualization libraries not available. Install with: pip install matplotlib seaborn pandas plotly")

@dataclass
class TestMetrics:
    """Comprehensive test metrics data structure"""
    timestamp: str
    test_suite: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    skipped_tests: int
    execution_time: float
    coverage_percentage: float
    memory_usage: float
    cpu_usage: float
    quality_score: float
    security_score: float
    performance_score: float
    maintainability_score: float
    reliability_score: float
    test_density: float
    code_complexity: float
    technical_debt: float

@dataclass
class TrendAnalysis:
    """Trend analysis results"""
    metric_name: str
    current_value: float
    previous_value: float
    trend_direction: str  # "improving", "declining", "stable"
    trend_percentage: float
    trend_significance: str  # "high", "medium", "low"
    recommendations: List[str]

class TestAnalyticsDashboard:
    """Advanced analytics dashboard for test metrics"""
    
    def __init__(self, db_path: str = "test_analytics.db"):
        self.db_path = Path(db_path)
        self.visualization_available = VISUALIZATION_AVAILABLE
        self.init_database()
        
    def init_database(self):
        """Initialize SQLite database for analytics storage"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create metrics table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS test_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        test_suite TEXT NOT NULL,
                        total_tests INTEGER,
                        passed_tests INTEGER,
                        failed_tests INTEGER,
                        skipped_tests INTEGER,
                        execution_time REAL,
                        coverage_percentage REAL,
                        memory_usage REAL,
                        cpu_usage REAL,
                        quality_score REAL,
                        security_score REAL,
                        performance_score REAL,
                        maintainability_score REAL,
                        reliability_score REAL,
                        test_density REAL,
                        code_complexity REAL,
                        technical_debt REAL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create indexes for better performance
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON test_metrics(timestamp)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_test_suite ON test_metrics(test_suite)")
                
                conn.commit()
                
        except Exception as e:
            print(f"❌ Error initializing database: {e}")
    
    def store_metrics(self, metrics: TestMetrics):
        """Store test metrics in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO test_metrics (
                        timestamp, test_suite, total_tests, passed_tests, failed_tests,
                        skipped_tests, execution_time, coverage_percentage, memory_usage,
                        cpu_usage, quality_score, security_score, performance_score,
                        maintainability_score, reliability_score, test_density,
                        code_complexity, technical_debt
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    metrics.timestamp, metrics.test_suite, metrics.total_tests,
                    metrics.passed_tests, metrics.failed_tests, metrics.skipped_tests,
                    metrics.execution_time, metrics.coverage_percentage, metrics.memory_usage,
                    metrics.cpu_usage, metrics.quality_score, metrics.security_score,
                    metrics.performance_score, metrics.maintainability_score,
                    metrics.reliability_score, metrics.test_density,
                    metrics.code_complexity, metrics.technical_debt
                ))
                
                conn.commit()
                return True
                
        except Exception as e:
            print(f"❌ Error storing metrics: {e}")
            return False
    
    def get_metrics_history(self, days: int = 30, test_suite: Optional[str] = None) -> List[TestMetrics]:
        """Retrieve metrics history from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Calculate date range
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days)
                
                query = """
                    SELECT * FROM test_metrics 
                    WHERE timestamp >= ? AND timestamp <= ?
                """
                params = [start_date.isoformat(), end_date.isoformat()]
                
                if test_suite:
                    query += " AND test_suite = ?"
                    params.append(test_suite)
                
                query += " ORDER BY timestamp DESC"
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                metrics_list = []
                for row in rows:
                    metrics = TestMetrics(
                        timestamp=row[1],
                        test_suite=row[2],
                        total_tests=row[3],
                        passed_tests=row[4],
                        failed_tests=row[5],
                        skipped_tests=row[6],
                        execution_time=row[7],
                        coverage_percentage=row[8],
                        memory_usage=row[9],
                        cpu_usage=row[10],
                        quality_score=row[11],
                        security_score=row[12],
                        performance_score=row[13],
                        maintainability_score=row[14],
                        reliability_score=row[15],
                        test_density=row[16],
                        code_complexity=row[17],
                        technical_debt=row[18]
                    )
                    metrics_list.append(metrics)
                
                return metrics_list
                
        except Exception as e:
            print(f"❌ Error retrieving metrics: {e}")
            return []
    
    def calculate_trends(self, metrics_list: List[TestMetrics]) -> List[TrendAnalysis]:
        """Calculate trend analysis for metrics"""
        if len(metrics_list) < 2:
            return []
        
        # Sort by timestamp
        sorted_metrics = sorted(metrics_list, key=lambda x: x.timestamp)
        
        trends = []
        metric_fields = [
            'execution_time', 'coverage_percentage', 'memory_usage', 'cpu_usage',
            'quality_score', 'security_score', 'performance_score',
            'maintainability_score', 'reliability_score', 'test_density',
            'code_complexity', 'technical_debt'
        ]
        
        for field in metric_fields:
            current_value = getattr(sorted_metrics[-1], field)
            previous_value = getattr(sorted_metrics[-2], field)
            
            if previous_value == 0:
                trend_percentage = 0
            else:
                trend_percentage = ((current_value - previous_value) / previous_value) * 100
            
            # Determine trend direction
            if abs(trend_percentage) < 1:
                trend_direction = "stable"
            elif trend_percentage > 0:
                trend_direction = "improving" if field in ['coverage_percentage', 'quality_score', 'security_score', 'performance_score', 'maintainability_score', 'reliability_score', 'test_density'] else "declining"
            else:
                trend_direction = "declining" if field in ['coverage_percentage', 'quality_score', 'security_score', 'performance_score', 'maintainability_score', 'reliability_score', 'test_density'] else "improving"
            
            # Determine significance
            if abs(trend_percentage) > 10:
                trend_significance = "high"
            elif abs(trend_percentage) > 5:
                trend_significance = "medium"
            else:
                trend_significance = "low"
            
            # Generate recommendations
            recommendations = self._generate_recommendations(field, trend_direction, trend_percentage)
            
            trend = TrendAnalysis(
                metric_name=field,
                current_value=current_value,
                previous_value=previous_value,
                trend_direction=trend_direction,
                trend_percentage=trend_percentage,
                trend_significance=trend_significance,
                recommendations=recommendations
            )
            trends.append(trend)
        
        return trends
    
    def _generate_recommendations(self, metric_name: str, trend_direction: str, trend_percentage: float) -> List[str]:
        """Generate recommendations based on trend analysis"""
        recommendations = []
        
        if metric_name == 'coverage_percentage':
            if trend_direction == "declining":
                recommendations.extend([
                    "Add more test cases to improve coverage",
                    "Review uncovered code paths",
                    "Consider adding integration tests"
                ])
            elif trend_direction == "improving":
                recommendations.append("Continue maintaining high coverage standards")
        
        elif metric_name == 'execution_time':
            if trend_direction == "declining":
                recommendations.extend([
                    "Optimize test execution with parallelization",
                    "Review slow-running tests",
                    "Consider test data optimization"
                ])
        
        elif metric_name == 'quality_score':
            if trend_direction == "declining":
                recommendations.extend([
                    "Review code quality metrics",
                    "Address technical debt",
                    "Improve code maintainability"
                ])
        
        elif metric_name == 'security_score':
            if trend_direction == "declining":
                recommendations.extend([
                    "Run security analysis tools",
                    "Review security test coverage",
                    "Update security dependencies"
                ])
        
        return recommendations
    
    def generate_dashboard_report(self, days: int = 30) -> Dict[str, Any]:
        """Generate comprehensive dashboard report"""
        print("📊 Generating Advanced Analytics Dashboard...")
        
        # Get metrics history
        metrics_list = self.get_metrics_history(days)
        
        if not metrics_list:
            return {"error": "No metrics data available"}
        
        # Calculate trends
        trends = self.calculate_trends(metrics_list)
        
        # Calculate summary statistics
        summary = self._calculate_summary_statistics(metrics_list)
        
        # Generate insights
        insights = self._generate_insights(metrics_list, trends)
        
        # Create report
        report = {
            "timestamp": datetime.now().isoformat(),
            "period_days": days,
            "total_measurements": len(metrics_list),
            "summary_statistics": summary,
            "trend_analysis": [asdict(trend) for trend in trends],
            "insights": insights,
            "recommendations": self._generate_overall_recommendations(trends),
            "quality_metrics": self._calculate_quality_metrics(metrics_list),
            "performance_metrics": self._calculate_performance_metrics(metrics_list)
        }
        
        return report
    
    def _calculate_summary_statistics(self, metrics_list: List[TestMetrics]) -> Dict[str, Any]:
        """Calculate summary statistics for metrics"""
        if not metrics_list:
            return {}
        
        # Group by test suite
        suite_stats = defaultdict(list)
        for metrics in metrics_list:
            suite_stats[metrics.test_suite].append(metrics)
        
        summary = {
            "overall": {
                "avg_execution_time": statistics.mean([m.execution_time for m in metrics_list]),
                "avg_coverage": statistics.mean([m.coverage_percentage for m in metrics_list]),
                "avg_quality_score": statistics.mean([m.quality_score for m in metrics_list]),
                "total_tests": sum([m.total_tests for m in metrics_list]),
                "total_passed": sum([m.passed_tests for m in metrics_list]),
                "total_failed": sum([m.failed_tests for m in metrics_list]),
                "success_rate": (sum([m.passed_tests for m in metrics_list]) / sum([m.total_tests for m in metrics_list])) * 100 if sum([m.total_tests for m in metrics_list]) > 0 else 0
            },
            "by_suite": {}
        }
        
        for suite, suite_metrics in suite_stats.items():
            summary["by_suite"][suite] = {
                "avg_execution_time": statistics.mean([m.execution_time for m in suite_metrics]),
                "avg_coverage": statistics.mean([m.coverage_percentage for m in suite_metrics]),
                "avg_quality_score": statistics.mean([m.quality_score for m in suite_metrics]),
                "measurements": len(suite_metrics)
            }
        
        return summary
    
    def _generate_insights(self, metrics_list: List[TestMetrics], trends: List[TrendAnalysis]) -> List[str]:
        """Generate insights from metrics and trends"""
        insights = []
        
        # Coverage insights
        coverage_values = [m.coverage_percentage for m in metrics_list]
        if coverage_values:
            avg_coverage = statistics.mean(coverage_values)
            if avg_coverage > 90:
                insights.append("🎯 Excellent test coverage maintained above 90%")
            elif avg_coverage > 80:
                insights.append("✅ Good test coverage, consider improving to 90%+")
            else:
                insights.append("⚠️ Test coverage below 80%, needs improvement")
        
        # Performance insights
        execution_times = [m.execution_time for m in metrics_list]
        if execution_times:
            avg_time = statistics.mean(execution_times)
            if avg_time < 30:
                insights.append("⚡ Fast test execution times maintained")
            elif avg_time < 60:
                insights.append("⏱️ Moderate test execution times, consider optimization")
            else:
                insights.append("🐌 Slow test execution times, optimization needed")
        
        # Quality insights
        quality_scores = [m.quality_score for m in metrics_list]
        if quality_scores:
            avg_quality = statistics.mean(quality_scores)
            if avg_quality > 8.0:
                insights.append("🏆 High quality scores maintained")
            elif avg_quality > 6.0:
                insights.append("📈 Good quality scores, room for improvement")
            else:
                insights.append("📉 Quality scores need attention")
        
        # Trend insights
        significant_trends = [t for t in trends if t.trend_significance == "high"]
        if significant_trends:
            improving_trends = [t for t in significant_trends if t.trend_direction == "improving"]
            declining_trends = [t for t in significant_trends if t.trend_direction == "declining"]
            
            if improving_trends:
                insights.append(f"📈 {len(improving_trends)} metrics showing significant improvement")
            if declining_trends:
                insights.append(f"📉 {len(declining_trends)} metrics showing significant decline")
        
        return insights
    
    def _generate_overall_recommendations(self, trends: List[TrendAnalysis]) -> List[str]:
        """Generate overall recommendations based on trends"""
        recommendations = []
        
        # High priority recommendations
        high_priority_trends = [t for t in trends if t.trend_significance == "high" and t.trend_direction == "declining"]
        if high_priority_trends:
            recommendations.append("🚨 High Priority: Address declining metrics immediately")
        
        # Coverage recommendations
        coverage_trend = next((t for t in trends if t.metric_name == "coverage_percentage"), None)
        if coverage_trend and coverage_trend.trend_direction == "declining":
            recommendations.append("📊 Improve test coverage with additional test cases")
        
        # Performance recommendations
        performance_trend = next((t for t in trends if t.metric_name == "execution_time"), None)
        if performance_trend and performance_trend.trend_direction == "declining":
            recommendations.append("⚡ Optimize test execution performance")
        
        # Quality recommendations
        quality_trend = next((t for t in trends if t.metric_name == "quality_score"), None)
        if quality_trend and quality_trend.trend_direction == "declining":
            recommendations.append("🏆 Focus on code quality improvements")
        
        return recommendations
    
    def _calculate_quality_metrics(self, metrics_list: List[TestMetrics]) -> Dict[str, Any]:
        """Calculate quality-related metrics"""
        if not metrics_list:
            return {}
        
        return {
            "avg_quality_score": statistics.mean([m.quality_score for m in metrics_list]),
            "avg_security_score": statistics.mean([m.security_score for m in metrics_list]),
            "avg_maintainability_score": statistics.mean([m.maintainability_score for m in metrics_list]),
            "avg_reliability_score": statistics.mean([m.reliability_score for m in metrics_list]),
            "avg_technical_debt": statistics.mean([m.technical_debt for m in metrics_list]),
            "avg_code_complexity": statistics.mean([m.code_complexity for m in metrics_list])
        }
    
    def _calculate_performance_metrics(self, metrics_list: List[TestMetrics]) -> Dict[str, Any]:
        """Calculate performance-related metrics"""
        if not metrics_list:
            return {}
        
        return {
            "avg_execution_time": statistics.mean([m.execution_time for m in metrics_list]),
            "avg_memory_usage": statistics.mean([m.memory_usage for m in metrics_list]),
            "avg_cpu_usage": statistics.mean([m.cpu_usage for m in metrics_list]),
            "avg_performance_score": statistics.mean([m.performance_score for m in metrics_list]),
            "avg_test_density": statistics.mean([m.test_density for m in metrics_list])
        }
    
    def create_visualizations(self, days: int = 30, output_dir: str = "analytics_charts"):
        """Create visualizations for analytics dashboard"""
        if not self.visualization_available:
            print("⚠️  Visualization libraries not available. Skipping chart generation.")
            return
        
        print("📊 Creating analytics visualizations...")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Get metrics data
        metrics_list = self.get_metrics_history(days)
        if not metrics_list:
            print("❌ No metrics data available for visualization")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame([asdict(m) for m in metrics_list])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Create time series plots
        self._create_time_series_plots(df, output_path)
        
        # Create quality metrics plots
        self._create_quality_plots(df, output_path)
        
        # Create performance plots
        self._create_performance_plots(df, output_path)
        
        # Create summary dashboard
        self._create_summary_dashboard(df, output_path)
        
        print(f"✅ Visualizations created in: {output_path}")
    
    def _create_time_series_plots(self, df: pd.DataFrame, output_path: Path):
        """Create time series plots for key metrics"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Test Coverage Over Time', 'Execution Time Over Time', 
                          'Quality Score Over Time', 'Success Rate Over Time'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Coverage over time
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['coverage_percentage'], 
                      mode='lines+markers', name='Coverage %'),
            row=1, col=1
        )
        
        # Execution time over time
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['execution_time'], 
                      mode='lines+markers', name='Execution Time (s)'),
            row=1, col=2
        )
        
        # Quality score over time
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['quality_score'], 
                      mode='lines+markers', name='Quality Score'),
            row=2, col=1
        )
        
        # Success rate over time
        df['success_rate'] = (df['passed_tests'] / df['total_tests']) * 100
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['success_rate'], 
                      mode='lines+markers', name='Success Rate %'),
            row=2, col=2
        )
        
        fig.update_layout(height=800, title_text="Test Metrics Time Series Analysis")
        fig.write_html(output_path / "time_series_analysis.html")
    
    def _create_quality_plots(self, df: pd.DataFrame, output_path: Path):
        """Create quality metrics visualizations"""
        quality_metrics = ['quality_score', 'security_score', 'maintainability_score', 'reliability_score']
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=quality_metrics
        )
        
        for i, metric in enumerate(quality_metrics):
            row = (i // 2) + 1
            col = (i % 2) + 1
            
            fig.add_trace(
                go.Box(y=df[metric], name=metric.replace('_', ' ').title()),
                row=row, col=col
            )
        
        fig.update_layout(height=600, title_text="Quality Metrics Distribution")
        fig.write_html(output_path / "quality_metrics.html")
    
    def _create_performance_plots(self, df: pd.DataFrame, output_path: Path):
        """Create performance metrics visualizations"""
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=('Execution Time Distribution', 'Memory Usage Over Time', 'CPU Usage Over Time')
        )
        
        # Execution time distribution
        fig.add_trace(
            go.Histogram(x=df['execution_time'], name='Execution Time'),
            row=1, col=1
        )
        
        # Memory usage over time
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['memory_usage'], 
                      mode='lines+markers', name='Memory Usage'),
            row=1, col=2
        )
        
        # CPU usage over time
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['cpu_usage'], 
                      mode='lines+markers', name='CPU Usage'),
            row=1, col=3
        )
        
        fig.update_layout(height=400, title_text="Performance Metrics Analysis")
        fig.write_html(output_path / "performance_metrics.html")
    
    def _create_summary_dashboard(self, df: pd.DataFrame, output_path: Path):
        """Create comprehensive summary dashboard"""
        # Calculate summary statistics
        summary_stats = {
            'Total Tests': df['total_tests'].sum(),
            'Passed Tests': df['passed_tests'].sum(),
            'Failed Tests': df['failed_tests'].sum(),
            'Avg Coverage': df['coverage_percentage'].mean(),
            'Avg Execution Time': df['execution_time'].mean(),
            'Avg Quality Score': df['quality_score'].mean()
        }
        
        # Create summary dashboard
        fig = go.Figure()
        
        # Add summary text
        summary_text = "<br>".join([f"<b>{k}:</b> {v:.2f}" if isinstance(v, float) else f"<b>{k}:</b> {v}" 
                                   for k, v in summary_stats.items()])
        
        fig.add_annotation(
            text=summary_text,
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=16)
        )
        
        fig.update_layout(
            title="Test Analytics Summary Dashboard",
            height=400,
            showlegend=False
        )
        
        fig.write_html(output_path / "summary_dashboard.html")
    
    def export_analytics_data(self, output_file: str = "analytics_export.json"):
        """Export analytics data to JSON file"""
        print("📤 Exporting analytics data...")
        
        # Get comprehensive data
        metrics_list = self.get_metrics_history(365)  # Last year
        trends = self.calculate_trends(metrics_list)
        report = self.generate_dashboard_report(365)
        
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "metrics_data": [asdict(m) for m in metrics_list],
            "trend_analysis": [asdict(t) for t in trends],
            "dashboard_report": report,
            "database_info": {
                "database_path": str(self.db_path),
                "total_records": len(metrics_list)
            }
        }
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, default=str)
            print(f"✅ Analytics data exported to: {output_file}")
            return True
        except Exception as e:
            print(f"❌ Error exporting data: {e}")
            return False


def main():
    """Main function for analytics dashboard"""
    print("📊 Advanced Test Analytics Dashboard")
    print("=" * 50)
    
    # Initialize dashboard
    dashboard = TestAnalyticsDashboard()
    
    # Generate sample data for demonstration
    print("📈 Generating sample analytics data...")
    
    sample_metrics = TestMetrics(
        timestamp=datetime.now().isoformat(),
        test_suite="comprehensive_tests",
        total_tests=150,
        passed_tests=145,
        failed_tests=3,
        skipped_tests=2,
        execution_time=25.5,
        coverage_percentage=92.3,
        memory_usage=256.7,
        cpu_usage=45.2,
        quality_score=8.7,
        security_score=9.1,
        performance_score=8.9,
        maintainability_score=8.5,
        reliability_score=9.0,
        test_density=0.85,
        code_complexity=6.2,
        technical_debt=12.5
    )
    
    # Store sample metrics
    dashboard.store_metrics(sample_metrics)
    
    # Generate dashboard report
    report = dashboard.generate_dashboard_report(30)
    
    # Print summary
    print("\n📊 Analytics Dashboard Summary:")
    print(f"  📈 Total measurements: {report.get('total_measurements', 0)}")
    print(f"  📅 Period: {report.get('period_days', 0)} days")
    
    if 'summary_statistics' in report:
        summary = report['summary_statistics'].get('overall', {})
        print(f"  ✅ Success rate: {summary.get('success_rate', 0):.1f}%")
        print(f"  📊 Avg coverage: {summary.get('avg_coverage', 0):.1f}%")
        print(f"  ⏱️ Avg execution time: {summary.get('avg_execution_time', 0):.1f}s")
        print(f"  🏆 Avg quality score: {summary.get('avg_quality_score', 0):.1f}/10")
    
    # Print insights
    if 'insights' in report:
        print("\n💡 Key Insights:")
        for insight in report['insights']:
            print(f"  {insight}")
    
    # Print recommendations
    if 'recommendations' in report:
        print("\n🎯 Recommendations:")
        for rec in report['recommendations']:
            print(f"  {rec}")
    
    # Create visualizations
    dashboard.create_visualizations(30, "analytics_charts")
    
    # Export data
    dashboard.export_analytics_data("analytics_export.json")
    
    print("\n🎉 Analytics dashboard generation completed!")
    print("📁 Check 'analytics_charts/' for visualizations")
    print("📄 Check 'analytics_export.json' for exported data")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())


