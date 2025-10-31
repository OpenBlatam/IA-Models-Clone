"""
ML NLP Benchmark Analytics System
Real, working data analysis and reporting for ML NLP Benchmark system
"""

import time
import json
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from collections import defaultdict, Counter
import statistics
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

logger = logging.getLogger(__name__)

@dataclass
class AnalyticsReport:
    """Analytics report structure"""
    report_id: str
    report_name: str
    report_type: str
    generated_at: datetime
    data_period: Tuple[datetime, datetime]
    summary: Dict[str, Any]
    insights: List[str]
    recommendations: List[str]
    charts: List[Dict[str, Any]]
    raw_data: Dict[str, Any]
    metadata: Dict[str, Any]

@dataclass
class DataInsight:
    """Data insight structure"""
    insight_id: str
    insight_type: str
    title: str
    description: str
    confidence: float
    impact: str
    data_points: List[Any]
    timestamp: datetime
    metadata: Dict[str, Any]

class MLNLPBenchmarkAnalytics:
    """Advanced analytics and reporting system"""
    
    def __init__(self):
        self.analytics_data = defaultdict(list)
        self.reports = {}
        self.insights = []
        self.lock = threading.RLock()
        
        # Analytics categories
        self.analytics_categories = {
            "usage": ["requests", "users", "endpoints", "methods"],
            "performance": ["response_time", "throughput", "error_rate", "cpu_usage", "memory_usage"],
            "content": ["text_length", "language", "sentiment", "topics", "entities"],
            "models": ["model_usage", "model_performance", "model_accuracy", "model_errors"],
            "system": ["health", "alerts", "resources", "capacity"]
        }
        
        # Report templates
        self.report_templates = {
            "daily_summary": {
                "name": "Daily Summary Report",
                "sections": ["usage", "performance", "content", "models"],
                "charts": ["usage_trend", "performance_metrics", "content_distribution", "model_performance"]
            },
            "weekly_analysis": {
                "name": "Weekly Analysis Report",
                "sections": ["usage", "performance", "content", "models", "system"],
                "charts": ["usage_trend", "performance_trend", "content_analysis", "model_comparison", "system_health"]
            },
            "monthly_report": {
                "name": "Monthly Report",
                "sections": ["usage", "performance", "content", "models", "system"],
                "charts": ["usage_summary", "performance_summary", "content_summary", "model_summary", "system_summary"]
            },
            "performance_analysis": {
                "name": "Performance Analysis Report",
                "sections": ["performance", "system"],
                "charts": ["response_time_distribution", "throughput_analysis", "resource_usage", "bottleneck_analysis"]
            },
            "content_analysis": {
                "name": "Content Analysis Report",
                "sections": ["content", "models"],
                "charts": ["language_distribution", "sentiment_analysis", "topic_modeling", "entity_extraction"]
            }
        }
    
    def add_data(self, category: str, data: Dict[str, Any]):
        """Add data to analytics"""
        with self.lock:
            data["timestamp"] = datetime.now()
            self.analytics_data[category].append(data)
            
            # Keep only last 10000 entries per category
            if len(self.analytics_data[category]) > 10000:
                self.analytics_data[category] = self.analytics_data[category][-10000:]
    
    def get_data(self, category: str, time_range: Optional[Tuple[datetime, datetime]] = None) -> List[Dict[str, Any]]:
        """Get analytics data for a category"""
        with self.lock:
            data = self.analytics_data.get(category, [])
            
            if time_range:
                start_time, end_time = time_range
                data = [d for d in data if start_time <= d["timestamp"] <= end_time]
            
            return data
    
    def analyze_usage_patterns(self, time_range: Optional[Tuple[datetime, datetime]] = None) -> Dict[str, Any]:
        """Analyze usage patterns"""
        usage_data = self.get_data("usage", time_range)
        
        if not usage_data:
            return {"error": "No usage data available"}
        
        df = pd.DataFrame(usage_data)
        
        # Convert timestamp to datetime if it's a string
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        analysis = {
            "total_requests": len(usage_data),
            "unique_users": df['user_id'].nunique() if 'user_id' in df.columns else 0,
            "unique_endpoints": df['endpoint'].nunique() if 'endpoint' in df.columns else 0,
            "request_methods": df['method'].value_counts().to_dict() if 'method' in df.columns else {},
            "endpoint_usage": df['endpoint'].value_counts().to_dict() if 'endpoint' in df.columns else {},
            "hourly_distribution": self._get_hourly_distribution(df),
            "daily_distribution": self._get_daily_distribution(df),
            "peak_usage_hour": self._get_peak_usage_hour(df),
            "peak_usage_day": self._get_peak_usage_day(df)
        }
        
        return analysis
    
    def analyze_performance_metrics(self, time_range: Optional[Tuple[datetime, datetime]] = None) -> Dict[str, Any]:
        """Analyze performance metrics"""
        performance_data = self.get_data("performance", time_range)
        
        if not performance_data:
            return {"error": "No performance data available"}
        
        df = pd.DataFrame(performance_data)
        
        analysis = {
            "total_requests": len(performance_data),
            "average_response_time": df['response_time'].mean() if 'response_time' in df.columns else 0,
            "median_response_time": df['response_time'].median() if 'response_time' in df.columns else 0,
            "p95_response_time": df['response_time'].quantile(0.95) if 'response_time' in df.columns else 0,
            "p99_response_time": df['response_time'].quantile(0.99) if 'response_time' in df.columns else 0,
            "min_response_time": df['response_time'].min() if 'response_time' in df.columns else 0,
            "max_response_time": df['response_time'].max() if 'response_time' in df.columns else 0,
            "error_rate": (df['status_code'] >= 400).mean() if 'status_code' in df.columns else 0,
            "success_rate": (df['status_code'] < 400).mean() if 'status_code' in df.columns else 0,
            "throughput_per_second": self._calculate_throughput(df),
            "response_time_distribution": self._get_response_time_distribution(df),
            "error_distribution": df['status_code'].value_counts().to_dict() if 'status_code' in df.columns else {}
        }
        
        return analysis
    
    def analyze_content_characteristics(self, time_range: Optional[Tuple[datetime, datetime]] = None) -> Dict[str, Any]:
        """Analyze content characteristics"""
        content_data = self.get_data("content", time_range)
        
        if not content_data:
            return {"error": "No content data available"}
        
        df = pd.DataFrame(content_data)
        
        analysis = {
            "total_texts": len(content_data),
            "average_text_length": df['text_length'].mean() if 'text_length' in df.columns else 0,
            "median_text_length": df['text_length'].median() if 'text_length' in df.columns else 0,
            "text_length_distribution": self._get_text_length_distribution(df),
            "language_distribution": df['language'].value_counts().to_dict() if 'language' in df.columns else {},
            "sentiment_distribution": df['sentiment'].value_counts().to_dict() if 'sentiment' in df.columns else {},
            "topic_distribution": self._get_topic_distribution(df),
            "entity_distribution": self._get_entity_distribution(df),
            "most_common_words": self._get_most_common_words(df),
            "complexity_analysis": self._analyze_complexity(df)
        }
        
        return analysis
    
    def analyze_model_performance(self, time_range: Optional[Tuple[datetime, datetime]] = None) -> Dict[str, Any]:
        """Analyze model performance"""
        model_data = self.get_data("models", time_range)
        
        if not model_data:
            return {"error": "No model data available"}
        
        df = pd.DataFrame(model_data)
        
        analysis = {
            "total_predictions": len(model_data),
            "unique_models": df['model_id'].nunique() if 'model_id' in df.columns else 0,
            "model_usage": df['model_id'].value_counts().to_dict() if 'model_id' in df.columns else {},
            "model_accuracy": self._calculate_model_accuracy(df),
            "model_performance": self._calculate_model_performance(df),
            "prediction_types": df['prediction_type'].value_counts().to_dict() if 'prediction_type' in df.columns else {},
            "confidence_distribution": self._get_confidence_distribution(df),
            "error_analysis": self._analyze_model_errors(df)
        }
        
        return analysis
    
    def generate_insights(self, time_range: Optional[Tuple[datetime, datetime]] = None) -> List[DataInsight]:
        """Generate data insights"""
        insights = []
        
        # Usage insights
        usage_analysis = self.analyze_usage_patterns(time_range)
        if "error" not in usage_analysis:
            insights.extend(self._generate_usage_insights(usage_analysis))
        
        # Performance insights
        performance_analysis = self.analyze_performance_metrics(time_range)
        if "error" not in performance_analysis:
            insights.extend(self._generate_performance_insights(performance_analysis))
        
        # Content insights
        content_analysis = self.analyze_content_characteristics(time_range)
        if "error" not in content_analysis:
            insights.extend(self._generate_content_insights(content_analysis))
        
        # Model insights
        model_analysis = self.analyze_model_performance(time_range)
        if "error" not in model_analysis:
            insights.extend(self._generate_model_insights(model_analysis))
        
        return insights
    
    def generate_report(self, report_type: str, time_range: Optional[Tuple[datetime, datetime]] = None) -> AnalyticsReport:
        """Generate analytics report"""
        if report_type not in self.report_templates:
            raise ValueError(f"Unknown report type: {report_type}")
        
        template = self.report_templates[report_type]
        report_id = f"{report_type}_{int(time.time())}"
        
        # Generate summary data
        summary = {}
        insights = []
        recommendations = []
        charts = []
        
        # Analyze each section
        for section in template["sections"]:
            if section == "usage":
                summary["usage"] = self.analyze_usage_patterns(time_range)
            elif section == "performance":
                summary["performance"] = self.analyze_performance_metrics(time_range)
            elif section == "content":
                summary["content"] = self.analyze_content_characteristics(time_range)
            elif section == "models":
                summary["models"] = self.analyze_model_performance(time_range)
            elif section == "system":
                summary["system"] = self.analyze_system_health(time_range)
        
        # Generate insights
        insights = self.generate_insights(time_range)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(summary, insights)
        
        # Generate charts
        for chart_type in template["charts"]:
            chart_data = self._generate_chart(chart_type, summary, time_range)
            if chart_data:
                charts.append(chart_data)
        
        # Create report
        report = AnalyticsReport(
            report_id=report_id,
            report_name=template["name"],
            report_type=report_type,
            generated_at=datetime.now(),
            data_period=time_range or (datetime.now() - timedelta(days=1), datetime.now()),
            summary=summary,
            insights=[insight.description for insight in insights],
            recommendations=recommendations,
            charts=charts,
            raw_data=summary,
            metadata={
                "template": template,
                "data_points": sum(len(self.get_data(category, time_range)) for category in self.analytics_categories.keys()),
                "generation_time": time.time()
            }
        )
        
        # Store report
        with self.lock:
            self.reports[report_id] = report
        
        return report
    
    def get_report(self, report_id: str) -> Optional[AnalyticsReport]:
        """Get analytics report"""
        return self.reports.get(report_id)
    
    def list_reports(self, report_type: Optional[str] = None) -> List[AnalyticsReport]:
        """List all reports"""
        reports = list(self.reports.values())
        
        if report_type:
            reports = [r for r in reports if r.report_type == report_type]
        
        return sorted(reports, key=lambda x: x.generated_at, reverse=True)
    
    def export_report(self, report_id: str, format: str = "json") -> str:
        """Export report in specified format"""
        report = self.get_report(report_id)
        if not report:
            raise ValueError(f"Report {report_id} not found")
        
        if format == "json":
            return json.dumps({
                "report_id": report.report_id,
                "report_name": report.report_name,
                "report_type": report.report_type,
                "generated_at": report.generated_at.isoformat(),
                "data_period": [report.data_period[0].isoformat(), report.data_period[1].isoformat()],
                "summary": report.summary,
                "insights": report.insights,
                "recommendations": report.recommendations,
                "metadata": report.metadata
            }, indent=2)
        elif format == "csv":
            # Export summary data as CSV
            csv_data = []
            for section, data in report.summary.items():
                if isinstance(data, dict):
                    for key, value in data.items():
                        csv_data.append({
                            "section": section,
                            "metric": key,
                            "value": value
                        })
            
            df = pd.DataFrame(csv_data)
            return df.to_csv(index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _get_hourly_distribution(self, df: pd.DataFrame) -> Dict[str, int]:
        """Get hourly distribution of requests"""
        if 'timestamp' not in df.columns:
            return {}
        
        df['hour'] = df['timestamp'].dt.hour
        return df['hour'].value_counts().sort_index().to_dict()
    
    def _get_daily_distribution(self, df: pd.DataFrame) -> Dict[str, int]:
        """Get daily distribution of requests"""
        if 'timestamp' not in df.columns:
            return {}
        
        df['day'] = df['timestamp'].dt.day_name()
        return df['day'].value_counts().to_dict()
    
    def _get_peak_usage_hour(self, df: pd.DataFrame) -> int:
        """Get peak usage hour"""
        if 'timestamp' not in df.columns:
            return 0
        
        df['hour'] = df['timestamp'].dt.hour
        return df['hour'].mode().iloc[0] if not df['hour'].mode().empty else 0
    
    def _get_peak_usage_day(self, df: pd.DataFrame) -> str:
        """Get peak usage day"""
        if 'timestamp' not in df.columns:
            return ""
        
        df['day'] = df['timestamp'].dt.day_name()
        return df['day'].mode().iloc[0] if not df['day'].mode().empty else ""
    
    def _calculate_throughput(self, df: pd.DataFrame) -> float:
        """Calculate throughput per second"""
        if 'timestamp' not in df.columns or len(df) < 2:
            return 0.0
        
        time_span = (df['timestamp'].max() - df['timestamp'].min()).total_seconds()
        return len(df) / max(time_span, 1)
    
    def _get_response_time_distribution(self, df: pd.DataFrame) -> Dict[str, float]:
        """Get response time distribution"""
        if 'response_time' not in df.columns:
            return {}
        
        return {
            "0-100ms": len(df[df['response_time'] < 0.1]) / len(df) * 100,
            "100ms-500ms": len(df[(df['response_time'] >= 0.1) & (df['response_time'] < 0.5)]) / len(df) * 100,
            "500ms-1s": len(df[(df['response_time'] >= 0.5) & (df['response_time'] < 1.0)]) / len(df) * 100,
            "1s-5s": len(df[(df['response_time'] >= 1.0) & (df['response_time'] < 5.0)]) / len(df) * 100,
            "5s+": len(df[df['response_time'] >= 5.0]) / len(df) * 100
        }
    
    def _get_text_length_distribution(self, df: pd.DataFrame) -> Dict[str, int]:
        """Get text length distribution"""
        if 'text_length' not in df.columns:
            return {}
        
        return {
            "0-100": len(df[df['text_length'] < 100]),
            "100-500": len(df[(df['text_length'] >= 100) & (df['text_length'] < 500)]),
            "500-1000": len(df[(df['text_length'] >= 500) & (df['text_length'] < 1000)]),
            "1000-5000": len(df[(df['text_length'] >= 1000) & (df['text_length'] < 5000)]),
            "5000+": len(df[df['text_length'] >= 5000])
        }
    
    def _get_topic_distribution(self, df: pd.DataFrame) -> Dict[str, int]:
        """Get topic distribution"""
        if 'topics' not in df.columns:
            return {}
        
        all_topics = []
        for topics in df['topics']:
            if isinstance(topics, list):
                all_topics.extend(topics)
            elif isinstance(topics, str):
                all_topics.append(topics)
        
        return Counter(all_topics).most_common(10)
    
    def _get_entity_distribution(self, df: pd.DataFrame) -> Dict[str, int]:
        """Get entity distribution"""
        if 'entities' not in df.columns:
            return {}
        
        all_entities = []
        for entities in df['entities']:
            if isinstance(entities, dict):
                for entity_type, entity_list in entities.items():
                    if isinstance(entity_list, list):
                        all_entities.extend([f"{entity_type}:{entity}" for entity in entity_list])
        
        return Counter(all_entities).most_common(10)
    
    def _get_most_common_words(self, df: pd.DataFrame) -> List[Tuple[str, int]]:
        """Get most common words"""
        if 'words' not in df.columns:
            return []
        
        all_words = []
        for words in df['words']:
            if isinstance(words, list):
                all_words.extend(words)
        
        return Counter(all_words).most_common(20)
    
    def _analyze_complexity(self, df: pd.DataFrame) -> Dict[str, float]:
        """Analyze text complexity"""
        if 'complexity' not in df.columns:
            return {}
        
        return {
            "average_complexity": df['complexity'].mean(),
            "median_complexity": df['complexity'].median(),
            "complexity_std": df['complexity'].std()
        }
    
    def _calculate_model_accuracy(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate model accuracy"""
        if 'model_id' not in df.columns or 'accuracy' not in df.columns:
            return {}
        
        return df.groupby('model_id')['accuracy'].mean().to_dict()
    
    def _calculate_model_performance(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Calculate model performance metrics"""
        if 'model_id' not in df.columns:
            return {}
        
        performance = {}
        for model_id in df['model_id'].unique():
            model_data = df[df['model_id'] == model_id]
            performance[model_id] = {
                "total_predictions": len(model_data),
                "average_confidence": model_data['confidence'].mean() if 'confidence' in model_data.columns else 0,
                "average_processing_time": model_data['processing_time'].mean() if 'processing_time' in model_data.columns else 0
            }
        
        return performance
    
    def _get_confidence_distribution(self, df: pd.DataFrame) -> Dict[str, int]:
        """Get confidence distribution"""
        if 'confidence' not in df.columns:
            return {}
        
        return {
            "0-0.5": len(df[df['confidence'] < 0.5]),
            "0.5-0.7": len(df[(df['confidence'] >= 0.5) & (df['confidence'] < 0.7)]),
            "0.7-0.9": len(df[(df['confidence'] >= 0.7) & (df['confidence'] < 0.9)]),
            "0.9-1.0": len(df[df['confidence'] >= 0.9])
        }
    
    def _analyze_model_errors(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze model errors"""
        if 'error' not in df.columns:
            return {}
        
        error_data = df[df['error'] == True]
        return {
            "total_errors": len(error_data),
            "error_rate": len(error_data) / len(df) * 100,
            "error_by_model": error_data['model_id'].value_counts().to_dict() if 'model_id' in error_data.columns else {},
            "error_types": error_data['error_type'].value_counts().to_dict() if 'error_type' in error_data.columns else {}
        }
    
    def _generate_usage_insights(self, usage_analysis: Dict[str, Any]) -> List[DataInsight]:
        """Generate usage insights"""
        insights = []
        
        if usage_analysis.get("total_requests", 0) > 1000:
            insights.append(DataInsight(
                insight_id=f"high_usage_{int(time.time())}",
                insight_type="usage",
                title="High Usage Volume",
                description=f"System processed {usage_analysis['total_requests']} requests, indicating high usage",
                confidence=0.9,
                impact="positive",
                data_points=[usage_analysis['total_requests']],
                timestamp=datetime.now(),
                metadata={}
            ))
        
        return insights
    
    def _generate_performance_insights(self, performance_analysis: Dict[str, Any]) -> List[DataInsight]:
        """Generate performance insights"""
        insights = []
        
        if performance_analysis.get("average_response_time", 0) > 2.0:
            insights.append(DataInsight(
                insight_id=f"slow_response_{int(time.time())}",
                insight_type="performance",
                title="Slow Response Times",
                description=f"Average response time is {performance_analysis['average_response_time']:.2f}s, which is above optimal",
                confidence=0.8,
                impact="negative",
                data_points=[performance_analysis['average_response_time']],
                timestamp=datetime.now(),
                metadata={}
            ))
        
        return insights
    
    def _generate_content_insights(self, content_analysis: Dict[str, Any]) -> List[DataInsight]:
        """Generate content insights"""
        insights = []
        
        if content_analysis.get("average_text_length", 0) > 1000:
            insights.append(DataInsight(
                insight_id=f"long_texts_{int(time.time())}",
                insight_type="content",
                title="Long Text Processing",
                description=f"Average text length is {content_analysis['average_text_length']:.0f} characters, indicating complex content",
                confidence=0.7,
                impact="neutral",
                data_points=[content_analysis['average_text_length']],
                timestamp=datetime.now(),
                metadata={}
            ))
        
        return insights
    
    def _generate_model_insights(self, model_analysis: Dict[str, Any]) -> List[DataInsight]:
        """Generate model insights"""
        insights = []
        
        if model_analysis.get("total_predictions", 0) > 500:
            insights.append(DataInsight(
                insight_id=f"high_model_usage_{int(time.time())}",
                insight_type="models",
                title="High Model Usage",
                description=f"Models processed {model_analysis['total_predictions']} predictions, showing active usage",
                confidence=0.9,
                impact="positive",
                data_points=[model_analysis['total_predictions']],
                timestamp=datetime.now(),
                metadata={}
            ))
        
        return insights
    
    def _generate_recommendations(self, summary: Dict[str, Any], insights: List[DataInsight]) -> List[str]:
        """Generate recommendations based on analysis"""
        recommendations = []
        
        # Performance recommendations
        if "performance" in summary:
            perf = summary["performance"]
            if perf.get("average_response_time", 0) > 2.0:
                recommendations.append("Consider optimizing response times by implementing caching or improving algorithms")
            
            if perf.get("error_rate", 0) > 5.0:
                recommendations.append("High error rate detected. Review error handling and input validation")
        
        # Usage recommendations
        if "usage" in summary:
            usage = summary["usage"]
            if usage.get("total_requests", 0) > 10000:
                recommendations.append("High usage volume detected. Consider scaling infrastructure")
        
        # Model recommendations
        if "models" in summary:
            models = summary["models"]
            if models.get("total_predictions", 0) > 1000:
                recommendations.append("High model usage detected. Consider model optimization or additional resources")
        
        return recommendations
    
    def _generate_chart(self, chart_type: str, summary: Dict[str, Any], time_range: Optional[Tuple[datetime, datetime]]) -> Optional[Dict[str, Any]]:
        """Generate chart data"""
        try:
            if chart_type == "usage_trend":
                return self._create_usage_trend_chart(summary)
            elif chart_type == "performance_metrics":
                return self._create_performance_metrics_chart(summary)
            elif chart_type == "content_distribution":
                return self._create_content_distribution_chart(summary)
            elif chart_type == "model_performance":
                return self._create_model_performance_chart(summary)
            else:
                return None
        except Exception as e:
            logger.error(f"Error generating chart {chart_type}: {e}")
            return None
    
    def _create_usage_trend_chart(self, summary: Dict[str, Any]) -> Dict[str, Any]:
        """Create usage trend chart"""
        if "usage" not in summary:
            return {}
        
        usage = summary["usage"]
        return {
            "type": "line",
            "title": "Usage Trend",
            "data": {
                "labels": list(usage.get("hourly_distribution", {}).keys()),
                "datasets": [{
                    "label": "Requests",
                    "data": list(usage.get("hourly_distribution", {}).values())
                }]
            }
        }
    
    def _create_performance_metrics_chart(self, summary: Dict[str, Any]) -> Dict[str, Any]:
        """Create performance metrics chart"""
        if "performance" not in summary:
            return {}
        
        perf = summary["performance"]
        return {
            "type": "bar",
            "title": "Performance Metrics",
            "data": {
                "labels": ["Avg Response Time", "P95 Response Time", "Error Rate", "Throughput"],
                "datasets": [{
                    "label": "Values",
                    "data": [
                        perf.get("average_response_time", 0),
                        perf.get("p95_response_time", 0),
                        perf.get("error_rate", 0) * 100,
                        perf.get("throughput_per_second", 0)
                    ]
                }]
            }
        }
    
    def _create_content_distribution_chart(self, summary: Dict[str, Any]) -> Dict[str, Any]:
        """Create content distribution chart"""
        if "content" not in summary:
            return {}
        
        content = summary["content"]
        return {
            "type": "pie",
            "title": "Content Distribution",
            "data": {
                "labels": list(content.get("language_distribution", {}).keys()),
                "datasets": [{
                    "data": list(content.get("language_distribution", {}).values())
                }]
            }
        }
    
    def _create_model_performance_chart(self, summary: Dict[str, Any]) -> Dict[str, Any]:
        """Create model performance chart"""
        if "models" not in summary:
            return {}
        
        models = summary["models"]
        return {
            "type": "bar",
            "title": "Model Performance",
            "data": {
                "labels": list(models.get("model_usage", {}).keys()),
                "datasets": [{
                    "label": "Usage Count",
                    "data": list(models.get("model_usage", {}).values())
                }]
            }
        }
    
    def analyze_system_health(self, time_range: Optional[Tuple[datetime, datetime]] = None) -> Dict[str, Any]:
        """Analyze system health"""
        system_data = self.get_data("system", time_range)
        
        if not system_data:
            return {"error": "No system data available"}
        
        df = pd.DataFrame(system_data)
        
        return {
            "total_alerts": len(system_data),
            "critical_alerts": len(df[df['level'] == 'critical']) if 'level' in df.columns else 0,
            "warning_alerts": len(df[df['level'] == 'warning']) if 'level' in df.columns else 0,
            "system_uptime": self._calculate_system_uptime(df),
            "resource_usage": self._analyze_resource_usage(df),
            "health_score": self._calculate_health_score(df)
        }
    
    def _calculate_system_uptime(self, df: pd.DataFrame) -> float:
        """Calculate system uptime percentage"""
        if 'status' not in df.columns:
            return 100.0
        
        uptime_count = len(df[df['status'] == 'healthy'])
        return (uptime_count / len(df)) * 100 if len(df) > 0 else 100.0
    
    def _analyze_resource_usage(self, df: pd.DataFrame) -> Dict[str, float]:
        """Analyze resource usage"""
        if 'cpu_usage' not in df.columns or 'memory_usage' not in df.columns:
            return {}
        
        return {
            "average_cpu_usage": df['cpu_usage'].mean(),
            "peak_cpu_usage": df['cpu_usage'].max(),
            "average_memory_usage": df['memory_usage'].mean(),
            "peak_memory_usage": df['memory_usage'].max()
        }
    
    def _calculate_health_score(self, df: pd.DataFrame) -> float:
        """Calculate overall health score"""
        if len(df) == 0:
            return 100.0
        
        # Simple health score calculation
        uptime_score = self._calculate_system_uptime(df)
        resource_score = 100.0  # Default, could be calculated based on resource usage
        
        return (uptime_score + resource_score) / 2

# Global analytics instance
ml_nlp_benchmark_analytics = MLNLPBenchmarkAnalytics()

def get_analytics() -> MLNLPBenchmarkAnalytics:
    """Get the global analytics instance"""
    return ml_nlp_benchmark_analytics

def add_analytics_data(category: str, data: Dict[str, Any]):
    """Add data to analytics"""
    ml_nlp_benchmark_analytics.add_data(category, data)

def generate_report(report_type: str, time_range: Optional[Tuple[datetime, datetime]] = None) -> AnalyticsReport:
    """Generate analytics report"""
    return ml_nlp_benchmark_analytics.generate_report(report_type, time_range)

def get_report(report_id: str) -> Optional[AnalyticsReport]:
    """Get analytics report"""
    return ml_nlp_benchmark_analytics.get_report(report_id)

def list_reports(report_type: Optional[str] = None) -> List[AnalyticsReport]:
    """List all reports"""
    return ml_nlp_benchmark_analytics.list_reports(report_type)

def export_report(report_id: str, format: str = "json") -> str:
    """Export report in specified format"""
    return ml_nlp_benchmark_analytics.export_report(report_id, format)











