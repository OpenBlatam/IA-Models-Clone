"""
Data Export System
==================

Advanced data export system for AI model analysis with comprehensive
export formats, data transformation, and automated export capabilities.
"""

import asyncio
import logging
import json
import csv
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
import hashlib
import io
import base64
import zipfile
import tarfile
import gzip
import pickle
import yaml
import xml.etree.ElementTree as ET
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class ExportFormat(str, Enum):
    """Export formats"""
    JSON = "json"
    CSV = "csv"
    EXCEL = "excel"
    PARQUET = "parquet"
    PICKLE = "pickle"
    XML = "xml"
    YAML = "yaml"
    HTML = "html"
    PDF = "pdf"
    PNG = "png"
    SVG = "svg"
    ZIP = "zip"
    TAR = "tar"
    GZIP = "gzip"
    SQLITE = "sqlite"
    MYSQL = "mysql"
    POSTGRESQL = "postgresql"
    MONGODB = "mongodb"
    REDIS = "redis"
    ELASTICSEARCH = "elasticsearch"


class ExportType(str, Enum):
    """Export types"""
    PERFORMANCE_DATA = "performance_data"
    ANALYTICS_RESULTS = "analytics_results"
    COMPARISON_RESULTS = "comparison_results"
    BENCHMARK_RESULTS = "benchmark_results"
    PREDICTION_RESULTS = "prediction_results"
    OPTIMIZATION_RESULTS = "optimization_results"
    VISUALIZATIONS = "visualizations"
    DASHBOARDS = "dashboards"
    REPORTS = "reports"
    CONFIGURATIONS = "configurations"
    LOGS = "logs"
    METRICS = "metrics"
    COMPREHENSIVE = "comprehensive"


class ExportStatus(str, Enum):
    """Export status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ExportConfig:
    """Export configuration"""
    export_id: str
    export_type: ExportType
    export_format: ExportFormat
    data_sources: List[str]
    filters: Dict[str, Any]
    transformations: List[Dict[str, Any]]
    output_options: Dict[str, Any]
    compression: bool = False
    encryption: bool = False
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class ExportResult:
    """Export result"""
    export_id: str
    export_config: ExportConfig
    status: ExportStatus
    file_path: str = ""
    file_size: int = 0
    record_count: int = 0
    processing_time: float = 0.0
    error_message: str = ""
    metadata: Dict[str, Any] = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.metadata is None:
            self.metadata = {}


@dataclass
class DataTransformation:
    """Data transformation configuration"""
    transformation_id: str
    name: str
    description: str
    transformation_type: str
    parameters: Dict[str, Any]
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


class DataExportSystem:
    """Advanced data export system for AI model analysis"""
    
    def __init__(self, max_exports: int = 1000, output_directory: str = "./exports"):
        self.max_exports = max_exports
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(exist_ok=True)
        
        self.export_configs: Dict[str, ExportConfig] = {}
        self.export_results: List[ExportResult] = []
        self.data_transformations: Dict[str, DataTransformation] = {}
        
        # Export settings
        self.default_compression = True
        self.default_encryption = False
        self.max_file_size = 100 * 1024 * 1024  # 100MB
        self.batch_size = 10000
        
        # Cache for exports
        self.export_cache = {}
        self.cache_ttl = 3600  # 1 hour
    
    async def create_export_config(self, 
                                 export_type: ExportType,
                                 export_format: ExportFormat,
                                 data_sources: List[str],
                                 filters: Dict[str, Any] = None,
                                 transformations: List[Dict[str, Any]] = None,
                                 output_options: Dict[str, Any] = None) -> ExportConfig:
        """Create export configuration"""
        try:
            export_id = hashlib.md5(f"{export_type}_{export_format}_{datetime.now()}".encode()).hexdigest()
            
            if filters is None:
                filters = {}
            if transformations is None:
                transformations = []
            if output_options is None:
                output_options = {}
            
            config = ExportConfig(
                export_id=export_id,
                export_type=export_type,
                export_format=export_format,
                data_sources=data_sources,
                filters=filters,
                transformations=transformations,
                output_options=output_options,
                compression=self.default_compression,
                encryption=self.default_encryption
            )
            
            self.export_configs[export_id] = config
            
            logger.info(f"Created export config: {export_type.value} -> {export_format.value}")
            
            return config
            
        except Exception as e:
            logger.error(f"Error creating export config: {str(e)}")
            raise e
    
    async def execute_export(self, 
                           export_config: ExportConfig,
                           data_provider: Any = None) -> ExportResult:
        """Execute export with given configuration"""
        try:
            start_time = datetime.now()
            
            # Create export result
            result = ExportResult(
                export_id=export_config.export_id,
                export_config=export_config,
                status=ExportStatus.PROCESSING,
                created_at=start_time
            )
            
            # Collect data from sources
            data = await self._collect_export_data(export_config, data_provider)
            
            # Apply transformations
            transformed_data = await self._apply_transformations(data, export_config.transformations)
            
            # Generate filename
            filename = await self._generate_filename(export_config)
            file_path = self.output_directory / filename
            
            # Export data
            await self._export_data(transformed_data, file_path, export_config)
            
            # Calculate metrics
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            # Update result
            result.status = ExportStatus.COMPLETED
            result.file_path = str(file_path)
            result.file_size = file_path.stat().st_size if file_path.exists() else 0
            result.record_count = len(transformed_data) if isinstance(transformed_data, list) else 1
            result.processing_time = processing_time
            result.metadata = {
                "data_sources": export_config.data_sources,
                "transformations_applied": len(export_config.transformations),
                "compression_used": export_config.compression,
                "encryption_used": export_config.encryption
            }
            
            # Store result
            self.export_results.append(result)
            
            logger.info(f"Completed export: {export_config.export_type.value} -> {filename}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing export: {str(e)}")
            
            # Create failed result
            failed_result = ExportResult(
                export_id=export_config.export_id,
                export_config=export_config,
                status=ExportStatus.FAILED,
                error_message=str(e),
                created_at=datetime.now()
            )
            
            self.export_results.append(failed_result)
            
            return failed_result
    
    async def export_performance_data(self, 
                                    model_names: List[str] = None,
                                    time_range_days: int = 30,
                                    export_format: ExportFormat = ExportFormat.CSV) -> ExportResult:
        """Export performance data"""
        try:
            if model_names is None:
                model_names = ["gpt-4", "claude-3", "gemini-pro"]
            
            # Create export config
            config = await self.create_export_config(
                export_type=ExportType.PERFORMANCE_DATA,
                export_format=export_format,
                data_sources=["performance_data"],
                filters={
                    "model_names": model_names,
                    "time_range_days": time_range_days
                }
            )
            
            # Execute export
            result = await self.execute_export(config)
            
            return result
            
        except Exception as e:
            logger.error(f"Error exporting performance data: {str(e)}")
            raise e
    
    async def export_analytics_results(self, 
                                     analytics_type: str = "comprehensive",
                                     export_format: ExportFormat = ExportFormat.JSON) -> ExportResult:
        """Export analytics results"""
        try:
            # Create export config
            config = await self.create_export_config(
                export_type=ExportType.ANALYTICS_RESULTS,
                export_format=export_format,
                data_sources=["analytics_data"],
                filters={
                    "analytics_type": analytics_type
                }
            )
            
            # Execute export
            result = await self.execute_export(config)
            
            return result
            
        except Exception as e:
            logger.error(f"Error exporting analytics results: {str(e)}")
            raise e
    
    async def export_comparison_results(self, 
                                      model_pairs: List[Tuple[str, str]] = None,
                                      export_format: ExportFormat = ExportFormat.EXCEL) -> ExportResult:
        """Export comparison results"""
        try:
            if model_pairs is None:
                model_pairs = [("gpt-4", "claude-3"), ("gpt-4", "gemini-pro"), ("claude-3", "gemini-pro")]
            
            # Create export config
            config = await self.create_export_config(
                export_type=ExportType.COMPARISON_RESULTS,
                export_format=export_format,
                data_sources=["comparison_data"],
                filters={
                    "model_pairs": model_pairs
                }
            )
            
            # Execute export
            result = await self.execute_export(config)
            
            return result
            
        except Exception as e:
            logger.error(f"Error exporting comparison results: {str(e)}")
            raise e
    
    async def export_benchmark_results(self, 
                                     benchmark_suite_id: str = None,
                                     export_format: ExportFormat = ExportFormat.PARQUET) -> ExportResult:
        """Export benchmark results"""
        try:
            # Create export config
            config = await self.create_export_config(
                export_type=ExportType.BENCHMARK_RESULTS,
                export_format=export_format,
                data_sources=["benchmark_data"],
                filters={
                    "benchmark_suite_id": benchmark_suite_id
                }
            )
            
            # Execute export
            result = await self.execute_export(config)
            
            return result
            
        except Exception as e:
            logger.error(f"Error exporting benchmark results: {str(e)}")
            raise e
    
    async def export_comprehensive_report(self, 
                                        report_type: str = "executive",
                                        export_format: ExportFormat = ExportFormat.PDF) -> ExportResult:
        """Export comprehensive report"""
        try:
            # Create export config
            config = await self.create_export_config(
                export_type=ExportType.REPORTS,
                export_format=export_format,
                data_sources=["performance_data", "analytics_data", "comparison_data", "benchmark_data"],
                filters={
                    "report_type": report_type
                },
                transformations=[
                    {
                        "type": "aggregate",
                        "parameters": {"group_by": "model_name", "aggregations": ["mean", "std", "min", "max"]}
                    },
                    {
                        "type": "format",
                        "parameters": {"template": "executive_summary"}
                    }
                ]
            )
            
            # Execute export
            result = await self.execute_export(config)
            
            return result
            
        except Exception as e:
            logger.error(f"Error exporting comprehensive report: {str(e)}")
            raise e
    
    async def create_data_transformation(self, 
                                       name: str,
                                       description: str,
                                       transformation_type: str,
                                       parameters: Dict[str, Any],
                                       input_schema: Dict[str, Any],
                                       output_schema: Dict[str, Any]) -> DataTransformation:
        """Create data transformation"""
        try:
            transformation_id = hashlib.md5(f"{name}_{transformation_type}_{datetime.now()}".encode()).hexdigest()
            
            transformation = DataTransformation(
                transformation_id=transformation_id,
                name=name,
                description=description,
                transformation_type=transformation_type,
                parameters=parameters,
                input_schema=input_schema,
                output_schema=output_schema
            )
            
            self.data_transformations[transformation_id] = transformation
            
            logger.info(f"Created data transformation: {name}")
            
            return transformation
            
        except Exception as e:
            logger.error(f"Error creating data transformation: {str(e)}")
            raise e
    
    async def get_export_analytics(self, 
                                 time_range_days: int = 30) -> Dict[str, Any]:
        """Get export analytics"""
        try:
            cutoff_date = datetime.now() - timedelta(days=time_range_days)
            recent_exports = [r for r in self.export_results if r.created_at >= cutoff_date]
            
            analytics = {
                "total_exports": len(recent_exports),
                "successful_exports": len([r for r in recent_exports if r.status == ExportStatus.COMPLETED]),
                "failed_exports": len([r for r in recent_exports if r.status == ExportStatus.FAILED]),
                "success_rate": 0.0,
                "export_types": {},
                "export_formats": {},
                "total_data_exported": 0,
                "average_processing_time": 0.0,
                "popular_exports": [],
                "export_trends": {}
            }
            
            if recent_exports:
                # Calculate success rate
                successful = len([r for r in recent_exports if r.status == ExportStatus.COMPLETED])
                analytics["success_rate"] = successful / len(recent_exports)
                
                # Analyze export types
                for result in recent_exports:
                    export_type = result.export_config.export_type.value
                    if export_type not in analytics["export_types"]:
                        analytics["export_types"][export_type] = 0
                    analytics["export_types"][export_type] += 1
                
                # Analyze export formats
                for result in recent_exports:
                    export_format = result.export_config.export_format.value
                    if export_format not in analytics["export_formats"]:
                        analytics["export_formats"][export_format] = 0
                    analytics["export_formats"][export_format] += 1
                
                # Calculate total data exported
                analytics["total_data_exported"] = sum(r.file_size for r in recent_exports if r.file_size > 0)
                
                # Calculate average processing time
                processing_times = [r.processing_time for r in recent_exports if r.processing_time > 0]
                if processing_times:
                    analytics["average_processing_time"] = np.mean(processing_times)
                
                # Find popular exports
                export_type_counts = analytics["export_types"]
                if export_type_counts:
                    popular_exports = sorted(export_type_counts.items(), key=lambda x: x[1], reverse=True)
                    analytics["popular_exports"] = [{"type": export_type, "count": count} for export_type, count in popular_exports[:5]]
                
                # Analyze export trends
                daily_exports = defaultdict(int)
                for result in recent_exports:
                    date_key = result.created_at.date()
                    daily_exports[date_key] += 1
                
                analytics["export_trends"] = {
                    date.isoformat(): count for date, count in daily_exports.items()
                }
            
            return analytics
            
        except Exception as e:
            logger.error(f"Error getting export analytics: {str(e)}")
            return {"error": str(e)}
    
    # Private helper methods
    async def _collect_export_data(self, 
                                 export_config: ExportConfig, 
                                 data_provider: Any = None) -> List[Dict[str, Any]]:
        """Collect data from specified sources"""
        try:
            data = []
            
            for source in export_config.data_sources:
                if source == "performance_data":
                    source_data = await self._collect_performance_data(export_config.filters)
                elif source == "analytics_data":
                    source_data = await self._collect_analytics_data(export_config.filters)
                elif source == "comparison_data":
                    source_data = await self._collect_comparison_data(export_config.filters)
                elif source == "benchmark_data":
                    source_data = await self._collect_benchmark_data(export_config.filters)
                else:
                    source_data = []
                
                data.extend(source_data)
            
            return data
            
        except Exception as e:
            logger.error(f"Error collecting export data: {str(e)}")
            return []
    
    async def _collect_performance_data(self, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Collect performance data"""
        try:
            # Generate sample performance data
            model_names = filters.get("model_names", ["gpt-4", "claude-3", "gemini-pro"])
            time_range_days = filters.get("time_range_days", 30)
            
            data = []
            end_date = datetime.now()
            start_date = end_date - timedelta(days=time_range_days)
            
            for model_name in model_names:
                # Generate sample data points
                for i in range(min(100, time_range_days * 2)):  # Limit data points
                    timestamp = start_date + timedelta(hours=i)
                    
                    data_point = {
                        "timestamp": timestamp.isoformat(),
                        "model_name": model_name,
                        "performance_score": 0.7 + np.random.normal(0, 0.1),
                        "response_time": 1.0 + np.random.normal(0, 0.3),
                        "cost_efficiency": 0.8 + np.random.normal(0, 0.05),
                        "accuracy": 0.85 + np.random.normal(0, 0.08),
                        "throughput": 100 + np.random.normal(0, 20)
                    }
                    
                    # Ensure values are in valid range
                    for key, value in data_point.items():
                        if isinstance(value, float) and key != "timestamp":
                            data_point[key] = max(0, min(1, value)) if key in ["performance_score", "cost_efficiency", "accuracy"] else max(0, value)
                    
                    data.append(data_point)
            
            return data
            
        except Exception as e:
            logger.error(f"Error collecting performance data: {str(e)}")
            return []
    
    async def _collect_analytics_data(self, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Collect analytics data"""
        try:
            # Generate sample analytics data
            analytics_type = filters.get("analytics_type", "comprehensive")
            
            data = []
            models = ["gpt-4", "claude-3", "gemini-pro"]
            
            for model_name in models:
                analytics_point = {
                    "model_name": model_name,
                    "analytics_type": analytics_type,
                    "timestamp": datetime.now().isoformat(),
                    "insights_count": np.random.randint(5, 20),
                    "confidence_score": 0.8 + np.random.normal(0, 0.1),
                    "trend_direction": np.random.choice(["improving", "declining", "stable"]),
                    "anomalies_detected": np.random.randint(0, 5),
                    "recommendations_count": np.random.randint(3, 10)
                }
                
                data.append(analytics_point)
            
            return data
            
        except Exception as e:
            logger.error(f"Error collecting analytics data: {str(e)}")
            return []
    
    async def _collect_comparison_data(self, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Collect comparison data"""
        try:
            # Generate sample comparison data
            model_pairs = filters.get("model_pairs", [("gpt-4", "claude-3")])
            
            data = []
            
            for model_a, model_b in model_pairs:
                comparison_point = {
                    "model_a": model_a,
                    "model_b": model_b,
                    "comparison_timestamp": datetime.now().isoformat(),
                    "winner": np.random.choice([model_a, model_b, "tie"]),
                    "confidence": 0.7 + np.random.normal(0, 0.15),
                    "effect_size": np.random.uniform(0.1, 0.8),
                    "statistical_significance": np.random.choice([True, False]),
                    "p_value": np.random.uniform(0.001, 0.1),
                    "practical_significance": np.random.choice(["small", "medium", "large"])
                }
                
                data.append(comparison_point)
            
            return data
            
        except Exception as e:
            logger.error(f"Error collecting comparison data: {str(e)}")
            return []
    
    async def _collect_benchmark_data(self, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Collect benchmark data"""
        try:
            # Generate sample benchmark data
            benchmark_suite_id = filters.get("benchmark_suite_id", "default_suite")
            
            data = []
            models = ["RandomForest", "LogisticRegression", "SVC", "GradientBoosting"]
            
            for model_name in models:
                benchmark_point = {
                    "benchmark_suite_id": benchmark_suite_id,
                    "model_name": model_name,
                    "benchmark_timestamp": datetime.now().isoformat(),
                    "accuracy": 0.7 + np.random.normal(0, 0.1),
                    "precision": 0.75 + np.random.normal(0, 0.08),
                    "recall": 0.72 + np.random.normal(0, 0.09),
                    "f1_score": 0.73 + np.random.normal(0, 0.07),
                    "execution_time": 1.0 + np.random.normal(0, 0.5),
                    "memory_usage": 100 + np.random.normal(0, 20),
                    "rank": np.random.randint(1, 5)
                }
                
                # Ensure values are in valid range
                for key, value in benchmark_point.items():
                    if isinstance(value, float) and key in ["accuracy", "precision", "recall", "f1_score"]:
                        benchmark_point[key] = max(0, min(1, value))
                    elif key in ["execution_time", "memory_usage"]:
                        benchmark_point[key] = max(0, value)
                
                data.append(benchmark_point)
            
            return data
            
        except Exception as e:
            logger.error(f"Error collecting benchmark data: {str(e)}")
            return []
    
    async def _apply_transformations(self, 
                                   data: List[Dict[str, Any]], 
                                   transformations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply data transformations"""
        try:
            transformed_data = data.copy()
            
            for transformation in transformations:
                transformation_type = transformation.get("type", "")
                parameters = transformation.get("parameters", {})
                
                if transformation_type == "filter":
                    transformed_data = await self._apply_filter_transformation(transformed_data, parameters)
                elif transformation_type == "aggregate":
                    transformed_data = await self._apply_aggregate_transformation(transformed_data, parameters)
                elif transformation_type == "sort":
                    transformed_data = await self._apply_sort_transformation(transformed_data, parameters)
                elif transformation_type == "format":
                    transformed_data = await self._apply_format_transformation(transformed_data, parameters)
                elif transformation_type == "calculate":
                    transformed_data = await self._apply_calculate_transformation(transformed_data, parameters)
            
            return transformed_data
            
        except Exception as e:
            logger.error(f"Error applying transformations: {str(e)}")
            return data
    
    async def _apply_filter_transformation(self, data: List[Dict[str, Any]], parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply filter transformation"""
        try:
            filtered_data = []
            
            for item in data:
                include = True
                
                for field, value in parameters.items():
                    if field in item and item[field] != value:
                        include = False
                        break
                
                if include:
                    filtered_data.append(item)
            
            return filtered_data
            
        except Exception as e:
            logger.error(f"Error applying filter transformation: {str(e)}")
            return data
    
    async def _apply_aggregate_transformation(self, data: List[Dict[str, Any]], parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply aggregate transformation"""
        try:
            group_by = parameters.get("group_by", "")
            aggregations = parameters.get("aggregations", ["mean"])
            
            if not group_by or not data:
                return data
            
            # Group data
            groups = defaultdict(list)
            for item in data:
                if group_by in item:
                    groups[item[group_by]].append(item)
            
            # Aggregate each group
            aggregated_data = []
            for group_key, group_items in groups.items():
                if not group_items:
                    continue
                
                aggregated_item = {group_by: group_key}
                
                # Get numeric fields
                numeric_fields = set()
                for item in group_items:
                    for key, value in item.items():
                        if isinstance(value, (int, float)) and key != group_by:
                            numeric_fields.add(key)
                
                # Apply aggregations
                for field in numeric_fields:
                    values = [item[field] for item in group_items if field in item and isinstance(item[field], (int, float))]
                    
                    if values:
                        for agg in aggregations:
                            if agg == "mean":
                                aggregated_item[f"{field}_mean"] = np.mean(values)
                            elif agg == "std":
                                aggregated_item[f"{field}_std"] = np.std(values)
                            elif agg == "min":
                                aggregated_item[f"{field}_min"] = np.min(values)
                            elif agg == "max":
                                aggregated_item[f"{field}_max"] = np.max(values)
                            elif agg == "count":
                                aggregated_item[f"{field}_count"] = len(values)
                
                aggregated_data.append(aggregated_item)
            
            return aggregated_data
            
        except Exception as e:
            logger.error(f"Error applying aggregate transformation: {str(e)}")
            return data
    
    async def _apply_sort_transformation(self, data: List[Dict[str, Any]], parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply sort transformation"""
        try:
            sort_field = parameters.get("field", "")
            sort_order = parameters.get("order", "asc")
            
            if not sort_field or not data:
                return data
            
            # Sort data
            reverse = sort_order.lower() == "desc"
            sorted_data = sorted(data, key=lambda x: x.get(sort_field, 0), reverse=reverse)
            
            return sorted_data
            
        except Exception as e:
            logger.error(f"Error applying sort transformation: {str(e)}")
            return data
    
    async def _apply_format_transformation(self, data: List[Dict[str, Any]], parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply format transformation"""
        try:
            template = parameters.get("template", "default")
            
            if template == "executive_summary":
                # Format for executive summary
                formatted_data = []
                for item in data:
                    formatted_item = {
                        "summary": f"Model: {item.get('model_name', 'Unknown')}",
                        "key_metrics": {
                            "performance": item.get("performance_score", 0),
                            "accuracy": item.get("accuracy", 0),
                            "efficiency": item.get("cost_efficiency", 0)
                        },
                        "timestamp": item.get("timestamp", datetime.now().isoformat())
                    }
                    formatted_data.append(formatted_item)
                
                return formatted_data
            
            return data
            
        except Exception as e:
            logger.error(f"Error applying format transformation: {str(e)}")
            return data
    
    async def _apply_calculate_transformation(self, data: List[Dict[str, Any]], parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply calculate transformation"""
        try:
            calculations = parameters.get("calculations", [])
            
            for item in data:
                for calc in calculations:
                    field = calc.get("field", "")
                    expression = calc.get("expression", "")
                    output_field = calc.get("output_field", f"{field}_calculated")
                    
                    if field in item and expression:
                        try:
                            # Simple expression evaluation (in practice, use a proper expression evaluator)
                            if expression == "performance_score * 100":
                                item[output_field] = item[field] * 100
                            elif expression == "accuracy + precision":
                                item[output_field] = item.get("accuracy", 0) + item.get("precision", 0)
                            # Add more expressions as needed
                        except Exception as e:
                            logger.warning(f"Error evaluating expression {expression}: {str(e)}")
            
            return data
            
        except Exception as e:
            logger.error(f"Error applying calculate transformation: {str(e)}")
            return data
    
    async def _generate_filename(self, export_config: ExportConfig) -> str:
        """Generate filename for export"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_type = export_config.export_type.value
            export_format = export_config.export_format.value
            
            filename = f"{export_type}_{timestamp}.{export_format}"
            
            if export_config.compression:
                filename += ".gz"
            
            return filename
            
        except Exception as e:
            logger.error(f"Error generating filename: {str(e)}")
            return f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    async def _export_data(self, 
                          data: List[Dict[str, Any]], 
                          file_path: Path, 
                          export_config: ExportConfig) -> None:
        """Export data to file"""
        try:
            export_format = export_config.export_format
            
            if export_format == ExportFormat.JSON:
                await self._export_json(data, file_path, export_config)
            elif export_format == ExportFormat.CSV:
                await self._export_csv(data, file_path, export_config)
            elif export_format == ExportFormat.EXCEL:
                await self._export_excel(data, file_path, export_config)
            elif export_format == ExportFormat.PARQUET:
                await self._export_parquet(data, file_path, export_config)
            elif export_format == ExportFormat.XML:
                await self._export_xml(data, file_path, export_config)
            elif export_format == ExportFormat.YAML:
                await self._export_yaml(data, file_path, export_config)
            elif export_format == ExportFormat.HTML:
                await self._export_html(data, file_path, export_config)
            else:
                # Default to JSON
                await self._export_json(data, file_path, export_config)
            
            # Apply compression if requested
            if export_config.compression:
                await self._compress_file(file_path)
            
        except Exception as e:
            logger.error(f"Error exporting data: {str(e)}")
            raise e
    
    async def _export_json(self, data: List[Dict[str, Any]], file_path: Path, export_config: ExportConfig) -> None:
        """Export data as JSON"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error exporting JSON: {str(e)}")
            raise e
    
    async def _export_csv(self, data: List[Dict[str, Any]], file_path: Path, export_config: ExportConfig) -> None:
        """Export data as CSV"""
        try:
            if not data:
                return
            
            df = pd.DataFrame(data)
            df.to_csv(file_path, index=False, encoding='utf-8')
        except Exception as e:
            logger.error(f"Error exporting CSV: {str(e)}")
            raise e
    
    async def _export_excel(self, data: List[Dict[str, Any]], file_path: Path, export_config: ExportConfig) -> None:
        """Export data as Excel"""
        try:
            if not data:
                return
            
            df = pd.DataFrame(data)
            df.to_excel(file_path, index=False, engine='openpyxl')
        except Exception as e:
            logger.error(f"Error exporting Excel: {str(e)}")
            raise e
    
    async def _export_parquet(self, data: List[Dict[str, Any]], file_path: Path, export_config: ExportConfig) -> None:
        """Export data as Parquet"""
        try:
            if not data:
                return
            
            df = pd.DataFrame(data)
            df.to_parquet(file_path, index=False)
        except Exception as e:
            logger.error(f"Error exporting Parquet: {str(e)}")
            raise e
    
    async def _export_xml(self, data: List[Dict[str, Any]], file_path: Path, export_config: ExportConfig) -> None:
        """Export data as XML"""
        try:
            root = ET.Element("data")
            
            for item in data:
                item_elem = ET.SubElement(root, "item")
                for key, value in item.items():
                    field_elem = ET.SubElement(item_elem, key)
                    field_elem.text = str(value)
            
            tree = ET.ElementTree(root)
            tree.write(file_path, encoding='utf-8', xml_declaration=True)
        except Exception as e:
            logger.error(f"Error exporting XML: {str(e)}")
            raise e
    
    async def _export_yaml(self, data: List[Dict[str, Any]], file_path: Path, export_config: ExportConfig) -> None:
        """Export data as YAML"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
        except Exception as e:
            logger.error(f"Error exporting YAML: {str(e)}")
            raise e
    
    async def _export_html(self, data: List[Dict[str, Any]], file_path: Path, export_config: ExportConfig) -> None:
        """Export data as HTML"""
        try:
            if not data:
                return
            
            df = pd.DataFrame(data)
            html_content = df.to_html(index=False, classes='table table-striped', table_id='data-table')
            
            full_html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Export Data</title>
                <style>
                    table {{ border-collapse: collapse; width: 100%; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                </style>
            </head>
            <body>
                <h1>Export Data</h1>
                {html_content}
            </body>
            </html>
            """
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(full_html)
        except Exception as e:
            logger.error(f"Error exporting HTML: {str(e)}")
            raise e
    
    async def _compress_file(self, file_path: Path) -> None:
        """Compress file using gzip"""
        try:
            compressed_path = file_path.with_suffix(file_path.suffix + '.gz')
            
            with open(file_path, 'rb') as f_in:
                with gzip.open(compressed_path, 'wb') as f_out:
                    f_out.writelines(f_in)
            
            # Remove original file
            file_path.unlink()
            
            # Rename compressed file
            compressed_path.rename(file_path)
            
        except Exception as e:
            logger.error(f"Error compressing file: {str(e)}")
            raise e


# Global export system instance
_export_system: Optional[DataExportSystem] = None


def get_data_export_system(max_exports: int = 1000, output_directory: str = "./exports") -> DataExportSystem:
    """Get or create global data export system instance"""
    global _export_system
    if _export_system is None:
        _export_system = DataExportSystem(max_exports, output_directory)
    return _export_system


# Example usage
async def main():
    """Example usage of the data export system"""
    export_system = get_data_export_system()
    
    # Export performance data
    perf_export = await export_system.export_performance_data(
        model_names=["gpt-4", "claude-3", "gemini-pro"],
        time_range_days=30,
        export_format=ExportFormat.CSV
    )
    print(f"Exported performance data: {perf_export.file_path}")
    
    # Export analytics results
    analytics_export = await export_system.export_analytics_results(
        analytics_type="comprehensive",
        export_format=ExportFormat.JSON
    )
    print(f"Exported analytics results: {analytics_export.file_path}")
    
    # Export comparison results
    comparison_export = await export_system.export_comparison_results(
        model_pairs=[("gpt-4", "claude-3"), ("gpt-4", "gemini-pro")],
        export_format=ExportFormat.EXCEL
    )
    print(f"Exported comparison results: {comparison_export.file_path}")
    
    # Export benchmark results
    benchmark_export = await export_system.export_benchmark_results(
        benchmark_suite_id="ml_benchmark_suite",
        export_format=ExportFormat.PARQUET
    )
    print(f"Exported benchmark results: {benchmark_export.file_path}")
    
    # Export comprehensive report
    report_export = await export_system.export_comprehensive_report(
        report_type="executive",
        export_format=ExportFormat.HTML
    )
    print(f"Exported comprehensive report: {report_export.file_path}")
    
    # Create data transformation
    transformation = await export_system.create_data_transformation(
        name="Performance Aggregator",
        description="Aggregates performance data by model",
        transformation_type="aggregate",
        parameters={"group_by": "model_name", "aggregations": ["mean", "std", "min", "max"]},
        input_schema={"model_name": "string", "performance_score": "float"},
        output_schema={"model_name": "string", "performance_score_mean": "float", "performance_score_std": "float"}
    )
    print(f"Created transformation: {transformation.transformation_id}")
    
    # Get export analytics
    analytics = await export_system.get_export_analytics()
    print(f"Export analytics: {analytics.get('total_exports', 0)} exports, {analytics.get('success_rate', 0):.1%} success rate")


if __name__ == "__main__":
    asyncio.run(main())

























