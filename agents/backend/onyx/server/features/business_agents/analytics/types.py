"""
Analytics Types and Definitions
===============================

Type definitions for advanced analytics and business intelligence.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Union, Callable
from datetime import datetime, timedelta
import uuid

class DataType(Enum):
    """Data type enumeration."""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATE = "date"
    DATETIME = "datetime"
    TIMESTAMP = "timestamp"
    JSON = "json"
    ARRAY = "array"
    OBJECT = "object"

class AggregationType(Enum):
    """Aggregation type enumeration."""
    SUM = "sum"
    COUNT = "count"
    AVG = "avg"
    MIN = "min"
    MAX = "max"
    MEDIAN = "median"
    PERCENTILE = "percentile"
    DISTINCT_COUNT = "distinct_count"
    FIRST = "first"
    LAST = "last"

class TimeGranularity(Enum):
    """Time granularity enumeration."""
    SECOND = "second"
    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    QUARTER = "quarter"
    YEAR = "year"

class ChartType(Enum):
    """Chart type enumeration."""
    LINE = "line"
    BAR = "bar"
    PIE = "pie"
    SCATTER = "scatter"
    AREA = "area"
    HISTOGRAM = "histogram"
    HEATMAP = "heatmap"
    TREEMAP = "treemap"
    SANKEY = "sankey"
    GAUGE = "gauge"
    TABLE = "table"
    KPI = "kpi"

@dataclass
class DataSource:
    """Data source definition."""
    id: str
    name: str
    type: str  # database, api, file, stream
    connection_string: str
    schema: str
    table: str
    credentials: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

@dataclass
class DataModel:
    """Data model definition."""
    id: str
    name: str
    description: str
    source_id: str
    fields: List[Dict[str, Any]] = field(default_factory=list)
    relationships: List[Dict[str, Any]] = field(default_factory=list)
    filters: List[Dict[str, Any]] = field(default_factory=list)
    calculated_fields: List[Dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

@dataclass
class Dimension:
    """Dimension definition."""
    name: str
    data_type: DataType
    description: str
    source_field: str
    display_name: str
    format: Optional[str] = None
    categories: List[str] = field(default_factory=list)
    hierarchy: List[str] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Metric:
    """Metric definition."""
    name: str
    data_type: DataType
    description: str
    source_field: str
    aggregation: AggregationType
    display_name: str
    format: Optional[str] = None
    unit: Optional[str] = None
    calculation: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=dict)

@dataclass
class KPI:
    """KPI (Key Performance Indicator) definition."""
    id: str
    name: str
    description: str
    metric: Metric
    target_value: float
    current_value: float
    unit: str
    trend: str  # up, down, stable
    status: str  # good, warning, critical
    calculation_period: TimeGranularity
    last_updated: datetime = field(default_factory=datetime.now)
    history: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class AnalyticsQuery:
    """Analytics query definition."""
    id: str
    name: str
    description: str
    data_model_id: str
    dimensions: List[str] = field(default_factory=list)
    metrics: List[str] = field(default_factory=list)
    filters: List[Dict[str, Any]] = field(default_factory=list)
    time_range: Dict[str, Any] = field(default_factory=dict)
    granularity: TimeGranularity = TimeGranularity.DAY
    limit: Optional[int] = None
    offset: int = 0
    order_by: List[Dict[str, str]] = field(default_factory=list)
    group_by: List[str] = field(default_factory=list)
    having: List[Dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class AnalyticsResult:
    """Analytics query result."""
    query_id: str
    data: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    row_count: int = 0
    columns: List[Dict[str, Any]] = field(default_factory=list)
    aggregations: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class BusinessRule:
    """Business rule definition."""
    id: str
    name: str
    description: str
    condition: str
    action: str
    priority: int = 100
    enabled: bool = True
    category: str = "general"
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

@dataclass
class Widget:
    """Dashboard widget definition."""
    id: str
    name: str
    type: ChartType
    query_id: str
    title: str
    description: str
    position: Dict[str, int] = field(default_factory=dict)  # x, y, width, height
    config: Dict[str, Any] = field(default_factory=dict)
    refresh_interval: int = 300  # seconds
    auto_refresh: bool = True
    drill_down_enabled: bool = False
    export_enabled: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

@dataclass
class Dashboard:
    """Dashboard definition."""
    id: str
    name: str
    description: str
    category: str
    widgets: List[Widget] = field(default_factory=list)
    layout: Dict[str, Any] = field(default_factory=dict)
    filters: List[Dict[str, Any]] = field(default_factory=list)
    permissions: Dict[str, List[str]] = field(default_factory=dict)
    is_public: bool = False
    auto_refresh: bool = True
    refresh_interval: int = 300  # seconds
    created_by: str = "system"
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

@dataclass
class ReportTemplate:
    """Report template definition."""
    id: str
    name: str
    description: str
    type: str  # pdf, excel, csv, html
    template_content: str
    query_id: str
    parameters: List[Dict[str, Any]] = field(default_factory=list)
    styling: Dict[str, Any] = field(default_factory=dict)
    header: Dict[str, Any] = field(default_factory=dict)
    footer: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

@dataclass
class ReportSchedule:
    """Report schedule definition."""
    id: str
    template_id: str
    name: str
    description: str
    schedule_type: str  # daily, weekly, monthly, custom
    schedule_config: Dict[str, Any] = field(default_factory=dict)
    recipients: List[str] = field(default_factory=list)
    format: str = "pdf"
    enabled: bool = True
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class ETLJob:
    """ETL job definition."""
    id: str
    name: str
    description: str
    source_config: Dict[str, Any] = field(default_factory=dict)
    transformation_config: Dict[str, Any] = field(default_factory=dict)
    target_config: Dict[str, Any] = field(default_factory=dict)
    schedule: Optional[ReportSchedule] = None
    status: str = "idle"  # idle, running, completed, failed
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class DataMart:
    """Data mart definition."""
    id: str
    name: str
    description: str
    data_model_id: str
    dimensions: List[Dimension] = field(default_factory=list)
    metrics: List[Metric] = field(default_factory=list)
    aggregations: List[Dict[str, Any]] = field(default_factory=list)
    refresh_schedule: Optional[ReportSchedule] = None
    last_refresh: Optional[datetime] = None
    size_mb: float = 0.0
    row_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class MLModel:
    """ML model for analytics."""
    id: str
    name: str
    description: str
    type: str  # classification, regression, clustering, time_series
    algorithm: str
    training_data: str
    features: List[str] = field(default_factory=list)
    target: str = ""
    accuracy: float = 0.0
    status: str = "trained"  # training, trained, deployed, failed
    version: str = "1.0.0"
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

@dataclass
class Prediction:
    """ML prediction result."""
    model_id: str
    input_data: Dict[str, Any] = field(default_factory=dict)
    prediction: Any = None
    confidence: float = 0.0
    probability: Dict[str, float] = field(default_factory=dict)
    explanation: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class Anomaly:
    """Anomaly detection result."""
    id: str
    type: str  # outlier, trend_change, pattern_break
    severity: str  # low, medium, high, critical
    description: str
    detected_at: datetime = field(default_factory=datetime.now)
    data_point: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    resolved: bool = False

@dataclass
class Insight:
    """Business insight."""
    id: str
    title: str
    description: str
    type: str  # trend, anomaly, correlation, recommendation
    category: str
    priority: str  # low, medium, high, critical
    data: Dict[str, Any] = field(default_factory=dict)
    visualization: Dict[str, Any] = field(default_factory=dict)
    actions: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    viewed: bool = False
    acknowledged: bool = False

@dataclass
class AnalyticsMetrics:
    """Analytics performance metrics."""
    total_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0
    average_query_time: float = 0.0
    cache_hit_rate: float = 0.0
    data_freshness: float = 0.0
    user_engagement: float = 0.0
    dashboard_views: int = 0
    report_generations: int = 0
    ml_predictions: int = 0
    anomalies_detected: int = 0
    insights_generated: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
