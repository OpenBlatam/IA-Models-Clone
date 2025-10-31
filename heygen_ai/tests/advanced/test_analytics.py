"""
Advanced Test Analytics Framework for HeyGen AI Testing System.
Comprehensive test analytics including trend analysis, predictive insights,
and intelligent test optimization recommendations.
"""

import json
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import logging
import sqlite3
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

@dataclass
class TestExecution:
    """Represents a test execution record."""
    test_id: str
    test_name: str
    execution_time: float
    success: bool
    duration: float
    memory_usage: float
    cpu_usage: float
    timestamp: datetime
    environment: str = "default"
    test_category: str = "unknown"
    error_message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TestTrend:
    """Represents a test trend analysis."""
    test_name: str
    trend_type: str  # success_rate, duration, frequency
    direction: str  # increasing, decreasing, stable
    slope: float
    r_squared: float
    confidence: float
    period_days: int
    data_points: int

@dataclass
class TestInsight:
    """Represents a test insight."""
    insight_id: str
    insight_type: str  # performance, reliability, efficiency, quality
    title: str
    description: str
    severity: str  # low, medium, high, critical
    confidence: float
    recommendations: List[str] = field(default_factory=list)
    affected_tests: List[str] = field(default_factory=list)
    generated_at: datetime = field(default_factory=datetime.now)

@dataclass
class TestAnalyticsReport:
    """Represents a comprehensive test analytics report."""
    report_id: str
    period_start: datetime
    period_end: datetime
    total_tests: int
    total_executions: int
    success_rate: float
    avg_duration: float
    trends: List[TestTrend] = field(default_factory=list)
    insights: List[TestInsight] = field(default_factory=list)
    predictions: Dict[str, Any] = field(default_factory=dict)
    generated_at: datetime = field(default_factory=datetime.now)

class TestDataCollector:
    """Collects and manages test execution data."""
    
    def __init__(self, db_path: str = "test_analytics.db"):
        self.db_path = Path(db_path)
        self.init_database()
    
    def init_database(self):
        """Initialize the analytics database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create test executions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS test_executions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                test_id TEXT NOT NULL,
                test_name TEXT NOT NULL,
                execution_time REAL NOT NULL,
                success BOOLEAN NOT NULL,
                duration REAL NOT NULL,
                memory_usage REAL NOT NULL,
                cpu_usage REAL NOT NULL,
                timestamp DATETIME NOT NULL,
                environment TEXT DEFAULT 'default',
                test_category TEXT DEFAULT 'unknown',
                error_message TEXT DEFAULT '',
                metadata TEXT
            )
        """)
        
        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_test_name ON test_executions(test_name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON test_executions(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_success ON test_executions(success)")
        
        conn.commit()
        conn.close()
    
    def add_test_execution(self, execution: TestExecution):
        """Add a test execution record."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO test_executions 
            (test_id, test_name, execution_time, success, duration, memory_usage, 
             cpu_usage, timestamp, environment, test_category, error_message, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            execution.test_id,
            execution.test_name,
            execution.execution_time,
            execution.success,
            execution.duration,
            execution.memory_usage,
            execution.cpu_usage,
            execution.timestamp.isoformat(),
            execution.environment,
            execution.test_category,
            execution.error_message,
            json.dumps(execution.metadata)
        ))
        
        conn.commit()
        conn.close()
    
    def get_test_executions(self, start_time: datetime, end_time: datetime,
                          test_name: Optional[str] = None) -> List[TestExecution]:
        """Get test executions for a time period."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = """
            SELECT test_id, test_name, execution_time, success, duration, memory_usage,
                   cpu_usage, timestamp, environment, test_category, error_message, metadata
            FROM test_executions
            WHERE timestamp BETWEEN ? AND ?
        """
        params = [start_time.isoformat(), end_time.isoformat()]
        
        if test_name:
            query += " AND test_name = ?"
            params.append(test_name)
        
        query += " ORDER BY timestamp"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        executions = []
        for row in rows:
            execution = TestExecution(
                test_id=row[0],
                test_name=row[1],
                execution_time=row[2],
                success=bool(row[3]),
                duration=row[4],
                memory_usage=row[5],
                cpu_usage=row[6],
                timestamp=datetime.fromisoformat(row[7]),
                environment=row[8],
                test_category=row[9],
                error_message=row[10],
                metadata=json.loads(row[11]) if row[11] else {}
            )
            executions.append(execution)
        
        conn.close()
        return executions
    
    def get_test_summary(self, test_name: str, days: int = 30) -> Dict[str, Any]:
        """Get summary statistics for a specific test."""
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        executions = self.get_test_executions(start_time, end_time, test_name)
        
        if not executions:
            return {}
        
        durations = [e.duration for e in executions]
        successes = [e.success for e in executions]
        
        return {
            "test_name": test_name,
            "total_executions": len(executions),
            "success_rate": sum(successes) / len(successes),
            "avg_duration": np.mean(durations),
            "min_duration": np.min(durations),
            "max_duration": np.max(durations),
            "std_duration": np.std(durations),
            "avg_memory_usage": np.mean([e.memory_usage for e in executions]),
            "avg_cpu_usage": np.mean([e.cpu_usage for e in executions]),
            "first_execution": min(e.timestamp for e in executions),
            "last_execution": max(e.timestamp for e in executions)
        }

class TrendAnalyzer:
    """Analyzes trends in test execution data."""
    
    def __init__(self):
        self.min_data_points = 5
    
    def analyze_trends(self, executions: List[TestExecution]) -> List[TestTrend]:
        """Analyze trends in test execution data."""
        trends = []
        
        # Group executions by test name
        executions_by_test = defaultdict(list)
        for execution in executions:
            executions_by_test[execution.test_name].append(execution)
        
        for test_name, test_executions in executions_by_test.items():
            if len(test_executions) < self.min_data_points:
                continue
            
            # Sort by timestamp
            test_executions.sort(key=lambda x: x.timestamp)
            
            # Analyze different trend types
            trends.extend(self._analyze_success_rate_trend(test_name, test_executions))
            trends.extend(self._analyze_duration_trend(test_name, test_executions))
            trends.extend(self._analyze_frequency_trend(test_name, test_executions))
        
        return trends
    
    def _analyze_success_rate_trend(self, test_name: str, executions: List[TestExecution]) -> List[TestTrend]:
        """Analyze success rate trend for a test."""
        # Group executions by day
        daily_data = defaultdict(list)
        for execution in executions:
            day = execution.timestamp.date()
            daily_data[day].append(execution.success)
        
        # Calculate daily success rates
        days = sorted(daily_data.keys())
        success_rates = [sum(daily_data[day]) / len(daily_data[day]) for day in days]
        
        if len(success_rates) < self.min_data_points:
            return []
        
        # Calculate trend
        x = np.arange(len(success_rates))
        slope, intercept, r_value, p_value, std_err = np.polyfit(x, success_rates, 1, full=True)
        
        # Determine direction
        if abs(slope) < 0.01:
            direction = "stable"
        elif slope > 0:
            direction = "increasing"
        else:
            direction = "decreasing"
        
        trend = TestTrend(
            test_name=test_name,
            trend_type="success_rate",
            direction=direction,
            slope=float(slope),
            r_squared=float(r_value**2) if len(r_value) > 0 else 0.0,
            confidence=1.0 - float(p_value[0]) if len(p_value) > 0 else 0.0,
            period_days=len(days),
            data_points=len(success_rates)
        )
        
        return [trend]
    
    def _analyze_duration_trend(self, test_name: str, executions: List[TestExecution]) -> List[TestTrend]:
        """Analyze duration trend for a test."""
        # Group executions by day
        daily_data = defaultdict(list)
        for execution in executions:
            day = execution.timestamp.date()
            daily_data[day].append(execution.duration)
        
        # Calculate daily average durations
        days = sorted(daily_data.keys())
        avg_durations = [np.mean(daily_data[day]) for day in days]
        
        if len(avg_durations) < self.min_data_points:
            return []
        
        # Calculate trend
        x = np.arange(len(avg_durations))
        slope, intercept, r_value, p_value, std_err = np.polyfit(x, avg_durations, 1, full=True)
        
        # Determine direction
        if abs(slope) < 0.01:
            direction = "stable"
        elif slope > 0:
            direction = "increasing"
        else:
            direction = "decreasing"
        
        trend = TestTrend(
            test_name=test_name,
            trend_type="duration",
            direction=direction,
            slope=float(slope),
            r_squared=float(r_value**2) if len(r_value) > 0 else 0.0,
            confidence=1.0 - float(p_value[0]) if len(p_value) > 0 else 0.0,
            period_days=len(days),
            data_points=len(avg_durations)
        )
        
        return [trend]
    
    def _analyze_frequency_trend(self, test_name: str, executions: List[TestExecution]) -> List[TestTrend]:
        """Analyze execution frequency trend for a test."""
        # Group executions by day
        daily_counts = defaultdict(int)
        for execution in executions:
            day = execution.timestamp.date()
            daily_counts[day] += 1
        
        # Get daily counts
        days = sorted(daily_counts.keys())
        counts = [daily_counts[day] for day in days]
        
        if len(counts) < self.min_data_points:
            return []
        
        # Calculate trend
        x = np.arange(len(counts))
        slope, intercept, r_value, p_value, std_err = np.polyfit(x, counts, 1, full=True)
        
        # Determine direction
        if abs(slope) < 0.01:
            direction = "stable"
        elif slope > 0:
            direction = "increasing"
        else:
            direction = "decreasing"
        
        trend = TestTrend(
            test_name=test_name,
            trend_type="frequency",
            direction=direction,
            slope=float(slope),
            r_squared=float(r_value**2) if len(r_value) > 0 else 0.0,
            confidence=1.0 - float(p_value[0]) if len(p_value) > 0 else 0.0,
            period_days=len(days),
            data_points=len(counts)
        )
        
        return [trend]

class InsightGenerator:
    """Generates intelligent insights from test data."""
    
    def __init__(self):
        self.insight_rules = [
            self._check_performance_degradation,
            self._check_reliability_issues,
            self._check_efficiency_concerns,
            self._check_quality_issues,
            self._check_resource_usage,
            self._check_frequency_patterns
        ]
    
    def generate_insights(self, executions: List[TestExecution], 
                         trends: List[TestTrend]) -> List[TestInsight]:
        """Generate insights from test data and trends."""
        insights = []
        
        # Group executions by test name
        executions_by_test = defaultdict(list)
        for execution in executions:
            executions_by_test[execution.test_name].append(execution)
        
        # Generate insights for each test
        for test_name, test_executions in executions_by_test.items():
            test_trends = [t for t in trends if t.test_name == test_name]
            
            for rule in self.insight_rules:
                insight = rule(test_name, test_executions, test_trends)
                if insight:
                    insights.append(insight)
        
        # Generate cross-test insights
        insights.extend(self._generate_cross_test_insights(executions, trends))
        
        return insights
    
    def _check_performance_degradation(self, test_name: str, executions: List[TestExecution],
                                     trends: List[TestTrend]) -> Optional[TestInsight]:
        """Check for performance degradation."""
        duration_trend = next((t for t in trends if t.trend_type == "duration"), None)
        
        if duration_trend and duration_trend.direction == "increasing" and duration_trend.confidence > 0.7:
            # Calculate degradation percentage
            recent_durations = [e.duration for e in executions[-10:]]
            older_durations = [e.duration for e in executions[:-10]] if len(executions) > 10 else []
            
            if older_durations:
                recent_avg = np.mean(recent_durations)
                older_avg = np.mean(older_durations)
                degradation = ((recent_avg - older_avg) / older_avg) * 100
                
                if degradation > 20:  # 20% degradation threshold
                    return TestInsight(
                        insight_id=f"perf_degradation_{test_name}",
                        insight_type="performance",
                        title=f"Performance Degradation Detected",
                        description=f"Test '{test_name}' has degraded by {degradation:.1f}% over time",
                        severity="high" if degradation > 50 else "medium",
                        confidence=duration_trend.confidence,
                        recommendations=[
                            "Investigate recent changes that might affect performance",
                            "Consider optimizing the test or underlying code",
                            "Monitor resource usage during test execution"
                        ],
                        affected_tests=[test_name]
                    )
        
        return None
    
    def _check_reliability_issues(self, test_name: str, executions: List[TestExecution],
                                trends: List[TestTrend]) -> Optional[TestInsight]:
        """Check for reliability issues."""
        success_trend = next((t for t in trends if t.trend_type == "success_rate"), None)
        
        if success_trend and success_trend.direction == "decreasing" and success_trend.confidence > 0.7:
            # Calculate current success rate
            recent_executions = executions[-20:] if len(executions) >= 20 else executions
            success_rate = sum(e.success for e in recent_executions) / len(recent_executions)
            
            if success_rate < 0.8:  # 80% success rate threshold
                return TestInsight(
                    insight_id=f"reliability_issue_{test_name}",
                    insight_type="reliability",
                    title=f"Reliability Issues Detected",
                    description=f"Test '{test_name}' has a declining success rate ({success_rate:.1%})",
                    severity="critical" if success_rate < 0.5 else "high",
                    confidence=success_trend.confidence,
                    recommendations=[
                        "Investigate recent failures and their root causes",
                        "Review test environment stability",
                        "Consider adding more robust error handling"
                    ],
                    affected_tests=[test_name]
                )
        
        return None
    
    def _check_efficiency_concerns(self, test_name: str, executions: List[TestExecution],
                                 trends: List[TestTrend]) -> Optional[TestInsight]:
        """Check for efficiency concerns."""
        if len(executions) < 10:
            return None
        
        # Check for high resource usage
        avg_memory = np.mean([e.memory_usage for e in executions[-10:]])
        avg_cpu = np.mean([e.cpu_usage for e in executions[-10:]])
        
        if avg_memory > 500:  # 500MB threshold
            return TestInsight(
                insight_id=f"memory_efficiency_{test_name}",
                insight_type="efficiency",
                title=f"High Memory Usage Detected",
                description=f"Test '{test_name}' uses {avg_memory:.1f}MB on average",
                severity="medium",
                confidence=0.8,
                recommendations=[
                    "Profile memory usage during test execution",
                    "Consider optimizing data structures or algorithms",
                    "Check for memory leaks"
                ],
                affected_tests=[test_name]
            )
        
        if avg_cpu > 80:  # 80% CPU threshold
            return TestInsight(
                insight_id=f"cpu_efficiency_{test_name}",
                insight_type="efficiency",
                title=f"High CPU Usage Detected",
                description=f"Test '{test_name}' uses {avg_cpu:.1f}% CPU on average",
                severity="medium",
                confidence=0.8,
                recommendations=[
                    "Profile CPU usage during test execution",
                    "Consider optimizing algorithms or parallelization",
                    "Check for infinite loops or inefficient operations"
                ],
                affected_tests=[test_name]
            )
        
        return None
    
    def _check_quality_issues(self, test_name: str, executions: List[TestExecution],
                            trends: List[TestTrend]) -> Optional[TestInsight]:
        """Check for quality issues."""
        if len(executions) < 5:
            return None
        
        # Check for inconsistent execution times
        durations = [e.duration for e in executions[-20:]]
        if len(durations) > 5:
            cv = np.std(durations) / np.mean(durations)  # Coefficient of variation
            
            if cv > 0.5:  # 50% variation threshold
                return TestInsight(
                    insight_id=f"quality_inconsistency_{test_name}",
                    insight_type="quality",
                    title=f"Inconsistent Execution Times",
                    description=f"Test '{test_name}' has high execution time variability (CV: {cv:.2f})",
                    severity="medium",
                    confidence=0.7,
                    recommendations=[
                        "Investigate factors causing execution time variability",
                        "Consider adding more deterministic test data",
                        "Review test environment consistency"
                    ],
                    affected_tests=[test_name]
                )
        
        return None
    
    def _check_resource_usage(self, test_name: str, executions: List[TestExecution],
                            trends: List[TestTrend]) -> Optional[TestInsight]:
        """Check for resource usage patterns."""
        if len(executions) < 10:
            return None
        
        # Check for increasing resource usage
        recent_executions = executions[-10:]
        older_executions = executions[:-10] if len(executions) > 10 else []
        
        if older_executions:
            recent_memory = np.mean([e.memory_usage for e in recent_executions])
            older_memory = np.mean([e.memory_usage for e in older_executions])
            
            if recent_memory > older_memory * 1.5:  # 50% increase
                return TestInsight(
                    insight_id=f"resource_increase_{test_name}",
                    insight_type="efficiency",
                    title=f"Increasing Resource Usage",
                    description=f"Test '{test_name}' memory usage has increased significantly",
                    severity="medium",
                    confidence=0.8,
                    recommendations=[
                        "Investigate recent changes affecting memory usage",
                        "Consider memory profiling and optimization",
                        "Monitor for potential memory leaks"
                    ],
                    affected_tests=[test_name]
                )
        
        return None
    
    def _check_frequency_patterns(self, test_name: str, executions: List[TestExecution],
                                trends: List[TestTrend]) -> Optional[TestInsight]:
        """Check for frequency patterns."""
        frequency_trend = next((t for t in trends if t.trend_type == "frequency"), None)
        
        if frequency_trend and frequency_trend.direction == "decreasing" and frequency_trend.confidence > 0.7:
            return TestInsight(
                insight_id=f"frequency_decrease_{test_name}",
                insight_type="quality",
                title=f"Decreasing Test Frequency",
                description=f"Test '{test_name}' is being executed less frequently",
                severity="low",
                confidence=frequency_trend.confidence,
                recommendations=[
                    "Review test scheduling and execution policies",
                    "Ensure test is still relevant and necessary",
                    "Consider increasing test frequency if important"
                ],
                affected_tests=[test_name]
            )
        
        return None
    
    def _generate_cross_test_insights(self, executions: List[TestExecution],
                                    trends: List[TestTrend]) -> List[TestInsight]:
        """Generate insights across multiple tests."""
        insights = []
        
        # Group executions by test category
        executions_by_category = defaultdict(list)
        for execution in executions:
            executions_by_category[execution.test_category].append(execution)
        
        # Analyze category-level patterns
        for category, category_executions in executions_by_category.items():
            if len(category_executions) < 20:
                continue
            
            # Calculate category success rate
            success_rate = sum(e.success for e in category_executions) / len(category_executions)
            
            if success_rate < 0.7:  # 70% threshold for category
                insights.append(TestInsight(
                    insight_id=f"category_reliability_{category}",
                    insight_type="quality",
                    title=f"Category Reliability Issues",
                    description=f"Test category '{category}' has low success rate ({success_rate:.1%})",
                    severity="high",
                    confidence=0.8,
                    recommendations=[
                        f"Investigate common issues in {category} tests",
                        "Review test environment for category-specific problems",
                        "Consider refactoring or improving test design"
                    ],
                    affected_tests=list(set(e.test_name for e in category_executions))
                ))
        
        return insights

class PredictiveAnalyzer:
    """Provides predictive analytics for test execution."""
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
    
    def train_models(self, executions: List[TestExecution]):
        """Train predictive models on test execution data."""
        if len(executions) < 50:
            return
        
        # Prepare data
        df = self._prepare_dataframe(executions)
        
        # Train duration prediction model
        self._train_duration_model(df)
        
        # Train success prediction model
        self._train_success_model(df)
    
    def _prepare_dataframe(self, executions: List[TestExecution]) -> pd.DataFrame:
        """Prepare DataFrame from test executions."""
        data = []
        for execution in executions:
            data.append({
                'test_name': execution.test_name,
                'execution_time': execution.execution_time,
                'success': execution.success,
                'duration': execution.duration,
                'memory_usage': execution.memory_usage,
                'cpu_usage': execution.cpu_usage,
                'timestamp': execution.timestamp,
                'environment': execution.environment,
                'test_category': execution.test_category,
                'hour': execution.timestamp.hour,
                'day_of_week': execution.timestamp.weekday(),
                'month': execution.timestamp.month
            })
        
        df = pd.DataFrame(data)
        
        # Add historical features
        df = df.sort_values('timestamp')
        df['prev_duration'] = df.groupby('test_name')['duration'].shift(1)
        df['prev_success'] = df.groupby('test_name')['success'].shift(1)
        df['duration_trend'] = df.groupby('test_name')['duration'].rolling(5).mean().values
        df['success_rate'] = df.groupby('test_name')['success'].rolling(10).mean().values
        
        # Fill NaN values
        df = df.fillna(method='bfill').fillna(method='ffill')
        
        return df
    
    def _train_duration_model(self, df: pd.DataFrame):
        """Train duration prediction model."""
        features = ['memory_usage', 'cpu_usage', 'hour', 'day_of_week', 'month', 
                   'prev_duration', 'duration_trend']
        
        X = df[features].values
        y = df['duration'].values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_scaled, y)
        
        self.models['duration'] = {
            'model': model,
            'features': features,
            'scaler': self.scaler
        }
    
    def _train_success_model(self, df: pd.DataFrame):
        """Train success prediction model."""
        features = ['duration', 'memory_usage', 'cpu_usage', 'hour', 'day_of_week', 
                   'month', 'prev_success', 'success_rate']
        
        X = df[features].values
        y = df['success'].values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_scaled, y)
        
        self.models['success'] = {
            'model': model,
            'features': features,
            'scaler': self.scaler
        }
    
    def predict_duration(self, test_name: str, current_metrics: Dict[str, Any]) -> float:
        """Predict test duration."""
        if 'duration' not in self.models:
            return 0.0
        
        model_data = self.models['duration']
        features = model_data['features']
        
        # Prepare input data
        input_data = []
        for feature in features:
            if feature in current_metrics:
                input_data.append(current_metrics[feature])
            else:
                input_data.append(0.0)  # Default value
        
        # Scale and predict
        X = np.array(input_data).reshape(1, -1)
        X_scaled = model_data['scaler'].transform(X)
        prediction = model_data['model'].predict(X_scaled)[0]
        
        return max(0.0, prediction)
    
    def predict_success_probability(self, test_name: str, current_metrics: Dict[str, Any]) -> float:
        """Predict test success probability."""
        if 'success' not in self.models:
            return 0.5
        
        model_data = self.models['success']
        features = model_data['features']
        
        # Prepare input data
        input_data = []
        for feature in features:
            if feature in current_metrics:
                input_data.append(current_metrics[feature])
            else:
                input_data.append(0.0)  # Default value
        
        # Scale and predict
        X = np.array(input_data).reshape(1, -1)
        X_scaled = model_data['scaler'].transform(X)
        prediction = model_data['model'].predict(X_scaled)[0]
        
        return max(0.0, min(1.0, prediction))

class TestAnalyticsFramework:
    """Main test analytics framework."""
    
    def __init__(self, db_path: str = "test_analytics.db"):
        self.collector = TestDataCollector(db_path)
        self.trend_analyzer = TrendAnalyzer()
        self.insight_generator = InsightGenerator()
        self.predictive_analyzer = PredictiveAnalyzer()
    
    def add_test_execution(self, execution: TestExecution):
        """Add a test execution record."""
        self.collector.add_test_execution(execution)
    
    def generate_analytics_report(self, days: int = 30) -> TestAnalyticsReport:
        """Generate comprehensive analytics report."""
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        # Get test executions
        executions = self.collector.get_test_executions(start_time, end_time)
        
        if not executions:
            return TestAnalyticsReport(
                report_id=f"report_{int(time.time())}",
                period_start=start_time,
                period_end=end_time,
                total_tests=0,
                total_executions=0,
                success_rate=0.0,
                avg_duration=0.0
            )
        
        # Calculate basic statistics
        total_tests = len(set(e.test_name for e in executions))
        total_executions = len(executions)
        success_rate = sum(e.success for e in executions) / total_executions
        avg_duration = np.mean([e.duration for e in executions])
        
        # Analyze trends
        trends = self.trend_analyzer.analyze_trends(executions)
        
        # Generate insights
        insights = self.insight_generator.generate_insights(executions, trends)
        
        # Train predictive models
        self.predictive_analyzer.train_models(executions)
        
        # Generate predictions
        predictions = self._generate_predictions(executions)
        
        return TestAnalyticsReport(
            report_id=f"report_{int(time.time())}",
            period_start=start_time,
            period_end=end_time,
            total_tests=total_tests,
            total_executions=total_executions,
            success_rate=success_rate,
            avg_duration=avg_duration,
            trends=trends,
            insights=insights,
            predictions=predictions
        )
    
    def _generate_predictions(self, executions: List[TestExecution]) -> Dict[str, Any]:
        """Generate predictions for future test executions."""
        predictions = {}
        
        # Get unique test names
        test_names = list(set(e.test_name for e in executions))
        
        for test_name in test_names:
            # Get recent executions for this test
            test_executions = [e for e in executions if e.test_name == test_name]
            if len(test_executions) < 5:
                continue
            
            # Get latest metrics
            latest_execution = max(test_executions, key=lambda x: x.timestamp)
            current_metrics = {
                'memory_usage': latest_execution.memory_usage,
                'cpu_usage': latest_execution.cpu_usage,
                'hour': datetime.now().hour,
                'day_of_week': datetime.now().weekday(),
                'month': datetime.now().month,
                'prev_duration': latest_execution.duration,
                'prev_success': float(latest_execution.success),
                'duration_trend': np.mean([e.duration for e in test_executions[-5:]]),
                'success_rate': sum(e.success for e in test_executions[-10:]) / min(10, len(test_executions))
            }
            
            # Make predictions
            predicted_duration = self.predictive_analyzer.predict_duration(test_name, current_metrics)
            predicted_success = self.predictive_analyzer.predict_success_probability(test_name, current_metrics)
            
            predictions[test_name] = {
                'predicted_duration': predicted_duration,
                'predicted_success_probability': predicted_success,
                'confidence': 0.8  # Placeholder
            }
        
        return predictions
    
    def get_test_insights(self, test_name: str, days: int = 30) -> List[TestInsight]:
        """Get insights for a specific test."""
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        executions = self.collector.get_test_executions(start_time, end_time, test_name)
        trends = self.trend_analyzer.analyze_trends(executions)
        insights = self.insight_generator.generate_insights(executions, trends)
        
        return insights
    
    def generate_visualizations(self, report: TestAnalyticsReport, output_dir: str = "analytics_plots"):
        """Generate visualizations for the analytics report."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        
        # Create comprehensive dashboard
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'Test Success Rate Over Time',
                'Test Duration Distribution',
                'Resource Usage Trends',
                'Test Execution Frequency',
                'Performance Trends by Test',
                'Insights Summary'
            ],
            specs=[[{"type": "scatter"}, {"type": "histogram"}],
                   [{"type": "scatter"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        # This is a simplified version - in practice, you'd generate actual plots
        # based on the report data
        
        # Save the plot
        plot_path = output_path / f"{report.report_id}_dashboard.html"
        fig.write_html(str(plot_path))
        
        return str(plot_path)

# Example usage and demo
def demo_test_analytics():
    """Demonstrate test analytics capabilities."""
    print("📊 Test Analytics Framework Demo")
    print("=" * 50)
    
    # Create analytics framework
    analytics = TestAnalyticsFramework()
    
    # Generate sample test executions
    print("📈 Generating sample test data...")
    test_names = ["test_login", "test_registration", "test_payment", "test_search", "test_api"]
    
    for i in range(100):
        test_name = test_names[i % len(test_names)]
        
        # Simulate some trends
        base_duration = 1.0 + (i % 10) * 0.1
        if i > 50:  # Performance degradation
            base_duration *= 1.2
        
        success = np.random.random() > 0.1  # 90% success rate
        if i > 80:  # Reliability issues
            success = np.random.random() > 0.3  # 70% success rate
        
        execution = TestExecution(
            test_id=f"test_{i}",
            test_name=test_name,
            execution_time=time.time(),
            success=success,
            duration=base_duration + np.random.normal(0, 0.1),
            memory_usage=50 + np.random.normal(0, 10),
            cpu_usage=30 + np.random.normal(0, 5),
            timestamp=datetime.now() - timedelta(days=30-i//3),
            environment="test",
            test_category="integration" if "api" in test_name else "unit"
        )
        
        analytics.add_test_execution(execution)
    
    # Generate analytics report
    print("\n📊 Generating analytics report...")
    report = analytics.generate_analytics_report(days=30)
    
    print(f"Report ID: {report.report_id}")
    print(f"Total Tests: {report.total_tests}")
    print(f"Total Executions: {report.total_executions}")
    print(f"Success Rate: {report.success_rate:.1%}")
    print(f"Average Duration: {report.avg_duration:.2f}s")
    print(f"Trends Found: {len(report.trends)}")
    print(f"Insights Generated: {len(report.insights)}")
    
    # Print insights
    if report.insights:
        print("\n💡 Key Insights:")
        for insight in report.insights[:5]:  # Show first 5
            print(f"  {insight.severity.upper()}: {insight.title}")
            print(f"    {insight.description}")
    
    # Print trends
    if report.trends:
        print("\n📈 Key Trends:")
        for trend in report.trends[:5]:  # Show first 5
            print(f"  {trend.test_name}: {trend.trend_type} is {trend.direction}")
    
    # Generate visualizations
    print("\n📊 Generating visualizations...")
    viz_path = analytics.generate_visualizations(report)
    print(f"Visualizations saved to: {viz_path}")

if __name__ == "__main__":
    # Run demo
    demo_test_analytics()
