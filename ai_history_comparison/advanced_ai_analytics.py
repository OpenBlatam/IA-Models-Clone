"""
Advanced AI Analytics and Intelligence System
============================================

Advanced analytics system for AI model intelligence, performance optimization,
and predictive insights with machine learning capabilities.
"""

import asyncio
import logging
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
import hashlib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score
import joblib
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class AnalyticsType(str, Enum):
    """Types of analytics"""
    PREDICTIVE = "predictive"
    DESCRIPTIVE = "descriptive"
    DIAGNOSTIC = "diagnostic"
    PRESCRIPTIVE = "prescriptive"
    REAL_TIME = "real_time"
    BATCH = "batch"
    STREAMING = "streaming"
    CUSTOM = "custom"


class ModelIntelligence(str, Enum):
    """Model intelligence types"""
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    COST_OPTIMIZATION = "cost_optimization"
    QUALITY_ENHANCEMENT = "quality_enhancement"
    EFFICIENCY_IMPROVEMENT = "efficiency_improvement"
    ADAPTIVE_LEARNING = "adaptive_learning"
    PREDICTIVE_MAINTENANCE = "predictive_maintenance"
    ANOMALY_DETECTION = "anomaly_detection"
    TREND_ANALYSIS = "trend_analysis"


class IntelligenceLevel(str, Enum):
    """Intelligence levels"""
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"
    AUTONOMOUS = "autonomous"
    SUPERINTELLIGENT = "superintelligent"


@dataclass
class AIInsight:
    """AI-generated insight"""
    insight_id: str
    insight_type: str
    title: str
    description: str
    confidence: float
    impact_score: float
    actionable: bool
    recommendations: List[str]
    supporting_data: Dict[str, Any]
    generated_at: datetime
    expires_at: Optional[datetime] = None


@dataclass
class PredictiveModel:
    """Predictive model for AI analytics"""
    model_id: str
    model_name: str
    model_type: str
    algorithm: str
    accuracy: float
    features: List[str]
    target: str
    trained_at: datetime
    last_updated: datetime
    performance_metrics: Dict[str, float]
    model_data: Any = None


@dataclass
class IntelligenceReport:
    """Comprehensive intelligence report"""
    report_id: str
    report_type: AnalyticsType
    intelligence_level: IntelligenceLevel
    generated_at: datetime
    insights: List[AIInsight]
    predictions: List[Dict[str, Any]]
    recommendations: List[str]
    risk_assessments: List[Dict[str, Any]]
    opportunities: List[Dict[str, Any]]
    performance_summary: Dict[str, Any]
    confidence_score: float


class AdvancedAIAnalytics:
    """Advanced AI analytics system with machine learning capabilities"""
    
    def __init__(self, max_history_days: int = 365):
        self.max_history_days = max_history_days
        self.performance_data: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.predictive_models: Dict[str, PredictiveModel] = {}
        self.insights_history: List[AIInsight] = []
        self.intelligence_reports: List[IntelligenceReport] = []
        
        # Machine learning components
        self.scalers = {}
        self.feature_importance = {}
        self.model_performance = {}
        
        # Analytics configuration
        self.analytics_config = {
            "prediction_horizon_days": 30,
            "confidence_threshold": 0.7,
            "anomaly_threshold": 2.0,
            "trend_analysis_window": 14,
            "clustering_enabled": True,
            "deep_learning_enabled": False
        }
        
        # Cache for computed results
        self.analytics_cache = {}
        self.cache_ttl = 1800  # 30 minutes
    
    async def record_performance_data(self, 
                                    model_name: str,
                                    performance_metrics: Dict[str, float],
                                    context: Dict[str, Any] = None,
                                    metadata: Dict[str, Any] = None) -> bool:
        """Record comprehensive performance data"""
        try:
            data_point = {
                "timestamp": datetime.now(),
                "model_name": model_name,
                "metrics": performance_metrics,
                "context": context or {},
                "metadata": metadata or {}
            }
            
            self.performance_data[model_name].append(data_point)
            
            # Clean old data
            await self._cleanup_old_data()
            
            # Invalidate cache
            self._invalidate_cache()
            
            logger.debug(f"Recorded performance data for {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error recording performance data: {str(e)}")
            return False
    
    async def generate_predictive_insights(self, 
                                         model_name: str,
                                         prediction_horizon: int = 30) -> List[AIInsight]:
        """Generate predictive insights using machine learning"""
        try:
            insights = []
            
            # Get performance data
            data = self.performance_data.get(model_name, [])
            if len(data) < 50:  # Need sufficient data
                return insights
            
            # Convert to DataFrame
            df = await self._prepare_dataframe(data)
            
            # Generate predictions for each metric
            for metric in df.columns:
                if metric in ['timestamp', 'model_name']:
                    continue
                
                # Train predictive model
                model = await self._train_predictive_model(df, metric)
                if model:
                    # Generate predictions
                    predictions = await self._generate_predictions(model, df, metric, prediction_horizon)
                    
                    # Create insight
                    insight = await self._create_predictive_insight(
                        model_name, metric, predictions, model.accuracy
                    )
                    insights.append(insight)
            
            # Store insights
            self.insights_history.extend(insights)
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating predictive insights: {str(e)}")
            return []
    
    async def analyze_performance_patterns(self, 
                                         model_name: str,
                                         analysis_type: AnalyticsType = AnalyticsType.DESCRIPTIVE) -> Dict[str, Any]:
        """Analyze performance patterns using advanced analytics"""
        try:
            data = self.performance_data.get(model_name, [])
            if not data:
                return {"error": "No data available"}
            
            df = await self._prepare_dataframe(data)
            
            analysis_results = {
                "model_name": model_name,
                "analysis_type": analysis_type.value,
                "analysis_timestamp": datetime.now().isoformat(),
                "data_points": len(df),
                "time_range": {
                    "start": df['timestamp'].min().isoformat(),
                    "end": df['timestamp'].max().isoformat()
                }
            }
            
            if analysis_type == AnalyticsType.DESCRIPTIVE:
                analysis_results.update(await self._descriptive_analysis(df))
            elif analysis_type == AnalyticsType.DIAGNOSTIC:
                analysis_results.update(await self._diagnostic_analysis(df))
            elif analysis_type == AnalyticsType.PRESCRIPTIVE:
                analysis_results.update(await self._prescriptive_analysis(df))
            elif analysis_type == AnalyticsType.PREDICTIVE:
                analysis_results.update(await self._predictive_analysis(df))
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error analyzing performance patterns: {str(e)}")
            return {"error": str(e)}
    
    async def detect_anomalies(self, 
                             model_name: str,
                             sensitivity: float = 0.1) -> List[Dict[str, Any]]:
        """Detect anomalies in performance data using machine learning"""
        try:
            data = self.performance_data.get(model_name, [])
            if len(data) < 20:
                return []
            
            df = await self._prepare_dataframe(data)
            anomalies = []
            
            # Use multiple anomaly detection methods
            for metric in df.columns:
                if metric in ['timestamp', 'model_name']:
                    continue
                
                # Statistical anomaly detection
                stat_anomalies = await self._statistical_anomaly_detection(df, metric)
                
                # Machine learning anomaly detection
                ml_anomalies = await self._ml_anomaly_detection(df, metric)
                
                # Combine results
                combined_anomalies = await self._combine_anomaly_results(
                    stat_anomalies, ml_anomalies, sensitivity
                )
                
                anomalies.extend(combined_anomalies)
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Error detecting anomalies: {str(e)}")
            return []
    
    async def optimize_model_performance(self, 
                                       model_name: str,
                                       optimization_goals: List[str]) -> Dict[str, Any]:
        """Optimize model performance using AI-driven recommendations"""
        try:
            data = self.performance_data.get(model_name, [])
            if not data:
                return {"error": "No data available"}
            
            df = await self._prepare_dataframe(data)
            
            optimization_results = {
                "model_name": model_name,
                "optimization_goals": optimization_goals,
                "optimization_timestamp": datetime.now().isoformat(),
                "current_performance": {},
                "optimization_opportunities": [],
                "recommendations": [],
                "expected_improvements": {}
            }
            
            # Analyze current performance
            for goal in optimization_goals:
                if goal in df.columns:
                    current_value = df[goal].mean()
                    optimization_results["current_performance"][goal] = current_value
            
            # Find optimization opportunities
            opportunities = await self._find_optimization_opportunities(df, optimization_goals)
            optimization_results["optimization_opportunities"] = opportunities
            
            # Generate recommendations
            recommendations = await self._generate_optimization_recommendations(
                df, optimization_goals, opportunities
            )
            optimization_results["recommendations"] = recommendations
            
            # Predict expected improvements
            improvements = await self._predict_improvements(df, optimization_goals, recommendations)
            optimization_results["expected_improvements"] = improvements
            
            return optimization_results
            
        except Exception as e:
            logger.error(f"Error optimizing model performance: {str(e)}")
            return {"error": str(e)}
    
    async def generate_intelligence_report(self, 
                                         report_type: AnalyticsType = AnalyticsType.COMPREHENSIVE,
                                         intelligence_level: IntelligenceLevel = IntelligenceLevel.ADVANCED) -> IntelligenceReport:
        """Generate comprehensive intelligence report"""
        try:
            report_id = hashlib.md5(f"{datetime.now()}_{report_type}_{intelligence_level}".encode()).hexdigest()
            
            # Collect all insights
            insights = []
            predictions = []
            recommendations = []
            risk_assessments = []
            opportunities = []
            
            # Generate insights for each model
            for model_name in self.performance_data.keys():
                # Predictive insights
                pred_insights = await self.generate_predictive_insights(model_name)
                insights.extend(pred_insights)
                
                # Performance analysis
                analysis = await self.analyze_performance_patterns(model_name, report_type)
                if "insights" in analysis:
                    insights.extend(analysis["insights"])
                
                # Anomaly detection
                anomalies = await self.detect_anomalies(model_name)
                if anomalies:
                    risk_assessments.extend(anomalies)
                
                # Optimization opportunities
                opt_results = await self.optimize_model_performance(model_name, ["quality_score", "response_time"])
                if "optimization_opportunities" in opt_results:
                    opportunities.extend(opt_results["optimization_opportunities"])
                if "recommendations" in opt_results:
                    recommendations.extend(opt_results["recommendations"])
            
            # Calculate confidence score
            confidence_score = await self._calculate_confidence_score(insights, predictions)
            
            # Create intelligence report
            report = IntelligenceReport(
                report_id=report_id,
                report_type=report_type,
                intelligence_level=intelligence_level,
                generated_at=datetime.now(),
                insights=insights,
                predictions=predictions,
                recommendations=recommendations,
                risk_assessments=risk_assessments,
                opportunities=opportunities,
                performance_summary=await self._generate_performance_summary(),
                confidence_score=confidence_score
            )
            
            # Store report
            self.intelligence_reports.append(report)
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating intelligence report: {str(e)}")
            return None
    
    # Private helper methods
    async def _prepare_dataframe(self, data: List[Dict[str, Any]]) -> pd.DataFrame:
        """Prepare DataFrame from performance data"""
        try:
            # Flatten the data
            flattened_data = []
            for item in data:
                flat_item = {
                    "timestamp": item["timestamp"],
                    "model_name": item["model_name"]
                }
                flat_item.update(item["metrics"])
                flattened_data.append(flat_item)
            
            df = pd.DataFrame(flattened_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
            
            return df
            
        except Exception as e:
            logger.error(f"Error preparing DataFrame: {str(e)}")
            return pd.DataFrame()
    
    async def _train_predictive_model(self, df: pd.DataFrame, target_column: str) -> Optional[PredictiveModel]:
        """Train predictive model for a specific metric"""
        try:
            # Prepare features
            feature_columns = [col for col in df.columns if col not in ['timestamp', 'model_name', target_column]]
            
            if not feature_columns:
                return None
            
            # Create time-based features
            df_features = df.copy()
            df_features['hour'] = df_features['timestamp'].dt.hour
            df_features['day_of_week'] = df_features['timestamp'].dt.dayofweek
            df_features['day_of_month'] = df_features['timestamp'].dt.day
            
            # Prepare training data
            X = df_features[feature_columns + ['hour', 'day_of_week', 'day_of_month']].fillna(0)
            y = df_features[target_column].fillna(0)
            
            if len(X) < 10:
                return None
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train multiple models and select best
            models = {
                'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
                'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
                'Ridge': Ridge(alpha=1.0),
                'LinearRegression': LinearRegression()
            }
            
            best_model = None
            best_score = -float('inf')
            best_name = ""
            
            for name, model in models.items():
                try:
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                    score = r2_score(y_test, y_pred)
                    
                    if score > best_score:
                        best_score = score
                        best_model = model
                        best_name = name
                except Exception as e:
                    logger.warning(f"Error training {name}: {str(e)}")
                    continue
            
            if best_model is None:
                return None
            
            # Create predictive model object
            model_id = hashlib.md5(f"{target_column}_{datetime.now()}".encode()).hexdigest()
            
            predictive_model = PredictiveModel(
                model_id=model_id,
                model_name=f"{target_column}_predictor",
                model_type="regression",
                algorithm=best_name,
                accuracy=best_score,
                features=feature_columns + ['hour', 'day_of_week', 'day_of_month'],
                target=target_column,
                trained_at=datetime.now(),
                last_updated=datetime.now(),
                performance_metrics={
                    "r2_score": best_score,
                    "mse": mean_squared_error(y_test, y_pred),
                    "mae": mean_absolute_error(y_test, y_pred)
                },
                model_data=best_model
            )
            
            # Store model
            self.predictive_models[model_id] = predictive_model
            self.scalers[model_id] = scaler
            
            return predictive_model
            
        except Exception as e:
            logger.error(f"Error training predictive model: {str(e)}")
            return None
    
    async def _generate_predictions(self, 
                                  model: PredictiveModel, 
                                  df: pd.DataFrame, 
                                  target_column: str, 
                                  horizon: int) -> List[Dict[str, Any]]:
        """Generate predictions using trained model"""
        try:
            predictions = []
            
            # Get last data point
            last_row = df.iloc[-1]
            last_timestamp = last_row['timestamp']
            
            # Generate predictions for future time points
            for i in range(1, horizon + 1):
                future_timestamp = last_timestamp + timedelta(days=i)
                
                # Prepare features for prediction
                feature_values = []
                for feature in model.features:
                    if feature in ['hour', 'day_of_week', 'day_of_month']:
                        if feature == 'hour':
                            feature_values.append(future_timestamp.hour)
                        elif feature == 'day_of_week':
                            feature_values.append(future_timestamp.weekday())
                        elif feature == 'day_of_month':
                            feature_values.append(future_timestamp.day)
                    else:
                        # Use recent average for other features
                        recent_values = df[feature].tail(7).mean()
                        feature_values.append(recent_values)
                
                # Scale features
                scaler = self.scalers.get(model.model_id)
                if scaler:
                    scaled_features = scaler.transform([feature_values])
                    
                    # Make prediction
                    prediction = model.model_data.predict(scaled_features)[0]
                    
                    predictions.append({
                        "timestamp": future_timestamp.isoformat(),
                        "predicted_value": float(prediction),
                        "confidence": model.accuracy,
                        "model_id": model.model_id
                    })
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error generating predictions: {str(e)}")
            return []
    
    async def _create_predictive_insight(self, 
                                       model_name: str, 
                                       metric: str, 
                                       predictions: List[Dict[str, Any]], 
                                       confidence: float) -> AIInsight:
        """Create predictive insight from predictions"""
        try:
            insight_id = hashlib.md5(f"{model_name}_{metric}_{datetime.now()}".encode()).hexdigest()
            
            # Analyze predictions
            if predictions:
                latest_prediction = predictions[-1]
                current_trend = "stable"
                
                if len(predictions) > 1:
                    first_pred = predictions[0]["predicted_value"]
                    last_pred = predictions[-1]["predicted_value"]
                    
                    if last_pred > first_pred * 1.05:
                        current_trend = "improving"
                    elif last_pred < first_pred * 0.95:
                        current_trend = "declining"
                
                # Generate insight
                if current_trend == "improving":
                    title = f"{model_name} {metric} Performance Improving"
                    description = f"Predicted {metric} will improve by {((latest_prediction - predictions[0]['predicted_value']) / predictions[0]['predicted_value'] * 100):.1f}% over the next 30 days"
                    recommendations = [
                        "Consider scaling up usage of this model",
                        "Monitor for continued improvement",
                        "Share best practices with other models"
                    ]
                elif current_trend == "declining":
                    title = f"{model_name} {metric} Performance Declining"
                    description = f"Predicted {metric} will decline by {((predictions[0]['predicted_value'] - latest_prediction) / predictions[0]['predicted_value'] * 100):.1f}% over the next 30 days"
                    recommendations = [
                        "Investigate potential causes of decline",
                        "Consider model retraining or replacement",
                        "Implement monitoring alerts"
                    ]
                else:
                    title = f"{model_name} {metric} Performance Stable"
                    description = f"Predicted {metric} will remain stable around {latest_prediction:.3f} over the next 30 days"
                    recommendations = [
                        "Continue current monitoring",
                        "Look for optimization opportunities",
                        "Consider incremental improvements"
                    ]
                
                return AIInsight(
                    insight_id=insight_id,
                    insight_type="predictive",
                    title=title,
                    description=description,
                    confidence=confidence,
                    impact_score=abs(latest_prediction - predictions[0]["predicted_value"]) / predictions[0]["predicted_value"],
                    actionable=True,
                    recommendations=recommendations,
                    supporting_data={
                        "predictions": predictions,
                        "trend": current_trend,
                        "model_name": model_name,
                        "metric": metric
                    },
                    generated_at=datetime.now(),
                    expires_at=datetime.now() + timedelta(days=7)
                )
            
        except Exception as e:
            logger.error(f"Error creating predictive insight: {str(e)}")
            return None
    
    async def _descriptive_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform descriptive analysis"""
        try:
            analysis = {
                "summary_statistics": {},
                "trends": {},
                "patterns": {}
            }
            
            # Calculate summary statistics for each metric
            for column in df.columns:
                if column not in ['timestamp', 'model_name']:
                    analysis["summary_statistics"][column] = {
                        "mean": float(df[column].mean()),
                        "median": float(df[column].median()),
                        "std": float(df[column].std()),
                        "min": float(df[column].min()),
                        "max": float(df[column].max()),
                        "count": int(df[column].count())
                    }
            
            # Analyze trends
            for column in df.columns:
                if column not in ['timestamp', 'model_name']:
                    # Calculate trend using linear regression
                    x = np.arange(len(df))
                    y = df[column].values
                    
                    if len(y) > 1:
                        slope = np.polyfit(x, y, 1)[0]
                        trend_direction = "improving" if slope > 0 else "declining" if slope < 0 else "stable"
                        
                        analysis["trends"][column] = {
                            "direction": trend_direction,
                            "slope": float(slope),
                            "strength": abs(slope) / df[column].std() if df[column].std() > 0 else 0
                        }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in descriptive analysis: {str(e)}")
            return {}
    
    async def _diagnostic_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform diagnostic analysis"""
        try:
            analysis = {
                "correlations": {},
                "anomalies": {},
                "seasonality": {}
            }
            
            # Calculate correlations between metrics
            numeric_columns = [col for col in df.columns if col not in ['timestamp', 'model_name']]
            if len(numeric_columns) > 1:
                correlation_matrix = df[numeric_columns].corr()
                analysis["correlations"] = correlation_matrix.to_dict()
            
            # Detect anomalies
            for column in numeric_columns:
                values = df[column].values
                if len(values) > 5:
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    
                    anomalies = []
                    for i, value in enumerate(values):
                        z_score = abs(value - mean_val) / std_val if std_val > 0 else 0
                        if z_score > 2.0:
                            anomalies.append({
                                "index": i,
                                "value": float(value),
                                "z_score": float(z_score),
                                "timestamp": df.iloc[i]['timestamp'].isoformat()
                            })
                    
                    analysis["anomalies"][column] = anomalies
            
            # Analyze seasonality
            if 'timestamp' in df.columns:
                df['hour'] = df['timestamp'].dt.hour
                df['day_of_week'] = df['timestamp'].dt.dayofweek
                
                for column in numeric_columns:
                    hourly_avg = df.groupby('hour')[column].mean()
                    daily_avg = df.groupby('day_of_week')[column].mean()
                    
                    analysis["seasonality"][column] = {
                        "hourly_pattern": hourly_avg.to_dict(),
                        "daily_pattern": daily_avg.to_dict()
                    }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in diagnostic analysis: {str(e)}")
            return {}
    
    async def _prescriptive_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform prescriptive analysis"""
        try:
            analysis = {
                "optimization_opportunities": [],
                "recommendations": [],
                "action_plans": []
            }
            
            # Find optimization opportunities
            for column in df.columns:
                if column not in ['timestamp', 'model_name']:
                    values = df[column].values
                    if len(values) > 10:
                        # Calculate improvement potential
                        current_avg = np.mean(values)
                        best_performance = np.max(values)
                        improvement_potential = (best_performance - current_avg) / current_avg
                        
                        if improvement_potential > 0.1:  # 10% improvement potential
                            analysis["optimization_opportunities"].append({
                                "metric": column,
                                "current_average": float(current_avg),
                                "best_performance": float(best_performance),
                                "improvement_potential": float(improvement_potential),
                                "priority": "high" if improvement_potential > 0.3 else "medium"
                            })
            
            # Generate recommendations
            for opportunity in analysis["optimization_opportunities"]:
                if opportunity["priority"] == "high":
                    analysis["recommendations"].append(
                        f"High priority: Optimize {opportunity['metric']} - potential {opportunity['improvement_potential']:.1%} improvement"
                    )
                else:
                    analysis["recommendations"].append(
                        f"Medium priority: Consider optimizing {opportunity['metric']} - potential {opportunity['improvement_potential']:.1%} improvement"
                    )
            
            # Create action plans
            if analysis["optimization_opportunities"]:
                analysis["action_plans"].append({
                    "plan_name": "Performance Optimization",
                    "description": "Systematic approach to improve model performance",
                    "steps": [
                        "Identify root causes of performance variations",
                        "Implement targeted optimizations",
                        "Monitor improvements and adjust strategies",
                        "Document best practices"
                    ],
                    "expected_impact": "10-30% performance improvement",
                    "timeline": "2-4 weeks"
                })
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in prescriptive analysis: {str(e)}")
            return {}
    
    async def _predictive_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform predictive analysis"""
        try:
            analysis = {
                "forecasts": {},
                "scenarios": {},
                "risk_assessments": {}
            }
            
            # Generate forecasts for each metric
            for column in df.columns:
                if column not in ['timestamp', 'model_name']:
                    values = df[column].values
                    if len(values) > 20:
                        # Simple linear trend forecast
                        x = np.arange(len(values))
                        slope, intercept = np.polyfit(x, values, 1)
                        
                        # Generate 30-day forecast
                        future_x = np.arange(len(values), len(values) + 30)
                        forecast = slope * future_x + intercept
                        
                        analysis["forecasts"][column] = {
                            "trend": "improving" if slope > 0 else "declining" if slope < 0 else "stable",
                            "forecast_values": forecast.tolist(),
                            "confidence": min(0.9, len(values) / 100),  # More data = higher confidence
                            "forecast_period": "30 days"
                        }
            
            # Generate scenarios
            analysis["scenarios"] = {
                "best_case": "Performance improves by 20% over next 30 days",
                "worst_case": "Performance declines by 15% over next 30 days",
                "most_likely": "Performance remains stable with minor fluctuations"
            }
            
            # Risk assessments
            for column in df.columns:
                if column not in ['timestamp', 'model_name']:
                    values = df[column].values
                    if len(values) > 10:
                        volatility = np.std(values) / np.mean(values) if np.mean(values) > 0 else 0
                        
                        risk_level = "low"
                        if volatility > 0.2:
                            risk_level = "high"
                        elif volatility > 0.1:
                            risk_level = "medium"
                        
                        analysis["risk_assessments"][column] = {
                            "volatility": float(volatility),
                            "risk_level": risk_level,
                            "recommendation": f"Monitor {column} closely" if risk_level == "high" else f"Standard monitoring for {column}"
                        }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in predictive analysis: {str(e)}")
            return {}
    
    async def _statistical_anomaly_detection(self, df: pd.DataFrame, column: str) -> List[Dict[str, Any]]:
        """Detect anomalies using statistical methods"""
        try:
            values = df[column].values
            if len(values) < 5:
                return []
            
            mean_val = np.mean(values)
            std_val = np.std(values)
            
            anomalies = []
            for i, value in enumerate(values):
                z_score = abs(value - mean_val) / std_val if std_val > 0 else 0
                if z_score > 2.0:
                    anomalies.append({
                        "index": i,
                        "value": float(value),
                        "z_score": float(z_score),
                        "method": "statistical",
                        "timestamp": df.iloc[i]['timestamp'].isoformat()
                    })
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Error in statistical anomaly detection: {str(e)}")
            return []
    
    async def _ml_anomaly_detection(self, df: pd.DataFrame, column: str) -> List[Dict[str, Any]]:
        """Detect anomalies using machine learning"""
        try:
            values = df[column].values.reshape(-1, 1)
            if len(values) < 10:
                return []
            
            # Use DBSCAN for anomaly detection
            dbscan = DBSCAN(eps=0.5, min_samples=3)
            clusters = dbscan.fit_predict(values)
            
            anomalies = []
            for i, cluster in enumerate(clusters):
                if cluster == -1:  # Outlier
                    anomalies.append({
                        "index": i,
                        "value": float(values[i][0]),
                        "method": "ml_dbscan",
                        "timestamp": df.iloc[i]['timestamp'].isoformat()
                    })
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Error in ML anomaly detection: {str(e)}")
            return []
    
    async def _combine_anomaly_results(self, 
                                     stat_anomalies: List[Dict[str, Any]], 
                                     ml_anomalies: List[Dict[str, Any]], 
                                     sensitivity: float) -> List[Dict[str, Any]]:
        """Combine results from different anomaly detection methods"""
        try:
            combined = []
            
            # Create index sets for each method
            stat_indices = {a["index"] for a in stat_anomalies}
            ml_indices = {a["index"] for a in ml_anomalies}
            
            # Find consensus anomalies
            consensus_indices = stat_indices.intersection(ml_indices)
            
            for index in consensus_indices:
                stat_anomaly = next(a for a in stat_anomalies if a["index"] == index)
                ml_anomaly = next(a for a in ml_anomalies if a["index"] == index)
                
                combined.append({
                    "index": index,
                    "value": stat_anomaly["value"],
                    "confidence": 0.9,  # High confidence for consensus
                    "methods": ["statistical", "ml_dbscan"],
                    "timestamp": stat_anomaly["timestamp"]
                })
            
            # Add high-confidence single-method anomalies based on sensitivity
            if sensitivity < 0.5:  # Lower sensitivity = more anomalies
                for anomaly in stat_anomalies:
                    if anomaly["index"] not in consensus_indices and anomaly["z_score"] > 3.0:
                        combined.append({
                            "index": anomaly["index"],
                            "value": anomaly["value"],
                            "confidence": 0.7,
                            "methods": ["statistical"],
                            "timestamp": anomaly["timestamp"]
                        })
            
            return combined
            
        except Exception as e:
            logger.error(f"Error combining anomaly results: {str(e)}")
            return []
    
    async def _find_optimization_opportunities(self, 
                                             df: pd.DataFrame, 
                                             goals: List[str]) -> List[Dict[str, Any]]:
        """Find optimization opportunities"""
        try:
            opportunities = []
            
            for goal in goals:
                if goal in df.columns:
                    values = df[goal].values
                    if len(values) > 5:
                        current_avg = np.mean(values)
                        best_performance = np.max(values)
                        worst_performance = np.min(values)
                        
                        # Calculate improvement potential
                        improvement_potential = (best_performance - current_avg) / current_avg
                        consistency_score = 1 - (np.std(values) / current_avg) if current_avg > 0 else 0
                        
                        opportunities.append({
                            "metric": goal,
                            "current_average": float(current_avg),
                            "best_performance": float(best_performance),
                            "worst_performance": float(worst_performance),
                            "improvement_potential": float(improvement_potential),
                            "consistency_score": float(consistency_score),
                            "priority": "high" if improvement_potential > 0.2 else "medium" if improvement_potential > 0.1 else "low"
                        })
            
            return opportunities
            
        except Exception as e:
            logger.error(f"Error finding optimization opportunities: {str(e)}")
            return []
    
    async def _generate_optimization_recommendations(self, 
                                                   df: pd.DataFrame, 
                                                   goals: List[str], 
                                                   opportunities: List[Dict[str, Any]]) -> List[str]:
        """Generate optimization recommendations"""
        try:
            recommendations = []
            
            for opportunity in opportunities:
                if opportunity["priority"] == "high":
                    recommendations.append(
                        f"URGENT: Optimize {opportunity['metric']} - {opportunity['improvement_potential']:.1%} improvement potential"
                    )
                elif opportunity["priority"] == "medium":
                    recommendations.append(
                        f"Consider optimizing {opportunity['metric']} - {opportunity['improvement_potential']:.1%} improvement potential"
                    )
                
                if opportunity["consistency_score"] < 0.7:
                    recommendations.append(
                        f"Improve consistency of {opportunity['metric']} - high variability detected"
                    )
            
            # Add general recommendations
            if len(opportunities) > 0:
                recommendations.extend([
                    "Implement continuous monitoring for all metrics",
                    "Set up automated alerts for performance degradation",
                    "Create performance baselines and targets",
                    "Document optimization strategies for future reference"
                ])
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating optimization recommendations: {str(e)}")
            return []
    
    async def _predict_improvements(self, 
                                  df: pd.DataFrame, 
                                  goals: List[str], 
                                  recommendations: List[str]) -> Dict[str, Any]:
        """Predict expected improvements from recommendations"""
        try:
            improvements = {}
            
            for goal in goals:
                if goal in df.columns:
                    values = df[goal].values
                    if len(values) > 5:
                        current_avg = np.mean(values)
                        best_performance = np.max(values)
                        
                        # Estimate improvement based on recommendations
                        if any("URGENT" in rec for rec in recommendations):
                            expected_improvement = (best_performance - current_avg) * 0.8  # 80% of potential
                        elif any("Consider optimizing" in rec for rec in recommendations):
                            expected_improvement = (best_performance - current_avg) * 0.5  # 50% of potential
                        else:
                            expected_improvement = (best_performance - current_avg) * 0.2  # 20% of potential
                        
                        improvements[goal] = {
                            "current_value": float(current_avg),
                            "expected_improvement": float(expected_improvement),
                            "improvement_percentage": float(expected_improvement / current_avg * 100),
                            "confidence": 0.7 if expected_improvement > 0 else 0.3
                        }
            
            return improvements
            
        except Exception as e:
            logger.error(f"Error predicting improvements: {str(e)}")
            return {}
    
    async def _calculate_confidence_score(self, 
                                        insights: List[AIInsight], 
                                        predictions: List[Dict[str, Any]]) -> float:
        """Calculate overall confidence score for the report"""
        try:
            if not insights and not predictions:
                return 0.0
            
            # Calculate confidence from insights
            insight_confidence = 0.0
            if insights:
                insight_confidence = np.mean([insight.confidence for insight in insights])
            
            # Calculate confidence from predictions
            prediction_confidence = 0.0
            if predictions:
                prediction_confidence = np.mean([pred.get("confidence", 0.5) for pred in predictions])
            
            # Weighted average
            total_items = len(insights) + len(predictions)
            if total_items > 0:
                overall_confidence = (insight_confidence * len(insights) + prediction_confidence * len(predictions)) / total_items
            else:
                overall_confidence = 0.0
            
            return float(overall_confidence)
            
        except Exception as e:
            logger.error(f"Error calculating confidence score: {str(e)}")
            return 0.0
    
    async def _generate_performance_summary(self) -> Dict[str, Any]:
        """Generate overall performance summary"""
        try:
            summary = {
                "total_models": len(self.performance_data),
                "total_data_points": sum(len(data) for data in self.performance_data.values()),
                "analytics_insights": len(self.insights_history),
                "predictive_models": len(self.predictive_models),
                "intelligence_reports": len(self.intelligence_reports)
            }
            
            # Calculate overall performance metrics
            all_metrics = {}
            for model_name, data in self.performance_data.items():
                for data_point in data:
                    for metric, value in data_point["metrics"].items():
                        if metric not in all_metrics:
                            all_metrics[metric] = []
                        all_metrics[metric].append(value)
            
            summary["overall_metrics"] = {}
            for metric, values in all_metrics.items():
                if values:
                    summary["overall_metrics"][metric] = {
                        "average": float(np.mean(values)),
                        "median": float(np.median(values)),
                        "std": float(np.std(values)),
                        "count": len(values)
                    }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating performance summary: {str(e)}")
            return {}
    
    async def _cleanup_old_data(self):
        """Clean up old data"""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.max_history_days)
            
            for model_name in list(self.performance_data.keys()):
                self.performance_data[model_name] = [
                    data for data in self.performance_data[model_name]
                    if data["timestamp"] >= cutoff_date
                ]
                
                if not self.performance_data[model_name]:
                    del self.performance_data[model_name]
            
            # Clean up old insights (keep last 1000)
            if len(self.insights_history) > 1000:
                self.insights_history = self.insights_history[-1000:]
            
            # Clean up old reports (keep last 100)
            if len(self.intelligence_reports) > 100:
                self.intelligence_reports = self.intelligence_reports[-100:]
            
        except Exception as e:
            logger.error(f"Error cleaning up old data: {str(e)}")
    
    def _invalidate_cache(self):
        """Invalidate analytics cache"""
        self.analytics_cache.clear()


# Global analytics instance
_analytics: Optional[AdvancedAIAnalytics] = None


def get_advanced_ai_analytics(max_history_days: int = 365) -> AdvancedAIAnalytics:
    """Get or create global advanced AI analytics instance"""
    global _analytics
    if _analytics is None:
        _analytics = AdvancedAIAnalytics(max_history_days)
    return _analytics


# Example usage
async def main():
    """Example usage of the advanced AI analytics system"""
    analytics = get_advanced_ai_analytics()
    
    # Record sample performance data
    models = ["gpt-4", "claude-3", "gemini-pro"]
    metrics = ["quality_score", "response_time", "cost_efficiency", "accuracy"]
    
    # Simulate performance data
    for i in range(100):
        for model in models:
            performance_metrics = {
                "quality_score": 0.7 + (i * 0.001) + np.random.normal(0, 0.05),
                "response_time": 2.0 - (i * 0.01) + np.random.normal(0, 0.2),
                "cost_efficiency": 0.8 + (i * 0.002) + np.random.normal(0, 0.03),
                "accuracy": 0.85 + (i * 0.001) + np.random.normal(0, 0.04)
            }
            
            await analytics.record_performance_data(
                model_name=model,
                performance_metrics=performance_metrics,
                context={"iteration": i, "test_type": "benchmark"},
                metadata={"version": "1.0", "environment": "production"}
            )
    
    # Generate predictive insights
    for model in models:
        insights = await analytics.generate_predictive_insights(model)
        print(f"Generated {len(insights)} insights for {model}")
    
    # Generate intelligence report
    report = await analytics.generate_intelligence_report()
    if report:
        print(f"Generated intelligence report with {len(report.insights)} insights")
        print(f"Confidence score: {report.confidence_score:.3f}")
    
    # Analyze performance patterns
    for model in models:
        analysis = await analytics.analyze_performance_patterns(model, AnalyticsType.DESCRIPTIVE)
        print(f"Analysis for {model}: {analysis.get('data_points', 0)} data points")


if __name__ == "__main__":
    asyncio.run(main())



























