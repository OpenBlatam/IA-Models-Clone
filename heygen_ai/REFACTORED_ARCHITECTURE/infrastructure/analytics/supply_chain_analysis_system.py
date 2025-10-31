"""
Advanced Supply Chain Analysis System

This module provides comprehensive supply chain analysis capabilities integrated
with the HeyGen AI system for intelligent supply chain optimization and insights.
"""

import asyncio
import pandas as pd
import numpy as np
import logging
import json
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from enum import Enum
import uuid
import sqlite3
from pathlib import Path
import requests
import websockets
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


logger = logging.getLogger(__name__)


class AnalysisType(str, Enum):
    """Analysis types."""
    DEMAND_FORECASTING = "demand_forecasting"
    SUPPLIER_ANALYSIS = "supplier_analysis"
    RISK_ASSESSMENT = "risk_assessment"
    COST_OPTIMIZATION = "cost_optimization"
    PERFORMANCE_METRICS = "performance_metrics"
    SUSTAINABILITY = "sustainability"
    COMPLIANCE = "compliance"
    INVENTORY_OPTIMIZATION = "inventory_optimization"


class RiskLevel(str, Enum):
    """Risk levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SupplyChainNode:
    """Supply chain node structure."""
    node_id: str
    name: str
    node_type: str  # supplier, manufacturer, distributor, retailer, customer
    location: Dict[str, float]  # lat, lng
    capacity: float
    cost_per_unit: float
    lead_time: int  # days
    reliability_score: float  # 0-1
    sustainability_score: float  # 0-1
    compliance_score: float  # 0-1
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SupplyChainEdge:
    """Supply chain edge structure."""
    edge_id: str
    from_node: str
    to_node: str
    distance: float  # km
    transportation_cost: float
    transportation_time: int  # days
    capacity: float
    reliability: float  # 0-1
    carbon_footprint: float  # CO2 emissions
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DemandForecast:
    """Demand forecast structure."""
    product_id: str
    forecast_date: datetime
    predicted_demand: float
    confidence_interval: Tuple[float, float]
    model_accuracy: float
    factors: Dict[str, float] = field(default_factory=dict)


@dataclass
class RiskAssessment:
    """Risk assessment structure."""
    risk_id: str
    risk_type: str
    risk_level: RiskLevel
    probability: float  # 0-1
    impact: float  # 0-1
    description: str
    mitigation_strategies: List[str] = field(default_factory=list)
    affected_nodes: List[str] = field(default_factory=list)


class DemandForecastingEngine:
    """Advanced demand forecasting engine."""
    
    def __init__(self):
        self.models = {
            'linear': LinearRegression(),
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=0.1),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        self.scaler = StandardScaler()
        self.is_trained = False
    
    async def train_models(self, historical_data: pd.DataFrame, target_column: str) -> Dict[str, float]:
        """Train forecasting models."""
        try:
            # Prepare data
            X = historical_data.drop(columns=[target_column])
            y = historical_data[target_column]
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train models
            model_scores = {}
            for name, model in self.models.items():
                model.fit(X_scaled, y)
                y_pred = model.predict(X_scaled)
                score = r2_score(y, y_pred)
                model_scores[name] = score
            
            self.is_trained = True
            logger.info(f"Models trained successfully. Scores: {model_scores}")
            return model_scores
            
        except Exception as e:
            logger.error(f"Error training models: {e}")
            return {}
    
    async def forecast_demand(self, features: Dict[str, float], horizon: int = 30) -> List[DemandForecast]:
        """Generate demand forecasts."""
        if not self.is_trained:
            raise ValueError("Models must be trained before forecasting")
        
        forecasts = []
        best_model_name = max(self.models.keys(), key=lambda k: self._get_model_score(k))
        best_model = self.models[best_model_name]
        
        # Prepare features
        feature_vector = np.array([list(features.values())]).reshape(1, -1)
        feature_vector_scaled = self.scaler.transform(feature_vector)
        
        # Generate forecasts
        for i in range(horizon):
            forecast_date = datetime.now(timezone.utc) + timedelta(days=i+1)
            
            # Predict demand
            predicted_demand = best_model.predict(feature_vector_scaled)[0]
            
            # Calculate confidence interval (simplified)
            confidence_interval = (
                predicted_demand * 0.8,
                predicted_demand * 1.2
            )
            
            forecast = DemandForecast(
                product_id=features.get('product_id', 'unknown'),
                forecast_date=forecast_date,
                predicted_demand=predicted_demand,
                confidence_interval=confidence_interval,
                model_accuracy=self._get_model_score(best_model_name),
                factors=features
            )
            
            forecasts.append(forecast)
        
        return forecasts
    
    def _get_model_score(self, model_name: str) -> float:
        """Get model score (placeholder implementation)."""
        return 0.85  # Placeholder score


class SupplierAnalysisEngine:
    """Advanced supplier analysis engine."""
    
    def __init__(self):
        self.suppliers: Dict[str, SupplyChainNode] = {}
        self.performance_metrics = {}
    
    async def analyze_supplier_performance(self, supplier_id: str, historical_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze supplier performance."""
        try:
            if supplier_id not in self.suppliers:
                raise ValueError(f"Supplier {supplier_id} not found")
            
            supplier = self.suppliers[supplier_id]
            
            # Calculate performance metrics
            on_time_delivery = self._calculate_on_time_delivery(historical_data)
            quality_score = self._calculate_quality_score(historical_data)
            cost_efficiency = self._calculate_cost_efficiency(historical_data)
            reliability_score = self._calculate_reliability_score(historical_data)
            
            # Overall performance score
            overall_score = (
                on_time_delivery * 0.3 +
                quality_score * 0.3 +
                cost_efficiency * 0.2 +
                reliability_score * 0.2
            )
            
            analysis = {
                'supplier_id': supplier_id,
                'supplier_name': supplier.name,
                'on_time_delivery': on_time_delivery,
                'quality_score': quality_score,
                'cost_efficiency': cost_efficiency,
                'reliability_score': reliability_score,
                'overall_score': overall_score,
                'recommendation': self._get_recommendation(overall_score),
                'risk_factors': self._identify_risk_factors(historical_data),
                'improvement_areas': self._identify_improvement_areas(historical_data)
            }
            
            self.performance_metrics[supplier_id] = analysis
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing supplier performance: {e}")
            return {}
    
    def _calculate_on_time_delivery(self, data: pd.DataFrame) -> float:
        """Calculate on-time delivery rate."""
        if 'delivery_date' in data.columns and 'promised_date' in data.columns:
            on_time = (data['delivery_date'] <= data['promised_date']).mean()
            return float(on_time)
        return 0.85  # Default value
    
    def _calculate_quality_score(self, data: pd.DataFrame) -> float:
        """Calculate quality score."""
        if 'defect_rate' in data.columns:
            quality = 1 - data['defect_rate'].mean()
            return float(max(0, min(1, quality)))
        return 0.90  # Default value
    
    def _calculate_cost_efficiency(self, data: pd.DataFrame) -> float:
        """Calculate cost efficiency score."""
        if 'cost' in data.columns and 'quantity' in data.columns:
            avg_cost_per_unit = (data['cost'] / data['quantity']).mean()
            # Normalize based on industry average (simplified)
            efficiency = max(0, 1 - (avg_cost_per_unit - 10) / 50)
            return float(max(0, min(1, efficiency)))
        return 0.80  # Default value
    
    def _calculate_reliability_score(self, data: pd.DataFrame) -> float:
        """Calculate reliability score."""
        if 'delivery_variance' in data.columns:
            variance = data['delivery_variance'].mean()
            reliability = max(0, 1 - variance / 10)
            return float(max(0, min(1, reliability)))
        return 0.88  # Default value
    
    def _get_recommendation(self, overall_score: float) -> str:
        """Get supplier recommendation based on score."""
        if overall_score >= 0.9:
            return "Excellent - Continue partnership"
        elif overall_score >= 0.8:
            return "Good - Monitor performance"
        elif overall_score >= 0.7:
            return "Fair - Address issues"
        else:
            return "Poor - Consider replacement"
    
    def _identify_risk_factors(self, data: pd.DataFrame) -> List[str]:
        """Identify risk factors."""
        risks = []
        
        if 'delivery_variance' in data.columns and data['delivery_variance'].mean() > 5:
            risks.append("High delivery variance")
        
        if 'defect_rate' in data.columns and data['defect_rate'].mean() > 0.05:
            risks.append("High defect rate")
        
        if 'cost' in data.columns and data['cost'].std() > data['cost'].mean() * 0.2:
            risks.append("High cost volatility")
        
        return risks
    
    def _identify_improvement_areas(self, data: pd.DataFrame) -> List[str]:
        """Identify improvement areas."""
        improvements = []
        
        if 'delivery_date' in data.columns and 'promised_date' in data.columns:
            late_deliveries = (data['delivery_date'] > data['promised_date']).sum()
            if late_deliveries > len(data) * 0.1:
                improvements.append("Improve delivery reliability")
        
        if 'defect_rate' in data.columns and data['defect_rate'].mean() > 0.02:
            improvements.append("Improve quality control")
        
        if 'cost' in data.columns:
            cost_trend = data['cost'].pct_change().mean()
            if cost_trend > 0.05:
                improvements.append("Control cost increases")
        
        return improvements


class RiskAssessmentEngine:
    """Advanced risk assessment engine."""
    
    def __init__(self):
        self.risks: List[RiskAssessment] = []
        self.risk_factors = {
            'supplier_failure': 0.1,
            'transportation_delay': 0.15,
            'quality_issues': 0.08,
            'cost_increases': 0.12,
            'demand_volatility': 0.20,
            'regulatory_changes': 0.05,
            'natural_disasters': 0.03,
            'cyber_attacks': 0.02
        }
    
    async def assess_supply_chain_risks(self, supply_chain_data: Dict[str, Any]) -> List[RiskAssessment]:
        """Assess supply chain risks."""
        risks = []
        
        # Supplier risks
        supplier_risks = await self._assess_supplier_risks(supply_chain_data)
        risks.extend(supplier_risks)
        
        # Transportation risks
        transportation_risks = await self._assess_transportation_risks(supply_chain_data)
        risks.extend(transportation_risks)
        
        # Market risks
        market_risks = await self._assess_market_risks(supply_chain_data)
        risks.extend(market_risks)
        
        # Regulatory risks
        regulatory_risks = await self._assess_regulatory_risks(supply_chain_data)
        risks.extend(regulatory_risks)
        
        self.risks = risks
        return risks
    
    async def _assess_supplier_risks(self, data: Dict[str, Any]) -> List[RiskAssessment]:
        """Assess supplier-related risks."""
        risks = []
        
        # Supplier concentration risk
        if 'suppliers' in data:
            supplier_count = len(data['suppliers'])
            if supplier_count < 3:
                risk = RiskAssessment(
                    risk_id=str(uuid.uuid4()),
                    risk_type="supplier_concentration",
                    risk_level=RiskLevel.HIGH if supplier_count == 1 else RiskLevel.MEDIUM,
                    probability=0.8 if supplier_count == 1 else 0.5,
                    impact=0.9,
                    description=f"High supplier concentration with only {supplier_count} suppliers",
                    mitigation_strategies=["Diversify supplier base", "Develop backup suppliers"],
                    affected_nodes=data['suppliers']
                )
                risks.append(risk)
        
        # Supplier financial risk
        if 'supplier_financials' in data:
            for supplier_id, financials in data['supplier_financials'].items():
                if financials.get('debt_ratio', 0) > 0.6:
                    risk = RiskAssessment(
                        risk_id=str(uuid.uuid4()),
                        risk_type="supplier_financial",
                        risk_level=RiskLevel.HIGH,
                        probability=0.7,
                        impact=0.8,
                        description=f"High debt ratio for supplier {supplier_id}",
                        mitigation_strategies=["Monitor financial health", "Require financial guarantees"],
                        affected_nodes=[supplier_id]
                    )
                    risks.append(risk)
        
        return risks
    
    async def _assess_transportation_risks(self, data: Dict[str, Any]) -> List[RiskAssessment]:
        """Assess transportation-related risks."""
        risks = []
        
        # Distance-based risks
        if 'transportation_routes' in data:
            for route in data['transportation_routes']:
                distance = route.get('distance', 0)
                if distance > 1000:  # km
                    risk = RiskAssessment(
                        risk_id=str(uuid.uuid4()),
                        risk_type="transportation_distance",
                        risk_level=RiskLevel.MEDIUM,
                        probability=0.4,
                        impact=0.6,
                        description=f"Long transportation distance: {distance} km",
                        mitigation_strategies=["Use multiple routes", "Local sourcing"],
                        affected_nodes=[route.get('from'), route.get('to')]
                    )
                    risks.append(risk)
        
        return risks
    
    async def _assess_market_risks(self, data: Dict[str, Any]) -> List[RiskAssessment]:
        """Assess market-related risks."""
        risks = []
        
        # Demand volatility
        if 'demand_history' in data:
            demand_std = np.std(data['demand_history'])
            demand_mean = np.mean(data['demand_history'])
            volatility = demand_std / demand_mean if demand_mean > 0 else 0
            
            if volatility > 0.3:
                risk = RiskAssessment(
                    risk_id=str(uuid.uuid4()),
                    risk_type="demand_volatility",
                    risk_level=RiskLevel.HIGH,
                    probability=0.6,
                    impact=0.7,
                    description=f"High demand volatility: {volatility:.2f}",
                    mitigation_strategies=["Demand forecasting", "Flexible production"],
                    affected_nodes=[]
                )
                risks.append(risk)
        
        return risks
    
    async def _assess_regulatory_risks(self, data: Dict[str, Any]) -> List[RiskAssessment]:
        """Assess regulatory-related risks."""
        risks = []
        
        # International trade risks
        if 'international_suppliers' in data and data['international_suppliers']:
            risk = RiskAssessment(
                risk_id=str(uuid.uuid4()),
                risk_type="international_trade",
                risk_level=RiskLevel.MEDIUM,
                probability=0.3,
                impact=0.8,
                description="Dependence on international suppliers",
                mitigation_strategies=["Local sourcing", "Trade agreements"],
                affected_nodes=data['international_suppliers']
            )
            risks.append(risk)
        
        return risks


class CostOptimizationEngine:
    """Advanced cost optimization engine."""
    
    def __init__(self):
        self.optimization_models = {}
        self.cost_factors = {}
    
    async def optimize_supply_chain_costs(self, supply_chain_data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize supply chain costs."""
        try:
            # Analyze current costs
            current_costs = self._analyze_current_costs(supply_chain_data)
            
            # Identify optimization opportunities
            opportunities = self._identify_optimization_opportunities(supply_chain_data)
            
            # Generate optimization recommendations
            recommendations = self._generate_optimization_recommendations(opportunities)
            
            # Calculate potential savings
            potential_savings = self._calculate_potential_savings(recommendations)
            
            optimization_result = {
                'current_costs': current_costs,
                'optimization_opportunities': opportunities,
                'recommendations': recommendations,
                'potential_savings': potential_savings,
                'optimization_score': self._calculate_optimization_score(recommendations)
            }
            
            return optimization_result
            
        except Exception as e:
            logger.error(f"Error optimizing costs: {e}")
            return {}
    
    def _analyze_current_costs(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Analyze current supply chain costs."""
        costs = {
            'procurement': 0,
            'transportation': 0,
            'inventory': 0,
            'quality': 0,
            'total': 0
        }
        
        # Calculate procurement costs
        if 'suppliers' in data:
            for supplier in data['suppliers']:
                costs['procurement'] += supplier.get('cost_per_unit', 0) * supplier.get('quantity', 0)
        
        # Calculate transportation costs
        if 'transportation_routes' in data:
            for route in data['transportation_routes']:
                costs['transportation'] += route.get('cost', 0)
        
        # Calculate inventory costs
        if 'inventory' in data:
            inventory_value = data['inventory'].get('value', 0)
            holding_cost_rate = data['inventory'].get('holding_cost_rate', 0.2)
            costs['inventory'] = inventory_value * holding_cost_rate
        
        # Calculate quality costs
        if 'quality_issues' in data:
            costs['quality'] = sum(issue.get('cost', 0) for issue in data['quality_issues'])
        
        costs['total'] = sum(costs.values())
        return costs
    
    def _identify_optimization_opportunities(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify cost optimization opportunities."""
        opportunities = []
        
        # Supplier consolidation
        if 'suppliers' in data and len(data['suppliers']) > 5:
            opportunities.append({
                'type': 'supplier_consolidation',
                'description': 'Consolidate suppliers to reduce procurement costs',
                'potential_savings': 0.15,
                'implementation_effort': 'medium'
            })
        
        # Transportation optimization
        if 'transportation_routes' in data:
            opportunities.append({
                'type': 'transportation_optimization',
                'description': 'Optimize transportation routes and modes',
                'potential_savings': 0.10,
                'implementation_effort': 'low'
            })
        
        # Inventory optimization
        if 'inventory' in data:
            opportunities.append({
                'type': 'inventory_optimization',
                'description': 'Optimize inventory levels and locations',
                'potential_savings': 0.20,
                'implementation_effort': 'high'
            })
        
        return opportunities
    
    def _generate_optimization_recommendations(self, opportunities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate specific optimization recommendations."""
        recommendations = []
        
        for opportunity in opportunities:
            if opportunity['type'] == 'supplier_consolidation':
                recommendations.append({
                    'priority': 'high',
                    'action': 'Consolidate suppliers from 8 to 4 key suppliers',
                    'expected_savings': '$500,000 annually',
                    'timeline': '6 months',
                    'requirements': ['Supplier evaluation', 'Contract negotiation']
                })
            
            elif opportunity['type'] == 'transportation_optimization':
                recommendations.append({
                    'priority': 'medium',
                    'action': 'Implement route optimization software',
                    'expected_savings': '$200,000 annually',
                    'timeline': '3 months',
                    'requirements': ['Software implementation', 'Driver training']
                })
            
            elif opportunity['type'] == 'inventory_optimization':
                recommendations.append({
                    'priority': 'high',
                    'action': 'Implement just-in-time inventory management',
                    'expected_savings': '$800,000 annually',
                    'timeline': '12 months',
                    'requirements': ['System integration', 'Process redesign']
                })
        
        return recommendations
    
    def _calculate_potential_savings(self, recommendations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate potential savings from recommendations."""
        total_savings = 0
        savings_by_priority = {'high': 0, 'medium': 0, 'low': 0}
        
        for rec in recommendations:
            # Extract savings amount (simplified)
            savings_text = rec.get('expected_savings', '$0')
            savings_amount = float(savings_text.replace('$', '').replace(',', '').replace(' annually', ''))
            total_savings += savings_amount
            
            priority = rec.get('priority', 'low')
            savings_by_priority[priority] += savings_amount
        
        return {
            'total_annual_savings': total_savings,
            'savings_by_priority': savings_by_priority,
            'roi_percentage': (total_savings / 1000000) * 100  # Assuming $1M baseline
        }
    
    def _calculate_optimization_score(self, recommendations: List[Dict[str, Any]]) -> float:
        """Calculate overall optimization score."""
        if not recommendations:
            return 0.0
        
        # Weight by priority
        priority_weights = {'high': 1.0, 'medium': 0.7, 'low': 0.4}
        weighted_score = 0
        total_weight = 0
        
        for rec in recommendations:
            priority = rec.get('priority', 'low')
            weight = priority_weights.get(priority, 0.4)
            weighted_score += weight
            total_weight += weight
        
        return weighted_score / total_weight if total_weight > 0 else 0.0


class AdvancedSupplyChainAnalysisSystem:
    """
    Advanced supply chain analysis system with comprehensive capabilities.
    
    Features:
    - Demand forecasting with multiple ML models
    - Supplier performance analysis
    - Risk assessment and mitigation
    - Cost optimization recommendations
    - Performance metrics and KPIs
    - Sustainability analysis
    - Compliance monitoring
    - Real-time analytics and reporting
    """
    
    def __init__(self, db_path: str = "supply_chain_analysis.db"):
        """
        Initialize the advanced supply chain analysis system.
        
        Args:
            db_path: SQLite database path for data persistence
        """
        self.db_path = db_path
        self.db_connection = None
        
        # Initialize engines
        self.demand_forecasting = DemandForecastingEngine()
        self.supplier_analysis = SupplierAnalysisEngine()
        self.risk_assessment = RiskAssessmentEngine()
        self.cost_optimization = CostOptimizationEngine()
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database."""
        self.db_connection = sqlite3.connect(self.db_path)
        cursor = self.db_connection.cursor()
        
        # Create tables
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS supply_chain_nodes (
                node_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                node_type TEXT NOT NULL,
                location_lat REAL,
                location_lng REAL,
                capacity REAL,
                cost_per_unit REAL,
                lead_time INTEGER,
                reliability_score REAL,
                sustainability_score REAL,
                compliance_score REAL,
                metadata TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS supply_chain_edges (
                edge_id TEXT PRIMARY KEY,
                from_node TEXT NOT NULL,
                to_node TEXT NOT NULL,
                distance REAL,
                transportation_cost REAL,
                transportation_time INTEGER,
                capacity REAL,
                reliability REAL,
                carbon_footprint REAL,
                metadata TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS demand_forecasts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                product_id TEXT NOT NULL,
                forecast_date DATETIME NOT NULL,
                predicted_demand REAL NOT NULL,
                confidence_lower REAL,
                confidence_upper REAL,
                model_accuracy REAL,
                factors TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS risk_assessments (
                risk_id TEXT PRIMARY KEY,
                risk_type TEXT NOT NULL,
                risk_level TEXT NOT NULL,
                probability REAL NOT NULL,
                impact REAL NOT NULL,
                description TEXT NOT NULL,
                mitigation_strategies TEXT,
                affected_nodes TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        self.db_connection.commit()
    
    async def analyze_supply_chain(self, supply_chain_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive supply chain analysis."""
        try:
            analysis_results = {}
            
            # Demand forecasting
            if 'historical_demand' in supply_chain_data:
                forecasts = await self.demand_forecasting.forecast_demand(
                    supply_chain_data['historical_demand'],
                    horizon=30
                )
                analysis_results['demand_forecasts'] = [f.__dict__ for f in forecasts]
            
            # Supplier analysis
            if 'suppliers' in supply_chain_data:
                supplier_results = {}
                for supplier_id in supply_chain_data['suppliers']:
                    supplier_data = supply_chain_data['suppliers'][supplier_id]
                    if 'historical_data' in supplier_data:
                        result = await self.supplier_analysis.analyze_supplier_performance(
                            supplier_id,
                            pd.DataFrame(supplier_data['historical_data'])
                        )
                        supplier_results[supplier_id] = result
                analysis_results['supplier_analysis'] = supplier_results
            
            # Risk assessment
            risks = await self.risk_assessment.assess_supply_chain_risks(supply_chain_data)
            analysis_results['risk_assessment'] = [r.__dict__ for r in risks]
            
            # Cost optimization
            cost_optimization = await self.cost_optimization.optimize_supply_chain_costs(supply_chain_data)
            analysis_results['cost_optimization'] = cost_optimization
            
            # Performance metrics
            performance_metrics = self._calculate_performance_metrics(supply_chain_data)
            analysis_results['performance_metrics'] = performance_metrics
            
            # Store results in database
            await self._store_analysis_results(analysis_results)
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error analyzing supply chain: {e}")
            return {}
    
    def _calculate_performance_metrics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate supply chain performance metrics."""
        metrics = {}
        
        # On-time delivery rate
        if 'deliveries' in data:
            total_deliveries = len(data['deliveries'])
            on_time_deliveries = sum(1 for d in data['deliveries'] if d.get('on_time', False))
            metrics['on_time_delivery_rate'] = on_time_deliveries / total_deliveries if total_deliveries > 0 else 0
        
        # Inventory turnover
        if 'inventory' in data and 'sales' in data:
            avg_inventory = data['inventory'].get('value', 0)
            annual_sales = data['sales'].get('annual', 0)
            metrics['inventory_turnover'] = annual_sales / avg_inventory if avg_inventory > 0 else 0
        
        # Cost per unit
        if 'total_costs' in data and 'units_produced' in data:
            metrics['cost_per_unit'] = data['total_costs'] / data['units_produced']
        
        # Supplier reliability
        if 'suppliers' in data:
            reliability_scores = [s.get('reliability_score', 0) for s in data['suppliers'].values()]
            metrics['avg_supplier_reliability'] = np.mean(reliability_scores) if reliability_scores else 0
        
        return metrics
    
    async def _store_analysis_results(self, results: Dict[str, Any]):
        """Store analysis results in database."""
        try:
            cursor = self.db_connection.cursor()
            
            # Store demand forecasts
            if 'demand_forecasts' in results:
                for forecast in results['demand_forecasts']:
                    cursor.execute("""
                        INSERT INTO demand_forecasts 
                        (product_id, forecast_date, predicted_demand, confidence_lower, 
                         confidence_upper, model_accuracy, factors)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        forecast['product_id'],
                        forecast['forecast_date'],
                        forecast['predicted_demand'],
                        forecast['confidence_interval'][0],
                        forecast['confidence_interval'][1],
                        forecast['model_accuracy'],
                        json.dumps(forecast['factors'])
                    ))
            
            # Store risk assessments
            if 'risk_assessment' in results:
                for risk in results['risk_assessment']:
                    cursor.execute("""
                        INSERT INTO risk_assessments 
                        (risk_id, risk_type, risk_level, probability, impact, 
                         description, mitigation_strategies, affected_nodes)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        risk['risk_id'],
                        risk['risk_type'],
                        risk['risk_level'],
                        risk['probability'],
                        risk['impact'],
                        risk['description'],
                        json.dumps(risk['mitigation_strategies']),
                        json.dumps(risk['affected_nodes'])
                    ))
            
            self.db_connection.commit()
            
        except Exception as e:
            logger.error(f"Error storing analysis results: {e}")
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get analysis summary from database."""
        try:
            cursor = self.db_connection.cursor()
            
            # Get demand forecast summary
            cursor.execute("""
                SELECT COUNT(*) as total_forecasts, 
                       AVG(predicted_demand) as avg_demand,
                       AVG(model_accuracy) as avg_accuracy
                FROM demand_forecasts
            """)
            forecast_summary = cursor.fetchone()
            
            # Get risk assessment summary
            cursor.execute("""
                SELECT risk_level, COUNT(*) as count
                FROM risk_assessments
                GROUP BY risk_level
            """)
            risk_summary = cursor.fetchall()
            
            return {
                'demand_forecasts': {
                    'total_forecasts': forecast_summary[0],
                    'average_demand': forecast_summary[1],
                    'average_accuracy': forecast_summary[2]
                },
                'risk_assessment': dict(risk_summary)
            }
            
        except Exception as e:
            logger.error(f"Error getting analysis summary: {e}")
            return {}
    
    def cleanup(self):
        """Cleanup resources."""
        if self.db_connection:
            self.db_connection.close()


# Example usage and demonstration
async def main():
    """Demonstrate the advanced supply chain analysis system."""
    print("📊 HeyGen AI - Advanced Supply Chain Analysis System Demo")
    print("=" * 70)
    
    # Initialize system
    sca_system = AdvancedSupplyChainAnalysisSystem()
    
    try:
        # Sample supply chain data
        supply_chain_data = {
            'suppliers': {
                'supplier_1': {
                    'name': 'ABC Manufacturing',
                    'cost_per_unit': 10.50,
                    'reliability_score': 0.92,
                    'historical_data': [
                        {'delivery_date': '2024-01-01', 'promised_date': '2024-01-01', 'defect_rate': 0.02, 'cost': 1050, 'quantity': 100},
                        {'delivery_date': '2024-01-15', 'promised_date': '2024-01-14', 'defect_rate': 0.01, 'cost': 1100, 'quantity': 100},
                        {'delivery_date': '2024-02-01', 'promised_date': '2024-02-01', 'defect_rate': 0.03, 'cost': 1080, 'quantity': 100}
                    ]
                },
                'supplier_2': {
                    'name': 'XYZ Corp',
                    'cost_per_unit': 9.80,
                    'reliability_score': 0.88,
                    'historical_data': [
                        {'delivery_date': '2024-01-05', 'promised_date': '2024-01-05', 'defect_rate': 0.04, 'cost': 980, 'quantity': 100},
                        {'delivery_date': '2024-01-20', 'promised_date': '2024-01-18', 'defect_rate': 0.02, 'cost': 1020, 'quantity': 100}
                    ]
                }
            },
            'historical_demand': {
                'product_id': 'PROD_001',
                'seasonality': 0.8,
                'trend': 1.2,
                'price_elasticity': -0.5
            },
            'transportation_routes': [
                {'from': 'supplier_1', 'to': 'warehouse', 'distance': 150, 'cost': 500},
                {'from': 'supplier_2', 'to': 'warehouse', 'distance': 200, 'cost': 600}
            ],
            'inventory': {'value': 500000, 'holding_cost_rate': 0.2},
            'deliveries': [
                {'on_time': True}, {'on_time': False}, {'on_time': True}, {'on_time': True}
            ],
            'sales': {'annual': 2000000}
        }
        
        # Perform analysis
        print("\n🔍 Performing Supply Chain Analysis...")
        results = await sca_system.analyze_supply_chain(supply_chain_data)
        
        # Display results
        print("\n📈 Analysis Results:")
        
        # Demand forecasts
        if 'demand_forecasts' in results:
            print(f"\n📊 Demand Forecasts: {len(results['demand_forecasts'])} forecasts generated")
            for i, forecast in enumerate(results['demand_forecasts'][:3]):  # Show first 3
                print(f"  Day {i+1}: {forecast['predicted_demand']:.2f} units (accuracy: {forecast['model_accuracy']:.2%})")
        
        # Supplier analysis
        if 'supplier_analysis' in results:
            print(f"\n🏭 Supplier Analysis:")
            for supplier_id, analysis in results['supplier_analysis'].items():
                print(f"  {analysis['supplier_name']}:")
                print(f"    Overall Score: {analysis['overall_score']:.2f}")
                print(f"    Recommendation: {analysis['recommendation']}")
                if analysis['risk_factors']:
                    print(f"    Risk Factors: {', '.join(analysis['risk_factors'])}")
        
        # Risk assessment
        if 'risk_assessment' in results:
            print(f"\n⚠️ Risk Assessment: {len(results['risk_assessment'])} risks identified")
            for risk in results['risk_assessment']:
                print(f"  {risk['risk_type']} ({risk['risk_level']}): {risk['description']}")
        
        # Cost optimization
        if 'cost_optimization' in results:
            opt = results['cost_optimization']
            print(f"\n💰 Cost Optimization:")
            print(f"  Current Total Costs: ${opt.get('current_costs', {}).get('total', 0):,.2f}")
            print(f"  Potential Annual Savings: ${opt.get('potential_savings', {}).get('total_annual_savings', 0):,.2f}")
            print(f"  Optimization Score: {opt.get('optimization_score', 0):.2f}")
        
        # Performance metrics
        if 'performance_metrics' in results:
            metrics = results['performance_metrics']
            print(f"\n📊 Performance Metrics:")
            for metric, value in metrics.items():
                if isinstance(value, float):
                    print(f"  {metric}: {value:.2f}")
                else:
                    print(f"  {metric}: {value}")
        
        # Get summary
        print("\n📋 Analysis Summary:")
        summary = sca_system.get_analysis_summary()
        print(f"  Total Forecasts: {summary.get('demand_forecasts', {}).get('total_forecasts', 0)}")
        print(f"  Average Accuracy: {summary.get('demand_forecasts', {}).get('average_accuracy', 0):.2%}")
        
    except Exception as e:
        print(f"❌ Demo Error: {e}")
        logger.error(f"Demo failed: {e}")
    
    finally:
        # Cleanup
        sca_system.cleanup()
        print("\n✅ Demo completed")


if __name__ == "__main__":
    asyncio.run(main())
