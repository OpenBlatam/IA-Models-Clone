"""
Complete System Example - AI History Analyzer
============================================

This example demonstrates how to use the complete AI History Analysis System
including all components: analyzer, ML predictor, alerts, dashboard, and integration.

Features demonstrated:
- System initialization and configuration
- Performance tracking and analysis
- ML-based predictions and anomaly detection
- Intelligent alerts and notifications
- Real-time dashboard integration
- Workflow chain optimization
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Import all system components
from ..ai_history_analyzer import (
    get_ai_history_analyzer, AIHistoryAnalyzer,
    ModelType, PerformanceMetric
)
from ..config import get_ai_history_config, AIHistoryConfig
from ..ml_predictor import get_ml_predictor, MLPredictor
from ..intelligent_alerts import (
    get_intelligent_alert_system, IntelligentAlertSystem,
    AlertRule, AlertSeverity, AlertType, NotificationChannel
)
from ..realtime_dashboard import get_realtime_dashboard, RealtimeDashboard
from ..integration_system import get_integration_system, AIHistoryIntegrationSystem
from ..comprehensive_system import (
    get_comprehensive_system, ComprehensiveAISystem,
    SystemConfiguration
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CompleteSystemExample:
    """Example demonstrating the complete AI History Analysis System"""
    
    def __init__(self):
        self.system: ComprehensiveAISystem = None
        self.analyzer: AIHistoryAnalyzer = None
        self.ml_predictor: MLPredictor = None
        self.alert_system: IntelligentAlertSystem = None
        self.dashboard: RealtimeDashboard = None
        self.integration: AIHistoryIntegrationSystem = None
    
    async def initialize_system(self):
        """Initialize the complete system"""
        logger.info("🚀 Initializing Complete AI History Analysis System...")
        
        # Create system configuration
        config = SystemConfiguration(
            enable_api=True,
            enable_dashboard=True,
            enable_ml_predictor=True,
            enable_alerts=True,
            enable_integration=True,
            api_port=8002,
            dashboard_port=8003,
            ml_model_storage_path="ml_models",
            log_level="INFO"
        )
        
        # Initialize comprehensive system
        self.system = get_comprehensive_system(config)
        await self.system.initialize()
        
        # Get component instances
        self.analyzer = self.system.analyzer
        self.ml_predictor = self.system.ml_predictor
        self.alert_system = self.system.alert_system
        self.dashboard = self.system.dashboard
        self.integration = self.system.integration_system
        
        logger.info("✅ System initialized successfully")
    
    async def setup_notification_channels(self):
        """Setup notification channels for alerts"""
        logger.info("📧 Setting up notification channels...")
        
        # Dashboard channel (always enabled)
        dashboard_channel = NotificationChannel(
            channel_id="dashboard",
            channel_type="dashboard",
            name="Dashboard Notifications",
            configuration={"show_popup": True, "sound": True}
        )
        self.alert_system.add_notification_channel(dashboard_channel)
        
        # Email channel (example configuration)
        email_channel = NotificationChannel(
            channel_id="email_admin",
            channel_type="email",
            name="Admin Email",
            configuration={
                "from_email": "alerts@ai-system.com",
                "to_email": "admin@ai-system.com",
                "smtp_server": "smtp.gmail.com",
                "smtp_port": 587
            }
        )
        self.alert_system.add_notification_channel(email_channel)
        
        # Slack channel (example configuration)
        slack_channel = NotificationChannel(
            channel_id="slack_alerts",
            channel_type="slack",
            name="Slack Alerts",
            configuration={
                "webhook_url": "https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK",
                "channel": "#ai-alerts",
                "username": "AI System Bot"
            }
        )
        self.alert_system.add_notification_channel(slack_channel)
        
        logger.info("✅ Notification channels configured")
    
    async def setup_custom_alert_rules(self):
        """Setup custom alert rules"""
        logger.info("🚨 Setting up custom alert rules...")
        
        # Custom quality rule for GPT-4
        gpt4_quality_rule = AlertRule(
            rule_id="gpt4_quality_monitoring",
            name="GPT-4 Quality Monitoring",
            description="Monitor GPT-4 quality score specifically",
            alert_type=AlertType.PERFORMANCE_DEGRADATION,
            severity=AlertSeverity.WARNING,
            model_name="gpt-4",
            metric=PerformanceMetric.QUALITY_SCORE,
            threshold_value=0.8,
            threshold_operator="less_than",
            time_window_minutes=30,
            cooldown_minutes=15,
            notification_channels=["dashboard", "email_admin"]
        )
        self.alert_system.add_alert_rule(gpt4_quality_rule)
        
        # Response time rule for Claude
        claude_response_rule = AlertRule(
            rule_id="claude_response_time",
            name="Claude Response Time Alert",
            description="Alert when Claude response time is too high",
            alert_type=AlertType.THRESHOLD_BREACH,
            severity=AlertSeverity.ERROR,
            model_name="claude-3-sonnet",
            metric=PerformanceMetric.RESPONSE_TIME,
            threshold_value=15.0,
            threshold_operator="greater_than",
            time_window_minutes=15,
            cooldown_minutes=10,
            notification_channels=["dashboard", "slack_alerts"]
        )
        self.alert_system.add_alert_rule(claude_response_rule)
        
        # Anomaly detection rule
        anomaly_rule = AlertRule(
            rule_id="anomaly_detection",
            name="Anomaly Detection",
            description="Detect anomalies in model performance",
            alert_type=AlertType.ANOMALY_DETECTED,
            severity=AlertSeverity.WARNING,
            metric=PerformanceMetric.QUALITY_SCORE,
            time_window_minutes=60,
            cooldown_minutes=30,
            notification_channels=["dashboard", "email_admin"]
        )
        self.alert_system.add_alert_rule(anomaly_rule)
        
        logger.info("✅ Custom alert rules configured")
    
    async def simulate_performance_data(self):
        """Simulate performance data for demonstration"""
        logger.info("📊 Simulating performance data...")
        
        models = ["gpt-4", "claude-3-sonnet", "gemini-1.5-pro"]
        metrics = [
            PerformanceMetric.QUALITY_SCORE,
            PerformanceMetric.RESPONSE_TIME,
            PerformanceMetric.COST_EFFICIENCY,
            PerformanceMetric.TOKEN_EFFICIENCY
        ]
        
        # Simulate 30 days of data
        for day in range(30):
            for model in models:
                for metric in metrics:
                    # Simulate realistic performance values
                    if metric == PerformanceMetric.QUALITY_SCORE:
                        base_value = 0.85 if model == "gpt-4" else 0.82
                        value = base_value + (0.1 * (0.5 - 0.5))  # Random variation
                    elif metric == PerformanceMetric.RESPONSE_TIME:
                        base_value = 2.5 if model == "gpt-4" else 3.0
                        value = base_value + (1.0 * (0.5 - 0.5))  # Random variation
                    elif metric == PerformanceMetric.COST_EFFICIENCY:
                        base_value = 0.75 if model == "gpt-4" else 0.80
                        value = base_value + (0.1 * (0.5 - 0.5))  # Random variation
                    elif metric == PerformanceMetric.TOKEN_EFFICIENCY:
                        base_value = 0.80 if model == "gpt-4" else 0.78
                        value = base_value + (0.1 * (0.5 - 0.5))  # Random variation
                    
                    # Record performance
                    self.analyzer.record_performance(
                        model_name=model,
                        model_type=ModelType.TEXT_GENERATION,
                        metric=metric,
                        value=value,
                        context=f"Simulated data for day {day}",
                        timestamp=datetime.now() - timedelta(days=30-day)
                    )
        
        logger.info("✅ Performance data simulation completed")
    
    async def demonstrate_analysis_capabilities(self):
        """Demonstrate analysis capabilities"""
        logger.info("🔍 Demonstrating analysis capabilities...")
        
        # Get performance summary for GPT-4
        gpt4_summary = self.analyzer.get_performance_summary("gpt-4", days=30)
        logger.info(f"📈 GPT-4 Performance Summary: {json.dumps(gpt4_summary, indent=2, default=str)}")
        
        # Analyze trends
        quality_trend = self.analyzer.analyze_trends(
            "gpt-4", PerformanceMetric.QUALITY_SCORE, days=30
        )
        if quality_trend:
            logger.info(f"📊 Quality Trend Analysis: {quality_trend.trend_direction} (strength: {quality_trend.trend_strength:.3f})")
        
        # Compare models
        comparison = self.analyzer.compare_models(
            "gpt-4", "claude-3-sonnet", PerformanceMetric.QUALITY_SCORE, days=30
        )
        if comparison:
            winner = comparison.model_a if comparison.comparison_score > 0 else comparison.model_b
            logger.info(f"🏆 Model Comparison: {winner} wins with confidence {comparison.confidence:.3f}")
        
        # Get model rankings
        rankings = self.analyzer.get_model_rankings(PerformanceMetric.QUALITY_SCORE, days=30)
        logger.info(f"🥇 Model Rankings: {[r['model_name'] for r in rankings[:3]]}")
        
        logger.info("✅ Analysis capabilities demonstrated")
    
    async def demonstrate_ml_predictions(self):
        """Demonstrate ML prediction capabilities"""
        logger.info("🤖 Demonstrating ML prediction capabilities...")
        
        # Train a performance prediction model
        training_result = await self.ml_predictor.train_performance_prediction_model(
            model_name="gpt-4",
            metric=PerformanceMetric.QUALITY_SCORE,
            algorithm="random_forest"
        )
        
        if training_result["success"]:
            logger.info(f"🎯 Model Training: R² Score = {training_result['r2_score']:.3f}")
            
            # Make a prediction
            prediction = await self.ml_predictor.predict_performance(
                model_name="gpt-4",
                metric=PerformanceMetric.QUALITY_SCORE
            )
            
            if prediction:
                logger.info(f"🔮 Performance Prediction: {prediction.predicted_value:.3f} (confidence: {prediction.confidence:.3f})")
        
        # Detect anomalies
        anomalies = await self.ml_predictor.detect_anomalies(
            "gpt-4", PerformanceMetric.QUALITY_SCORE, days=30
        )
        logger.info(f"🚨 Anomalies Detected: {len(anomalies)}")
        
        # Generate optimization recommendations
        recommendations = await self.ml_predictor.generate_optimization_recommendations(
            "gpt-4", PerformanceMetric.QUALITY_SCORE
        )
        logger.info(f"💡 Optimization Recommendations: {len(recommendations)}")
        
        logger.info("✅ ML prediction capabilities demonstrated")
    
    async def demonstrate_alert_system(self):
        """Demonstrate alert system capabilities"""
        logger.info("🚨 Demonstrating alert system capabilities...")
        
        # Start alert monitoring
        await self.alert_system.start_monitoring()
        
        # Simulate some performance issues to trigger alerts
        self.analyzer.record_performance(
            model_name="gpt-4",
            model_type=ModelType.TEXT_GENERATION,
            metric=PerformanceMetric.QUALITY_SCORE,
            value=0.6,  # Low quality to trigger alert
            context="Simulated quality degradation"
        )
        
        # Wait for alerts to be processed
        await asyncio.sleep(5)
        
        # Get active alerts
        active_alerts = self.alert_system.get_active_alerts()
        logger.info(f"🔔 Active Alerts: {len(active_alerts)}")
        
        # Get alert statistics
        alert_stats = self.alert_system.get_alert_statistics()
        logger.info(f"📊 Alert Statistics: {alert_stats}")
        
        logger.info("✅ Alert system capabilities demonstrated")
    
    async def demonstrate_dashboard_integration(self):
        """Demonstrate dashboard integration"""
        logger.info("📈 Demonstrating dashboard integration...")
        
        # Get current metrics
        metrics = self.dashboard._collect_dashboard_metrics()
        logger.info(f"📊 Dashboard Metrics: {metrics.active_models} models, {metrics.total_measurements} measurements")
        
        # Get model performance snapshots
        await self.dashboard._update_performance_snapshots()
        snapshots = self.dashboard.performance_snapshots
        logger.info(f"📈 Performance Snapshots: {len(snapshots)} models tracked")
        
        logger.info("✅ Dashboard integration demonstrated")
    
    async def demonstrate_workflow_integration(self):
        """Demonstrate workflow chain integration"""
        logger.info("🔗 Demonstrating workflow chain integration...")
        
        # Get performance insights
        insights = self.integration.get_performance_insights()
        logger.info(f"💡 Performance Insights: {len(insights)} insights generated")
        
        # Get model recommendations
        recommendation = await self.integration.get_model_recommendation(
            task_type="document_generation",
            content_size=5000,
            priority="balanced"
        )
        
        if recommendation:
            logger.info(f"🎯 Model Recommendation: {recommendation.recommended_model}")
            logger.info(f"📝 Reasoning: {recommendation.reasoning}")
        
        logger.info("✅ Workflow integration demonstrated")
    
    async def demonstrate_system_health_monitoring(self):
        """Demonstrate system health monitoring"""
        logger.info("🏥 Demonstrating system health monitoring...")
        
        # Get system status
        status = self.system.get_system_status()
        if status:
            logger.info(f"💚 System Status: {status.overall_status}")
            logger.info(f"📊 Components: {len(status.components)} components monitored")
            logger.info(f"⏱️ Uptime: {status.system_uptime:.1f} seconds")
        
        # Get performance history
        performance_history = self.system.get_performance_history(limit=10)
        logger.info(f"📈 Performance History: {len(performance_history)} recent metrics")
        
        logger.info("✅ System health monitoring demonstrated")
    
    async def run_complete_demonstration(self):
        """Run the complete system demonstration"""
        logger.info("🎬 Starting Complete System Demonstration...")
        
        try:
            # Initialize system
            await self.initialize_system()
            
            # Setup components
            await self.setup_notification_channels()
            await self.setup_custom_alert_rules()
            
            # Simulate data
            await self.simulate_performance_data()
            
            # Demonstrate capabilities
            await self.demonstrate_analysis_capabilities()
            await self.demonstrate_ml_predictions()
            await self.demonstrate_alert_system()
            await self.demonstrate_dashboard_integration()
            await self.demonstrate_workflow_integration()
            await self.demonstrate_system_health_monitoring()
            
            logger.info("🎉 Complete system demonstration finished successfully!")
            
            # Keep system running for a while to show real-time features
            logger.info("⏳ Keeping system running for 60 seconds to demonstrate real-time features...")
            await asyncio.sleep(60)
            
        except Exception as e:
            logger.error(f"❌ Error during demonstration: {str(e)}")
            raise
        
        finally:
            # Cleanup
            if self.alert_system:
                await self.alert_system.stop_monitoring()
            if self.system:
                await self.system.shutdown()
            
            logger.info("🧹 System cleanup completed")


async def main():
    """Main function to run the complete system example"""
    example = CompleteSystemExample()
    await example.run_complete_demonstration()


if __name__ == "__main__":
    asyncio.run(main())

























