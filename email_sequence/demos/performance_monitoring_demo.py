from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import gradio as gr
import asyncio
import logging
import json
import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import random
import time
import threading
from collections import deque
import sys
from typing import Any, List, Dict, Optional
"""
Real-time Performance Monitoring Demo

Interactive demo showcasing real-time model performance monitoring,
system health tracking, and predictive analytics for the email sequence system.
"""


# Add the parent directory to the path
sys.path.append(str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PerformanceMonitoringDemo:
    """Real-time performance monitoring demo"""
    
    def __init__(self) -> Any:
        self.monitoring_data = {
            "timestamps": deque(maxlen=1000),
            "model_performance": deque(maxlen=1000),
            "system_health": deque(maxlen=1000),
            "user_activity": deque(maxlen=1000),
            "error_rates": deque(maxlen=1000),
            "response_times": deque(maxlen=1000)
        }
        self.is_monitoring = False
        self.monitoring_thread = None
        
        logger.info("Performance Monitoring Demo initialized")
    
    def create_performance_dashboard(self) -> gr.Blocks:
        """Create real-time performance monitoring dashboard"""
        
        with gr.Blocks(title="Real-time Performance Monitoring", theme=gr.themes.Soft()) as demo:
            
            gr.Markdown("""
            # 📊 Real-time Performance Monitoring Dashboard
            
            Monitor system performance, model accuracy, and user activity in real-time.
            Track key metrics and receive alerts for performance issues.
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Monitoring Controls")
                    
                    # Start/Stop monitoring
                    monitoring_btn = gr.Button("🟢 Start Monitoring", variant="primary", size="lg")
                    
                    # Monitoring interval
                    update_interval = gr.Slider(
                        minimum=1,
                        maximum=30,
                        value=5,
                        step=1,
                        label="Update Interval (seconds)",
                        info="How often to update metrics"
                    )
                    
                    # Alert thresholds
                    performance_threshold = gr.Slider(
                        minimum=0.5,
                        maximum=1.0,
                        value=0.8,
                        step=0.05,
                        label="Performance Threshold",
                        info="Alert when performance drops below this level"
                    )
                    
                    response_time_threshold = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=3,
                        step=0.5,
                        label="Response Time Threshold (seconds)",
                        info="Alert when response time exceeds this level"
                    )
                    
                    # System status
                    system_status = gr.JSON(
                        label="System Status",
                        value={"status": "Stopped", "last_update": None}
                    )
                
                with gr.Column(scale=2):
                    gr.Markdown("### Real-time Metrics")
                    
                    # Performance chart
                    performance_chart = gr.Plot(
                        label="Model Performance Over Time"
                    )
                    
                    # System health chart
                    health_chart = gr.Plot(
                        label="System Health Metrics"
                    )
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Key Metrics")
                    
                    # Current metrics
                    current_metrics = gr.JSON(
                        label="Current Metrics"
                    )
                    
                    # Performance alerts
                    alerts = gr.JSON(
                        label="Active Alerts"
                    )
                
                with gr.Column(scale=1):
                    gr.Markdown("### User Activity")
                    
                    # User activity chart
                    activity_chart = gr.Plot(
                        label="User Activity Heatmap"
                    )
                    
                    # Activity summary
                    activity_summary = gr.JSON(
                        label="Activity Summary"
                    )
            
            # Monitoring control function
            def toggle_monitoring(
                button_text: str,
                update_interval: int,
                performance_threshold: float,
                response_time_threshold: float
            ) -> Tuple[str, Dict, List[go.Figure], Dict, Dict, Dict]:
                
                if "Start" in button_text:
                    # Start monitoring
                    self.start_monitoring(update_interval)
                    new_button_text = "🔴 Stop Monitoring"
                    status = {"status": "Running", "last_update": datetime.now().isoformat()}
                else:
                    # Stop monitoring
                    self.stop_monitoring()
                    new_button_text = "🟢 Start Monitoring"
                    status = {"status": "Stopped", "last_update": datetime.now().isoformat()}
                
                # Get current data
                current_data = self.get_current_metrics()
                alerts_data = self.get_alerts(performance_threshold, response_time_threshold)
                charts = self.create_monitoring_charts()
                activity_data = self.get_activity_summary()
                
                return new_button_text, status, charts, current_data, alerts_data, activity_data
            
            # Update function for real-time updates
            def update_dashboard() -> Tuple[Dict, List[go.Figure], Dict, Dict, Dict]:
                
                current_data = self.get_current_metrics()
                alerts_data = self.get_alerts(0.8, 3.0)  # Default thresholds
                charts = self.create_monitoring_charts()
                activity_data = self.get_activity_summary()
                status = {"status": "Running" if self.is_monitoring else "Stopped", "last_update": datetime.now().isoformat()}
                
                return status, charts, current_data, alerts_data, activity_data
            
            # Connect the monitoring button
            monitoring_btn.click(
                fn=toggle_monitoring,
                inputs=[
                    monitoring_btn,
                    update_interval,
                    performance_threshold,
                    response_time_threshold
                ],
                outputs=[
                    monitoring_btn,
                    system_status,
                    performance_chart,
                    health_chart,
                    current_metrics,
                    alerts,
                    activity_chart,
                    activity_summary
                ]
            )
            
            # Auto-refresh every 5 seconds when monitoring is active
            demo.load(update_dashboard, outputs=[
                system_status,
                performance_chart,
                health_chart,
                current_metrics,
                alerts,
                activity_chart,
                activity_summary
            ])
        
        return demo
    
    def start_monitoring(self, interval: int):
        """Start real-time monitoring"""
        
        if not self.is_monitoring:
            self.is_monitoring = True
            self.monitoring_thread = threading.Thread(
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                target=self._monitoring_loop,
                args=(interval,),
                daemon=True
            )
            self.monitoring_thread.start()
            logger.info(f"Started monitoring with {interval}s interval")
    
    def stop_monitoring(self) -> Any:
        """Stop real-time monitoring"""
        
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=1)
        logger.info("Stopped monitoring")
    
    def _monitoring_loop(self, interval: int):
        """Main monitoring loop"""
        
        while self.is_monitoring:
            try:
                # Generate new monitoring data
                self._generate_monitoring_data()
                time.sleep(interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(interval)
    
    def _generate_monitoring_data(self) -> Any:
        """Generate new monitoring data points"""
        
        timestamp = datetime.now()
        
        # Model performance (simulated)
        performance = {
            "accuracy": round(random.uniform(0.75, 0.95), 3),
            "precision": round(random.uniform(0.70, 0.90), 3),
            "recall": round(random.uniform(0.65, 0.85), 3),
            "f1_score": round(random.uniform(0.70, 0.90), 3)
        }
        
        # System health metrics
        health = {
            "cpu_usage": round(random.uniform(20, 80), 1),
            "memory_usage": round(random.uniform(30, 85), 1),
            "gpu_usage": round(random.uniform(10, 90), 1),
            "disk_usage": round(random.uniform(40, 75), 1),
            "network_latency": round(random.uniform(10, 100), 1)
        }
        
        # User activity
        activity = {
            "active_users": random.randint(5, 50),
            "requests_per_minute": random.randint(10, 200),
            "sequences_generated": random.randint(1, 20),
            "evaluations_performed": random.randint(5, 30)
        }
        
        # Error rates
        error_rate = round(random.uniform(0.001, 0.05), 4)
        
        # Response times
        response_time = round(random.uniform(0.5, 5.0), 2)
        
        # Store data
        self.monitoring_data["timestamps"].append(timestamp)
        self.monitoring_data["model_performance"].append(performance)
        self.monitoring_data["system_health"].append(health)
        self.monitoring_data["user_activity"].append(activity)
        self.monitoring_data["error_rates"].append(error_rate)
        self.monitoring_data["response_times"].append(response_time)
    
    def get_current_metrics(self) -> Dict:
        """Get current metrics"""
        
        if not self.monitoring_data["timestamps"]:
            return {"error": "No monitoring data available"}
        
        latest_idx = -1
        
        return {
            "timestamp": self.monitoring_data["timestamps"][latest_idx].isoformat(),
            "model_performance": self.monitoring_data["model_performance"][latest_idx],
            "system_health": self.monitoring_data["system_health"][latest_idx],
            "user_activity": self.monitoring_data["user_activity"][latest_idx],
            "error_rate": self.monitoring_data["error_rates"][latest_idx],
            "response_time": self.monitoring_data["response_times"][latest_idx]
        }
    
    def get_alerts(self, performance_threshold: float, response_time_threshold: float) -> List[Dict]:
        """Get current alerts based on thresholds"""
        
        alerts = []
        
        if not self.monitoring_data["model_performance"]:
            return alerts
        
        latest_performance = self.monitoring_data["model_performance"][-1]
        latest_response_time = self.monitoring_data["response_times"][-1]
        latest_health = self.monitoring_data["system_health"][-1]
        
        # Performance alerts
        if latest_performance["accuracy"] < performance_threshold:
            alerts.append({
                "type": "performance",
                "severity": "warning",
                "message": f"Model accuracy ({latest_performance['accuracy']:.3f}) below threshold ({performance_threshold})",
                "timestamp": datetime.now().isoformat()
            })
        
        # Response time alerts
        if latest_response_time > response_time_threshold:
            alerts.append({
                "type": "response_time",
                "severity": "error",
                "message": f"Response time ({latest_response_time}s) above threshold ({response_time_threshold}s)",
                "timestamp": datetime.now().isoformat()
            })
        
        # System health alerts
        if latest_health["cpu_usage"] > 80:
            alerts.append({
                "type": "system_health",
                "severity": "warning",
                "message": f"High CPU usage: {latest_health['cpu_usage']}%",
                "timestamp": datetime.now().isoformat()
            })
        
        if latest_health["memory_usage"] > 85:
            alerts.append({
                "type": "system_health",
                "severity": "error",
                "message": f"High memory usage: {latest_health['memory_usage']}%",
                "timestamp": datetime.now().isoformat()
            })
        
        return alerts
    
    def create_monitoring_charts(self) -> List[go.Figure]:
        """Create monitoring charts"""
        
        if not self.monitoring_data["timestamps"]:
            return [go.Figure(), go.Figure()]
        
        # Performance chart
        timestamps = list(self.monitoring_data["timestamps"])
        performance_data = list(self.monitoring_data["model_performance"])
        
        fig1 = go.Figure()
        
        metrics = ["accuracy", "precision", "recall", "f1_score"]
        colors = ["blue", "red", "green", "orange"]
        
        for metric, color in zip(metrics, colors):
            values = [p[metric] for p in performance_data]
            fig1.add_trace(go.Scatter(
                x=timestamps,
                y=values,
                mode='lines',
                name=metric.title(),
                line=dict(color=color)
            ))
        
        fig1.update_layout(
            title="Model Performance Over Time",
            xaxis_title="Time",
            yaxis_title="Score",
            hovermode='x unified'
        )
        
        # System health chart
        health_data = list(self.monitoring_data["system_health"])
        
        fig2 = go.Figure()
        
        health_metrics = ["cpu_usage", "memory_usage", "gpu_usage", "disk_usage"]
        health_colors = ["red", "blue", "green", "orange"]
        
        for metric, color in zip(health_metrics, health_colors):
            values = [h[metric] for h in health_data]
            fig2.add_trace(go.Scatter(
                x=timestamps,
                y=values,
                mode='lines',
                name=metric.replace('_', ' ').title(),
                line=dict(color=color)
            ))
        
        fig2.update_layout(
            title="System Health Metrics",
            xaxis_title="Time",
            yaxis_title="Usage (%)",
            hovermode='x unified'
        )
        
        return [fig1, fig2]
    
    def get_activity_summary(self) -> Dict:
        """Get user activity summary"""
        
        if not self.monitoring_data["user_activity"]:
            return {"error": "No activity data available"}
        
        activity_data = list(self.monitoring_data["user_activity"])
        
        # Calculate averages
        avg_active_users = np.mean([a["active_users"] for a in activity_data])
        avg_requests = np.mean([a["requests_per_minute"] for a in activity_data])
        avg_sequences = np.mean([a["sequences_generated"] for a in activity_data])
        avg_evaluations = np.mean([a["evaluations_performed"] for a in activity_data])
        
        return {
            "average_active_users": round(avg_active_users, 1),
            "average_requests_per_minute": round(avg_requests, 1),
            "average_sequences_generated": round(avg_sequences, 1),
            "average_evaluations_performed": round(avg_evaluations, 1),
            "total_data_points": len(activity_data)
        }
    
    def create_predictive_analytics_demo(self) -> gr.Blocks:
        """Create predictive analytics demo"""
        
        with gr.Blocks(title="Predictive Analytics Demo", theme=gr.themes.Soft()) as demo:
            
            gr.Markdown("""
            # 🔮 Predictive Analytics Demo
            
            Explore predictive analytics for email sequence performance,
            user behavior forecasting, and system optimization predictions.
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Prediction Controls")
                    
                    # Prediction type
                    prediction_type = gr.Dropdown(
                        choices=[
                            "Sequence Performance",
                            "User Engagement",
                            "System Load",
                            "Revenue Forecasting",
                            "Churn Prediction"
                        ],
                        value="Sequence Performance",
                        label="Prediction Type",
                        info="Choose what to predict"
                    )
                    
                    # Prediction horizon
                    prediction_horizon = gr.Slider(
                        minimum=1,
                        maximum=30,
                        value=7,
                        step=1,
                        label="Prediction Horizon (days)",
                        info="How far into the future to predict"
                    )
                    
                    # Confidence level
                    confidence_level = gr.Slider(
                        minimum=0.8,
                        maximum=0.99,
                        value=0.95,
                        step=0.01,
                        label="Confidence Level",
                        info="Prediction confidence interval"
                    )
                    
                    # Generate prediction button
                    predict_btn = gr.Button("🔮 Generate Prediction", variant="primary", size="lg")
                
                with gr.Column(scale=2):
                    gr.Markdown("### Prediction Results")
                    
                    # Prediction chart
                    prediction_chart = gr.Plot(
                        label="Prediction Visualization"
                    )
                    
                    # Prediction details
                    prediction_details = gr.JSON(
                        label="Prediction Details"
                    )
                    
                    # Prediction insights
                    prediction_insights = gr.Markdown(
                        label="Prediction Insights"
                    )
            
            # Prediction function
            def generate_prediction(
                prediction_type: str,
                horizon: int,
                confidence: float
            ) -> Tuple[go.Figure, Dict, str]:
                
                try:
                    # Generate prediction data
                    prediction_data = self._generate_prediction_data(
                        prediction_type, horizon, confidence
                    )
                    
                    # Create prediction chart
                    chart = self._create_prediction_chart(prediction_data)
                    
                    # Create insights
                    insights = self._create_prediction_insights(prediction_data)
                    
                    return chart, prediction_data, insights
                    
                except Exception as e:
                    logger.error(f"Error generating prediction: {e}")
                    return go.Figure(), {"error": str(e)}, f"Error: {str(e)}"
            
            # Connect the predict button
            predict_btn.click(
                fn=generate_prediction,
                inputs=[prediction_type, prediction_horizon, confidence_level],
                outputs=[prediction_chart, prediction_details, prediction_insights]
            )
        
        return demo
    
    def _generate_prediction_data(
        self,
        prediction_type: str,
        horizon: int,
        confidence: float
    ) -> Dict:
        """Generate prediction data"""
        
        # Generate historical data
        historical_dates = [datetime.now() - timedelta(days=i) for i in range(30, 0, -1)]
        
        if prediction_type == "Sequence Performance":
            # Simulate sequence performance trends
            base_performance = 0.8
            trend = np.linspace(0, 0.1, 30)  # Slight upward trend
            noise = np.random.normal(0, 0.02, 30)
            historical_values = base_performance + trend + noise
            
            # Generate future predictions
            future_dates = [datetime.now() + timedelta(days=i) for i in range(1, horizon + 1)]
            future_trend = np.linspace(0.1, 0.15, horizon)  # Continued improvement
            future_noise = np.random.normal(0, 0.03, horizon)
            future_values = base_performance + 0.1 + future_trend + future_noise
            
            # Confidence intervals
            confidence_interval = 0.05 * (1 - confidence)  # Wider for lower confidence
            upper_bound = future_values + confidence_interval
            lower_bound = future_values - confidence_interval
            
            return {
                "type": prediction_type,
                "historical_dates": [d.strftime("%Y-%m-%d") for d in historical_dates],
                "historical_values": historical_values.tolist(),
                "future_dates": [d.strftime("%Y-%m-%d") for d in future_dates],
                "future_values": future_values.tolist(),
                "upper_bound": upper_bound.tolist(),
                "lower_bound": lower_bound.tolist(),
                "confidence_level": confidence,
                "trend": "increasing",
                "reliability_score": round(random.uniform(0.7, 0.95), 3)
            }
        
        elif prediction_type == "User Engagement":
            # Simulate user engagement trends
            base_engagement = 0.6
            seasonal_pattern = 0.1 * np.sin(np.linspace(0, 4*np.pi, 30))  # Weekly pattern
            historical_values = base_engagement + seasonal_pattern + np.random.normal(0, 0.05, 30)
            
            future_dates = [datetime.now() + timedelta(days=i) for i in range(1, horizon + 1)]
            future_seasonal = 0.1 * np.sin(np.linspace(4*np.pi, 4*np.pi + horizon*np.pi/7, horizon))
            future_values = base_engagement + future_seasonal + np.random.normal(0, 0.05, horizon)
            
            confidence_interval = 0.08 * (1 - confidence)
            upper_bound = future_values + confidence_interval
            lower_bound = future_values - confidence_interval
            
            return {
                "type": prediction_type,
                "historical_dates": [d.strftime("%Y-%m-%d") for d in historical_dates],
                "historical_values": historical_values.tolist(),
                "future_dates": [d.strftime("%Y-%m-%d") for d in future_dates],
                "future_values": future_values.tolist(),
                "upper_bound": upper_bound.tolist(),
                "lower_bound": lower_bound.tolist(),
                "confidence_level": confidence,
                "trend": "stable",
                "reliability_score": round(random.uniform(0.75, 0.9), 3)
            }
        
        else:
            # Default prediction
            return {
                "type": prediction_type,
                "historical_dates": [d.strftime("%Y-%m-%d") for d in historical_dates],
                "historical_values": [random.uniform(0.5, 0.9) for _ in range(30)],
                "future_dates": [(datetime.now() + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(1, horizon + 1)],
                "future_values": [random.uniform(0.5, 0.9) for _ in range(horizon)],
                "upper_bound": [random.uniform(0.6, 0.95) for _ in range(horizon)],
                "lower_bound": [random.uniform(0.4, 0.85) for _ in range(horizon)],
                "confidence_level": confidence,
                "trend": "stable",
                "reliability_score": round(random.uniform(0.6, 0.8), 3)
            }
    
    def _create_prediction_chart(self, prediction_data: Dict) -> go.Figure:
        """Create prediction visualization chart"""
        
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=prediction_data["historical_dates"],
            y=prediction_data["historical_values"],
            mode='lines+markers',
            name='Historical',
            line=dict(color='blue', width=2)
        ))
        
        # Future predictions
        fig.add_trace(go.Scatter(
            x=prediction_data["future_dates"],
            y=prediction_data["future_values"],
            mode='lines+markers',
            name='Prediction',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        # Confidence interval
        fig.add_trace(go.Scatter(
            x=prediction_data["future_dates"] + prediction_data["future_dates"][::-1],
            y=prediction_data["upper_bound"] + prediction_data["lower_bound"][::-1],
            fill='toself',
            fillcolor='rgba(255,0,0,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name=f'{prediction_data["confidence_level"]:.0%} Confidence'
        ))
        
        fig.update_layout(
            title=f"Prediction: {prediction_data['type']}",
            xaxis_title="Date",
            yaxis_title="Value",
            hovermode='x unified',
            showlegend=True
        )
        
        return fig
    
    def _create_prediction_insights(self, prediction_data: Dict) -> str:
        """Create prediction insights report"""
        
        insights = f"# 🔮 Prediction Insights: {prediction_data['type']}\n\n"
        
        insights += f"**Prediction Horizon:** {len(prediction_data['future_dates'])} days\n"
        insights += f"**Confidence Level:** {prediction_data['confidence_level']:.0%}\n"
        insights += f"**Reliability Score:** {prediction_data['reliability_score']:.3f}\n\n"
        
        insights += "## Key Findings\n\n"
        
        if prediction_data['trend'] == 'increasing':
            insights += "📈 **Trend:** Upward trajectory predicted\n"
            insights += "✅ **Recommendation:** Continue current strategies\n"
        elif prediction_data['trend'] == 'decreasing':
            insights += "📉 **Trend:** Downward trajectory predicted\n"
            insights += "⚠️ **Recommendation:** Review and optimize strategies\n"
        else:
            insights += "➡️ **Trend:** Stable performance expected\n"
            insights += "🔄 **Recommendation:** Monitor for opportunities\n"
        
        insights += f"\n## Confidence Analysis\n\n"
        insights += f"The prediction has a {prediction_data['confidence_level']:.0%} confidence level, "
        insights += f"indicating {'high' if prediction_data['confidence_level'] > 0.9 else 'moderate'} reliability.\n\n"
        
        insights += "## Action Items\n\n"
        insights += "1. **Monitor Performance:** Track actual vs predicted values\n"
        insights += "2. **Adjust Strategies:** Modify approaches based on predictions\n"
        insights += "3. **Resource Planning:** Prepare for predicted changes\n"
        insights += "4. **Risk Management:** Plan for worst-case scenarios\n"
        
        return insights


def create_performance_demo_launcher():
    """Create launcher for performance monitoring demos"""
    
    demo = PerformanceMonitoringDemo()
    
    with gr.Blocks(title="Performance Monitoring Demos", theme=gr.themes.Soft()) as app:
        
        gr.Markdown("""
        # 📊 Performance Monitoring & Analytics Demos
        
        Real-time monitoring and predictive analytics for the email sequence system.
        Track performance, predict trends, and optimize system operations.
        """)
        
        with gr.Tabs():
            with gr.Tab("📈 Real-time Monitoring"):
                monitoring_demo = demo.create_performance_dashboard()
            
            with gr.Tab("🔮 Predictive Analytics"):
                predictive_demo = demo.create_predictive_analytics_demo()
        
        gr.Markdown("""
        ---
        
        **Note:** These demos use simulated data for demonstration purposes.
        In production, the system would use real monitoring data and actual predictions.
        """)
    
    return app


def main():
    """Launch the performance monitoring demos"""
    
    app = create_performance_demo_launcher()
    
    app.launch(
        server_name="0.0.0.0",
        server_port=7862,
        share=True,
        debug=True,
        show_error=True
    )


match __name__:
    case "__main__":
    main() 