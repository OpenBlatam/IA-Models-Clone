#!/usr/bin/env python3
"""
Advanced Predictive Interface for Facebook Posts Analysis v3.0
Next-generation Gradio interface with advanced AI capabilities
"""

import gradio as gr
import torch
import numpy as np
import json
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from pathlib import Path
import logging

# Import our enhanced components
from advanced_predictive_system import AdvancedPredictiveSystem, AdvancedPredictiveConfig
from enhanced_integrated_system import EnhancedIntegratedSystem, EnhancedIntegratedSystemConfig
from enhanced_performance_engine import EnhancedPerformanceOptimizationEngine, EnhancedPerformanceConfig


class AdvancedPredictiveInterface:
    """Advanced Gradio interface with predictive capabilities"""
    
    def __init__(self):
        self.predictive_system = None
        self.enhanced_system = None
        self.is_initialized = False
        self.demo_data = self._load_demo_data()
        
        # Performance tracking
        self.request_history = []
        self.performance_metrics = {}
        self.prediction_cache = {}
        
        # Initialize system
        self._initialize_system()
    
    def _initialize_system(self):
        """Initialize the advanced predictive system"""
        try:
            # Initialize predictive system
            predictive_config = AdvancedPredictiveConfig()
            self.predictive_system = AdvancedPredictiveSystem(predictive_config)
            
            # Initialize enhanced system
            enhanced_config = EnhancedIntegratedSystemConfig(
                environment="development",
                enable_ai_agents=True,
                enable_performance_monitoring=True,
                enable_health_checks=True,
                log_level="INFO"
            )
            self.enhanced_system = EnhancedIntegratedSystem(enhanced_config)
            self.enhanced_system.start()
            
            self.is_initialized = True
            logging.info("✅ Advanced Predictive System initialized successfully!")
            
        except Exception as e:
            logging.error(f"❌ Failed to initialize system: {e}")
            self.is_initialized = False
    
    def _load_demo_data(self) -> Dict[str, Any]:
        """Load demo data for the interface"""
        return {
            'sample_content': [
                "🚀 Discover our revolutionary AI-powered content optimization platform!",
                "💡 Transform your social media strategy with cutting-edge technology",
                "🔥 Boost engagement and reach with intelligent content recommendations",
                "📈 Maximize your ROI with data-driven content optimization",
                "🎯 Target the right audience with AI-powered insights",
                "🌟 Experience the future of social media marketing today!",
                "💪 Empower your brand with predictive analytics and AI insights",
                "🎨 Create compelling content that resonates with your audience"
            ],
            'content_types': ['Post', 'Story', 'Reel', 'Video', 'Image', 'Carousel', 'Live'],
            'audience_sizes': ['Small (1K-10K)', 'Medium (10K-100K)', 'Large (100K-1M)', 'Enterprise (1M+)'],
            'time_periods': ['Morning (6AM-12PM)', 'Afternoon (12PM-6PM)', 'Evening (6PM-12AM)', 'Night (12AM-6AM)'],
            'trending_topics': ['#AI', '#Innovation', '#Technology', '#Marketing', '#SocialMedia', '#DigitalTransformation'],
            'audience_demographics': {
                'age_ranges': ['18-24', '25-34', '35-44', '45-54', '55+'],
                'interests': ['Technology', 'Business', 'Entertainment', 'Sports', 'Fashion', 'Food'],
                'locations': ['United States', 'Europe', 'Asia', 'Latin America', 'Global']
            }
        }
    
    def create_interface(self) -> gr.Blocks:
        """Create the advanced predictive Gradio interface"""
        with gr.Blocks(
            title="🚀 Advanced Predictive Facebook Content Optimization System v3.0",
            theme=gr.themes.Soft(),
            css="""
            .gradio-container {
                max-width: 1400px !important;
                margin: 0 auto !important;
            }
            .header {
                text-align: center;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border-radius: 10px;
                margin-bottom: 20px;
            }
            .metric-card {
                background: white;
                padding: 15px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                margin: 10px 0;
            }
            .prediction-result {
                background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                color: white;
                padding: 20px;
                border-radius: 10px;
                margin: 15px 0;
            }
            .recommendation-box {
                background: #f8f9fa;
                border-left: 4px solid #007bff;
                padding: 15px;
                margin: 10px 0;
                border-radius: 5px;
            }
            """
        ) as interface:
            
            # Header
            with gr.Row():
                gr.HTML("""
                <div class="header">
                    <h1>🚀 Advanced Predictive Facebook Content Optimization System v3.0</h1>
                    <p>Next-generation AI-powered prediction and forecasting capabilities</p>
                </div>
                """)
            
            # Main tabs
            with gr.Tabs():
                
                # Tab 1: Advanced Content Analysis
                with gr.Tab("🔮 Advanced Content Analysis", id=1):
                    self._create_advanced_analysis_tab()
                
                # Tab 2: Predictive Analytics
                with gr.Tab("📊 Predictive Analytics", id=2):
                    self._create_predictive_analytics_tab()
                
                # Tab 3: Audience Intelligence
                with gr.Tab("🎯 Audience Intelligence", id=3):
                    self._create_audience_intelligence_tab()
                
                # Tab 4: Performance Forecasting
                with gr.Tab("📈 Performance Forecasting", id=4):
                    self._create_performance_forecasting_tab()
                
                # Tab 5: System Health & Metrics
                with gr.Tab("🏥 System Health & Metrics", id=5):
                    self._create_system_health_tab()
            
            # Footer
            with gr.Row():
                gr.HTML("""
                <div style="text-align: center; padding: 20px; color: #666;">
                    <p>Advanced Predictive System v3.0 | Powered by AI & Machine Learning</p>
                </div>
                """)
        
        return interface
    
    def _create_advanced_analysis_tab(self):
        """Create the advanced content analysis tab"""
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("## 🔮 Advanced Content Analysis")
                gr.Markdown("Analyze your Facebook content with next-generation AI capabilities")
                
                # Content input
                content_input = gr.Textbox(
                    label="📝 Enter your Facebook content",
                    placeholder="Paste your Facebook post content here...",
                    lines=5,
                    max_lines=10
                )
                
                # Analysis options
                with gr.Row():
                    enable_viral_prediction = gr.Checkbox(label="🔮 Viral Prediction", value=True)
                    enable_sentiment_analysis = gr.Checkbox(label="😊 Advanced Sentiment", value=True)
                    enable_context_analysis = gr.Checkbox(label="🌍 Context Analysis", value=True)
                
                # Context inputs
                with gr.Accordion("🌍 Context & Environment", open=False):
                    trending_topics = gr.Textbox(
                        label="🔥 Trending Topics (comma-separated)",
                        placeholder="#AI, #Innovation, #Technology",
                        value=", ".join(self.demo_data['trending_topics'])
                    )
                    audience_demographics = gr.Dropdown(
                        label="👥 Target Audience",
                        choices=self.demo_data['audience_demographics']['age_ranges'],
                        value="25-34"
                    )
                    posting_time = gr.Dropdown(
                        label="⏰ Posting Time",
                        choices=self.demo_data['time_periods'],
                        value="Afternoon (12PM-6PM)"
                    )
                
                # Analyze button
                analyze_btn = gr.Button("🚀 Analyze Content", variant="primary", size="lg")
                
            with gr.Column(scale=2):
                # Results display
                gr.Markdown("## 📊 Analysis Results")
                
                # Viral prediction results
                viral_results = gr.HTML(label="🔮 Viral Prediction")
                
                # Sentiment analysis results
                sentiment_results = gr.HTML(label="😊 Sentiment Analysis")
                
                # Context analysis results
                context_results = gr.HTML(label="🌍 Context Analysis")
                
                # Recommendations
                recommendations_output = gr.HTML(label="💡 AI Recommendations")
        
        # Event handlers
        analyze_btn.click(
            fn=self._analyze_content_advanced,
            inputs=[content_input, enable_viral_prediction, enable_sentiment_analysis, 
                   enable_context_analysis, trending_topics, audience_demographics, posting_time],
            outputs=[viral_results, sentiment_results, context_results, recommendations_output]
        )
    
    def _create_predictive_analytics_tab(self):
        """Create the predictive analytics tab"""
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("## 📊 Predictive Analytics Dashboard")
                gr.Markdown("Real-time predictions and forecasting for your content strategy")
                
                # Content for prediction
                pred_content = gr.Textbox(
                    label="📝 Content for Prediction",
                    placeholder="Enter content to analyze...",
                    lines=3
                )
                
                # Prediction type
                prediction_type = gr.Radio(
                    label="🎯 Prediction Type",
                    choices=["Viral Potential", "Engagement Forecast", "Audience Response", "Trend Prediction"],
                    value="Viral Potential"
                )
                
                # Time horizon
                time_horizon = gr.Slider(
                    label="⏰ Prediction Horizon (days)",
                    minimum=1,
                    maximum=90,
                    value=30,
                    step=1
                )
                
                # Generate prediction button
                predict_btn = gr.Button("🔮 Generate Prediction", variant="primary")
                
            with gr.Column(scale=2):
                # Prediction results
                gr.Markdown("## 🔮 Prediction Results")
                
                prediction_output = gr.HTML(label="Prediction Results")
                
                # Prediction visualization
                prediction_chart = gr.Plot(label="📈 Prediction Visualization")
        
        # Event handlers
        predict_btn.click(
            fn=self._generate_prediction,
            inputs=[pred_content, prediction_type, time_horizon],
            outputs=[prediction_output, prediction_chart]
        )
    
    def _create_audience_intelligence_tab(self):
        """Create the audience intelligence tab"""
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("## 🎯 Audience Intelligence")
                gr.Markdown("Advanced audience segmentation and behavioral analysis")
                
                # Audience data input
                gr.Markdown("### 📊 Sample Audience Data")
                sample_data = gr.DataFrame(
                    value=self._generate_sample_audience_data(),
                    label="Sample Audience Data",
                    interactive=False
                )
                
                # Segmentation options
                with gr.Row():
                    segment_count = gr.Slider(
                        label="🔢 Number of Segments",
                        minimum=3,
                        maximum=10,
                        value=5,
                        step=1
                    )
                    enable_behavioral_analysis = gr.Checkbox(label="🧠 Behavioral Analysis", value=True)
                
                # Segment button
                segment_btn = gr.Button("🎯 Create Segments", variant="primary")
                
            with gr.Column(scale=2):
                # Segmentation results
                gr.Markdown("## 🎯 Segmentation Results")
                
                segments_output = gr.HTML(label="Audience Segments")
                
                # Segment visualization
                segment_chart = gr.Plot(label="📊 Segment Distribution")
        
        # Event handlers
        segment_btn.click(
            fn=self._create_audience_segments,
            inputs=[segment_count, enable_behavioral_analysis],
            outputs=[segments_output, segment_chart]
        )
    
    def _create_performance_forecasting_tab(self):
        """Create the performance forecasting tab"""
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("## 📈 Performance Forecasting")
                gr.Markdown("Predict future performance and optimize your content strategy")
                
                # Content input
                forecast_content = gr.Textbox(
                    label="📝 Content for Forecasting",
                    placeholder="Enter content to forecast...",
                    lines=3
                )
                
                # Audience segment
                audience_segment = gr.Dropdown(
                    label="👥 Target Segment",
                    choices=["segment_0", "segment_1", "segment_2", "segment_3", "segment_4"],
                    value="segment_0"
                )
                
                # Posting schedule
                posting_schedule = gr.Dropdown(
                    label="📅 Posting Schedule",
                    choices=["Daily", "Weekly", "Bi-weekly", "Monthly"],
                    value="Weekly"
                )
                
                # Forecast button
                forecast_btn = gr.Button("📈 Generate Forecast", variant="primary")
                
            with gr.Column(scale=2):
                # Forecast results
                gr.Markdown("## 📈 Forecast Results")
                
                forecast_output = gr.HTML(label="Performance Forecast")
                
                # Forecast chart
                forecast_chart = gr.Plot(label="📊 Forecast Visualization")
        
        # Event handlers
        forecast_btn.click(
            fn=self._generate_performance_forecast,
            inputs=[forecast_content, audience_segment, posting_schedule],
            outputs=[forecast_output, forecast_chart]
        )
    
    def _create_system_health_tab(self):
        """Create the system health and metrics tab"""
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("## 🏥 System Health & Performance")
                gr.Markdown("Monitor system performance and health metrics")
                
                # Refresh button
                refresh_btn = gr.Button("🔄 Refresh Metrics", variant="primary")
                
                # System status
                system_status = gr.HTML(label="🟢 System Status")
                
            with gr.Column(scale=2):
                # Performance metrics
                gr.Markdown("## 📊 Performance Metrics")
                
                performance_metrics = gr.HTML(label="System Metrics")
                
                # Performance chart
                performance_chart = gr.Plot(label="📈 Performance Trends")
        
        # Event handlers
        refresh_btn.click(
            fn=self._refresh_system_metrics,
            inputs=[],
            outputs=[system_status, performance_metrics, performance_chart]
        )
    
    def _analyze_content_advanced(self, content: str, enable_viral: bool, 
                                 enable_sentiment: bool, enable_context: bool,
                                 trending_topics: str, audience_demographics: str, 
                                 posting_time: str) -> tuple:
        """Advanced content analysis with multiple AI models"""
        if not content.strip():
            return "❌ Please enter content to analyze", "", "", ""
        
        try:
            results = []
            
            # Prepare context
            context = {
                'trending_topics': [topic.strip() for topic in trending_topics.split(',') if topic.strip()],
                'audience_demographics': audience_demographics,
                'timing': posting_time
            }
            
            # Viral prediction
            viral_html = ""
            if enable_viral:
                viral_result = self.predictive_system.predict_viral_potential(content, context)
                viral_html = self._format_viral_results(viral_result)
            
            # Sentiment analysis
            sentiment_html = ""
            if enable_sentiment:
                sentiment_result = self.predictive_system.analyze_sentiment_advanced(content, context)
                sentiment_html = self._format_sentiment_results(sentiment_result)
            
            # Context analysis
            context_html = ""
            if enable_context:
                context_analysis = self._analyze_context_factors(content, context)
                context_html = self._format_context_results(context_analysis)
            
            # Generate recommendations
            recommendations = self._generate_combined_recommendations(
                viral_result if enable_viral else None,
                sentiment_result if enable_sentiment else None,
                context_analysis if enable_context else None
            )
            recommendations_html = self._format_recommendations(recommendations)
            
            return viral_html, sentiment_html, context_html, recommendations_html
            
        except Exception as e:
            error_msg = f"❌ Error during analysis: {str(e)}"
            return error_msg, error_msg, error_msg, error_msg
    
    def _generate_prediction(self, content: str, prediction_type: str, 
                           time_horizon: int) -> tuple:
        """Generate predictions based on content and type"""
        if not content.strip():
            return "❌ Please enter content for prediction", None
        
        try:
            if prediction_type == "Viral Potential":
                result = self.predictive_system.predict_viral_potential(content)
                output_html = self._format_viral_results(result)
                chart = self._create_viral_prediction_chart(result)
                
            elif prediction_type == "Engagement Forecast":
                result = self.predictive_system.forecast_engagement(
                    content, "segment_0", datetime.now()
                )
                output_html = self._format_engagement_forecast(result)
                chart = self._create_engagement_forecast_chart(result)
                
            else:
                output_html = f"🎯 {prediction_type} analysis completed"
                chart = self._create_generic_chart()
            
            return output_html, chart
            
        except Exception as e:
            error_msg = f"❌ Error generating prediction: {str(e)}"
            return error_msg, None
    
    def _create_audience_segments(self, segment_count: int, 
                                 enable_behavioral: bool) -> tuple:
        """Create audience segments"""
        try:
            # Generate sample audience data
            audience_data = self._generate_sample_audience_data()
            
            # Create segments
            segments_result = self.predictive_system.segment_audience(audience_data)
            
            # Format output
            segments_html = self._format_segments_results(segments_result)
            
            # Create visualization
            chart = self._create_segments_chart(segments_result)
            
            return segments_html, chart
            
        except Exception as e:
            error_msg = f"❌ Error creating segments: {str(e)}"
            return error_msg, None
    
    def _generate_performance_forecast(self, content: str, audience_segment: str,
                                     posting_schedule: str) -> tuple:
        """Generate performance forecast"""
        if not content.strip():
            return "❌ Please enter content for forecasting", None
        
        try:
            # Generate forecast
            forecast_result = self.predictive_system.forecast_engagement(
                content, audience_segment, datetime.now()
            )
            
            # Format output
            forecast_html = self._format_engagement_forecast(forecast_result)
            
            # Create visualization
            chart = self._create_forecast_chart(forecast_result, posting_schedule)
            
            return forecast_html, chart
            
        except Exception as e:
            error_msg = f"❌ Error generating forecast: {str(e)}"
            return error_msg, None
    
    def _refresh_system_metrics(self) -> tuple:
        """Refresh system health and performance metrics"""
        try:
            # Get system status
            system_status = "🟢 System Operational" if self.is_initialized else "🔴 System Error"
            
            # Get performance metrics
            if self.predictive_system:
                metrics = self.predictive_system.get_system_metrics()
                metrics_html = self._format_system_metrics(metrics)
            else:
                metrics_html = "❌ System not initialized"
            
            # Create performance chart
            chart = self._create_performance_chart()
            
            return system_status, metrics_html, chart
            
        except Exception as e:
            error_msg = f"❌ Error refreshing metrics: {str(e)}"
            return "🔴 System Error", error_msg, None
    
    def _format_viral_results(self, result: Dict[str, Any]) -> str:
        """Format viral prediction results"""
        if 'error' in result:
            return f"❌ Error: {result['error']}"
        
        viral_score = result.get('viral_score', 0)
        confidence = result.get('confidence', 0)
        recommendations = result.get('recommendations', [])
        
        # Color coding based on viral score
        if viral_score > 0.7:
            color = "🟢"
            status = "High Viral Potential"
        elif viral_score > 0.4:
            color = "🟡"
            status = "Medium Viral Potential"
        else:
            color = "🔴"
            status = "Low Viral Potential"
        
        html = f"""
        <div class="prediction-result">
            <h3>{color} Viral Prediction Results</h3>
            <p><strong>Viral Score:</strong> {viral_score:.3f} ({viral_score*100:.1f}%)</p>
            <p><strong>Status:</strong> {status}</p>
            <p><strong>Confidence:</strong> {confidence:.2f}</p>
            <p><strong>Viral Probability:</strong> {result.get('viral_probability', 'N/A')}</p>
        </div>
        """
        
        if recommendations:
            html += "<h4>💡 Recommendations:</h4><ul>"
            for rec in recommendations:
                html += f"<li>{rec}</li>"
            html += "</ul>"
        
        return html
    
    def _format_sentiment_results(self, result: Dict[str, Any]) -> str:
        """Format sentiment analysis results"""
        if 'error' in result:
            return f"❌ Error: {result['error']}"
        
        primary_emotion = result.get('primary_emotion', 'neutral')
        emotion_confidence = result.get('emotion_confidence', 0)
        sentiment_score = result.get('sentiment_score', 0)
        emotion_breakdown = result.get('emotion_breakdown', {})
        
        # Emotion emoji mapping
        emotion_emojis = {
            'joy': '😊', 'sadness': '😢', 'anger': '😠', 
            'fear': '😨', 'surprise': '😲', 'disgust': '🤢', 'neutral': '😐'
        }
        
        html = f"""
        <div class="prediction-result">
            <h3>😊 Advanced Sentiment Analysis</h3>
            <p><strong>Primary Emotion:</strong> {emotion_emojis.get(primary_emotion, '😐')} {primary_emotion.title()}</p>
            <p><strong>Confidence:</strong> {emotion_confidence:.2f}</p>
            <p><strong>Overall Sentiment:</strong> {sentiment_score:.3f}</p>
        </div>
        
        <h4>📊 Emotion Breakdown:</h4>
        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px;">
        """
        
        for emotion, score in emotion_breakdown.items():
            emoji = emotion_emojis.get(emotion, '😐')
            percentage = score * 100
            html += f"""
            <div class="metric-card">
                <p><strong>{emoji} {emotion.title()}:</strong> {percentage:.1f}%</p>
            </div>
            """
        
        html += "</div>"
        
        if result.get('recommendations'):
            html += "<h4>💡 Recommendations:</h4><ul>"
            for rec in result['recommendations']:
                html += f"<li>{rec}</li>"
            html += "</ul>"
        
        return html
    
    def _format_context_results(self, context_analysis: Dict[str, Any]) -> str:
        """Format context analysis results"""
        html = """
        <div class="prediction-result">
            <h3>🌍 Context Analysis Results</h3>
        """
        
        for key, value in context_analysis.items():
            if isinstance(value, dict):
                html += f"<p><strong>{key.replace('_', ' ').title()}:</strong></p><ul>"
                for sub_key, sub_value in value.items():
                    html += f"<li>{sub_key}: {sub_value}</li>"
                html += "</ul>"
            else:
                html += f"<p><strong>{key.replace('_', ' ').title()}:</strong> {value}</p>"
        
        html += "</div>"
        return html
    
    def _format_recommendations(self, recommendations: List[str]) -> str:
        """Format AI recommendations"""
        if not recommendations:
            return "<p>No specific recommendations available.</p>"
        
        html = """
        <div class="recommendation-box">
            <h4>💡 AI-Powered Recommendations</h4>
            <ul>
        """
        
        for rec in recommendations:
            html += f"<li>{rec}</li>"
        
        html += "</ul></div>"
        return html
    
    def _format_engagement_forecast(self, result: Dict[str, Any]) -> str:
        """Format engagement forecast results"""
        if 'error' in result:
            return f"❌ Error: {result['error']}"
        
        forecast = result.get('forecast', {})
        predicted_engagement = forecast.get('predicted_engagement', 0)
        confidence_interval = forecast.get('confidence_interval', (0, 0))
        
        html = f"""
        <div class="prediction-result">
            <h3>📈 Engagement Forecast</h3>
            <p><strong>Predicted Engagement:</strong> {predicted_engagement:.3f}</p>
            <p><strong>Confidence Interval:</strong> {confidence_interval[0]:.3f} - {confidence_interval[1]:.3f}</p>
            <p><strong>Confidence Level:</strong> {forecast.get('confidence_level', 0):.2f}</p>
        </div>
        """
        
        if result.get('recommendations'):
            html += "<h4>💡 Optimization Tips:</h4><ul>"
            for rec in result['recommendations']:
                html += f"<li>{rec}</li>"
            html += "</ul>"
        
        return html
    
    def _format_segments_results(self, result: Dict[str, Any]) -> str:
        """Format audience segmentation results"""
        if 'error' in result:
            return f"❌ Error: {result['error']}"
        
        segments = result.get('segments', {})
        insights = result.get('insights', {})
        
        html = f"""
        <div class="prediction-result">
            <h3>🎯 Audience Segmentation Results</h3>
            <p><strong>Total Segments:</strong> {len(segments)}</p>
            <p><strong>Total Audience:</strong> {insights.get('total_size', 0):,}</p>
            <p><strong>Largest Segment:</strong> {insights.get('largest_segment', 'N/A')}</p>
            <p><strong>Highest Engagement:</strong> {insights.get('highest_engagement', 'N/A')}</p>
        </div>
        
        <h4>📊 Segment Details:</h4>
        """
        
        for segment_name, segment_data in segments.items():
            size = segment_data.get('size', 0)
            engagement = segment_data.get('engagement_rate', 0)
            
            html += f"""
            <div class="metric-card">
                <h5>{segment_name.title()}</h5>
                <p><strong>Size:</strong> {size:,}</p>
                <p><strong>Engagement Rate:</strong> {engagement:.3f}</p>
            </div>
            """
        
        return html
    
    def _format_system_metrics(self, metrics: Dict[str, Any]) -> str:
        """Format system performance metrics"""
        html = """
        <div class="prediction-result">
            <h3>📊 System Performance Metrics</h3>
        """
        
        for key, value in metrics.items():
            if key == 'timestamp':
                continue
            elif isinstance(value, dict):
                html += f"<p><strong>{key.replace('_', ' ').title()}:</strong></p><ul>"
                for sub_key, sub_value in value.items():
                    html += f"<li>{sub_key}: {sub_value}</li>"
                html += "</ul>"
            else:
                html += f"<p><strong>{key.replace('_', ' ').title()}:</strong> {value}</p>"
        
        html += "</div>"
        return html
    
    def _analyze_context_factors(self, content: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze contextual factors for content"""
        analysis = {}
        
        # Trend relevance
        if 'trending_topics' in context:
            trend_relevance = self._calculate_trend_relevance(content, context['trending_topics'])
            analysis['trend_relevance'] = trend_relevance
        
        # Timing optimization
        if 'timing' in context:
            timing_analysis = self._analyze_timing_optimization(context['timing'])
            analysis['timing_optimization'] = timing_analysis
        
        # Audience alignment
        if 'audience_demographics' in context:
            audience_alignment = self._analyze_audience_alignment(content, context['audience_demographics'])
            analysis['audience_alignment'] = audience_alignment
        
        return analysis
    
    def _calculate_trend_relevance(self, content: str, trending_topics: List[str]) -> Dict[str, Any]:
        """Calculate relevance to trending topics"""
        relevance_scores = {}
        total_relevance = 0
        
        for topic in trending_topics:
            # Simple keyword matching
            topic_clean = topic.replace('#', '').lower()
            content_lower = content.lower()
            
            if topic_clean in content_lower:
                score = 0.8
            elif any(word in content_lower for word in topic_clean.split()):
                score = 0.5
            else:
                score = 0.1
            
            relevance_scores[topic] = score
            total_relevance += score
        
        avg_relevance = total_relevance / len(trending_topics) if trending_topics else 0
        
        return {
            'average_relevance': avg_relevance,
            'topic_scores': relevance_scores,
            'trending_topics_count': len(trending_topics)
        }
    
    def _analyze_timing_optimization(self, timing: str) -> Dict[str, Any]:
        """Analyze timing optimization"""
        timing_scores = {
            'Morning (6AM-12PM)': 0.7,
            'Afternoon (12PM-6PM)': 0.9,
            'Evening (6PM-12AM)': 0.8,
            'Night (12AM-6AM)': 0.3
        }
        
        optimal_score = timing_scores.get(timing, 0.5)
        
        return {
            'selected_timing': timing,
            'timing_score': optimal_score,
            'optimal_timing': max(timing_scores, key=timing_scores.get),
            'timing_recommendation': "Consider posting during peak hours for better engagement" if optimal_score < 0.8 else "Excellent timing choice!"
        }
    
    def _analyze_audience_alignment(self, content: str, demographics: str) -> Dict[str, Any]:
        """Analyze content alignment with target audience"""
        # Simple demographic analysis
        age_keywords = {
            '18-24': ['trendy', 'viral', 'challenge', 'fun', 'cool'],
            '25-34': ['professional', 'career', 'business', 'growth', 'success'],
            '35-44': ['family', 'parenting', 'career', 'balance', 'experience'],
            '45-54': ['experience', 'wisdom', 'career', 'family', 'values'],
            '55+': ['wisdom', 'experience', 'legacy', 'family', 'values']
        }
        
        target_keywords = age_keywords.get(demographics, [])
        content_lower = content.lower()
        
        keyword_matches = sum(1 for keyword in target_keywords if keyword in content_lower)
        alignment_score = min(keyword_matches / len(target_keywords), 1.0) if target_keywords else 0
        
        return {
            'target_demographics': demographics,
            'alignment_score': alignment_score,
            'keyword_matches': keyword_matches,
            'total_keywords': len(target_keywords),
            'recommendation': "Content aligns well with target audience" if alignment_score > 0.6 else "Consider adjusting content for better audience alignment"
        }
    
    def _generate_combined_recommendations(self, viral_result: Optional[Dict], 
                                         sentiment_result: Optional[Dict],
                                         context_analysis: Optional[Dict]) -> List[str]:
        """Generate combined recommendations from all analyses"""
        recommendations = []
        
        # Viral recommendations
        if viral_result and 'recommendations' in viral_result:
            recommendations.extend(viral_result['recommendations'])
        
        # Sentiment recommendations
        if sentiment_result and 'recommendations' in sentiment_result:
            recommendations.extend(sentiment_result['recommendations'])
        
        # Context recommendations
        if context_analysis:
            if 'timing_optimization' in context_analysis:
                timing_rec = context_analysis['timing_optimization'].get('timing_recommendation')
                if timing_rec:
                    recommendations.append(timing_rec)
            
            if 'audience_alignment' in context_analysis:
                audience_rec = context_analysis['audience_alignment'].get('recommendation')
                if audience_rec:
                    recommendations.append(audience_rec)
        
        # Remove duplicates and limit
        unique_recommendations = list(dict.fromkeys(recommendations))
        return unique_recommendations[:10]  # Limit to top 10
    
    def _generate_sample_audience_data(self) -> pd.DataFrame:
        """Generate sample audience data for demonstration"""
        np.random.seed(42)
        
        n_samples = 1000
        
        data = {
            'age': np.random.normal(35, 10, n_samples).clip(18, 65),
            'gender': np.random.choice(['Male', 'Female', 'Other'], n_samples),
            'engagement_rate': np.random.beta(2, 5, n_samples),
            'posting_frequency': np.random.poisson(3, n_samples),
            'content_type_preference': np.random.choice(['Post', 'Story', 'Video', 'Image'], n_samples),
            'posting_time': pd.date_range('2024-01-01', periods=n_samples, freq='H')
        }
        
        return pd.DataFrame(data)
    
    def _create_viral_prediction_chart(self, result: Dict[str, Any]) -> go.Figure:
        """Create viral prediction visualization"""
        if 'error' in result:
            return go.Figure()
        
        viral_score = result.get('viral_score', 0)
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=viral_score * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Viral Potential Score"},
            delta={'reference': 50},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgray"},
                    {'range': [30, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig.update_layout(height=400, title="Viral Potential Gauge")
        return fig
    
    def _create_engagement_forecast_chart(self, result: Dict[str, Any]) -> go.Figure:
        """Create engagement forecast visualization"""
        if 'error' in result:
            return go.Figure()
        
        forecast = result.get('forecast', {})
        predicted_engagement = forecast.get('predicted_engagement', 0)
        confidence_interval = forecast.get('confidence_interval', (0, 0))
        
        fig = go.Figure()
        
        # Main prediction
        fig.add_trace(go.Bar(
            x=['Predicted Engagement'],
            y=[predicted_engagement],
            name='Prediction',
            marker_color='blue'
        ))
        
        # Confidence interval
        fig.add_trace(go.Scatter(
            x=['Predicted Engagement'],
            y=[confidence_interval[0], confidence_interval[1]],
            mode='markers',
            name='Confidence Interval',
            marker=dict(color='red', size=10)
        ))
        
        fig.update_layout(
            title="Engagement Forecast",
            yaxis_title="Engagement Rate",
            height=400
        )
        
        return fig
    
    def _create_segments_chart(self, result: Dict[str, Any]) -> go.Figure:
        """Create audience segments visualization"""
        if 'error' in result:
            return go.Figure()
        
        segments = result.get('segments', {})
        
        if not segments:
            return go.Figure()
        
        segment_names = list(segments.keys())
        segment_sizes = [seg['size'] for seg in segments.values()]
        engagement_rates = [seg['engagement_rate'] for seg in segments.values()]
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Segment Sizes', 'Engagement Rates'),
            specs=[[{"type": "pie"}, {"type": "bar"}]]
        )
        
        # Pie chart for segment sizes
        fig.add_trace(
            go.Pie(labels=segment_names, values=segment_sizes, name="Segment Sizes"),
            row=1, col=1
        )
        
        # Bar chart for engagement rates
        fig.add_trace(
            go.Bar(x=segment_names, y=engagement_rates, name="Engagement Rates"),
            row=1, col=2
        )
        
        fig.update_layout(height=400, title="Audience Segmentation Analysis")
        return fig
    
    def _create_forecast_chart(self, result: Dict[str, Any], posting_schedule: str) -> go.Figure:
        """Create performance forecast visualization"""
        if 'error' in result:
            return go.Figure()
        
        # Generate future dates based on posting schedule
        if posting_schedule == "Daily":
            dates = pd.date_range(start=datetime.now(), periods=30, freq='D')
        elif posting_schedule == "Weekly":
            dates = pd.date_range(start=datetime.now(), periods=12, freq='W')
        elif posting_schedule == "Bi-weekly":
            dates = pd.date_range(start=datetime.now(), periods=6, freq='2W')
        else:  # Monthly
            dates = pd.date_range(start=datetime.now(), periods=6, freq='M')
        
        # Generate forecast values with some variation
        base_engagement = result.get('forecast', {}).get('predicted_engagement', 0.5)
        np.random.seed(42)
        forecast_values = base_engagement + np.random.normal(0, 0.1, len(dates))
        forecast_values = np.clip(forecast_values, 0, 1)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=forecast_values,
            mode='lines+markers',
            name='Forecasted Engagement',
            line=dict(color='blue', width=2),
            marker=dict(size=6)
        ))
        
        fig.update_layout(
            title=f"Performance Forecast - {posting_schedule} Schedule",
            xaxis_title="Date",
            yaxis_title="Engagement Rate",
            height=400
        )
        
        return fig
    
    def _create_generic_chart(self) -> go.Figure:
        """Create a generic chart placeholder"""
        fig = go.Figure()
        fig.add_annotation(
            text="Chart visualization will be generated based on prediction type",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(height=400, title="Chart Placeholder")
        return fig
    
    def _create_performance_chart(self) -> go.Figure:
        """Create system performance visualization"""
        # Generate sample performance data
        dates = pd.date_range(start=datetime.now() - timedelta(days=7), periods=7, freq='D')
        cpu_usage = np.random.uniform(20, 80, 7)
        memory_usage = np.random.uniform(30, 70, 7)
        response_time = np.random.uniform(0.1, 0.5, 7)
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('CPU Usage', 'Memory Usage', 'Response Time', 'System Health'),
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "indicator"}]]
        )
        
        # CPU Usage
        fig.add_trace(
            go.Scatter(x=dates, y=cpu_usage, mode='lines+markers', name='CPU %'),
            row=1, col=1
        )
        
        # Memory Usage
        fig.add_trace(
            go.Scatter(x=dates, y=memory_usage, mode='lines+markers', name='Memory %'),
            row=1, col=2
        )
        
        # Response Time
        fig.add_trace(
            go.Scatter(x=dates, y=response_time, mode='lines+markers', name='Response (s)'),
            row=2, col=1
        )
        
        # System Health Indicator
        avg_health = (100 - np.mean(cpu_usage) + (100 - np.mean(memory_usage)) + (1 - np.mean(response_time)) * 100) / 3
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=avg_health,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "System Health"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "green"},
                    'steps': [
                        {'range': [0, 50], 'color': "red"},
                        {'range': [50, 80], 'color': "yellow"},
                        {'range': [80, 100], 'color': "green"}
                    ]
                }
            ),
            row=2, col=2
        )
        
        fig.update_layout(height=600, title="System Performance Overview")
        return fig
    
    def launch(self, **kwargs):
        """Launch the advanced predictive interface"""
        interface = self.create_interface()
        return interface.launch(**kwargs)


# Example usage and testing
if __name__ == "__main__":
    # Initialize interface
    interface = AdvancedPredictiveInterface()
    
    # Launch the interface
    interface.launch(
        server_name="0.0.0.0",
        server_port=7862,  # Different port to avoid conflicts
        share=False,
        debug=True,
        show_error=True,
        quiet=False
    )

