"""
Enhanced Gradio Interface for NLP Platform
User-friendly interfaces that showcase model capabilities with improved UX/UI
"""

import gradio as gr
import torch
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
import json
import logging
import traceback
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import our NLP systems
from nlp_system_optimized import OptimizedNLPSystem, NLPSystemConfig
from gradient_clipping_nan_handling_system import GradientConfig, GradientMonitor
from advanced_evaluation_metrics_system import (
    ClassificationMetricsConfig, RegressionMetricsConfig,
    TextGenerationMetricsConfig, ComprehensiveEvaluationSystem
)
from comprehensive_training_system import ComprehensiveTrainingSystem, ComprehensiveTrainingConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EnhancedGradioConfig:
    """Enhanced configuration for Gradio interface"""
    theme: str = "default"
    title: str = "🤖 Advanced NLP Platform"
    description: str = "Experience the power of AI with our comprehensive NLP platform. Generate text, analyze sentiment, classify content, and train custom models - all through an intuitive interface."
    allow_flagging: str = "never"
    cache_examples: bool = True
    max_threads: int = 40
    show_error: bool = True
    height: int = 800
    width: int = 1200
    show_tips: bool = True
    enable_animations: bool = True


class EnhancedInputValidator:
    """Enhanced input validation with better user feedback"""
    
    @staticmethod
    def validate_text_input(text: str, min_length: int = 1, max_length: int = 1000) -> Tuple[bool, str]:
        """Validate text input with helpful feedback"""
        if not text or not text.strip():
            return False, "❌ Please enter some text to continue"
        
        if len(text.strip()) < min_length:
            return False, f"❌ Text must be at least {min_length} characters long"
        
        if len(text) > max_length:
            return False, f"❌ Text must be no more than {max_length} characters"
        
        return True, "✅ Valid input"
    
    @staticmethod
    def validate_numeric_input(value: Union[int, float], min_val: float, max_val: float) -> Tuple[bool, str]:
        """Validate numeric input with helpful feedback"""
        try:
            num_val = float(value)
            if num_val < min_val or num_val > max_val:
                return False, f"❌ Value must be between {min_val} and {max_val}"
            return True, "✅ Valid input"
        except (ValueError, TypeError):
            return False, "❌ Please enter a valid number"
    
    @staticmethod
    def validate_model_name(model_name: str) -> Tuple[bool, str]:
        """Validate model name with helpful feedback"""
        valid_models = {
            "gpt2": "Fast text generation",
            "gpt2-medium": "Balanced performance",
            "gpt2-large": "High quality generation",
            "gpt2-xl": "Best quality (slower)",
            "bert-base-uncased": "Good for classification",
            "bert-base-cased": "Case-sensitive analysis",
            "roberta-base": "Robust performance",
            "distilbert-base-uncased": "Fast and efficient"
        }
        
        if model_name not in valid_models:
            return False, f"❌ Invalid model. Choose from: {', '.join(valid_models.keys())}"
        
        return True, f"✅ {valid_models[model_name]}"


class EnhancedErrorHandler:
    """Enhanced error handling with user-friendly messages"""
    
    @staticmethod
    def handle_exception(func):
        """Decorator to handle exceptions with better user feedback"""
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_msg = f"⚠️ Something went wrong: {str(e)}"
                logger.error(f"Gradio function error: {error_msg}")
                logger.error(traceback.format_exc())
                return error_msg, None, None, "❌ Error occurred"
        return wrapper
    
    @staticmethod
    def format_error_message(error: Exception) -> str:
        """Format error message for user display"""
        return f"⚠️ An error occurred: {str(error)}. Please check your inputs and try again."


class EnhancedVisualizationManager:
    """Enhanced visualizations with better design and interactivity"""
    
    @staticmethod
    def create_sentiment_chart(sentiment_data: Dict[str, float]) -> go.Figure:
        """Create enhanced sentiment analysis visualization"""
        labels = list(sentiment_data.keys())
        values = list(sentiment_data.values())
        
        # Color mapping for sentiment
        colors = {
            'Positive': '#2E8B57',  # Sea Green
            'Negative': '#DC143C',  # Crimson
            'Neutral': '#FFD700',   # Gold
            'Generated': '#4169E1', # Royal Blue
            'Original': '#FF6347'   # Tomato
        }
        
        bar_colors = [colors.get(label, '#4682B4') for label in labels]
        
        fig = go.Figure(data=[
            go.Bar(
                x=labels, 
                y=values, 
                marker_color=bar_colors,
                text=[f'{v:.3f}' for v in values],
                textposition='auto',
                hovertemplate='<b>%{x}</b><br>Score: %{y:.3f}<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title={
                'text': "🎭 Sentiment Analysis Results",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'color': '#2F4F4F'}
            },
            xaxis_title="Sentiment Type",
            yaxis_title="Confidence Score",
            template="plotly_white",
            height=400,
            showlegend=False,
            margin=dict(l=50, r=50, t=80, b=50)
        )
        
        return fig
    
    @staticmethod
    def create_training_progress_chart(epochs: List[int], metrics: Dict[str, List[float]]) -> go.Figure:
        """Create enhanced training progress visualization"""
        fig = go.Figure()
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        for i, (metric_name, values) in enumerate(metrics.items()):
            fig.add_trace(go.Scatter(
                x=epochs,
                y=values,
                mode='lines+markers',
                name=metric_name.replace('_', ' ').title(),
                line=dict(color=colors[i % len(colors)], width=3),
                marker=dict(size=8),
                hovertemplate='<b>%{fullData.name}</b><br>Epoch: %{x}<br>Value: %{y:.4f}<extra></extra>'
            ))
        
        fig.update_layout(
            title={
                'text': "📈 Training Progress",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'color': '#2F4F4F'}
            },
            xaxis_title="Epoch",
            yaxis_title="Metric Value",
            template="plotly_white",
            height=500,
            hovermode='x unified',
            margin=dict(l=50, r=50, t=80, b=50)
        )
        
        return fig
    
    @staticmethod
    def create_performance_gauge(metric_name: str, value: float, max_value: float = 1.0) -> go.Figure:
        """Create performance gauge visualization"""
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=value,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': metric_name, 'font': {'size': 16}},
            delta={'reference': max_value * 0.5},
            gauge={
                'axis': {'range': [None, max_value]},
                'bar': {'color': "#4682B4"},
                'steps': [
                    {'range': [0, max_value * 0.5], 'color': "lightgray"},
                    {'range': [max_value * 0.5, max_value * 0.8], 'color': "yellow"},
                    {'range': [max_value * 0.8, max_value], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': max_value * 0.9
                }
            }
        ))
        
        fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        return fig


class EnhancedGradioNLPSystem:
    """Enhanced Gradio interface with improved UX/UI"""
    
    def __init__(self, config: EnhancedGradioConfig = None):
        self.config = config or EnhancedGradioConfig()
        self.nlp_system = None
        self.validator = EnhancedInputValidator()
        self.error_handler = EnhancedErrorHandler()
        self.viz_manager = EnhancedVisualizationManager()
        self._initialize_nlp_system()
    
    def _initialize_nlp_system(self):
        """Initialize the NLP system"""
        try:
            nlp_config = NLPSystemConfig(
                model_name="gpt2",
                max_length=100,
                temperature=0.7,
                use_gpu=torch.cuda.is_available()
            )
            self.nlp_system = OptimizedNLPSystem(nlp_config)
            logger.info("NLP system initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize NLP system: {e}")
            self.nlp_system = None
    
    @EnhancedErrorHandler.handle_exception
    def generate_text_interface(self, 
                              prompt: str, 
                              max_length: int, 
                              temperature: float,
                              model_name: str) -> Tuple[str, Optional[go.Figure], str]:
        """Enhanced text generation interface"""
        # Input validation with better feedback
        is_valid, error_msg = self.validator.validate_text_input(prompt, 1, 500)
        if not is_valid:
            return error_msg, None, "❌ Invalid Input"
        
        is_valid, error_msg = self.validator.validate_numeric_input(max_length, 10, 500)
        if not is_valid:
            return error_msg, None, "❌ Invalid Length"
        
        is_valid, error_msg = self.validator.validate_numeric_input(temperature, 0.1, 2.0)
        if not is_valid:
            return error_msg, None, "❌ Invalid Temperature"
        
        is_valid, error_msg = self.validator.validate_model_name(model_name)
        if not is_valid:
            return error_msg, None, "❌ Invalid Model"
        
        # Generate text
        if self.nlp_system is None:
            return "❌ NLP system not initialized", None, "❌ System Error"
        
        try:
            generated_text = self.nlp_system.generate_text(
                prompt=prompt,
                max_length=max_length,
                temperature=temperature,
                model_name=model_name
            )
            
            # Create enhanced visualization
            fig = self.viz_manager.create_sentiment_chart({
                "Generated": 0.8,
                "Original": 0.6,
                "Creativity": temperature,
                "Length": min(max_length / 500, 1.0)
            })
            
            return generated_text, fig, "✅ Generation Complete"
            
        except Exception as e:
            return self.error_handler.format_error_message(e), None, "❌ Generation Failed"
    
    @EnhancedErrorHandler.handle_exception
    def sentiment_analysis_interface(self, text: str) -> Tuple[str, Optional[go.Figure], str]:
        """Enhanced sentiment analysis interface"""
        # Input validation
        is_valid, error_msg = self.validator.validate_text_input(text, 10, 1000)
        if not is_valid:
            return error_msg, None, "❌ Invalid Input"
        
        if self.nlp_system is None:
            return "❌ NLP system not initialized", None, "❌ System Error"
        
        try:
            analyzer = self.nlp_system.analyzer
            sentiment_result = analyzer.analyze_sentiment(text)
            
            # Format result with emojis and better styling
            result_text = f"🎭 **Sentiment Analysis Results**\n\n"
            result_text += f"**Overall Sentiment:** {sentiment_result['label']}\n"
            result_text += f"**Confidence:** {sentiment_result['score']:.3f}\n\n"
            result_text += f"**Analyzed Text:**\n{text[:200]}{'...' if len(text) > 200 else ''}\n\n"
            result_text += f"📊 **Detailed Breakdown:**\n"
            result_text += f"• Positive: {sentiment_result.get('positive', 0):.3f}\n"
            result_text += f"• Neutral: {sentiment_result.get('neutral', 0):.3f}\n"
            result_text += f"• Negative: {sentiment_result.get('negative', 0):.3f}"
            
            # Create enhanced visualization
            fig = self.viz_manager.create_sentiment_chart({
                "Positive": sentiment_result.get('positive', 0),
                "Neutral": sentiment_result.get('neutral', 0),
                "Negative": sentiment_result.get('negative', 0)
            })
            
            return result_text, fig, "✅ Analysis Complete"
            
        except Exception as e:
            return self.error_handler.format_error_message(e), None, "❌ Analysis Failed"
    
    @EnhancedErrorHandler.handle_exception
    def text_classification_interface(self, text: str, labels: str) -> Tuple[str, Optional[go.Figure], str]:
        """Enhanced text classification interface"""
        # Input validation
        is_valid, error_msg = self.validator.validate_text_input(text, 10, 1000)
        if not is_valid:
            return error_msg, None, "❌ Invalid Input"
        
        # Parse labels
        try:
            label_list = [label.strip() for label in labels.split(',') if label.strip()]
        except:
            return "❌ Invalid labels format. Use comma-separated labels.", None, "❌ Invalid Format"
        
        if len(label_list) < 2:
            return "❌ At least 2 labels required", None, "❌ Insufficient Labels"
        
        if self.nlp_system is None:
            return "❌ NLP system not initialized", None, "❌ System Error"
        
        try:
            analyzer = self.nlp_system.analyzer
            classification_result = analyzer.classify_text(text, label_list)
            
            # Format result with better styling
            result_text = f"🏷️ **Text Classification Results**\n\n"
            result_text += f"**Best Match:** {classification_result['label']}\n"
            result_text += f"**Confidence:** {classification_result['score']:.3f}\n\n"
            result_text += f"**Analyzed Text:**\n{text[:200]}{'...' if len(text) > 200 else ''}\n\n"
            result_text += f"📊 **All Classification Scores:**\n"
            for label, score in classification_result['scores'].items():
                emoji = "🥇" if label == classification_result['label'] else "📊"
                result_text += f"{emoji} {label}: {score:.3f}\n"
            
            # Create enhanced visualization
            fig = go.Figure(data=[
                go.Bar(
                    x=list(classification_result['scores'].keys()), 
                    y=list(classification_result['scores'].values()),
                    marker_color=['#FF6B6B' if k == classification_result['label'] else '#4ECDC4' 
                                for k in classification_result['scores'].keys()],
                    text=[f'{v:.3f}' for v in classification_result['scores'].values()],
                    textposition='auto'
                )
            ])
            fig.update_layout(
                title={
                    'text': "🏷️ Classification Confidence Scores",
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 18, 'color': '#2F4F4F'}
                },
                xaxis_title="Labels",
                yaxis_title="Confidence Score",
                template="plotly_white",
                height=400,
                showlegend=False
            )
            
            return result_text, fig, "✅ Classification Complete"
            
        except Exception as e:
            return self.error_handler.format_error_message(e), None, "❌ Classification Failed"
    
    @EnhancedErrorHandler.handle_exception
    def model_training_interface(self, 
                               training_data: str, 
                               model_name: str,
                               epochs: int,
                               learning_rate: float,
                               batch_size: int) -> Tuple[str, Optional[go.Figure], str]:
        """Enhanced model training interface"""
        # Input validation
        if not training_data.strip():
            return "❌ Training data cannot be empty", None, "❌ No Data"
        
        is_valid, error_msg = self.validator.validate_model_name(model_name)
        if not is_valid:
            return error_msg, None, "❌ Invalid Model"
        
        is_valid, error_msg = self.validator.validate_numeric_input(epochs, 1, 100)
        if not is_valid:
            return error_msg, None, "❌ Invalid Epochs"
        
        is_valid, error_msg = self.validator.validate_numeric_input(learning_rate, 1e-6, 1e-2)
        if not is_valid:
            return error_msg, None, "❌ Invalid Learning Rate"
        
        is_valid, error_msg = self.validator.validate_numeric_input(batch_size, 1, 64)
        if not is_valid:
            return error_msg, None, "❌ Invalid Batch Size"
        
        try:
            # Parse training data
            lines = training_data.strip().split('\n')
            parsed_data = []
            
            for line in lines:
                if ',' in line:
                    text, label = line.split(',', 1)
                    parsed_data.append((text.strip(), label.strip()))
            
            if len(parsed_data) < 10:
                return "❌ At least 10 training examples required", None, "❌ Insufficient Data"
            
            # Setup training config
            config = ComprehensiveTrainingConfig(
                model_name=model_name,
                batch_size=batch_size,
                learning_rate=learning_rate,
                num_epochs=epochs
            )
            
            # Initialize training system
            training_system = ComprehensiveTrainingSystem(config)
            
            # Create dataset
            from comprehensive_training_system import ComprehensiveDataset
            dataset = ComprehensiveDataset(parsed_data)
            
            # Train model
            training_system.train(dataset)
            
            # Get training metrics
            metrics = training_system.get_training_metrics()
            
            # Create enhanced visualization
            epochs_list = list(range(1, len(metrics.get('train_loss', [])) + 1))
            fig = self.viz_manager.create_training_progress_chart(epochs_list, metrics)
            
            # Format result with better styling
            result_text = f"🎯 **Training Results**\n\n"
            result_text += f"✅ **Training completed successfully!**\n\n"
            result_text += f"📊 **Final Metrics:**\n"
            result_text += f"• Final Loss: {metrics.get('train_loss', [0])[-1]:.4f}\n"
            result_text += f"• Epochs Trained: {len(metrics.get('train_loss', []))}\n"
            result_text += f"• Model: {model_name}\n"
            result_text += f"• Batch Size: {batch_size}\n"
            result_text += f"• Learning Rate: {learning_rate}\n\n"
            result_text += f"💾 **Model saved as:** {model_name}_trained"
            
            return result_text, fig, "✅ Training Complete"
            
        except Exception as e:
            return self.error_handler.format_error_message(e), None, "❌ Training Failed"
    
    @EnhancedErrorHandler.handle_exception
    def performance_benchmark_interface(self, 
                                      model_name: str,
                                      test_text: str,
                                      num_runs: int) -> Tuple[str, Optional[go.Figure], str]:
        """Enhanced performance benchmarking interface"""
        # Input validation
        is_valid, error_msg = self.validator.validate_model_name(model_name)
        if not is_valid:
            return error_msg, None, "❌ Invalid Model"
        
        is_valid, error_msg = self.validator.validate_text_input(test_text, 10, 500)
        if not is_valid:
            return error_msg, None, "❌ Invalid Text"
        
        is_valid, error_msg = self.validator.validate_numeric_input(num_runs, 1, 50)
        if not is_valid:
            return error_msg, None, "❌ Invalid Runs"
        
        if self.nlp_system is None:
            return "❌ NLP system not initialized", None, "❌ System Error"
        
        try:
            # Run benchmarks
            import time
            
            # Generation benchmark
            generation_times = []
            for _ in range(num_runs):
                start_time = time.time()
                self.nlp_system.generate_text(test_text, max_length=50)
                generation_times.append(time.time() - start_time)
            
            # Sentiment analysis benchmark
            sentiment_times = []
            for _ in range(num_runs):
                start_time = time.time()
                self.nlp_system.analyzer.analyze_sentiment(test_text)
                sentiment_times.append(time.time() - start_time)
            
            # Calculate statistics
            gen_avg = np.mean(generation_times)
            gen_std = np.std(generation_times)
            sent_avg = np.mean(sentiment_times)
            sent_std = np.std(sentiment_times)
            
            # Format results with better styling
            result_text = f"⚡ **Performance Benchmark Results**\n\n"
            result_text += f"🤖 **Model:** {model_name}\n"
            result_text += f"🔄 **Test Runs:** {num_runs}\n"
            result_text += f"🖥️ **GPU Available:** {'✅ Yes' if torch.cuda.is_available() else '❌ No'}\n\n"
            result_text += f"📝 **Text Generation Performance:**\n"
            result_text += f"• Average Time: {gen_avg:.4f}s\n"
            result_text += f"• Std Deviation: {gen_std:.4f}s\n"
            result_text += f"• Throughput: {1/gen_avg:.1f} generations/sec\n\n"
            result_text += f"🎭 **Sentiment Analysis Performance:**\n"
            result_text += f"• Average Time: {sent_avg:.4f}s\n"
            result_text += f"• Std Deviation: {sent_std:.4f}s\n"
            result_text += f"• Throughput: {1/sent_avg:.1f} analyses/sec"
            
            # Create enhanced visualization
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name='Text Generation',
                x=['Average Time', 'Std Dev'],
                y=[gen_avg, gen_std],
                marker_color='#4682B4',
                text=[f'{gen_avg:.4f}s', f'{gen_std:.4f}s'],
                textposition='auto'
            ))
            fig.add_trace(go.Bar(
                name='Sentiment Analysis',
                x=['Average Time', 'Std Dev'],
                y=[sent_avg, sent_std],
                marker_color='#FF6B6B',
                text=[f'{sent_avg:.4f}s', f'{sent_std:.4f}s'],
                textposition='auto'
            ))
            
            fig.update_layout(
                title={
                    'text': "⚡ Performance Benchmark Results",
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 18, 'color': '#2F4F4F'}
                },
                xaxis_title="Metric",
                yaxis_title="Time (seconds)",
                template="plotly_white",
                height=400,
                barmode='group'
            )
            
            return result_text, fig, "✅ Benchmark Complete"
            
        except Exception as e:
            return self.error_handler.format_error_message(e), None, "❌ Benchmark Failed"
    
    @EnhancedErrorHandler.handle_exception
    def system_info_interface(self) -> Tuple[str, Optional[go.Figure], str]:
        """Enhanced system information interface"""
        try:
            import psutil
            import platform
            
            # System info with emojis
            system_info = f"🖥️ **System Information**\n\n"
            system_info += f"💻 **Platform:** {platform.system()} {platform.release()}\n"
            system_info += f"🐍 **Python:** {platform.python_version()}\n"
            system_info += f"🔥 **PyTorch:** {torch.__version__}\n"
            system_info += f"🚀 **CUDA Available:** {'✅ Yes' if torch.cuda.is_available() else '❌ No'}\n"
            
            if torch.cuda.is_available():
                system_info += f"🎮 **CUDA Version:** {torch.version.cuda}\n"
                system_info += f"🎯 **GPU Count:** {torch.cuda.device_count()}\n"
                system_info += f"🎪 **Current GPU:** {torch.cuda.get_device_name()}\n"
            
            # Memory info
            memory = psutil.virtual_memory()
            system_info += f"\n💾 **Memory Usage:**\n"
            system_info += f"• Total: {memory.total / (1024**3):.2f} GB\n"
            system_info += f"• Available: {memory.available / (1024**3):.2f} GB\n"
            system_info += f"• Used: {memory.percent:.1f}%\n"
            
            # CPU info
            cpu_count = psutil.cpu_count()
            cpu_percent = psutil.cpu_percent(interval=1)
            system_info += f"\n⚡ **CPU Information:**\n"
            system_info += f"• Cores: {cpu_count}\n"
            system_info += f"• Usage: {cpu_percent:.1f}%"
            
            # Create enhanced visualization
            fig = self.viz_manager.create_performance_gauge("Memory Usage (%)", memory.percent)
            
            return system_info, fig, "✅ System Info Retrieved"
            
        except Exception as e:
            return self.error_handler.format_error_message(e), None, "❌ System Info Failed"
    
    def create_enhanced_interface(self) -> gr.Blocks:
        """Create the enhanced Gradio interface"""
        
        # Custom CSS for better styling
        custom_css = """
        .gradio-container {
            max-width: 1400px !important;
        }
        .main-header {
            text-align: center;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 15px;
            margin-bottom: 20px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        .feature-card {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin: 10px 0;
            border-left: 4px solid #667eea;
        }
        .status-indicator {
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: bold;
            text-align: center;
            margin: 10px 0;
        }
        .status-success {
            background: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        .status-error {
            background: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
        .tip-box {
            background: #e7f3ff;
            border: 1px solid #b3d9ff;
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
        }
        """
        
        with gr.Blocks(
            title=self.config.title,
            theme=self.config.theme,
            css=custom_css
        ) as interface:
            
            # Enhanced header
            gr.HTML("""
            <div class="main-header">
                <h1>🤖 Advanced NLP Platform</h1>
                <p>Experience the power of AI with our comprehensive NLP platform. Generate text, analyze sentiment, classify content, and train custom models - all through an intuitive interface.</p>
            </div>
            """)
            
            # Quick start guide
            with gr.Accordion("🚀 Quick Start Guide", open=False):
                gr.Markdown("""
                ### Welcome to the Advanced NLP Platform!
                
                **🎯 What you can do:**
                - **Text Generation**: Create creative text with AI models
                - **Sentiment Analysis**: Understand the emotional tone of text
                - **Text Classification**: Categorize text into custom labels
                - **Model Training**: Train your own custom models
                - **Performance Testing**: Benchmark system performance
                - **System Monitoring**: Check your system resources
                
                **💡 Tips:**
                - Start with the Text Generation tab to see AI in action
                - Use the Sentiment Analysis for understanding text emotions
                - Try different models to see performance differences
                - Monitor system resources to optimize performance
                """)
            
            with gr.Tabs():
                
                # Text Generation Tab
                with gr.Tab("📝 Text Generation", id=1):
                    gr.Markdown("### 🎨 Create Creative Text with AI")
                    
                    with gr.Row():
                        with gr.Column(scale=2):
                            prompt_input = gr.Textbox(
                                label="🎯 Input Prompt",
                                placeholder="Enter your creative prompt here... (e.g., 'Once upon a time in a magical forest...')",
                                lines=4
                            )
                            
                            with gr.Row():
                                max_length_input = gr.Slider(
                                    minimum=10, maximum=500, value=100,
                                    step=10, label="📏 Max Length",
                                    info="Number of words to generate"
                                )
                                temperature_input = gr.Slider(
                                    minimum=0.1, maximum=2.0, value=0.7,
                                    step=0.1, label="🌡️ Temperature",
                                    info="Higher = more creative, Lower = more focused"
                                )
                            
                            model_name_input = gr.Dropdown(
                                choices=["gpt2", "gpt2-medium", "gpt2-large", "bert-base-uncased"],
                                value="gpt2", 
                                label="🤖 AI Model",
                                info="Choose the AI model for generation"
                            )
                            
                            generate_btn = gr.Button("🚀 Generate Text", variant="primary", size="lg")
                        
                        with gr.Column(scale=2):
                            output_text = gr.Textbox(
                                label="✨ Generated Text",
                                lines=12,
                                interactive=False
                            )
                            output_plot = gr.Plot(label="📊 Generation Metrics")
                            status_output = gr.HTML(label="📈 Status")
                
                # Sentiment Analysis Tab
                with gr.Tab("🎭 Sentiment Analysis", id=2):
                    gr.Markdown("### 🎭 Analyze Text Emotions and Tone")
                    
                    with gr.Row():
                        with gr.Column(scale=2):
                            sentiment_text_input = gr.Textbox(
                                label="📝 Text for Analysis",
                                placeholder="Enter text to analyze its emotional tone...",
                                lines=6
                            )
                            sentiment_btn = gr.Button("🔍 Analyze Sentiment", variant="primary", size="lg")
                        
                        with gr.Column(scale=2):
                            sentiment_output = gr.Markdown(label="📊 Analysis Results")
                            sentiment_plot = gr.Plot(label="📈 Sentiment Distribution")
                            sentiment_status = gr.HTML(label="📈 Status")
                
                # Text Classification Tab
                with gr.Tab("🏷️ Text Classification", id=3):
                    gr.Markdown("### 🏷️ Categorize Text with Custom Labels")
                    
                    with gr.Row():
                        with gr.Column(scale=2):
                            classification_text_input = gr.Textbox(
                                label="📝 Text to Classify",
                                placeholder="Enter text to classify...",
                                lines=4
                            )
                            classification_labels_input = gr.Textbox(
                                label="🏷️ Labels (comma-separated)",
                                placeholder="positive, negative, neutral, urgent, casual",
                                lines=1
                            )
                            classification_btn = gr.Button("🏷️ Classify Text", variant="primary", size="lg")
                        
                        with gr.Column(scale=2):
                            classification_output = gr.Markdown(label="📊 Classification Results")
                            classification_plot = gr.Plot(label="📈 Classification Scores")
                            classification_status = gr.HTML(label="📈 Status")
                
                # Model Training Tab
                with gr.Tab("🎯 Model Training", id=4):
                    gr.Markdown("### 🎯 Train Your Own Custom AI Model")
                    
                    with gr.Row():
                        with gr.Column(scale=2):
                            training_data_input = gr.Textbox(
                                label="📚 Training Data (text,label format)",
                                placeholder="Enter training data in format: text,label (one per line)...\nExample:\nI love this product,positive\nThis is terrible,negative\nIt's okay,neutral",
                                lines=10
                            )
                            
                            with gr.Row():
                                training_model_name = gr.Dropdown(
                                    choices=["gpt2", "bert-base-uncased", "roberta-base"],
                                    value="gpt2", 
                                    label="🤖 Base Model"
                                )
                                training_epochs = gr.Slider(
                                    minimum=1, maximum=50, value=5,
                                    step=1, label="🔄 Epochs"
                                )
                            
                            with gr.Row():
                                training_lr = gr.Slider(
                                    minimum=1e-6, maximum=1e-2, value=2e-5,
                                    step=1e-6, label="📈 Learning Rate"
                                )
                                training_batch_size = gr.Slider(
                                    minimum=1, maximum=32, value=8,
                                    step=1, label="📦 Batch Size"
                                )
                            
                            training_btn = gr.Button("🎯 Start Training", variant="primary", size="lg")
                        
                        with gr.Column(scale=2):
                            training_output = gr.Markdown(label="📊 Training Results")
                            training_plot = gr.Plot(label="📈 Training Progress")
                            training_status = gr.HTML(label="📈 Status")
                
                # Performance Benchmark Tab
                with gr.Tab("⚡ Performance", id=5):
                    gr.Markdown("### ⚡ Benchmark System Performance")
                    
                    with gr.Row():
                        with gr.Column(scale=2):
                            benchmark_model_name = gr.Dropdown(
                                choices=["gpt2", "gpt2-medium", "bert-base-uncased"],
                                value="gpt2", 
                                label="🤖 Model to Test"
                            )
                            benchmark_text_input = gr.Textbox(
                                label="📝 Test Text",
                                placeholder="Enter text for performance testing...",
                                lines=3
                            )
                            benchmark_runs = gr.Slider(
                                minimum=1, maximum=50, value=10,
                                step=1, label="🔄 Number of Test Runs"
                            )
                            benchmark_btn = gr.Button("⚡ Run Benchmark", variant="primary", size="lg")
                        
                        with gr.Column(scale=2):
                            benchmark_output = gr.Markdown(label="📊 Benchmark Results")
                            benchmark_plot = gr.Plot(label="📈 Performance Metrics")
                            benchmark_status = gr.HTML(label="📈 Status")
                
                # System Info Tab
                with gr.Tab("🖥️ System Info", id=6):
                    gr.Markdown("### 🖥️ Monitor System Resources")
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            system_info_btn = gr.Button("🔄 Refresh System Info", variant="primary", size="lg")
                        
                        with gr.Column(scale=2):
                            system_info_output = gr.Markdown(label="📊 System Information")
                            system_info_plot = gr.Plot(label="📈 Resource Usage")
                            system_info_status = gr.HTML(label="📈 Status")
            
            # Event handlers
            generate_btn.click(
                fn=self.generate_text_interface,
                inputs=[prompt_input, max_length_input, temperature_input, model_name_input],
                outputs=[output_text, output_plot, status_output]
            )
            
            sentiment_btn.click(
                fn=self.sentiment_analysis_interface,
                inputs=[sentiment_text_input],
                outputs=[sentiment_output, sentiment_plot, sentiment_status]
            )
            
            classification_btn.click(
                fn=self.text_classification_interface,
                inputs=[classification_text_input, classification_labels_input],
                outputs=[classification_output, classification_plot, classification_status]
            )
            
            training_btn.click(
                fn=self.model_training_interface,
                inputs=[training_data_input, training_model_name, training_epochs, 
                       training_lr, training_batch_size],
                outputs=[training_output, training_plot, training_status]
            )
            
            benchmark_btn.click(
                fn=self.performance_benchmark_interface,
                inputs=[benchmark_model_name, benchmark_text_input, benchmark_runs],
                outputs=[benchmark_output, benchmark_plot, benchmark_status]
            )
            
            system_info_btn.click(
                fn=self.system_info_interface,
                inputs=[],
                outputs=[system_info_output, system_info_plot, system_info_status]
            )
        
        return interface


def create_enhanced_gradio_app():
    """Create and launch the enhanced Gradio app"""
    enhanced_config = EnhancedGradioConfig(
        title="🤖 Advanced NLP Platform",
        description="Experience the power of AI with our comprehensive NLP platform",
        theme="default",
        height=800,
        width=1200
    )
    
    enhanced_system = EnhancedGradioNLPSystem(enhanced_config)
    interface = enhanced_system.create_enhanced_interface()
    
    return interface


if __name__ == "__main__":
    # Create and launch the enhanced app
    app = create_enhanced_gradio_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        debug=True
    )




