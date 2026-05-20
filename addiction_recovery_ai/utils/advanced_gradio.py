"""
Advanced Gradio Interface with Enhanced Features
Modern, interactive, and production-ready Gradio apps
"""

import gradio as gr
import torch
import numpy as np
from typing import Optional, Dict, List, Any, Tuple
import logging
from PIL import Image
import io
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class AdvancedRecoveryGradio:
    """
    Advanced Gradio interface with:
    - Real-time inference
    - Visualization
    - Model comparison
    - Performance metrics
    - Interactive charts
    """
    
    def __init__(
        self,
        sentiment_analyzer=None,
        progress_predictor=None,
        relapse_predictor=None,
        llm_coach=None,
        visualizer=None
    ):
        """
        Initialize advanced Gradio interface
        
        Args:
            sentiment_analyzer: Sentiment analyzer model
            progress_predictor: Progress predictor model
            relapse_predictor: Relapse risk predictor
            llm_coach: LLM coaching model
            visualizer: Progress visualizer
        """
        self.sentiment_analyzer = sentiment_analyzer
        self.progress_predictor = progress_predictor
        self.relapse_predictor = relapse_predictor
        self.llm_coach = llm_coach
        self.visualizer = visualizer
        
        # History tracking
        self.history = []
    
    def create_interface(self) -> gr.Blocks:
        """Create advanced Gradio interface"""
        theme = gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="gray",
            font=("Helvetica", "Arial", "sans-serif")
        )
        
        with gr.Blocks(theme=theme, title="Addiction Recovery AI - Advanced") as interface:
            gr.Markdown(
                """
                # 🎯 Addiction Recovery AI - Advanced Deep Learning Platform
                
                Powered by PyTorch, Transformers, and Diffusion Models
                
                **Features**: Real-time inference, progress tracking, relapse prediction, AI coaching, and visualization
                """
            )
            
            with gr.Tabs():
                # Tab 1: Sentiment Analysis
                with gr.Tab("📊 Sentiment Analysis"):
                    self._create_sentiment_tab()
                
                # Tab 2: Progress Prediction
                with gr.Tab("📈 Progress Prediction"):
                    self._create_progress_tab()
                
                # Tab 3: Relapse Risk
                with gr.Tab("⚠️ Relapse Risk"):
                    self._create_relapse_tab()
                
                # Tab 4: AI Coaching
                with gr.Tab("🤖 AI Coaching"):
                    self._create_coaching_tab()
                
                # Tab 5: Visualization
                with gr.Tab("🎨 Visualization"):
                    self._create_visualization_tab()
                
                # Tab 6: Model Comparison
                with gr.Tab("🔬 Model Comparison"):
                    self._create_comparison_tab()
                
                # Tab 7: Performance Metrics
                with gr.Tab("⚡ Performance"):
                    self._create_performance_tab()
            
            # Footer
            gr.Markdown(
                """
                ---
                **Note**: This is a support tool and does NOT replace professional medical advice.
                """
            )
        
        return interface
    
    def _create_sentiment_tab(self):
        """Create sentiment analysis tab"""
        with gr.Row():
            with gr.Column(scale=2):
                text_input = gr.Textbox(
                    label="Enter text to analyze",
                    placeholder="How are you feeling today?",
                    lines=5
                )
                analyze_btn = gr.Button("Analyze Sentiment", variant="primary")
                
                with gr.Row():
                    batch_input = gr.File(
                        label="Upload batch file (JSON)",
                        file_types=[".json"]
                    )
                    batch_btn = gr.Button("Analyze Batch", variant="secondary")
            
            with gr.Column(scale=1):
                sentiment_output = gr.Label(label="Sentiment")
                confidence_output = gr.Number(label="Confidence", precision=3)
                probabilities_output = gr.JSON(label="All Probabilities")
                
                # Attention visualization
                attention_plot = gr.Plot(label="Attention Weights")
        
        def analyze_sentiment(text: str, return_attention: bool = True):
            """Analyze sentiment"""
            if not text or not self.sentiment_analyzer:
                return None, 0, {}, None
            
            try:
                result = self.sentiment_analyzer.analyze(text, return_attention=return_attention)
                
                # Format output
                label = result.get("label", "NEUTRAL")
                score = result.get("score", 0.5)
                probs = result.get("probabilities", {})
                
                # Create attention plot if available
                attention_img = None
                if return_attention and "attention_weights" in result:
                    attention_img = self._plot_attention(result["attention_weights"])
                
                return {
                    label: score
                }, score, probs, attention_img
            except Exception as e:
                logger.error(f"Sentiment analysis failed: {e}")
                return None, 0, {}, None
        
        analyze_btn.click(
            fn=analyze_sentiment,
            inputs=[text_input],
            outputs=[sentiment_output, confidence_output, probabilities_output, attention_plot]
        )
    
    def _create_progress_tab(self):
        """Create progress prediction tab"""
        with gr.Row():
            with gr.Column():
                days_sober = gr.Number(label="Days Sober", value=30, precision=0)
                cravings_level = gr.Slider(0, 10, value=3, label="Cravings Level")
                stress_level = gr.Slider(0, 10, value=4, label="Stress Level")
                support_level = gr.Slider(0, 10, value=8, label="Support Level")
                mood_score = gr.Slider(0, 10, value=7, label="Mood Score")
                
                predict_btn = gr.Button("Predict Progress", variant="primary")
            
            with gr.Column():
                progress_output = gr.Number(label="Progress Score", precision=3)
                progress_gauge = gr.Gauge(
                    label="Progress",
                    minimum=0,
                    maximum=1,
                    value=0
                )
                progress_chart = gr.Plot(label="Progress Over Time")
        
        def predict_progress(days, cravings, stress, support, mood):
            """Predict progress"""
            if not self.progress_predictor:
                return 0, 0, None
            
            try:
                # Normalize features
                features = torch.tensor([[
                    days / 365,  # Normalize days
                    cravings / 10,
                    stress / 10,
                    support / 10,
                    mood / 10
                ]], dtype=torch.float32)
                
                progress = self.progress_predictor.predict_progress(features)
                
                # Create chart
                chart = self._plot_progress_timeline(progress)
                
                return progress, progress, chart
            except Exception as e:
                logger.error(f"Progress prediction failed: {e}")
                return 0, 0, None
        
        predict_btn.click(
            fn=predict_progress,
            inputs=[days_sober, cravings_level, stress_level, support_level, mood_score],
            outputs=[progress_output, progress_gauge, progress_chart]
        )
    
    def _create_relapse_tab(self):
        """Create relapse risk tab"""
        with gr.Row():
            with gr.Column():
                sequence_input = gr.Dataframe(
                    label="Daily Features (Sequence)",
                    headers=["Day", "Cravings", "Stress", "Mood", "Support", "Sleep"],
                    row_count=7,
                    col_count=6
                )
                predict_risk_btn = gr.Button("Predict Relapse Risk", variant="primary")
            
            with gr.Column():
                risk_output = gr.Number(label="Relapse Risk", precision=3)
                risk_gauge = gr.Gauge(
                    label="Risk Level",
                    minimum=0,
                    maximum=1,
                    value=0
                )
                risk_chart = gr.Plot(label="Risk Over Time")
                alert_output = gr.Markdown(label="Alert")
        
        def predict_risk(sequence_df):
            """Predict relapse risk"""
            if not self.relapse_predictor or sequence_df is None:
                return 0, 0, None, ""
            
            try:
                # Convert dataframe to tensor
                features = sequence_df.iloc[:, 1:].values.astype(np.float32)
                sequence_tensor = torch.tensor(features).unsqueeze(0)
                
                risk = self.relapse_predictor.predict_risk(sequence_tensor)
                
                # Create chart
                chart = self._plot_risk_timeline(sequence_df, risk)
                
                # Alert
                if risk > 0.7:
                    alert = "🔴 **HIGH RISK** - Immediate attention recommended"
                elif risk > 0.5:
                    alert = "🟡 **MODERATE RISK** - Monitor closely"
                else:
                    alert = "🟢 **LOW RISK** - Continue current approach"
                
                return risk, risk, chart, alert
            except Exception as e:
                logger.error(f"Relapse risk prediction failed: {e}")
                return 0, 0, None, ""
        
        predict_risk_btn.click(
            fn=predict_risk,
            inputs=[sequence_input],
            outputs=[risk_output, risk_gauge, risk_chart, alert_output]
        )
    
    def _create_coaching_tab(self):
        """Create AI coaching tab"""
        with gr.Row():
            with gr.Column():
                situation_input = gr.Textbox(
                    label="Current Situation",
                    placeholder="Describe your current situation...",
                    lines=3
                )
                days_sober_input = gr.Number(label="Days Sober", value=30, precision=0)
                challenge_input = gr.Textbox(
                    label="Current Challenge (Optional)",
                    placeholder="What challenge are you facing?",
                    lines=2
                )
                generate_btn = gr.Button("Generate Coaching", variant="primary")
            
            with gr.Column():
                coaching_output = gr.Textbox(
                    label="AI Coaching Message",
                    lines=10,
                    interactive=False
                )
                regenerate_btn = gr.Button("Regenerate", variant="secondary")
        
        def generate_coaching(situation, days, challenge):
            """Generate coaching message"""
            if not self.llm_coach or not situation:
                return ""
            
            try:
                message = self.llm_coach.generate_coaching_message(
                    user_situation=situation,
                    days_sober=int(days),
                    current_challenge=challenge if challenge else None
                )
                return message
            except Exception as e:
                logger.error(f"Coaching generation failed: {e}")
                return f"Error: {str(e)}"
        
        generate_btn.click(
            fn=generate_coaching,
            inputs=[situation_input, days_sober_input, challenge_input],
            outputs=[coaching_output]
        )
        
        regenerate_btn.click(
            fn=generate_coaching,
            inputs=[situation_input, days_sober_input, challenge_input],
            outputs=[coaching_output]
        )
    
    def _create_visualization_tab(self):
        """Create visualization tab"""
        with gr.Row():
            with gr.Column():
                prompt_input = gr.Textbox(
                    label="Visualization Prompt",
                    placeholder="Recovery journey, progress, motivation...",
                    lines=2
                )
                days_sober_viz = gr.Number(label="Days Sober", value=30, precision=0)
                progress_score_viz = gr.Slider(0, 1, value=0.75, label="Progress Score")
                
                generate_image_btn = gr.Button("Generate Image", variant="primary")
            
            with gr.Column():
                generated_image = gr.Image(label="Generated Progress Image")
                chart_type = gr.Radio(
                    ["Line", "Bar", "Radar"],
                    label="Chart Type",
                    value="Line"
                )
                generate_chart_btn = gr.Button("Generate Chart", variant="secondary")
                generated_chart = gr.Plot(label="Progress Chart")
        
        def generate_image(prompt, days, progress):
            """Generate progress image"""
            if not self.visualizer:
                return None
            
            try:
                image = self.visualizer.generate_progress_image(
                    prompt=prompt,
                    days_sober=int(days),
                    progress_score=progress
                )
                return image
            except Exception as e:
                logger.error(f"Image generation failed: {e}")
                return None
        
        def generate_chart(chart_type):
            """Generate progress chart"""
            # Mock data for demonstration
            days = list(range(1, 31))
            scores = np.random.rand(30).cumsum() / 30
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            if chart_type == "Line":
                ax.plot(days, scores, marker='o', linewidth=2)
                ax.fill_between(days, scores, alpha=0.3)
            elif chart_type == "Bar":
                ax.bar(days, scores, alpha=0.7)
            elif chart_type == "Radar":
                # Simplified radar
                categories = ["Progress", "Mood", "Support", "Health", "Stability"]
                values = [0.8, 0.7, 0.9, 0.75, 0.85]
                angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
                values += values[:1]
                angles += angles[:1]
                
                ax = plt.subplot(111, projection='polar')
                ax.plot(angles, values, 'o-', linewidth=2)
                ax.fill(angles, values, alpha=0.25)
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(categories)
            
            ax.set_xlabel("Days")
            ax.set_ylabel("Progress Score")
            ax.set_title("Recovery Progress")
            ax.grid(True, alpha=0.3)
            
            return fig
        
        generate_image_btn.click(
            fn=generate_image,
            inputs=[prompt_input, days_sober_viz, progress_score_viz],
            outputs=[generated_image]
        )
        
        generate_chart_btn.click(
            fn=generate_chart,
            inputs=[chart_type],
            outputs=[generated_chart]
        )
    
    def _create_comparison_tab(self):
        """Create model comparison tab"""
        gr.Markdown("### Model Performance Comparison")
        
        with gr.Row():
            model1_output = gr.JSON(label="Model 1 Metrics")
            model2_output = gr.JSON(label="Model 2 Metrics")
        
        comparison_chart = gr.Plot(label="Comparison Chart")
        
        def compare_models():
            """Compare models"""
            # Mock comparison data
            metrics = {
                "Model 1": {
                    "Accuracy": 0.85,
                    "Precision": 0.82,
                    "Recall": 0.88,
                    "F1": 0.85,
                    "Latency (ms)": 15
                },
                "Model 2": {
                    "Accuracy": 0.88,
                    "Precision": 0.85,
                    "Recall": 0.90,
                    "F1": 0.87,
                    "Latency (ms)": 20
                }
            }
            
            # Create comparison chart
            fig, ax = plt.subplots(figsize=(10, 6))
            models = list(metrics.keys())
            metric_names = ["Accuracy", "Precision", "Recall", "F1"]
            
            x = np.arange(len(metric_names))
            width = 0.35
            
            for i, model in enumerate(models):
                values = [metrics[model][m] for m in metric_names]
                ax.bar(x + i * width, values, width, label=model)
            
            ax.set_xlabel("Metrics")
            ax.set_ylabel("Score")
            ax.set_title("Model Comparison")
            ax.set_xticks(x + width / 2)
            ax.set_xticklabels(metric_names)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            return metrics["Model 1"], metrics["Model 2"], fig
        
        compare_btn = gr.Button("Compare Models", variant="primary")
        compare_btn.click(
            fn=compare_models,
            outputs=[model1_output, model2_output, comparison_chart]
        )
    
    def _create_performance_tab(self):
        """Create performance metrics tab"""
        gr.Markdown("### Real-time Performance Metrics")
        
        with gr.Row():
            with gr.Column():
                latency_output = gr.Number(label="Average Latency (ms)", precision=2)
                throughput_output = gr.Number(label="Throughput (req/s)", precision=2)
                gpu_memory_output = gr.Number(label="GPU Memory (GB)", precision=2)
            
            with gr.Column():
                performance_chart = gr.Plot(label="Performance Over Time")
        
        def get_performance():
            """Get performance metrics"""
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / 1024**3
            else:
                gpu_memory = 0
            
            # Mock metrics
            latency = 15.5
            throughput = 64.5
            
            # Create performance chart
            fig, ax = plt.subplots(figsize=(10, 6))
            time_points = list(range(1, 21))
            latency_data = np.random.normal(latency, 2, 20)
            throughput_data = np.random.normal(throughput, 5, 20)
            
            ax2 = ax.twinx()
            line1 = ax.plot(time_points, latency_data, 'b-', label='Latency (ms)')
            line2 = ax2.plot(time_points, throughput_data, 'r-', label='Throughput (req/s)')
            
            ax.set_xlabel("Time")
            ax.set_ylabel("Latency (ms)", color='b')
            ax2.set_ylabel("Throughput (req/s)", color='r')
            ax.tick_params(axis='y', labelcolor='b')
            ax2.tick_params(axis='y', labelcolor='r')
            
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax.legend(lines, labels, loc='upper left')
            
            ax.grid(True, alpha=0.3)
            
            return latency, throughput, gpu_memory, fig
        
        refresh_btn = gr.Button("Refresh Metrics", variant="primary")
        refresh_btn.click(
            fn=get_performance,
            outputs=[latency_output, throughput_output, gpu_memory_output, performance_chart]
        )
    
    def _plot_attention(self, attention_weights: List[List[float]]) -> plt.Figure:
        """Plot attention weights"""
        fig, ax = plt.subplots(figsize=(10, 6))
        attention_array = np.array(attention_weights)
        im = ax.imshow(attention_array, cmap='viridis', aspect='auto')
        ax.set_xlabel("Token Position")
        ax.set_ylabel("Head")
        ax.set_title("Attention Weights")
        plt.colorbar(im, ax=ax)
        return fig
    
    def _plot_progress_timeline(self, current_progress: float) -> plt.Figure:
        """Plot progress timeline"""
        fig, ax = plt.subplots(figsize=(10, 6))
        days = list(range(1, 31))
        progress = np.linspace(0, current_progress, 30)
        ax.plot(days, progress, marker='o', linewidth=2)
        ax.fill_between(days, progress, alpha=0.3)
        ax.set_xlabel("Days")
        ax.set_ylabel("Progress Score")
        ax.set_title("Progress Timeline")
        ax.grid(True, alpha=0.3)
        return fig
    
    def _plot_risk_timeline(self, sequence_df, current_risk: float) -> plt.Figure:
        """Plot risk timeline"""
        fig, ax = plt.subplots(figsize=(10, 6))
        days = sequence_df.iloc[:, 0].values
        risks = np.random.rand(len(days)) * 0.3 + current_risk - 0.15
        risks = np.clip(risks, 0, 1)
        ax.plot(days, risks, marker='o', linewidth=2, color='red')
        ax.axhline(y=0.7, color='r', linestyle='--', label='High Risk Threshold')
        ax.axhline(y=0.5, color='orange', linestyle='--', label='Moderate Risk Threshold')
        ax.fill_between(days, risks, alpha=0.3, color='red')
        ax.set_xlabel("Days")
        ax.set_ylabel("Risk Score")
        ax.set_title("Relapse Risk Timeline")
        ax.legend()
        ax.grid(True, alpha=0.3)
        return fig


def create_advanced_gradio_app(
    sentiment_analyzer=None,
    progress_predictor=None,
    relapse_predictor=None,
    llm_coach=None,
    visualizer=None
) -> gr.Blocks:
    """Factory function for advanced Gradio app"""
    interface = AdvancedRecoveryGradio(
        sentiment_analyzer=sentiment_analyzer,
        progress_predictor=progress_predictor,
        relapse_predictor=relapse_predictor,
        llm_coach=llm_coach,
        visualizer=visualizer
    )
    return interface.create_interface()













