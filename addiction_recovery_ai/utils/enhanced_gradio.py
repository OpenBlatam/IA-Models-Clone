"""
Enhanced Gradio Interface with Advanced Features
"""

import gradio as gr
from typing import Optional, Dict, List, Any
import logging
import time
import json

logger = logging.getLogger(__name__)


class EnhancedRecoveryGradio:
    """Enhanced Gradio interface with advanced features"""
    
    def __init__(self, analyzer, engine=None):
        """
        Initialize enhanced Gradio interface
        
        Args:
            analyzer: Recovery analyzer
            engine: Ultra-fast engine (optional)
        """
        self.analyzer = analyzer
        self.engine = engine or analyzer
        self.history = []
    
    def create_interface(self) -> gr.Blocks:
        """Create enhanced Gradio interface"""
        with gr.Blocks(title="Enhanced Recovery AI", theme=gr.themes.Soft()) as demo:
            gr.Markdown("# 🚀 Enhanced Addiction Recovery AI")
            gr.Markdown("Advanced AI-powered recovery support system")
            
            with gr.Tabs():
                # Tab 1: Quick Analysis
                with gr.Tab("⚡ Quick Analysis"):
                    with gr.Row():
                        with gr.Column():
                            text_input = gr.Textbox(
                                label="Journal Entry",
                                placeholder="How are you feeling today?",
                                lines=5
                            )
                            analyze_btn = gr.Button("Analyze Sentiment", variant="primary")
                        
                        with gr.Column():
                            sentiment_output = gr.JSON(label="Sentiment Analysis")
                    
                    analyze_btn.click(
                        fn=self.analyze_sentiment_wrapper,
                        inputs=[text_input],
                        outputs=[sentiment_output]
                    )
                
                # Tab 2: Progress Tracking
                with gr.Tab("📊 Progress Tracking"):
                    with gr.Row():
                        with gr.Column():
                            days_sober = gr.Number(label="Days Sober", value=30)
                            cravings = gr.Slider(0, 10, 5, label="Cravings Level")
                            stress = gr.Slider(0, 10, 5, label="Stress Level")
                            support = gr.Slider(0, 10, 5, label="Support Level")
                            mood = gr.Slider(0, 10, 5, label="Mood Score")
                            
                            predict_btn = gr.Button("Predict Progress", variant="primary")
                        
                        with gr.Column():
                            progress_output = gr.JSON(label="Progress Prediction")
                            progress_chart = gr.Plot(label="Progress Chart")
                    
                    predict_btn.click(
                        fn=self.predict_progress_wrapper,
                        inputs=[days_sober, cravings, stress, support, mood],
                        outputs=[progress_output, progress_chart]
                    )
                
                # Tab 3: Relapse Risk
                with gr.Tab("⚠️ Relapse Risk"):
                    with gr.Row():
                        with gr.Column():
                            risk_text = gr.Textbox(
                                label="Recent Patterns",
                                placeholder="Describe recent triggers and challenges...",
                                lines=5
                            )
                            check_risk_btn = gr.Button("Check Risk", variant="primary")
                        
                        with gr.Column():
                            risk_output = gr.JSON(label="Risk Assessment")
                            risk_gauge = gr.Label(label="Risk Level")
                    
                    check_risk_btn.click(
                        fn=self.check_risk_wrapper,
                        inputs=[risk_text],
                        outputs=[risk_output, risk_gauge]
                    )
                
                # Tab 4: AI Coaching
                with gr.Tab("🤖 AI Coaching"):
                    with gr.Row():
                        with gr.Column():
                            situation = gr.Textbox(
                                label="Your Situation",
                                placeholder="What challenge are you facing?",
                                lines=3
                            )
                            challenge = gr.Textbox(
                                label="Current Challenge",
                                placeholder="Specific challenge...",
                                lines=2
                            )
                            coaching_btn = gr.Button("Get Coaching", variant="primary")
                        
                        with gr.Column():
                            coaching_output = gr.Textbox(
                                label="Personalized Coaching",
                                lines=10
                            )
                    
                    coaching_btn.click(
                        fn=self.get_coaching_wrapper,
                        inputs=[situation, challenge],
                        outputs=[coaching_output]
                    )
                
                # Tab 5: Batch Processing
                with gr.Tab("📦 Batch Processing"):
                    with gr.Row():
                        with gr.Column():
                            batch_input = gr.Dataframe(
                                label="Batch Data",
                                headers=["Days", "Cravings", "Stress", "Support", "Mood"]
                            )
                            batch_btn = gr.Button("Process Batch", variant="primary")
                        
                        with gr.Column():
                            batch_output = gr.Dataframe(label="Batch Results")
                            batch_stats = gr.JSON(label="Statistics")
                    
                    batch_btn.click(
                        fn=self.process_batch_wrapper,
                        inputs=[batch_input],
                        outputs=[batch_output, batch_stats]
                    )
                
                # Tab 6: Benchmark
                with gr.Tab("⚙️ Benchmark"):
                    with gr.Row():
                        benchmark_btn = gr.Button("Run Benchmark", variant="primary")
                        benchmark_output = gr.JSON(label="Performance Metrics")
                    
                    benchmark_btn.click(
                        fn=self.run_benchmark,
                        inputs=[],
                        outputs=[benchmark_output]
                    )
            
            # History
            with gr.Accordion("📜 History", open=False):
                history_output = gr.JSON(label="Recent Analyses")
                clear_btn = gr.Button("Clear History")
                clear_btn.click(fn=self.clear_history, outputs=[history_output])
        
        return demo
    
    def analyze_sentiment_wrapper(self, text: str) -> Dict:
        """Wrapper for sentiment analysis"""
        start = time.time()
        result = self.engine.analyze_sentiment(text)
        elapsed = (time.time() - start) * 1000
        
        result["inference_time_ms"] = elapsed
        self.history.append({"type": "sentiment", "result": result, "timestamp": time.time()})
        
        return result
    
    def predict_progress_wrapper(
        self, days, cravings, stress, support, mood
    ) -> tuple:
        """Wrapper for progress prediction"""
        features = {
            "days_sober": days,
            "cravings_level": cravings,
            "stress_level": stress,
            "support_level": support,
            "mood_score": mood
        }
        
        start = time.time()
        progress = self.engine.predict_progress(features)
        elapsed = (time.time() - start) * 1000
        
        result = {
            "progress": progress,
            "percentage": f"{progress * 100:.1f}%",
            "inference_time_ms": elapsed
        }
        
        # Simple chart data
        chart_data = {
            "progress": progress * 100,
            "days": days
        }
        
        self.history.append({"type": "progress", "result": result, "timestamp": time.time()})
        
        return result, chart_data
    
    def check_risk_wrapper(self, text: str) -> tuple:
        """Wrapper for risk check"""
        # Simplified - in practice, extract features from text
        sequence = [{
            "cravings_level": 5,
            "stress_level": 5,
            "mood_score": 5,
            "triggers_count": 1,
            "consumed": 0.0
        }] * 30
        
        start = time.time()
        risk = self.engine.predict_relapse_risk(sequence)
        elapsed = (time.time() - start) * 1000
        
        result = {
            "risk": risk,
            "risk_level": "High" if risk > 0.7 else "Medium" if risk > 0.4 else "Low",
            "inference_time_ms": elapsed
        }
        
        label = {"High": 2, "Medium": 1, "Low": 0}[result["risk_level"]]
        
        self.history.append({"type": "risk", "result": result, "timestamp": time.time()})
        
        return result, label
    
    def get_coaching_wrapper(self, situation: str, challenge: str) -> str:
        """Wrapper for coaching"""
        days_sober = 30  # Default
        coaching = self.engine.generate_coaching(
            situation, days_sober, challenge
        )
        
        self.history.append({
            "type": "coaching",
            "coaching": coaching,
            "timestamp": time.time()
        })
        
        return coaching
    
    def process_batch_wrapper(self, df) -> tuple:
        """Wrapper for batch processing"""
        if df is None or len(df) == 0:
            return None, {}
        
        # Convert to sequences
        sequences = []
        for _, row in df.iterrows():
            seq = [{
                "cravings_level": row.get("Cravings", 5),
                "stress_level": row.get("Stress", 5),
                "mood_score": row.get("Mood", 5),
                "triggers_count": 1,
                "consumed": 0.0
            }] * 30
            sequences.append(seq)
        
        # Batch predict
        start = time.time()
        risks = self.engine.predict_relapse_batch(sequences)
        elapsed = (time.time() - start) * 1000
        
        # Create results dataframe
        results_df = df.copy()
        results_df["Risk"] = risks
        results_df["Risk_Level"] = [
            "High" if r > 0.7 else "Medium" if r > 0.4 else "Low"
            for r in risks
        ]
        
        stats = {
            "total": len(risks),
            "avg_risk": sum(risks) / len(risks),
            "high_risk": sum(1 for r in risks if r > 0.7),
            "inference_time_ms": elapsed,
            "throughput": len(risks) / (elapsed / 1000)
        }
        
        return results_df, stats
    
    def run_benchmark(self) -> Dict:
        """Run performance benchmark"""
        if hasattr(self.engine, "benchmark"):
            return self.engine.benchmark(num_runs=100)
        return {"error": "Benchmark not available"}
    
    def clear_history(self) -> List:
        """Clear history"""
        self.history = []
        return []


def create_enhanced_gradio_app(analyzer, engine=None) -> gr.Blocks:
    """Create enhanced Gradio app"""
    interface = EnhancedRecoveryGradio(analyzer, engine)
    return interface.create_interface()

