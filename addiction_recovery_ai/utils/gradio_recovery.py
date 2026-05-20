"""
Gradio Interface for Addiction Recovery AI
"""

import gradio as gr
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class RecoveryGradioInterface:
    """Gradio interface for recovery AI"""
    
    def __init__(self, enhanced_analyzer, counseling_service=None):
        """
        Initialize Gradio interface
        
        Args:
            enhanced_analyzer: EnhancedAddictionAnalyzer instance
            counseling_service: CounselingService instance
        """
        self.analyzer = enhanced_analyzer
        self.counseling = counseling_service
    
    def create_interface(self) -> gr.Blocks:
        """Create Gradio interface"""
        with gr.Blocks(title="Addiction Recovery AI", theme=gr.themes.Soft()) as interface:
            gr.Markdown("# 🎯 Addiction Recovery AI - Deep Learning Powered")
            gr.Markdown("AI-powered recovery support system with deep learning")
            
            with gr.Tabs():
                # Sentiment Analysis Tab
                with gr.Tab("Sentiment Analysis"):
                    self._create_sentiment_tab()
                
                # Progress Prediction Tab
                with gr.Tab("Progress Prediction"):
                    self._create_progress_tab()
                
                # Relapse Risk Tab
                with gr.Tab("Relapse Risk"):
                    self._create_relapse_tab()
                
                # AI Coaching Tab
                with gr.Tab("AI Coaching"):
                    self._create_coaching_tab()
            
        return interface
    
    def _create_sentiment_tab(self):
        """Create sentiment analysis tab"""
        with gr.Row():
            with gr.Column():
                sentiment_input = gr.Textbox(
                    label="Enter your thoughts or journal entry",
                    lines=10,
                    placeholder="How are you feeling today?"
                )
                analyze_btn = gr.Button("Analyze Sentiment", variant="primary")
            
            with gr.Column():
                sentiment_result = gr.JSON(label="Sentiment Analysis")
                sentiment_label = gr.Textbox(
                    label="Sentiment",
                    interactive=False
                )
        
        def analyze_sentiment(text):
            if not self.analyzer:
                return {}, "Not available"
            
            result = self.analyzer.analyze_sentiment(text)
            label = result.get("label", "NEUTRAL")
            score = result.get("score", 0.5)
            
            return result, f"{label} (confidence: {score:.2%})"
        
        analyze_btn.click(
            fn=analyze_sentiment,
            inputs=sentiment_input,
            outputs=[sentiment_result, sentiment_label]
        )
    
    def _create_progress_tab(self):
        """Create progress prediction tab"""
        with gr.Row():
            with gr.Column():
                days_sober = gr.Number(label="Days Sober", value=30)
                cravings_level = gr.Slider(0, 10, value=5, label="Cravings Level")
                stress_level = gr.Slider(0, 10, value=5, label="Stress Level")
                support_level = gr.Slider(0, 10, value=7, label="Support Level")
                mood_score = gr.Slider(0, 10, value=6, label="Mood Score")
                
                predict_btn = gr.Button("Predict Progress", variant="primary")
            
            with gr.Column():
                progress_score = gr.Slider(0, 100, interactive=False, label="Progress Score (%)")
                progress_result = gr.JSON(label="Prediction Details")
        
        def predict_progress(days, cravings, stress, support, mood):
            if not self.analyzer:
                return 0, {}
            
            features = {
                "days_sober": days,
                "cravings_level": cravings,
                "stress_level": stress,
                "support_level": support,
                "mood_score": mood
            }
            
            progress = self.analyzer.predict_progress(features)
            return progress * 100, {"progress": progress, "features": features}
        
        predict_btn.click(
            fn=predict_progress,
            inputs=[days_sober, cravings_level, stress_level, support_level, mood_score],
            outputs=[progress_score, progress_result]
        )
    
    def _create_relapse_tab(self):
        """Create relapse risk tab"""
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Recent Days Data")
                recent_cravings = gr.Slider(0, 10, value=5, label="Recent Cravings")
                recent_stress = gr.Slider(0, 10, value=5, label="Recent Stress")
                recent_mood = gr.Slider(0, 10, value=6, label="Recent Mood")
                triggers_count = gr.Number(label="Triggers Encountered", value=2)
                
                risk_btn = gr.Button("Assess Relapse Risk", variant="primary")
            
            with gr.Column():
                risk_score = gr.Slider(0, 100, interactive=False, label="Relapse Risk (%)")
                risk_level = gr.Textbox(label="Risk Level", interactive=False)
                recommendations = gr.Textbox(
                    label="Recommendations",
                    lines=5,
                    interactive=False
                )
        
        def assess_risk(cravings, stress, mood, triggers):
            if not self.analyzer:
                return 0, "Not available", "Not available"
            
            sequence = [{
                "cravings_level": cravings,
                "stress_level": stress,
                "mood_score": mood,
                "triggers_count": triggers,
                "consumed": 0.0
            }]
            
            risk = self.analyzer.predict_relapse_risk(sequence)
            
            if risk < 0.3:
                level = "Low"
                rec = "Continue your current strategies. You're doing well!"
            elif risk < 0.6:
                level = "Moderate"
                rec = "Be cautious. Consider reaching out to your support network."
            else:
                level = "High"
                rec = "High risk detected. Please contact your support system or counselor immediately."
            
            return risk * 100, level, rec
        
        risk_btn.click(
            fn=assess_risk,
            inputs=[recent_cravings, recent_stress, recent_mood, triggers_count],
            outputs=[risk_score, risk_level, recommendations]
        )
    
    def _create_coaching_tab(self):
        """Create AI coaching tab"""
        with gr.Row():
            with gr.Column():
                situation_input = gr.Textbox(
                    label="Your Current Situation",
                    lines=5,
                    placeholder="Describe your current situation..."
                )
                days_sober_input = gr.Number(label="Days Sober", value=0)
                challenge_input = gr.Textbox(
                    label="Current Challenge (Optional)",
                    lines=3,
                    placeholder="What challenges are you facing?"
                )
                coach_btn = gr.Button("Get AI Coaching", variant="primary")
            
            with gr.Column():
                coaching_output = gr.Textbox(
                    label="AI Coaching Message",
                    lines=10,
                    interactive=False
                )
        
        def get_coaching(situation, days, challenge):
            if not self.analyzer:
                return "AI coaching not available"
            
            challenge_text = challenge if challenge else None
            coaching = self.analyzer.generate_coaching(
                situation, int(days), challenge_text
            )
            return coaching
        
        coach_btn.click(
            fn=get_coaching,
            inputs=[situation_input, days_sober_input, challenge_input],
            outputs=coaching_output
        )
    
    def launch(
        self,
        server_name: str = "0.0.0.0",
        server_port: int = 7860,
        share: bool = False
    ):
        """Launch interface"""
        interface = self.create_interface()
        interface.launch(server_name=server_name, server_port=server_port, share=share)


def create_recovery_gradio_app(enhanced_analyzer, counseling_service=None) -> gr.Blocks:
    """Create Gradio app"""
    interface = RecoveryGradioInterface(enhanced_analyzer, counseling_service)
    return interface.create_interface()

