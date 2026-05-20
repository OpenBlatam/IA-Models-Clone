"""
Prediction Demo with Gradio
============================

Interactive demo for model predictions.
"""

import gradio as gr
import torch
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

try:
    from ..ml.models import EventDurationPredictor, RoutineCompletionPredictor
    from ..ml.data.preprocessing import FeatureExtractor
    from ..ml.prediction_service import PredictionService
    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False
    logger.warning("ML models not available")


def predict_event_duration(
    event_type: str,
    start_time: str,
    location: str,
    description: str
) -> str:
    """
    Predict event duration.
    
    Args:
        event_type: Type of event
        start_time: Start time
        location: Location
        description: Description
    
    Returns:
        Prediction result
    """
    if not MODELS_AVAILABLE:
        return "Models not available. Please install dependencies."
    
    try:
        # Create service
        service = PredictionService()
        
        # Create event data
        event_data = {
            "type": event_type,
            "start_time": start_time,
            "location": location,
            "description": description
        }
        
        # Predict
        prediction = service.predict_event_duration(
            event_type=event_type,
            historical_events=[],
            event_data=event_data
        )
        
        return f"""
        **Predicted Duration:** {prediction.predicted_duration_hours} hours
        **Confidence:** {prediction.confidence:.1%}
        **Method:** {prediction.factors.get('method', 'unknown')}
        **Recommendation:** {prediction.recommendation}
        """
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return f"Error: {str(e)}"


def predict_routine_completion(
    routine_type: str,
    scheduled_time: str,
    day_of_week: int
) -> str:
    """
    Predict routine completion.
    
    Args:
        routine_type: Type of routine
        scheduled_time: Scheduled time
        day_of_week: Day of week (0-6)
    
    Returns:
        Prediction result
    """
    if not MODELS_AVAILABLE:
        return "Models not available. Please install dependencies."
    
    try:
        # Create service
        service = PredictionService()
        
        # Create routine data
        routine_data = {
            "type": routine_type,
            "scheduled_time": scheduled_time,
            "day_of_week": day_of_week
        }
        
        # Predict
        prediction = service.predict_routine_completion(
            routine_id="demo",
            completion_history=[],
            routine_data=routine_data
        )
        
        return f"""
        **Predicted Completion Rate:** {prediction.predicted_completion_rate:.1%}
        **Confidence:** {prediction.confidence:.1%}
        **Optimal Time:** {prediction.optimal_time or 'N/A'}
        **Recommendation:** {prediction.recommendation}
        """
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return f"Error: {str(e)}"


def create_prediction_demo() -> gr.Blocks:
    """
    Create Gradio demo for predictions.
    
    Returns:
        Gradio Blocks interface
    """
    with gr.Blocks(title="Artist Manager AI - Predictions") as demo:
        gr.Markdown("# 🎯 Artist Manager AI - Prediction Demo")
        gr.Markdown("Predict event durations and routine completion rates using ML models.")
        
        with gr.Tabs():
            # Event Duration Tab
            with gr.Tab("Event Duration"):
                gr.Markdown("### Predict Event Duration")
                
                with gr.Row():
                    event_type = gr.Dropdown(
                        choices=["concert", "interview", "photoshoot", "rehearsal", "meeting"],
                        label="Event Type",
                        value="concert"
                    )
                    start_time = gr.Textbox(
                        label="Start Time (ISO format)",
                        value="2024-01-01T20:00:00",
                        placeholder="YYYY-MM-DDTHH:MM:SS"
                    )
                
                location = gr.Textbox(
                    label="Location",
                    value="Venue A",
                    placeholder="Event location"
                )
                
                description = gr.Textbox(
                    label="Description",
                    value="Concert performance",
                    placeholder="Event description"
                )
                
                event_btn = gr.Button("Predict Duration", variant="primary")
                event_output = gr.Markdown()
                
                event_btn.click(
                    predict_event_duration,
                    inputs=[event_type, start_time, location, description],
                    outputs=event_output
                )
            
            # Routine Completion Tab
            with gr.Tab("Routine Completion"):
                gr.Markdown("### Predict Routine Completion")
                
                with gr.Row():
                    routine_type = gr.Dropdown(
                        choices=["exercise", "practice", "meal", "rest"],
                        label="Routine Type",
                        value="exercise"
                    )
                    scheduled_time = gr.Textbox(
                        label="Scheduled Time",
                        value="09:00",
                        placeholder="HH:MM"
                    )
                
                day_of_week = gr.Slider(
                    minimum=0,
                    maximum=6,
                    value=0,
                    step=1,
                    label="Day of Week (0=Monday, 6=Sunday)"
                )
                
                routine_btn = gr.Button("Predict Completion", variant="primary")
                routine_output = gr.Markdown()
                
                routine_btn.click(
                    predict_routine_completion,
                    inputs=[routine_type, scheduled_time, day_of_week],
                    outputs=routine_output
                )
        
        gr.Markdown("---")
        gr.Markdown("Built with PyTorch and Gradio")
    
    return demo


if __name__ == "__main__":
    demo = create_prediction_demo()
    demo.launch(share=True)




