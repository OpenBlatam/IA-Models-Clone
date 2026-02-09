"""
Advanced Gradio Demo
====================

Advanced interactive demo with multiple features.
"""

import gradio as gr
import torch
import numpy as np
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def create_advanced_demo(
    prediction_service,
    text_generator,
    image_generator=None
):
    """
    Create advanced Gradio demo.
    
    Args:
        prediction_service: Prediction service
        text_generator: Text generator
        image_generator: Optional image generator
    
    Returns:
        Gradio interface
    """
    
    def predict_event_duration(
        event_type: str,
        event_name: str,
        priority: str,
        location: str
    ) -> Dict[str, Any]:
        """
        Predict event duration.
        
        Args:
            event_type: Type of event
            event_name: Name of event
            priority: Priority level
            location: Location
        
        Returns:
            Prediction results
        """
        try:
            # Prepare input
            event_data = {
                "type": event_type,
                "name": event_name,
                "priority": priority,
                "location": location
            }
            
            # Predict
            prediction = prediction_service.predict_event_duration(event_data)
            
            return {
                "predicted_duration": f"{prediction['duration']:.2f} hours",
                "confidence": f"{prediction.get('confidence', 0.0) * 100:.1f}%"
            }
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return {"error": str(e)}
    
    def generate_text(
        prompt: str,
        max_length: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """
        Generate text.
        
        Args:
            prompt: Input prompt
            max_length: Maximum length
            temperature: Temperature
            top_p: Top-p sampling
        
        Returns:
            Generated text
        """
        try:
            result = text_generator.generate(
                prompt,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p
            )
            return result["text"]
        except Exception as e:
            logger.error(f"Text generation error: {str(e)}")
            return f"Error: {str(e)}"
    
    def generate_image(
        prompt: str,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5
    ) -> Optional[np.ndarray]:
        """
        Generate image.
        
        Args:
            prompt: Text prompt
            num_inference_steps: Number of steps
            guidance_scale: Guidance scale
        
        Returns:
            Generated image
        """
        if image_generator is None:
            return None
        
        try:
            image = image_generator.generate(
                prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
            )
            return image
        except Exception as e:
            logger.error(f"Image generation error: {str(e)}")
            return None
    
    # Create interfaces
    with gr.Blocks(title="Artist Manager AI - Advanced Demo") as demo:
        gr.Markdown("# 🎨 Artist Manager AI - Advanced Demo")
        
        with gr.Tabs():
            # Event Prediction Tab
            with gr.Tab("Event Prediction"):
                with gr.Row():
                    with gr.Column():
                        event_type_input = gr.Dropdown(
                            choices=["Concert", "Recording", "Meeting", "Rehearsal"],
                            label="Event Type",
                            value="Concert"
                        )
                        event_name_input = gr.Textbox(
                            label="Event Name",
                            value="Summer Tour"
                        )
                        priority_input = gr.Dropdown(
                            choices=["Low", "Medium", "High", "Critical"],
                            label="Priority",
                            value="High"
                        )
                        location_input = gr.Textbox(
                            label="Location",
                            value="Main Stage"
                        )
                        predict_btn = gr.Button("Predict Duration", variant="primary")
                    
                    with gr.Column():
                        duration_output = gr.Textbox(label="Predicted Duration")
                        confidence_output = gr.Textbox(label="Confidence")
                
                predict_btn.click(
                    fn=predict_event_duration,
                    inputs=[event_type_input, event_name_input, priority_input, location_input],
                    outputs=[duration_output, confidence_output]
                )
            
            # Text Generation Tab
            with gr.Tab("Text Generation"):
                with gr.Row():
                    with gr.Column():
                        prompt_input = gr.Textbox(
                            label="Prompt",
                            placeholder="Enter your prompt here...",
                            lines=3
                        )
                        max_length_input = gr.Slider(
                            minimum=10,
                            maximum=500,
                            value=100,
                            step=10,
                            label="Max Length"
                        )
                        temperature_input = gr.Slider(
                            minimum=0.1,
                            maximum=2.0,
                            value=0.7,
                            step=0.1,
                            label="Temperature"
                        )
                        top_p_input = gr.Slider(
                            minimum=0.1,
                            maximum=1.0,
                            value=0.9,
                            step=0.05,
                            label="Top-p"
                        )
                        generate_btn = gr.Button("Generate", variant="primary")
                    
                    with gr.Column():
                        text_output = gr.Textbox(
                            label="Generated Text",
                            lines=10
                        )
                
                generate_btn.click(
                    fn=generate_text,
                    inputs=[prompt_input, max_length_input, temperature_input, top_p_input],
                    outputs=[text_output]
                )
            
            # Image Generation Tab (if available)
            if image_generator is not None:
                with gr.Tab("Image Generation"):
                    with gr.Row():
                        with gr.Column():
                            image_prompt_input = gr.Textbox(
                                label="Image Prompt",
                                placeholder="Describe the image you want to generate...",
                                lines=3
                            )
                            steps_input = gr.Slider(
                                minimum=10,
                                maximum=100,
                                value=50,
                                step=5,
                                label="Inference Steps"
                            )
                            guidance_input = gr.Slider(
                                minimum=1.0,
                                maximum=20.0,
                                value=7.5,
                                step=0.5,
                                label="Guidance Scale"
                            )
                            image_generate_btn = gr.Button("Generate Image", variant="primary")
                        
                        with gr.Column():
                            image_output = gr.Image(label="Generated Image")
                    
                    image_generate_btn.click(
                        fn=generate_image,
                        inputs=[image_prompt_input, steps_input, guidance_input],
                        outputs=[image_output]
                    )
        
        # Examples
        gr.Markdown("## 📚 Examples")
        gr.Examples(
            examples=[
                ["Concert", "Summer Festival", "High", "Outdoor Stage"],
                ["Recording", "Album Session", "Critical", "Studio A"],
            ],
            inputs=[event_type_input, event_name_input, priority_input, location_input]
        )
    
    return demo




