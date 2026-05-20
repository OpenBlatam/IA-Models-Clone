"""
Gradio Integration
Interactive demos for model inference and visualization
"""

import gradio as gr
import torch
import numpy as np
from PIL import Image
from typing import Optional, Dict, Any, List
import logging
from pathlib import Path

from ..inference import InferencePipeline
from ..utils import TrainingVisualizer

logger = logging.getLogger(__name__)


class GradioApp:
    """
    Gradio application for model inference
    """
    
    def __init__(
        self,
        inference_pipeline: InferencePipeline,
        title: str = "MobileNet Inference",
        description: str = "Upload images for classification",
    ):
        """
        Initialize Gradio app
        
        Args:
            inference_pipeline: Inference pipeline instance
            title: App title
            description: App description
        """
        self.pipeline = inference_pipeline
        self.title = title
        self.description = description
    
    def predict_image(
        self,
        image: Image.Image,
        top_k: int = 5,
        confidence_threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Predict on single image
        
        Args:
            image: PIL Image
            top_k: Number of top predictions
            confidence_threshold: Confidence threshold
            
        Returns:
            Prediction results
        """
        try:
            if image is None:
                return {"error": "Please upload an image"}
            
            result = self.pipeline.predict(
                image,
                return_probabilities=True,
                top_k=top_k
            )
            
            # Format results for display
            output = {
                "prediction": result.get('class_name', result.get('predictions', [])[0]),
                "confidence": f"{result.get('confidence', 0.0) * 100:.2f}%",
            }
            
            if 'top_k' in result:
                top_k_results = []
                for idx, prob in zip(result['top_k']['indices'][0], result['top_k']['probabilities'][0]):
                    top_k_results.append({
                        'class': result.get('top_k_classes', [])[0][idx] if 'top_k_classes' in result else f"Class {idx}",
                        'probability': f"{prob * 100:.2f}%"
                    })
                output['top_k'] = top_k_results
            
            return output
        except Exception as e:
            logger.error(f"Error in prediction: {e}", exc_info=True)
            return {"error": str(e)}
    
    def predict_batch_images(
        self,
        images: List[Image.Image],
    ) -> List[Dict[str, Any]]:
        """
        Predict on batch of images
        
        Args:
            images: List of PIL Images
            
        Returns:
            List of prediction results
        """
        try:
            if not images:
                return [{"error": "Please upload images"}]
            
            results = self.pipeline.predict_batch(images)
            return results
        except Exception as e:
            logger.error(f"Error in batch prediction: {e}", exc_info=True)
            return [{"error": str(e)}]
    
    def create_interface(self) -> gr.Blocks:
        """
        Create Gradio interface
        
        Returns:
            Gradio Blocks interface
        """
        with gr.Blocks(title=self.title) as interface:
            gr.Markdown(f"# {self.title}")
            gr.Markdown(self.description)
            
            with gr.Row():
                with gr.Column():
                    image_input = gr.Image(type="pil", label="Upload Image")
                    top_k_slider = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=5,
                        step=1,
                        label="Top K Predictions"
                    )
                    confidence_slider = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.5,
                        step=0.05,
                        label="Confidence Threshold"
                    )
                    predict_btn = gr.Button("Predict", variant="primary")
                
                with gr.Column():
                    prediction_output = gr.JSON(label="Prediction Results")
                    confidence_output = gr.Textbox(label="Confidence")
            
            with gr.Row():
                batch_images = gr.File(
                    file_count="multiple",
                    file_types=["image"],
                    label="Upload Multiple Images"
                )
                batch_predict_btn = gr.Button("Predict Batch", variant="secondary")
                batch_output = gr.JSON(label="Batch Results")
            
            # Examples
            gr.Examples(
                examples=[],
                inputs=image_input,
            )
            
            # Event handlers
            predict_btn.click(
                fn=self.predict_image,
                inputs=[image_input, top_k_slider, confidence_slider],
                outputs=[prediction_output, confidence_output]
            )
            
            batch_predict_btn.click(
                fn=self.predict_batch_images,
                inputs=[batch_images],
                outputs=[batch_output]
            )
        
        return interface
    
    def launch(
        self,
        server_name: str = "0.0.0.0",
        server_port: int = 7860,
        share: bool = False,
    ) -> None:
        """
        Launch Gradio app
        
        Args:
            server_name: Server hostname
            server_port: Server port
            share: Create public link
        """
        interface = self.create_interface()
        interface.launch(
            server_name=server_name,
            server_port=server_port,
            share=share
        )


class TrainingMonitorApp:
    """
    Gradio app for monitoring training progress
    """
    
    def __init__(self, metrics_collector=None):
        """
        Initialize training monitor
        
        Args:
            metrics_collector: MetricsCollector instance
        """
        self.metrics_collector = metrics_collector
        self.history = {}
    
    def update_metrics(self, metrics: Dict[str, float]) -> None:
        """Update training metrics"""
        self.history.update(metrics)
    
    def create_interface(self) -> gr.Blocks:
        """Create monitoring interface"""
        with gr.Blocks(title="Training Monitor") as interface:
            gr.Markdown("# Training Progress Monitor")
            
            with gr.Row():
                loss_plot = gr.Plot(label="Loss")
                accuracy_plot = gr.Plot(label="Accuracy")
            
            with gr.Row():
                metrics_table = gr.Dataframe(
                    headers=["Metric", "Value"],
                    label="Current Metrics"
                )
                refresh_btn = gr.Button("Refresh", variant="primary")
            
            def update_display():
                if self.metrics_collector:
                    summary = self.metrics_collector.get_summary()
                    data = [[k, f"{v['latest']:.4f}"] for k, v in summary.items()]
                    return data
                return []
            
            refresh_btn.click(fn=update_display, outputs=[metrics_table])
        
        return interface



