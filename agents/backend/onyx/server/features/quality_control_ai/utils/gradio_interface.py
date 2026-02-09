"""
Gradio Interface for Quality Control AI
"""

import gradio as gr
import numpy as np
import cv2
from PIL import Image
from typing import Optional, Tuple, Dict
import logging
import torch

logger = logging.getLogger(__name__)


class QualityControlGradioInterface:
    """
    Gradio interface for quality control AI inference and visualization
    """
    
    def __init__(
        self,
        quality_inspector,
        title: str = "Quality Control AI",
        description: str = "Upload an image to inspect for defects and anomalies"
    ):
        """
        Initialize Gradio interface
        
        Args:
            quality_inspector: QualityInspector instance
            title: Interface title
            description: Interface description
        """
        self.inspector = quality_inspector
        self.title = title
        self.description = description
        
        logger.info("Gradio interface initialized")
    
    def inspect_image(self, image: Image.Image) -> Tuple[np.ndarray, Dict]:
        """
        Inspect image and return visualization
        
        Args:
            image: PIL Image
            
        Returns:
            Tuple of (visualized_image, results_dict)
        """
        if image is None:
            return None, {}
        
        try:
            # Convert PIL to numpy
            img_array = np.array(image)
            
            # Convert RGB to BGR if needed
            if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            # Inspect
            result = self.inspector.inspect_frame(img_array)
            
            if not result.get("success", False):
                return image, {"error": result.get("error", "Inspection failed")}
            
            # Visualize results
            vis_image = self._visualize_results(img_array, result)
            
            # Convert back to RGB for display
            if len(vis_image.shape) == 3:
                vis_image = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)
            
            # Format results
            results_text = self._format_results(result)
            
            return Image.fromarray(vis_image), results_text
            
        except Exception as e:
            logger.error(f"Error in inspection: {e}", exc_info=True)
            return image, {"error": str(e)}
    
    def _visualize_results(self, image: np.ndarray, result: Dict) -> np.ndarray:
        """Visualize inspection results on image"""
        vis_image = image.copy()
        
        # Draw defects
        defects = result.get("defects", [])
        for defect in defects:
            x, y, w, h = defect["location"]
            severity = defect.get("severity", "minor")
            
            # Color by severity
            color_map = {
                "minor": (0, 255, 255),      # Yellow
                "moderate": (0, 165, 255),   # Orange
                "severe": (0, 0, 255),       # Red
                "critical": (0, 0, 139)      # Dark red
            }
            color = color_map.get(severity, (0, 255, 0))
            
            # Draw rectangle
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, 2)
            
            # Draw label
            label = f"{defect['type']}: {defect['severity']}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(vis_image, (x, y - label_size[1] - 10), 
                         (x + label_size[0], y), color, -1)
            cv2.putText(vis_image, label, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Draw anomalies
        anomalies = result.get("anomalies", [])
        for anomaly in anomalies:
            x, y, w, h = anomaly["location"]
            severity = anomaly.get("severity", "low")
            
            color_map = {
                "low": (0, 255, 255),
                "medium": (0, 165, 255),
                "high": (0, 0, 255)
            }
            color = color_map.get(severity, (0, 255, 0))
            
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, 1)
        
        # Add quality score text
        quality_score = result.get("quality_score", 0)
        status = result.get("summary", {}).get("status", "unknown")
        
        text = f"Quality Score: {quality_score:.1f}/100 - Status: {status.upper()}"
        cv2.putText(vis_image, text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return vis_image
    
    def _format_results(self, result: Dict) -> Dict:
        """Format results for display"""
        summary = result.get("summary", {})
        
        formatted = {
            "Quality Score": f"{result.get('quality_score', 0):.2f}/100",
            "Status": summary.get("status", "unknown").upper(),
            "Total Objects": result.get("objects_detected", 0),
            "Total Anomalies": result.get("anomalies_detected", 0),
            "Total Defects": result.get("defects_detected", 0),
            "Recommendation": summary.get("recommendation", "N/A")
        }
        
        # Add defect counts
        defect_counts = summary.get("defect_counts", {})
        if defect_counts:
            formatted["Defect Breakdown"] = defect_counts
        
        return formatted
    
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
                    image_input = gr.Image(
                        type="pil",
                        label="Upload Image for Inspection"
                    )
                    inspect_btn = gr.Button("Inspect Quality", variant="primary")
                
                with gr.Column():
                    image_output = gr.Image(
                        type="pil",
                        label="Inspection Results"
                    )
                    results_output = gr.JSON(
                        label="Inspection Results"
                    )
            
            # Examples
            gr.Examples(
                examples=[],  # Can add example images here
                inputs=image_input
            )
            
            # Event handlers
            inspect_btn.click(
                fn=self.inspect_image,
                inputs=image_input,
                outputs=[image_output, results_output]
            )
            
            image_input.upload(
                fn=self.inspect_image,
                inputs=image_input,
                outputs=[image_output, results_output]
            )
        
        return interface
    
    def launch(
        self,
        server_name: str = "0.0.0.0",
        server_port: int = 7860,
        share: bool = False
    ):
        """
        Launch Gradio interface
        
        Args:
            server_name: Server name
            server_port: Server port
            share: Whether to create public link
        """
        interface = self.create_interface()
        interface.launch(
            server_name=server_name,
            server_port=server_port,
            share=share
        )


def create_gradio_app(quality_inspector) -> gr.Blocks:
    """
    Create Gradio app for quality control
    
    Args:
        quality_inspector: QualityInspector instance
        
    Returns:
        Gradio Blocks interface
    """
    interface = QualityControlGradioInterface(quality_inspector)
    return interface.create_interface()

