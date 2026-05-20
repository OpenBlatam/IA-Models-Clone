"""
Gradio Integration for Dermatology AI
Creates interactive demos for model inference and visualization
"""

import logging
from typing import Optional, Dict, Any, List
import numpy as np
from PIL import Image
import io

logger = logging.getLogger(__name__)

try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False
    logger.warning("Gradio not available. Install with: pip install gradio")


class GradioDemo:
    """
    Gradio demo interface for skin analysis
    Implements proper error handling and input validation
    """
    
    def __init__(
        self,
        analyzer: Any,
        model_manager: Optional[Any] = None,
        title: str = "Dermatology AI - Skin Analysis",
        description: str = "Upload an image to analyze skin quality and get recommendations"
    ):
        if not GRADIO_AVAILABLE:
            raise ImportError("Gradio is required. Install with: pip install gradio")
        
        self.analyzer = analyzer
        self.model_manager = model_manager
        self.title = title
        self.description = description
        self.interface = None
    
    def analyze_image_gradio(
        self,
        image: Image.Image,
        use_advanced: bool = True,
        enhance: bool = False
    ) -> Dict[str, Any]:
        """
        Analyze image with Gradio interface
        Includes proper error handling and input validation
        """
        try:
            # Validate input
            if image is None:
                return {"error": "Please upload an image"}
            
            # Convert PIL to numpy if needed
            if isinstance(image, Image.Image):
                img_array = np.array(image)
            else:
                img_array = image
            
            # Validate image dimensions
            if len(img_array.shape) < 2:
                return {"error": "Invalid image format"}
            
            # Run analysis
            result = self.analyzer.analyze_image(
                img_array,
                use_cache=False  # Don't cache in demo
            )
            
            # Format results for display
            formatted_result = self._format_results(result)
            
            return formatted_result
        
        except Exception as e:
            logger.error(f"Error in Gradio analysis: {str(e)}", exc_info=True)
            return {
                "error": f"Analysis failed: {str(e)}",
                "details": "Please try again with a different image"
            }
    
    def _format_results(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Format analysis results for display"""
        formatted = {
            "status": "success",
            "overall_score": result.get("quality_scores", {}).get("overall_score", 0),
            "skin_type": result.get("skin_type", "unknown"),
            "conditions": [],
            "recommendations": result.get("recommendations_priority", [])
        }
        
        # Format conditions
        for condition in result.get("conditions", []):
            formatted["conditions"].append({
                "name": condition.get("name", ""),
                "confidence": f"{condition.get('confidence', 0) * 100:.1f}%",
                "severity": condition.get("severity", "unknown")
            })
        
        # Format quality scores
        quality_scores = result.get("quality_scores", {})
        formatted["quality_scores"] = {
            k: f"{v:.1f}" if isinstance(v, (int, float)) else v
            for k, v in quality_scores.items()
        }
        
        return formatted
    
    def create_interface(self) -> gr.Blocks:
        """Create Gradio interface"""
        with gr.Blocks(title=self.title, theme=gr.themes.Soft()) as demo:
            gr.Markdown(f"# {self.title}")
            gr.Markdown(self.description)
            
            with gr.Row():
                with gr.Column():
                    image_input = gr.Image(
                        type="pil",
                        label="Upload Skin Image",
                        sources=["upload", "webcam"]
                    )
                    
                    with gr.Row():
                        use_advanced = gr.Checkbox(
                            label="Use Advanced Analysis",
                            value=True
                        )
                        enhance = gr.Checkbox(
                            label="Enhance Image",
                            value=False
                        )
                    
                    analyze_btn = gr.Button("Analyze", variant="primary")
                
                with gr.Column():
                    overall_score = gr.Number(
                        label="Overall Score",
                        precision=1
                    )
                    
                    skin_type = gr.Textbox(
                        label="Skin Type"
                    )
                    
                    conditions_output = gr.JSON(
                        label="Detected Conditions"
                    )
                    
                    quality_scores_output = gr.JSON(
                        label="Quality Scores"
                    )
                    
                    recommendations_output = gr.JSON(
                        label="Recommendations"
                    )
            
            # Examples
            gr.Examples(
                examples=[],  # Add example images here
                inputs=image_input
            )
            
            # Event handlers
            analyze_btn.click(
                fn=self.analyze_image_gradio,
                inputs=[image_input, use_advanced, enhance],
                outputs=[
                    overall_score,
                    skin_type,
                    conditions_output,
                    quality_scores_output,
                    recommendations_output
                ]
            )
            
            # Error handling
            demo.load(
                fn=lambda: {"status": "ready"},
                outputs=None
            )
        
        self.interface = demo
        return demo
    
    def launch(
        self,
        server_name: str = "0.0.0.0",
        server_port: int = 7860,
        share: bool = False,
        debug: bool = False
    ):
        """Launch Gradio interface"""
        if self.interface is None:
            self.create_interface()
        
        logger.info(f"Launching Gradio interface on {server_name}:{server_port}")
        self.interface.launch(
            server_name=server_name,
            server_port=server_port,
            share=share,
            debug=debug
        )


class ModelComparisonDemo:
    """
    Gradio demo for comparing different models
    """
    
    def __init__(self, models: Dict[str, Any]):
        if not GRADIO_AVAILABLE:
            raise ImportError("Gradio is required")
        
        self.models = models
    
    def compare_models(
        self,
        image: Image.Image,
        selected_models: List[str]
    ) -> Dict[str, Any]:
        """Compare predictions from multiple models"""
        if image is None:
            return {"error": "Please upload an image"}
        
        results = {}
        img_array = np.array(image)
        
        for model_name in selected_models:
            if model_name in self.models:
                try:
                    model = self.models[model_name]
                    result = model.predict(img_array)
                    results[model_name] = result
                except Exception as e:
                    results[model_name] = {"error": str(e)}
        
        return results
    
    def create_interface(self) -> gr.Blocks:
        """Create comparison interface"""
        with gr.Blocks(title="Model Comparison") as demo:
            gr.Markdown("# Model Comparison Demo")
            
            with gr.Row():
                image_input = gr.Image(type="pil", label="Upload Image")
                model_selection = gr.CheckboxGroup(
                    choices=list(self.models.keys()),
                    label="Select Models to Compare"
                )
            
            compare_btn = gr.Button("Compare Models", variant="primary")
            results_output = gr.JSON(label="Comparison Results")
            
            compare_btn.click(
                fn=self.compare_models,
                inputs=[image_input, model_selection],
                outputs=results_output
            )
        
        return demo













