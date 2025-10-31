"""
Gradio Integration System
Implements interactive demos, user-friendly interfaces, error handling, and input validation
"""

import gradio as gr
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import json
import logging
from typing import Dict, List, Any, Optional, Callable, Union
from pathlib import Path
import traceback
import warnings
from datetime import datetime
import io
import base64

class GradioInterfaceBuilder:
    """Builder for creating Gradio interfaces with best practices"""
    
    def __init__(self, title: str = "AI Model Demo", description: str = ""):
        self.title = title
        self.description = description
        self.components = []
        self.event_handlers = []
        self.css_custom = ""
        self.theme = gr.themes.Soft()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
    def add_component(self, component_type: str, **kwargs):
        """Add a component to the interface"""
        self.components.append({
            "type": component_type,
            "kwargs": kwargs
        })
        return self
    
    def add_event_handler(self, event_type: str, handler: Callable, **kwargs):
        """Add an event handler"""
        self.event_handlers.append({
            "event_type": event_type,
            "handler": handler,
            "kwargs": kwargs
        })
        return self
    
    def set_custom_css(self, css: str):
        """Set custom CSS for styling"""
        self.css_custom = css
        return self
    
    def set_theme(self, theme):
        """Set Gradio theme"""
        self.theme = theme
        return self
    
    def build(self):
        """Build the Gradio interface"""
        # Create interface
        interface = gr.Interface(
            fn=self._create_demo_function(),
            inputs=self._create_inputs(),
            outputs=self._create_outputs(),
            title=self.title,
            description=self.description,
            theme=self.theme,
            css=self.css_custom
        )
        
        # Add event handlers
        for handler_info in self.event_handlers:
            interface.load(handler_info["handler"], **handler_info["kwargs"])
        
        return interface
    
    def _create_demo_function(self):
        """Create the main demo function"""
        def demo_function(*args):
            try:
                # Process inputs and call appropriate handlers
                return self._process_inputs(*args)
            except Exception as e:
                self.logger.error(f"Error in demo function: {e}")
                return self._create_error_output(str(e))
        
        return demo_function
    
    def _create_inputs(self):
        """Create input components"""
        inputs = []
        for component in self.components:
            if component["type"] == "textbox":
                inputs.append(gr.Textbox(**component["kwargs"]))
            elif component["type"] == "slider":
                inputs.append(gr.Slider(**component["kwargs"]))
            elif component["type"] == "dropdown":
                inputs.append(gr.Dropdown(**component["kwargs"]))
            elif component["type"] == "image":
                inputs.append(gr.Image(**component["kwargs"]))
            elif component["type"] == "number":
                inputs.append(gr.Number(**component["kwargs"]))
            elif component["type"] == "checkbox":
                inputs.append(gr.Checkbox(**component["kwargs"]))
        
        return inputs
    
    def _create_outputs(self):
        """Create output components"""
        return [
            gr.Image(label="Generated Image"),
            gr.Textbox(label="Status"),
            gr.JSON(label="Metadata")
        ]
    
    def _process_inputs(self, *args):
        """Process inputs and return outputs"""
        # Placeholder - override in subclasses
        return None, "Processing complete", {}
    
    def _create_error_output(self, error_msg: str):
        """Create error output"""
        return None, f"Error: {error_msg}", {"error": error_msg}

class ModelDemoInterface:
    """Base class for model demo interfaces"""
    
    def __init__(self, model: Optional[nn.Module] = None, device: str = "cpu"):
        self.model = model
        self.device = device
        self.logger = logging.getLogger(__name__)
        
        if self.model:
            self.model.to(device)
            self.model.eval()
    
    def create_interface(self) -> gr.Interface:
        """Create the Gradio interface"""
        raise NotImplementedError("Subclasses must implement create_interface")

class TextGenerationInterface(ModelDemoInterface):
    """Interface for text generation models"""
    
    def __init__(self, model, tokenizer, device: str = "cpu"):
        super().__init__(model, device)
        self.tokenizer = tokenizer
        
    def create_interface(self) -> gr.Interface:
        """Create text generation interface"""
        
        def generate_text(prompt: str, max_length: int, temperature: float, 
                         top_p: float, do_sample: bool) -> tuple:
            try:
                # Input validation
                if not prompt.strip():
                    return "", "Error: Please provide a prompt", {"error": "Empty prompt"}
                
                if max_length < 1 or max_length > 1000:
                    return "", "Error: Invalid max length", {"error": "Invalid max length"}
                
                if temperature < 0.1 or temperature > 2.0:
                    return "", "Error: Invalid temperature", {"error": "Invalid temperature"}
                
                # Generate text
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_length=max_length,
                        temperature=temperature,
                        top_p=top_p,
                        do_sample=do_sample,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Metadata
                metadata = {
                    "prompt": prompt,
                    "max_length": max_length,
                    "temperature": temperature,
                    "top_p": top_p,
                    "do_sample": do_sample,
                    "generated_length": len(generated_text),
                    "timestamp": datetime.now().isoformat()
                }
                
                return generated_text, "Generation successful", metadata
                
            except Exception as e:
                self.logger.error(f"Error in text generation: {e}")
                return "", f"Error: {str(e)}", {"error": str(e)}
        
        # Create interface
        interface = gr.Interface(
            fn=generate_text,
            inputs=[
                gr.Textbox(
                    label="Prompt",
                    placeholder="Enter your prompt here...",
                    lines=3,
                    max_lines=10
                ),
                gr.Slider(
                    minimum=10,
                    maximum=500,
                    value=100,
                    step=1,
                    label="Max Length"
                ),
                gr.Slider(
                    minimum=0.1,
                    maximum=2.0,
                    value=0.7,
                    step=0.1,
                    label="Temperature"
                ),
                gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.9,
                    step=0.1,
                    label="Top-p (nucleus sampling)"
                ),
                gr.Checkbox(
                    label="Enable sampling",
                    value=True
                )
            ],
            outputs=[
                gr.Textbox(label="Generated Text", lines=10),
                gr.Textbox(label="Status"),
                gr.JSON(label="Metadata")
            ],
            title="Text Generation Demo",
            description="Generate text using AI language models",
            theme=gr.themes.Soft(),
            css="""
            .gradio-container {
                max-width: 1200px !important;
                margin: auto !important;
            }
            .status-box {
                border-radius: 8px;
                padding: 10px;
                margin: 10px 0;
            }
            .success { background-color: #d4edda; border: 1px solid #c3e6cb; }
            .error { background-color: #f8d7da; border: 1px solid #f5c6cb; }
            """
        )
        
        return interface

class ImageGenerationInterface(ModelDemoInterface):
    """Interface for image generation models"""
    
    def __init__(self, model, device: str = "cpu"):
        super().__init__(model, device)
        
    def create_interface(self) -> gr.Interface:
        """Create image generation interface"""
        
        def generate_image(prompt: str, negative_prompt: str, num_steps: int,
                          guidance_scale: float, seed: int, width: int, height: int) -> tuple:
            try:
                # Input validation
                if not prompt.strip():
                    return None, "Error: Please provide a prompt", {"error": "Empty prompt"}
                
                if num_steps < 10 or num_steps > 100:
                    return None, "Error: Invalid number of steps", {"error": "Invalid steps"}
                
                if guidance_scale < 1.0 or guidance_scale > 20.0:
                    return None, "Error: Invalid guidance scale", {"error": "Invalid guidance scale"}
                
                if width < 64 or width > 2048 or height < 64 or height > 2048:
                    return None, "Error: Invalid dimensions", {"error": "Invalid dimensions"}
                
                # Set seed for reproducibility
                if seed != -1:
                    torch.manual_seed(seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed(seed)
                
                # Generate image (placeholder - replace with actual model)
                # This is a mock generation for demonstration
                mock_image = self._create_mock_image(width, height, prompt)
                
                # Metadata
                metadata = {
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "num_steps": num_steps,
                    "guidance_scale": guidance_scale,
                    "seed": seed,
                    "width": width,
                    "height": height,
                    "timestamp": datetime.now().isoformat()
                }
                
                return mock_image, "Image generation successful", metadata
                
            except Exception as e:
                self.logger.error(f"Error in image generation: {e}")
                return None, f"Error: {str(e)}", {"error": str(e)}
        
        def _create_mock_image(self, width: int, height: int, prompt: str):
            """Create a mock image for demonstration"""
            # Create a simple gradient image
            img_array = np.random.rand(height, width, 3)
            
            # Add some pattern based on prompt
            if "blue" in prompt.lower():
                img_array[:, :, 0] = 0.8
            if "red" in prompt.lower():
                img_array[:, :, 2] = 0.8
            if "green" in prompt.lower():
                img_array[:, :, 1] = 0.8
            
            # Convert to PIL Image
            img = Image.fromarray((img_array * 255).astype(np.uint8))
            return img
        
        # Create interface
        interface = gr.Interface(
            fn=generate_image,
            inputs=[
                gr.Textbox(
                    label="Prompt",
                    placeholder="Describe the image you want to generate...",
                    lines=2
                ),
                gr.Textbox(
                    label="Negative Prompt",
                    placeholder="What you don't want in the image...",
                    lines=2
                ),
                gr.Slider(
                    minimum=10,
                    maximum=100,
                    value=50,
                    step=1,
                    label="Number of Steps"
                ),
                gr.Slider(
                    minimum=1.0,
                    maximum=20.0,
                    value=7.5,
                    step=0.5,
                    label="Guidance Scale"
                ),
                gr.Number(
                    label="Seed (-1 for random)",
                    value=-1,
                    minimum=-1,
                    maximum=999999
                ),
                gr.Slider(
                    minimum=256,
                    maximum=1024,
                    value=512,
                    step=64,
                    label="Width"
                ),
                gr.Slider(
                    minimum=256,
                    maximum=1024,
                    value=512,
                    step=64,
                    label="Height"
                )
            ],
            outputs=[
                gr.Image(label="Generated Image", type="pil"),
                gr.Textbox(label="Status"),
                gr.JSON(label="Metadata")
            ],
            title="Image Generation Demo",
            description="Generate images using AI diffusion models",
            theme=gr.themes.Soft(),
            css="""
            .gradio-container {
                max-width: 1400px !important;
                margin: auto !important;
            }
            .image-output {
                border-radius: 12px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            }
            """
        )
        
        return interface

class ModelComparisonInterface(ModelDemoInterface):
    """Interface for comparing multiple models"""
    
    def __init__(self, models: Dict[str, nn.Module], device: str = "cpu"):
        super().__init__(None, device)
        self.models = {name: model.to(device) for name, model in models.items()}
        
    def create_interface(self) -> gr.Interface:
        """Create model comparison interface"""
        
        def compare_models(prompt: str, model_names: List[str], 
                          parameters: Dict[str, Any]) -> tuple:
            try:
                # Input validation
                if not prompt.strip():
                    return None, "Error: Please provide a prompt", {"error": "Empty prompt"}
                
                if not model_names:
                    return None, "Error: Please select at least one model", {"error": "No models selected"}
                
                results = {}
                images = []
                
                for model_name in model_names:
                    if model_name in self.models:
                        try:
                            # Generate with selected model
                            model = self.models[model_name]
                            model.eval()
                            
                            # Mock generation (replace with actual model inference)
                            mock_image = self._create_comparison_image(prompt, model_name)
                            images.append(mock_image)
                            
                            results[model_name] = {
                                "status": "success",
                                "generation_time": np.random.uniform(1.0, 5.0),
                                "parameters": parameters
                            }
                            
                        except Exception as e:
                            results[model_name] = {
                                "status": "error",
                                "error": str(e)
                            }
                    else:
                        results[model_name] = {
                            "status": "error",
                            "error": "Model not found"
                        }
                
                # Combine images horizontally
                if images:
                    combined_image = self._combine_images_horizontally(images)
                else:
                    combined_image = None
                
                metadata = {
                    "prompt": prompt,
                    "models": model_names,
                    "parameters": parameters,
                    "results": results,
                    "timestamp": datetime.now().isoformat()
                }
                
                return combined_image, "Comparison complete", metadata
                
            except Exception as e:
                self.logger.error(f"Error in model comparison: {e}")
                return None, f"Error: {str(e)}", {"error": str(e)}
        
        def _create_comparison_image(self, prompt: str, model_name: str):
            """Create a comparison image for demonstration"""
            # Create a simple image with model name
            img_array = np.ones((256, 256, 3), dtype=np.uint8) * 255
            
            # Add some visual elements
            if "blue" in prompt.lower():
                img_array[:, :, 0] = 200
            if "red" in prompt.lower():
                img_array[:, :, 2] = 200
            
            # Convert to PIL Image
            img = Image.fromarray(img_array)
            return img
        
        def _combine_images_horizontally(self, images: List[Image.Image]):
            """Combine multiple images horizontally"""
            if not images:
                return None
            
            # Calculate total width and max height
            total_width = sum(img.width for img in images)
            max_height = max(img.height for img in images)
            
            # Create combined image
            combined = Image.new('RGB', (total_width, max_height), 'white')
            
            x_offset = 0
            for img in images:
                combined.paste(img, (x_offset, 0))
                x_offset += img.width
            
            return combined
        
        # Create interface
        interface = gr.Interface(
            fn=compare_models,
            inputs=[
                gr.Textbox(
                    label="Prompt",
                    placeholder="Enter prompt for model comparison...",
                    lines=2
                ),
                gr.CheckboxGroup(
                    choices=list(self.models.keys()),
                    label="Select Models to Compare",
                    value=list(self.models.keys())[:2]
                ),
                gr.JSON(
                    label="Model Parameters",
                    value={"temperature": 0.7, "max_length": 100},
                    interactive=True
                )
            ],
            outputs=[
                gr.Image(label="Comparison Results", type="pil"),
                gr.Textbox(label="Status"),
                gr.JSON(label="Detailed Results")
            ],
            title="Model Comparison Demo",
            description="Compare multiple AI models side by side",
            theme=gr.themes.Soft(),
            css="""
            .gradio-container {
                max-width: 1600px !important;
                margin: auto !important;
            }
            .comparison-output {
                border-radius: 12px;
                box-shadow: 0 6px 12px rgba(0,0,0,0.15);
            }
            """
        )
        
        return interface

class ErrorHandler:
    """Comprehensive error handling for Gradio interfaces"""
    
    @staticmethod
    def handle_input_validation(value: Any, validation_rules: Dict[str, Any]) -> tuple[bool, str]:
        """Validate input values"""
        try:
            if "required" in validation_rules and validation_rules["required"]:
                if value is None or (isinstance(value, str) and not value.strip()):
                    return False, "This field is required"
            
            if "min_length" in validation_rules and isinstance(value, str):
                if len(value) < validation_rules["min_length"]:
                    return False, f"Minimum length is {validation_rules['min_length']} characters"
            
            if "max_length" in validation_rules and isinstance(value, str):
                if len(value) > validation_rules["max_length"]:
                    return False, f"Maximum length is {validation_rules['max_length']} characters"
            
            if "min_value" in validation_rules and isinstance(value, (int, float)):
                if value < validation_rules["min_value"]:
                    return False, f"Minimum value is {validation_rules['min_value']}"
            
            if "max_value" in validation_rules and isinstance(value, (int, float)):
                if value > validation_rules["max_value"]:
                    return False, f"Maximum value is {validation_rules['max_value']}"
            
            if "pattern" in validation_rules and isinstance(value, str):
                import re
                if not re.match(validation_rules["pattern"], value):
                    return False, f"Invalid format: {validation_rules['pattern']}"
            
            return True, "Validation passed"
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    @staticmethod
    def create_error_output(error: Exception, context: str = "") -> tuple:
        """Create standardized error output"""
        error_msg = f"Error in {context}: {str(error)}" if context else str(error)
        
        # Log error with traceback
        logging.error(f"{error_msg}\n{traceback.format_exc()}")
        
        return None, error_msg, {"error": str(error), "traceback": traceback.format_exc()}

class InputValidator:
    """Input validation utilities for Gradio interfaces"""
    
    @staticmethod
    def validate_text_input(text: str, min_length: int = 1, max_length: int = 1000) -> tuple[bool, str]:
        """Validate text input"""
        if not text or not text.strip():
            return False, "Text cannot be empty"
        
        if len(text) < min_length:
            return False, f"Text must be at least {min_length} characters long"
        
        if len(text) > max_length:
            return False, f"Text cannot exceed {max_length} characters"
        
        return True, "Text is valid"
    
    @staticmethod
    def validate_numeric_input(value: Union[int, float], min_val: float, max_val: float) -> tuple[bool, str]:
        """Validate numeric input"""
        if not isinstance(value, (int, float)):
            return False, "Value must be a number"
        
        if value < min_val:
            return False, f"Value must be at least {min_val}"
        
        if value > max_val:
            return False, f"Value cannot exceed {max_val}"
        
        return True, "Value is valid"
    
    @staticmethod
    def validate_image_input(image: Image.Image, max_size: tuple = (2048, 2048)) -> tuple[bool, str]:
        """Validate image input"""
        if image is None:
            return False, "Image is required"
        
        if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
            return False, f"Image dimensions cannot exceed {max_size[0]}x{max_size[1]}"
        
        return True, "Image is valid"

def main():
    """Example usage of the Gradio integration system"""
    
    # Create a simple mock model for demonstration
    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 1)
        
        def forward(self, x):
            return self.linear(x)
    
    # Create interfaces
    text_interface = TextGenerationInterface(
        model=MockModel(),
        tokenizer=None,  # Mock tokenizer
        device="cpu"
    )
    
    image_interface = ImageGenerationInterface(
        model=MockModel(),
        device="cpu"
    )
    
    # Create model comparison interface
    models = {
        "Model A": MockModel(),
        "Model B": MockModel(),
        "Model C": MockModel()
    }
    
    comparison_interface = ModelComparisonInterface(
        models=models,
        device="cpu"
    )
    
    # Launch interfaces
    print("Launching Gradio interfaces...")
    
    # Launch text generation interface
    text_interface.create_interface().launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )

if __name__ == "__main__":
    main()
