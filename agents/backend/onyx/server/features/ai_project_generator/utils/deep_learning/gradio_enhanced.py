"""Enhanced Gradio Interfaces"""

def generate_enhanced_gradio_code(task: str = "text_generation") -> str:
    if task == "text_generation":
        return '''"""
Enhanced Gradio Interface for Text Generation
=============================================
"""

import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextGenerationInterface:
    """Interfaz Gradio mejorada para generación de texto."""
    
    def __init__(self, model_path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """Inicializa interfaz."""
        self.device = device
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """Carga modelo y tokenizer."""
        try:
            logger.info(f"Loading model from {self.model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            ).to(self.device)
            self.model.eval()
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def generate(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1
    ) -> str:
        """Genera texto."""
        if not prompt:
            return "Por favor, ingresa un prompt."
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return generated_text
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            return f"Error: {str(e)}"
    
    def create_interface(self) -> gr.Interface:
        """Crea interfaz Gradio."""
        return gr.Interface(
            fn=self.generate,
            inputs=[
                gr.Textbox(label="Prompt", placeholder="Escribe tu prompt aquí...", lines=3),
                gr.Slider(minimum=50, maximum=500, value=100, step=10, label="Max Length"),
                gr.Slider(minimum=0.1, maximum=2.0, value=0.7, step=0.1, label="Temperature"),
                gr.Slider(minimum=0.1, maximum=1.0, value=0.9, step=0.05, label="Top-p"),
                gr.Slider(minimum=1, maximum=100, value=50, step=1, label="Top-k"),
                gr.Slider(minimum=1.0, maximum=2.0, value=1.1, step=0.1, label="Repetition Penalty")
            ],
            outputs=gr.Textbox(label="Generated Text", lines=10),
            title="Text Generation Demo",
            description="Genera texto usando un modelo de lenguaje",
            examples=[
                ["The future of AI is", 100, 0.7, 0.9, 50, 1.1],
                ["Once upon a time", 150, 0.8, 0.95, 40, 1.2]
            ]
        )


if __name__ == "__main__":
    interface = TextGenerationInterface("path/to/model")
    interface.create_interface().launch(share=True)
'''
    else:
        return '''"""
Enhanced Gradio Interface for Image Generation
==============================================
"""

import gradio as gr
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageGenerationInterface:
    """Interfaz Gradio mejorada para generación de imágenes."""
    
    def __init__(self, model_id: str = "runwayml/stable-diffusion-v1-5"):
        """Inicializa interfaz."""
        self.model_id = model_id
        self.pipe = None
        self._load_model()
    
    def _load_model(self):
        """Carga pipeline de difusión."""
        try:
            logger.info(f"Loading diffusion model: {self.model_id}")
            self.pipe = StableDiffusionPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            self.pipe = self.pipe.to("cuda" if torch.cuda.is_available() else "cpu")
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None
    ) -> Image:
        """Genera imagen."""
        if not prompt:
            return None
        
        try:
            generator = torch.Generator().manual_seed(seed) if seed else None
            
            image = self.pipe(
                prompt,
                negative_prompt=negative_prompt if negative_prompt else None,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator
            ).images[0]
            
            return image
        except Exception as e:
            logger.error(f"Error generating image: {e}")
            return None
    
    def create_interface(self) -> gr.Interface:
        """Crea interfaz Gradio."""
        return gr.Interface(
            fn=self.generate,
            inputs=[
                gr.Textbox(label="Prompt", placeholder="Describe la imagen...", lines=2),
                gr.Textbox(label="Negative Prompt", placeholder="Lo que no quieres...", lines=2),
                gr.Slider(minimum=20, maximum=100, value=50, step=5, label="Inference Steps"),
                gr.Slider(minimum=1.0, maximum=20.0, value=7.5, step=0.5, label="Guidance Scale"),
                gr.Number(label="Seed (optional)", value=None, precision=0)
            ],
            outputs=gr.Image(label="Generated Image"),
            title="Image Generation Demo",
            description="Genera imágenes usando Stable Diffusion",
            examples=[
                ["a beautiful sunset over mountains", "", 50, 7.5, None],
                ["a futuristic city at night", "blurry, low quality", 60, 8.0, None]
            ]
        )


if __name__ == "__main__":
    interface = ImageGenerationInterface()
    interface.create_interface().launch(share=True)
'''

