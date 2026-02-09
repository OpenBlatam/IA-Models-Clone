"""
Gradio Demos Service - Demos interactivos con Gradio
=====================================================

Sistema para crear demos interactivos de modelos usando Gradio.
Sigue mejores prácticas de Gradio y UX.
"""

import logging
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
import io

logger = logging.getLogger(__name__)

# Try to import Gradio
try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False
    logger.warning("Gradio not available. Install with: pip install gradio")


@dataclass
class GradioDemoConfig:
    """Configuración de demo Gradio"""
    title: str
    description: str = ""
    theme: str = "default"  # default, soft, monochrome
    share: bool = False
    server_name: str = "0.0.0.0"
    server_port: int = 7860
    show_error: bool = True
    show_tips: bool = True


class GradioDemosService:
    """Servicio para crear demos interactivos con Gradio"""
    
    def __init__(self):
        """Inicializar servicio"""
        self.demos: Dict[str, Any] = {}
        logger.info(f"GradioDemosService initialized (Gradio: {GRADIO_AVAILABLE})")
    
    def create_llm_demo(
        self,
        demo_id: str,
        generate_fn: Callable[[str, Optional[int], Optional[float]], str],
        config: Optional[GradioDemoConfig] = None
    ) -> Any:
        """
        Crear demo interactivo para LLM.
        
        Args:
            demo_id: ID único del demo
            generate_fn: Función que genera texto (prompt, max_tokens, temperature) -> text
            config: Configuración del demo
        
        Returns:
            Gradio Interface
        """
        if not GRADIO_AVAILABLE:
            raise RuntimeError("Gradio not available. Install with: pip install gradio")
        
        config = config or GradioDemoConfig(
            title="LLM Text Generation",
            description="Generate text using a Large Language Model"
        )
        
        def generate_wrapper(
            prompt: str,
            max_tokens: int = 100,
            temperature: float = 1.0
        ) -> str:
            """Wrapper con manejo de errores"""
            try:
                if not prompt or not prompt.strip():
                    return "Please enter a prompt."
                
                result = generate_fn(prompt, max_tokens, temperature)
                return result
            except Exception as e:
                logger.error(f"Error in LLM generation: {e}", exc_info=True)
                return f"Error: {str(e)}"
        
        # Create interface
        interface = gr.Interface(
            fn=generate_wrapper,
            inputs=[
                gr.Textbox(
                    label="Prompt",
                    placeholder="Enter your prompt here...",
                    lines=3,
                ),
                gr.Slider(
                    minimum=1,
                    maximum=500,
                    value=100,
                    step=1,
                    label="Max Tokens"
                ),
                gr.Slider(
                    minimum=0.1,
                    maximum=2.0,
                    value=1.0,
                    step=0.1,
                    label="Temperature"
                ),
            ],
            outputs=gr.Textbox(
                label="Generated Text",
                lines=10,
            ),
            title=config.title,
            description=config.description,
            theme=config.theme,
            examples=[
                ["Write a short story about AI", 200, 0.8],
                ["Explain quantum computing", 150, 0.7],
                ["Write a poem", 100, 1.2],
            ] if config.show_tips else None,
        )
        
        self.demos[demo_id] = interface
        logger.info(f"LLM demo created: {demo_id}")
        return interface
    
    def create_image_generation_demo(
        self,
        demo_id: str,
        generate_fn: Callable[[str, Optional[str], Optional[int]], bytes],
        config: Optional[GradioDemoConfig] = None
    ) -> Any:
        """
        Crear demo para generación de imágenes.
        
        Args:
            demo_id: ID único del demo
            generate_fn: Función que genera imagen (prompt, negative_prompt, steps) -> image_bytes
            config: Configuración del demo
        
        Returns:
            Gradio Interface
        """
        if not GRADIO_AVAILABLE:
            raise RuntimeError("Gradio not available")
        
        config = config or GradioDemoConfig(
            title="Image Generation",
            description="Generate images using diffusion models"
        )
        
        def generate_wrapper(
            prompt: str,
            negative_prompt: Optional[str] = None,
            num_steps: int = 50
        ) -> Optional[Any]:
            """Wrapper con manejo de errores"""
            try:
                if not prompt or not prompt.strip():
                    return None
                
                image_bytes = generate_fn(prompt, negative_prompt, num_steps)
                
                # Convert bytes to PIL Image for Gradio
                try:
                    from PIL import Image
                    image = Image.open(io.BytesIO(image_bytes))
                    return image
                except Exception as e:
                    logger.error(f"Error converting image: {e}")
                    return None
                    
            except Exception as e:
                logger.error(f"Error in image generation: {e}", exc_info=True)
                return None
        
        interface = gr.Interface(
            fn=generate_wrapper,
            inputs=[
                gr.Textbox(
                    label="Prompt",
                    placeholder="A beautiful landscape...",
                    lines=2,
                ),
                gr.Textbox(
                    label="Negative Prompt (optional)",
                    placeholder="blurry, low quality",
                    lines=2,
                ),
                gr.Slider(
                    minimum=10,
                    maximum=100,
                    value=50,
                    step=5,
                    label="Inference Steps"
                ),
            ],
            outputs=gr.Image(label="Generated Image"),
            title=config.title,
            description=config.description,
            theme=config.theme,
            examples=[
                ["A futuristic city at sunset", "", 50],
                ["A cute cat playing", "blurry", 40],
            ] if config.show_tips else None,
        )
        
        self.demos[demo_id] = interface
        logger.info(f"Image generation demo created: {demo_id}")
        return interface
    
    def create_chatbot_demo(
        self,
        demo_id: str,
        chat_fn: Callable[[str, List[Tuple[str, str]]], str],
        config: Optional[GradioDemoConfig] = None
    ) -> Any:
        """
        Crear demo de chatbot conversacional.
        
        Args:
            demo_id: ID único del demo
            chat_fn: Función de chat (message, history) -> response
            config: Configuración del demo
        
        Returns:
            Gradio ChatInterface
        """
        if not GRADIO_AVAILABLE:
            raise RuntimeError("Gradio not available")
        
        config = config or GradioDemoConfig(
            title="AI Chatbot",
            description="Chat with an AI assistant"
        )
        
        def chat_wrapper(message: str, history: List[Tuple[str, str]]) -> str:
            """Wrapper con manejo de errores"""
            try:
                if not message or not message.strip():
                    return ""
                
                response = chat_fn(message, history)
                return response
            except Exception as e:
                logger.error(f"Error in chat: {e}", exc_info=True)
                return f"I apologize, but I encountered an error: {str(e)}"
        
        # Use ChatInterface for better UX
        if hasattr(gr, "ChatInterface"):
            interface = gr.ChatInterface(
                fn=chat_wrapper,
                title=config.title,
                description=config.description,
                theme=config.theme,
            )
        else:
            # Fallback to regular Interface
            interface = gr.Interface(
                fn=lambda msg: chat_wrapper(msg, []),
                inputs=gr.Textbox(label="Message", lines=2),
                outputs=gr.Textbox(label="Response", lines=5),
                title=config.title,
                description=config.description,
            )
        
        self.demos[demo_id] = interface
        logger.info(f"Chatbot demo created: {demo_id}")
        return interface
    
    def launch_demo(
        self,
        demo_id: str,
        config: Optional[GradioDemoConfig] = None
    ) -> None:
        """
        Lanzar demo.
        
        Args:
            demo_id: ID del demo
            config: Configuración de lanzamiento
        """
        if demo_id not in self.demos:
            raise ValueError(f"Demo {demo_id} not found")
        
        demo = self.demos[demo_id]
        config = config or GradioDemoConfig()
        
        logger.info(f"Launching demo {demo_id} on {config.server_name}:{config.server_port}")
        
        demo.launch(
            server_name=config.server_name,
            server_port=config.server_port,
            share=config.share,
            show_error=config.show_error,
        )
    
    def list_demos(self) -> List[str]:
        """Listar demos disponibles"""
        return list(self.demos.keys())




