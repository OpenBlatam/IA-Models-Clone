from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import gradio as gr
from video_pipeline import crear_video_ugc_langchain
import logging
from typing import List, Optional
import asyncio

from typing import Any, List, Dict, Optional
logger = logging.getLogger("os_content.gradio")

# Dummy langchain_service for demo; replace with real instance if available
def get_langchain_service():
    
    """get_langchain_service function."""
return None

async def generate_ugc_video(
    prompt: str, 
    images: List[gr.File], 
    videos: List[gr.File], 
    duration_per_image: float
) -> tuple[Optional[str], str]:
    """Generate UGC video with improved error handling and validation"""
    try:
        # Input validation
        if not prompt or not prompt.strip():
            return None, "Error: Debes ingresar un prompt válido."
        
        if not images and not videos:
            return None, "Error: Debes subir al menos una imagen o video."
        
        # Validate duration
        if duration_per_image < 1 or duration_per_image > 10:
            return None, "Error: La duración por imagen debe estar entre 1 y 10 segundos."
        
        # Process file paths
        image_paths = []
        video_paths = []
        
        if images:
            for img in images:
                if hasattr(img, 'name') and img.name:
                    image_paths.append(img.name)
                else:
                    logger.warning("Invalid image file provided")
        
        if videos:
            for vid in videos:
                if hasattr(vid, 'name') and vid.name:
                    video_paths.append(vid.name)
                else:
                    logger.warning("Invalid video file provided")
        
        if not image_paths and not video_paths:
            return None, "Error: No se pudieron procesar los archivos subidos."
        
        # Generate video
        output_path = await crear_video_ugc_langchain(
            image_paths=image_paths,
            video_paths=video_paths,
            text_prompt=prompt,
            duration_per_image=duration_per_image,
            langchain_service=get_langchain_service()
        )
        
        if output_path and output_path != "Error":
            return output_path, "Video generado exitosamente."
        else:
            return None, "Error: No se pudo generar el video."
            
    except Exception as e:
        logger.error(f"Error en la generación de video: {str(e)}")
        return None, f"Error en la generación: {str(e)}"

# Create Gradio interface with improved configuration
demo = gr.Interface(
    fn=generate_ugc_video,
    inputs=[
        gr.Textbox(
            label="Prompt", 
            placeholder="Describe el video que quieres generar...",
            lines=3,
            max_lines=5
        ),
        gr.File(
            label="Imágenes", 
            file_count="multiple", 
            type="file", 
            file_types=["image"],
            height=100
        ),
        gr.File(
            label="Videos", 
            file_count="multiple", 
            type="file", 
            file_types=["video"],
            height=100
        ),
        gr.Slider(
            minimum=1, 
            maximum=10, 
            value=3, 
            step=0.5, 
            label="Duración por imagen (segundos)",
            info="Duración en segundos para cada imagen en el video"
        )
    ],
    outputs=[
        gr.Video(label="Video generado"),
        gr.Textbox(label="Mensaje", interactive=False)
    ],
    title="Generador de Video UGC",
    description="Sube imágenes y/o videos, escribe un prompt y genera un video automáticamente.",
    allow_flagging="never",
    theme=gr.themes.Soft(),
    css="""
        .gradio-container {
            max-width: 1200px;
            margin: 0 auto;
        }
        .file-upload {
            border: 2px dashed #ccc;
            border-radius: 8px;
            padding: 20px;
            text-align: center;
        }
    """
)

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Launch with improved settings
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=False,
        show_error=True
    ) 