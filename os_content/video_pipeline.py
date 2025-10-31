from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import os
import tempfile
import numpy as np
from typing import List, Optional, Any
from PIL import Image, ImageOps
from tqdm import tqdm
import cv2
from gtts import gTTS
from moviepy.editor import ImageClip, VideoFileClip, concatenate_videoclips, AudioFileClip
import logging
from pathlib import Path
import gc

    import asyncio
from typing import Any, List, Dict, Optional
logger = logging.getLogger("os_content.video")

def calcular_calidad_media(ruta_archivo: str) -> float:
    """Calculate media quality with improved error handling"""
    try:
        if not os.path.exists(ruta_archivo):
            logger.warning(f"File does not exist: {ruta_archivo}")
            return 0.0
            
        extension = Path(ruta_archivo).suffix.lower()
        
        if extension in [".jpg", ".jpeg", ".png"]:
            imagen = cv2.imread(ruta_archivo)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            if imagen is None:
                logger.warning(f"Could not read image: {ruta_archivo}")
                return 0.0
                
            escala_grises = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
            var_laplaciano = cv2.Laplacian(escala_grises, cv2.CV_64F).var()
            brillo = np.mean(escala_grises) / 255.0
            puntaje = 0.5 * var_laplaciano + 0.5 * brillo * 10
            
            # Clean up memory
            del imagen, escala_grises
            return float(np.clip(puntaje, 0, 10))
            
        elif extension in [".mp4", ".mov", ".avi"]:
            captura = cv2.VideoCapture(ruta_archivo)
            if not captura.isOpened():
                logger.warning(f"Could not open video: {ruta_archivo}")
                return 0.0
                
            ret, cuadro = captura.read()
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            captura.release()
            
            if not ret or cuadro is None:
                logger.warning(f"Could not read video frame: {ruta_archivo}")
                return 0.0
                
            escala_grises = cv2.cvtColor(cuadro, cv2.COLOR_BGR2GRAY)
            var_laplaciano = cv2.Laplacian(escala_grises, cv2.CV_64F).var()
            brillo = np.mean(escala_grises) / 255.0
            puntaje = 0.5 * var_laplaciano + 0.5 * brillo * 10
            
            # Clean up memory
            del cuadro, escala_grises
            return float(np.clip(puntaje, 0, 10))
            
        logger.warning(f"Unsupported file type: {extension}")
        return 0.0
        
    except Exception as e:
        logger.error(f"Error calculating media quality for {ruta_archivo}: {e}")
        return 0.0

def preprocesar_imagen(ruta: str, resolucion: tuple) -> str:
    """Preprocess image with improved error handling and memory management"""
    try:
        if not os.path.exists(ruta):
            logger.error(f"Image file does not exist: {ruta}")
            return ruta
            
        imagen = Image.open(ruta)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        imagen = ImageOps.exif_transpose(imagen)
        imagen = ImageOps.fit(imagen, resolucion, Image.LANCZOS)
        
        # Create temp file with better naming
        temp_dir = Path(tempfile.gettempdir())
        temp_filename = f"pre_{Path(ruta).stem}_{os.urandom(4).hex()}.jpg"
        ruta_temp = temp_dir / temp_filename
        
        imagen.save(ruta_temp, format="JPEG", quality=95, optimize=True)
        imagen.close()
        
        return str(ruta_temp)
        
    except Exception as e:
        logger.error(f"Error preprocessing image {ruta}: {e}")
        return ruta

def generar_voiceover(texto: str, idioma: str = "es") -> str:
    """Generate voiceover with improved error handling"""
    try:
        if not texto or not texto.strip():
            logger.warning("Empty text for voiceover generation")
            return ""
            
        tts = gTTS(text=texto, lang=idioma, slow=False)
        ruta_audio = Path(tempfile.gettempdir()) / f"voiceover_{os.urandom(8).hex()}.mp3"
        tts.save(str(ruta_audio))
        
        return str(ruta_audio)
        
    except Exception as e:
        logger.error(f"Error generating voiceover: {e}")
        return ""

def crear_video_ugc(
    rutas_imagenes: List[str],
    rutas_videos: List[str],
    guion: str,
    ruta_salida: Optional[str] = None,
    duracion_por_imagen: float = 3.0,
    resolucion: tuple = (1080, 1920),
    ruta_audio: Optional[str] = None,
    idioma: str = "es",
) -> str:
    """Create UGC video with improved memory management and error handling"""
    clips = []
    medios = []
    
    try:
        # Calculate quality for all media files
        for ruta in rutas_imagenes + rutas_videos:
            if os.path.exists(ruta):
                calidad = calcular_calidad_media(ruta)
                medios.append((ruta, calidad))
            else:
                logger.warning(f"Media file does not exist: {ruta}")
        
        # Sort by quality (highest first)
        medios.sort(key=lambda x: -x[1])
        
        if not medios:
            raise ValueError("No se encontraron imágenes o videos válidos para componer el video ad.")
        
        # Process media files
        for ruta, calidad in tqdm(medios, desc="Procesando media para video"):
            try:
                extension = Path(ruta).suffix.lower()
                
                if extension in [".jpg", ".jpeg", ".png"]:
                    ruta_pre = preprocesar_imagen(ruta, resolucion)
                    clip_img = ImageClip(ruta_pre).set_duration(duracion_por_imagen).resize(resolucion)
                    clips.append(clip_img)
                    
                elif extension in [".mp4", ".mov", ".avi"]:
                    clip_vid = VideoFileClip(ruta).resize(resolucion)
                    clips.append(clip_vid)
                    
            except Exception as e:
                logger.error(f"Error processing media file {ruta}: {e}")
                continue
        
        if not clips:
            raise ValueError("No se pudieron procesar imágenes o videos válidos.")
        
        # Concatenate clips
        clip_final = concatenate_videoclips(clips, method="compose")
        
        # Add audio
        audio_final = ruta_audio or generar_voiceover(guion, idioma=idioma)
        if audio_final and os.path.exists(audio_final):
            try:
                audio_clip = AudioFileClip(audio_final)
                clip_final = clip_final.set_audio(audio_clip)
            except Exception as e:
                logger.warning(f"Could not add audio: {e}")
        
        # Set output path
        if not ruta_salida:
            ruta_salida = Path(tempfile.gettempdir()) / f"ugc_video_{os.urandom(8).hex()}.mp4"
        
        # Write video file
        clip_final.write_videofile(
            str(ruta_salida), 
            fps=30, 
            codec="libx264", 
            audio_codec="aac",
            verbose=False,
            logger=None
        )
        
        # Clean up
        clip_final.close()
        for clip in clips:
            try:
                clip.close()
            except:
                pass
        
        # Force garbage collection
        gc.collect()
        
        return str(ruta_salida)
        
    except Exception as e:
        logger.error(f"Error creating UGC video: {e}")
        # Clean up clips on error
        for clip in clips:
            try:
                clip.close()
            except:
                pass
        raise

def generar_guion_langchain(texto_prompt: str, servicio_langchain: Any = None) -> str:
    """Generate script using LangChain with improved error handling"""
    if not servicio_langchain:
        return texto_prompt
    
    try:
        prompt = f"Genera un guion atractivo para un video social media basado en: {texto_prompt}"
        
        if hasattr(servicio_langchain, 'generate_script') and callable(servicio_langchain.generate_script):
            return servicio_langchain.generate_script(prompt)
            
        if hasattr(servicio_langchain, 'generate_ads') and callable(servicio_langchain.generate_ads):
            scripts = servicio_langchain.generate_ads(prompt, num_ads=1)
            return scripts[0] if scripts else texto_prompt
            
        return texto_prompt
        
    except Exception as e:
        logger.error(f"Error generating script with LangChain: {e}")
        return texto_prompt

async def crear_video_ugc_langchain(
    rutas_imagenes: List[str],
    rutas_videos: List[str],
    texto_prompt: str,
    ruta_salida: Optional[str] = None,
    duracion_por_imagen: float = 3.0,
    resolucion: tuple = (1080, 1920),
    ruta_audio: Optional[str] = None,
    servicio_langchain: Any = None,
    idioma: str = "es",
) -> str:
    """Async wrapper for creating UGC video with LangChain"""
    
    try:
        loop = asyncio.get_running_loop()
        guion = await loop.run_in_executor(None, generar_guion_langchain, texto_prompt, servicio_langchain)
        
        return await loop.run_in_executor(
            None,
            crear_video_ugc,
            rutas_imagenes,
            rutas_videos,
            guion,
            ruta_salida,
            duracion_por_imagen,
            resolucion,
            ruta_audio,
            idioma
        )
        
    except Exception as e:
        logger.error(f"Error in async video creation: {e}")
        raise

# Alias for backward compatibility
create_ugc_video_ad_with_langchain = crear_video_ugc_langchain 