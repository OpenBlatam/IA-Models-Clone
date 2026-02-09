"""
Generación de Música Optimizada

Proporciona:
- Generación en batch optimizada
- Pipeline paralelo
- Optimización de memoria
- Pre-carga de modelos
- Generación incremental
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import torch
import numpy as np

logger = logging.getLogger(__name__)


class OptimizedMusicGenerator:
    """Generador de música optimizado"""
    
    def __init__(
        self,
        model_name: str = "facebook/musicgen-medium",
        device: Optional[str] = None,
        max_workers: int = 4,
        use_compile: bool = True
    ):
        """
        Args:
            model_name: Nombre del modelo
            device: Dispositivo (cuda/cpu)
            max_workers: Número máximo de workers
            use_compile: Usar torch.compile
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_workers = max_workers
        self.use_compile = use_compile
        
        self.model = None
        self.processor = None
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        logger.info(f"OptimizedMusicGenerator initialized (device: {self.device})")
    
    def _load_model(self):
        """Carga el modelo (lazy loading)"""
        if self.model is None:
            try:
                from transformers import AutoProcessor, MusicgenForConditionalGeneration
                
                logger.info(f"Loading model: {self.model_name}")
                self.processor = AutoProcessor.from_pretrained(self.model_name)
                self.model = MusicgenForConditionalGeneration.from_pretrained(
                    self.model_name
                ).to(self.device)
                
                if self.use_compile and hasattr(torch, 'compile'):
                    logger.info("Compiling model with torch.compile")
                    self.model = torch.compile(self.model)
                
                self.model.eval()
                logger.info("Model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                raise
    
    async def generate_batch_async(
        self,
        prompts: List[str],
        durations: Optional[List[int]] = None,
        max_concurrent: int = 4
    ) -> List[np.ndarray]:
        """
        Genera música en batch de forma asíncrona
        
        Args:
            prompts: Lista de prompts
            durations: Lista de duraciones (opcional)
            max_concurrent: Máximo de generaciones concurrentes
        
        Returns:
            Lista de audios generados
        """
        if durations is None:
            durations = [30] * len(prompts)
        
        # Limitar concurrencia
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def generate_one(prompt: str, duration: int, index: int):
            async with semaphore:
                try:
                    audio = await asyncio.to_thread(
                        self.generate_single,
                        prompt,
                        duration
                    )
                    return (index, audio)
                except Exception as e:
                    logger.error(f"Error generating audio {index}: {e}")
                    return (index, None)
        
        # Generar todas en paralelo
        tasks = [
            generate_one(prompt, durations[i], i)
            for i, prompt in enumerate(prompts)
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Ordenar por índice y extraer audios
        results.sort(key=lambda x: x[0])
        return [audio for _, audio in results if audio is not None]
    
    def generate_single(
        self,
        prompt: str,
        duration: int = 30
    ) -> np.ndarray:
        """
        Genera un solo audio
        
        Args:
            prompt: Prompt de texto
            duration: Duración en segundos
        
        Returns:
            Audio generado
        """
        self._load_model()
        
        try:
            # Procesar input
            inputs = self.processor(
                text=[prompt],
                padding=True,
                return_tensors="pt"
            ).to(self.device)
            
            # Generar
            with torch.no_grad():
                audio_values = self.model.generate(
                    **inputs,
                    max_new_tokens=int(duration * 50),  # Aproximado
                    do_sample=True,
                    guidance_scale=3.0
                )
            
            # Convertir a numpy
            audio = audio_values[0, 0].cpu().numpy()
            
            return audio
        except Exception as e:
            logger.error(f"Error generating audio: {e}")
            raise
    
    def generate_batch_optimized(
        self,
        prompts: List[str],
        durations: Optional[List[int]] = None,
        batch_size: int = 4
    ) -> List[np.ndarray]:
        """
        Genera música en batch optimizado
        
        Args:
            prompts: Lista de prompts
            durations: Lista de duraciones
            batch_size: Tamaño del batch
        
        Returns:
            Lista de audios
        """
        if durations is None:
            durations = [30] * len(prompts)
        
        self._load_model()
        results = []
        
        # Procesar en batches
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            batch_durations = durations[i:i + batch_size]
            
            try:
                # Procesar batch
                inputs = self.processor(
                    text=batch_prompts,
                    padding=True,
                    return_tensors="pt"
                ).to(self.device)
                
                # Generar batch
                with torch.no_grad():
                    audio_values = self.model.generate(
                        **inputs,
                        max_new_tokens=int(max(batch_durations) * 50),
                        do_sample=True,
                        guidance_scale=3.0
                    )
                
                # Convertir a numpy
                for j in range(len(batch_prompts)):
                    audio = audio_values[j, 0].cpu().numpy()
                    results.append(audio)
            
            except Exception as e:
                logger.error(f"Error in batch generation: {e}")
                # Generar individualmente como fallback
                for prompt, duration in zip(batch_prompts, batch_durations):
                    try:
                        audio = self.generate_single(prompt, duration)
                        results.append(audio)
                    except Exception as e2:
                        logger.error(f"Error generating single: {e2}")
                        results.append(None)
        
        return results
    
    def generate_incremental(
        self,
        prompt: str,
        duration: int = 30,
        chunk_size: int = 5,
        callback: Optional[Callable[[np.ndarray, int], None]] = None
    ) -> np.ndarray:
        """
        Genera música de forma incremental (chunk por chunk)
        
        Args:
            prompt: Prompt de texto
            duration: Duración total
            chunk_size: Tamaño de cada chunk en segundos
            callback: Callback para cada chunk generado
        
        Returns:
            Audio completo
        """
        self._load_model()
        
        chunks = []
        total_chunks = (duration + chunk_size - 1) // chunk_size
        
        for chunk_idx in range(total_chunks):
            try:
                # Generar chunk
                chunk_duration = min(chunk_size, duration - chunk_idx * chunk_size)
                
                inputs = self.processor(
                    text=[prompt],
                    padding=True,
                    return_tensors="pt"
                ).to(self.device)
                
                with torch.no_grad():
                    audio_values = self.model.generate(
                        **inputs,
                        max_new_tokens=int(chunk_duration * 50),
                        do_sample=True,
                        guidance_scale=3.0
                    )
                
                chunk = audio_values[0, 0].cpu().numpy()
                chunks.append(chunk)
                
                # Callback si se proporciona
                if callback:
                    callback(chunk, chunk_idx)
                
                logger.info(f"Generated chunk {chunk_idx + 1}/{total_chunks}")
            
            except Exception as e:
                logger.error(f"Error generating chunk {chunk_idx}: {e}")
                # Rellenar con silencio
                chunk = np.zeros(int(chunk_duration * 32000))
                chunks.append(chunk)
        
        # Concatenar chunks
        full_audio = np.concatenate(chunks)
        return full_audio[:int(duration * 32000)]  # Asegurar duración exacta
    
    def optimize_memory(self):
        """Optimiza el uso de memoria"""
        if self.model:
            # Limpiar caché de CUDA
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Cambiar a modo eval
            self.model.eval()
            
            logger.info("Memory optimized")
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del generador"""
        stats = {
            "model_name": self.model_name,
            "device": self.device,
            "model_loaded": self.model is not None,
            "use_compile": self.use_compile,
            "max_workers": self.max_workers
        }
        
        if torch.cuda.is_available() and self.device == "cuda":
            stats["gpu_memory_allocated"] = torch.cuda.memory_allocated() / (1024**3)
            stats["gpu_memory_reserved"] = torch.cuda.memory_reserved() / (1024**3)
        
        return stats


# Instancia global
_optimized_generator: Optional[OptimizedMusicGenerator] = None


def get_optimized_generator(
    model_name: str = "facebook/musicgen-medium",
    device: Optional[str] = None
) -> OptimizedMusicGenerator:
    """Obtiene la instancia global del generador optimizado"""
    global _optimized_generator
    if _optimized_generator is None:
        _optimized_generator = OptimizedMusicGenerator(
            model_name=model_name,
            device=device
        )
    return _optimized_generator

