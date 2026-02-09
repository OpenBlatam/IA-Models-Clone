import logging
import torch
import numpy as np
from typing import Optional, Any

logger = logging.getLogger(__name__)


class ModelLoader:
    def __init__(self, model_name: str, device: Optional[str] = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None
    
    def load(self):
        if self.model is not None:
            return
        
        try:
            from audiocraft.models import MusicGen
            
            logger.info(f"Loading model: {self.model_name}")
            self.model = MusicGen.get_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            
            try:
                if hasattr(torch, 'compile') and self.device == 'cuda':
                    self.model = torch.compile(self.model, mode='reduce-overhead')
                    logger.info("Model compiled with torch.compile")
            except Exception:
                pass
            
            logger.info("Model loaded successfully")
        except ImportError:
            from transformers import AutoProcessor, MusicgenForConditionalGeneration
            
            logger.info(f"Loading model via transformers: {self.model_name}")
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = MusicgenForConditionalGeneration.from_pretrained(
                self.model_name
            ).to(self.device)
            self.model.eval()
            
            try:
                if hasattr(torch, 'compile') and self.device == 'cuda':
                    self.model = torch.compile(self.model, mode='reduce-overhead')
                    logger.info("Model compiled with torch.compile")
            except Exception:
                pass
            
            logger.info("Model loaded via transformers")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def generate_audio(
        self,
        prompt: str,
        duration: int,
        guidance_scale: float,
        temperature: float,
        fast_mode: bool = False
    ) -> Any:
        if not prompt:
            raise ValueError("Prompt cannot be empty")
        
        self.load()
        
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        if hasattr(self.model, 'set_generation_params'):
            params = {
                'duration': duration,
                'temperature': temperature,
                'cfg_coef': guidance_scale,
            }
            if not fast_mode:
                params.update({'top_k': 250, 'top_p': 0.0, 'use_sampling': True})
            else:
                params.update({'top_k': 50, 'top_p': 0.0, 'use_sampling': True})
            self.model.set_generation_params(**params)
            try:
                with torch.inference_mode(), torch.cuda.amp.autocast(enabled=self.device == 'cuda'):
                    audio = self.model.generate([prompt])
            except AttributeError:
                with torch.cuda.amp.autocast(enabled=self.device == 'cuda'):
                    audio = self.model.generate([prompt])
            
            result = audio[0].cpu().numpy()
            if result.dtype == np.float32:
                return result
            return result.astype(np.float32, copy=False)
        else:
            if self.processor is None:
                raise RuntimeError("Processor not loaded")
            
            inputs = self.processor(
                text=[prompt],
                padding=True,
                return_tensors="pt"
            ).to(self.device)
            
            try:
                with torch.inference_mode(), torch.cuda.amp.autocast(enabled=self.device == 'cuda'):
                    audio_values = self.model.generate(
                        **inputs,
                        max_new_tokens=int(duration * 50),
                        do_sample=True,
                        guidance_scale=guidance_scale,
                        temperature=temperature,
                        num_beams=1 if fast_mode else 3
                    )
            except AttributeError:
                with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.device == 'cuda'):
                    audio_values = self.model.generate(
                        **inputs,
                        max_new_tokens=int(duration * 50),
                        do_sample=True,
                        guidance_scale=guidance_scale,
                        temperature=temperature,
                        num_beams=1 if fast_mode else 3
                    )
            
            result = audio_values[0, 0].cpu().numpy()
            if result.dtype == np.float32:
                return result
            return result.astype(np.float32, copy=False)
    
    def optimize_memory(self):
        if self.model:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                torch.cuda.ipc_collect()
            
            self.model.eval()
            
            if hasattr(self.model, 'half'):
                try:
                    self.model = self.model.half()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    logger.info("Model converted to half precision")
                except Exception as e:
                    logger.warning(f"Could not convert to half precision: {e}")
    
    def clear_cache(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.model = None
        self.processor = None
        logger.info("Cache cleared")

