from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
TIMEOUT_SECONDS = 60

import spacy
from transformers import pipeline
import logging
from typing import List, Dict, Any, Optional
import torch
from diffusers import StableDiffusionPipeline
import asyncio
from functools import lru_cache
import gc
from cache_manager import cache, model_cache, cached
from async_processor import processor, run_in_thread
from typing import Any, List, Dict, Optional
# Dependencies:
#   spacy
#   transformers
#   torch
#   diffusers


# Performance libraries

logger = logging.getLogger("os_content.nlp")

SUPPORTED_LANGS = {"es", "en"}

# Global model cache
_model_cache = {}
_sentiment_cache = {}
_diffusion_cache = {}

@cached(ttl=3600, key_prefix="spacy_model")
async def get_spacy_model(lang: str):
    """Get spaCy model with improved caching and error handling"""
    if lang not in SUPPORTED_LANGS:
        logger.warning(f"Unsupported spaCy language: {lang}, defaulting to 'es'.")
        lang = 'es'
    
    # Check model cache first
    cached_model = model_cache.get_model(f"spacy_{lang}")
    if cached_model:
        return cached_model
    
    try:
        if lang == 'es':
            model = spacy.load('es_core_news_sm')
        elif lang == 'en':
            model = spacy.load('en_core_web_sm')
        else:
            model = spacy.blank(lang)
        
        # Cache the model
        model_cache.set_model(f"spacy_{lang}", model)
        logger.info(f"Loaded spaCy model for language: {lang}")
        return model
        
    except Exception as e:
        logger.error(f"Failed to load spaCy model for {lang}: {e}")
        fallback_model = spacy.blank(lang)
        model_cache.set_model(f"spacy_{lang}", fallback_model)
        return fallback_model

@cached(ttl=1800, key_prefix="sentiment_pipeline")
async def get_sentiment_pipeline(lang: str):
    """Get sentiment analysis pipeline with caching"""
    # Check model cache first
    cached_pipeline = model_cache.get_pipeline(f"sentiment_{lang}")
    if cached_pipeline:
        return cached_pipeline
    
    try:
        pipeline_instance = pipeline(
            'sentiment-analysis', 
            model='nlptown/bert-base-multilingual-uncased-sentiment',
            device=0 if torch.cuda.is_available() else -1
        )
        
        # Cache the pipeline
        model_cache.set_pipeline(f"sentiment_{lang}", pipeline_instance)
        logger.info("Loaded sentiment analysis pipeline")
        return pipeline_instance
        
    except Exception as e:
        logger.error(f"Failed to load sentiment pipeline: {e}")
        return None

@cached(ttl=7200, key_prefix="diffusion_pipeline")
async def get_diffusion_pipeline(model_id: str = "runwayml/stable-diffusion-v1-5", device: Optional[str] = None):
    """Get diffusion pipeline with memory optimization"""
    if not model_id:
        logger.error("model_id is required for Diffusers.")
        return None
    
    cache_key = f"diffusion_{model_id}_{device}"
    cached_pipeline = model_cache.get_pipeline(cache_key)
    if cached_pipeline:
        return cached_pipeline
    
    try:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Memory optimization for GPU
        if device == "cuda":
            torch.cuda.empty_cache()
        
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id, 
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )
        pipe = pipe.to(device)
        
        # Enable memory efficient attention if available
        if hasattr(pipe, 'enable_attention_slicing'):
            pipe.enable_attention_slicing()
        
        # Cache the pipeline
        model_cache.set_pipeline(cache_key, pipe)
        logger.info(f"Loaded diffusion pipeline: {model_id}")
        return pipe
        
    except Exception as e:
        logger.error(f"Failed to load Diffusers model: {e}")
        return None

@cached(ttl=300, key_prefix="image_generation")
async def generate_image_from_text(
    prompt: str, 
    model_id: str = "runwayml/stable-diffusion-v1-5", 
    device: Optional[str] = None, 
    num_inference_steps: int = 25
):
    """Generate image from text with improved error handling"""
    if not prompt or not prompt.strip():
        return {"error": "Prompt cannot be empty. Please provide a description for the image."}
    
    pipe = await get_diffusion_pipeline(model_id, device)
    if pipe is None:
        return {"error": "Image generation model could not be loaded. Please try again later or contact support."}
    
    try:
        current_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        with torch.autocast(current_device):
            image = pipe(
                prompt, 
                num_inference_steps=num_inference_steps,
                guidance_scale=7.5
            ).images[0]
        
        # Clean up memory
        if current_device == "cuda":
            torch.cuda.empty_cache()
        
        return image
        
    except Exception as e:
        logger.error(f"Error generating image with Diffusers: {e}")
        return {"error": "An error occurred while generating the image. Please try again later."}

async def analyze_nlp_sync(text: str, lang: str = 'es') -> Dict[str, Any]:
    """Synchronous NLP analysis with improved error handling"""
    if not text or not text.strip():
        return {"error": "Text cannot be empty. Please provide input text for analysis."}
    
    result = {}
    
    try:
        # Get spaCy model
        nlp_spacy = await get_spacy_model(lang)
        if nlp_spacy is None:
            return {"error": "NLP model could not be loaded. Please try again later or contact support."}
        
        # Process text
        doc = nlp_spacy(text)
        
        # Extract entities
        result['entities'] = [
            {'text': ent.text, 'label': ent.label_}
            for ent in doc.ents
        ]
        
        # Extract tokens
        result['tokens'] = [token.text for token in doc]
        
        # Sentiment analysis
        sentiment_pipeline = await get_sentiment_pipeline(lang)
        if sentiment_pipeline:
            try:
                sentiment = sentiment_pipeline(text[:512])  # Limit text length for sentiment
                result['sentiment'] = sentiment
            except Exception as e:
                logger.error(f"Sentiment analysis error: {e}")
                result['sentiment'] = 'error'
                result['error'] = 'An error occurred during sentiment analysis.'
        else:
            result['sentiment'] = 'not available'
            result['error'] = 'Sentiment analysis is temporarily unavailable.'
        
        # Clean up memory
        del doc
        if lang in SUPPORTED_LANGS:
            gc.collect()
        
    except Exception as e:
        logger.error(f"General NLP error: {e}")
        result['error'] = 'An internal error occurred during NLP analysis. Please try again later.'
    
    return result

async def analyze_nlp(text: str, lang: str = 'es') -> Dict[str, Any]:
    """Async NLP analysis with thread pool executor and caching"""
    if not text or not text.strip():
        return {"error": "Text cannot be empty. Please provide input text for analysis."}
    
    # Check cache first
    cache_key = f"nlp_analysis:{hash(text)}:{lang}"
    cached_result = await cache.get(cache_key)
    if cached_result:
        return cached_result
    
    try:
        # Run analysis in thread pool for CPU-intensive tasks
        result = await run_in_thread(analyze_nlp_sync, text, lang)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        
        # Cache the result
        await cache.set(cache_key, result, ttl=1800)  # Cache for 30 minutes
        
        return result
        
    except Exception as e:
        logger.error(f"Async NLP error: {e}")
        return {"error": "An internal error occurred during NLP analysis. Please try again later."}

async def batch_analyze_nlp(texts: List[str], lang: str = 'es') -> List[Dict[str, Any]]:
    """Batch NLP analysis for multiple texts"""
    if not texts:
        return []
    
    # Create tasks for batch processing
    tasks = []
    for text in texts:
        task = processor.submit_task(
            analyze_nlp,
            text,
            lang,
            priority=processor.TaskPriority.NORMAL,
            timeout=30
        )
        tasks.append(task)
    
    # Wait for all tasks to complete
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Handle exceptions
    processed_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"Batch NLP analysis failed for text {i}: {result}")
            processed_results.append({"error": str(result)})
        else:
            processed_results.append(result)
    
    return processed_results

def cleanup_models():
    """Clean up model cache to free memory"""
    global _model_cache, _sentiment_cache, _diffusion_cache
    
    # Clear global caches
    for cache_dict in [_model_cache, _sentiment_cache, _diffusion_cache]:
        cache_dict.clear()
    
    # Clear model cache
    model_cache.clear()
    
    # Clear multi-level cache
    asyncio.create_task(cache.clear())
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    gc.collect()
    logger.info("Model cache cleaned up")

async def get_nlp_stats() -> Dict[str, Any]:
    """Get NLP system statistics"""
    cache_stats = cache.get_stats()
    
    return {
        "cache_stats": cache_stats,
        "model_cache_size": len(model_cache.model_cache),
        "pipeline_cache_size": len(model_cache.pipeline_cache),
        "supported_languages": list(SUPPORTED_LANGS),
        "gpu_available": torch.cuda.is_available(),
        "gpu_memory_allocated": torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    } 