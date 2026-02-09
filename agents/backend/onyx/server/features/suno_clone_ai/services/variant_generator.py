import logging
import asyncio
import numpy as np
from typing import List, Dict, Any, Callable, Optional

logger = logging.getLogger(__name__)


class VariantGenerator:
    @staticmethod
    def get_guidance_variations(base: float, count: int) -> List[float]:
        if count <= 4:
            return [
                base - 0.5,
                base,
                base + 0.5,
                base + 1.0
            ][:count]
        step = 0.5
        return [base - 0.5 + (i * step) for i in range(count)]
    
    @staticmethod
    def get_temperature_variations(base: float, count: int) -> List[float]:
        if count <= 4:
            return [
                base - 0.2,
                base,
                base + 0.2,
                base + 0.4
            ][:count]
        step = 0.2
        return [base - 0.2 + (i * step) for i in range(count)]
    
    @staticmethod
    def generate_variants_sync(
        generate_fn: Callable,
        prompt: str,
        num_variants: int,
        duration: int,
        sample_rate: Optional[int],
        base_guidance_scale: float,
        base_temperature: float
    ) -> List[Dict[str, Any]]:
        guidance_variations = VariantGenerator.get_guidance_variations(
            base_guidance_scale, num_variants
        )
        temperature_variations = VariantGenerator.get_temperature_variations(
            base_temperature, num_variants
        )
        
        variants = [None] * num_variants
        guidance_len = len(guidance_variations)
        temp_len = len(temperature_variations)
        
        for i in range(num_variants):
            guidance = guidance_variations[i % guidance_len]
            temperature = temperature_variations[i % temp_len]
            
            audio = generate_fn(
                prompt, duration, sample_rate, guidance, temperature
            )
            
            variants[i] = {
                'audio': audio,
                'variant_id': i + 1,
                'guidance_scale': guidance,
                'temperature': temperature,
                'prompt': prompt
            }
        
        return variants
    
    @staticmethod
    async def generate_variants_async(
        generate_async_fn: Callable,
        prompt: str,
        num_variants: int,
        duration: int,
        sample_rate: Optional[int],
        base_guidance_scale: float,
        base_temperature: float,
        max_concurrent: int = 2
    ) -> List[Dict[str, Any]]:
        guidance_variations = VariantGenerator.get_guidance_variations(
            base_guidance_scale, num_variants
        )
        temperature_variations = VariantGenerator.get_temperature_variations(
            base_temperature, num_variants
        )
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        guidance_len = len(guidance_variations)
        temp_len = len(temperature_variations)
        
        async def generate_one(i: int):
            async with semaphore:
                guidance = guidance_variations[i % guidance_len]
                temperature = temperature_variations[i % temp_len]
                
                audio = await generate_async_fn(
                    prompt, duration, sample_rate, guidance, temperature
                )
                
                return {
                    'audio': audio,
                    'variant_id': i + 1,
                    'guidance_scale': guidance,
                    'temperature': temperature,
                    'prompt': prompt
                }
        
        tasks = [generate_one(i) for i in range(num_variants)]
        return list(await asyncio.gather(*tasks))

