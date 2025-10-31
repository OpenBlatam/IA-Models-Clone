from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import torch
import asyncio
from typing import Dict, List, Optional, Union, Any
import logging
from concurrent.futures import ThreadPoolExecutor
import time
from dataclasses import asdict
from .model import ProductDescriptionModel
from .config import ProductDescriptionConfig, ECOMMERCE_CONFIG, LUXURY_CONFIG, TECHNICAL_CONFIG
        import hashlib
from typing import Any, List, Dict, Optional
"""
Product Description Generator
=============================

High-level generator class that orchestrates the model for product description generation.
"""



logger = logging.getLogger(__name__)


class ProductDescriptionGenerator:
    """
    High-level Product Description Generator
    
    Features:
    - Async/sync generation modes
    - Batch processing
    - Multiple style presets
    - Performance optimization
    - Caching and rate limiting
    """
    
    def __init__(self, config: Optional[ProductDescriptionConfig] = None):
        
    """__init__ function."""
self.config = config or ProductDescriptionConfig()
        self.model: Optional[ProductDescriptionModel] = None
        self.is_initialized = False
        
        # Performance tracking
        self.stats = {
            "total_generations": 0,
            "total_time": 0.0,
            "avg_time_per_generation": 0.0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        
        # Simple in-memory cache
        self.cache = {}
        self.cache_max_size = self.config.cache_max_size
        
        # Thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    async def initialize(self) -> bool:
        """Initialize the generator and load model."""
        try:
            logger.info("Initializing Product Description Generator...")
            
            # Initialize model
            self.model = ProductDescriptionModel(self.config.model)
            
            # Move to device
            device = torch.device(self.config.model.device)
            self.model.to(device)
            
            # Set to evaluation mode
            self.model.eval()
            
            self.is_initialized = True
            logger.info("Product Description Generator initialized successfully!")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize generator: {e}")
            return False
    
    def generate(
        self,
        product_name: str,
        features: List[str],
        category: str = "general",
        brand: str = "unknown",
        price_range: str = "medium",
        style: str = "professional",
        tone: str = "friendly",
        max_length: int = 300,
        temperature: float = 0.7,
        num_variations: int = 1,
        use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Generate product descriptions synchronously.
        
        Args:
            product_name: Name of the product
            features: List of product features
            category: Product category
            brand: Brand name
            price_range: Price range (low, medium, high, luxury)
            style: Writing style (professional, casual, luxury, technical, creative)
            tone: Writing tone (friendly, formal, enthusiastic, informative, persuasive)
            max_length: Maximum description length
            temperature: Generation temperature (0.1-2.0)
            num_variations: Number of variations to generate
            use_cache: Whether to use caching
            
        Returns:
            List of generated descriptions with metadata
        """
        if not self.is_initialized:
            raise RuntimeError("Generator not initialized. Call await generator.initialize() first.")
        
        start_time = time.time()
        
        # Check cache
        cache_key = self._generate_cache_key(
            product_name, features, category, brand, style, tone, max_length, temperature
        )
        
        if use_cache and cache_key in self.cache:
            self.stats["cache_hits"] += 1
            return self.cache[cache_key]
        
        self.stats["cache_misses"] += 1
        
        # Generate descriptions
        results = self.model.generate_description(
            product_name=product_name,
            features=features,
            category=category,
            brand=brand,
            style=style,
            tone=tone,
            max_length=max_length,
            temperature=temperature,
            num_return_sequences=num_variations
        )
        
        # Enhance results with additional metadata
        enhanced_results = []
        for result in results:
            enhanced_result = {
                **result,
                "generation_params": {
                    "temperature": temperature,
                    "max_length": max_length,
                    "style": style,
                    "tone": tone,
                    "price_range": price_range
                },
                "timestamps": {
                    "generated_at": time.time(),
                    "generation_time_ms": (time.time() - start_time) * 1000
                }
            }
            enhanced_results.append(enhanced_result)
        
        # Cache results
        if use_cache:
            self._add_to_cache(cache_key, enhanced_results)
        
        # Update stats
        self._update_stats(start_time)
        
        return enhanced_results
    
    async def generate_async(
        self,
        product_name: str,
        features: List[str],
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Generate product descriptions asynchronously."""
        loop = asyncio.get_event_loop()
        
        return await loop.run_in_executor(
            self.executor,
            self.generate,
            product_name,
            features,
            **kwargs
        )
    
    def generate_batch(
        self,
        products: List[Dict[str, Any]],
        use_cache: bool = True,
        max_concurrent: int = 4
    ) -> List[List[Dict[str, Any]]]:
        """
        Generate descriptions for multiple products.
        
        Args:
            products: List of product dictionaries with keys:
                     ['product_name', 'features', 'category', 'brand', etc.]
            use_cache: Whether to use caching
            max_concurrent: Maximum concurrent generations
            
        Returns:
            List of results for each product
        """
        if not self.is_initialized:
            raise RuntimeError("Generator not initialized.")
        
        results = []
        
        for product in products:
            try:
                product_results = self.generate(
                    product_name=product["product_name"],
                    features=product["features"],
                    category=product.get("category", "general"),
                    brand=product.get("brand", "unknown"),
                    price_range=product.get("price_range", "medium"),
                    style=product.get("style", "professional"),
                    tone=product.get("tone", "friendly"),
                    max_length=product.get("max_length", 300),
                    temperature=product.get("temperature", 0.7),
                    num_variations=product.get("num_variations", 1),
                    use_cache=use_cache
                )
                results.append(product_results)
                
            except Exception as e:
                logger.error(f"Error generating description for {product.get('product_name', 'unknown')}: {e}")
                results.append([])
        
        return results
    
    async def generate_batch_async(
        self,
        products: List[Dict[str, Any]],
        max_concurrent: int = 4
    ) -> List[List[Dict[str, Any]]]:
        """Generate descriptions for multiple products asynchronously."""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def generate_single(product) -> Any:
            async with semaphore:
                return await self.generate_async(
                    product_name=product["product_name"],
                    features=product["features"],
                    category=product.get("category", "general"),
                    brand=product.get("brand", "unknown"),
                    price_range=product.get("price_range", "medium"),
                    style=product.get("style", "professional"),
                    tone=product.get("tone", "friendly"),
                    max_length=product.get("max_length", 300),
                    temperature=product.get("temperature", 0.7),
                    num_variations=product.get("num_variations", 1)
                )
        
        tasks = [generate_single(product) for product in products]
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    def generate_with_preset(
        self,
        product_name: str,
        features: List[str],
        preset: str = "ecommerce",
        **override_params
    ) -> List[Dict[str, Any]]:
        """
        Generate using predefined presets.
        
        Presets:
        - ecommerce: Standard e-commerce descriptions
        - luxury: High-end luxury product descriptions
        - technical: Technical/detailed descriptions
        """
        presets = {
            "ecommerce": {
                "style": "professional",
                "tone": "friendly",
                "temperature": 0.7,
                "max_length": 200
            },
            "luxury": {
                "style": "luxury",
                "tone": "sophisticated",
                "temperature": 0.8,
                "max_length": 350
            },
            "technical": {
                "style": "technical",
                "tone": "informative",
                "temperature": 0.6,
                "max_length": 500
            }
        }
        
        if preset not in presets:
            raise ValueError(f"Unknown preset: {preset}. Available: {list(presets.keys())}")
        
        params = presets[preset]
        params.update(override_params)
        
        return self.generate(
            product_name=product_name,
            features=features,
            **params
        )
    
    def _generate_cache_key(self, *args) -> str:
        """Generate cache key from parameters."""
        key_string = "|".join(str(arg) for arg in args)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _add_to_cache(self, key: str, value: Any):
        """Add result to cache with size limit."""
        if len(self.cache) >= self.cache_max_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[key] = value
    
    def _update_stats(self, start_time: float):
        """Update generation statistics."""
        generation_time = time.time() - start_time
        self.stats["total_generations"] += 1
        self.stats["total_time"] += generation_time
        self.stats["avg_time_per_generation"] = (
            self.stats["total_time"] / self.stats["total_generations"]
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get generation statistics."""
        cache_hit_rate = 0
        if self.stats["cache_hits"] + self.stats["cache_misses"] > 0:
            cache_hit_rate = self.stats["cache_hits"] / (
                self.stats["cache_hits"] + self.stats["cache_misses"]
            )
        
        return {
            **self.stats,
            "cache_hit_rate": cache_hit_rate,
            "cache_size": len(self.cache),
            "is_initialized": self.is_initialized,
            "model_device": self.config.model.device if self.model else None
        }
    
    def clear_cache(self) -> Any:
        """Clear the generation cache."""
        self.cache.clear()
        logger.info("Cache cleared")
    
    def __del__(self) -> Any:
        """Cleanup resources."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False) 