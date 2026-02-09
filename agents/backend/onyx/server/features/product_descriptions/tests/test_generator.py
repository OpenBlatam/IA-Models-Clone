from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import pytest
import asyncio
from unittest.mock import Mock, patch
import torch
from product_descriptions.core.generator import ProductDescriptionGenerator
from product_descriptions.core.config import ProductDescriptionConfig, ModelConfig
from typing import Any, List, Dict, Optional
import logging
"""
Tests for Product Description Generator
=======================================

Unit and integration tests for the generator functionality.
"""




class TestProductDescriptionGenerator:
    """Test suite for ProductDescriptionGenerator."""
    
    @pytest.fixture
    def config(self) -> Any:
        """Test configuration."""
        return ProductDescriptionConfig(
            model=ModelConfig(
                model_name="microsoft/DialoGPT-small",  # Smaller model for testing
                device="cpu",  # Force CPU for testing
                mixed_precision=False
            )
        )
    
    @pytest.fixture
    async def generator(self, config) -> Any:
        """Initialized generator for testing."""
        gen = ProductDescriptionGenerator(config)
        await gen.initialize()
        return gen
    
    def test_generator_initialization(self, config) -> Any:
        """Test generator initialization."""
        generator = ProductDescriptionGenerator(config)
        assert generator.config == config
        assert not generator.is_initialized
        assert generator.model is None
    
    @pytest.mark.asyncio
    async def test_async_initialization(self, config) -> Any:
        """Test async initialization."""
        generator = ProductDescriptionGenerator(config)
        result = await generator.initialize()
        
        assert result is True
        assert generator.is_initialized
        assert generator.model is not None
    
    @pytest.mark.asyncio
    async def test_generate_description(self, generator) -> Any:
        """Test basic description generation."""
        results = generator.generate(
            product_name="Test Product",
            features=["feature1", "feature2"],
            category="general",
            brand="TestBrand"
        )
        
        assert len(results) == 1
        assert "description" in results[0]
        assert "quality_score" in results[0]
        assert "seo_score" in results[0]
        assert "metadata" in results[0]
        
        # Check metadata
        metadata = results[0]["metadata"]
        assert metadata["product_name"] == "Test Product"
        assert metadata["category"] == "general"
        assert metadata["brand"] == "TestBrand"
    
    @pytest.mark.asyncio
    async def test_generate_multiple_variations(self, generator) -> Any:
        """Test generating multiple variations."""
        results = generator.generate(
            product_name="Test Product",
            features=["feature1", "feature2"],
            num_variations=3
        )
        
        assert len(results) == 3
        
        # Check that descriptions are different
        descriptions = [r["description"] for r in results]
        assert len(set(descriptions)) >= 2  # At least 2 should be different
    
    @pytest.mark.asyncio
    async def test_async_generation(self, generator) -> Any:
        """Test async generation."""
        results = await generator.generate_async(
            product_name="Async Test Product",
            features=["async_feature1", "async_feature2"]
        )
        
        assert len(results) == 1
        assert "description" in results[0]
    
    def test_batch_generation(self, generator) -> Any:
        """Test batch generation."""
        products = [
            {
                "product_name": "Product 1",
                "features": ["feature1", "feature2"],
                "category": "electronics"
            },
            {
                "product_name": "Product 2", 
                "features": ["feature3", "feature4"],
                "category": "home"
            }
        ]
        
        results = generator.generate_batch(products)
        
        assert len(results) == 2
        assert len(results[0]) == 1  # Default num_variations
        assert len(results[1]) == 1
    
    @pytest.mark.asyncio
    async def test_batch_async_generation(self, generator) -> Any:
        """Test async batch generation."""
        products = [
            {
                "product_name": "Async Product 1",
                "features": ["async_feature1"],
                "category": "general"
            },
            {
                "product_name": "Async Product 2",
                "features": ["async_feature2"],
                "category": "general"
            }
        ]
        
        results = await generator.generate_batch_async(products)
        
        assert len(results) == 2
    
    def test_preset_generation(self, generator) -> Any:
        """Test preset-based generation."""
        results = generator.generate_with_preset(
            product_name="Preset Test Product",
            features=["preset_feature1", "preset_feature2"],
            preset="ecommerce"
        )
        
        assert len(results) == 1
        assert "description" in results[0]
    
    def test_invalid_preset(self, generator) -> Any:
        """Test invalid preset handling."""
        with pytest.raises(ValueError, match="Unknown preset"):
            generator.generate_with_preset(
                product_name="Test",
                features=["test"],
                preset="invalid_preset"
            )
    
    def test_caching(self, generator) -> Any:
        """Test caching functionality."""
        # First generation
        results1 = generator.generate(
            product_name="Cache Test",
            features=["cache_feature"],
            use_cache=True
        )
        
        # Second generation with same parameters
        results2 = generator.generate(
            product_name="Cache Test",
            features=["cache_feature"],
            use_cache=True
        )
        
        # Should get same results from cache
        assert results1[0]["description"] == results2[0]["description"]
        
        # Check stats
        stats = generator.get_stats()
        assert stats["cache_hits"] > 0
    
    def test_cache_disable(self, generator) -> Any:
        """Test disabling cache."""
        results1 = generator.generate(
            product_name="No Cache Test",
            features=["no_cache_feature"],
            use_cache=False
        )
        
        results2 = generator.generate(
            product_name="No Cache Test",
            features=["no_cache_feature"],
            use_cache=False
        )
        
        # Results might be different since no caching
        assert "description" in results1[0]
        assert "description" in results2[0]
    
    def test_stats_tracking(self, generator) -> Any:
        """Test statistics tracking."""
        initial_stats = generator.get_stats()
        
        # Generate some descriptions
        generator.generate(
            product_name="Stats Test",
            features=["stats_feature"]
        )
        
        final_stats = generator.get_stats()
        
        assert final_stats["total_generations"] > initial_stats["total_generations"]
        assert final_stats["total_time"] > initial_stats["total_time"]
    
    def test_clear_cache(self, generator) -> Any:
        """Test cache clearing."""
        # Generate and cache something
        generator.generate(
            product_name="Clear Cache Test",
            features=["clear_feature"],
            use_cache=True
        )
        
        stats_before = generator.get_stats()
        cache_size_before = stats_before["cache_size"]
        
        # Clear cache
        generator.clear_cache()
        
        stats_after = generator.get_stats()
        assert stats_after["cache_size"] == 0
        assert stats_after["cache_size"] < cache_size_before
    
    def test_error_handling(self, generator) -> Any:
        """Test error handling for invalid inputs."""
        # Empty features should still work but might produce different results
        results = generator.generate(
            product_name="Error Test",
            features=[],  # Empty features
            category="general"
        )
        
        # Should still generate something
        assert len(results) == 1
    
    @pytest.mark.asyncio
    async def test_uninitialized_generator(self, config) -> Any:
        """Test using uninitialized generator."""
        generator = ProductDescriptionGenerator(config)
        
        with pytest.raises(RuntimeError, match="Generator not initialized"):
            generator.generate(
                product_name="Test",
                features=["test"]
            )


@pytest.mark.integration
class TestGeneratorIntegration:
    """Integration tests with real model loading."""
    
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_full_pipeline(self) -> Any:
        """Test complete generation pipeline."""
        config = ProductDescriptionConfig()
        generator = ProductDescriptionGenerator(config)
        
        # Initialize
        await generator.initialize()
        
        # Generate
        results = generator.generate(
            product_name="Integration Test Product",
            features=["high quality", "durable", "affordable"],
            category="general",
            brand="TestBrand",
            style="professional",
            tone="friendly",
            temperature=0.7,
            max_length=200
        )
        
        # Validate results
        assert len(results) == 1
        result = results[0]
        
        assert len(result["description"]) > 10
        assert 0 <= result["quality_score"] <= 1
        assert 0 <= result["seo_score"] <= 1
        assert result["metadata"]["word_count"] > 0
        assert result["metadata"]["char_count"] > 0 