"""
Tests para el generador de variantes
"""

import pytest
from unittest.mock import Mock
import numpy as np

from services.variant_generator import VariantGenerator


@pytest.mark.unit
class TestVariantGenerator:
    """Tests para el generador de variantes"""
    
    def test_get_guidance_variations_small_count(self):
        """Test de variaciones de guidance con count pequeño"""
        variations = VariantGenerator.get_guidance_variations(base=3.0, count=3)
        
        assert len(variations) == 3
        assert all(isinstance(v, float) for v in variations)
        assert 2.5 in variations  # base - 0.5
        assert 3.0 in variations  # base
    
    def test_get_guidance_variations_large_count(self):
        """Test de variaciones de guidance con count grande"""
        variations = VariantGenerator.get_guidance_variations(base=3.0, count=10)
        
        assert len(variations) == 10
        assert all(isinstance(v, float) for v in variations)
        assert variations[0] == 2.5  # base - 0.5
    
    def test_get_temperature_variations_small_count(self):
        """Test de variaciones de temperature con count pequeño"""
        variations = VariantGenerator.get_temperature_variations(base=0.9, count=3)
        
        assert len(variations) == 3
        assert all(isinstance(v, float) for v in variations)
        assert 0.7 in variations  # base - 0.2
        assert 0.9 in variations  # base
    
    def test_get_temperature_variations_large_count(self):
        """Test de variaciones de temperature con count grande"""
        variations = VariantGenerator.get_temperature_variations(base=0.9, count=10)
        
        assert len(variations) == 10
        assert all(isinstance(v, float) for v in variations)
        assert variations[0] == 0.7  # base - 0.2
    
    def test_generate_variants_sync(self):
        """Test de generación síncrona de variantes"""
        def mock_generate_fn(prompt, duration, sample_rate, guidance, temperature):
            return np.array([0.1, 0.2, 0.3])
        
        variants = VariantGenerator.generate_variants_sync(
            generate_fn=mock_generate_fn,
            prompt="Test song",
            num_variants=3,
            duration=30,
            sample_rate=44100,
            base_guidance_scale=3.0,
            base_temperature=0.9
        )
        
        assert len(variants) == 3
        for variant in variants:
            assert "audio" in variant
            assert "variant_id" in variant
            assert "guidance_scale" in variant
            assert "temperature" in variant
            assert "prompt" in variant
            assert variant["prompt"] == "Test song"
    
    @pytest.mark.asyncio
    async def test_generate_variants_async(self):
        """Test de generación asíncrona de variantes"""
        async def mock_generate_async_fn(prompt, duration, sample_rate, guidance, temperature):
            return np.array([0.1, 0.2, 0.3])
        
        variants = await VariantGenerator.generate_variants_async(
            generate_async_fn=mock_generate_async_fn,
            prompt="Test song",
            num_variants=3,
            duration=30,
            sample_rate=44100,
            base_guidance_scale=3.0,
            base_temperature=0.9
        )
        
        assert len(variants) == 3
        for variant in variants:
            assert "audio" in variant
            assert "variant_id" in variant
            assert "guidance_scale" in variant
            assert "temperature" in variant


@pytest.mark.integration
class TestVariantGeneratorIntegration:
    """Tests de integración para generador de variantes"""
    
    def test_full_variant_generation_workflow(self):
        """Test del flujo completo de generación de variantes"""
        def mock_generate_fn(prompt, duration, sample_rate, guidance, temperature):
            return np.array([0.1, 0.2, 0.3])
        
        # Generar variaciones
        guidance_vars = VariantGenerator.get_guidance_variations(3.0, 5)
        temp_vars = VariantGenerator.get_temperature_variations(0.9, 5)
        
        assert len(guidance_vars) == 5
        assert len(temp_vars) == 5
        
        # Generar variantes
        variants = VariantGenerator.generate_variants_sync(
            generate_fn=mock_generate_fn,
            prompt="Test",
            num_variants=5,
            duration=30,
            sample_rate=44100,
            base_guidance_scale=3.0,
            base_temperature=0.9
        )
        
        assert len(variants) == 5
        # Verificar que cada variante tiene diferentes parámetros
        guidance_scales = [v["guidance_scale"] for v in variants]
        temperatures = [v["temperature"] for v in variants]
        
        assert len(set(guidance_scales)) > 1 or len(set(temperatures)) > 1



