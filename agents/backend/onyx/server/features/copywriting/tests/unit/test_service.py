"""
Unit tests for copywriting service layer.
"""
import pytest
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any, List
import asyncio

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from service import CopywritingService
from models import (
    CopywritingInput,
    CopywritingOutput,
    Feedback
)


class TestCopywritingService:
    """Test cases for CopywritingService."""
    
    @pytest.fixture
    def service(self):
        """Create a service instance for testing."""
        return CopywritingService()
    
    @pytest.fixture
    def sample_request(self):
        """Create a sample copywriting request."""
        return CopywritingRequest(
            product_description="Zapatos deportivos de alta gama",
            target_platform="Instagram",
            tone="inspirational",
            target_audience="Jóvenes activos",
            key_points=["Comodidad", "Estilo", "Durabilidad"],
            instructions="Enfatiza la innovación",
            restrictions=["no mencionar precio"],
            creativity_level=0.8,
            language="es"
        )
    
    @pytest.mark.asyncio
    async def test_generate_copywriting_success(self, service, sample_request):
        """Test successful copywriting generation."""
        with patch.object(service, '_call_ai_model') as mock_call:
            mock_call.return_value = {
                "variants": [
                    {
                        "headline": "¡Descubre la Comodidad Perfecta!",
                        "primary_text": "Zapatos deportivos diseñados para tu máximo rendimiento",
                        "call_to_action": "Compra ahora",
                        "hashtags": ["#deportes", "#comodidad"]
                    }
                ],
                "model_used": "gpt-3.5-turbo",
                "generation_time": 2.5,
                "extra_metadata": {"tokens_used": 150}
            }
            
            result = await service.generate_copywriting(sample_request, "gpt-3.5-turbo")
            
            assert isinstance(result, CopywritingResponse)
            assert len(result.variants) == 1
            assert result.model_used == "gpt-3.5-turbo"
            assert result.generation_time == 2.5
            mock_call.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_generate_copywriting_invalid_model(self, service, sample_request):
        """Test copywriting generation with invalid model."""
        with pytest.raises(ValueError, match="Modelo no soportado"):
            await service.generate_copywriting(sample_request, "invalid_model")
    
    @pytest.mark.asyncio
    async def test_generate_copywriting_ai_error(self, service, sample_request):
        """Test copywriting generation with AI model error."""
        with patch.object(service, '_call_ai_model') as mock_call:
            mock_call.side_effect = Exception("AI model error")
            
            with pytest.raises(Exception, match="AI model error"):
                await service.generate_copywriting(sample_request, "gpt-3.5-turbo")
    
    @pytest.mark.asyncio
    async def test_batch_generate_copywriting_success(self, service):
        """Test successful batch copywriting generation."""
        requests = [
            CopywritingRequest(
                product_description="Zapatos deportivos",
                target_platform="Instagram",
                tone="inspirational",
                language="es"
            ),
            CopywritingRequest(
                product_description="Reloj inteligente",
                target_platform="Facebook",
                tone="informative",
                language="es"
            )
        ]
        
        batch_request = BatchCopywritingRequest(requests=requests)
        
        with patch.object(service, 'generate_copywriting') as mock_generate:
            mock_generate.side_effect = [
                CopywritingResponse(
                    variants=[{"headline": "Zapatos", "primary_text": "Texto"}],
                    model_used="gpt-3.5-turbo",
                    generation_time=1.0,
                    extra_metadata={}
                ),
                CopywritingResponse(
                    variants=[{"headline": "Reloj", "primary_text": "Texto"}],
                    model_used="gpt-3.5-turbo",
                    generation_time=1.5,
                    extra_metadata={}
                )
            ]
            
            result = await service.batch_generate_copywriting(batch_request)
            
            assert len(result.results) == 2
            assert mock_generate.call_count == 2
    
    @pytest.mark.asyncio
    async def test_batch_generate_copywriting_too_large(self, service):
        """Test batch generation with too many requests."""
        requests = [
            CopywritingRequest(
                product_description=f"Producto {i}",
                target_platform="Instagram",
                tone="inspirational",
                language="es"
            )
            for i in range(25)  # Exceeds limit
        ]
        
        batch_request = BatchCopywritingRequest(requests=requests)
        
        with pytest.raises(ValueError, match="batch máximo"):
            await service.batch_generate_copywriting(batch_request)
    
    def test_get_available_models(self, service):
        """Test getting available models."""
        models = service.get_available_models()
        
        assert isinstance(models, list)
        assert "gpt-3.5-turbo" in models
        assert "gpt-4" in models
        assert len(models) > 0
    
    def test_validate_model_success(self, service):
        """Test model validation with valid model."""
        assert service.validate_model("gpt-3.5-turbo") is True
        assert service.validate_model("gpt-4") is True
    
    def test_validate_model_failure(self, service):
        """Test model validation with invalid model."""
        assert service.validate_model("invalid_model") is False
        assert service.validate_model("") is False
        assert service.validate_model(None) is False
    
    @pytest.mark.asyncio
    async def test_call_ai_model_success(self, service, sample_request):
        """Test successful AI model call."""
        with patch('openai.ChatCompletion.create') as mock_create:
            mock_create.return_value = {
                "choices": [
                    {
                        "message": {
                            "content": '{"variants": [{"headline": "Test", "primary_text": "Content"}]}'
                        }
                    }
                ],
                "usage": {"total_tokens": 100}
            }
            
            result = await service._call_ai_model(sample_request, "gpt-3.5-turbo")
            
            assert "variants" in result
            assert len(result["variants"]) == 1
            assert result["variants"][0]["headline"] == "Test"
    
    @pytest.mark.asyncio
    async def test_call_ai_model_invalid_response(self, service, sample_request):
        """Test AI model call with invalid response format."""
        with patch('openai.ChatCompletion.create') as mock_create:
            mock_create.return_value = {
                "choices": [
                    {
                        "message": {
                            "content": "Invalid JSON response"
                        }
                    }
                ]
            }
            
            with pytest.raises(ValueError, match="Respuesta inválida"):
                await service._call_ai_model(sample_request, "gpt-3.5-turbo")
    
    @pytest.mark.asyncio
    async def test_call_ai_model_api_error(self, service, sample_request):
        """Test AI model call with API error."""
        with patch('openai.ChatCompletion.create') as mock_create:
            mock_create.side_effect = Exception("API Error")
            
            with pytest.raises(Exception, match="API Error"):
                await service._call_ai_model(sample_request, "gpt-3.5-turbo")
    
    def test_format_prompt(self, service, sample_request):
        """Test prompt formatting."""
        prompt = service._format_prompt(sample_request)
        
        assert isinstance(prompt, str)
        assert "Zapatos deportivos de alta gama" in prompt
        assert "Instagram" in prompt
        assert "inspirational" in prompt
        assert "Jóvenes activos" in prompt
        assert "Comodidad" in prompt
        assert "Estilo" in prompt
        assert "Durabilidad" in prompt
    
    def test_format_prompt_minimal(self, service):
        """Test prompt formatting with minimal data."""
        request = CopywritingRequest(
            product_description="Producto de prueba",
            target_platform="Facebook",
            tone="informative",
            language="es"
        )
        
        prompt = service._format_prompt(request)
        
        assert isinstance(prompt, str)
        assert "Producto de prueba" in prompt
        assert "Facebook" in prompt
        assert "informative" in prompt
        assert "es" in prompt
    
    def test_parse_ai_response_success(self, service):
        """Test successful AI response parsing."""
        response_text = '{"variants": [{"headline": "Test", "primary_text": "Content"}]}'
        
        result = service._parse_ai_response(response_text)
        
        assert "variants" in result
        assert len(result["variants"]) == 1
        assert result["variants"][0]["headline"] == "Test"
    
    def test_parse_ai_response_invalid_json(self, service):
        """Test AI response parsing with invalid JSON."""
        response_text = "Invalid JSON"
        
        with pytest.raises(ValueError, match="JSON inválido"):
            service._parse_ai_response(response_text)
    
    def test_parse_ai_response_missing_variants(self, service):
        """Test AI response parsing with missing variants."""
        response_text = '{"other_field": "value"}'
        
        with pytest.raises(ValueError, match="variants"):
            service._parse_ai_response(response_text)





