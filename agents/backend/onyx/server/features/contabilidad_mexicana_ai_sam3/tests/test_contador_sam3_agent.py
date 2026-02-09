"""
Tests for ContadorSAM3Agent
===========================

Refactored to:
- Use test helpers for common patterns
- Eliminate duplicate mock setup
- Standardize test assertions
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from contabilidad_mexicana_ai_sam3 import ContadorSAM3Agent, ContadorSAM3Config
from .test_helpers import (
    create_mock_openrouter_response,
    patch_openrouter_client,
    assert_task_submitted
)


@pytest.fixture
def config():
    """Create test configuration."""
    return ContadorSAM3Config(
        openrouter=ContadorSAM3Config.OpenRouterConfig(
            api_key="test-key",
            model="anthropic/claude-3.5-sonnet"
        )
    )


@pytest.fixture
async def agent(config):
    """Create test agent."""
    agent = ContadorSAM3Agent(config=config)
    yield agent
    await agent.close()


@pytest.mark.asyncio
async def test_calcular_impuestos(agent):
    """Test tax calculation."""
    mock_response = create_mock_openrouter_response(
        response_text="ISR calculado: $5,000",
        tokens_used=100
    )
    
    with patch_openrouter_client(agent, mock_response):
        task_id = await agent.calcular_impuestos(
            regimen="RESICO",
            tipo_impuesto="ISR",
            datos={"ingresos": 100000, "gastos": 30000}
        )
        
        # Wait a bit for processing
        await asyncio.sleep(0.1)
        
        await assert_task_submitted(agent, task_id)


@pytest.mark.asyncio
async def test_asesoria_fiscal(agent):
    """Test fiscal advice."""
    mock_response = create_mock_openrouter_response(
        response_text="Puedes deducir gastos de home office...",
        tokens_used=150
    )
    
    with patch_openrouter_client(agent, mock_response):
        task_id = await agent.asesoria_fiscal(
            pregunta="¿Puedo deducir gastos de home office?",
            contexto={"regimen": "RESICO"}
        )
        
        assert task_id is not None


@pytest.mark.asyncio
async def test_guia_fiscal(agent):
    """Test fiscal guide."""
    mock_response = create_mock_openrouter_response(
        response_text="Guía completa sobre deducciones...",
        tokens_used=200
    )
    
    with patch_openrouter_client(agent, mock_response):
        task_id = await agent.guia_fiscal(
            tema="Deducciones RESICO",
            nivel_detalle="completo"
        )
        
        assert task_id is not None


@pytest.mark.asyncio
async def test_tramite_sat(agent):
    """Test SAT procedure."""
    mock_response = create_mock_openrouter_response(
        response_text="Información sobre alta en RFC...",
        tokens_used=120
    )
    
    with patch_openrouter_client(agent, mock_response):
        task_id = await agent.tramite_sat(
            tipo_tramite="Alta en RFC",
            detalles={"persona_fisica": True}
        )
        
        assert task_id is not None


@pytest.mark.asyncio
async def test_ayuda_declaracion(agent):
    """Test declaration assistance."""
    mock_response = create_mock_openrouter_response(
        response_text="Guía para preparar declaración...",
        tokens_used=180
    )
    
    with patch_openrouter_client(agent, mock_response):
        task_id = await agent.ayuda_declaracion(
            tipo_declaracion="mensual",
            periodo="2024-01",
            datos={"rfc": "ABC123456789"}
        )
        
        assert task_id is not None


@pytest.mark.asyncio
async def test_task_priority(agent):
    """Test task priority."""
    task_id_high = await agent.calcular_impuestos(
        regimen="RESICO",
        tipo_impuesto="ISR",
        datos={"ingresos": 100000},
        priority=10
    )
    
    task_id_low = await agent.calcular_impuestos(
        regimen="RESICO",
        tipo_impuesto="ISR",
        datos={"ingresos": 50000},
        priority=1
    )
    
    assert task_id_high != task_id_low
    
    status_high = await agent.get_task_status(task_id_high)
    status_low = await agent.get_task_status(task_id_low)
    
    assert status_high["priority"] > status_low["priority"]

