"""
Pytest configuration and fixtures for Social Video Transcriber AI
"""

import pytest
import asyncio
from typing import Generator

from ..config.settings import Settings


@pytest.fixture(scope="session")
def event_loop() -> Generator:
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def test_settings() -> Settings:
    """Create test settings"""
    return Settings(
        environment="test",
        openrouter_api_key="test_key",
        whisper_model="tiny",
        rate_limit_enabled=False,
    )


@pytest.fixture
def sample_transcription_text() -> str:
    """Sample transcription text for testing"""
    return """
    ¡Hola! Bienvenidos a mi canal.
    
    Hoy vamos a hablar sobre un tema muy importante.
    Este es el problema que muchos enfrentan.
    
    Pero no te preocupes, tengo la solución perfecta.
    
    Primero, necesitas hacer esto.
    Segundo, asegúrate de seguir estos pasos.
    Tercero, no olvides este detalle importante.
    
    Si te gustó este video, dale like y suscríbete.
    ¡Hasta la próxima!
    """


@pytest.fixture
def sample_segments() -> list:
    """Sample transcription segments for testing"""
    from ..core.models import TranscriptionSegment
    
    return [
        TranscriptionSegment(
            id=0,
            start_time=0.0,
            end_time=3.0,
            text="¡Hola! Bienvenidos a mi canal.",
            confidence=0.95,
        ),
        TranscriptionSegment(
            id=1,
            start_time=3.0,
            end_time=8.0,
            text="Hoy vamos a hablar sobre un tema muy importante.",
            confidence=0.92,
        ),
        TranscriptionSegment(
            id=2,
            start_time=8.0,
            end_time=12.0,
            text="Este es el problema que muchos enfrentan.",
            confidence=0.90,
        ),
    ]












