"""
Tests básicos para servicios
"""

import pytest
from ..services.profile_extractor import ProfileExtractor
from ..services.identity_analyzer import IdentityAnalyzer
from ..core.models import SocialProfile, Platform


@pytest.mark.asyncio
async def test_profile_extractor_initialization():
    """Test de inicialización del extractor"""
    extractor = ProfileExtractor()
    assert extractor is not None


@pytest.mark.asyncio
async def test_identity_analyzer_initialization():
    """Test de inicialización del analizador"""
    analyzer = IdentityAnalyzer()
    assert analyzer is not None


@pytest.mark.asyncio
async def test_build_identity_with_empty_profiles():
    """Test de construcción de identidad con perfiles vacíos"""
    analyzer = IdentityAnalyzer()
    identity = await analyzer.build_identity()
    assert identity is not None
    assert identity.profile_id is not None




