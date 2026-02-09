"""
Tests modulares para AudioProcessor
"""

import pytest
import numpy as np
from tests.helpers.test_helpers import create_mock_audio
from tests.helpers.assertion_helpers import assert_audio_valid, assert_audio_processed


class TestAudioProcessor:
    """Tests para AudioProcessor"""
    
    @pytest.fixture
    def audio_processor(self):
        """Fixture para crear instancia de AudioProcessor"""
        try:
            from core.audio_processor import get_audio_processor
            return get_audio_processor(sample_rate=44100)
        except ImportError:
            pytest.skip("AudioProcessor not available")
    
    @pytest.fixture
    def sample_audio(self):
        """Fixture para audio de prueba"""
        return create_mock_audio(duration=1.0, sample_rate=44100)
    
    @pytest.mark.unit
    def test_normalize_audio(self, audio_processor, sample_audio):
        """Test de normalización de audio"""
        normalized = audio_processor.normalize(sample_audio)
        
        assert_audio_valid(normalized)
        assert_audio_processed(sample_audio, normalized, should_differ=True)
    
    @pytest.mark.unit
    def test_apply_fade(self, audio_processor, sample_audio):
        """Test de aplicación de fade"""
        faded = audio_processor.apply_fade(
            sample_audio,
            fade_in=0.1,
            fade_out=0.1
        )
        
        assert_audio_valid(faded)
        assert_audio_processed(sample_audio, faded, should_differ=True)
    
    @pytest.mark.unit
    def test_trim_silence(self, audio_processor, sample_audio):
        """Test de eliminación de silencio"""
        trimmed = audio_processor.trim_silence(sample_audio)
        
        assert_audio_valid(trimmed)
        # El audio puede ser más corto después de trim
        assert len(trimmed) <= len(sample_audio)
    
    @pytest.mark.unit
    def test_apply_reverb(self, audio_processor, sample_audio):
        """Test de aplicación de reverb"""
        reverbed = audio_processor.apply_reverb(
            sample_audio,
            room_size=0.5,
            damping=0.5
        )
        
        assert_audio_valid(reverbed)
        assert_audio_processed(sample_audio, reverbed, should_differ=True)
    
    @pytest.mark.unit
    def test_apply_eq(self, audio_processor, sample_audio):
        """Test de aplicación de EQ"""
        eq_audio = audio_processor.apply_eq(
            sample_audio,
            low_gain=1.0,
            mid_gain=0.5,
            high_gain=0.0
        )
        
        assert_audio_valid(eq_audio)
        assert_audio_processed(sample_audio, eq_audio, should_differ=True)
    
    @pytest.mark.unit
    def test_change_tempo(self, audio_processor, sample_audio):
        """Test de cambio de tempo"""
        original_length = len(sample_audio)
        
        faster = audio_processor.change_tempo(sample_audio, factor=1.5)
        
        assert_audio_valid(faster)
        # Audio más rápido debería ser más corto
        assert len(faster) < original_length
    
    @pytest.mark.unit
    def test_change_pitch(self, audio_processor, sample_audio):
        """Test de cambio de pitch"""
        pitched = audio_processor.change_pitch(sample_audio, semitones=2.0)
        
        assert_audio_valid(pitched)
        assert_audio_processed(sample_audio, pitched, should_differ=True)
    
    @pytest.mark.unit
    def test_mix_audio(self, audio_processor, sample_audio):
        """Test de mezcla de audio"""
        audio2 = create_mock_audio(duration=1.0, frequency=880.0)
        
        mixed = audio_processor.mix_audio([sample_audio, audio2])
        
        assert_audio_valid(mixed)
        # Audio mezclado debería tener longitud similar
        assert len(mixed) > 0
    
    @pytest.mark.unit
    def test_analyze_audio(self, audio_processor, sample_audio):
        """Test de análisis de audio"""
        analysis = audio_processor.analyze_audio(sample_audio)
        
        assert isinstance(analysis, dict)
        assert "duration" in analysis or "sample_rate" in analysis


class TestAudioProcessorEdgeCases:
    """Tests de casos edge para AudioProcessor"""
    
    @pytest.fixture
    def audio_processor(self):
        try:
            from core.audio_processor import get_audio_processor
            return get_audio_processor(sample_rate=44100)
        except ImportError:
            pytest.skip("AudioProcessor not available")
    
    @pytest.mark.edge_case
    def test_normalize_empty_audio(self, audio_processor):
        """Test con audio vacío"""
        empty_audio = np.array([])
        
        try:
            normalized = audio_processor.normalize(empty_audio)
            assert_audio_valid(normalized)
        except (ValueError, IndexError):
            # Es esperado que falle con audio vacío
            pass
    
    @pytest.mark.edge_case
    def test_apply_fade_extreme_values(self, audio_processor):
        """Test con valores extremos de fade"""
        audio = create_mock_audio()
        
        # Fade muy largo
        faded = audio_processor.apply_fade(audio, fade_in=10.0, fade_out=10.0)
        assert_audio_valid(faded)
    
    @pytest.mark.edge_case
    def test_change_tempo_extreme_factor(self, audio_processor):
        """Test con factor de tempo extremo"""
        audio = create_mock_audio()
        
        # Tempo muy rápido
        fast = audio_processor.change_tempo(audio, factor=10.0)
        assert_audio_valid(fast)
        
        # Tempo muy lento
        slow = audio_processor.change_tempo(audio, factor=0.1)
        assert_audio_valid(slow)
    
    @pytest.mark.edge_case
    def test_mix_many_audio_tracks(self, audio_processor):
        """Test mezclando muchas pistas"""
        tracks = [create_mock_audio(frequency=440.0 * (i + 1)) for i in range(10)]
        
        mixed = audio_processor.mix_audio(tracks)
        
        assert_audio_valid(mixed)

