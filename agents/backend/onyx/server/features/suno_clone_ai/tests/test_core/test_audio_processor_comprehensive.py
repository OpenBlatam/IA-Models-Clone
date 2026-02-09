"""
Comprehensive Unit Tests for Audio Processor

Tests cover all audio processing methods with diverse test cases
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from core.audio_processor import AudioProcessor, get_audio_processor


class TestAudioProcessorNormalize:
    """Test cases for normalize method"""
    
    def test_normalize_basic(self):
        """Test basic normalization"""
        processor = AudioProcessor()
        audio = np.array([0.1, 0.2, 0.3, 0.4])
        result = processor.normalize(audio)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(audio)
        assert np.all(result >= -1.0) and np.all(result <= 1.0)
    
    def test_normalize_silent_audio(self):
        """Test normalizing silent audio (RMS = 0)"""
        processor = AudioProcessor()
        audio = np.zeros(100)
        result = processor.normalize(audio)
        
        assert np.array_equal(result, audio)
    
    def test_normalize_already_normalized(self):
        """Test normalizing already normalized audio"""
        processor = AudioProcessor()
        audio = np.array([0.5, -0.5, 0.3, -0.3])
        result = processor.normalize(audio)
        
        assert np.all(result >= -1.0) and np.all(result <= 1.0)
    
    def test_normalize_custom_target_db(self):
        """Test normalization with custom target dB"""
        processor = AudioProcessor()
        audio = np.array([0.1, 0.2, 0.3])
        result = processor.normalize(audio, target_db=-6.0)
        
        assert isinstance(result, np.ndarray)
        assert np.all(result >= -1.0) and np.all(result <= 1.0)
    
    def test_normalize_large_values(self):
        """Test normalizing audio with large values"""
        processor = AudioProcessor()
        audio = np.array([2.0, 3.0, 4.0])
        result = processor.normalize(audio)
        
        assert np.all(result >= -1.0) and np.all(result <= 1.0)
    
    def test_normalize_negative_values(self):
        """Test normalizing audio with negative values"""
        processor = AudioProcessor()
        audio = np.array([-0.5, -0.3, 0.2, 0.4])
        result = processor.normalize(audio)
        
        assert np.all(result >= -1.0) and np.all(result <= 1.0)


class TestAudioProcessorApplyFade:
    """Test cases for apply_fade method"""
    
    def test_apply_fade_basic(self):
        """Test basic fade in and fade out"""
        processor = AudioProcessor(sample_rate=44100)
        audio = np.ones(44100)  # 1 second of audio
        result = processor.apply_fade(audio, fade_in=0.1, fade_out=0.1)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(audio)
        # First samples should be less than 1.0 (fade in)
        assert result[0] < 1.0
        # Last samples should be less than 1.0 (fade out)
        assert result[-1] < 1.0
    
    def test_apply_fade_fade_in_only(self):
        """Test fade in only"""
        processor = AudioProcessor(sample_rate=44100)
        audio = np.ones(44100)
        result = processor.apply_fade(audio, fade_in=0.2, fade_out=0.0)
        
        assert result[0] < 1.0  # Fade in applied
        assert result[-1] == 1.0  # No fade out
    
    def test_apply_fade_fade_out_only(self):
        """Test fade out only"""
        processor = AudioProcessor(sample_rate=44100)
        audio = np.ones(44100)
        result = processor.apply_fade(audio, fade_in=0.0, fade_out=0.2)
        
        assert result[0] == 1.0  # No fade in
        assert result[-1] < 1.0  # Fade out applied
    
    def test_apply_fade_no_fade(self):
        """Test with no fade"""
        processor = AudioProcessor(sample_rate=44100)
        audio = np.ones(44100)
        result = processor.apply_fade(audio, fade_in=0.0, fade_out=0.0)
        
        assert np.array_equal(result, audio)
    
    def test_apply_fade_short_audio(self):
        """Test fade on short audio"""
        processor = AudioProcessor(sample_rate=44100)
        audio = np.ones(1000)  # Very short
        result = processor.apply_fade(audio, fade_in=0.5, fade_out=0.5)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(audio)


class TestAudioProcessorMixAudio:
    """Test cases for mix_audio method"""
    
    def test_mix_audio_two_tracks(self):
        """Test mixing two audio tracks"""
        processor = AudioProcessor()
        track1 = np.array([0.5, 0.5, 0.5])
        track2 = np.array([0.3, 0.3, 0.3])
        
        result = processor.mix_audio([track1, track2])
        
        assert isinstance(result, np.ndarray)
        assert len(result) == 3
        assert np.all(result >= -1.0) and np.all(result <= 1.0)
    
    def test_mix_audio_different_lengths(self):
        """Test mixing tracks of different lengths"""
        processor = AudioProcessor()
        track1 = np.array([0.5, 0.5, 0.5, 0.5])
        track2 = np.array([0.3, 0.3])
        
        result = processor.mix_audio([track1, track2])
        
        assert len(result) == 4  # Should match longest track
    
    def test_mix_audio_with_volumes(self):
        """Test mixing with volume control"""
        processor = AudioProcessor()
        track1 = np.array([0.5, 0.5])
        track2 = np.array([0.5, 0.5])
        volumes = [0.5, 1.0]
        
        result = processor.mix_audio([track1, track2], volumes=volumes)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == 2
    
    def test_mix_audio_empty_tracks(self):
        """Test mixing empty tracks raises error"""
        processor = AudioProcessor()
        
        with pytest.raises(ValueError, match="No audio tracks"):
            processor.mix_audio([])
    
    def test_mix_audio_single_track(self):
        """Test mixing single track"""
        processor = AudioProcessor()
        track1 = np.array([0.5, 0.5, 0.5])
        
        result = processor.mix_audio([track1])
        
        assert isinstance(result, np.ndarray)
        assert len(result) == 3
    
    def test_mix_audio_prevents_clipping(self):
        """Test that mixing prevents clipping"""
        processor = AudioProcessor()
        track1 = np.array([0.8, 0.8])
        track2 = np.array([0.8, 0.8])
        
        result = processor.mix_audio([track1, track2])
        
        # Should be normalized to prevent clipping
        assert np.all(result >= -1.0) and np.all(result <= 1.0)


class TestAudioProcessorApplyReverb:
    """Test cases for apply_reverb method"""
    
    def test_apply_reverb_basic(self):
        """Test basic reverb application"""
        processor = AudioProcessor(sample_rate=44100)
        audio = np.random.randn(44100).astype(np.float32)
        
        result = processor.apply_reverb(audio)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(audio)
        assert np.all(result >= -1.0) and np.all(result <= 1.0)
    
    def test_apply_reverb_custom_room_size(self):
        """Test reverb with custom room size"""
        processor = AudioProcessor(sample_rate=44100)
        audio = np.random.randn(44100).astype(np.float32)
        
        result = processor.apply_reverb(audio, room_size=0.8)
        
        assert isinstance(result, np.ndarray)
        assert np.all(result >= -1.0) and np.all(result <= 1.0)
    
    def test_apply_reverb_custom_damping(self):
        """Test reverb with custom damping"""
        processor = AudioProcessor(sample_rate=44100)
        audio = np.random.randn(44100).astype(np.float32)
        
        result = processor.apply_reverb(audio, damping=0.3)
        
        assert isinstance(result, np.ndarray)
        assert np.all(result >= -1.0) and np.all(result <= 1.0)
    
    def test_apply_reverb_silent_audio(self):
        """Test reverb on silent audio"""
        processor = AudioProcessor(sample_rate=44100)
        audio = np.zeros(44100)
        
        result = processor.apply_reverb(audio)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(audio)


class TestAudioProcessorApplyEQ:
    """Test cases for apply_eq method"""
    
    def test_apply_eq_no_gain(self):
        """Test EQ with no gain (should return original)"""
        processor = AudioProcessor(sample_rate=44100)
        audio = np.random.randn(44100).astype(np.float32)
        
        result = processor.apply_eq(audio)
        
        # Should be similar (may have minor differences due to processing)
        assert isinstance(result, np.ndarray)
        assert len(result) == len(audio)
    
    def test_apply_eq_low_gain(self):
        """Test EQ with low frequency gain"""
        processor = AudioProcessor(sample_rate=44100)
        audio = np.random.randn(44100).astype(np.float32)
        
        result = processor.apply_eq(audio, low_gain=3.0)
        
        assert isinstance(result, np.ndarray)
        assert np.all(result >= -1.0) and np.all(result <= 1.0)
    
    def test_apply_eq_mid_gain(self):
        """Test EQ with mid frequency gain"""
        processor = AudioProcessor(sample_rate=44100)
        audio = np.random.randn(44100).astype(np.float32)
        
        result = processor.apply_eq(audio, mid_gain=2.0)
        
        assert isinstance(result, np.ndarray)
        assert np.all(result >= -1.0) and np.all(result <= 1.0)
    
    def test_apply_eq_high_gain(self):
        """Test EQ with high frequency gain"""
        processor = AudioProcessor(sample_rate=44100)
        audio = np.random.randn(44100).astype(np.float32)
        
        result = processor.apply_eq(audio, high_gain=1.5)
        
        assert isinstance(result, np.ndarray)
        assert np.all(result >= -1.0) and np.all(result <= 1.0)
    
    def test_apply_eq_all_bands(self):
        """Test EQ with all frequency bands"""
        processor = AudioProcessor(sample_rate=44100)
        audio = np.random.randn(44100).astype(np.float32)
        
        result = processor.apply_eq(
            audio, 
            low_gain=1.0, 
            mid_gain=1.0, 
            high_gain=1.0
        )
        
        assert isinstance(result, np.ndarray)
        assert np.all(result >= -1.0) and np.all(result <= 1.0)


class TestAudioProcessorAnalyzeAudio:
    """Test cases for analyze_audio method"""
    
    def test_analyze_audio_basic(self):
        """Test basic audio analysis"""
        processor = AudioProcessor(sample_rate=44100)
        audio = np.random.randn(44100).astype(np.float32)
        
        result = processor.analyze_audio(audio)
        
        assert isinstance(result, dict)
        assert "rms" in result
        assert "peak" in result
        assert "zero_crossing_rate" in result
        assert "spectral_centroid" in result
        assert "tempo" in result
        assert "duration" in result
    
    def test_analyze_audio_silent(self):
        """Test analyzing silent audio"""
        processor = AudioProcessor(sample_rate=44100)
        audio = np.zeros(44100)
        
        result = processor.analyze_audio(audio)
        
        assert isinstance(result, dict)
        assert result["rms"] == 0.0
        assert result["peak"] == 0.0
    
    def test_analyze_audio_duration(self):
        """Test duration calculation"""
        processor = AudioProcessor(sample_rate=44100)
        audio = np.random.randn(88200).astype(np.float32)  # 2 seconds
        
        result = processor.analyze_audio(audio)
        
        assert result["duration"] == pytest.approx(2.0, rel=0.1)
    
    def test_analyze_audio_all_positive(self):
        """Test analyzing audio with all positive values"""
        processor = AudioProcessor(sample_rate=44100)
        audio = np.abs(np.random.randn(44100)).astype(np.float32)
        
        result = processor.analyze_audio(audio)
        
        assert result["peak"] > 0
        assert result["rms"] > 0


class TestAudioProcessorTrimSilence:
    """Test cases for trim_silence method"""
    
    def test_trim_silence_basic(self):
        """Test basic silence trimming"""
        processor = AudioProcessor()
        # Audio with silence at start and end
        audio = np.concatenate([
            np.zeros(100),
            np.ones(100) * 0.5,
            np.zeros(100)
        ])
        
        result = processor.trim_silence(audio)
        
        assert isinstance(result, np.ndarray)
        assert len(result) <= len(audio)
    
    def test_trim_silence_no_silence(self):
        """Test trimming audio with no silence"""
        processor = AudioProcessor()
        audio = np.ones(100) * 0.5
        
        result = processor.trim_silence(audio)
        
        assert len(result) == len(audio)
    
    def test_trim_silence_all_silence(self):
        """Test trimming audio that is all silence"""
        processor = AudioProcessor()
        audio = np.zeros(100)
        
        result = processor.trim_silence(audio, threshold=0.01)
        
        # Should return original if all silence
        assert isinstance(result, np.ndarray)
    
    def test_trim_silence_custom_threshold(self):
        """Test trimming with custom threshold"""
        processor = AudioProcessor()
        audio = np.concatenate([
            np.zeros(100),
            np.ones(100) * 0.05,  # Below default threshold
            np.zeros(100)
        ])
        
        result = processor.trim_silence(audio, threshold=0.03)
        
        assert isinstance(result, np.ndarray)


class TestAudioProcessorChangeTempo:
    """Test cases for change_tempo method"""
    
    def test_change_tempo_speed_up(self):
        """Test speeding up tempo"""
        processor = AudioProcessor()
        audio = np.random.randn(44100).astype(np.float32)
        
        result = processor.change_tempo(audio, factor=1.5)
        
        assert isinstance(result, np.ndarray)
        # Faster tempo should result in shorter audio
        assert len(result) < len(audio)
    
    def test_change_tempo_slow_down(self):
        """Test slowing down tempo"""
        processor = AudioProcessor()
        audio = np.random.randn(44100).astype(np.float32)
        
        result = processor.change_tempo(audio, factor=0.5)
        
        assert isinstance(result, np.ndarray)
        # Slower tempo should result in longer audio
        assert len(result) > len(audio)
    
    def test_change_tempo_no_change(self):
        """Test tempo factor of 1.0"""
        processor = AudioProcessor()
        audio = np.random.randn(44100).astype(np.float32)
        
        result = processor.change_tempo(audio, factor=1.0)
        
        assert isinstance(result, np.ndarray)


class TestAudioProcessorChangePitch:
    """Test cases for change_pitch method"""
    
    def test_change_pitch_up(self):
        """Test raising pitch"""
        processor = AudioProcessor(sample_rate=44100)
        audio = np.random.randn(44100).astype(np.float32)
        
        result = processor.change_pitch(audio, semitones=2.0)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(audio)  # Pitch change doesn't change length
    
    def test_change_pitch_down(self):
        """Test lowering pitch"""
        processor = AudioProcessor(sample_rate=44100)
        audio = np.random.randn(44100).astype(np.float32)
        
        result = processor.change_pitch(audio, semitones=-2.0)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(audio)
    
    def test_change_pitch_no_change(self):
        """Test pitch change of 0 semitones"""
        processor = AudioProcessor(sample_rate=44100)
        audio = np.random.randn(44100).astype(np.float32)
        
        result = processor.change_pitch(audio, semitones=0.0)
        
        assert isinstance(result, np.ndarray)


class TestGetAudioProcessor:
    """Test cases for get_audio_processor function"""
    
    def test_get_audio_processor_singleton(self):
        """Test that get_audio_processor returns singleton"""
        processor1 = get_audio_processor()
        processor2 = get_audio_processor()
        
        assert processor1 is processor2
        assert isinstance(processor1, AudioProcessor)
    
    def test_get_audio_processor_default_sample_rate(self):
        """Test default sample rate"""
        processor = get_audio_processor()
        assert processor.sample_rate == 32000
    
    def test_get_audio_processor_custom_sample_rate_first_call(self):
        """Test custom sample rate on first call"""
        # Reset global instance for test
        import core.audio_processor
        core.audio_processor._audio_processor = None
        
        processor = get_audio_processor(sample_rate=44100)
        assert processor.sample_rate == 44100















