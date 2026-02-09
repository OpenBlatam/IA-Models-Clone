"""
Comprehensive Unit Tests for Audio Exporter

Tests cover audio export functionality with diverse test cases
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from core.audio_exporter import AudioExporter


class TestAudioExporter:
    """Test cases for AudioExporter class"""
    
    def test_audio_exporter_init(self):
        """Test initializing audio exporter"""
        exporter = AudioExporter()
        assert "wav" in exporter.supported_formats
        assert "mp3" in exporter.supported_formats
        assert "flac" in exporter.supported_formats
    
    def test_export_to_format_unsupported_format(self):
        """Test exporting with unsupported format raises error"""
        exporter = AudioExporter()
        
        with pytest.raises(ValueError, match="Unsupported format"):
            exporter.export_to_format("input.wav", "output.xyz", format="xyz")
    
    def test_export_to_format_file_not_found(self):
        """Test exporting non-existent file raises error"""
        exporter = AudioExporter()
        
        with pytest.raises(FileNotFoundError):
            exporter.export_to_format("nonexistent.wav", "output.mp3")
    
    @patch('core.audio_exporter.sf.read')
    @patch('core.audio_exporter.sf.write')
    def test_export_to_wav_basic(self, mock_write, mock_read):
        """Test exporting to WAV format"""
        exporter = AudioExporter()
        
        # Mock audio data
        mock_audio = np.array([0.1, 0.2, 0.3])
        mock_read.return_value = (mock_audio, 44100)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.wav"
            output_path = Path(tmpdir) / "output.wav"
            
            # Create input file
            input_path.touch()
            
            result = exporter._export_to_wav(input_path, output_path, None)
            
            assert result == str(output_path)
            mock_read.assert_called_once()
            mock_write.assert_called_once()
    
    @patch('core.audio_exporter.sf.read')
    @patch('core.audio_exporter.sf.write')
    @patch('core.audio_exporter.signal.resample')
    def test_export_to_wav_with_resample(self, mock_resample, mock_write, mock_read):
        """Test exporting to WAV with resampling"""
        exporter = AudioExporter()
        
        mock_audio = np.array([0.1, 0.2, 0.3])
        mock_read.return_value = (mock_audio, 44100)
        mock_resample.return_value = np.array([0.1, 0.2])
        
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.wav"
            output_path = Path(tmpdir) / "output.wav"
            input_path.touch()
            
            result = exporter._export_to_wav(input_path, output_path, 22050)
            
            assert result == str(output_path)
            mock_resample.assert_called_once()
    
    @patch('core.audio_exporter.sf.read')
    def test_export_to_wav_error_handling(self, mock_read):
        """Test error handling in WAV export"""
        exporter = AudioExporter()
        mock_read.side_effect = Exception("Read error")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.wav"
            output_path = Path(tmpdir) / "output.wav"
            input_path.touch()
            
            with pytest.raises(Exception):
                exporter._export_to_wav(input_path, output_path, None)
    
    @patch('core.audio_exporter.subprocess.run')
    def test_export_with_ffmpeg_mp3(self, mock_subprocess):
        """Test exporting to MP3 with ffmpeg"""
        exporter = AudioExporter()
        mock_subprocess.return_value = Mock(returncode=0)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.wav"
            output_path = Path(tmpdir) / "output.mp3"
            input_path.touch()
            
            result = exporter._export_with_ffmpeg(
                input_path, output_path, "mp3", "192k", None
            )
            
            assert result == str(output_path)
            mock_subprocess.assert_called_once()
            call_args = mock_subprocess.call_args[0][0]
            assert "ffmpeg" in call_args
            assert "mp3" in str(call_args)
    
    @patch('core.audio_exporter.subprocess.run')
    def test_export_with_ffmpeg_custom_bitrate(self, mock_subprocess):
        """Test exporting with custom bitrate"""
        exporter = AudioExporter()
        mock_subprocess.return_value = Mock(returncode=0)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.wav"
            output_path = Path(tmpdir) / "output.mp3"
            input_path.touch()
            
            exporter._export_with_ffmpeg(
                input_path, output_path, "mp3", "320k", None
            )
            
            call_args = mock_subprocess.call_args[0][0]
            assert "320k" in str(call_args)
    
    @patch('core.audio_exporter.subprocess.run')
    def test_export_with_ffmpeg_custom_sample_rate(self, mock_subprocess):
        """Test exporting with custom sample rate"""
        exporter = AudioExporter()
        mock_subprocess.return_value = Mock(returncode=0)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.wav"
            output_path = Path(tmpdir) / "output.mp3"
            input_path.touch()
            
            exporter._export_with_ffmpeg(
                input_path, output_path, "mp3", "192k", 48000
            )
            
            call_args = mock_subprocess.call_args[0][0]
            assert "48000" in str(call_args)
    
    @patch('core.audio_exporter.subprocess.run')
    def test_export_with_ffmpeg_error(self, mock_subprocess):
        """Test ffmpeg error handling"""
        exporter = AudioExporter()
        mock_subprocess.return_value = Mock(returncode=1)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.wav"
            output_path = Path(tmpdir) / "output.mp3"
            input_path.touch()
            
            with pytest.raises(RuntimeError, match="ffmpeg"):
                exporter._export_with_ffmpeg(
                    input_path, output_path, "mp3", "192k", None
                )
    
    @patch('core.audio_exporter.AudioExporter._export_to_wav')
    def test_export_to_format_wav(self, mock_export_wav):
        """Test export_to_format calls _export_to_wav for WAV"""
        exporter = AudioExporter()
        mock_export_wav.return_value = "output.wav"
        
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.wav"
            output_path = Path(tmpdir) / "output.wav"
            input_path.touch()
            
            result = exporter.export_to_format(
                str(input_path), str(output_path), format="wav"
            )
            
            assert result == "output.wav"
            mock_export_wav.assert_called_once()
    
    @patch('core.audio_exporter.AudioExporter._export_with_ffmpeg')
    def test_export_to_format_mp3(self, mock_export_ffmpeg):
        """Test export_to_format calls _export_with_ffmpeg for MP3"""
        exporter = AudioExporter()
        mock_export_ffmpeg.return_value = "output.mp3"
        
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.wav"
            output_path = Path(tmpdir) / "output.mp3"
            input_path.touch()
            
            result = exporter.export_to_format(
                str(input_path), str(output_path), format="mp3"
            )
            
            assert result == "output.mp3"
            mock_export_ffmpeg.assert_called_once()
    
    def test_export_to_format_all_supported_formats(self):
        """Test all supported formats are accepted"""
        exporter = AudioExporter()
        
        for fmt in exporter.supported_formats:
            # Should not raise ValueError for format
            assert fmt in exporter.supported_formats















