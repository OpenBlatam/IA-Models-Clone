"""
Testing Utilities for Music Generation
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


class MusicGeneratorTester:
    """Test utilities for music generators"""
    
    def __init__(self):
        """Initialize tester"""
        logger.info("MusicGeneratorTester initialized")
    
    def test_generation(
        self,
        generator,
        text: str = "A test song",
        duration: int = 5
    ) -> Dict[str, Any]:
        """
        Test music generation
        
        Args:
            generator: Music generator instance
            text: Test text
            duration: Test duration
        
        Returns:
            Test results
        """
        import time
        
        try:
            start = time.time()
            audio = generator.generate_from_text(text, duration=duration)
            elapsed = time.time() - start
            
            # Validate audio
            is_valid = self._validate_audio(audio)
            
            return {
                "status": "passed" if is_valid else "failed",
                "duration_seconds": elapsed,
                "audio_shape": audio.shape if isinstance(audio, np.ndarray) else None,
                "audio_dtype": str(audio.dtype) if isinstance(audio, np.ndarray) else None,
                "has_nan": np.isnan(audio).any() if isinstance(audio, np.ndarray) else None,
                "has_inf": np.isinf(audio).any() if isinstance(audio, np.ndarray) else None,
                "audio_range": (audio.min(), audio.max()) if isinstance(audio, np.ndarray) else None
            }
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e)
            }
    
    def _validate_audio(self, audio: np.ndarray) -> bool:
        """Validate audio array"""
        if not isinstance(audio, np.ndarray):
            return False
        
        if audio.size == 0:
            return False
        
        if np.isnan(audio).any():
            return False
        
        if np.isinf(audio).any():
            return False
        
        # Check if audio is in reasonable range
        if np.abs(audio).max() > 10:
            return False
        
        return True
    
    def test_batch_generation(
        self,
        generator,
        texts: List[str],
        duration: int = 5
    ) -> Dict[str, Any]:
        """
        Test batch generation
        
        Args:
            generator: Music generator instance
            texts: List of test texts
            duration: Test duration
        
        Returns:
            Test results
        """
        import time
        
        try:
            start = time.time()
            if hasattr(generator, 'generate_batch'):
                audios = generator.generate_batch(texts, duration=duration)
            else:
                audios = [generator.generate_from_text(text, duration=duration) for text in texts]
            elapsed = time.time() - start
            
            valid_count = sum(1 for audio in audios if self._validate_audio(audio))
            
            return {
                "status": "passed" if valid_count == len(texts) else "failed",
                "duration_seconds": elapsed,
                "total": len(texts),
                "valid": valid_count,
                "invalid": len(texts) - valid_count,
                "throughput": len(texts) / elapsed if elapsed > 0 else 0
            }
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e)
            }
    
    def benchmark(
        self,
        generator,
        iterations: int = 10,
        duration: int = 5
    ) -> Dict[str, Any]:
        """
        Benchmark generator
        
        Args:
            generator: Music generator instance
            iterations: Number of iterations
            duration: Test duration
        
        Returns:
            Benchmark results
        """
        import time
        
        times = []
        successes = 0
        
        for i in range(iterations):
            try:
                start = time.time()
                audio = generator.generate_from_text(f"Test song {i}", duration=duration)
                elapsed = time.time() - start
                
                if self._validate_audio(audio):
                    times.append(elapsed)
                    successes += 1
            except Exception as e:
                logger.warning(f"Benchmark iteration {i} failed: {e}")
        
        if not times:
            return {
                "status": "failed",
                "error": "No successful iterations"
            }
        
        return {
            "status": "passed",
            "iterations": iterations,
            "successes": successes,
            "failures": iterations - successes,
            "avg_time_seconds": sum(times) / len(times),
            "min_time_seconds": min(times),
            "max_time_seconds": max(times),
            "throughput": successes / sum(times) if sum(times) > 0 else 0
        }


class AudioProcessorTester:
    """Test utilities for audio processors"""
    
    def __init__(self):
        """Initialize tester"""
        logger.info("AudioProcessorTester initialized")
    
    def test_processor(
        self,
        processor,
        audio: np.ndarray
    ) -> Dict[str, Any]:
        """
        Test audio processor
        
        Args:
            processor: Audio processor instance
            audio: Test audio
        
        Returns:
            Test results
        """
        import time
        
        results = {}
        
        # Test normalize
        try:
            start = time.time()
            normalized = processor.normalize(audio)
            results["normalize"] = {
                "status": "passed",
                "time_ms": (time.time() - start) * 1000,
                "output_range": (normalized.min(), normalized.max())
            }
        except Exception as e:
            results["normalize"] = {
                "status": "failed",
                "error": str(e)
            }
        
        # Test compressor
        try:
            start = time.time()
            compressed = processor.apply_compressor(audio)
            results["compressor"] = {
                "status": "passed",
                "time_ms": (time.time() - start) * 1000
            }
        except Exception as e:
            results["compressor"] = {
                "status": "failed",
                "error": str(e)
            }
        
        # Test reverb
        try:
            start = time.time()
            reverb = processor.apply_reverb(audio)
            results["reverb"] = {
                "status": "passed",
                "time_ms": (time.time() - start) * 1000
            }
        except Exception as e:
            results["reverb"] = {
                "status": "failed",
                "error": str(e)
            }
        
        return results

