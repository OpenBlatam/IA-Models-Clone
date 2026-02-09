"""
Pipeline Module

Complete music generation pipeline with all stages integrated.
"""

from typing import Optional, Dict, Any, List
import numpy as np
import logging

logger = logging.getLogger(__name__)


class MusicGenerationPipeline:
    """
    Complete pipeline for music generation from prompt to final mix.
    """
    
    def __init__(
        self,
        generator_type: str = "audiocraft",
        model_name: Optional[str] = None,
        sample_rate: int = 44100,
        enable_analysis: bool = True,
        enable_post_processing: bool = True,
        enable_mixing: bool = False
    ):
        """
        Initialize music generation pipeline.
        
        Args:
            generator_type: Type of generator to use
            model_name: Model name
            sample_rate: Sample rate for audio
            enable_analysis: Enable audio analysis
            enable_post_processing: Enable post-processing
            enable_mixing: Enable mixing stage
        """
        self.generator_type = generator_type
        self.model_name = model_name
        self.sample_rate = sample_rate
        self.enable_analysis = enable_analysis
        self.enable_post_processing = enable_post_processing
        self.enable_mixing = enable_mixing
        
        # Lazy initialization
        self._generator = None
        self._post_processor = None
        self._analyzer = None
        self._mixer = None
        self._stem_separator = None
    
    def _get_generator(self):
        """Get or create generator."""
        if self._generator is None:
            from .generators import create_generator
            self._generator = create_generator(
                generator_type=self.generator_type,
                model_name=self.model_name
            )
        return self._generator
    
    def _get_post_processor(self):
        """Get or create post-processor."""
        if self._post_processor is None:
            from .post_processing import AudioPostProcessor
            self._post_processor = AudioPostProcessor(sample_rate=self.sample_rate)
        return self._post_processor
    
    def _get_analyzer(self):
        """Get or create analyzer."""
        if self._analyzer is None:
            from .analysis import AudioAnalyzer
            self._analyzer = AudioAnalyzer(sample_rate=self.sample_rate)
        return self._analyzer
    
    def _get_mixer(self):
        """Get or create mixer."""
        if self._mixer is None:
            from .mixing import AudioMixer
            self._mixer = AudioMixer(sample_rate=self.sample_rate)
        return self._mixer
    
    def _get_stem_separator(self):
        """Get or create stem separator."""
        if self._stem_separator is None:
            from .mixing import StemSeparator
            self._stem_separator = StemSeparator()
        return self._stem_separator
    
    def generate(
        self,
        prompt: str,
        duration: int = 30,
        post_process: bool = True,
        analyze: bool = True,
        return_analysis: bool = False,
        **generator_kwargs
    ) -> Dict[str, Any]:
        """
        Generate music with optional post-processing and analysis.
        
        Args:
            prompt: Text prompt for generation
            duration: Duration in seconds
            post_process: Apply post-processing
            analyze: Perform audio analysis
            return_analysis: Return analysis results
            **generator_kwargs: Additional generator parameters
            
        Returns:
            Dictionary with audio and optional analysis
        """
        # Generate
        generator = self._get_generator()
        audio = generator.generate(
            prompt=prompt,
            duration=duration,
            **generator_kwargs
        )
        
        result = {
            "audio": audio,
            "prompt": prompt,
            "duration": duration,
            "sample_rate": self.sample_rate
        }
        
        # Post-process
        if post_process and self.enable_post_processing:
            processor = self._get_post_processor()
            audio = processor.process_full_pipeline(audio)
            result["audio"] = audio
            result["post_processed"] = True
        
        # Analyze
        if analyze and self.enable_analysis:
            analyzer = self._get_analyzer()
            analysis = analyzer.full_analysis(audio)
            if return_analysis:
                result["analysis"] = analysis
            result["tempo"] = analysis.get("tempo", {})
            result["key"] = analysis.get("key", {})
        
        return result
    
    def generate_with_stems(
        self,
        prompt: str,
        duration: int = 30,
        mix_stems: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate music and separate into stems.
        
        Args:
            prompt: Text prompt
            duration: Duration in seconds
            mix_stems: Mix stems back together
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with audio, stems, and mixed result
        """
        # Generate base audio
        result = self.generate(prompt, duration, **kwargs)
        audio = result["audio"]
        
        # Separate stems
        separator = self._get_stem_separator()
        stems = separator.separate(audio, sample_rate=self.sample_rate)
        result["stems"] = stems
        
        # Mix stems if requested
        if mix_stems and self.enable_mixing:
            mixer = self._get_mixer()
            mixed = mixer.mix_tracks(
                [stems["drums"], stems["bass"], stems["vocals"], stems["other"]],
                volumes=[0.8, 0.7, 1.0, 0.6]
            )
            result["mixed"] = mixed
        
        return result
    
    def generate_batch(
        self,
        prompts: List[str],
        duration: int = 30,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Generate multiple tracks in batch.
        
        Args:
            prompts: List of prompts
            duration: Duration for each track
            **kwargs: Additional parameters
            
        Returns:
            List of generation results
        """
        generator = self._get_generator()
        audio_list = generator.generate_batch(prompts, duration=duration, **kwargs)
        
        results = []
        for i, (prompt, audio) in enumerate(zip(prompts, audio_list)):
            result = {
                "audio": audio,
                "prompt": prompt,
                "duration": duration,
                "sample_rate": self.sample_rate
            }
            
            # Post-process
            if self.enable_post_processing:
                processor = self._get_post_processor()
                audio = processor.process_full_pipeline(audio)
                result["audio"] = audio
            
            results.append(result)
        
        return results


class VoiceMusicPipeline:
    """
    Pipeline for generating music with vocals.
    """
    
    def __init__(
        self,
        generator_type: str = "audiocraft",
        voice_model: str = "tts_models/multilingual/multi-dataset/xtts_v3",
        sample_rate: int = 44100
    ):
        self.generator_type = generator_type
        self.voice_model = voice_model
        self.sample_rate = sample_rate
        
        self._music_generator = None
        self._voice_synthesizer = None
        self._mixer = None
    
    def _get_music_generator(self):
        """Get music generator."""
        if self._music_generator is None:
            from .generators import create_generator
            self._music_generator = create_generator(self.generator_type)
        return self._music_generator
    
    def _get_voice_synthesizer(self):
        """Get voice synthesizer."""
        if self._voice_synthesizer is None:
            from .voice_synthesis import VoiceSynthesizer
            self._voice_synthesizer = VoiceSynthesizer(model_name=self.voice_model)
        return self._voice_synthesizer
    
    def _get_mixer(self):
        """Get mixer."""
        if self._mixer is None:
            from .mixing import AudioMixer
            self._mixer = AudioMixer(sample_rate=self.sample_rate)
        return self._mixer
    
    def generate_with_vocals(
        self,
        music_prompt: str,
        lyrics: str,
        duration: int = 30,
        voice_reference: Optional[str] = None,
        language: str = "en",
        vocal_volume: float = 1.0,
        music_volume: float = 0.7
    ) -> Dict[str, Any]:
        """
        Generate music with synthesized vocals.
        
        Args:
            music_prompt: Prompt for music generation
            lyrics: Lyrics text
            duration: Duration in seconds
            voice_reference: Optional reference audio for voice cloning
            language: Language code
            vocal_volume: Volume for vocals (0.0 to 1.0)
            music_volume: Volume for music (0.0 to 1.0)
            
        Returns:
            Dictionary with mixed audio and separate tracks
        """
        # Generate music
        music_gen = self._get_music_generator()
        music = music_gen.generate(music_prompt, duration=duration)
        
        # Synthesize vocals
        voice_synth = self._get_voice_synthesizer()
        if voice_reference:
            from .voice_synthesis import VoiceCloner
            cloner = VoiceCloner()
            vocals = cloner.clone_voice(lyrics, voice_reference, language=language)
        else:
            vocals = voice_synth.synthesize(lyrics, language=language)
        
        # Mix music and vocals
        mixer = self._get_mixer()
        mixed = mixer.mix_tracks(
            [music, vocals],
            volumes=[music_volume, vocal_volume],
            panning=[0.0, 0.0]  # Center both
        )
        
        return {
            "audio": mixed,
            "music": music,
            "vocals": vocals,
            "sample_rate": self.sample_rate
        }















