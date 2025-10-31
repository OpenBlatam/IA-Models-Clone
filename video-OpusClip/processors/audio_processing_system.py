"""
Advanced Audio Processing System

Handles background music selection, audio enhancement, sound effects,
and music copyright compliance for professional video content.
"""

from __future__ import annotations
from typing import List, Dict, Optional, Any, Tuple, Union
import asyncio
import numpy as np
import librosa
import soundfile as sf
from pydub import AudioSegment
from pydub.effects import normalize, compress_dynamic_range
import structlog
import time
import json
import requests
import aiohttp
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import hashlib
import re

from ..models.video_models import VideoClipRequest, VideoClipResponse
from ..error_handling import ErrorHandler, ProcessingError, ValidationError

logger = structlog.get_logger("audio_processing_system")
error_handler = ErrorHandler()

class AudioGenre(Enum):
    """Audio genres for background music."""
    ELECTRONIC = "electronic"
    POP = "pop"
    ROCK = "rock"
    HIP_HOP = "hip_hop"
    CLASSICAL = "classical"
    JAZZ = "jazz"
    AMBIENT = "ambient"
    UPBEAT = "upbeat"
    CALM = "calm"
    EPIC = "epic"

class AudioMood(Enum):
    """Audio moods for content matching."""
    HAPPY = "happy"
    SAD = "sad"
    EXCITING = "exciting"
    RELAXING = "relaxing"
    DRAMATIC = "dramatic"
    MYSTERIOUS = "mysterious"
    INSPIRATIONAL = "inspirational"
    ENERGETIC = "energetic"
    ROMANTIC = "romantic"
    NOSTALGIC = "nostalgic"

class AudioQuality(Enum):
    """Audio quality levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    PROFESSIONAL = "professional"

@dataclass
class AudioTrack:
    """Represents an audio track."""
    track_id: str
    title: str
    artist: str
    duration: float
    genre: AudioGenre
    mood: AudioMood
    tempo: float  # BPM
    key: str  # Musical key
    energy_level: float  # 0-1
    copyright_status: str
    file_path: Optional[str] = None
    url: Optional[str] = None
    metadata: Dict[str, Any] = None

@dataclass
class AudioEnhancement:
    """Audio enhancement parameters."""
    normalize: bool = True
    compress: bool = True
    eq_boost: bool = True
    noise_reduction: bool = True
    stereo_width: float = 1.0
    reverb: float = 0.0
    delay: float = 0.0
    chorus: float = 0.0

@dataclass
class SoundEffect:
    """Represents a sound effect."""
    effect_id: str
    name: str
    category: str
    duration: float
    file_path: str
    volume: float = 0.8
    timing: Optional[float] = None  # When to play in video

class MusicLibrary:
    """Manages music library and selection."""
    
    def __init__(self):
        self.tracks = []
        self.genres = {}
        self.moods = {}
        self._load_music_library()
    
    def _load_music_library(self):
        """Load music library from database or files."""
        try:
            # Placeholder - would load from actual music library
            self.tracks = self._generate_sample_tracks()
            self._categorize_tracks()
            
            logger.info(f"Loaded {len(self.tracks)} tracks in music library")
            
        except Exception as e:
            logger.error(f"Failed to load music library: {e}")
            self.tracks = []
    
    def _generate_sample_tracks(self) -> List[AudioTrack]:
        """Generate sample tracks for demo."""
        tracks = []
        
        # Sample tracks with different characteristics
        sample_data = [
            {"title": "Upbeat Electronic", "genre": AudioGenre.ELECTRONIC, "mood": AudioMood.ENERGETIC, "tempo": 128, "energy": 0.9},
            {"title": "Calm Ambient", "genre": AudioGenre.AMBIENT, "mood": AudioMood.RELAXING, "tempo": 60, "energy": 0.3},
            {"title": "Epic Orchestral", "genre": AudioGenre.CLASSICAL, "mood": AudioMood.DRAMATIC, "tempo": 100, "energy": 0.8},
            {"title": "Happy Pop", "genre": AudioGenre.POP, "mood": AudioMood.HAPPY, "tempo": 120, "energy": 0.7},
            {"title": "Mysterious Jazz", "genre": AudioGenre.JAZZ, "mood": AudioMood.MYSTERIOUS, "tempo": 90, "energy": 0.5},
            {"title": "Inspiring Rock", "genre": AudioGenre.ROCK, "mood": AudioMood.INSPIRATIONAL, "tempo": 140, "energy": 0.8},
            {"title": "Romantic Piano", "genre": AudioGenre.CLASSICAL, "mood": AudioMood.ROMANTIC, "tempo": 80, "energy": 0.4},
            {"title": "Energetic Hip Hop", "genre": AudioGenre.HIP_HOP, "mood": AudioMood.EXCITING, "tempo": 95, "energy": 0.9},
        ]
        
        for i, data in enumerate(sample_data):
            track = AudioTrack(
                track_id=f"track_{i}",
                title=data["title"],
                artist="Sample Artist",
                duration=30.0,  # 30 seconds
                genre=data["genre"],
                mood=data["mood"],
                tempo=data["tempo"],
                key="C Major",
                energy_level=data["energy"],
                copyright_status="royalty_free",
                metadata={"bpm": data["tempo"], "key": "C Major"}
            )
            tracks.append(track)
        
        return tracks
    
    def _categorize_tracks(self):
        """Categorize tracks by genre and mood."""
        try:
            # Group by genre
            for track in self.tracks:
                genre = track.genre.value
                if genre not in self.genres:
                    self.genres[genre] = []
                self.genres[genre].append(track)
            
            # Group by mood
            for track in self.tracks:
                mood = track.mood.value
                if mood not in self.moods:
                    self.moods[mood] = []
                self.moods[mood].append(track)
                
        except Exception as e:
            logger.error(f"Track categorization failed: {e}")
    
    async def find_matching_tracks(self, 
                                 content_analysis: Dict[str, Any],
                                 duration: float,
                                 max_tracks: int = 5) -> List[AudioTrack]:
        """Find tracks that match content characteristics."""
        try:
            # Analyze content to determine music requirements
            mood = self._analyze_content_mood(content_analysis)
            genre_preference = self._analyze_genre_preference(content_analysis)
            energy_level = self._analyze_energy_requirement(content_analysis)
            
            # Filter tracks based on requirements
            matching_tracks = []
            
            for track in self.tracks:
                score = 0.0
                
                # Mood matching
                if track.mood == mood:
                    score += 0.4
                elif track.mood.value in self._get_related_moods(mood):
                    score += 0.2
                
                # Genre preference
                if track.genre.value == genre_preference:
                    score += 0.3
                elif track.genre.value in self._get_related_genres(genre_preference):
                    score += 0.15
                
                # Energy level matching
                energy_diff = abs(track.energy_level - energy_level)
                score += (1 - energy_diff) * 0.2
                
                # Duration compatibility
                if abs(track.duration - duration) < 5.0:  # Within 5 seconds
                    score += 0.1
                
                if score > 0.5:  # Minimum threshold
                    matching_tracks.append((track, score))
            
            # Sort by score and return top matches
            matching_tracks.sort(key=lambda x: x[1], reverse=True)
            
            return [track for track, score in matching_tracks[:max_tracks]]
            
        except Exception as e:
            logger.error(f"Track matching failed: {e}")
            return []
    
    def _analyze_content_mood(self, content_analysis: Dict[str, Any]) -> AudioMood:
        """Analyze content to determine required mood."""
        try:
            # Get sentiment and engagement data
            sentiment = content_analysis.get("sentiment", "neutral")
            engagement_level = content_analysis.get("engagement_level", 0.5)
            content_type = content_analysis.get("content_type", "general")
            
            # Map to audio moods
            if sentiment == "positive" and engagement_level > 0.7:
                return AudioMood.HAPPY
            elif sentiment == "negative" and engagement_level > 0.7:
                return AudioMood.DRAMATIC
            elif engagement_level > 0.8:
                return AudioMood.ENERGETIC
            elif content_type in ["tutorial", "educational"]:
                return AudioMood.INSPIRATIONAL
            elif content_type in ["lifestyle", "beauty"]:
                return AudioMood.ROMANTIC
            else:
                return AudioMood.UPBEAT
                
        except Exception as e:
            logger.error(f"Content mood analysis failed: {e}")
            return AudioMood.UPBEAT
    
    def _analyze_genre_preference(self, content_analysis: Dict[str, Any]) -> str:
        """Analyze content to determine genre preference."""
        try:
            content_type = content_analysis.get("content_type", "general")
            keywords = content_analysis.get("keywords", [])
            
            # Map content type to genre
            genre_mapping = {
                "tech": "electronic",
                "gaming": "electronic",
                "lifestyle": "pop",
                "fitness": "upbeat",
                "cooking": "jazz",
                "travel": "ambient",
                "education": "classical",
                "comedy": "pop",
                "dance": "electronic"
            }
            
            # Check content type
            if content_type in genre_mapping:
                return genre_mapping[content_type]
            
            # Check keywords
            for keyword in keywords:
                if keyword.lower() in ["tech", "ai", "digital"]:
                    return "electronic"
                elif keyword.lower() in ["music", "dance", "party"]:
                    return "pop"
                elif keyword.lower() in ["nature", "peaceful", "calm"]:
                    return "ambient"
            
            return "pop"  # Default
            
        except Exception as e:
            logger.error(f"Genre preference analysis failed: {e}")
            return "pop"
    
    def _analyze_energy_requirement(self, content_analysis: Dict[str, Any]) -> float:
        """Analyze content to determine energy requirement."""
        try:
            engagement_level = content_analysis.get("engagement_level", 0.5)
            content_type = content_analysis.get("content_type", "general")
            
            # Base energy on engagement level
            energy = engagement_level
            
            # Adjust based on content type
            energy_adjustments = {
                "fitness": 0.2,
                "dance": 0.3,
                "gaming": 0.2,
                "comedy": 0.1,
                "tutorial": -0.1,
                "lifestyle": 0.0,
                "education": -0.2
            }
            
            if content_type in energy_adjustments:
                energy += energy_adjustments[content_type]
            
            return max(0.0, min(1.0, energy))
            
        except Exception as e:
            logger.error(f"Energy requirement analysis failed: {e}")
            return 0.5
    
    def _get_related_moods(self, mood: AudioMood) -> List[str]:
        """Get related moods for better matching."""
        mood_relations = {
            AudioMood.HAPPY: ["energetic", "upbeat"],
            AudioMood.SAD: ["dramatic", "mysterious"],
            AudioMood.EXCITING: ["energetic", "happy"],
            AudioMood.RELAXING: ["calm", "ambient"],
            AudioMood.DRAMATIC: ["epic", "mysterious"],
            AudioMood.MYSTERIOUS: ["dramatic", "ambient"],
            AudioMood.INSPIRATIONAL: ["epic", "happy"],
            AudioMood.ENERGETIC: ["exciting", "happy"],
            AudioMood.ROMANTIC: ["calm", "nostalgic"],
            AudioMood.NOSTALGIC: ["romantic", "calm"]
        }
        
        return mood_relations.get(mood, [])
    
    def _get_related_genres(self, genre: str) -> List[str]:
        """Get related genres for better matching."""
        genre_relations = {
            "electronic": ["pop", "ambient"],
            "pop": ["electronic", "rock"],
            "rock": ["pop", "hip_hop"],
            "hip_hop": ["electronic", "pop"],
            "classical": ["jazz", "ambient"],
            "jazz": ["classical", "blues"],
            "ambient": ["electronic", "classical"],
            "upbeat": ["pop", "electronic"],
            "calm": ["ambient", "classical"],
            "epic": ["classical", "rock"]
        }
        
        return genre_relations.get(genre, [])

class AudioEnhancer:
    """Enhances audio quality and applies effects."""
    
    def __init__(self):
        self.enhancement_presets = self._load_enhancement_presets()
    
    def _load_enhancement_presets(self) -> Dict[str, AudioEnhancement]:
        """Load audio enhancement presets."""
        return {
            "minimal": AudioEnhancement(
                normalize=True,
                compress=False,
                eq_boost=False,
                noise_reduction=True
            ),
            "balanced": AudioEnhancement(
                normalize=True,
                compress=True,
                eq_boost=True,
                noise_reduction=True,
                stereo_width=1.1
            ),
            "enhanced": AudioEnhancement(
                normalize=True,
                compress=True,
                eq_boost=True,
                noise_reduction=True,
                stereo_width=1.2,
                reverb=0.1,
                delay=0.05
            ),
            "professional": AudioEnhancement(
                normalize=True,
                compress=True,
                eq_boost=True,
                noise_reduction=True,
                stereo_width=1.3,
                reverb=0.15,
                delay=0.08,
                chorus=0.05
            )
        }
    
    async def enhance_audio(self, 
                          audio_path: str, 
                          output_path: str,
                          enhancement: AudioEnhancement = None,
                          quality: AudioQuality = AudioQuality.HIGH) -> Dict[str, Any]:
        """Enhance audio file with specified parameters."""
        try:
            logger.info(f"Enhancing audio: {audio_path}")
            
            if enhancement is None:
                enhancement = self.enhancement_presets["balanced"]
            
            # Load audio file
            audio = AudioSegment.from_file(audio_path)
            
            # Apply enhancements
            enhanced_audio = audio
            
            # Normalize
            if enhancement.normalize:
                enhanced_audio = normalize(enhanced_audio)
            
            # Compress dynamic range
            if enhancement.compress:
                enhanced_audio = compress_dynamic_range(enhanced_audio)
            
            # EQ boost (simplified)
            if enhancement.eq_boost:
                enhanced_audio = self._apply_eq_boost(enhanced_audio)
            
            # Noise reduction (simplified)
            if enhancement.noise_reduction:
                enhanced_audio = self._apply_noise_reduction(enhanced_audio)
            
            # Stereo width adjustment
            if enhancement.stereo_width != 1.0:
                enhanced_audio = self._adjust_stereo_width(enhanced_audio, enhancement.stereo_width)
            
            # Reverb
            if enhancement.reverb > 0:
                enhanced_audio = self._apply_reverb(enhanced_audio, enhancement.reverb)
            
            # Delay
            if enhancement.delay > 0:
                enhanced_audio = self._apply_delay(enhanced_audio, enhancement.delay)
            
            # Chorus
            if enhancement.chorus > 0:
                enhanced_audio = self._apply_chorus(enhanced_audio, enhancement.chorus)
            
            # Export enhanced audio
            enhanced_audio.export(output_path, format="mp3", bitrate=self._get_bitrate(quality))
            
            return {
                "input_path": audio_path,
                "output_path": output_path,
                "enhancement_applied": enhancement.__dict__,
                "quality": quality.value,
                "duration": len(enhanced_audio) / 1000.0
            }
            
        except Exception as e:
            logger.error(f"Audio enhancement failed: {e}")
            raise ProcessingError(f"Audio enhancement failed: {e}")
    
    def _apply_eq_boost(self, audio: AudioSegment) -> AudioSegment:
        """Apply EQ boost to audio."""
        try:
            # Simple EQ boost - would use more sophisticated EQ
            # For now, just increase volume slightly
            return audio + 2  # 2dB boost
            
        except Exception as e:
            logger.error(f"EQ boost failed: {e}")
            return audio
    
    def _apply_noise_reduction(self, audio: AudioSegment) -> AudioSegment:
        """Apply noise reduction to audio."""
        try:
            # Simple noise reduction - would use more sophisticated algorithms
            # For now, just apply a gentle low-pass filter effect
            return audio.low_pass_filter(8000)
            
        except Exception as e:
            logger.error(f"Noise reduction failed: {e}")
            return audio
    
    def _adjust_stereo_width(self, audio: AudioSegment, width: float) -> AudioSegment:
        """Adjust stereo width of audio."""
        try:
            # Simple stereo width adjustment
            if width > 1.0:
                # Widen stereo
                left = audio.split_to_mono()[0]
                right = audio.split_to_mono()[1]
                
                # Create stereo with widened effect
                widened = AudioSegment.from_mono_audiosegments(left, right)
                return widened
            else:
                return audio
                
        except Exception as e:
            logger.error(f"Stereo width adjustment failed: {e}")
            return audio
    
    def _apply_reverb(self, audio: AudioSegment, amount: float) -> AudioSegment:
        """Apply reverb effect to audio."""
        try:
            # Simple reverb simulation - would use more sophisticated reverb
            # For now, just add a slight echo
            delay_ms = int(50 * amount)  # 50ms base delay
            echo = audio - 6  # 6dB quieter
            return audio.overlay(echo, delay_ms)
            
        except Exception as e:
            logger.error(f"Reverb application failed: {e}")
            return audio
    
    def _apply_delay(self, audio: AudioSegment, amount: float) -> AudioSegment:
        """Apply delay effect to audio."""
        try:
            # Simple delay effect
            delay_ms = int(200 * amount)  # 200ms base delay
            delayed = audio - 8  # 8dB quieter
            return audio.overlay(delayed, delay_ms)
            
        except Exception as e:
            logger.error(f"Delay application failed: {e}")
            return audio
    
    def _apply_chorus(self, audio: AudioSegment, amount: float) -> AudioSegment:
        """Apply chorus effect to audio."""
        try:
            # Simple chorus simulation - would use more sophisticated chorus
            # For now, just add slight pitch variation
            pitch_shift = int(5 * amount)  # 5 cents pitch shift
            if pitch_shift > 0:
                return audio._spawn(audio.raw_data, overrides={"frame_rate": int(audio.frame_rate * (1 + pitch_shift/100))})
            else:
                return audio
                
        except Exception as e:
            logger.error(f"Chorus application failed: {e}")
            return audio
    
    def _get_bitrate(self, quality: AudioQuality) -> str:
        """Get bitrate for audio quality."""
        bitrates = {
            AudioQuality.LOW: "128k",
            AudioQuality.MEDIUM: "192k",
            AudioQuality.HIGH: "320k",
            AudioQuality.PROFESSIONAL: "320k"
        }
        return bitrates.get(quality, "192k")

class SoundEffectLibrary:
    """Manages sound effects library."""
    
    def __init__(self):
        self.effects = []
        self.categories = {}
        self._load_sound_effects()
    
    def _load_sound_effects(self):
        """Load sound effects library."""
        try:
            # Placeholder - would load from actual sound effects library
            self.effects = self._generate_sample_effects()
            self._categorize_effects()
            
            logger.info(f"Loaded {len(self.effects)} sound effects")
            
        except Exception as e:
            logger.error(f"Failed to load sound effects: {e}")
            self.effects = []
    
    def _generate_sample_effects(self) -> List[SoundEffect]:
        """Generate sample sound effects for demo."""
        effects = []
        
        sample_effects = [
            {"name": "Whoosh", "category": "transition", "duration": 0.5},
            {"name": "Pop", "category": "accent", "duration": 0.2},
            {"name": "Click", "category": "interface", "duration": 0.1},
            {"name": "Swoosh", "category": "transition", "duration": 0.8},
            {"name": "Ding", "category": "notification", "duration": 0.3},
            {"name": "Applause", "category": "reaction", "duration": 2.0},
            {"name": "Laugh", "category": "reaction", "duration": 1.5},
            {"name": "Gasp", "category": "reaction", "duration": 0.8},
            {"name": "Drum Roll", "category": "build_up", "duration": 3.0},
            {"name": "Cymbal Crash", "category": "accent", "duration": 1.0}
        ]
        
        for i, effect_data in enumerate(sample_effects):
            effect = SoundEffect(
                effect_id=f"effect_{i}",
                name=effect_data["name"],
                category=effect_data["category"],
                duration=effect_data["duration"],
                file_path=f"/tmp/sound_effects/{effect_data['name'].lower()}.wav",
                volume=0.8
            )
            effects.append(effect)
        
        return effects
    
    def _categorize_effects(self):
        """Categorize sound effects by type."""
        try:
            for effect in self.effects:
                category = effect.category
                if category not in self.categories:
                    self.categories[category] = []
                self.categories[category].append(effect)
                
        except Exception as e:
            logger.error(f"Effect categorization failed: {e}")
    
    async def suggest_effects(self, 
                            content_analysis: Dict[str, Any],
                            video_duration: float) -> List[SoundEffect]:
        """Suggest sound effects based on content analysis."""
        try:
            suggestions = []
            
            # Analyze content for effect opportunities
            content_type = content_analysis.get("content_type", "general")
            engagement_points = content_analysis.get("engagement_points", [])
            keywords = content_analysis.get("keywords", [])
            
            # Suggest effects based on content type
            if content_type == "comedy":
                suggestions.extend(self._get_effects_by_category("reaction"))
            elif content_type == "tutorial":
                suggestions.extend(self._get_effects_by_category("interface"))
            elif content_type == "dance":
                suggestions.extend(self._get_effects_by_category("accent"))
            
            # Suggest transition effects
            if len(engagement_points) > 1:
                suggestions.extend(self._get_effects_by_category("transition"))
            
            # Suggest effects based on keywords
            for keyword in keywords:
                if keyword.lower() in ["amazing", "wow", "incredible"]:
                    suggestions.extend(self._get_effects_by_category("reaction"))
                elif keyword.lower() in ["click", "tap", "press"]:
                    suggestions.extend(self._get_effects_by_category("interface"))
            
            # Remove duplicates and limit suggestions
            unique_suggestions = list({effect.effect_id: effect for effect in suggestions}.values())
            
            return unique_suggestions[:5]  # Limit to 5 suggestions
            
        except Exception as e:
            logger.error(f"Effect suggestion failed: {e}")
            return []
    
    def _get_effects_by_category(self, category: str) -> List[SoundEffect]:
        """Get effects by category."""
        return self.categories.get(category, [])

class AudioMixer:
    """Mixes multiple audio tracks together."""
    
    def __init__(self):
        self.mixing_presets = self._load_mixing_presets()
    
    def _load_mixing_presets(self) -> Dict[str, Dict[str, float]]:
        """Load audio mixing presets."""
        return {
            "background_music": {
                "volume": 0.3,  # 30% volume
                "fade_in": 1.0,  # 1 second fade in
                "fade_out": 1.0,  # 1 second fade out
                "ducking": True  # Duck when speech is present
            },
            "sound_effects": {
                "volume": 0.8,  # 80% volume
                "fade_in": 0.1,  # 0.1 second fade in
                "fade_out": 0.1,  # 0.1 second fade out
                "ducking": False
            },
            "voice": {
                "volume": 1.0,  # 100% volume
                "fade_in": 0.0,  # No fade in
                "fade_out": 0.0,  # No fade out
                "ducking": False
            }
        }
    
    async def mix_audio(self, 
                       tracks: List[Tuple[str, str, Dict[str, float]]],  # (file_path, track_type, settings)
                       output_path: str,
                       duration: float) -> Dict[str, Any]:
        """Mix multiple audio tracks together."""
        try:
            logger.info(f"Mixing {len(tracks)} audio tracks")
            
            # Start with silence
            mixed_audio = AudioSegment.silent(duration=int(duration * 1000))
            
            # Mix each track
            for file_path, track_type, settings in tracks:
                try:
                    # Load track
                    track = AudioSegment.from_file(file_path)
                    
                    # Apply settings
                    track = self._apply_track_settings(track, settings)
                    
                    # Overlay on mixed audio
                    mixed_audio = mixed_audio.overlay(track)
                    
                except Exception as e:
                    logger.warning(f"Failed to mix track {file_path}: {e}")
                    continue
            
            # Export mixed audio
            mixed_audio.export(output_path, format="mp3", bitrate="192k")
            
            return {
                "output_path": output_path,
                "tracks_mixed": len(tracks),
                "duration": len(mixed_audio) / 1000.0,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Audio mixing failed: {e}")
            raise ProcessingError(f"Audio mixing failed: {e}")
    
    def _apply_track_settings(self, track: AudioSegment, settings: Dict[str, float]) -> AudioSegment:
        """Apply settings to audio track."""
        try:
            # Volume adjustment
            volume = settings.get("volume", 1.0)
            track = track - (60 - (60 * volume))  # Convert to dB
            
            # Fade in
            fade_in = settings.get("fade_in", 0.0)
            if fade_in > 0:
                track = track.fade_in(int(fade_in * 1000))
            
            # Fade out
            fade_out = settings.get("fade_out", 0.0)
            if fade_out > 0:
                track = track.fade_out(int(fade_out * 1000))
            
            return track
            
        except Exception as e:
            logger.error(f"Track settings application failed: {e}")
            return track

class AudioProcessingSystem:
    """Main audio processing system."""
    
    def __init__(self):
        self.music_library = MusicLibrary()
        self.audio_enhancer = AudioEnhancer()
        self.sound_effects = SoundEffectLibrary()
        self.audio_mixer = AudioMixer()
    
    async def process_video_audio(self, 
                                video_path: str,
                                content_analysis: Dict[str, Any],
                                output_path: str) -> Dict[str, Any]:
        """Process video audio with music and effects."""
        try:
            logger.info(f"Processing audio for video: {video_path}")
            
            # Extract audio from video
            audio_path = await self._extract_audio(video_path)
            
            # Enhance original audio
            enhanced_audio_path = await self._enhance_audio(audio_path)
            
            # Find matching background music
            matching_tracks = await self.music_library.find_matching_tracks(
                content_analysis, 
                content_analysis.get("duration", 30.0)
            )
            
            # Suggest sound effects
            suggested_effects = await self.sound_effects.suggest_effects(
                content_analysis,
                content_analysis.get("duration", 30.0)
            )
            
            # Mix audio tracks
            tracks_to_mix = [
                (enhanced_audio_path, "voice", self.audio_mixer.mixing_presets["voice"])
            ]
            
            if matching_tracks:
                # Add background music (simplified - would load actual track)
                tracks_to_mix.append(
                    (enhanced_audio_path, "background_music", self.audio_mixer.mixing_presets["background_music"])
                )
            
            # Mix all tracks
            mixed_audio_path = await self._mix_audio_tracks(tracks_to_mix, output_path)
            
            return {
                "input_video": video_path,
                "output_audio": mixed_audio_path,
                "enhanced_audio": enhanced_audio_path,
                "matching_tracks": len(matching_tracks),
                "suggested_effects": len(suggested_effects),
                "processing_successful": True
            }
            
        except Exception as e:
            logger.error(f"Audio processing failed: {e}")
            raise ProcessingError(f"Audio processing failed: {e}")
    
    async def _extract_audio(self, video_path: str) -> str:
        """Extract audio from video."""
        try:
            audio_path = f"/tmp/audio_{int(time.time())}.wav"
            
            # Use ffmpeg to extract audio
            import subprocess
            result = subprocess.run([
                "ffmpeg", "-i", video_path, "-vn", "-acodec", "pcm_s16le", 
                "-ar", "44100", "-ac", "2", audio_path, "-y"
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                raise ProcessingError(f"Audio extraction failed: {result.stderr}")
            
            return audio_path
            
        except Exception as e:
            logger.error(f"Audio extraction failed: {e}")
            raise ProcessingError(f"Audio extraction failed: {e}")
    
    async def _enhance_audio(self, audio_path: str) -> str:
        """Enhance audio quality."""
        try:
            enhanced_path = f"/tmp/enhanced_audio_{int(time.time())}.wav"
            
            await self.audio_enhancer.enhance_audio(
                audio_path, 
                enhanced_path,
                quality=AudioQuality.HIGH
            )
            
            return enhanced_path
            
        except Exception as e:
            logger.error(f"Audio enhancement failed: {e}")
            return audio_path  # Return original if enhancement fails
    
    async def _mix_audio_tracks(self, tracks: List[Tuple[str, str, Dict]], output_path: str) -> str:
        """Mix audio tracks together."""
        try:
            duration = 30.0  # Default duration - would calculate from video
            
            result = await self.audio_mixer.mix_audio(tracks, output_path, duration)
            
            return output_path
            
        except Exception as e:
            logger.error(f"Audio mixing failed: {e}")
            raise ProcessingError(f"Audio mixing failed: {e}")

# Export the main class
__all__ = ["AudioProcessingSystem", "MusicLibrary", "AudioEnhancer", "SoundEffectLibrary", "AudioMixer"]


