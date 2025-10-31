"""
Voice Library Service for HeyGen AI
==================================

Provides comprehensive voice management, storage, and retrieval
for enterprise-grade AI voice synthesis and cloning.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
import uuid
import json
import numpy as np

# Core imports
from .base_service import BaseService, ServiceType, HealthCheckResult, ServiceStatus
from .error_handler import ErrorHandler, with_error_handling, with_retry
from .config_manager import ConfigurationManager
from .logging_service import LoggingService

# Audio processing imports
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class Voice:
    """Voice information."""
    
    voice_id: str
    name: str
    description: str
    language: str  # en, es, fr, de, etc.
    accent: Optional[str] = None  # american, british, australian, etc.
    gender: Optional[str] = None  # male, female, neutral
    age_group: Optional[str] = None  # child, teen, adult, senior
    voice_type: str = "natural"  # natural, synthetic, cloned, etc.
    emotion_style: str = "neutral"  # neutral, happy, sad, angry, etc.
    file_path: str = ""
    sample_path: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    last_used: Optional[datetime] = None
    usage_count: int = 0
    quality_rating: float = 0.0
    duration_seconds: float = 0.0


@dataclass
class VoiceCategory:
    """Voice category definition."""
    
    category_id: str
    name: str
    description: str
    parent_category: Optional[str] = None
    voice_count: int = 0
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class VoiceStyle:
    """Voice style definition."""
    
    style_id: str
    name: str
    description: str
    characteristics: List[str] = field(default_factory=list)
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class VoiceRequest:
    """Request for voice operations."""
    
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    operation: str = ""  # create, update, delete, search, clone
    voice_data: Optional[Dict[str, Any]] = None
    search_criteria: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class VoiceResult:
    """Result of voice operation."""
    
    request_id: str
    operation: str
    success: bool
    voice_id: Optional[str] = None
    voices: Optional[List[Voice]] = None
    error_message: Optional[str] = None
    processing_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


class VoiceLibraryService(BaseService):
    """Service for managing voice library."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the voice library service."""
        super().__init__("VoiceLibraryService", ServiceType.LIBRARY, config)
        
        # Voice storage
        self.voices: Dict[str, Voice] = {}
        self.categories: Dict[str, VoiceCategory] = {}
        self.styles: Dict[str, VoiceStyle] = {}
        
        # File management
        self.voice_directory = Path("./voices")
        self.sample_directory = Path("./voice_samples")
        
        # Error handling
        self.error_handler = ErrorHandler()
        
        # Configuration manager
        self.config_manager = ConfigurationManager()
        
        # Logging service
        self.logging_service = LoggingService()
        
        # Performance tracking
        self.library_stats = {
            "total_voices": 0,
            "active_voices": 0,
            "total_categories": 0,
            "total_styles": 0,
            "total_operations": 0,
            "successful_operations": 0,
            "failed_operations": 0
        }
        
        # Default configuration
        self.default_config = {
            "max_voice_size_mb": 100,
            "supported_formats": ["wav", "mp3", "ogg", "flac", "m4a"],
            "sample_duration_seconds": 10,
            "max_tags_per_voice": 10,
            "auto_generate_samples": True,
            "quality_threshold": 0.7
        }

    async def _initialize_service_impl(self) -> None:
        """Initialize voice library services."""
        try:
            logger.info("Initializing voice library service...")
            
            # Check dependencies
            await self._check_dependencies()
            
            # Create directories
            await self._create_directories()
            
            # Load default categories and styles
            await self._load_default_categories_styles()
            
            # Load existing voices
            await self._load_existing_voices()
            
            # Validate configuration
            await self._validate_configuration()
            
            logger.info("Voice library service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize voice library service: {e}")
            raise

    async def _check_dependencies(self) -> None:
        """Check required dependencies."""
        missing_deps = []
        
        if not LIBROSA_AVAILABLE:
            missing_deps.append("librosa")
        
        if not SOUNDFILE_AVAILABLE:
            missing_deps.append("soundfile")
        
        if missing_deps:
            logger.warning(f"Missing dependencies: {missing_deps}")
            logger.warning("Some voice features may not be available")

    async def _create_directories(self) -> None:
        """Create necessary directories."""
        try:
            self.voice_directory.mkdir(exist_ok=True)
            self.sample_directory.mkdir(exist_ok=True)
            logger.info("Voice directories created successfully")
            
        except Exception as e:
            logger.warning(f"Failed to create some directories: {e}")

    async def _load_default_categories_styles(self) -> None:
        """Load default voice categories and styles."""
        try:
            # Create default categories
            default_categories = [
                VoiceCategory("cat_natural", "Natural", "Natural human voices"),
                VoiceCategory("cat_synthetic", "Synthetic", "AI-generated synthetic voices"),
                VoiceCategory("cat_cloned", "Cloned", "Voice cloned from samples"),
                VoiceCategory("cat_professional", "Professional", "Professional and business voices"),
                VoiceCategory("cat_casual", "Casual", "Casual and everyday voices"),
                VoiceCategory("cat_character", "Character", "Character and animated voices"),
                VoiceCategory("cat_multilingual", "Multilingual", "Multi-language voices"),
                VoiceCategory("cat_emotion", "Emotional", "Emotionally expressive voices")
            ]
            
            for category in default_categories:
                self.categories[category.category_id] = category
                self.library_stats["total_categories"] += 1
            
            # Create default styles
            default_styles = [
                VoiceStyle("style_neutral", "Neutral", "Neutral and balanced voices", 
                          ["clear", "balanced", "professional"]),
                VoiceStyle("style_warm", "Warm", "Warm and friendly voices", 
                          ["friendly", "approachable", "welcoming"]),
                VoiceStyle("style_authoritative", "Authoritative", "Confident and authoritative voices", 
                          ["confident", "strong", "leadership"]),
                VoiceStyle("style_energetic", "Energetic", "High-energy and enthusiastic voices", 
                          ["enthusiastic", "dynamic", "engaging"]),
                VoiceStyle("style_calm", "Calm", "Calm and soothing voices", 
                          ["soothing", "relaxed", "peaceful"]),
                VoiceStyle("style_playful", "Playful", "Fun and playful voices", 
                          ["fun", "entertaining", "engaging"])
            ]
            
            for style in default_styles:
                self.styles[style.style_id] = style
                self.library_stats["total_styles"] += 1
            
            logger.info(f"Loaded {len(default_categories)} categories and {len(default_styles)} styles")
            
        except Exception as e:
            logger.warning(f"Failed to load some default categories/styles: {e}")

    async def _load_existing_voices(self) -> None:
        """Load existing voices from storage."""
        try:
            # For now, we'll create some sample voices
            # In production, this would load from database or file system
            
            sample_voices = [
                Voice(
                    voice_id="voice_sample_1",
                    name="Professional Male",
                    description="A professional male voice for business content",
                    language="en",
                    accent="american",
                    gender="male",
                    age_group="adult",
                    voice_type="natural",
                    emotion_style="neutral",
                    tags=["professional", "business", "male", "adult", "american"],
                    file_path="./voices/sample_1.wav",
                    sample_path="./voice_samples/sample_1.wav",
                    quality_rating=0.9,
                    duration_seconds=30.0
                ),
                Voice(
                    voice_id="voice_sample_2",
                    name="Friendly Female",
                    description="A warm and friendly female voice",
                    language="en",
                    accent="british",
                    gender="female",
                    age_group="adult",
                    voice_type="natural",
                    emotion_style="warm",
                    tags=["friendly", "warm", "female", "adult", "british"],
                    file_path="./voices/sample_2.wav",
                    sample_path="./voice_samples/sample_2.wav",
                    quality_rating=0.85,
                    duration_seconds=25.0
                ),
                Voice(
                    voice_id="voice_sample_3",
                    name="Energetic Teen",
                    description="An energetic young voice for dynamic content",
                    language="en",
                    accent="australian",
                    gender="neutral",
                    age_group="teen",
                    voice_type="synthetic",
                    emotion_style="energetic",
                    tags=["energetic", "young", "dynamic", "teen", "australian"],
                    file_path="./voices/sample_3.wav",
                    sample_path="./voice_samples/sample_3.wav",
                    quality_rating=0.8,
                    duration_seconds=20.0
                )
            ]
            
            for voice in sample_voices:
                self.voices[voice.voice_id] = voice
                self.library_stats["total_voices"] += 1
                self.library_stats["active_voices"] += 1
                
                # Update category count
                if voice.voice_type in self.categories:
                    self.categories[voice.voice_type].voice_count += 1
            
            logger.info(f"Loaded {len(sample_voices)} sample voices")
            
        except Exception as e:
            logger.warning(f"Failed to load some existing voices: {e}")

    async def _validate_configuration(self) -> None:
        """Validate voice library configuration."""
        if not self.categories:
            raise RuntimeError("No voice categories configured")
        
        if not self.styles:
            raise RuntimeError("No voice styles configured")

    @with_error_handling
    @with_retry(max_attempts=3)
    async def create_voice(self, name: str, description: str, language: str,
                          voice_type: str, file_path: str, **kwargs) -> str:
        """Create a new voice."""
        start_time = time.time()
        
        try:
            logger.info(f"Creating voice: {name}")
            
            # Validate voice type
            if voice_type not in self.categories:
                raise ValueError(f"Invalid voice type: {voice_type}")
            
            # Validate file
            if not Path(file_path).exists():
                raise ValueError(f"Voice file not found: {file_path}")
            
            # Generate voice ID
            voice_id = str(uuid.uuid4())
            
            # Generate sample if enabled
            sample_path = ""
            duration_seconds = 0.0
            if self.default_config["auto_generate_samples"] and LIBROSA_AVAILABLE:
                sample_path, duration_seconds = await self._generate_sample(file_path, voice_id)
            
            # Calculate quality rating
            quality_rating = await self._calculate_quality_rating(file_path)
            
            # Create voice
            voice = Voice(
                voice_id=voice_id,
                name=name,
                description=description,
                language=language,
                voice_type=voice_type,
                file_path=file_path,
                sample_path=sample_path,
                duration_seconds=duration_seconds,
                quality_rating=quality_rating,
                tags=kwargs.get("tags", []),
                accent=kwargs.get("accent"),
                gender=kwargs.get("gender"),
                age_group=kwargs.get("age_group"),
                emotion_style=kwargs.get("emotion_style", "neutral"),
                metadata=kwargs.get("metadata", {})
            )
            
            # Store voice
            self.voices[voice_id] = voice
            self.library_stats["total_voices"] += 1
            self.library_stats["active_voices"] += 1
            
            # Update category count
            if voice_type in self.categories:
                self.categories[voice_type].voice_count += 1
            
            # Update statistics
            self._update_operation_stats(time.time() - start_time, True)
            
            logger.info(f"Voice created successfully: {voice_id}")
            return voice_id
            
        except Exception as e:
            self._update_operation_stats(time.time() - start_time, False)
            logger.error(f"Failed to create voice: {e}")
            raise

    async def _generate_sample(self, file_path: str, voice_id: str) -> Tuple[str, float]:
        """Generate sample for voice."""
        try:
            if not LIBROSA_AVAILABLE or not SOUNDFILE_AVAILABLE:
                return "", 0.0
            
            # Load audio file
            audio, sample_rate = librosa.load(file_path, sr=None)
            
            # Get duration
            duration_seconds = len(audio) / sample_rate
            
            # Extract sample (first 10 seconds or full duration if shorter)
            sample_duration = min(self.default_config["sample_duration_seconds"], duration_seconds)
            sample_length = int(sample_duration * sample_rate)
            sample_audio = audio[:sample_length]
            
            # Save sample
            sample_path = self.sample_directory / f"{voice_id}_sample.wav"
            sf.write(str(sample_path), sample_audio, sample_rate)
            
            return str(sample_path), duration_seconds
            
        except Exception as e:
            logger.warning(f"Failed to generate sample: {e}")
            return "", 0.0

    async def _calculate_quality_rating(self, file_path: str) -> float:
        """Calculate quality rating for voice file."""
        try:
            if not LIBROSA_AVAILABLE:
                return 0.5  # Default rating
            
            # Load audio
            audio, sample_rate = librosa.load(file_path, sr=None)
            
            # Calculate various quality metrics
            # Signal-to-noise ratio (simplified)
            signal_power = np.mean(audio**2)
            noise_power = np.var(audio)
            snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
            
            # Dynamic range
            dynamic_range = np.max(audio) - np.min(audio)
            
            # Spectral centroid (brightness)
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sample_rate))
            
            # Normalize metrics to 0-1 range
            snr_score = min(max((snr + 20) / 40, 0), 1)  # Assume SNR range -20 to +20 dB
            dynamic_score = min(dynamic_range / 2, 1)  # Assume max range of 2
            spectral_score = min(spectral_centroid / 4000, 1)  # Assume max centroid of 4000 Hz
            
            # Weighted average
            quality_rating = 0.4 * snr_score + 0.3 * dynamic_score + 0.3 * spectral_score
            
            return max(min(quality_rating, 1.0), 0.0)
            
        except Exception as e:
            logger.warning(f"Failed to calculate quality rating: {e}")
            return 0.5

    @with_error_handling
    async def search_voices(self, criteria: Dict[str, Any]) -> List[Voice]:
        """Search voices based on criteria."""
        try:
            logger.info(f"Searching voices with criteria: {criteria}")
            
            matching_voices = []
            
            for voice in self.voices.values():
                if not voice.is_active:
                    continue
                
                # Check voice type
                if "voice_type" in criteria and voice.voice_type != criteria["voice_type"]:
                    continue
                
                # Check language
                if "language" in criteria and voice.language != criteria["language"]:
                    continue
                
                # Check gender
                if "gender" in criteria and voice.gender != criteria["gender"]:
                    continue
                
                # Check age group
                if "age_group" in criteria and voice.age_group != criteria["age_group"]:
                    continue
                
                # Check emotion style
                if "emotion_style" in criteria and voice.emotion_style != criteria["emotion_style"]:
                    continue
                
                # Check quality threshold
                if "min_quality" in criteria and voice.quality_rating < criteria["min_quality"]:
                    continue
                
                # Check tags
                if "tags" in criteria:
                    required_tags = criteria["tags"]
                    if not all(tag in voice.tags for tag in required_tags):
                        continue
                
                # Check name/description (case-insensitive)
                if "search_text" in criteria:
                    search_text = criteria["search_text"].lower()
                    if (search_text not in voice.name.lower() and 
                        search_text not in voice.description.lower()):
                        continue
                
                matching_voices.append(voice)
            
            # Sort by quality rating (highest first)
            matching_voices.sort(key=lambda x: x.quality_rating, reverse=True)
            
            logger.info(f"Found {len(matching_voices)} matching voices")
            return matching_voices
            
        except Exception as e:
            logger.error(f"Voice search failed: {e}")
            return []

    @with_error_handling
    async def get_voice(self, voice_id: str) -> Optional[Voice]:
        """Get voice by ID."""
        try:
            if voice_id not in self.voices:
                return None
            
            voice = self.voices[voice_id]
            
            # Update usage statistics
            voice.last_used = datetime.now()
            voice.usage_count += 1
            self.voices[voice_id] = voice
            
            return voice
            
        except Exception as e:
            logger.error(f"Failed to get voice: {e}")
            return None

    @with_error_handling
    async def update_voice(self, voice_id: str, updates: Dict[str, Any]) -> bool:
        """Update voice information."""
        try:
            logger.info(f"Updating voice: {voice_id}")
            
            if voice_id not in self.voices:
                raise ValueError(f"Voice not found: {voice_id}")
            
            voice = self.voices[voice_id]
            
            # Update fields
            for field, value in updates.items():
                if hasattr(voice, field) and field not in ["voice_id", "created_at"]:
                    setattr(voice, field, value)
            
            # Update storage
            self.voices[voice_id] = voice
            
            logger.info(f"Voice updated successfully: {voice_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update voice: {e}")
            raise

    @with_error_handling
    async def delete_voice(self, voice_id: str) -> bool:
        """Delete a voice."""
        try:
            logger.info(f"Deleting voice: {voice_id}")
            
            if voice_id not in self.voices:
                raise ValueError(f"Voice not found: {voice_id}")
            
            voice = self.voices[voice_id]
            
            # Update category count
            if voice.voice_type in self.categories:
                self.categories[voice.voice_type].voice_count -= 1
            
            # Remove voice
            del self.voices[voice_id]
            self.library_stats["total_voices"] -= 1
            self.library_stats["active_voices"] -= 1
            
            logger.info(f"Voice deleted successfully: {voice_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete voice: {e}")
            raise

    @with_error_handling
    async def get_categories(self) -> List[VoiceCategory]:
        """Get all voice categories."""
        return list(self.categories.values())

    @with_error_handling
    async def get_styles(self) -> List[VoiceStyle]:
        """Get all voice styles."""
        return list(self.styles.values())

    @with_error_handling
    async def get_high_quality_voices(self, min_quality: float = 0.8, limit: int = 10) -> List[Voice]:
        """Get high-quality voices."""
        try:
            # Filter by quality and return top voices
            high_quality_voices = [
                voice for voice in self.voices.values() 
                if voice.is_active and voice.quality_rating >= min_quality
            ]
            
            # Sort by quality rating
            high_quality_voices.sort(key=lambda x: x.quality_rating, reverse=True)
            
            return high_quality_voices[:limit]
            
        except Exception as e:
            logger.error(f"Failed to get high-quality voices: {e}")
            return []

    @with_error_handling
    async def get_voices_by_language(self, language: str) -> List[Voice]:
        """Get voices by language."""
        try:
            return [
                voice for voice in self.voices.values() 
                if voice.is_active and voice.language == language
            ]
            
        except Exception as e:
            logger.error(f"Failed to get voices by language: {e}")
            return []

    def _update_operation_stats(self, processing_time: float, success: bool):
        """Update operation statistics."""
        self.library_stats["total_operations"] += 1
        
        if success:
            self.library_stats["successful_operations"] += 1
        else:
            self.library_stats["failed_operations"] += 1

    async def health_check(self) -> HealthCheckResult:
        """Check the health of the voice library service."""
        try:
            # Check base service health
            base_health = await super().health_check()
            
            # Check dependencies
            dependencies = {
                "librosa": LIBROSA_AVAILABLE,
                "soundfile": SOUNDFILE_AVAILABLE
            }
            
            # Check storage
            storage_status = {
                "voice_directory_exists": self.voice_directory.exists(),
                "sample_directory_exists": self.sample_directory.exists(),
                "voice_files_count": len(list(self.voice_directory.glob("*.wav"))) + len(list(self.voice_directory.glob("*.mp3")))
            }
            
            # Check library content
            library_status = {
                "total_voices": self.library_stats["total_voices"],
                "active_voices": self.library_stats["active_voices"],
                "total_categories": self.library_stats["total_categories"],
                "total_styles": self.library_stats["total_styles"]
            }
            
            # Check operations
            operation_status = {
                "total_operations": self.library_stats["total_operations"],
                "successful_operations": self.library_stats["successful_operations"],
                "failed_operations": self.library_stats["failed_operations"]
            }
            
            # Update base health
            base_health.details.update({
                "dependencies": dependencies,
                "storage": storage_status,
                "library": library_status,
                "operations": operation_status,
                "library_stats": self.library_stats
            })
            
            return base_health
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return HealthCheckResult(
                status=ServiceStatus.UNHEALTHY,
                error_message=str(e)
            )

    async def cleanup_temp_files(self) -> None:
        """Clean up temporary voice files."""
        try:
            temp_dir = Path("./temp")
            if temp_dir.exists():
                for voice_file in temp_dir.glob("voice_*"):
                    voice_file.unlink()
                    logger.debug(f"Cleaned up temp file: {voice_file}")
        except Exception as e:
            logger.warning(f"Failed to cleanup temp files: {e}")

    async def shutdown(self) -> None:
        """Shutdown the voice library service."""
        try:
            # Clear data
            self.voices.clear()
            self.categories.clear()
            self.styles.clear()
            
            logger.info("Voice library service shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")


