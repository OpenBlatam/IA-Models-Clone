"""
Music and Audio Library Service for HeyGen AI
============================================

Provides comprehensive music tracks, sound effects, and audio management
for enterprise-grade AI video generation with professional audio accompaniment.
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

try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class MusicTrack:
    """Music track information."""
    
    track_id: str
    title: str
    artist: str
    album: Optional[str] = None
    genre: str = "ambient"  # ambient, electronic, classical, jazz, rock, pop, etc.
    mood: str = "neutral"  # happy, sad, energetic, calm, dramatic, etc.
    tempo_bpm: int = 120  # beats per minute
    key: Optional[str] = None  # musical key (C, D, E, F, G, A, B)
    duration_seconds: float = 0.0
    file_path: str = ""
    preview_path: str = ""
    waveform_path: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    last_used: Optional[datetime] = None
    usage_count: int = 0
    quality_rating: float = 0.0
    license_type: str = "royalty_free"  # royalty_free, licensed, creative_commons
    price: float = 0.0


@dataclass
class SoundEffect:
    """Sound effect information."""
    
    effect_id: str
    name: str
    description: str
    category: str = "general"  # general, nature, technology, human, animals, etc.
    type: str = "one_shot"  # one_shot, loop, transition, ambient
    duration_seconds: float = 0.0
    file_path: str = ""
    preview_path: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    last_used: Optional[datetime] = None
    usage_count: int = 0
    quality_rating: float = 0.0
    license_type: str = "royalty_free"
    price: float = 0.0


@dataclass
class AudioCategory:
    """Audio category definition."""
    
    category_id: str
    name: str
    description: str
    parent_category: Optional[str] = None
    track_count: int = 0
    effect_count: int = 0
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class AudioStyle:
    """Audio style definition."""
    
    style_id: str
    name: str
    description: str
    characteristics: List[str] = field(default_factory=list)
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class AudioRequest:
    """Request for audio operations."""
    
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    operation: str = ""  # create, update, delete, search, generate
    audio_data: Optional[Dict[str, Any]] = None
    search_criteria: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class AudioResult:
    """Result of audio operations."""
    
    success: bool = False
    message: str = ""
    data: Optional[Union[MusicTrack, SoundEffect, List[Union[MusicTrack, SoundEffect]]]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class MusicAudioLibraryService(BaseService):
    """Service for managing music and audio library."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the music and audio library service."""
        super().__init__("MusicAudioLibraryService", ServiceType.LIBRARY, config)
        
        # Audio storage
        self.music_tracks: Dict[str, MusicTrack] = {}
        self.sound_effects: Dict[str, SoundEffect] = {}
        self.categories: Dict[str, AudioCategory] = {}
        self.styles: Dict[str, AudioStyle] = {}
        
        # File management
        self.music_directory = Path("./music")
        self.effects_directory = Path("./sound_effects")
        self.previews_directory = Path("./audio_previews")
        self.waveforms_directory = Path("./waveforms")
        
        # Error handling
        self.error_handler = ErrorHandler()
        
        # Configuration manager
        self.config_manager = ConfigurationManager()
        
        # Logging service
        self.logging_service = LoggingService()
        
        # Performance tracking
        self.library_stats = {
            "total_tracks": 0,
            "total_effects": 0,
            "active_tracks": 0,
            "active_effects": 0,
            "total_categories": 0,
            "total_styles": 0,
            "total_operations": 0,
            "successful_operations": 0,
            "failed_operations": 0
        }
        
        # Default configuration
        self.default_config = {
            "max_track_size_mb": 100,
            "max_effect_size_mb": 50,
            "supported_formats": ["mp3", "wav", "ogg", "flac", "m4a", "aac"],
            "preview_duration_seconds": 30,
            "waveform_resolution": (800, 200),
            "max_tags_per_audio": 15,
            "auto_generate_previews": True,
            "auto_generate_waveforms": True,
            "quality_threshold": 0.7
        }
        
        # Supported genres and moods
        self.supported_genres = [
            "ambient", "electronic", "classical", "jazz", "rock", "pop", "folk",
            "country", "hip_hop", "r&b", "reggae", "blues", "world", "soundtrack"
        ]
        
        self.supported_moods = [
            "happy", "sad", "energetic", "calm", "dramatic", "mysterious",
            "romantic", "tense", "peaceful", "uplifting", "melancholic", "powerful"
        ]
        
        self.supported_effect_categories = [
            "general", "nature", "technology", "human", "animals", "transportation",
            "household", "weather", "machines", "weapons", "magic", "sci_fi"
        ]

    async def _initialize_service_impl(self) -> bool:
        """Initialize the music and audio library service."""
        try:
            logger.info("Initializing Music and Audio Library Service...")
            
            # Check dependencies
            if not await self._check_dependencies():
                logger.warning("Some dependencies are not available")
            
            # Create directories
            await self._create_directories()
            
            # Load default categories and styles
            await self._load_default_categories_styles()
            
            # Load existing audio files
            await self._load_existing_audio()
            
            # Validate configuration
            await self._validate_configuration()
            
            logger.info("Music and Audio Library Service initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Music and Audio Library Service: {e}")
            return False

    async def _check_dependencies(self) -> bool:
        """Check if required dependencies are available."""
        dependencies_available = True
        
        if not LIBROSA_AVAILABLE:
            logger.warning("librosa not available - audio analysis features limited")
            dependencies_available = False
        
        if not SOUNDFILE_AVAILABLE:
            logger.warning("soundfile not available - audio file operations limited")
            dependencies_available = False
        
        if not PYDUB_AVAILABLE:
            logger.warning("pydub not available - audio manipulation features limited")
            dependencies_available = False
        
        return dependencies_available

    async def _create_directories(self):
        """Create necessary directories."""
        directories = [
            self.music_directory,
            self.effects_directory,
            self.previews_directory,
            self.waveforms_directory
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {directory}")

    async def _load_default_categories_styles(self):
        """Load default audio categories and styles."""
        # Default categories
        default_categories = [
            AudioCategory("music", "Music", "Musical compositions and tracks"),
            AudioCategory("sound_effects", "Sound Effects", "Sound effects and audio elements"),
            AudioCategory("ambient", "Ambient", "Background and atmospheric sounds"),
            AudioCategory("transitions", "Transitions", "Audio transition effects"),
            AudioCategory("jingles", "Jingles", "Short musical identifiers")
        ]
        
        for category in default_categories:
            self.categories[category.category_id] = category
        
        # Default styles
        default_styles = [
            AudioStyle("modern", "Modern", ["contemporary", "trendy", "current"]),
            AudioStyle("classic", "Classic", ["timeless", "traditional", "established"]),
            AudioStyle("minimalist", "Minimalist", ["simple", "clean", "sparse"]),
            AudioStyle("complex", "Complex", ["layered", "rich", "detailed"]),
            AudioStyle("experimental", "Experimental", ["innovative", "avant-garde", "unconventional"])
        ]
        
        for style in default_styles:
            self.styles[style.style_id] = style
        
        logger.info(f"Loaded {len(default_categories)} categories and {len(default_styles)} styles")

    async def _load_existing_audio(self):
        """Load existing audio files from directories."""
        # Load music tracks
        for music_file in self.music_directory.glob("**/*.*"):
            if music_file.suffix.lower() in ['.mp3', '.wav', '.ogg', '.flac', '.m4a', '.aac']:
                await self._load_music_track(music_file)
        
        # Load sound effects
        for effect_file in self.effects_directory.glob("**/*.*"):
            if effect_file.suffix.lower() in ['.mp3', '.wav', '.ogg', '.flac', '.m4a', '.aac']:
                await self._load_sound_effect(effect_file)
        
        logger.info(f"Loaded {len(self.music_tracks)} music tracks and {len(self.sound_effects)} sound effects")

    async def _load_music_track(self, file_path: Path):
        """Load a music track from file."""
        try:
            track_id = str(uuid.uuid4())
            track = MusicTrack(
                track_id=track_id,
                title=file_path.stem,
                artist="Unknown",
                file_path=str(file_path),
                duration_seconds=0.0
            )
            
            # Analyze audio file if possible
            if LIBROSA_AVAILABLE:
                await self._analyze_music_track(track)
            
            self.music_tracks[track_id] = track
            self.library_stats["total_tracks"] += 1
            
        except Exception as e:
            logger.error(f"Failed to load music track {file_path}: {e}")

    async def _load_sound_effect(self, file_path: Path):
        """Load a sound effect from file."""
        try:
            effect_id = str(uuid.uuid4())
            effect = SoundEffect(
                effect_id=effect_id,
                name=file_path.stem,
                description="Sound effect",
                file_path=str(file_path),
                duration_seconds=0.0
            )
            
            # Analyze audio file if possible
            if LIBROSA_AVAILABLE:
                await self._analyze_sound_effect(effect)
            
            self.sound_effects[effect_id] = effect
            self.library_stats["total_effects"] += 1
            
        except Exception as e:
            logger.error(f"Failed to load sound effect {file_path}: {e}")

    async def _analyze_music_track(self, track: MusicTrack):
        """Analyze music track for metadata."""
        try:
            if not LIBROSA_AVAILABLE:
                return
            
            # Load audio file
            y, sr = librosa.load(track.file_path)
            
            # Get duration
            track.duration_seconds = len(y) / sr
            
            # Get tempo
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            track.tempo_bpm = int(tempo)
            
            # Get key (simplified)
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
            key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            key_idx = np.argmax(np.sum(chroma, axis=1))
            track.key = key_names[key_idx]
            
            # Generate preview if enabled
            if self.default_config["auto_generate_previews"]:
                await self._generate_preview(track)
            
            # Generate waveform if enabled
            if self.default_config["auto_generate_waveforms"]:
                await self._generate_waveform(track)
                
        except Exception as e:
            logger.error(f"Failed to analyze music track {track.track_id}: {e}")

    async def _analyze_sound_effect(self, effect: SoundEffect):
        """Analyze sound effect for metadata."""
        try:
            if not LIBROSA_AVAILABLE:
                return
            
            # Load audio file
            y, sr = librosa.load(effect.file_path)
            
            # Get duration
            effect.duration_seconds = len(y) / sr
            
            # Generate preview if enabled
            if self.default_config["auto_generate_previews"]:
                await self._generate_preview(effect)
                
        except Exception as e:
            logger.error(f"Failed to analyze sound effect {effect.effect_id}: {e}")

    async def _generate_preview(self, audio_item: Union[MusicTrack, SoundEffect]):
        """Generate preview for audio item."""
        try:
            if not PYDUB_AVAILABLE:
                return
            
            # Load audio
            audio = AudioSegment.from_file(audio_item.file_path)
            
            # Create preview (first 30 seconds or full duration if shorter)
            preview_duration = min(self.default_config["preview_duration_seconds"] * 1000, len(audio))
            preview = audio[:preview_duration]
            
            # Save preview
            preview_path = self.previews_directory / f"{audio_item.track_id if hasattr(audio_item, 'track_id') else audio_item.effect_id}_preview.mp3"
            preview.export(str(preview_path), format="mp3")
            
            if hasattr(audio_item, 'preview_path'):
                audio_item.preview_path = str(preview_path)
            else:
                audio_item.preview_path = str(preview_path)
                
        except Exception as e:
            logger.error(f"Failed to generate preview: {e}")

    async def _generate_waveform(self, track: MusicTrack):
        """Generate waveform visualization for music track."""
        try:
            if not LIBROSA_AVAILABLE:
                return
            
            # Load audio
            y, sr = librosa.load(track.file_path)
            
            # Generate waveform data
            # This is a simplified waveform generation
            # In production, you might want to use matplotlib or other visualization libraries
            
            waveform_path = self.waveforms_directory / f"{track.track_id}_waveform.json"
            waveform_data = {
                "samples": y[::1000].tolist(),  # Downsample for visualization
                "sample_rate": sr,
                "duration": track.duration_seconds
            }
            
            with open(waveform_path, 'w') as f:
                json.dump(waveform_data, f)
            
            track.waveform_path = str(waveform_path)
            
        except Exception as e:
            logger.error(f"Failed to generate waveform: {e}")

    async def _validate_configuration(self):
        """Validate service configuration."""
        try:
            # Check if directories exist
            for directory in [self.music_directory, self.effects_directory]:
                if not directory.exists():
                    logger.warning(f"Directory does not exist: {directory}")
            
            # Validate supported formats
            if not self.default_config["supported_formats"]:
                logger.warning("No supported audio formats configured")
            
            logger.info("Configuration validation completed")
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")

    @with_error_handling
    @with_retry
    async def create_music_track(self, track_data: Dict[str, Any]) -> AudioResult:
        """Create a new music track."""
        try:
            track_id = str(uuid.uuid4())
            track = MusicTrack(
                track_id=track_id,
                title=track_data.get("title", "Untitled"),
                artist=track_data.get("artist", "Unknown"),
                album=track_data.get("album"),
                genre=track_data.get("genre", "ambient"),
                mood=track_data.get("mood", "neutral"),
                tempo_bpm=track_data.get("tempo_bpm", 120),
                key=track_data.get("key"),
                file_path=track_data.get("file_path", ""),
                tags=track_data.get("tags", [])
            )
            
            # Analyze and process the track
            if track.file_path and Path(track.file_path).exists():
                await self._analyze_music_track(track)
            
            self.music_tracks[track_id] = track
            self.library_stats["total_tracks"] += 1
            self.library_stats["active_tracks"] += 1
            
            await self._update_operation_stats(True)
            
            return AudioResult(
                success=True,
                message="Music track created successfully",
                data=track
            )
            
        except Exception as e:
            await self._update_operation_stats(False)
            raise e

    @with_error_handling
    @with_retry
    async def create_sound_effect(self, effect_data: Dict[str, Any]) -> AudioResult:
        """Create a new sound effect."""
        try:
            effect_id = str(uuid.uuid4())
            effect = SoundEffect(
                effect_id=effect_id,
                name=effect_data.get("name", "Untitled"),
                description=effect_data.get("description", "Sound effect"),
                category=effect_data.get("category", "general"),
                type=effect_data.get("type", "one_shot"),
                file_path=effect_data.get("file_path", ""),
                tags=effect_data.get("tags", [])
            )
            
            # Analyze and process the effect
            if effect.file_path and Path(effect.file_path).exists():
                await self._analyze_sound_effect(effect)
            
            self.sound_effects[effect_id] = effect
            self.library_stats["total_effects"] += 1
            self.library_stats["active_effects"] += 1
            
            await self._update_operation_stats(True)
            
            return AudioResult(
                success=True,
                message="Sound effect created successfully",
                data=effect
            )
            
        except Exception as e:
            await self._update_operation_stats(False)
            raise e

    @with_error_handling
    @with_retry
    async def search_audio(self, search_criteria: Dict[str, Any]) -> AudioResult:
        """Search for audio items based on criteria."""
        try:
            results = []
            
            # Search music tracks
            if search_criteria.get("type") in ["music", "all", None]:
                for track in self.music_tracks.values():
                    if self._matches_criteria(track, search_criteria):
                        results.append(track)
            
            # Search sound effects
            if search_criteria.get("type") in ["effects", "all", None]:
                for effect in self.sound_effects.values():
                    if self._matches_criteria(effect, search_criteria):
                        results.append(effect)
            
            await self._update_operation_stats(True)
            
            return AudioResult(
                success=True,
                message=f"Found {len(results)} audio items",
                data=results
            )
            
        except Exception as e:
            await self._update_operation_stats(False)
            raise e

    def _matches_criteria(self, audio_item: Union[MusicTrack, SoundEffect], criteria: Dict[str, Any]) -> bool:
        """Check if audio item matches search criteria."""
        try:
            # Text search
            if criteria.get("query"):
                query = criteria["query"].lower()
                if hasattr(audio_item, 'title'):
                    if query not in audio_item.title.lower():
                        return False
                elif hasattr(audio_item, 'name'):
                    if query not in audio_item.name.lower():
                        return False
            
            # Genre filter (for music tracks)
            if criteria.get("genre") and hasattr(audio_item, 'genre'):
                if audio_item.genre.lower() != criteria["genre"].lower():
                    return False
            
            # Mood filter (for music tracks)
            if criteria.get("mood") and hasattr(audio_item, 'mood'):
                if audio_item.mood.lower() != criteria["mood"].lower():
                    return False
            
            # Category filter (for sound effects)
            if criteria.get("category") and hasattr(audio_item, 'category'):
                if audio_item.category.lower() != criteria["category"].lower():
                    return False
            
            # Duration filter
            if criteria.get("min_duration"):
                if audio_item.duration_seconds < criteria["min_duration"]:
                    return False
            
            if criteria.get("max_duration"):
                if audio_item.duration_seconds > criteria["max_duration"]:
                    return False
            
            # Tempo filter (for music tracks)
            if criteria.get("min_tempo") and hasattr(audio_item, 'tempo_bpm'):
                if audio_item.tempo_bpm < criteria["min_tempo"]:
                    return False
            
            if criteria.get("max_tempo") and hasattr(audio_item, 'tempo_bpm'):
                if audio_item.tempo_bpm > criteria["max_tempo"]:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error matching criteria: {e}")
            return False

    @with_error_handling
    @with_retry
    async def get_audio_by_id(self, audio_id: str, audio_type: str = "auto") -> AudioResult:
        """Get audio item by ID."""
        try:
            # Auto-detect type
            if audio_type == "auto":
                if audio_id in self.music_tracks:
                    audio_type = "music"
                elif audio_id in self.sound_effects:
                    audio_type = "effects"
                else:
                    return AudioResult(
                        success=False,
                        message="Audio item not found"
                    )
            
            if audio_type == "music":
                track = self.music_tracks.get(audio_id)
                if track:
                    await self._update_operation_stats(True)
                    return AudioResult(
                        success=True,
                        message="Music track found",
                        data=track
                    )
            elif audio_type == "effects":
                effect = self.sound_effects.get(audio_id)
                if effect:
                    await self._update_operation_stats(True)
                    return AudioResult(
                        success=True,
                        message="Sound effect found",
                        data=effect
                    )
            
            await self._update_operation_stats(False)
            return AudioResult(
                success=False,
                message="Audio item not found"
            )
            
        except Exception as e:
            await self._update_operation_stats(False)
            raise e

    @with_error_handling
    @with_retry
    async def get_categories(self) -> AudioResult:
        """Get all audio categories."""
        try:
            await self._update_operation_stats(True)
            return AudioResult(
                success=True,
                message=f"Found {len(self.categories)} categories",
                data=list(self.categories.values())
            )
        except Exception as e:
            await self._update_operation_stats(False)
            raise e

    @with_error_handling
    @with_retry
    async def get_styles(self) -> AudioResult:
        """Get all audio styles."""
        try:
            await self._update_operation_stats(True)
            return AudioResult(
                success=True,
                message=f"Found {len(self.styles)} styles",
                data=list(self.styles.values())
            )
        except Exception as e:
            await self._update_operation_stats(False)
            raise e

    @with_error_handling
    @with_retry
    async def get_popular_audio(self, limit: int = 10, audio_type: str = "all") -> AudioResult:
        """Get most popular audio items."""
        try:
            results = []
            
            if audio_type in ["music", "all"]:
                popular_tracks = sorted(
                    self.music_tracks.values(),
                    key=lambda x: x.usage_count,
                    reverse=True
                )[:limit]
                results.extend(popular_tracks)
            
            if audio_type in ["effects", "all"]:
                popular_effects = sorted(
                    self.sound_effects.values(),
                    key=lambda x: x.usage_count,
                    reverse=True
                )[:limit]
                results.extend(popular_effects)
            
            await self._update_operation_stats(True)
            
            return AudioResult(
                success=True,
                message=f"Found {len(results)} popular audio items",
                data=results
            )
            
        except Exception as e:
            await self._update_operation_stats(False)
            raise e

    async def _update_operation_stats(self, success: bool):
        """Update operation statistics."""
        self.library_stats["total_operations"] += 1
        if success:
            self.library_stats["successful_operations"] += 1
        else:
            self.library_stats["failed_operations"] += 1

    async def health_check(self) -> HealthCheckResult:
        """Check the health of the music and audio library service."""
        try:
            # Call parent health check
            parent_health = await super().health_check()
            
            # Check dependencies
            dependencies_healthy = await self._check_dependencies()
            
            # Check file system
            directories_exist = all(
                directory.exists() for directory in [
                    self.music_directory,
                    self.effects_directory,
                    self.previews_directory,
                    self.waveforms_directory
                ]
            )
            
            # Check audio files
            audio_files_accessible = len(self.music_tracks) > 0 or len(self.sound_effects) > 0
            
            # Determine overall health
            if (parent_health.status == ServiceStatus.HEALTHY and
                dependencies_healthy and
                directories_exist and
                audio_files_accessible):
                status = ServiceStatus.HEALTHY
                message = "Music and Audio Library Service is healthy"
            else:
                status = ServiceStatus.DEGRADED
                message = "Music and Audio Library Service has some issues"
            
            return HealthCheckResult(
                service_name=self.service_name,
                status=status,
                message=message,
                details={
                    "parent_health": parent_health.details,
                    "dependencies_healthy": dependencies_healthy,
                    "directories_exist": directories_exist,
                    "audio_files_accessible": audio_files_accessible,
                    "total_tracks": len(self.music_tracks),
                    "total_effects": len(self.sound_effects),
                    "total_categories": len(self.categories),
                    "total_styles": len(self.styles)
                }
            )
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return HealthCheckResult(
                service_name=self.service_name,
                status=ServiceStatus.UNHEALTHY,
                message=f"Health check failed: {e}",
                details={"error": str(e)}
            )

    async def cleanup_temp_files(self):
        """Clean up temporary files."""
        try:
            # Clean up previews older than 24 hours
            cutoff_time = datetime.now() - timedelta(hours=24)
            
            for preview_file in self.previews_directory.glob("*_preview.*"):
                if preview_file.stat().st_mtime < cutoff_time.timestamp():
                    preview_file.unlink()
                    logger.info(f"Cleaned up old preview: {preview_file}")
            
            logger.info("Temp file cleanup completed")
            
        except Exception as e:
            logger.error(f"Temp file cleanup failed: {e}")

    async def shutdown(self):
        """Shutdown the music and audio library service."""
        try:
            logger.info("Shutting down Music and Audio Library Service...")
            
            # Clean up temp files
            await self.cleanup_temp_files()
            
            # Save statistics
            await self._save_statistics()
            
            logger.info("Music and Audio Library Service shutdown completed")
            
        except Exception as e:
            logger.error(f"Shutdown failed: {e}")

    async def _save_statistics(self):
        """Save service statistics."""
        try:
            stats_file = Path("./logs/music_audio_library_stats.json")
            stats_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(stats_file, 'w') as f:
                json.dump(self.library_stats, f, indent=2, default=str)
            
            logger.info("Statistics saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to save statistics: {e}")
