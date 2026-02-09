#!/usr/bin/env python3
"""
Configuration Management System
==============================

Centralized configuration management for the HeyGen AI system with:
- Environment variable support
- Configuration validation
- Multiple environment profiles
- Dynamic configuration updates
"""

import os
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field, asdict
from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)

# =============================================================================
# Base Configuration Models
# =============================================================================

class BaseConfig(BaseModel):
    """Base configuration class with validation."""
    
    class Config:
        extra = "forbid"  # Prevent additional fields
        validate_assignment = True  # Validate on assignment

@dataclass
class AvatarConfig:
    """Avatar generation configuration."""
    
    # Model settings
    default_style: str = "realistic"
    default_quality: str = "high"
    default_resolution: str = "1024x1024"
    
    # Generation settings
    enable_expressions: bool = True
    enable_lighting: bool = True
    enable_lip_sync: bool = True
    enable_enhancement: bool = True
    
    # Pipeline settings
    stable_diffusion_model: str = "runwayml/stable-diffusion-v1-5"
    stable_diffusion_xl_model: str = "stabilityai/stable-diffusion-xl-base-1.0"
    enable_xformers: bool = True
    enable_attention_slicing: bool = True
    
    # Quality settings
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    enable_safety_checker: bool = True

@dataclass
class VoiceConfig:
    """Voice synthesis configuration."""
    
    # Model settings
    default_voice_id: str = "voice_001"
    default_language: str = "en"
    default_quality: str = "high"
    
    # TTS settings
    tts_model: str = "tts_models/en/ljspeech/tacotron2-DDC"
    enable_voice_cloning: bool = True
    enable_emotion_control: bool = True
    enable_speed_control: bool = True
    enable_pitch_control: bool = True
    
    # Audio settings
    sample_rate: int = 22050
    audio_format: str = "wav"
    enable_noise_reduction: bool = True
    enable_audio_enhancement: bool = True
    
    # ElevenLabs integration
    elevenlabs_api_key: Optional[str] = None
    enable_elevenlabs: bool = False

@dataclass
class VideoConfig:
    """Video rendering configuration."""
    
    # Quality settings
    default_quality: str = "high"
    default_resolution: str = "1080p"
    default_fps: int = 30
    default_format: str = "mp4"
    
    # Rendering settings
    enable_hardware_acceleration: bool = True
    enable_parallel_processing: bool = True
    enable_caching: bool = True
    cache_ttl_hours: int = 24
    
    # Effects settings
    enable_fade_effects: bool = True
    enable_text_overlay: bool = True
    enable_watermark: bool = False
    enable_color_correction: bool = True
    
    # Output settings
    output_directory: str = "./generated_videos"
    temp_directory: str = "./temp"
    enable_compression: bool = True
    compression_quality: int = 85

@dataclass
class ProcessingConfig:
    """Processing pipeline configuration."""
    
    # Job management
    max_concurrent_jobs: int = 3
    job_timeout_minutes: int = 30
    enable_job_queue: bool = True
    enable_retry_on_failure: bool = True
    max_retry_attempts: int = 3
    
    # Performance settings
    enable_gpu_acceleration: bool = True
    enable_mixed_precision: bool = True
    enable_gradient_checkpointing: bool = False
    enable_attention_slicing: bool = True
    
    # Memory settings
    max_memory_usage_gb: float = 8.0
    enable_memory_optimization: bool = True
    enable_model_offloading: bool = False

@dataclass
class SystemConfig:
    """System-level configuration."""
    
    # Environment
    environment: str = "development"
    debug_mode: bool = False
    log_level: str = "INFO"
    
    # Security
    enable_authentication: bool = False
    enable_rate_limiting: bool = True
    max_requests_per_minute: int = 60
    
    # Monitoring
    enable_health_checks: bool = True
    enable_metrics: bool = True
    enable_logging: bool = True
    
    # Storage
    data_directory: str = "./data"
    models_directory: str = "./models"
    cache_directory: str = "./cache"

# =============================================================================
# Main Configuration Class
# =============================================================================

class HeyGenAIConfig(BaseConfig):
    """Main configuration class for the HeyGen AI system."""
    
    # Component configurations
    avatar: AvatarConfig = Field(default_factory=AvatarConfig)
    voice: VoiceConfig = Field(default_factory=VoiceConfig)
    video: VideoConfig = Field(default_factory=VideoConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    system: SystemConfig = Field(default_factory=SystemConfig)
    
    # Validation methods
    @validator('avatar')
    def validate_avatar_config(cls, v):
        if v.default_quality not in ['low', 'medium', 'high', 'ultra']:
            raise ValueError("Avatar quality must be one of: low, medium, high, ultra")
        return v
    
    @validator('voice')
    def validate_voice_config(cls, v):
        if v.sample_rate not in [8000, 16000, 22050, 44100, 48000]:
            raise ValueError("Sample rate must be one of: 8000, 16000, 22050, 44100, 48000")
        return v
    
    @validator('video')
    def validate_video_config(cls, v):
        if v.default_fps not in [24, 25, 30, 50, 60]:
            raise ValueError("FPS must be one of: 24, 25, 30, 50, 60")
        return v
    
    @validator('processing')
    def validate_processing_config(cls, v):
        if v.max_concurrent_jobs < 1:
            raise ValueError("Max concurrent jobs must be at least 1")
        return v

# =============================================================================
# Configuration Manager
# =============================================================================

class ConfigurationManager:
    """Manages system configuration with environment support and validation."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self._get_default_config_path()
        self.config: Optional[HeyGenAIConfig] = None
        self.environment = os.getenv("HEYGEN_ENV", "development")
        
        # Load configuration
        self._load_configuration()
    
    def _get_default_config_path(self) -> str:
        """Get default configuration file path."""
        return "./config/heygen_ai_config.json"
    
    def _load_configuration(self):
        """Load configuration from file and environment variables."""
        try:
            # Load base configuration
            self.config = self._load_from_file()
            
            # Override with environment variables
            self._override_from_environment()
            
            # Validate configuration
            self._validate_configuration()
            
            logger.info(f"Configuration loaded successfully for environment: {self.environment}")
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            # Use default configuration
            self.config = HeyGenAIConfig()
            logger.info("Using default configuration")
    
    def _load_from_file(self) -> HeyGenAIConfig:
        """Load configuration from JSON file."""
        if not os.path.exists(self.config_path):
            logger.warning(f"Configuration file not found: {self.config_path}")
            return HeyGenAIConfig()
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            # Load environment-specific configuration
            env_config = config_data.get(self.environment, {})
            
            # Merge with base configuration
            base_config = config_data.get("base", {})
            merged_config = {**base_config, **env_config}
            
            return HeyGenAIConfig(**merged_config)
            
        except Exception as e:
            logger.error(f"Failed to parse configuration file: {e}")
            return HeyGenAIConfig()
    
    def _override_from_environment(self):
        """Override configuration with environment variables."""
        if not self.config:
            return
        
        # Avatar settings
        if os.getenv("HEYGEN_AVATAR_DEFAULT_STYLE"):
            self.config.avatar.default_style = os.getenv("HEYGEN_AVATAR_DEFAULT_STYLE")
        
        if os.getenv("HEYGEN_AVATAR_DEFAULT_QUALITY"):
            self.config.avatar.default_quality = os.getenv("HEYGEN_AVATAR_DEFAULT_QUALITY")
        
        # Voice settings
        if os.getenv("HEYGEN_VOICE_DEFAULT_LANGUAGE"):
            self.config.voice.default_language = os.getenv("HEYGEN_VOICE_DEFAULT_LANGUAGE")
        
        if os.getenv("HEYGEN_VOICE_ELEVENLABS_API_KEY"):
            self.config.voice.elevenlabs_api_key = os.getenv("HEYGEN_VOICE_ELEVENLABS_API_KEY")
            self.config.voice.enable_elevenlabs = True
        
        # Video settings
        if os.getenv("HEYGEN_VIDEO_DEFAULT_RESOLUTION"):
            self.config.video.default_resolution = os.getenv("HEYGEN_VIDEO_DEFAULT_RESOLUTION")
        
        if os.getenv("HEYGEN_VIDEO_OUTPUT_DIRECTORY"):
            self.config.video.output_directory = os.getenv("HEYGEN_VIDEO_OUTPUT_DIRECTORY")
        
        # Processing settings
        if os.getenv("HEYGEN_PROCESSING_MAX_CONCURRENT_JOBS"):
            try:
                self.config.processing.max_concurrent_jobs = int(
                    os.getenv("HEYGEN_PROCESSING_MAX_CONCURRENT_JOBS")
                )
            except ValueError:
                pass
        
        # System settings
        if os.getenv("HEYGEN_SYSTEM_ENVIRONMENT"):
            self.config.system.environment = os.getenv("HEYGEN_SYSTEM_ENVIRONMENT")
        
        if os.getenv("HEYGEN_SYSTEM_DEBUG_MODE"):
            self.config.system.debug_mode = os.getenv("HEYGEN_SYSTEM_DEBUG_MODE").lower() == "true"
        
        if os.getenv("HEYGEN_SYSTEM_LOG_LEVEL"):
            self.config.system.log_level = os.getenv("HEYGEN_SYSTEM_LOG_LEVEL")
    
    def _validate_configuration(self):
        """Validate the loaded configuration."""
        if not self.config:
            raise ValueError("Configuration not loaded")
        
        # Validate directories exist or can be created
        self._validate_directories()
        
        # Validate model paths
        self._validate_model_paths()
        
        # Validate API keys
        self._validate_api_keys()
    
    def _validate_directories(self):
        """Validate and create necessary directories."""
        directories = [
            self.config.video.output_directory,
            self.config.video.temp_directory,
            self.config.system.data_directory,
            self.config.system.models_directory,
            self.config.system.cache_directory
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            logger.debug(f"Directory validated: {directory}")
    
    def _validate_model_paths(self):
        """Validate that model paths are accessible."""
        # This would check if models are available
        # For now, just log the paths
        logger.debug(f"Avatar models: {self.config.avatar.stable_diffusion_model}")
        logger.debug(f"Voice models: {self.config.voice.tts_model}")
    
    def _validate_api_keys(self):
        """Validate API keys if required."""
        if self.config.voice.enable_elevenlabs and not self.config.voice.elevenlabs_api_key:
            logger.warning("ElevenLabs enabled but no API key provided")
    
    def get_config(self) -> HeyGenAIConfig:
        """Get the current configuration."""
        if not self.config:
            raise RuntimeError("Configuration not loaded")
        return self.config
    
    def update_config(self, updates: Dict[str, Any]):
        """Update configuration with new values."""
        if not self.config:
            raise RuntimeError("Configuration not loaded")
        
        # Convert to dict, update, and recreate
        config_dict = asdict(self.config)
        
        # Deep update
        self._deep_update(config_dict, updates)
        
        # Recreate configuration object
        self.config = HeyGenAIConfig(**config_dict)
        
        # Validate updated configuration
        self._validate_configuration()
        
        logger.info("Configuration updated successfully")
    
    def _deep_update(self, base_dict: Dict[str, Any], updates: Dict[str, Any]):
        """Recursively update nested dictionary."""
        for key, value in updates.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def save_config(self, file_path: Optional[str] = None):
        """Save current configuration to file."""
        if not self.config:
            raise RuntimeError("Configuration not loaded")
        
        save_path = file_path or self.config_path
        
        try:
            # Ensure directory exists
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Convert to dict and save
            config_dict = asdict(self.config)
            
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Configuration saved to: {save_path}")
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            raise
    
    def reload_config(self):
        """Reload configuration from file."""
        logger.info("Reloading configuration...")
        self._load_configuration()
    
    def get_environment_config(self) -> Dict[str, Any]:
        """Get environment-specific configuration."""
        if not self.config:
            return {}
        
        return {
            "environment": self.config.system.environment,
            "debug_mode": self.config.system.debug_mode,
            "log_level": self.config.system.log_level
        }
    
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.config.system.environment.lower() == "production"
    
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.config.system.environment.lower() == "development"
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration."""
        if not self.config:
            return {}
        
        return {
            "level": self.config.system.log_level,
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "enable_file_logging": self.config.system.enable_logging,
            "log_file": f"{self.config.system.data_directory}/heygen_ai.log"
        }

# =============================================================================
# Configuration Factory
# =============================================================================

class ConfigurationFactory:
    """Factory for creating different configuration profiles."""
    
    @staticmethod
    def create_development_config() -> HeyGenAIConfig:
        """Create development configuration."""
        return HeyGenAIConfig(
            system=SystemConfig(
                environment="development",
                debug_mode=True,
                log_level="DEBUG"
            ),
            processing=ProcessingConfig(
                max_concurrent_jobs=2,
                enable_gpu_acceleration=False,
                max_memory_usage_gb=4.0
            )
        )
    
    @staticmethod
    def create_production_config() -> HeyGenAIConfig:
        """Create production configuration."""
        return HeyGenAIConfig(
            system=SystemConfig(
                environment="production",
                debug_mode=False,
                log_level="INFO",
                enable_authentication=True,
                enable_rate_limiting=True
            ),
            processing=ProcessingConfig(
                max_concurrent_jobs=5,
                enable_gpu_acceleration=True,
                max_memory_usage_gb=16.0
            )
        )
    
    @staticmethod
    def create_testing_config() -> HeyGenAIConfig:
        """Create testing configuration."""
        return HeyGenAIConfig(
            system=SystemConfig(
                environment="testing",
                debug_mode=True,
                log_level="DEBUG"
            ),
            processing=ProcessingConfig(
                max_concurrent_jobs=1,
                enable_gpu_acceleration=False,
                max_memory_usage_gb=2.0
            )
        )

# =============================================================================
# Example Usage
# =============================================================================

def main():
    """Example usage of the configuration system."""
    try:
        # Create configuration manager
        config_manager = ConfigurationManager()
        
        # Get configuration
        config = config_manager.get_config()
        
        print(f"Environment: {config.system.environment}")
        print(f"Debug mode: {config.system.debug_mode}")
        print(f"Max concurrent jobs: {config.processing.max_concurrent_jobs}")
        print(f"Avatar quality: {config.avatar.default_quality}")
        print(f"Voice language: {config.voice.default_language}")
        print(f"Video resolution: {config.video.default_resolution}")
        
        # Update configuration
        config_manager.update_config({
            "avatar": {"default_quality": "ultra"},
            "video": {"default_fps": 60}
        })
        
        # Save configuration
        config_manager.save_config()
        
        print("Configuration updated and saved successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()


