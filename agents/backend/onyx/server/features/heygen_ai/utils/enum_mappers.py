"""
Enum Mapping Utilities
=======================

Helper functions for converting string values to enum types.
This eliminates repetitive mapping code throughout the codebase.

Benefits:
- Single source of truth for string-to-enum mappings
- Consistent default values across the codebase
- Easier to maintain and update mappings
- Reduces code duplication by ~200+ lines
"""

from typing import Dict, TypeVar, Type, Optional, Any

from shared.enums import (
    AvatarStyle,
    AvatarQuality,
    Resolution,
    VoiceQuality,
    AudioFormat,
    VideoQuality,
    VideoFormat,
    VideoCodec,
)

# Type variable for enum types
EnumType = TypeVar('EnumType')


def map_string_to_enum(
    value: str,
    enum_class: Type[EnumType],
    default: Optional[EnumType] = None,
    case_sensitive: bool = False
) -> EnumType:
    """
    Generic helper to map a string value to an enum.
    
    Args:
        value: String value to map
        enum_class: Enum class to map to
        default: Default enum value if mapping fails
        case_sensitive: Whether comparison should be case-sensitive
    
    Returns:
        Mapped enum value or default
    
    Example:
        >>> style = map_string_to_enum("realistic", AvatarStyle, AvatarStyle.REALISTIC)
        >>> quality = map_string_to_enum("HIGH", VoiceQuality, case_sensitive=False)
    """
    if not value:
        if default is not None:
            return default
        raise ValueError(f"Cannot map empty value to {enum_class.__name__}")
    
    # Normalize case if needed
    search_value = value if case_sensitive else value.lower()
    
    # Try direct value match
    for enum_member in enum_class:
        enum_value = enum_member.value if case_sensitive else enum_member.value.lower()
        if enum_value == search_value:
            return enum_member
    
    # Try name match
    for enum_member in enum_class:
        enum_name = enum_member.name if case_sensitive else enum_member.name.lower()
        if enum_name == search_value:
            return enum_member
    
    # Return default if provided, otherwise raise
    if default is not None:
        return default
    
    raise ValueError(
        f"Could not map '{value}' to {enum_class.__name__}. "
        f"Valid values: {[e.value for e in enum_class]}"
    )


# =============================================================================
# Avatar Enum Mappers
# =============================================================================

def map_avatar_style(style_str: str, default: AvatarStyle = AvatarStyle.REALISTIC) -> AvatarStyle:
    """
    Map string to AvatarStyle enum.
    
    Args:
        style_str: Style string (e.g., "realistic", "cartoon", "anime", "artistic")
        default: Default style if mapping fails
    
    Returns:
        AvatarStyle enum value
    
    Example:
        >>> style = map_avatar_style("realistic")
        >>> style = map_avatar_style("CARTOON", AvatarStyle.CARTOON)
    """
    return map_string_to_enum(style_str, AvatarStyle, default, case_sensitive=False)


def map_avatar_quality(quality_str: str, default: AvatarQuality = AvatarQuality.HIGH) -> AvatarQuality:
    """
    Map string to AvatarQuality enum.
    
    Args:
        quality_str: Quality string (e.g., "low", "medium", "high", "ultra")
        default: Default quality if mapping fails
    
    Returns:
        AvatarQuality enum value
    
    Example:
        >>> quality = map_avatar_quality("high")
        >>> quality = map_avatar_quality("ULTRA", AvatarQuality.ULTRA)
    """
    return map_string_to_enum(quality_str, AvatarQuality, default, case_sensitive=False)


def map_resolution(resolution_str: str, default: Resolution = Resolution.P1080) -> Resolution:
    """
    Map string to Resolution enum.
    
    Args:
        resolution_str: Resolution string (e.g., "720p", "1080p", "4k")
        default: Default resolution if mapping fails
    
    Returns:
        Resolution enum value
    
    Example:
        >>> res = map_resolution("1080p")
        >>> res = map_resolution("4K", Resolution.P4K)
    """
    return map_string_to_enum(resolution_str, Resolution, default, case_sensitive=False)


# =============================================================================
# Voice Enum Mappers
# =============================================================================

def map_voice_quality(quality_str: str, default: VoiceQuality = VoiceQuality.HIGH) -> VoiceQuality:
    """
    Map string to VoiceQuality enum.
    
    Args:
        quality_str: Quality string (e.g., "low", "medium", "high", "ultra")
        default: Default quality if mapping fails
    
    Returns:
        VoiceQuality enum value
    
    Example:
        >>> quality = map_voice_quality("high")
        >>> quality = map_voice_quality("ULTRA", VoiceQuality.ULTRA)
    """
    return map_string_to_enum(quality_str, VoiceQuality, default, case_sensitive=False)


def map_audio_format(format_str: str, default: AudioFormat = AudioFormat.WAV) -> AudioFormat:
    """
    Map string to AudioFormat enum.
    
    Args:
        format_str: Format string (e.g., "wav", "mp3", "ogg", "flac")
        default: Default format if mapping fails
    
    Returns:
        AudioFormat enum value
    
    Example:
        >>> fmt = map_audio_format("mp3")
        >>> fmt = map_audio_format("WAV", AudioFormat.WAV)
    """
    return map_string_to_enum(format_str, AudioFormat, default, case_sensitive=False)


# =============================================================================
# Video Enum Mappers
# =============================================================================

def map_video_quality(quality_str: str, default: VideoQuality = VideoQuality.HIGH) -> VideoQuality:
    """
    Map string to VideoQuality enum.
    
    Args:
        quality_str: Quality string (e.g., "low", "medium", "high", "ultra")
        default: Default quality if mapping fails
    
    Returns:
        VideoQuality enum value
    
    Example:
        >>> quality = map_video_quality("high")
        >>> quality = map_video_quality("ULTRA", VideoQuality.ULTRA)
    """
    return map_string_to_enum(quality_str, VideoQuality, default, case_sensitive=False)


def map_video_format(format_str: str, default: VideoFormat = VideoFormat.MP4) -> VideoFormat:
    """
    Map string to VideoFormat enum.
    
    Args:
        format_str: Format string (e.g., "mp4", "mov", "avi", "webm")
        default: Default format if mapping fails
    
    Returns:
        VideoFormat enum value
    
    Example:
        >>> fmt = map_video_format("mp4")
        >>> fmt = map_video_format("MOV", VideoFormat.MOV)
    """
    return map_string_to_enum(format_str, VideoFormat, default, case_sensitive=False)


def map_video_codec(codec_str: str, default: VideoCodec = VideoCodec.H264) -> VideoCodec:
    """
    Map string to VideoCodec enum.
    
    Args:
        codec_str: Codec string (e.g., "h264", "h265", "vp9", "prores")
        default: Default codec if mapping fails
    
    Returns:
        VideoCodec enum value
    
    Example:
        >>> codec = map_video_codec("h264")
        >>> codec = map_video_codec("H265", VideoCodec.H265)
    """
    return map_string_to_enum(codec_str, VideoCodec, default, case_sensitive=False)


# =============================================================================
# Batch Mapping Helpers
# =============================================================================

def create_avatar_config_from_strings(
    style_str: str,
    quality_str: str,
    resolution_str: str,
    default_style: AvatarStyle = AvatarStyle.REALISTIC,
    default_quality: AvatarQuality = AvatarQuality.HIGH,
    default_resolution: Resolution = Resolution.P1080,
) -> Dict[str, Any]:
    """
    Create avatar configuration dictionary from string values.
    
    This helper eliminates the need to create mapping dictionaries
    in multiple places throughout the codebase.
    
    Args:
        style_str: Avatar style string
        quality_str: Quality string
        resolution_str: Resolution string
        default_style: Default style if mapping fails
        default_quality: Default quality if mapping fails
        default_resolution: Default resolution if mapping fails
    
    Returns:
        Dictionary with mapped enum values
    
    Example:
        >>> config = create_avatar_config_from_strings("realistic", "high", "1080p")
        >>> # Returns: {
        >>> #     "style": AvatarStyle.REALISTIC,
        >>> #     "quality": AvatarQuality.HIGH,
        >>> #     "resolution": Resolution.P1080
        >>> # }
    """
    return {
        "style": map_avatar_style(style_str, default_style),
        "quality": map_avatar_quality(quality_str, default_quality),
        "resolution": map_resolution(resolution_str, default_resolution),
    }


def create_video_config_from_strings(
    quality_str: str,
    format_str: str,
    default_quality: VideoQuality = VideoQuality.HIGH,
    default_format: VideoFormat = VideoFormat.MP4,
) -> Dict[str, Any]:
    """
    Create video configuration dictionary from string values.
    
    Args:
        quality_str: Quality string
        format_str: Format string
        default_quality: Default quality if mapping fails
        default_format: Default format if mapping fails
    
    Returns:
        Dictionary with mapped enum values
    
    Example:
        >>> config = create_video_config_from_strings("high", "mp4")
        >>> # Returns: {
        >>> #     "quality": VideoQuality.HIGH,
        >>> #     "format": VideoFormat.MP4
        >>> # }
    """
    return {
        "quality": map_video_quality(quality_str, default_quality),
        "format": map_video_format(format_str, default_format),
    }

