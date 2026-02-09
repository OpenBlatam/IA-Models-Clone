"""
Input Validator
Advanced input validation
"""

from typing import Dict, Any, List, Optional
import re
import logging

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Validation error"""
    pass


class InputValidator:
    """Advanced input validation"""
    
    def __init__(self):
        self.max_script_length = 10000
        self.max_title_length = 200
        self.allowed_languages = ["en", "es", "fr", "de", "it", "pt", "ja", "zh", "ko"]
        self.allowed_resolutions = [
            "1920x1080", "1280x720", "1080x1920",
            "1080x1080", "720x1280"
        ]
        self.allowed_fps = [24, 30, 60]
    
    def validate_script(self, script: Dict[str, Any]) -> List[str]:
        """
        Validate script input
        
        Args:
            script: Script dictionary
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        if "text" not in script:
            errors.append("Script text is required")
        elif not isinstance(script["text"], str):
            errors.append("Script text must be a string")
        elif len(script["text"].strip()) == 0:
            errors.append("Script text cannot be empty")
        elif len(script["text"]) > self.max_script_length:
            errors.append(f"Script text exceeds maximum length of {self.max_script_length}")
        
        if "language" in script:
            if script["language"] not in self.allowed_languages:
                errors.append(f"Language must be one of: {', '.join(self.allowed_languages)}")
        
        return errors
    
    def validate_video_config(self, config: Dict[str, Any]) -> List[str]:
        """Validate video configuration"""
        errors = []
        
        if "resolution" in config:
            if config["resolution"] not in self.allowed_resolutions:
                errors.append(f"Resolution must be one of: {', '.join(self.allowed_resolutions)}")
        
        if "fps" in config:
            if config["fps"] not in self.allowed_fps:
                errors.append(f"FPS must be one of: {', '.join(map(str, self.allowed_fps))}")
        
        if "duration" in config:
            if not isinstance(config["duration"], (int, float)):
                errors.append("Duration must be a number")
            elif config["duration"] <= 0:
                errors.append("Duration must be positive")
            elif config["duration"] > 3600:
                errors.append("Duration cannot exceed 3600 seconds")
        
        return errors
    
    def validate_audio_config(self, config: Dict[str, Any]) -> List[str]:
        """Validate audio configuration"""
        errors = []
        
        if "voice" in config:
            if not isinstance(config["voice"], str):
                errors.append("Voice must be a string")
        
        if "speed" in config:
            if not isinstance(config["speed"], (int, float)):
                errors.append("Speed must be a number")
            elif config["speed"] < 0.5 or config["speed"] > 2.0:
                errors.append("Speed must be between 0.5 and 2.0")
        
        return errors
    
    def validate_request(self, request: Dict[str, Any]) -> List[str]:
        """
        Validate complete request
        
        Args:
            request: Request dictionary
            
        Returns:
            List of validation errors
        """
        errors = []
        
        if "script" not in request:
            errors.append("Script is required")
        else:
            errors.extend(self.validate_script(request["script"]))
        
        if "video_config" in request:
            errors.extend(self.validate_video_config(request["video_config"]))
        
        if "audio_config" in request:
            errors.extend(self.validate_audio_config(request["audio_config"]))
        
        return errors
    
    def validate_file_upload(self, filename: str, max_size_mb: int = 100) -> List[str]:
        """Validate file upload"""
        errors = []
        
        # Check extension
        allowed_extensions = [".txt", ".md", ".doc", ".docx"]
        file_ext = "." + filename.split(".")[-1].lower() if "." in filename else ""
        
        if file_ext not in allowed_extensions:
            errors.append(f"File extension must be one of: {', '.join(allowed_extensions)}")
        
        # Check filename length
        if len(filename) > 255:
            errors.append("Filename too long")
        
        # Check for dangerous characters
        if re.search(r'[<>:"|?*]', filename):
            errors.append("Filename contains invalid characters")
        
        return errors


_input_validator: Optional[InputValidator] = None


def get_input_validator() -> InputValidator:
    """Get input validator instance (singleton)"""
    global _input_validator
    if _input_validator is None:
        _input_validator = InputValidator()
    return _input_validator

