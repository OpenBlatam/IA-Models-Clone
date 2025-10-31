#!/usr/bin/env python3
"""
Named Exports Example for Video-OpusClip
Demonstrates named exports for commands and utility functions
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
from functools import wraps

# Named exports for security commands
__all__ = [
    # Security commands
    'authenticate_user',
    'validate_input',
    'encrypt_data',
    'decrypt_data',
    'check_rate_limit',
    'block_suspicious_ip',
    
    # Video processing commands
    'process_video',
    'extract_audio',
    'generate_thumbnail',
    'analyze_frames',
    'compress_video',
    
    # Database commands
    'create_user',
    'update_user',
    'delete_user',
    'get_user_by_id',
    'get_all_users',
    
    # Utility functions
    'sanitize_input',
    'validate_email',
    'hash_password',
    'generate_token',
    'log_security_event',
    
    # Configuration
    'SecurityConfig',
    'VideoConfig',
    'DatabaseConfig'
]

# Configuration classes
@dataclass
class SecurityConfig:
    """Security configuration with named exports"""
    secret_key: str = "your-secret-key"
    encryption_key: str = "your-encryption-key"
    max_login_attempts: int = 5
    lockout_duration: int = 900
    rate_limit_requests: int = 100
    rate_limit_window: int = 60

@dataclass
class VideoConfig:
    """Video processing configuration"""
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    supported_formats: List[str] = None
    output_quality: str = "high"
    enable_compression: bool = True
    
    def __post_init__(self):
        if self.supported_formats is None:
            self.supported_formats = [".mp4", ".avi", ".mov", ".mkv"]

@dataclass
class DatabaseConfig:
    """Database configuration"""
    host: str = "localhost"
    port: int = 5432
    database: str = "video_opusclip"
    username: str = "postgres"
    password: str = "password"

# Security commands with named exports
def authenticate_user(username: str, password: str, config: SecurityConfig) -> Dict[str, Any]:
    """Authenticate user with username and password"""
    import hashlib
    
    # Mock user database
    users = {
        "admin": "5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d8",  # "password"
        "user": "5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d8"   # "password"
    }
    
    hashed_password = hashlib.sha256(password.encode()).hexdigest()
    
    if username in users and users[username] == hashed_password:
        return {
            "success": True,
            "user": username,
            "message": "Authentication successful"
        }
    
    return {
        "success": False,
        "message": "Invalid credentials"
    }

def validate_input(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate input data for security"""
    import re
    
    validation_results = {}
    
    # Validate email
    if "email" in input_data:
        email = input_data["email"]
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        validation_results["email"] = {
            "valid": bool(re.match(email_pattern, email)),
            "value": email
        }
    
    # Validate password
    if "password" in input_data:
        password = input_data["password"]
        validation_results["password"] = {
            "valid": len(password) >= 8,
            "strength": "strong" if len(password) >= 12 else "medium"
        }
    
    # Check for suspicious content
    if "content" in input_data:
        content = input_data["content"]
        suspicious_patterns = [
            r'<script.*?</script>',
            r'javascript:',
            r'vbscript:',
            r'data:',
        ]
        
        detected_patterns = []
        for pattern in suspicious_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                detected_patterns.append(pattern)
        
        validation_results["content"] = {
            "valid": len(detected_patterns) == 0,
            "suspicious_patterns": detected_patterns
        }
    
    return validation_results

def encrypt_data(data: str, config: SecurityConfig) -> str:
    """Encrypt sensitive data"""
    from cryptography.fernet import Fernet
    
    cipher = Fernet(config.encryption_key.encode())
    return cipher.encrypt(data.encode()).decode()

def decrypt_data(encrypted_data: str, config: SecurityConfig) -> str:
    """Decrypt encrypted data"""
    from cryptography.fernet import Fernet
    
    cipher = Fernet(config.encryption_key.encode())
    return cipher.decrypt(encrypted_data.encode()).decode()

def check_rate_limit(client_ip: str, request_history: Dict[str, List[float]], config: SecurityConfig) -> bool:
    """Check if client has exceeded rate limit"""
    import time
    
    current_time = time.time()
    
    if client_ip not in request_history:
        return True
    
    # Remove old requests
    recent_requests = [
        req_time for req_time in request_history[client_ip]
        if current_time - req_time < config.rate_limit_window
    ]
    
    return len(recent_requests) < config.rate_limit_requests

def block_suspicious_ip(client_ip: str, blocked_ips: Dict[str, float], config: SecurityConfig) -> None:
    """Block suspicious IP address"""
    import time
    
    blocked_ips[client_ip] = time.time()
    logging.warning(f"Suspicious IP blocked: {client_ip}")

# Video processing commands with named exports
def process_video(video_path: str, config: VideoConfig) -> Dict[str, Any]:
    """Process video file"""
    import os
    
    if not os.path.exists(video_path):
        return {
            "success": False,
            "error": "Video file not found"
        }
    
    file_size = os.path.getsize(video_path)
    if file_size > config.max_file_size:
        return {
            "success": False,
            "error": "Video file too large"
        }
    
    # Mock video processing
    return {
        "success": True,
        "processed_path": f"{video_path}_processed.mp4",
        "file_size": file_size,
        "processing_time": 30.5
    }

def extract_audio(video_path: str) -> Dict[str, Any]:
    """Extract audio from video"""
    # Mock audio extraction
    return {
        "success": True,
        "audio_path": f"{video_path}_audio.wav",
        "duration": 120.5,
        "format": "WAV"
    }

def generate_thumbnail(video_path: str) -> Dict[str, Any]:
    """Generate thumbnail from video"""
    # Mock thumbnail generation
    return {
        "success": True,
        "thumbnail_path": f"{video_path}_thumb.jpg",
        "dimensions": "1920x1080"
    }

def analyze_frames(video_path: str) -> Dict[str, Any]:
    """Analyze video frames"""
    # Mock frame analysis
    return {
        "success": True,
        "total_frames": 3600,
        "fps": 30,
        "duration": 120.0,
        "key_frames": [0, 30, 60, 90, 120]
    }

def compress_video(video_path: str, quality: str = "medium") -> Dict[str, Any]:
    """Compress video file"""
    # Mock video compression
    compression_ratios = {
        "low": 0.3,
        "medium": 0.5,
        "high": 0.7
    }
    
    ratio = compression_ratios.get(quality, 0.5)
    
    return {
        "success": True,
        "compressed_path": f"{video_path}_compressed.mp4",
        "compression_ratio": ratio,
        "original_size": 100 * 1024 * 1024,  # 100MB
        "compressed_size": int(100 * 1024 * 1024 * ratio)
    }

# Database commands with named exports
def create_user(user_data: Dict[str, Any], config: DatabaseConfig) -> Dict[str, Any]:
    """Create new user in database"""
    # Mock database operation
    user_id = len(user_data) + 1
    
    return {
        "success": True,
        "user_id": user_id,
        "message": "User created successfully",
        "user_data": {**user_data, "id": user_id}
    }

def update_user(user_id: int, user_data: Dict[str, Any], config: DatabaseConfig) -> Dict[str, Any]:
    """Update existing user"""
    # Mock database operation
    return {
        "success": True,
        "user_id": user_id,
        "message": "User updated successfully",
        "updated_data": user_data
    }

def delete_user(user_id: int, config: DatabaseConfig) -> Dict[str, Any]:
    """Delete user from database"""
    # Mock database operation
    return {
        "success": True,
        "user_id": user_id,
        "message": "User deleted successfully"
    }

def get_user_by_id(user_id: int, config: DatabaseConfig) -> Optional[Dict[str, Any]]:
    """Get user by ID"""
    # Mock database operation
    users = {
        1: {"id": 1, "username": "admin", "email": "admin@example.com"},
        2: {"id": 2, "username": "user", "email": "user@example.com"}
    }
    
    return users.get(user_id)

def get_all_users(config: DatabaseConfig) -> List[Dict[str, Any]]:
    """Get all users from database"""
    # Mock database operation
    return [
        {"id": 1, "username": "admin", "email": "admin@example.com"},
        {"id": 2, "username": "user", "email": "user@example.com"}
    ]

# Utility functions with named exports
def sanitize_input(input_text: str) -> str:
    """Sanitize user input"""
    import re
    
    # Remove dangerous characters
    sanitized = re.sub(r'[<>"\']', '', input_text)
    # Remove script tags
    sanitized = re.sub(r'<script.*?</script>', '', sanitized, flags=re.IGNORECASE)
    
    return sanitized.strip()

def validate_email(email: str) -> bool:
    """Validate email address format"""
    import re
    
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(email_pattern, email))

def hash_password(password: str) -> str:
    """Hash password using SHA-256"""
    import hashlib
    
    return hashlib.sha256(password.encode()).hexdigest()

def generate_token(user_id: int, config: SecurityConfig) -> str:
    """Generate authentication token"""
    import jwt
    import time
    
    payload = {
        "user_id": user_id,
        "exp": time.time() + 3600,  # 1 hour
        "iat": time.time()
    }
    
    return jwt.encode(payload, config.secret_key, algorithm="HS256")

def log_security_event(event_type: str, details: Dict[str, Any]) -> None:
    """Log security event"""
    from datetime import datetime
    
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "event_type": event_type,
        "details": details
    }
    
    logging.info(f"SECURITY_EVENT: {json.dumps(log_entry)}")

# Command registry for easy access
COMMAND_REGISTRY = {
    # Security commands
    "authenticate_user": authenticate_user,
    "validate_input": validate_input,
    "encrypt_data": encrypt_data,
    "decrypt_data": decrypt_data,
    "check_rate_limit": check_rate_limit,
    "block_suspicious_ip": block_suspicious_ip,
    
    # Video processing commands
    "process_video": process_video,
    "extract_audio": extract_audio,
    "generate_thumbnail": generate_thumbnail,
    "analyze_frames": analyze_frames,
    "compress_video": compress_video,
    
    # Database commands
    "create_user": create_user,
    "update_user": update_user,
    "delete_user": delete_user,
    "get_user_by_id": get_user_by_id,
    "get_all_users": get_all_users,
    
    # Utility functions
    "sanitize_input": sanitize_input,
    "validate_email": validate_email,
    "hash_password": hash_password,
    "generate_token": generate_token,
    "log_security_event": log_security_event
}

# Command executor
def execute_command(command_name: str, *args, **kwargs) -> Any:
    """Execute command by name"""
    if command_name not in COMMAND_REGISTRY:
        raise ValueError(f"Unknown command: {command_name}")
    
    command_func = COMMAND_REGISTRY[command_name]
    return command_func(*args, **kwargs)

# Example usage
async def main():
    """Example usage of named exports"""
    print("🚀 Named Exports Example")
    
    # Initialize configurations
    security_config = SecurityConfig()
    video_config = VideoConfig()
    database_config = DatabaseConfig()
    
    # Example 1: Security commands
    print("\n🔒 Security Commands:")
    
    # Authenticate user
    auth_result = authenticate_user("admin", "password", security_config)
    print(f"   Authentication: {'✅' if auth_result['success'] else '❌'}")
    
    # Validate input
    input_data = {
        "email": "user@example.com",
        "password": "strongpassword123",
        "content": "Normal content"
    }
    validation_result = validate_input(input_data)
    print(f"   Input validation: {validation_result}")
    
    # Encrypt data
    encrypted = encrypt_data("sensitive data", security_config)
    decrypted = decrypt_data(encrypted, security_config)
    print(f"   Encryption/Decryption: {'✅' if decrypted == 'sensitive data' else '❌'}")
    
    # Example 2: Video processing commands
    print("\n🎥 Video Processing Commands:")
    
    video_path = "sample_video.mp4"
    
    # Process video
    process_result = process_video(video_path, video_config)
    print(f"   Video processing: {'✅' if process_result['success'] else '❌'}")
    
    # Extract audio
    audio_result = extract_audio(video_path)
    print(f"   Audio extraction: {'✅' if audio_result['success'] else '❌'}")
    
    # Generate thumbnail
    thumbnail_result = generate_thumbnail(video_path)
    print(f"   Thumbnail generation: {'✅' if thumbnail_result['success'] else '❌'}")
    
    # Example 3: Database commands
    print("\n🗄️ Database Commands:")
    
    # Create user
    user_data = {"username": "newuser", "email": "newuser@example.com"}
    create_result = create_user(user_data, database_config)
    print(f"   User creation: {'✅' if create_result['success'] else '❌'}")
    
    # Get user
    user = get_user_by_id(1, database_config)
    print(f"   Get user: {'✅' if user else '❌'}")
    
    # Example 4: Utility functions
    print("\n🔧 Utility Functions:")
    
    # Sanitize input
    sanitized = sanitize_input("<script>alert('xss')</script>Hello World")
    print(f"   Input sanitization: {sanitized}")
    
    # Validate email
    is_valid_email = validate_email("test@example.com")
    print(f"   Email validation: {'✅' if is_valid_email else '❌'}")
    
    # Hash password
    hashed = hash_password("mypassword")
    print(f"   Password hashing: {hashed[:20]}...")
    
    # Example 5: Command registry
    print("\n📋 Command Registry:")
    
    # Execute command by name
    result = execute_command("validate_email", "test@example.com")
    print(f"   Command execution: {'✅' if result else '❌'}")
    
    # List available commands
    print(f"   Available commands: {len(COMMAND_REGISTRY)}")
    for category in ["Security", "Video Processing", "Database", "Utility"]:
        commands = [cmd for cmd in COMMAND_REGISTRY.keys() if cmd.startswith(category.lower().replace(" ", "_"))]
        print(f"     {category}: {len(commands)} commands")
    
    print("\n🎯 Named exports example completed!")

if __name__ == "__main__":
    asyncio.run(main()) 