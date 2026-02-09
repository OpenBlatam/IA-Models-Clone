#!/usr/bin/env python3
"""
Type Hints and Pydantic v2 Validation for Video-OpusClip
Demonstrates comprehensive type hints and structured validation
"""

import asyncio
import time
import json
import hashlib
import re
from typing import Dict, List, Any, Optional, Union, Literal, TypedDict, Callable
from typing_extensions import Annotated
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum

from pydantic import BaseModel, Field, validator, root_validator, ConfigDict
from pydantic.types import StrictStr, StrictInt, StrictFloat, StrictBool
import aiohttp
import aiofiles
import aioredis
import asyncpg

# Enums for type safety
class SecurityLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class VideoFormat(str, Enum):
    MP4 = "mp4"
    AVI = "avi"
    MOV = "mov"
    MKV = "mkv"
    WEBM = "webm"

class ProcessingQuality(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ULTRA = "ultra"

class DatabaseOperation(str, Enum):
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"

# TypedDict for structured data
class VideoMetadata(TypedDict):
    duration: float
    frame_count: int
    resolution: str
    file_size: int
    codec: str
    bitrate: Optional[float]

class UserPermissions(TypedDict):
    can_upload: bool
    can_process: bool
    can_delete: bool
    max_file_size: int
    allowed_formats: List[VideoFormat]

# Pydantic v2 Models for validation
class SecurityConfig(BaseModel):
    """Security configuration with validation"""
    model_config = ConfigDict(strict=True, extra="forbid")
    
    secret_key: Annotated[str, Field(min_length=32, max_length=256)]
    encryption_key: Annotated[str, Field(min_length=32, max_length=256)]
    salt: Annotated[str, Field(min_length=16, max_length=64)]
    max_login_attempts: Annotated[int, Field(ge=1, le=10)]
    lockout_duration: Annotated[int, Field(ge=60, le=3600)]
    rate_limit_requests: Annotated[int, Field(ge=10, le=1000)]
    rate_limit_window: Annotated[int, Field(ge=60, le=3600)]
    jwt_expiration_minutes: Annotated[int, Field(ge=5, le=1440)]
    
    @validator("secret_key", "encryption_key")
    def validate_key_strength(cls, v: str) -> str:
        if len(set(v)) < 20:
            raise ValueError("Key must contain at least 20 unique characters")
        return v

class VideoConfig(BaseModel):
    """Video processing configuration with validation"""
    model_config = ConfigDict(strict=True, extra="forbid")
    
    max_file_size: Annotated[int, Field(ge=1024*1024, le=1024*1024*1024)]  # 1MB to 1GB
    supported_formats: List[VideoFormat]
    output_quality: ProcessingQuality
    enable_compression: bool = True
    max_duration: Annotated[int, Field(ge=1, le=3600)] = 300  # 1 second to 1 hour
    max_resolution: Annotated[str, Field(regex=r"^\d+x\d+$")] = "1920x1080"
    
    @validator("supported_formats")
    def validate_formats(cls, v: List[VideoFormat]) -> List[VideoFormat]:
        if not v:
            raise ValueError("At least one video format must be supported")
        return v
    
    @validator("max_resolution")
    def validate_resolution(cls, v: str) -> str:
        width, height = map(int, v.split('x'))
        if width > 7680 or height > 4320:  # 8K limit
            raise ValueError("Resolution too high")
        return v

class DatabaseConfig(BaseModel):
    """Database configuration with validation"""
    model_config = ConfigDict(strict=True, extra="forbid")
    
    host: Annotated[str, Field(min_length=1, max_length=255)]
    port: Annotated[int, Field(ge=1, le=65535)]
    database: Annotated[str, Field(min_length=1, max_length=64)]
    username: Annotated[str, Field(min_length=1, max_length=64)]
    password: Annotated[str, Field(min_length=1, max_length=128)]
    max_connections: Annotated[int, Field(ge=1, le=100)]
    connection_timeout: Annotated[float, Field(ge=1.0, le=60.0)] = 30.0
    
    @validator("host")
    def validate_host(cls, v: str) -> str:
        if not re.match(r"^[a-zA-Z0-9.-]+$", v):
            raise ValueError("Invalid host format")
        return v

class UserRegistrationRequest(BaseModel):
    """User registration request with validation"""
    model_config = ConfigDict(strict=True, extra="forbid")
    
    username: Annotated[str, Field(min_length=3, max_length=50, regex=r"^[a-zA-Z0-9_]+$")]
    email: Annotated[str, Field(regex=r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")]
    password: Annotated[str, Field(min_length=8, max_length=128)]
    confirm_password: Annotated[str, Field(min_length=8, max_length=128)]
    
    @validator("password")
    def validate_password_strength(cls, v: str) -> str:
        if not re.search(r"[A-Z]", v):
            raise ValueError("Password must contain uppercase letter")
        if not re.search(r"[a-z]", v):
            raise ValueError("Password must contain lowercase letter")
        if not re.search(r"\d", v):
            raise ValueError("Password must contain digit")
        if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", v):
            raise ValueError("Password must contain special character")
        return v
    
    @root_validator
    def validate_passwords_match(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        password = values.get("password")
        confirm_password = values.get("confirm_password")
        
        if password and confirm_password and password != confirm_password:
            raise ValueError("Passwords do not match")
        
        return values

class VideoProcessingRequest(BaseModel):
    """Video processing request with validation"""
    model_config = ConfigDict(strict=True, extra="forbid")
    
    video_path: Annotated[str, Field(min_length=1, max_length=500)]
    output_format: VideoFormat
    quality: ProcessingQuality
    processing_options: Dict[str, Any] = Field(default_factory=dict)
    metadata: Optional[VideoMetadata] = None
    
    @validator("video_path")
    def validate_video_path(cls, v: str) -> str:
        if not Path(v).suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
            raise ValueError("Invalid video file format")
        return v
    
    @validator("processing_options")
    def validate_processing_options(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        allowed_options = {"crop", "resize", "filter", "codec", "bitrate"}
        invalid_options = set(v.keys()) - allowed_options
        if invalid_options:
            raise ValueError(f"Invalid processing options: {invalid_options}")
        return v

class AuthenticationRequest(BaseModel):
    """Authentication request with validation"""
    model_config = ConfigDict(strict=True, extra="forbid")
    
    username: Annotated[str, Field(min_length=1, max_length=50)]
    password: Annotated[str, Field(min_length=1, max_length=128)]
    client_ip: Annotated[str, Field(regex=r"^(\d{1,3}\.){3}\d{1,3}$")]
    user_agent: Optional[str] = Field(None, max_length=500)
    remember_me: bool = False
    
    @validator("client_ip")
    def validate_ip_address(cls, v: str) -> str:
        parts = v.split('.')
        for part in parts:
            if not 0 <= int(part) <= 255:
                raise ValueError("Invalid IP address")
        return v

# Functions with comprehensive type hints
class SecurityOperations:
    """Security operations with type hints and validation"""
    
    def __init__(self, config: SecurityConfig) -> None:
        self.config = config
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate security configuration"""
        if not self.config:
            raise ValueError("Security configuration is required")
    
    def validate_password_strength(self, password: str) -> Dict[str, Union[bool, int, str, List[str]]]:
        """Validate password strength with type hints"""
        import re
        
        checks: Dict[str, bool] = {
            "length": len(password) >= 8,
            "uppercase": bool(re.search(r'[A-Z]', password)),
            "lowercase": bool(re.search(r'[a-z]', password)),
            "digit": bool(re.search(r'\d', password)),
            "special": bool(re.search(r'[!@#$%^&*(),.?":{}|<>]', password))
        }
        
        failed_checks: List[str] = [name for name, passed in checks.items() if not passed]
        score: int = sum(checks.values())
        strength: str = "weak" if score < 3 else "medium" if score < 5 else "strong"
        
        return {
            "valid": score >= 4,
            "score": score,
            "strength": strength,
            "failed_checks": failed_checks
        }
    
    def hash_password(self, password: str, salt: Optional[str] = None) -> Dict[str, str]:
        """Hash password with type hints"""
        if salt is None:
            salt = hashlib.sha256(str(time.time()).encode()).hexdigest()[:16]
        
        # Multiple rounds for security
        hashed: str = password + salt
        for _ in range(100000):
            hashed = hashlib.sha256(hashed.encode()).hexdigest()
        
        return {
            "hash": hashed,
            "salt": salt
        }
    
    def verify_password(self, password: str, hashed_password: str, salt: str) -> bool:
        """Verify password with type hints"""
        test_hash: str = password + salt
        for _ in range(100000):
            test_hash = hashlib.sha256(test_hash.encode()).hexdigest()
        
        return test_hash == hashed_password
    
    def validate_email_format(self, email: str) -> bool:
        """Validate email format with type hints"""
        import re
        pattern: str = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
    
    def sanitize_input(self, text: str) -> str:
        """Sanitize input with type hints"""
        import re
        
        # Remove dangerous characters
        sanitized: str = re.sub(r'[<>"\']', '', text)
        # Remove script tags
        sanitized = re.sub(r'<script.*?</script>', '', sanitized, flags=re.IGNORECASE)
        # Remove SQL injection patterns
        sanitized = re.sub(r'(\b(union|select|insert|update|delete|drop|create|alter)\b)', '', sanitized, flags=re.IGNORECASE)
        
        return sanitized.strip()

class VideoOperations:
    """Video operations with type hints and validation"""
    
    def __init__(self, config: VideoConfig) -> None:
        self.config = config
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate video configuration"""
        if not self.config:
            raise ValueError("Video configuration is required")
    
    def calculate_video_metrics(self, metadata: VideoMetadata) -> Dict[str, Union[float, str, int]]:
        """Calculate video metrics with type hints"""
        duration: float = metadata.get("duration", 0.0)
        frame_count: int = metadata.get("frame_count", 0)
        resolution: str = metadata.get("resolution", "1920x1080")
        file_size: int = metadata.get("file_size", 0)
        
        # Calculate metrics
        fps: float = frame_count / duration if duration > 0 else 0.0
        bitrate: float = (file_size * 8) / duration if duration > 0 else 0.0
        
        return {
            "fps": round(fps, 2),
            "bitrate": round(bitrate, 2),
            "resolution": resolution,
            "aspect_ratio": self._calculate_aspect_ratio(resolution),
            "compression_ratio": self._calculate_compression_ratio(metadata)
        }
    
    def _calculate_aspect_ratio(self, resolution: str) -> str:
        """Calculate aspect ratio with type hints"""
        try:
            width, height = map(int, resolution.split('x'))
            gcd: int = self._gcd(width, height)
            return f"{width//gcd}:{height//gcd}"
        except (ValueError, ZeroDivisionError):
            return "16:9"
    
    def _gcd(self, a: int, b: int) -> int:
        """Calculate GCD with type hints"""
        while b:
            a, b = b, a % b
        return a
    
    def _calculate_compression_ratio(self, metadata: VideoMetadata) -> float:
        """Calculate compression ratio with type hints"""
        original_size: int = metadata.get("original_size", 0)
        compressed_size: int = metadata.get("file_size", 0)
        
        if original_size > 0:
            return round(compressed_size / original_size, 3)
        return 1.0
    
    def validate_video_file(self, file_path: Path) -> Dict[str, Union[bool, str, int]]:
        """Validate video file with type hints"""
        if not file_path.exists():
            return {"valid": False, "error": "File does not exist"}
        
        file_size: int = file_path.stat().st_size
        if file_size > self.config.max_file_size:
            return {
                "valid": False,
                "error": f"File too large: {file_size} bytes (max: {self.config.max_file_size})"
            }
        
        file_extension: str = file_path.suffix.lower()
        if file_extension not in [f".{fmt.value}" for fmt in self.config.supported_formats]:
            return {
                "valid": False,
                "error": f"Unsupported format: {file_extension}"
            }
        
        return {
            "valid": True,
            "file_size": file_size,
            "format": file_extension
        }

class DatabaseOperations:
    """Database operations with type hints and validation"""
    
    def __init__(self, config: DatabaseConfig) -> None:
        self.config = config
        self._validate_config()
        self.pool: Optional[asyncpg.Pool] = None
    
    def _validate_config(self) -> None:
        """Validate database configuration"""
        if not self.config:
            raise ValueError("Database configuration is required")
    
    async def initialize_pool(self) -> None:
        """Initialize database connection pool"""
        self.pool = await asyncpg.create_pool(
            host=self.config.host,
            port=self.config.port,
            database=self.config.database,
            user=self.config.username,
            password=self.config.password,
            max_size=self.config.max_connections,
            command_timeout=self.config.connection_timeout
        )
    
    async def close_pool(self) -> None:
        """Close database connection pool"""
        if self.pool:
            await self.pool.close()
    
    async def execute_query(
        self, 
        query: str, 
        *args: Any, 
        operation: DatabaseOperation = DatabaseOperation.READ
    ) -> Dict[str, Union[bool, str, List[Dict[str, Any]], int]]:
        """Execute database query with type hints"""
        if not self.pool:
            raise RuntimeError("Database pool not initialized")
        
        try:
            async with self.pool.acquire() as conn:
                if operation == DatabaseOperation.READ:
                    rows = await conn.fetch(query, *args)
                    return {
                        "success": True,
                        "data": [dict(row) for row in rows],
                        "count": len(rows)
                    }
                else:
                    result = await conn.execute(query, *args)
                    return {
                        "success": True,
                        "result": result,
                        "operation": operation.value
                    }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def create_user(self, user_data: UserRegistrationRequest) -> Dict[str, Union[bool, str, Dict[str, Any]]]:
        """Create user with type hints"""
        query: str = """
            INSERT INTO users (username, email, password_hash, created_at)
            VALUES ($1, $2, $3, $4)
            RETURNING id, username, email, created_at
        """
        
        # Hash password
        security_ops = SecurityOperations(SecurityConfig(
            secret_key="temp_key" * 8,
            encryption_key="temp_enc_key" * 8,
            salt="temp_salt" * 4,
            max_login_attempts=5,
            lockout_duration=900,
            rate_limit_requests=100,
            rate_limit_window=60,
            jwt_expiration_minutes=30
        ))
        
        hashed_password: Dict[str, str] = security_ops.hash_password(user_data.password)
        
        return await self.execute_query(
            query,
            user_data.username,
            user_data.email,
            hashed_password["hash"],
            datetime.utcnow(),
            operation=DatabaseOperation.CREATE
        )

# Async functions with type hints
class AsyncVideoProcessor:
    """Async video processor with type hints and validation"""
    
    def __init__(self, video_config: VideoConfig) -> None:
        self.config = video_config
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def initialize_session(self) -> None:
        """Initialize HTTP session"""
        self.session = aiohttp.ClientSession()
    
    async def close_session(self) -> None:
        """Close HTTP session"""
        if self.session:
            await self.session.close()
    
    async def fetch_video_metadata(self, video_id: str) -> Dict[str, Union[bool, str, VideoMetadata]]:
        """Fetch video metadata with type hints"""
        if not self.session:
            await self.initialize_session()
        
        url: str = f"https://api.example.com/videos/{video_id}"
        
        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    data: Dict[str, Any] = await response.json()
                    return {
                        "success": True,
                        "data": VideoMetadata(**data)
                    }
                else:
                    return {
                        "success": False,
                        "error": f"HTTP {response.status}"
                    }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def process_video(
        self, 
        request: VideoProcessingRequest
    ) -> Dict[str, Union[bool, str, Dict[str, Any]]]:
        """Process video with type hints"""
        # Validate request
        try:
            validated_request = VideoProcessingRequest(**request.dict())
        except Exception as e:
            return {
                "success": False,
                "error": f"Validation error: {str(e)}"
            }
        
        # Validate video file
        video_path = Path(validated_request.video_path)
        validation_result = self._validate_video_file(video_path)
        
        if not validation_result["valid"]:
            return {
                "success": False,
                "error": validation_result["error"]
            }
        
        # Simulate processing
        processing_time: float = 30.5
        output_path: str = f"processed_{video_path.name}"
        
        return {
            "success": True,
            "output_path": output_path,
            "processing_time": processing_time,
            "quality": validated_request.quality.value,
            "format": validated_request.output_format.value
        }
    
    def _validate_video_file(self, file_path: Path) -> Dict[str, Union[bool, str]]:
        """Validate video file"""
        if not file_path.exists():
            return {"valid": False, "error": "File does not exist"}
        
        file_size: int = file_path.stat().st_size
        if file_size > self.config.max_file_size:
            return {
                "valid": False,
                "error": f"File too large: {file_size} bytes"
            }
        
        return {"valid": True}

# Function decorators with type hints
def validate_inputs(*validators: Callable[[Any], bool]) -> Callable:
    """Decorator to validate function inputs with type hints"""
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Validate arguments
            for validator_func in validators:
                if not validator_func(args[0] if args else None):
                    raise ValueError(f"Validation failed for {func.__name__}")
            return func(*args, **kwargs)
        return wrapper
    return decorator

def log_execution_time(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to log execution time with type hints"""
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time: float = time.time()
        result: Any = func(*args, **kwargs)
        execution_time: float = time.time() - start_time
        print(f"{func.__name__} executed in {execution_time:.4f} seconds")
        return result
    return wrapper

# Example usage
async def main() -> None:
    """Example usage of type hints and validation"""
    print("🔍 Type Hints and Validation Example")
    
    # Initialize configurations with validation
    try:
        security_config = SecurityConfig(
            secret_key="my-super-secret-key-that-is-very-long-and-secure",
            encryption_key="my-encryption-key-that-is-also-very-long-and-secure",
            salt="my-salt-that-is-long-enough",
            max_login_attempts=5,
            lockout_duration=900,
            rate_limit_requests=100,
            rate_limit_window=60,
            jwt_expiration_minutes=30
        )
        print("✅ Security config validated")
    except Exception as e:
        print(f"❌ Security config validation failed: {e}")
        return
    
    try:
        video_config = VideoConfig(
            max_file_size=100 * 1024 * 1024,  # 100MB
            supported_formats=[VideoFormat.MP4, VideoFormat.AVI, VideoFormat.MOV],
            output_quality=ProcessingQuality.HIGH,
            enable_compression=True,
            max_duration=300,
            max_resolution="1920x1080"
        )
        print("✅ Video config validated")
    except Exception as e:
        print(f"❌ Video config validation failed: {e}")
        return
    
    try:
        database_config = DatabaseConfig(
            host="localhost",
            port=5432,
            database="video_db",
            username="postgres",
            password="password",
            max_connections=10,
            connection_timeout=30.0
        )
        print("✅ Database config validated")
    except Exception as e:
        print(f"❌ Database config validation failed: {e}")
        return
    
    # Test user registration validation
    print("\n👤 User Registration Validation:")
    try:
        user_request = UserRegistrationRequest(
            username="testuser",
            email="test@example.com",
            password="StrongPass123!",
            confirm_password="StrongPass123!"
        )
        print("✅ User registration request validated")
    except Exception as e:
        print(f"❌ User registration validation failed: {e}")
    
    # Test video processing validation
    print("\n🎥 Video Processing Validation:")
    try:
        video_request = VideoProcessingRequest(
            video_path="sample_video.mp4",
            output_format=VideoFormat.MP4,
            quality=ProcessingQuality.HIGH,
            processing_options={"crop": "16:9", "resize": "1920x1080"}
        )
        print("✅ Video processing request validated")
    except Exception as e:
        print(f"❌ Video processing validation failed: {e}")
    
    # Test security operations
    print("\n🔒 Security Operations:")
    security_ops = SecurityOperations(security_config)
    
    # Password validation
    password_result = security_ops.validate_password_strength("WeakPass")
    print(f"   Password validation: {'✅' if password_result['valid'] else '❌'}")
    print(f"   Strength: {password_result['strength']}")
    print(f"   Failed checks: {password_result['failed_checks']}")
    
    # Email validation
    email_valid = security_ops.validate_email_format("user@example.com")
    print(f"   Email validation: {'✅' if email_valid else '❌'}")
    
    # Input sanitization
    sanitized = security_ops.sanitize_input("<script>alert('xss')</script>Hello World")
    print(f"   Input sanitization: {sanitized}")
    
    # Test video operations
    print("\n🎬 Video Operations:")
    video_ops = VideoOperations(video_config)
    
    # Video metrics calculation
    metadata: VideoMetadata = {
        "duration": 120.5,
        "frame_count": 3600,
        "resolution": "1920x1080",
        "file_size": 50 * 1024 * 1024,  # 50MB
        "codec": "h264",
        "bitrate": 5000.0
    }
    
    metrics = video_ops.calculate_video_metrics(metadata)
    print(f"   Video metrics: FPS={metrics['fps']}, Bitrate={metrics['bitrate']}")
    print(f"   Aspect ratio: {metrics['aspect_ratio']}")
    
    # Test async video processor
    print("\n⚡ Async Video Processor:")
    async_processor = AsyncVideoProcessor(video_config)
    
    # Simulate video processing
    processing_result = await async_processor.process_video(video_request)
    print(f"   Processing result: {'✅' if processing_result['success'] else '❌'}")
    if processing_result['success']:
        print(f"   Output path: {processing_result['output_path']}")
        print(f"   Processing time: {processing_result['processing_time']}s")
    
    print("\n🎯 Type hints and validation example completed!")

if __name__ == "__main__":
    asyncio.run(main()) 