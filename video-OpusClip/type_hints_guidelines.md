# Type Hints and Pydantic v2 Validation Guidelines

## 🔍 Comprehensive Type Hints and Validation

### Core Principles
- **Type Safety**: All functions must have complete type hints
- **Validation**: Use Pydantic v2 models for structured data validation
- **Documentation**: Type hints serve as inline documentation
- **IDE Support**: Enable better autocomplete and error detection

## 📋 Type Hint Categories

### 1. Basic Type Hints
```python
from typing import Dict, List, Any, Optional, Union, Literal, Callable
from typing_extensions import Annotated

# Basic types
def process_video(video_path: str) -> Dict[str, Any]:
    """Process video with type hints"""
    pass

# Optional parameters
def validate_user(user_id: Optional[int] = None) -> bool:
    """Validate user with optional parameter"""
    pass

# Union types
def get_data(source: Union[str, Path, bytes]) -> Dict[str, Any]:
    """Get data from multiple source types"""
    pass

# Literal types
def set_quality(quality: Literal["low", "medium", "high"]) -> None:
    """Set quality with literal values"""
    pass

# Annotated types with constraints
def validate_email(email: Annotated[str, Field(regex=r"^[^@]+@[^@]+\.[^@]+$")]) -> bool:
    """Validate email with regex constraint"""
    pass
```

### 2. Complex Type Hints
```python
from typing import TypedDict, Protocol, Generic, TypeVar

# TypedDict for structured data
class VideoMetadata(TypedDict):
    duration: float
    frame_count: int
    resolution: str
    file_size: int
    codec: str
    bitrate: Optional[float]

# Protocol for structural typing
class VideoProcessor(Protocol):
    def process(self, video_path: str) -> Dict[str, Any]: ...
    def validate(self, metadata: VideoMetadata) -> bool: ...

# Generic types
T = TypeVar('T')
class Result(Generic[T]):
    def __init__(self, data: T, success: bool) -> None:
        self.data = data
        self.success = success

# Callable types
VideoCallback = Callable[[str, VideoMetadata], Dict[str, Any]]
```

## 🛡️ Pydantic v2 Models

### 1. Basic Model Structure
```python
from pydantic import BaseModel, Field, validator, root_validator, ConfigDict

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
```

### 2. Enum Integration
```python
from enum import Enum

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

class VideoConfig(BaseModel):
    """Video processing configuration with validation"""
    model_config = ConfigDict(strict=True, extra="forbid")
    
    max_file_size: Annotated[int, Field(ge=1024*1024, le=1024*1024*1024)]
    supported_formats: List[VideoFormat]
    output_quality: ProcessingQuality
    enable_compression: bool = True
    max_duration: Annotated[int, Field(ge=1, le=3600)] = 300
    max_resolution: Annotated[str, Field(regex=r"^\d+x\d+$")] = "1920x1080"
    
    @validator("supported_formats")
    def validate_formats(cls, v: List[VideoFormat]) -> List[VideoFormat]:
        if not v:
            raise ValueError("At least one video format must be supported")
        return v
```

### 3. Complex Validation
```python
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
```

## 🔧 Function Signatures with Type Hints

### 1. CPU-bound Functions
```python
def validate_password_strength(password: str) -> Dict[str, Union[bool, int, str, List[str]]]:
    """Validate password strength with comprehensive type hints"""
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

def hash_password(password: str, salt: Optional[str] = None) -> Dict[str, str]:
    """Hash password with type hints"""
    if salt is None:
        salt = hashlib.sha256(str(time.time()).encode()).hexdigest()[:16]
    
    hashed: str = password + salt
    for _ in range(100000):
        hashed = hashlib.sha256(hashed.encode()).hexdigest()
    
    return {
        "hash": hashed,
        "salt": salt
    }

def calculate_video_metrics(metadata: VideoMetadata) -> Dict[str, Union[float, str, int]]:
    """Calculate video metrics with type hints"""
    duration: float = metadata.get("duration", 0.0)
    frame_count: int = metadata.get("frame_count", 0)
    resolution: str = metadata.get("resolution", "1920x1080")
    file_size: int = metadata.get("file_size", 0)
    
    fps: float = frame_count / duration if duration > 0 else 0.0
    bitrate: float = (file_size * 8) / duration if duration > 0 else 0.0
    
    return {
        "fps": round(fps, 2),
        "bitrate": round(bitrate, 2),
        "resolution": resolution,
        "aspect_ratio": _calculate_aspect_ratio(resolution),
        "compression_ratio": _calculate_compression_ratio(metadata)
    }
```

### 2. I/O-bound Functions
```python
async def fetch_video_metadata(video_id: str) -> Dict[str, Union[bool, str, VideoMetadata]]:
    """Fetch video metadata with type hints"""
    async with aiohttp.ClientSession() as session:
        url: str = f"https://api.example.com/videos/{video_id}"
        
        try:
            async with session.get(url) as response:
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

async def save_video_to_storage(video_path: str, video_data: bytes) -> Dict[str, Union[bool, str, int]]:
    """Save video to storage with type hints"""
    try:
        async with aiofiles.open(video_path, 'wb') as f:
            await f.write(video_data)
        
        return {
            "success": True,
            "path": video_path,
            "size": len(video_data)
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

async def execute_database_query(
    query: str, 
    *args: Any, 
    operation: DatabaseOperation = DatabaseOperation.READ
) -> Dict[str, Union[bool, str, List[Dict[str, Any]], int]]:
    """Execute database query with type hints"""
    if not pool:
        raise RuntimeError("Database pool not initialized")
    
    try:
        async with pool.acquire() as conn:
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
```

### 3. Class Methods with Type Hints
```python
class SecurityOperations:
    """Security operations with type hints and validation"""
    
    def __init__(self, config: SecurityConfig) -> None:
        self.config = config
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate security configuration"""
        if not self.config:
            raise ValueError("Security configuration is required")
    
    def validate_email_format(self, email: str) -> bool:
        """Validate email format with type hints"""
        import re
        pattern: str = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
    
    def sanitize_input(self, text: str) -> str:
        """Sanitize input with type hints"""
        import re
        
        sanitized: str = re.sub(r'[<>"\']', '', text)
        sanitized = re.sub(r'<script.*?</script>', '', sanitized, flags=re.IGNORECASE)
        sanitized = re.sub(r'(\b(union|select|insert|update|delete|drop|create|alter)\b)', '', sanitized, flags=re.IGNORECASE)
        
        return sanitized.strip()

class VideoOperations:
    """Video operations with type hints and validation"""
    
    def __init__(self, config: VideoConfig) -> None:
        self.config = config
        self._validate_config()
    
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
```

## 🎯 Decorators with Type Hints

### 1. Validation Decorators
```python
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
```

### 2. Type-Safe Decorators
```python
from functools import wraps
from typing import TypeVar, ParamSpec

T = TypeVar('T')
P = ParamSpec('P')

def type_safe(func: Callable[P, T]) -> Callable[P, T]:
    """Decorator to ensure type safety"""
    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        # Type checking logic here
        return func(*args, **kwargs)
    return wrapper
```

## 📊 Best Practices

### 1. **Comprehensive Type Coverage**
```python
# ✅ Good: Complete type hints
def process_video(
    video_path: str,
    quality: ProcessingQuality,
    output_format: VideoFormat,
    metadata: Optional[VideoMetadata] = None
) -> Dict[str, Union[bool, str, Dict[str, Any]]]:
    """Process video with complete type hints"""
    pass

# ❌ Bad: Missing type hints
def process_video(video_path, quality, output_format, metadata=None):
    """Process video without type hints"""
    pass
```

### 2. **Use Annotated for Constraints**
```python
# ✅ Good: Annotated with constraints
def validate_email(email: Annotated[str, Field(regex=r"^[^@]+@[^@]+\.[^@]+$")]) -> bool:
    """Validate email with regex constraint"""
    pass

# ❌ Bad: No constraints
def validate_email(email: str) -> bool:
    """Validate email without constraints"""
    pass
```

### 3. **Structured Validation with Pydantic**
```python
# ✅ Good: Pydantic model with validation
class UserRequest(BaseModel):
    username: Annotated[str, Field(min_length=3, max_length=50)]
    email: Annotated[str, Field(regex=r"^[^@]+@[^@]+\.[^@]+$")]
    
    @validator("username")
    def validate_username(cls, v: str) -> str:
        if not v.isalnum():
            raise ValueError("Username must be alphanumeric")
        return v

# ❌ Bad: Manual validation
def create_user(username: str, email: str) -> Dict[str, Any]:
    if len(username) < 3:
        raise ValueError("Username too short")
    # ... more manual validation
```

### 4. **Return Type Specifications**
```python
# ✅ Good: Specific return types
def get_user_data(user_id: int) -> Dict[str, Union[str, int, bool]]:
    """Get user data with specific return type"""
    return {
        "id": user_id,
        "name": "John Doe",
        "active": True
    }

# ❌ Bad: Generic return type
def get_user_data(user_id: int) -> Any:
    """Get user data with generic return type"""
    return {"id": user_id, "name": "John Doe"}
```

## 🚨 Common Mistakes

### 1. **Missing Type Hints**
```python
# ❌ Wrong: No type hints
def process_data(data):
    return data.upper()

# ✅ Correct: With type hints
def process_data(data: str) -> str:
    return data.upper()
```

### 2. **Incorrect Union Types**
```python
# ❌ Wrong: Incorrect union syntax
def get_value() -> str | int:
    return "hello"

# ✅ Correct: Proper union syntax
def get_value() -> Union[str, int]:
    return "hello"
```

### 3. **Missing Validation**
```python
# ❌ Wrong: No validation
def create_user(username: str, email: str) -> Dict[str, Any]:
    return {"username": username, "email": email}

# ✅ Correct: With validation
def create_user(request: UserRegistrationRequest) -> Dict[str, Any]:
    return {"username": request.username, "email": request.email}
```

### 4. **Inconsistent Return Types**
```python
# ❌ Wrong: Inconsistent return types
def get_user(user_id: int) -> Dict[str, Any]:
    if user_id < 0:
        return None  # Inconsistent!
    return {"id": user_id, "name": "John"}

# ✅ Correct: Consistent return types
def get_user(user_id: int) -> Optional[Dict[str, Any]]:
    if user_id < 0:
        return None
    return {"id": user_id, "name": "John"}
```

## 📋 Implementation Checklist

### Type Hints
- [ ] All function parameters have type hints
- [ ] All function return values have type hints
- [ ] Complex types use proper imports (Union, Optional, etc.)
- [ ] Generic types are properly defined
- [ ] Callable types are specified for callbacks

### Pydantic Validation
- [ ] Structured data uses Pydantic models
- [ ] Field constraints are properly defined
- [ ] Custom validators are implemented
- [ ] Root validators for cross-field validation
- [ ] Enum types are used for constrained values

### Error Handling
- [ ] Validation errors are properly caught
- [ ] Type errors are handled gracefully
- [ ] Meaningful error messages are provided
- [ ] Error types are consistent

### Documentation
- [ ] Type hints serve as documentation
- [ ] Complex types are explained
- [ ] Validation rules are documented
- [ ] Examples are provided

Following these guidelines ensures type safety, better IDE support, and more maintainable code in the Video-OpusClip system. 