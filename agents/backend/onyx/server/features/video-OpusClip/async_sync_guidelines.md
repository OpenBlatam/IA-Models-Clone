# Async/Sync Guidelines for Video-OpusClip

## 🔄 When to Use `def` vs `async def`

### 💻 Use `def` for CPU-bound Operations

**CPU-bound operations** are computations that primarily use the CPU and don't involve waiting for external resources.

#### Examples of CPU-bound operations:

```python
# ✅ Use `def` for these operations

def validate_password_strength(password: str) -> Dict[str, Any]:
    """CPU-bound password validation"""
    import re
    
    checks = {
        "length": len(password) >= 8,
        "uppercase": bool(re.search(r'[A-Z]', password)),
        "lowercase": bool(re.search(r'[a-z]', password)),
        "digit": bool(re.search(r'\d', password)),
        "special": bool(re.search(r'[!@#$%^&*(),.?":{}|<>]', password))
    }
    
    score = sum(checks.values())
    strength = "weak" if score < 3 else "medium" if score < 5 else "strong"
    
    return {
        "valid": score >= 4,
        "score": score,
        "strength": strength,
        "checks": checks
    }

def hash_password(password: str, salt: str = None) -> Dict[str, str]:
    """CPU-bound password hashing"""
    import hashlib
    
    if salt is None:
        salt = hashlib.sha256(str(time.time()).encode()).hexdigest()[:16]
    
    # Multiple rounds for security (CPU-intensive)
    hashed = password + salt
    for _ in range(100000):
        hashed = hashlib.sha256(hashed.encode()).hexdigest()
    
    return {"hash": hashed, "salt": salt}

def encrypt_data(data: str, key: bytes) -> str:
    """CPU-bound data encryption"""
    from cryptography.fernet import Fernet
    cipher = Fernet(key)
    return cipher.encrypt(data.encode()).decode()

def validate_email_format(email: str) -> bool:
    """CPU-bound email validation"""
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def sanitize_input(text: str) -> str:
    """CPU-bound input sanitization"""
    import re
    
    # Remove dangerous characters
    sanitized = re.sub(r'[<>"\']', '', text)
    # Remove script tags
    sanitized = re.sub(r'<script.*?</script>', '', sanitized, flags=re.IGNORECASE)
    # Remove SQL injection patterns
    sanitized = re.sub(r'(\b(union|select|insert|update|delete|drop|create|alter)\b)', '', sanitized, flags=re.IGNORECASE)
    
    return sanitized.strip()

def calculate_video_metrics(video_data: Dict[str, Any]) -> Dict[str, Any]:
    """CPU-bound video metrics calculation"""
    duration = video_data.get("duration", 0)
    frame_count = video_data.get("frame_count", 0)
    resolution = video_data.get("resolution", "1920x1080")
    
    # Complex calculations
    fps = frame_count / duration if duration > 0 else 0
    bitrate = (video_data.get("file_size", 0) * 8) / duration if duration > 0 else 0
    
    return {
        "fps": round(fps, 2),
        "bitrate": round(bitrate, 2),
        "resolution": resolution,
        "aspect_ratio": _calculate_aspect_ratio(resolution),
        "compression_ratio": _calculate_compression_ratio(video_data)
    }

def process_video_frames(frames: List[str]) -> Dict[str, Any]:
    """CPU-bound frame processing"""
    processed_frames = []
    total_pixels = 0
    
    for frame in frames:
        # CPU-intensive frame processing
        processed_frame = _process_single_frame(frame)
        processed_frames.append(processed_frame)
        total_pixels += processed_frame.get("pixel_count", 0)
    
    return {
        "processed_frames": len(processed_frames),
        "total_pixels": total_pixels,
        "frames": processed_frames
    }
```

### 🌐 Use `async def` for I/O-bound Operations

**I/O-bound operations** involve waiting for external resources like network requests, file system operations, or database queries.

#### Examples of I/O-bound operations:

```python
# ✅ Use `async def` for these operations

async def fetch_video_metadata(video_id: str) -> Dict[str, Any]:
    """I/O-bound: Fetch video metadata from API"""
    async with aiohttp.ClientSession() as session:
        url = f"https://api.example.com/videos/{video_id}"
        
        async with session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                return {"success": True, "data": data}
            else:
                return {"success": False, "error": f"HTTP {response.status}"}

async def save_video_to_storage(video_path: str, video_data: bytes) -> Dict[str, Any]:
    """I/O-bound: Save video to file system"""
    try:
        async with aiofiles.open(video_path, 'wb') as f:
            await f.write(video_data)
        
        return {
            "success": True,
            "path": video_path,
            "size": len(video_data)
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

async def load_video_from_storage(video_path: str) -> Dict[str, Any]:
    """I/O-bound: Load video from file system"""
    try:
        async with aiofiles.open(video_path, 'rb') as f:
            video_data = await f.read()
        
        return {
            "success": True,
            "data": video_data,
            "size": len(video_data)
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

async def cache_video_metadata(video_id: str, metadata: Dict[str, Any]) -> bool:
    """I/O-bound: Cache metadata in Redis"""
    redis_client = aioredis.from_url("redis://localhost")
    try:
        await redis_client.setex(
            f"video:{video_id}",
            3600,  # 1 hour TTL
            json.dumps(metadata)
        )
        return True
    except Exception:
        return False
    finally:
        await redis_client.close()

async def save_user_to_database(user_data: Dict[str, Any]) -> Dict[str, Any]:
    """I/O-bound: Save user to database"""
    pool = await asyncpg.create_pool("postgresql://user:password@localhost/db")
    
    try:
        async with pool.acquire() as conn:
            query = """
                INSERT INTO users (username, email, created_at)
                VALUES ($1, $2, $3)
                RETURNING id, username, email, created_at
            """
            row = await conn.fetchrow(
                query,
                user_data["username"],
                user_data["email"],
                datetime.utcnow()
            )
            
            return {"success": True, "user": dict(row)}
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        await pool.close()

async def upload_video_to_cloud(video_data: bytes, filename: str) -> Dict[str, Any]:
    """I/O-bound: Upload video to cloud storage"""
    async with aiohttp.ClientSession() as session:
        url = "https://storage.example.com/upload"
        
        async with session.post(url, data={"file": video_data}) as response:
            if response.status == 200:
                result = await response.json()
                return {
                    "success": True,
                    "url": result.get("url"),
                    "filename": filename
                }
            else:
                return {
                    "success": False,
                    "error": f"Upload failed: HTTP {response.status}"
                }

async def send_notification(user_id: int, message: str) -> bool:
    """I/O-bound: Send notification to user"""
    async with aiohttp.ClientSession() as session:
        url = "https://notifications.example.com/send"
        data = {
            "user_id": user_id,
            "message": message,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        async with session.post(url, json=data) as response:
            return response.status == 200
```

## 🔀 Hybrid Operations

**Hybrid operations** combine both CPU-bound and I/O-bound work. Use `async def` for the main function and call CPU-bound functions with `await asyncio.to_thread()`.

```python
# ✅ Hybrid operations combining CPU and I/O

async def process_and_save_video(video_data: bytes, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Hybrid: Process video (CPU) and save (I/O)"""
    # CPU-bound: Process video metadata (run in thread pool)
    processed_metadata = await asyncio.to_thread(
        calculate_video_metrics, metadata
    )
    
    # I/O-bound: Save to storage
    filename = f"video_{int(time.time())}.mp4"
    save_result = await save_video_to_storage(filename, video_data)
    
    if save_result["success"]:
        # I/O-bound: Cache metadata
        await cache_video_metadata(filename, processed_metadata)
        
        return {
            "success": True,
            "filename": filename,
            "metadata": processed_metadata,
            "size": save_result["size"]
        }
    else:
        return {
            "success": False,
            "error": save_result["error"]
        }

async def validate_and_register_user(user_data: Dict[str, Any]) -> Dict[str, Any]:
    """Hybrid: Validate user data (CPU) and save to database (I/O)"""
    # CPU-bound: Validate input (run in thread pool)
    email_valid = await asyncio.to_thread(
        validate_email_format, user_data["email"]
    )
    if not email_valid:
        return {"success": False, "error": "Invalid email format"}
    
    password_validation = await asyncio.to_thread(
        validate_password_strength, user_data["password"]
    )
    if not password_validation["valid"]:
        return {"success": False, "error": "Password too weak"}
    
    # CPU-bound: Hash password (run in thread pool)
    hashed_password = await asyncio.to_thread(
        hash_password, user_data["password"]
    )
    
    # I/O-bound: Save to database
    sanitized_data = {
        "username": await asyncio.to_thread(sanitize_input, user_data["username"]),
        "email": user_data["email"],
        "password_hash": hashed_password["hash"],
        "password_salt": hashed_password["salt"]
    }
    
    db_result = await save_user_to_database(sanitized_data)
    
    if db_result["success"]:
        # I/O-bound: Send welcome notification
        await send_notification(
            db_result["user"]["id"],
            f"Welcome {sanitized_data['username']}!"
        )
        
        return {"success": True, "user": db_result["user"]}
    else:
        return {"success": False, "error": db_result["error"]}
```

## 📋 Best Practices

### 1. **Identify Operation Type**
```python
# CPU-bound: Mathematical calculations, data processing, encryption
def cpu_intensive_function():
    # Heavy computations
    pass

# I/O-bound: Network requests, file operations, database queries
async def io_intensive_function():
    # Waiting for external resources
    pass
```

### 2. **Use Thread Pool for CPU-bound in Async Context**
```python
async def hybrid_function():
    # CPU-bound work in thread pool
    result = await asyncio.to_thread(cpu_intensive_function)
    
    # I/O-bound work
    await io_intensive_function()
    
    return result
```

### 3. **Concurrent I/O Operations**
```python
async def concurrent_io_operations():
    # Run multiple I/O operations concurrently
    results = await asyncio.gather(
        fetch_video_metadata("video1"),
        fetch_video_metadata("video2"),
        fetch_video_metadata("video3")
    )
    return results
```

### 4. **Error Handling**
```python
async def robust_io_operation():
    try:
        result = await fetch_external_data()
        return {"success": True, "data": result}
    except aiohttp.ClientError as e:
        return {"success": False, "error": f"Network error: {e}"}
    except Exception as e:
        return {"success": False, "error": f"Unexpected error: {e}"}
```

### 5. **Resource Management**
```python
async def managed_io_operation():
    # Use context managers for proper resource cleanup
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.json()
```

## 🚨 Common Mistakes

### ❌ Don't use `async def` for CPU-bound operations
```python
# ❌ Wrong: CPU-bound operation with async
async def validate_password(password: str) -> bool:
    # This doesn't involve I/O, should be sync
    return len(password) >= 8
```

### ❌ Don't use `def` for I/O-bound operations
```python
# ❌ Wrong: I/O-bound operation with sync
def fetch_data(url: str) -> Dict:
    # This involves network I/O, should be async
    response = requests.get(url)  # Blocking!
    return response.json()
```

### ❌ Don't forget to await async functions
```python
# ❌ Wrong: Not awaiting async function
async def process_data():
    result = fetch_data("http://api.example.com")  # Missing await!
    return result
```

### ✅ Correct patterns
```python
# ✅ Correct: CPU-bound with def
def validate_password(password: str) -> bool:
    return len(password) >= 8

# ✅ Correct: I/O-bound with async def
async def fetch_data(url: str) -> Dict:
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.json()

# ✅ Correct: Properly awaiting async functions
async def process_data():
    result = await fetch_data("http://api.example.com")
    return result
```

## 🎯 Summary

- **Use `def`** for CPU-intensive operations (validation, encryption, calculations)
- **Use `async def`** for I/O operations (network, file system, database)
- **Use `asyncio.to_thread()`** to run CPU-bound operations in async context
- **Use `asyncio.gather()`** for concurrent I/O operations
- **Always await** async functions
- **Use context managers** for proper resource cleanup

Following these guidelines ensures optimal performance and proper resource utilization in the Video-OpusClip system. 