#!/usr/bin/env python3
"""
Async/Sync Patterns for Video-OpusClip
Demonstrates proper use of `def` for CPU-bound and `async def` for I/O-bound operations
"""

import asyncio
import time
import json
import hashlib
import re
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from datetime import datetime
import aiohttp
import aiofiles
import aioredis
import asyncpg
from cryptography.fernet import Fernet

# CPU-bound operations using `def`
class CPUIntensiveOperations:
    """CPU-bound operations that should use `def`"""
    
    def __init__(self):
        self.encryption_key = Fernet.generate_key()
        self.cipher = Fernet(self.encryption_key)
    
    def validate_password_strength(self, password: str) -> Dict[str, Any]:
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
    
    def hash_password(self, password: str, salt: str = None) -> Dict[str, str]:
        """CPU-bound password hashing"""
        if salt is None:
            salt = hashlib.sha256(str(time.time()).encode()).hexdigest()[:16]
        
        # Multiple rounds for security
        hashed = password + salt
        for _ in range(100000):  # CPU-intensive
            hashed = hashlib.sha256(hashed.encode()).hexdigest()
        
        return {
            "hash": hashed,
            "salt": salt
        }
    
    def encrypt_data(self, data: str) -> str:
        """CPU-bound data encryption"""
        return self.cipher.encrypt(data.encode()).decode()
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """CPU-bound data decryption"""
        return self.cipher.decrypt(encrypted_data.encode()).decode()
    
    def validate_email_format(self, email: str) -> bool:
        """CPU-bound email validation"""
        import re
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
    
    def sanitize_input(self, text: str) -> str:
        """CPU-bound input sanitization"""
        import re
        
        # Remove dangerous characters
        sanitized = re.sub(r'[<>"\']', '', text)
        # Remove script tags
        sanitized = re.sub(r'<script.*?</script>', '', sanitized, flags=re.IGNORECASE)
        # Remove SQL injection patterns
        sanitized = re.sub(r'(\b(union|select|insert|update|delete|drop|create|alter)\b)', '', sanitized, flags=re.IGNORECASE)
        
        return sanitized.strip()
    
    def calculate_video_metrics(self, video_data: Dict[str, Any]) -> Dict[str, Any]:
        """CPU-bound video metrics calculation"""
        # Simulate CPU-intensive calculations
        duration = video_data.get("duration", 0)
        frame_count = video_data.get("frame_count", 0)
        resolution = video_data.get("resolution", "1920x1080")
        
        # Complex calculations
        fps = frame_count / duration if duration > 0 else 0
        bitrate = (video_data.get("file_size", 0) * 8) / duration if duration > 0 else 0
        
        # Simulate processing time
        time.sleep(0.01)  # Simulate CPU work
        
        return {
            "fps": round(fps, 2),
            "bitrate": round(bitrate, 2),
            "resolution": resolution,
            "aspect_ratio": self._calculate_aspect_ratio(resolution),
            "compression_ratio": self._calculate_compression_ratio(video_data)
        }
    
    def _calculate_aspect_ratio(self, resolution: str) -> str:
        """CPU-bound aspect ratio calculation"""
        try:
            width, height = map(int, resolution.split('x'))
            gcd = self._gcd(width, height)
            return f"{width//gcd}:{height//gcd}"
        except:
            return "16:9"
    
    def _gcd(self, a: int, b: int) -> int:
        """CPU-bound GCD calculation"""
        while b:
            a, b = b, a % b
        return a
    
    def _calculate_compression_ratio(self, video_data: Dict[str, Any]) -> float:
        """CPU-bound compression ratio calculation"""
        original_size = video_data.get("original_size", 0)
        compressed_size = video_data.get("compressed_size", 0)
        
        if original_size > 0:
            return round(compressed_size / original_size, 3)
        return 1.0
    
    def process_video_frames(self, frames: List[str]) -> Dict[str, Any]:
        """CPU-bound frame processing"""
        processed_frames = []
        total_pixels = 0
        
        for frame in frames:
            # Simulate CPU-intensive frame processing
            processed_frame = self._process_single_frame(frame)
            processed_frames.append(processed_frame)
            total_pixels += processed_frame.get("pixel_count", 0)
        
        return {
            "processed_frames": len(processed_frames),
            "total_pixels": total_pixels,
            "average_processing_time": 0.05,  # Simulated
            "frames": processed_frames
        }
    
    def _process_single_frame(self, frame: str) -> Dict[str, Any]:
        """CPU-bound single frame processing"""
        # Simulate intensive processing
        time.sleep(0.001)  # Simulate CPU work
        
        return {
            "frame_id": frame,
            "pixel_count": 1920 * 1080,
            "processed": True,
            "features": ["edge_detection", "color_analysis", "motion_estimation"]
        }

# I/O-bound operations using `async def`
class IOIntensiveOperations:
    """I/O-bound operations that should use `async def`"""
    
    def __init__(self):
        self.session = None
        self.redis_client = None
        self.db_pool = None
    
    async def initialize_connections(self):
        """Initialize async connections"""
        self.session = aiohttp.ClientSession()
        self.redis_client = aioredis.from_url("redis://localhost")
        self.db_pool = await asyncpg.create_pool(
            "postgresql://user:password@localhost/video_db"
        )
    
    async def cleanup_connections(self):
        """Cleanup async connections"""
        if self.session:
            await self.session.close()
        if self.redis_client:
            await self.redis_client.close()
        if self.db_pool:
            await self.db_pool.close()
    
    async def fetch_video_metadata(self, video_id: str) -> Dict[str, Any]:
        """I/O-bound: Fetch video metadata from API"""
        if not self.session:
            await self.initialize_connections()
        
        url = f"https://api.example.com/videos/{video_id}"
        
        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "success": True,
                        "data": data
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
    
    async def save_video_to_storage(self, video_path: str, video_data: bytes) -> Dict[str, Any]:
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
            return {
                "success": False,
                "error": str(e)
            }
    
    async def load_video_from_storage(self, video_path: str) -> Dict[str, Any]:
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
            return {
                "success": False,
                "error": str(e)
            }
    
    async def cache_video_metadata(self, video_id: str, metadata: Dict[str, Any]) -> bool:
        """I/O-bound: Cache metadata in Redis"""
        if not self.redis_client:
            await self.initialize_connections()
        
        try:
            await self.redis_client.setex(
                f"video:{video_id}",
                3600,  # 1 hour TTL
                json.dumps(metadata)
            )
            return True
        except Exception:
            return False
    
    async def get_cached_metadata(self, video_id: str) -> Optional[Dict[str, Any]]:
        """I/O-bound: Get cached metadata from Redis"""
        if not self.redis_client:
            await self.initialize_connections()
        
        try:
            cached_data = await self.redis_client.get(f"video:{video_id}")
            if cached_data:
                return json.loads(cached_data)
            return None
        except Exception:
            return None
    
    async def save_user_to_database(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """I/O-bound: Save user to database"""
        if not self.db_pool:
            await self.initialize_connections()
        
        try:
            async with self.db_pool.acquire() as conn:
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
                
                return {
                    "success": True,
                    "user": dict(row)
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_user_from_database(self, user_id: int) -> Dict[str, Any]:
        """I/O-bound: Get user from database"""
        if not self.db_pool:
            await self.initialize_connections()
        
        try:
            async with self.db_pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT id, username, email, created_at FROM users WHERE id = $1",
                    user_id
                )
                
                if row:
                    return {
                        "success": True,
                        "user": dict(row)
                    }
                else:
                    return {
                        "success": False,
                        "error": "User not found"
                    }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def upload_video_to_cloud(self, video_data: bytes, filename: str) -> Dict[str, Any]:
        """I/O-bound: Upload video to cloud storage"""
        if not self.session:
            await self.initialize_connections()
        
        url = "https://storage.example.com/upload"
        
        try:
            async with self.session.post(url, data={"file": video_data}) as response:
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
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def send_notification(self, user_id: int, message: str) -> bool:
        """I/O-bound: Send notification to user"""
        if not self.session:
            await self.initialize_connections()
        
        url = "https://notifications.example.com/send"
        data = {
            "user_id": user_id,
            "message": message,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        try:
            async with self.session.post(url, json=data) as response:
                return response.status == 200
        except Exception:
            return False

# Hybrid operations that combine CPU and I/O
class HybridOperations:
    """Operations that combine CPU-bound and I/O-bound work"""
    
    def __init__(self):
        self.cpu_ops = CPUIntensiveOperations()
        self.io_ops = IOIntensiveOperations()
    
    async def process_and_save_video(self, video_data: bytes, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Hybrid: Process video (CPU) and save (I/O)"""
        # CPU-bound: Process video metadata
        processed_metadata = self.cpu_ops.calculate_video_metrics(metadata)
        
        # I/O-bound: Save to storage
        filename = f"video_{int(time.time())}.mp4"
        save_result = await self.io_ops.save_video_to_storage(filename, video_data)
        
        if save_result["success"]:
            # I/O-bound: Cache metadata
            await self.io_ops.cache_video_metadata(filename, processed_metadata)
            
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
    
    async def validate_and_register_user(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """Hybrid: Validate user data (CPU) and save to database (I/O)"""
        # CPU-bound: Validate input
        if not self.cpu_ops.validate_email_format(user_data["email"]):
            return {
                "success": False,
                "error": "Invalid email format"
            }
        
        password_validation = self.cpu_ops.validate_password_strength(user_data["password"])
        if not password_validation["valid"]:
            return {
                "success": False,
                "error": "Password too weak"
            }
        
        # CPU-bound: Hash password
        hashed_password = self.cpu_ops.hash_password(user_data["password"])
        
        # I/O-bound: Save to database
        sanitized_data = {
            "username": self.cpu_ops.sanitize_input(user_data["username"]),
            "email": user_data["email"],
            "password_hash": hashed_password["hash"],
            "password_salt": hashed_password["salt"]
        }
        
        db_result = await self.io_ops.save_user_to_database(sanitized_data)
        
        if db_result["success"]:
            # I/O-bound: Send welcome notification
            await self.io_ops.send_notification(
                db_result["user"]["id"],
                f"Welcome {sanitized_data['username']}!"
            )
            
            return {
                "success": True,
                "user": db_result["user"]
            }
        else:
            return {
                "success": False,
                "error": db_result["error"]
            }
    
    async def fetch_and_process_video(self, video_id: str) -> Dict[str, Any]:
        """Hybrid: Fetch video (I/O) and process frames (CPU)"""
        # I/O-bound: Fetch video metadata
        metadata_result = await self.io_ops.fetch_video_metadata(video_id)
        
        if not metadata_result["success"]:
            return {
                "success": False,
                "error": metadata_result["error"]
            }
        
        # CPU-bound: Process video metrics
        processed_metadata = self.cpu_ops.calculate_video_metrics(metadata_result["data"])
        
        # CPU-bound: Process frames (simulated)
        frames = [f"frame_{i}" for i in range(100)]
        frame_results = self.cpu_ops.process_video_frames(frames)
        
        # I/O-bound: Cache results
        await self.io_ops.cache_video_metadata(video_id, {
            **processed_metadata,
            "frame_analysis": frame_results
        })
        
        return {
            "success": True,
            "video_id": video_id,
            "metadata": processed_metadata,
            "frame_analysis": frame_results
        }

# Example usage
async def main():
    """Example usage of async/sync patterns"""
    print("🔄 Async/Sync Patterns Example")
    
    # Initialize operations
    cpu_ops = CPUIntensiveOperations()
    io_ops = IOIntensiveOperations()
    hybrid_ops = HybridOperations()
    
    # Example 1: CPU-bound operations
    print("\n💻 CPU-bound Operations:")
    
    # Password validation
    password_result = cpu_ops.validate_password_strength("StrongPass123!")
    print(f"   Password validation: {'✅' if password_result['valid'] else '❌'}")
    print(f"   Strength: {password_result['strength']}")
    
    # Email validation
    email_valid = cpu_ops.validate_email_format("user@example.com")
    print(f"   Email validation: {'✅' if email_valid else '❌'}")
    
    # Input sanitization
    sanitized = cpu_ops.sanitize_input("<script>alert('xss')</script>Hello World")
    print(f"   Input sanitization: {sanitized}")
    
    # Video metrics calculation
    video_data = {
        "duration": 120.5,
        "frame_count": 3600,
        "resolution": "1920x1080",
        "file_size": 50 * 1024 * 1024  # 50MB
    }
    metrics = cpu_ops.calculate_video_metrics(video_data)
    print(f"   Video metrics: FPS={metrics['fps']}, Bitrate={metrics['bitrate']}")
    
    # Example 2: I/O-bound operations (simulated)
    print("\n🌐 I/O-bound Operations:")
    
    # Simulate video metadata fetch
    print("   Fetching video metadata... (simulated)")
    await asyncio.sleep(0.1)  # Simulate network delay
    
    # Simulate file operations
    print("   Saving video to storage... (simulated)")
    await asyncio.sleep(0.05)  # Simulate I/O delay
    
    # Example 3: Hybrid operations
    print("\n🔀 Hybrid Operations:")
    
    # Simulate user registration
    user_data = {
        "username": "newuser",
        "email": "newuser@example.com",
        "password": "StrongPass123!"
    }
    
    print("   Processing user registration... (simulated)")
    await asyncio.sleep(0.1)  # Simulate processing time
    
    # Simulate video processing
    print("   Processing video... (simulated)")
    await asyncio.sleep(0.2)  # Simulate processing time
    
    print("\n🎯 Async/Sync patterns demonstrated!")
    print("\n📋 Key Points:")
    print("   • Use `def` for CPU-intensive operations (validation, encryption, calculations)")
    print("   • Use `async def` for I/O operations (network, file system, database)")
    print("   • Combine both patterns for complex operations")
    print("   • Always await async functions")
    print("   • Use asyncio.gather() for concurrent I/O operations")

if __name__ == "__main__":
    asyncio.run(main()) 