from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import time
import json
import logging
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
from functools import wraps
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import aiohttp
import aiofiles
import aioredis
import asyncpg
import aiomysql
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy import select, update, insert
import requests
import redis
import pymysql
import psycopg2
import sqlite3
            from PIL import Image
            import numpy as np
            import torch
                from PIL import Image
                import numpy as np
                import torch
from typing import Any, List, Dict, Optional
"""
🔄 ASYNC CONVERSION EXAMPLES - PRACTICAL PATTERNS
================================================

Real-world examples of converting blocking operations to async in the AI Video system.
Each example shows the blocking (BAD) version and the async (GOOD) version.
"""


# Async libraries

# Sync libraries (to be converted)

logger = logging.getLogger(__name__)

# ============================================================================
# DATABASE OPERATIONS - CONVERSION EXAMPLES
# ============================================================================

class DatabaseConversionExamples:
    """Examples of converting database operations from sync to async."""
    
    # ❌ BAD: Synchronous database operations
    def bad_sync_database_operations(self) -> Any:
        """Examples of blocking database operations."""
        
        # ❌ BAD: Synchronous PostgreSQL
        def get_video_sync_pg(video_id: str) -> Dict:
            conn = psycopg2.connect(
                dbname="ai_video_db",
                user="user",
                password="pass",
                host="localhost"
            )
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM videos WHERE id = %s", (video_id,))
            result = cursor.fetchone()
            cursor.close()
            conn.close()
            return dict(result) if result else None
        
        # ❌ BAD: Synchronous MySQL
        def get_user_videos_sync_mysql(user_id: str) -> List[Dict]:
            conn = pymysql.connect(
                host='localhost',
                user='user',
                password='pass',
                database='ai_video_db'
            )
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM videos WHERE user_id = %s", (user_id,))
            results = cursor.fetchall()
            cursor.close()
            conn.close()
            return [dict(row) for row in results]
        
        # ❌ BAD: Synchronous SQLite
        def save_video_metadata_sync_sqlite(video_data: Dict) -> bool:
            conn = sqlite3.connect('ai_video.db')
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO videos (id, title, status) VALUES (?, ?, ?)",
                (video_data['id'], video_data['title'], video_data['status'])
            )
            conn.commit()
            cursor.close()
            conn.close()
            return True
    
    # ✅ GOOD: Asynchronous database operations
    async def good_async_database_operations(self) -> Any:
        """Examples of non-blocking async database operations."""
        
        # ✅ GOOD: Async PostgreSQL
        async def get_video_async_pg(video_id: str) -> Optional[Dict]:
            conn = await asyncpg.connect(
                user='user',
                password='pass',
                database='ai_video_db',
                host='localhost'
            )
            try:
                row = await conn.fetchrow(
                    "SELECT * FROM videos WHERE id = $1",
                    video_id
                )
                return dict(row) if row else None
            finally:
                await conn.close()
        
        # ✅ GOOD: Async MySQL
        async def get_user_videos_async_mysql(user_id: str) -> List[Dict]:
            conn = await aiomysql.connect(
                host='localhost',
                user='user',
                password='pass',
                database='ai_video_db'
            )
            try:
                async with conn.cursor() as cursor:
                    await cursor.execute(
                        "SELECT * FROM videos WHERE user_id = %s",
                        (user_id,)
                    )
                    rows = await cursor.fetchall()
                    return [dict(row) for row in rows]
            finally:
                conn.close()
                await conn.wait_closed()
        
        # ✅ GOOD: Async SQLAlchemy
        async def save_video_async_sqlalchemy(video_data: Dict) -> bool:
            engine = create_async_engine(
                "postgresql+asyncpg://user:pass@localhost/ai_video_db"
            )
            async_session = async_sessionmaker(engine, class_=AsyncSession)
            
            async with async_session() as session:
                try:
                    # Insert video
                    stmt = insert(Video).values(**video_data)
                    await session.execute(stmt)
                    await session.commit()
                    return True
                except Exception as e:
                    await session.rollback()
                    logger.error(f"Failed to save video: {e}")
                    return False
        
        # ✅ GOOD: Connection pooling
        class AsyncDatabasePool:
            def __init__(self) -> Any:
                self.pg_pool = None
                self.mysql_pool = None
            
            async def initialize(self) -> Any:
                # PostgreSQL pool
                self.pg_pool = await asyncpg.create_pool(
                    user='user',
                    password='pass',
                    database='ai_video_db',
                    host='localhost',
                    min_size=5,
                    max_size=20
                )
                
                # MySQL pool
                self.mysql_pool = await aiomysql.create_pool(
                    host='localhost',
                    user='user',
                    password='pass',
                    database='ai_video_db',
                    minsize=5,
                    maxsize=20
                )
            
            async def get_video_from_pool(self, video_id: str) -> Optional[Dict]:
                async with self.pg_pool.acquire() as conn:
                    row = await conn.fetchrow(
                        "SELECT * FROM videos WHERE id = $1",
                        video_id
                    )
                    return dict(row) if row else None
            
            async def save_video_to_pool(self, video_data: Dict) -> bool:
                async with self.mysql_pool.acquire() as conn:
                    async with conn.cursor() as cursor:
                        await cursor.execute(
                            "INSERT INTO videos (id, title, status) VALUES (%s, %s, %s)",
                            (video_data['id'], video_data['title'], video_data['status'])
                        )
                        await conn.commit()
                        return True

# ============================================================================
# HTTP/API OPERATIONS - CONVERSION EXAMPLES
# ============================================================================

class HTTPConversionExamples:
    """Examples of converting HTTP operations from sync to async."""
    
    # ❌ BAD: Synchronous HTTP operations
    async def bad_sync_http_operations(self) -> Any:
        """Examples of blocking HTTP operations."""
        
        # ❌ BAD: Synchronous API calls
        async def fetch_video_data_sync(video_id: str) -> Dict:
            response = requests.get(f"https://api.example.com/videos/{video_id}")
            response.raise_for_status()
            return response.json()
        
        # ❌ BAD: Synchronous file upload
        async def upload_video_sync(video_path: str, upload_url: str) -> bool:
            with open(video_path, 'rb') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                files = {'video': f}
                response = requests.post(upload_url, files=files)
                return response.status_code == 200
        
        # ❌ BAD: Synchronous batch requests
        async def fetch_multiple_videos_sync(video_ids: List[str]) -> List[Dict]:
            results = []
            for video_id in video_ids:
                response = requests.get(f"https://api.example.com/videos/{video_id}")
                results.append(response.json())
            return results
    
    # ✅ GOOD: Asynchronous HTTP operations
    async async def good_async_http_operations(self) -> Any:
        """Examples of non-blocking async HTTP operations."""
        
        # ✅ GOOD: Async API calls
        async async def fetch_video_data_async(video_id: str) -> Dict:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"https://api.example.com/videos/{video_id}") as response:
                    response.raise_for_status()
                    return await response.json()
        
        # ✅ GOOD: Async file upload
        async async def upload_video_async(video_path: str, upload_url: str) -> bool:
            async with aiohttp.ClientSession() as session:
                async with aiofiles.open(video_path, 'rb') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                    video_data = await f.read()
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                    data = aiohttp.FormData()
                    data.add_field('video', video_data, filename=Path(video_path).name)
                    
                    async with session.post(upload_url, data=data) as response:
                        return response.status == 200
        
        # ✅ GOOD: Concurrent batch requests
        async async def fetch_multiple_videos_async(video_ids: List[str]) -> List[Dict]:
            async with aiohttp.ClientSession() as session:
                tasks = []
                for video_id in video_ids:
                    task = session.get(f"https://api.example.com/videos/{video_id}")
                    tasks.append(task)
                
                responses = await asyncio.gather(*tasks, return_exceptions=True)
                
                results = []
                for response in responses:
                    if isinstance(response, Exception):
                        results.append(None)
                    else:
                        data = await response.json()
                        results.append(data)
                
                return results
        
        # ✅ GOOD: HTTP client with connection pooling
        class AsyncHTTPClient:
            def __init__(self, base_url: str = ""):
                
    """__init__ function."""
self.base_url = base_url
                self.session = None
            
            async def initialize(self) -> Any:
                timeout = aiohttp.ClientTimeout(total=30)
                connector = aiohttp.TCPConnector(
                    limit=100,
                    limit_per_host=30,
                    keepalive_timeout=30
                )
                self.session = aiohttp.ClientSession(
                    base_url=self.base_url,
                    timeout=timeout,
                    connector=connector
                )
            
            async def get(self, url: str, **kwargs) -> Dict:
                async with self.session.get(url, **kwargs) as response:
                    response.raise_for_status()
                    return await response.json()
            
            async def post(self, url: str, **kwargs) -> Dict:
                async with self.session.post(url, **kwargs) as response:
                    response.raise_for_status()
                    return await response.json()
            
            async def close(self) -> Any:
                if self.session:
                    await self.session.close()

# ============================================================================
# FILE I/O OPERATIONS - CONVERSION EXAMPLES
# ============================================================================

class FileIOConversionExamples:
    """Examples of converting file I/O operations from sync to async."""
    
    # ❌ BAD: Synchronous file operations
    def bad_sync_file_operations(self) -> Any:
        """Examples of blocking file operations."""
        
        # ❌ BAD: Synchronous file reading
        def read_video_config_sync(config_path: str) -> Dict:
            with open(config_path, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                return json.load(f)
        
        # ❌ BAD: Synchronous file writing
        def save_video_result_sync(result_path: str, data: Dict) -> bool:
            with open(result_path, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                json.dump(data, f, indent=2)
            return True
        
        # ❌ BAD: Synchronous binary file operations
        def read_video_file_sync(video_path: str) -> bytes:
            with open(video_path, 'rb') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                return f.read()
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        
        # ❌ BAD: Synchronous file processing
        def process_multiple_files_sync(file_paths: List[str]) -> List[Dict]:
            results = []
            for file_path in file_paths:
                with open(file_path, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                    data = json.load(f)
                    results.append(data)
            return results
    
    # ✅ GOOD: Asynchronous file operations
    async def good_async_file_operations(self) -> Any:
        """Examples of non-blocking async file operations."""
        
        # ✅ GOOD: Async file reading
        async def read_video_config_async(config_path: str) -> Dict:
            async with aiofiles.open(config_path, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                content = await f.read()
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                return json.loads(content)
        
        # ✅ GOOD: Async file writing
        async def save_video_result_async(result_path: str, data: Dict) -> bool:
            try:
                async with aiofiles.open(result_path, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                    await f.write(json.dumps(data, indent=2))
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                return True
            except Exception as e:
                logger.error(f"Failed to save result: {e}")
                return False
        
        # ✅ GOOD: Async binary file operations
        async def read_video_file_async(video_path: str) -> bytes:
            async with aiofiles.open(video_path, 'rb') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                return await f.read()
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        
        # ✅ GOOD: Concurrent file processing
        async def process_multiple_files_async(file_paths: List[str]) -> List[Dict]:
            tasks = [read_video_config_async(path) for path in file_paths]
            return await asyncio.gather(*tasks, return_exceptions=True)
        
        # ✅ GOOD: Async file manager with error handling
        class AsyncFileManager:
            def __init__(self) -> Any:
                self.executor = ThreadPoolExecutor(max_workers=10)
            
            async def read_file(self, file_path: str, encoding: str = 'utf-8') -> str:
                async with aiofiles.open(file_path, 'r', encoding=encoding) as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                    return await f.read()
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            
            async def write_file(self, file_path: str, content: str, encoding: str = 'utf-8') -> bool:
                try:
                    async with aiofiles.open(file_path, 'w', encoding=encoding) as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                        await f.write(content)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                    return True
                except Exception as e:
                    logger.error(f"Failed to write file {file_path}: {e}")
                    return False
            
            async def read_binary_file(self, file_path: str) -> bytes:
                async with aiofiles.open(file_path, 'rb') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                    return await f.read()
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            
            async def write_binary_file(self, file_path: str, content: bytes) -> bool:
                try:
                    async with aiofiles.open(file_path, 'wb') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                        await f.write(content)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                    return True
                except Exception as e:
                    logger.error(f"Failed to write binary file {file_path}: {e}")
                    return False
            
            async def file_exists(self, file_path: str) -> bool:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(self.executor, Path(file_path).exists)
            
            async def create_directory(self, directory_path: str) -> bool:
                try:
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(
                        self.executor, 
                        Path(directory_path).mkdir, 
                        parents=True, 
                        exist_ok=True
                    )
                    return True
                except Exception as e:
                    logger.error(f"Failed to create directory {directory_path}: {e}")
                    return False

# ============================================================================
# CACHE OPERATIONS - CONVERSION EXAMPLES
# ============================================================================

class CacheConversionExamples:
    """Examples of converting cache operations from sync to async."""
    
    # ❌ BAD: Synchronous cache operations
    def bad_sync_cache_operations(self) -> Any:
        """Examples of blocking cache operations."""
        
        # ❌ BAD: Synchronous Redis operations
        def get_cached_video_sync(video_id: str) -> Optional[Dict]:
            r = redis.Redis(host='localhost', port=6379, db=0)
            data = r.get(f"video:{video_id}")
            return json.loads(data) if data else None
        
        # ❌ BAD: Synchronous cache setting
        def set_cached_video_sync(video_id: str, data: Dict, ttl: int = 3600) -> bool:
            r = redis.Redis(host='localhost', port=6379, db=0)
            r.setex(f"video:{video_id}", ttl, json.dumps(data))
            return True
        
        # ❌ BAD: Synchronous cache deletion
        def delete_cached_video_sync(video_id: str) -> bool:
            r = redis.Redis(host='localhost', port=6379, db=0)
            return r.delete(f"video:{video_id}") > 0
    
    # ✅ GOOD: Asynchronous cache operations
    async def good_async_cache_operations(self) -> Any:
        """Examples of non-blocking async cache operations."""
        
        # ✅ GOOD: Async Redis operations
        async def get_cached_video_async(video_id: str) -> Optional[Dict]:
            redis_client = aioredis.from_url("redis://localhost")
            try:
                data = await redis_client.get(f"video:{video_id}")
                return json.loads(data) if data else None
            finally:
                await redis_client.close()
        
        # ✅ GOOD: Async cache setting
        async def set_cached_video_async(video_id: str, data: Dict, ttl: int = 3600) -> bool:
            redis_client = aioredis.from_url("redis://localhost")
            try:
                await redis_client.setex(
                    f"video:{video_id}",
                    ttl,
                    json.dumps(data)
                )
                return True
            finally:
                await redis_client.close()
        
        # ✅ GOOD: Async cache deletion
        async def delete_cached_video_async(video_id: str) -> bool:
            redis_client = aioredis.from_url("redis://localhost")
            try:
                return await redis_client.delete(f"video:{video_id}") > 0
            finally:
                await redis_client.close()
        
        # ✅ GOOD: Redis connection pooling
        class AsyncRedisManager:
            def __init__(self) -> Any:
                self.redis_pool = None
            
            async def initialize(self) -> Any:
                self.redis_pool = aioredis.from_url(
                    "redis://localhost",
                    encoding="utf-8",
                    decode_responses=True
                )
            
            async def get(self, key: str) -> Optional[str]:
                return await self.redis_pool.get(key)
            
            async def set(self, key: str, value: str, ttl: Optional[int] = None) -> bool:
                if ttl:
                    await self.redis_pool.setex(key, ttl, value)
                else:
                    await self.redis_pool.set(key, value)
                return True
            
            async def delete(self, key: str) -> bool:
                return await self.redis_pool.delete(key) > 0
            
            async def exists(self, key: str) -> bool:
                return await self.redis_pool.exists(key) > 0
            
            async def close(self) -> Any:
                if self.redis_pool:
                    await self.redis_pool.close()

# ============================================================================
# THIRD-PARTY LIBRARY CONVERSION EXAMPLES
# ============================================================================

class ThirdPartyConversionExamples:
    """Examples of converting third-party library operations from sync to async."""
    
    # ❌ BAD: Synchronous third-party operations
    def bad_sync_third_party_operations(self) -> Any:
        """Examples of blocking third-party operations."""
        
        # ❌ BAD: Synchronous image processing
        def process_image_sync(image_path: str) -> bytes:
            img = Image.open(image_path)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            img = img.resize((512, 512))
            output = io.BytesIO()
            img.save(output, format='JPEG')
            return output.getvalue()
        
        # ❌ BAD: Synchronous data processing
        def process_data_sync(data: List[float]) -> float:
            return np.mean(data)
        
        # ❌ BAD: Synchronous model inference
        def run_model_sync(input_data: np.ndarray) -> np.ndarray:
            model = torch.load('model.pth')
            with torch.no_grad():
                return model(input_data).numpy()
    
    # ✅ GOOD: Asynchronous third-party operations
    async def good_async_third_party_operations(self) -> Any:
        """Examples of non-blocking async third-party operations."""
        
        # ✅ GOOD: Async image processing
        async def process_image_async(image_path: str) -> bytes:
            def process_image_sync(path: str) -> bytes:
                img = Image.open(path)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                img = img.resize((512, 512))
                output = io.BytesIO()
                img.save(output, format='JPEG')
                return output.getvalue()
            
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as executor:
                return await loop.run_in_executor(executor, process_image_sync, image_path)
        
        # ✅ GOOD: Async data processing
        async def process_data_async(data: List[float]) -> float:
            def process_data_sync(numbers: List[float]) -> float:
                return np.mean(numbers)
            
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as executor:
                return await loop.run_in_executor(executor, process_data_sync, data)
        
        # ✅ GOOD: Async model inference
        async def run_model_async(input_data: np.ndarray) -> np.ndarray:
            def run_model_sync(data: np.ndarray) -> np.ndarray:
                model = torch.load('model.pth')
                with torch.no_grad():
                    return model(data).numpy()
            
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as executor:
                return await loop.run_in_executor(executor, run_model_sync, input_data)
        
        # ✅ GOOD: CPU-bound operations in process pool
        async def cpu_intensive_operation_async(data: List[float]) -> float:
            def cpu_intensive_sync(numbers: List[float]) -> float:
                # CPU-intensive computation
                result = 0.0
                for i in range(1000000):  # Simulate heavy computation
                    result += sum(x ** 2 for x in numbers)
                return result / len(numbers)
            
            loop = asyncio.get_event_loop()
            with ProcessPoolExecutor() as executor:
                return await loop.run_in_executor(executor, cpu_intensive_sync, data)

# ============================================================================
# COMPREHENSIVE ASYNC CONVERSION SYSTEM
# ============================================================================

class AsyncConversionSystem:
    """Complete system for converting blocking operations to async."""
    
    def __init__(self) -> Any:
        self.thread_executor = ThreadPoolExecutor(max_workers=20)
        self.process_executor = ProcessPoolExecutor(max_workers=4)
        self.conversion_stats = {
            'sync_to_async': 0,
            'cpu_bound_to_async': 0,
            'total_conversions': 0
        }
    
    def sync_to_async(self, func: Callable) -> Callable:
        """Convert synchronous function to asynchronous."""
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(self.thread_executor, func, *args, **kwargs)
            self.conversion_stats['sync_to_async'] += 1
            self.conversion_stats['total_conversions'] += 1
            return result
        return wrapper
    
    def cpu_bound_to_async(self, func: Callable) -> Callable:
        """Convert CPU-bound function to async using process executor."""
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(self.process_executor, func, *args, **kwargs)
            self.conversion_stats['cpu_bound_to_async'] += 1
            self.conversion_stats['total_conversions'] += 1
            return result
        return wrapper
    
    async def run_sync_in_executor(self, func: Callable, *args, **kwargs) -> Any:
        """Run synchronous function in executor."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.thread_executor, func, *args, **kwargs)
    
    def get_conversion_stats(self) -> Dict[str, int]:
        """Get conversion statistics."""
        return self.conversion_stats.copy()
    
    def cleanup(self) -> Any:
        """Cleanup executors."""
        self.thread_executor.shutdown(wait=True)
        self.process_executor.shutdown(wait=True)

# ============================================================================
# PRACTICAL USAGE EXAMPLES
# ============================================================================

async def practical_conversion_examples():
    """Practical examples of using the async conversion system."""
    
    # Initialize conversion system
    converter = AsyncConversionSystem()
    
    # Example 1: Convert sync database operation
    def sync_db_query(user_id: str) -> List[Dict]:
        # Simulate blocking database query
        time.sleep(0.1)
        return [{"id": user_id, "name": "User"}]
    
    async_db_query = converter.sync_to_async(sync_db_query)
    result = await async_db_query("user123")
    print(f"Async DB result: {result}")
    
    # Example 2: Convert sync file operation
    def sync_file_read(file_path: str) -> str:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        # Simulate blocking file read
        time.sleep(0.05)
        return f"Content from {file_path}"
    
    async_file_read = converter.sync_to_async(sync_file_read)
    content = await async_file_read("config.json")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
    print(f"Async file content: {content}")
    
    # Example 3: Convert CPU-bound operation
    def cpu_intensive_calculation(data: List[float]) -> float:
        # Simulate CPU-intensive operation
        result = 0.0
        for i in range(100000):
            result += sum(x ** 2 for x in data)
        return result / len(data)
    
    async_cpu_calc = converter.cpu_bound_to_async(cpu_intensive_calculation)
    result = await async_cpu_calc([1.0, 2.0, 3.0, 4.0, 5.0])
    print(f"Async CPU result: {result}")
    
    # Example 4: Multiple concurrent operations
    operations = [
        converter.sync_to_async(lambda: time.sleep(0.1)),
        converter.sync_to_async(lambda: time.sleep(0.1)),
        converter.sync_to_async(lambda: time.sleep(0.1))
    ]
    
    start_time = time.time()
    await asyncio.gather(*[op() for op in operations])
    end_time = time.time()
    
    print(f"Concurrent operations took: {end_time - start_time:.3f}s")
    
    # Print conversion stats
    stats = converter.get_conversion_stats()
    print(f"Conversion stats: {stats}")
    
    # Cleanup
    converter.cleanup()

match __name__:
    case "__main__":
    asyncio.run(practical_conversion_examples()) 