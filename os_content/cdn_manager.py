from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

import asyncio
import aiohttp
import hashlib
import time
import os
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import structlog
from cache_manager import cache
from typing import Any, List, Dict, Optional
import logging
#!/usr/bin/env python3
"""
CDN Manager for OS Content UGC Video Generator
Optimizes content delivery and caching
"""


logger = structlog.get_logger("os_content.cdn")

class CDNManager:
    """CDN manager for content delivery optimization"""
    
    def __init__(self, 
                 cdn_url: str = "",
                 local_cache_dir: str = "./cdn_cache",
                 max_cache_size: int = 1024 * 1024 * 1024,  # 1GB
                 cache_ttl: int = 3600):
        
        
    """__init__ function."""
self.cdn_url = cdn_url.rstrip('/')
        self.local_cache_dir = Path(local_cache_dir)
        self.max_cache_size = max_cache_size
        self.cache_ttl = cache_ttl
        
        # Create cache directory
        self.local_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Statistics
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_requests = 0
        self.start_time = time.time()
        
        # HTTP session
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def start(self) -> Any:
        """Start the CDN manager"""
        timeout = aiohttp.ClientTimeout(total=30)
        self.session = aiohttp.ClientSession(timeout=timeout)
        logger.info("CDN manager started")
    
    async def stop(self) -> Any:
        """Stop the CDN manager"""
        if self.session:
            await self.session.close()
        logger.info("CDN manager stopped")
    
    def _generate_cache_key(self, content_id: str, content_type: str) -> str:
        """Generate cache key for content"""
        return hashlib.sha256(f"{content_id}:{content_type}".encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get local cache file path"""
        return self.local_cache_dir / f"{cache_key}.cache"
    
    async async def upload_content(self, 
                           content_path: str, 
                           content_id: str, 
                           content_type: str = "video") -> str:
        """Upload content to CDN and return CDN URL"""
        try:
            if not self.cdn_url:
                # No CDN configured, return local path
                return content_path
            
            cache_key = self._generate_cache_key(content_id, content_type)
            cdn_url = f"{self.cdn_url}/{content_type}/{cache_key}"
            
            # Check if already uploaded
            if await self._check_cdn_exists(cdn_url):
                logger.info(f"Content {content_id} already exists in CDN")
                return cdn_url
            
            # Upload to CDN
            await self._upload_to_cdn(content_path, cdn_url)
            
            # Cache the CDN URL locally
            await cache.set(f"cdn_url:{content_id}", cdn_url, ttl=self.cache_ttl)
            
            logger.info(f"Content {content_id} uploaded to CDN: {cdn_url}")
            return cdn_url
            
        except Exception as e:
            logger.error(f"Failed to upload content {content_id}: {e}")
            return content_path
    
    async def get_content_url(self, 
                            content_id: str, 
                            content_type: str = "video",
                            local_path: Optional[str] = None) -> str:
        """Get optimized content URL (CDN or local)"""
        self.total_requests += 1
        
        try:
            # Check cache first
            cached_url = await cache.get(f"cdn_url:{content_id}")
            if cached_url:
                self.cache_hits += 1
                return cached_url
            
            # If no CDN URL cached and we have a local path, upload to CDN
            if local_path and self.cdn_url:
                cdn_url = await self.upload_content(local_path, content_id, content_type)
                return cdn_url
            
            # Fallback to local path
            self.cache_misses += 1
            return local_path or f"/content/{content_id}"
            
        except Exception as e:
            logger.error(f"Failed to get content URL for {content_id}: {e}")
            return local_path or f"/content/{content_id}"
    
    async async def download_content(self, 
                             content_id: str, 
                             content_type: str = "video") -> Optional[str]:
        """Download content from CDN to local cache"""
        try:
            cache_key = self._generate_cache_key(content_id, content_type)
            cdn_url = f"{self.cdn_url}/{content_type}/{cache_key}"
            local_path = self._get_cache_path(cache_key)
            
            # Check if already cached locally
            if local_path.exists():
                self.cache_hits += 1
                return str(local_path)
            
            # Download from CDN
            await self._download_from_cdn(cdn_url, local_path)
            
            self.cache_misses += 1
            return str(local_path)
            
        except Exception as e:
            logger.error(f"Failed to download content {content_id}: {e}")
            return None
    
    async def _check_cdn_exists(self, cdn_url: str) -> bool:
        """Check if content exists in CDN"""
        try:
            async with self.session.head(cdn_url) as response:
                return response.status == 200
        except Exception:
            return False
    
    async def _upload_to_cdn(self, local_path: str, cdn_url: str):
        """Upload content to CDN"""
        try:
            with open(local_path, 'rb') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                data = f.read()
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            
            async with self.session.put(cdn_url, data=data) as response:
                if response.status not in [200, 201]:
                    raise Exception(f"Upload failed with status {response.status}")
                    
        except Exception as e:
            logger.error(f"Failed to upload to CDN {cdn_url}: {e}")
            raise
    
    async def _download_from_cdn(self, cdn_url: str, local_path: Path):
        """Download content from CDN"""
        try:
            async with self.session.get(cdn_url) as response:
                if response.status == 200:
                    content = await response.read()
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                    with open(local_path, 'wb') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                        f.write(content)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                else:
                    raise Exception(f"Download failed with status {response.status}")
                    
        except Exception as e:
            logger.error(f"Failed to download from CDN {cdn_url}: {e}")
            raise
    
    async def cleanup_cache(self, max_age: int = 86400):
        """Cleanup old cache files"""
        try:
            current_time = time.time()
            deleted_count = 0
            
            for cache_file in self.local_cache_dir.glob("*.cache"):
                if current_time - cache_file.stat().st_mtime > max_age:
                    cache_file.unlink()
                    deleted_count += 1
            
            logger.info(f"Cleaned up {deleted_count} old cache files")
            
        except Exception as e:
            logger.error(f"Failed to cleanup cache: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get CDN manager statistics"""
        uptime = time.time() - self.start_time
        cache_hit_rate = 0
        if self.total_requests > 0:
            cache_hit_rate = (self.cache_hits / self.total_requests) * 100
        
        # Calculate cache size
        cache_size = 0
        cache_files = 0
        for cache_file in self.local_cache_dir.glob("*.cache"):
            cache_size += cache_file.stat().st_size
            cache_files += 1
        
        return {
            "uptime": uptime,
            "total_requests": self.total_requests,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": cache_hit_rate,
            "cdn_url": self.cdn_url,
            "local_cache_size": cache_size,
            "local_cache_files": cache_files,
            "max_cache_size": self.max_cache_size,
            "cache_ttl": self.cache_ttl
        }

# Global CDN manager instance
cdn_manager = CDNManager()

async def initialize_cdn_manager(cdn_url: str = "", **kwargs):
    """Initialize the CDN manager"""
    global cdn_manager
    cdn_manager = CDNManager(cdn_url=cdn_url, **kwargs)
    await cdn_manager.start()
    logger.info("CDN manager initialized")

async def cleanup_cdn_manager():
    """Cleanup the CDN manager"""
    await cdn_manager.stop()
    logger.info("CDN manager cleaned up") 