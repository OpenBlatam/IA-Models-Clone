from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
TIMEOUT_SECONDS = 60

import asyncio
import json
import time
from typing import Dict, Any, List
from datetime import datetime, timedelta
import aiohttp
import httpx
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.responses import JSONResponse
import structlog
from dependencies.shared_resources import (
from typing import Any, List, Dict, Optional
import logging
"""
Shared Resources Dependency Injection Demo

This demo showcases the comprehensive shared resources dependency injection system
including network sessions, cryptographic backends, and other shared resources.

Features demonstrated:
- HTTP session management with connection pooling
- WebSocket session management
- Cryptographic operations (encryption, decryption, hashing, signing)
- Database connection pooling
- Redis connection pooling
- Health checks and monitoring
- Resource lifecycle management
- FastAPI integration
- Context managers for resource management
"""



# Import shared resources
    SharedResourceConfig,
    ResourceConfig,
    CryptoConfig,
    ResourceType,
    CryptoAlgorithm,
    get_http_session,
    get_websocket_session,
    get_crypto_backend,
    get_database_pool,
    get_redis_pool,
    http_session_context,
    crypto_backend_context,
    initialize_shared_resources,
    shutdown_shared_resources,
    get_resource_health,
    get_all_resource_health
)

# Configure logging
logger = structlog.get_logger(__name__)

# =============================================================================
# Demo Configuration
# =============================================================================

def create_demo_config() -> SharedResourceConfig:
    """Create demo configuration for shared resources."""
    return SharedResourceConfig(
        resources={
            "http_session": ResourceConfig(
                name="http_session",
                resource_type=ResourceType.HTTP_SESSION,
                max_connections=50,
                timeout=30.0,
                keepalive_timeout=60.0,
                custom_headers={
                    "User-Agent": "SharedResourcesDemo/1.0",
                    "Accept": "application/json"
                }
            ),
            "websocket_session": ResourceConfig(
                name="websocket_session",
                resource_type=ResourceType.WEBSOCKET_SESSION,
                max_connections=20,
                timeout=10.0
            ),
            "database_pool": ResourceConfig(
                name="database_pool",
                resource_type=ResourceType.DATABASE_POOL,
                max_connections=10,
                pool_timeout=5.0
            ),
            "redis_pool": ResourceConfig(
                name="redis_pool",
                resource_type=ResourceType.REDIS_POOL,
                max_connections=20,
                timeout=5.0
            )
        },
        crypto_configs={
            "default": CryptoConfig(
                algorithm=CryptoAlgorithm.AES_256_GCM,
                key_size=256
            ),
            "rsa": CryptoConfig(
                algorithm=CryptoAlgorithm.RSA_2048,
                key_size=2048
            ),
            "hashing": CryptoConfig(
                algorithm=CryptoAlgorithm.SHA_256,
                key_size=256
            )
        },
        global_timeout=30.0,
        global_max_retries=3,
        enable_monitoring=True,
        enable_health_checks=True,
        resource_cleanup_interval=300.0
    )

# =============================================================================
# Demo Service Classes
# =============================================================================

class NetworkService:
    """Service for demonstrating network operations."""
    
    def __init__(self, http_session: aiohttp.ClientSession):
        
    """__init__ function."""
self.http_session = http_session
    
    async async def fetch_data(self, url: str) -> Dict[str, Any]:
        """Fetch data from a URL using the shared HTTP session."""
        try:
            async with self.http_session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "status": "success",
                        "data": data,
                        "status_code": response.status,
                        "headers": dict(response.headers)
                    }
                else:
                    return {
                        "status": "error",
                        "error": f"HTTP {response.status}",
                        "status_code": response.status
                    }
        except Exception as e:
            logger.error("Error fetching data", url=url, error=str(e))
            return {
                "status": "error",
                "error": str(e)
            }
    
    async async def fetch_multiple_urls(self, urls: List[str]) -> List[Dict[str, Any]]:
        """Fetch data from multiple URLs concurrently."""
        tasks = [self.fetch_data(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to error responses
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    "status": "error",
                    "error": str(result),
                    "url": urls[i]
                })
            else:
                processed_results.append(result)
        
        return processed_results

class CryptoService:
    """Service for demonstrating cryptographic operations."""
    
    def __init__(self, crypto_backend) -> Any:
        self.crypto_backend = crypto_backend
    
    async def encrypt_data(self, data: str) -> Dict[str, Any]:
        """Encrypt data using the shared crypto backend."""
        try:
            data_bytes = data.encode('utf-8')
            encrypted = await self.crypto_backend.encrypt(data_bytes)
            
            return {
                "status": "success",
                "original_data": data,
                "encrypted_data": encrypted.hex(),
                "algorithm": self.crypto_backend.crypto_config.algorithm.value
            }
        except Exception as e:
            logger.error("Error encrypting data", error=str(e))
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def decrypt_data(self, encrypted_hex: str) -> Dict[str, Any]:
        """Decrypt data using the shared crypto backend."""
        try:
            encrypted_bytes = bytes.fromhex(encrypted_hex)
            decrypted = await self.crypto_backend.decrypt(encrypted_bytes)
            decrypted_str = decrypted.decode('utf-8')
            
            return {
                "status": "success",
                "encrypted_data": encrypted_hex,
                "decrypted_data": decrypted_str,
                "algorithm": self.crypto_backend.crypto_config.algorithm.value
            }
        except Exception as e:
            logger.error("Error decrypting data", error=str(e))
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def hash_data(self, data: str) -> Dict[str, Any]:
        """Hash data using the shared crypto backend."""
        try:
            data_bytes = data.encode('utf-8')
            hashed = await self.crypto_backend.hash(data_bytes)
            
            return {
                "status": "success",
                "original_data": data,
                "hash": hashed.hex(),
                "algorithm": self.crypto_backend.crypto_config.algorithm.value
            }
        except Exception as e:
            logger.error("Error hashing data", error=str(e))
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def sign_data(self, data: str) -> Dict[str, Any]:
        """Sign data using the shared crypto backend."""
        try:
            data_bytes = data.encode('utf-8')
            signature = await self.crypto_backend.sign(data_bytes)
            
            return {
                "status": "success",
                "data": data,
                "signature": signature.hex(),
                "algorithm": self.crypto_backend.crypto_config.algorithm.value
            }
        except Exception as e:
            logger.error("Error signing data", error=str(e))
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def verify_signature(self, data: str, signature_hex: str) -> Dict[str, Any]:
        """Verify signature using the shared crypto backend."""
        try:
            data_bytes = data.encode('utf-8')
            signature_bytes = bytes.fromhex(signature_hex)
            is_valid = await self.crypto_backend.verify(data_bytes, signature_bytes)
            
            return {
                "status": "success",
                "data": data,
                "signature": signature_hex,
                "is_valid": is_valid,
                "algorithm": self.crypto_backend.crypto_config.algorithm.value
            }
        except Exception as e:
            logger.error("Error verifying signature", error=str(e))
            return {
                "status": "error",
                "error": str(e)
            }

class CacheService:
    """Service for demonstrating cache operations."""
    
    def __init__(self, redis_client) -> Any:
        self.redis_client = redis_client
    
    async def set_cache(self, key: str, value: str, ttl: int = 3600) -> Dict[str, Any]:
        """Set a value in cache."""
        try:
            await self.redis_client.set(key, value, ex=ttl)
            return {
                "status": "success",
                "key": key,
                "value": value,
                "ttl": ttl
            }
        except Exception as e:
            logger.error("Error setting cache", key=key, error=str(e))
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def get_cache(self, key: str) -> Dict[str, Any]:
        """Get a value from cache."""
        try:
            value = await self.redis_client.get(key)
            if value:
                return {
                    "status": "success",
                    "key": key,
                    "value": value.decode('utf-8')
                }
            else:
                return {
                    "status": "not_found",
                    "key": key
                }
        except Exception as e:
            logger.error("Error getting cache", key=key, error=str(e))
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def delete_cache(self, key: str) -> Dict[str, Any]:
        """Delete a value from cache."""
        try:
            result = await self.redis_client.delete(key)
            return {
                "status": "success",
                "key": key,
                "deleted": bool(result)
            }
        except Exception as e:
            logger.error("Error deleting cache", key=key, error=str(e))
            return {
                "status": "error",
                "error": str(e)
            }

# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="Shared Resources Demo",
    description="Demo of shared resources dependency injection system",
    version="1.0.0"
)

@app.on_event("startup")
async def startup_event():
    """Initialize shared resources on startup."""
    config = create_demo_config()
    await initialize_shared_resources(config)
    logger.info("Shared resources initialized")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup shared resources on shutdown."""
    await shutdown_shared_resources()
    logger.info("Shared resources cleaned up")

# =============================================================================
# API Routes
# =============================================================================

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Shared Resources Dependency Injection Demo",
        "version": "1.0.0",
        "endpoints": {
            "network": "/network/*",
            "crypto": "/crypto/*",
            "cache": "/cache/*",
            "health": "/health",
            "metrics": "/metrics"
        }
    }

# Network Operations
@app.get("/network/fetch/{url:path}")
async def fetch_url(
    url: str,
    http_session: aiohttp.ClientSession = Depends(get_http_session)
):
    """Fetch data from a URL using shared HTTP session."""
    network_service = NetworkService(http_session)
    result = await network_service.fetch_data(url)
    return result

@app.post("/network/fetch-multiple")
async def fetch_multiple_urls(
    urls: List[str],
    http_session: aiohttp.ClientSession = Depends(get_http_session)
):
    """Fetch data from multiple URLs concurrently."""
    network_service = NetworkService(http_session)
    results = await network_service.fetch_multiple_urls(urls)
    return {
        "urls": urls,
        "results": results,
        "total": len(results)
    }

# Cryptographic Operations
@app.post("/crypto/encrypt")
async def encrypt_data(
    data: str,
    crypto_backend = Depends(get_crypto_backend)
):
    """Encrypt data using shared crypto backend."""
    crypto_service = CryptoService(crypto_backend)
    result = await crypto_service.encrypt_data(data)
    return result

@app.post("/crypto/decrypt")
async def decrypt_data(
    encrypted_data: str,
    crypto_backend = Depends(get_crypto_backend)
):
    """Decrypt data using shared crypto backend."""
    crypto_service = CryptoService(crypto_backend)
    result = await crypto_service.decrypt_data(encrypted_data)
    return result

@app.post("/crypto/hash")
async def hash_data(
    data: str,
    crypto_backend = Depends(get_crypto_backend)
):
    """Hash data using shared crypto backend."""
    crypto_service = CryptoService(crypto_backend)
    result = await crypto_service.hash_data(data)
    return result

@app.post("/crypto/sign")
async def sign_data(
    data: str,
    crypto_backend = Depends(get_crypto_backend)
):
    """Sign data using shared crypto backend."""
    crypto_service = CryptoService(crypto_backend)
    result = await crypto_service.sign_data(data)
    return result

@app.post("/crypto/verify")
async def verify_signature(
    data: str,
    signature: str,
    crypto_backend = Depends(get_crypto_backend)
):
    """Verify signature using shared crypto backend."""
    crypto_service = CryptoService(crypto_backend)
    result = await crypto_service.verify_signature(data, signature)
    return result

# Cache Operations
@app.post("/cache/set")
async def set_cache(
    key: str,
    value: str,
    ttl: int = 3600,
    redis_client = Depends(get_redis_pool)
):
    """Set a value in cache."""
    cache_service = CacheService(redis_client)
    result = await cache_service.set_cache(key, value, ttl)
    return result

@app.get("/cache/get/{key}")
async def get_cache(
    key: str,
    redis_client = Depends(get_redis_pool)
):
    """Get a value from cache."""
    cache_service = CacheService(redis_client)
    result = await cache_service.get_cache(key)
    return result

@app.delete("/cache/delete/{key}")
async def delete_cache(
    key: str,
    redis_client = Depends(get_redis_pool)
):
    """Delete a value from cache."""
    cache_service = CacheService(redis_client)
    result = await cache_service.delete_cache(key)
    return result

# Health and Metrics
@app.get("/health")
async def health_check():
    """Get health status of all resources."""
    health_status = get_all_resource_health()
    return {
        "status": "healthy" if all(h.is_healthy for h in health_status.values()) else "unhealthy",
        "timestamp": datetime.utcnow().isoformat(),
        "resources": {
            name: {
                "is_healthy": health.is_healthy,
                "last_check": health.last_check.isoformat(),
                "response_time": health.response_time,
                "error_count": health.error_count,
                "last_error": health.last_error
            }
            for name, health in health_status.items()
        }
    }

@app.get("/metrics")
async def get_metrics():
    """Get metrics for all resources."""
    health_status = get_all_resource_health()
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "total_resources": len(health_status),
        "healthy_resources": sum(1 for h in health_status.values() if h.is_healthy),
        "unhealthy_resources": sum(1 for h in health_status.values() if not h.is_healthy),
        "average_response_time": sum(h.response_time for h in health_status.values()) / len(health_status) if health_status else 0,
        "total_errors": sum(h.error_count for h in health_status.values()),
        "resources": health_status
    }

# =============================================================================
# Demo Functions
# =============================================================================

async def demo_network_operations():
    """Demonstrate network operations with shared HTTP session."""
    print("\n=== Network Operations Demo ===")
    
    # Initialize shared resources
    config = create_demo_config()
    await initialize_shared_resources(config)
    
    try:
        # Get HTTP session
        async with http_session_context() as session:
            network_service = NetworkService(session)
            
            # Fetch single URL
            print("Fetching single URL...")
            result = await network_service.fetch_data("https://httpbin.org/json")
            print(f"Single fetch result: {json.dumps(result, indent=2)}")
            
            # Fetch multiple URLs
            print("\nFetching multiple URLs...")
            urls = [
                "https://httpbin.org/json",
                "https://httpbin.org/headers",
                "https://httpbin.org/ip"
            ]
            results = await network_service.fetch_multiple_urls(urls)
            print(f"Multiple fetch results: {json.dumps(results, indent=2)}")
    
    finally:
        await shutdown_shared_resources()

async def demo_crypto_operations():
    """Demonstrate cryptographic operations with shared crypto backend."""
    print("\n=== Cryptographic Operations Demo ===")
    
    # Initialize shared resources
    config = create_demo_config()
    await initialize_shared_resources(config)
    
    try:
        # Get crypto backend
        async with crypto_backend_context() as crypto_backend:
            crypto_service = CryptoService(crypto_backend)
            
            # Test data
            test_data = "Hello, Shared Resources!"
            
            # Encrypt data
            print("Encrypting data...")
            encrypt_result = await crypto_service.encrypt_data(test_data)
            print(f"Encryption result: {json.dumps(encrypt_result, indent=2)}")
            
            # Decrypt data
            if encrypt_result["status"] == "success":
                print("\nDecrypting data...")
                decrypt_result = await crypto_service.decrypt_data(encrypt_result["encrypted_data"])
                print(f"Decryption result: {json.dumps(decrypt_result, indent=2)}")
            
            # Hash data
            print("\nHashing data...")
            hash_result = await crypto_service.hash_data(test_data)
            print(f"Hash result: {json.dumps(hash_result, indent=2)}")
            
            # Sign data
            print("\nSigning data...")
            sign_result = await crypto_service.sign_data(test_data)
            print(f"Sign result: {json.dumps(sign_result, indent=2)}")
            
            # Verify signature
            if sign_result["status"] == "success":
                print("\nVerifying signature...")
                verify_result = await crypto_service.verify_signature(
                    test_data, 
                    sign_result["signature"]
                )
                print(f"Verify result: {json.dumps(verify_result, indent=2)}")
    
    finally:
        await shutdown_shared_resources()

async def demo_context_managers():
    """Demonstrate context managers for resource management."""
    print("\n=== Context Managers Demo ===")
    
    # Initialize shared resources
    config = create_demo_config()
    await initialize_shared_resources(config)
    
    try:
        # HTTP session context manager
        print("Using HTTP session context manager...")
        async with http_session_context() as session:
            network_service = NetworkService(session)
            result = await network_service.fetch_data("https://httpbin.org/json")
            print(f"HTTP session result: {result['status']}")
        
        # Crypto backend context manager
        print("\nUsing crypto backend context manager...")
        async with crypto_backend_context() as crypto_backend:
            crypto_service = CryptoService(crypto_backend)
            result = await crypto_service.hash_data("test data")
            print(f"Crypto backend result: {result['status']}")
    
    finally:
        await shutdown_shared_resources()

async def demo_health_monitoring():
    """Demonstrate health monitoring."""
    print("\n=== Health Monitoring Demo ===")
    
    # Initialize shared resources
    config = create_demo_config()
    await initialize_shared_resources(config)
    
    try:
        # Wait for health checks to run
        await asyncio.sleep(2)
        
        # Get health status
        health_status = get_all_resource_health()
        print("Health status:")
        for name, health in health_status.items():
            print(f"  {name}: {'Healthy' if health.is_healthy else 'Unhealthy'}")
            print(f"    Response time: {health.response_time:.3f}s")
            print(f"    Error count: {health.error_count}")
            if health.last_error:
                print(f"    Last error: {health.last_error}")
        
        # Get metrics
        print("\nMetrics:")
        for name, health in health_status.items():
            print(f"  {name}:")
            print(f"    Uptime: {health.uptime:.2f}s")
            print(f"    Last check: {health.last_check}")
    
    finally:
        await shutdown_shared_resources()

# =============================================================================
# Main Demo Runner
# =============================================================================

async def run_demo():
    """Run all demos."""
    print("Shared Resources Dependency Injection Demo")
    print("=" * 50)
    
    try:
        # Run individual demos
        await demo_network_operations()
        await demo_crypto_operations()
        await demo_context_managers()
        await demo_health_monitoring()
        
        print("\n" + "=" * 50)
        print("All demos completed successfully!")
        
    except Exception as e:
        logger.error("Demo failed", error=str(e))
        print(f"Demo failed: {e}")

if __name__ == "__main__":
    # Run the demo
    asyncio.run(run_demo()) 