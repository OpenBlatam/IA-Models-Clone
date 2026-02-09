from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
BUFFER_SIZE = 1024

import asyncio
import time
import logging
from pathlib import Path
import tempfile
import os
    from performance_monitor import monitor
    from nlp_utils import analyze_nlp
    from video_pipeline import crear_video_ugc_langchain
    from api import app
    from fastapi.testclient import TestClient
    import psutil
    import gc
        from nlp_utils import analyze_nlp_sync
from typing import Any, List, Dict, Optional
#!/usr/bin/env python3
"""
Test runner for OS Content UGC Video Generator
Validates optimizations and system functionality
"""


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("os_content.tests")

async def test_performance_monitoring():
    """Test performance monitoring functionality"""
    logger.info("Testing performance monitoring...")
    
    
    # Simulate some requests
    for i in range(10):
        monitor.record_request(0.1 + i * 0.01, success=i < 9)
        await asyncio.sleep(0.01)
    
    # Check metrics
    assert monitor.request_count == 10
    assert monitor.get_success_rate() == 90.0
    assert monitor.get_average_processing_time() > 0
    
    logger.info("✅ Performance monitoring test passed")

async def test_nlp_utils():
    """Test NLP utilities"""
    logger.info("Testing NLP utilities...")
    
    
    # Test NLP analysis
    result = await analyze_nlp("Hola mundo", "es")
    assert "tokens" in result
    assert "entities" in result
    
    logger.info("✅ NLP utilities test passed")

async def test_video_pipeline():
    """Test video pipeline functionality"""
    logger.info("Testing video pipeline...")
    
    
    # Create test files
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        f.write(b"fake image data")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        test_image = f.name
    
    try:
        # Test video creation (should handle errors gracefully)
        result = await crear_video_ugc_langchain(
            image_paths=[test_image],
            video_paths=[],
            text_prompt="Test video",
            duration_per_image=3.0
        )
        logger.info("✅ Video pipeline test passed")
    except Exception as e:
        logger.warning(f"Video pipeline test failed (expected for fake data): {e}")
    finally:
        os.unlink(test_image)

async def test_api_endpoints():
    """Test API endpoints"""
    logger.info("Testing API endpoints...")
    
    
    client = TestClient(app)
    
    # Test health endpoint
    response = client.get("/os-content/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    
    # Test metrics endpoint
    response = client.get("/os-content/metrics")
    assert response.status_code == 200
    assert "uptime" in response.json()
    
    logger.info("✅ API endpoints test passed")

async def test_memory_optimization():
    """Test memory optimization"""
    logger.info("Testing memory optimization...")
    
    
    process = psutil.Process()
    initial_memory = process.memory_info().rss
    
    # Simulate some operations
    for i in range(100):
        result = analyze_nlp_sync("Test text for memory optimization", "es")
        gc.collect()
    
    final_memory = process.memory_info().rss
    memory_increase = (final_memory - initial_memory) / 1024 / 1024  # MB
    
    logger.info(f"Memory increase: {memory_increase:.2f} MB")
    assert memory_increase < 100  # Should not increase more than 100MB
    
    logger.info("✅ Memory optimization test passed")

async def run_all_tests():
    """Run all tests"""
    logger.info("Starting OS Content optimization tests...")
    
    start_time = time.time()
    
    try:
        await test_performance_monitoring()
        await test_nlp_utils()
        await test_video_pipeline()
        await test_api_endpoints()
        await test_memory_optimization()
        
        total_time = time.time() - start_time
        logger.info(f"✅ All tests passed in {total_time:.2f} seconds")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        raise

match __name__:
    case "__main__":
    asyncio.run(run_all_tests()) 