"""
Test AI Intelligence Module

This script tests the AI Intelligence module capabilities including
NLP, computer vision, reasoning, and multimodal processing.
"""

import asyncio
import logging
import time
from typing import Dict, Any

# Import the AI Intelligence module
from modules.ai_intelligence import (
    AIIntelligenceModule,
    AIIntelligenceConfig,
    AITaskType,
    ReasoningType,
    create_ai_intelligence_module
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# TEST FUNCTIONS
# ============================================================================

async def test_nlp_processing(ai_module: AIIntelligenceModule):
    """Test NLP processing capabilities."""
    logger.info("🧠 Testing NLP Processing...")
    
    test_cases = [
        ("I love this amazing product!", "sentiment"),
        ("This is a terrible experience.", "sentiment"),
        ("The weather is nice today.", "sentiment"),
        ("Technology is advancing rapidly.", "classification"),
        ("Machine learning algorithms are powerful.", "classification"),
        ("This is a long text that needs to be summarized into a shorter version.", "summarization"),
        ("Hello world, how are you?", "translation"),
        ("Artificial intelligence is transforming industries.", "analysis")
    ]
    
    for text, task in test_cases:
        try:
            result = await ai_module.process_nlp_task(text, task)
            logger.info(f"✅ {task}: '{text[:30]}...' -> {result['result']}")
        except Exception as e:
            logger.error(f"❌ {task} failed: {e}")

async def test_vision_processing(ai_module: AIIntelligenceModule):
    """Test computer vision processing capabilities."""
    logger.info("👁️ Testing Vision Processing...")
    
    # Simulate different image data sizes
    test_images = [
        (b"small_image_data", "object_detection"),
        (b"medium_image_data_for_testing", "classification"),
        (b"large_image_data_for_comprehensive_testing", "segmentation"),
        (b"face_image_data_for_recognition", "face_recognition"),
        (b"general_image_data", "analysis")
    ]
    
    for image_data, task in test_images:
        try:
            result = await ai_module.process_vision_task(image_data, task)
            logger.info(f"✅ {task}: {len(image_data)} bytes -> {result['result']}")
        except Exception as e:
            logger.error(f"❌ {task} failed: {e}")

async def test_reasoning_capabilities(ai_module: AIIntelligenceModule):
    """Test automated reasoning capabilities."""
    logger.info("🧮 Testing Reasoning Capabilities...")
    
    test_queries = [
        ("If all humans are mortal and Socrates is human, what can we conclude?", ReasoningType.LOGICAL),
        ("What is the symbolic representation of a logical AND operation?", ReasoningType.SYMBOLIC),
        ("How certain are we about this fuzzy classification?", ReasoningType.FUZZY),
        ("What is the quantum state of this reasoning problem?", ReasoningType.QUANTUM),
        ("Can you provide a general analysis of this situation?", ReasoningType.LOGICAL)
    ]
    
    for query, reasoning_type in test_queries:
        try:
            result = await ai_module.process_reasoning_task(query, reasoning_type)
            logger.info(f"✅ {reasoning_type.value}: '{query[:40]}...' -> {result['result']['conclusion']}")
        except Exception as e:
            logger.error(f"❌ {reasoning_type.value} failed: {e}")

async def test_multimodal_processing(ai_module: AIIntelligenceModule):
    """Test multimodal processing capabilities."""
    logger.info("🔄 Testing Multimodal Processing...")
    
    test_cases = [
        ("A beautiful sunset over mountains", b"sunset_image_data", "analysis"),
        ("A busy city street with cars and people", b"city_image_data", "detection"),
        ("A peaceful forest with tall trees", b"forest_image_data", "classification"),
        ("A modern office building", b"office_image_data", "description")
    ]
    
    for text, image_data, task in test_cases:
        try:
            result = await ai_module.process_multimodal_task(text, image_data, task)
            logger.info(f"✅ Multimodal {task}: '{text[:30]}...' + {len(image_data)} bytes -> Success: {result['success']}")
        except Exception as e:
            logger.error(f"❌ Multimodal {task} failed: {e}")

async def test_task_management(ai_module: AIIntelligenceModule):
    """Test task management and queuing capabilities."""
    logger.info("📋 Testing Task Management...")
    
    from modules.ai_intelligence import AITask
    
    # Create test tasks
    tasks = [
        AITask(
            task_id="nlp_001",
            task_type=AITaskType.NLP,
            input_data={"text": "Test NLP task", "task": "sentiment"},
            priority=1
        ),
        AITask(
            task_id="vision_001",
            task_type=AITaskType.VISION,
            input_data={"image_data": b"test_image", "task": "detection"},
            priority=2
        ),
        AITask(
            task_id="reasoning_001",
            task_type=AITaskType.REASONING,
            input_data={"query": "Test reasoning", "reasoning_type": "logical"},
            priority=3
        )
    ]
    
    # Add tasks to queue
    for task in tasks:
        task_id = await ai_module.add_task(task)
        logger.info(f"✅ Added task {task_id} to queue")
    
    # Wait for tasks to be processed
    await asyncio.sleep(2)
    
    # Check task status
    for task in tasks:
        status = await ai_module.get_task_status(task.task_id)
        if status:
            logger.info(f"📊 Task {task.task_id}: {status['status']} in {status.get('processing_time', 0):.3f}s")
        else:
            logger.warning(f"⚠️ Task {task.task_id}: Status not found")

async def test_performance_benchmark(ai_module: AIIntelligenceModule):
    """Test performance and benchmark capabilities."""
    logger.info("⚡ Testing Performance Benchmark...")
    
    # Test NLP performance
    start_time = time.perf_counter()
    nlp_results = []
    for i in range(10):
        result = await ai_module.process_nlp_task(f"Test text {i}", "sentiment")
        nlp_results.append(result)
    nlp_time = time.perf_counter() - start_time
    
    # Test vision performance
    start_time = time.perf_counter()
    vision_results = []
    for i in range(10):
        result = await ai_module.process_vision_task(f"image_data_{i}".encode(), "detection")
        vision_results.append(result)
    vision_time = time.perf_counter() - start_time
    
    # Test reasoning performance
    start_time = time.perf_counter()
    reasoning_results = []
    for i in range(10):
        result = await ai_module.process_reasoning_task(f"Test query {i}", ReasoningType.LOGICAL)
        reasoning_results.append(result)
    reasoning_time = time.perf_counter() - start_time
    
    logger.info(f"📊 Performance Results:")
    logger.info(f"   NLP: 10 tasks in {nlp_time:.3f}s ({nlp_time/10:.3f}s per task)")
    logger.info(f"   Vision: 10 tasks in {vision_time:.3f}s ({vision_time/10:.3f}s per task)")
    logger.info(f"   Reasoning: 10 tasks in {reasoning_time:.3f}s ({reasoning_time/10:.3f}s per task)")

async def test_health_and_metrics(ai_module: AIIntelligenceModule):
    """Test health checking and metrics collection."""
    logger.info("🏥 Testing Health and Metrics...")
    
    # Check module health
    health = await ai_module.health_check()
    logger.info(f"✅ Module Health: {health['status']}")
    logger.info(f"   Active tasks: {health.get('active_tasks', 0)}")
    logger.info(f"   Queued tasks: {health.get('queued_tasks', 0)}")
    
    # Check engine health
    if 'engines' in health:
        for engine_name, engine_health in health['engines'].items():
            if engine_health:
                logger.info(f"   {engine_name} engine: {engine_health.get('status', 'unknown')}")
    
    # Get metrics
    metrics = await ai_module.get_metrics()
    logger.info(f"📈 Module Metrics:")
    logger.info(f"   Total tasks: {metrics.total_tasks_processed}")
    logger.info(f"   NLP tasks: {metrics.nlp_tasks}")
    logger.info(f"   Vision tasks: {metrics.vision_tasks}")
    logger.info(f"   Reasoning tasks: {metrics.reasoning_tasks}")
    logger.info(f"   Multimodal tasks: {metrics.multimodal_tasks}")
    logger.info(f"   Average processing time: {metrics.average_processing_time:.3f}s")
    logger.info(f"   Success rate: {metrics.success_rate:.1f}%")

# ============================================================================
# MAIN TEST FUNCTION
# ============================================================================

async def main():
    """Main test function."""
    logger.info("🚀 Starting AI Intelligence Module Tests")
    
    # Create AI Intelligence module
    config = AIIntelligenceConfig(
        name="test_ai_intelligence",
        enable_nlp=True,
        enable_vision=True,
        enable_reasoning=True,
        enable_multimodal=True,
        max_concurrent_tasks=5,
        enable_quantum_optimization=True,
        enable_neural_acceleration=True,
        enable_real_time_processing=True
    )
    
    ai_module = create_ai_intelligence_module(config)
    
    try:
        # Initialize module
        logger.info("🔧 Initializing AI Intelligence module...")
        success = await ai_module.initialize()
        if not success:
            logger.error("❌ Failed to initialize AI Intelligence module")
            return
        
        logger.info("✅ AI Intelligence module initialized successfully")
        
        # Run tests
        await test_nlp_processing(ai_module)
        await test_vision_processing(ai_module)
        await test_reasoning_capabilities(ai_module)
        await test_multimodal_processing(ai_module)
        await test_task_management(ai_module)
        await test_performance_benchmark(ai_module)
        await test_health_and_metrics(ai_module)
        
        logger.info("🎉 All tests completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        raise
    
    finally:
        # Shutdown module
        logger.info("🔄 Shutting down AI Intelligence module...")
        await ai_module.shutdown()
        logger.info("✅ AI Intelligence module shutdown completed")

# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    asyncio.run(main())
