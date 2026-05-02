"""
Verification script for Enhanced Autonomous Features
"""
import sys
import os
import asyncio
import logging
from datetime import datetime

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from unified_ai_model.core.continuous_agent import ContinuousAgent
    from unified_ai_model.core.experience_driven_learning import ExperienceDrivenLearning
    from unified_ai_model.core.reasoning_engine import ReasoningEngine
except ImportError:
    from backend.unified_ai_model.core.continuous_agent import ContinuousAgent
    from backend.unified_ai_model.core.experience_driven_learning import ExperienceDrivenLearning
    from backend.unified_ai_model.core.reasoning_engine import ReasoningEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def verify_enhancements():
    logger.info("Verifying Enhanced Features...")
    
    try:
        # 1. Verify Experience Driven Learning
        logger.info("1. Testing ExperienceDrivenLearning...")
        exp_learning = ExperienceDrivenLearning(storage_path="./test_data/experience")
        
        # Add experience
        await exp_learning.add_experience(
            task_id="test-1",
            instruction="Write a python script",
            result="print('hello')",
            outcome="success"
        )
        
        # Retrieve similar
        similar = await exp_learning.retrieve_similar("Write a python script")
        assert len(similar) > 0
        logger.info(f"✅ Retrieved {len(similar)} similar experiences")
        
        formatted = await exp_learning.format_experiences_for_prompt(similar)
        assert "Write a python script" in formatted
        logger.info("✅ Experience formatting correct")
        
        # 2. Verify Reasoning Engine Self-Correction
        logger.info("2. Checking ReasoningEngine methods...")
        # We can't easily test the full LLM loop without mocking, 
        # but we can check if the methods exist
        agent = ContinuousAgent(name="EnhancementAgent")
        
        assert hasattr(agent.reasoning_engine, "_critique_response")
        assert hasattr(agent.reasoning_engine, "_refine_response")
        logger.info("✅ ReasoningEngine has self-correction methods")
        
        # 3. Verify Agent Integration
        logger.info("3. Checking Agent Integration...")
        assert agent.experience_learning is not None
        logger.info("✅ Agent has experience_learning initialized")
        
        logger.info("Enhancement verification PASSED!")
        
    except Exception as e:
        logger.error(f"Enhancement verification FAILED: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(verify_enhancements())
