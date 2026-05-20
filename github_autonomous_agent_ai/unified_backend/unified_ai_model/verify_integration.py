"""
Verification script for Autonomous Agent Integration
"""
import sys
import os
import asyncio
import logging

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from unified_ai_model.core.continuous_agent import ContinuousAgent
    from unified_ai_model.config import get_config
except ImportError:
    # Fallback for running from project root
    from backend.unified_ai_model.core.continuous_agent import ContinuousAgent
    from backend.unified_ai_model.config import get_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def verify():
    logger.info("Verifying ContinuousAgent integration...")
    
    try:
        # Initialize agent
        agent = ContinuousAgent(name="VerificationAgent")
        
        # Check components
        components = [
            ("knowledge_base", agent.knowledge_base),
            ("learning_engine", agent.learning_engine),
            ("reasoning_engine", agent.reasoning_engine),
            ("autonomous_handler", agent.autonomous_handler),
            ("world_model", agent.world_model),
            ("self_reflection_engine", agent.self_reflection_engine)
        ]
        
        for name, component in components:
            if component:
                logger.info(f"✅ {name} initialized successfully")
            else:
                logger.warning(f"⚠️ {name} is None (might be disabled in config)")
        
        # Check config
        config = get_config()
        logger.info(f"Config autonomous settings: {config.autonomous}")
        
        logger.info("Integration verification PASSED!")
        
    except Exception as e:
        logger.error(f"Integration verification FAILED: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(verify())
