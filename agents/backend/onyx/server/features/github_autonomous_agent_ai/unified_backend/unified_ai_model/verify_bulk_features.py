"""
Verification script for Bulk Features (Workflow & Analytics)
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
    from unified_ai_model.core.workflow_engine import WorkflowEngine
    from unified_ai_model.core.analytics import AnalyticsSystem
except ImportError:
    from backend.unified_ai_model.core.continuous_agent import ContinuousAgent
    from backend.unified_ai_model.core.workflow_engine import WorkflowEngine
    from backend.unified_ai_model.core.analytics import AnalyticsSystem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def verify_bulk_features():
    logger.info("Verifying Bulk Features...")
    
    try:
        # 1. Verify Analytics System
        logger.info("1. Testing AnalyticsSystem...")
        analytics = AnalyticsSystem(db_path="./test_data/analytics.db")
        
        analytics.log_task(
            task_id="test-task-1",
            description="Test task",
            priority=1,
            status="completed",
            processing_time=100.5,
            tokens_used=50,
            result_length=200
        )
        
        stats = analytics.get_task_analytics()
        assert stats['total_tasks'] > 0
        logger.info(f"✅ Analytics logged and retrieved: {stats}")
        
        # 2. Verify Workflow Engine
        logger.info("2. Testing WorkflowEngine...")
        engine = WorkflowEngine()
        
        # Create a simple workflow
        workflow = engine.create_workflow(
            workflow_id="test-workflow",
            name="Test Workflow",
            description="A simple test workflow",
            tasks=[
                {
                    "id": "task1",
                    "name": "First Task",
                    "type": "file_operation",
                    "parameters": {
                        "operation": "write",
                        "file_path": "./test_data/test_file.txt",
                        "content": "Hello World"
                    }
                },
                {
                    "id": "task2",
                    "name": "Second Task",
                    "type": "file_operation",
                    "parameters": {
                        "operation": "read",
                        "file_path": "./test_data/test_file.txt"
                    },
                    "dependencies": ["task1"]
                }
            ]
        )
        
        # Execute workflow
        result = await engine.execute_workflow("test-workflow")
        assert result['status'] == "completed"
        assert result['completed_tasks'] == 2
        logger.info("✅ Workflow executed successfully")
        
        # 3. Verify Agent Integration
        logger.info("3. Checking Agent Integration...")
        agent = ContinuousAgent(name="BulkAgent")
        assert agent.workflow_engine is not None
        assert agent.analytics is not None
        logger.info("✅ Agent has workflow_engine and analytics initialized")
        
        logger.info("Bulk features verification PASSED!")
        
    except Exception as e:
        logger.error(f"Bulk features verification FAILED: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(verify_bulk_features())
