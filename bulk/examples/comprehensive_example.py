"""
BUL Comprehensive Example
=========================

This example demonstrates all the features of the BUL system including:
- Query submission and processing
- Webhook notifications
- Cache management
- Document generation
- Dashboard monitoring
"""

import asyncio
import logging
import time
from typing import List, Dict, Any

from ..core.bul_engine import BULEngine
from ..core.continuous_processor import ContinuousProcessor
from ..config.bul_config import BULConfig
from ..utils.webhook_manager import WebhookEndpoint, WebhookEvent, get_global_webhook_manager
from ..utils.cache_manager import get_global_cache_manager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BULExample:
    """Comprehensive example of BUL system usage."""
    
    def __init__(self):
        self.config = BULConfig()
        self.engine = BULEngine(self.config)
        self.processor = ContinuousProcessor(self.config)
        self.webhook_manager = get_global_webhook_manager()
        self.cache_manager = get_global_cache_manager()
        
        # Example queries for different business areas
        self.example_queries = [
            {
                "query": "Create a comprehensive marketing strategy for a new coffee shop in downtown",
                "priority": 1,
                "business_area": "marketing",
                "description": "Marketing Strategy for Coffee Shop"
            },
            {
                "query": "Develop a sales process for B2B software sales team with CRM integration",
                "priority": 2,
                "business_area": "sales",
                "description": "B2B Sales Process"
            },
            {
                "query": "Write an operational manual for customer service procedures and escalation",
                "priority": 3,
                "business_area": "operations",
                "description": "Customer Service Operations"
            },
            {
                "query": "Create HR policies for remote work, flexible schedules, and performance management",
                "priority": 2,
                "business_area": "hr",
                "description": "HR Remote Work Policies"
            },
            {
                "query": "Generate a financial plan and budget for a tech startup with 12-month projections",
                "priority": 1,
                "business_area": "finance",
                "description": "Startup Financial Plan"
            },
            {
                "query": "Develop legal compliance policies for data protection and privacy regulations",
                "priority": 2,
                "business_area": "legal",
                "description": "Data Protection Compliance"
            },
            {
                "query": "Create technical documentation for API integration and system architecture",
                "priority": 3,
                "business_area": "technical",
                "description": "API Technical Documentation"
            },
            {
                "query": "Write content strategy for social media marketing and blog posts",
                "priority": 3,
                "business_area": "content",
                "description": "Content Marketing Strategy"
            },
            {
                "query": "Develop business strategy and competitive analysis for market expansion",
                "priority": 1,
                "business_area": "strategy",
                "description": "Business Expansion Strategy"
            },
            {
                "query": "Create customer service training materials and support procedures",
                "priority": 3,
                "business_area": "customer_service",
                "description": "Customer Service Training"
            }
        ]
    
    async def setup_webhooks(self):
        """Setup webhook endpoints for notifications."""
        logger.info("Setting up webhook endpoints...")
        
        # Add a webhook endpoint (in real usage, this would be your actual endpoint)
        webhook_endpoint = WebhookEndpoint(
            url="https://webhook.site/your-webhook-url",  # Replace with actual URL
            events=[
                WebhookEvent.TASK_CREATED,
                WebhookEvent.TASK_COMPLETED,
                WebhookEvent.DOCUMENT_GENERATED,
                WebhookEvent.ERROR_OCCURRED
            ],
            secret="your-webhook-secret",
            timeout=30,
            retry_attempts=3
        )
        
        self.webhook_manager.add_endpoint(webhook_endpoint)
        logger.info("Webhook endpoint configured")
    
    async def setup_cache_demo(self):
        """Demonstrate cache functionality."""
        logger.info("Demonstrating cache functionality...")
        
        # Cache some example data
        await self.cache_manager.set("example_query", "Create a marketing strategy", ttl=300)
        await self.cache_manager.set("example_result", {"documents": 3, "processing_time": 45.2}, ttl=600)
        
        # Retrieve from cache
        cached_query = await self.cache_manager.get("example_query")
        cached_result = await self.cache_manager.get("example_result")
        
        logger.info(f"Cached query: {cached_query}")
        logger.info(f"Cached result: {cached_result}")
        
        # Show cache stats
        stats = self.cache_manager.get_stats()
        logger.info(f"Cache stats: {stats}")
    
    async def submit_sample_queries(self) -> List[str]:
        """Submit sample queries to the system."""
        logger.info("Submitting sample queries...")
        
        task_ids = []
        
        for i, query_data in enumerate(self.example_queries[:5]):  # Submit first 5 queries
            try:
                task_id = await self.engine.submit_query(
                    query=query_data["query"],
                    priority=query_data["priority"]
                )
                task_ids.append(task_id)
                logger.info(f"Submitted query {i+1}: {query_data['description']} (ID: {task_id})")
                
                # Small delay between submissions
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Failed to submit query {i+1}: {e}")
        
        return task_ids
    
    async def monitor_processing(self, task_ids: List[str]):
        """Monitor the processing of submitted tasks."""
        logger.info("Monitoring task processing...")
        
        completed_tasks = set()
        max_wait_time = 300  # 5 minutes max wait
        start_time = time.time()
        
        while len(completed_tasks) < len(task_ids) and (time.time() - start_time) < max_wait_time:
            for task_id in task_ids:
                if task_id in completed_tasks:
                    continue
                
                status = self.engine.get_task_status(task_id)
                if status:
                    logger.info(f"Task {task_id[:8]}... - Status: {status['status']}")
                    
                    if status['status'] == 'completed':
                        completed_tasks.add(task_id)
                        
                        # Get generated documents
                        documents = self.engine.get_completed_documents(task_id)
                        if documents:
                            logger.info(f"Task {task_id[:8]}... generated {len(documents)} documents")
                            
                            # Show document details
                            for doc in documents:
                                logger.info(f"  - Document: {doc['type']} ({doc['business_area']})")
                                logger.info(f"    File: {doc['file_path']}")
            
            if len(completed_tasks) < len(task_ids):
                await asyncio.sleep(5)  # Wait 5 seconds before checking again
        
        logger.info(f"Processing monitoring completed. {len(completed_tasks)}/{len(task_ids)} tasks completed")
    
    async def demonstrate_agents(self):
        """Demonstrate the business area agents."""
        logger.info("Demonstrating business area agents...")
        
        agent_manager = self.engine.sme_agent_manager
        agents_info = agent_manager.get_all_agents_info()
        
        for area, info in agents_info.items():
            if info:
                logger.info(f"\n{area.title()} Agent:")
                logger.info(f"  Document Types: {', '.join(info['supported_document_types'])}")
                logger.info(f"  Expertise Areas: {', '.join(info['expertise_areas'][:3])}...")
                logger.info(f"  Priority: {info['priority']}")
    
    async def show_system_stats(self):
        """Show comprehensive system statistics."""
        logger.info("System Statistics:")
        logger.info("=" * 50)
        
        # Engine stats
        engine_stats = self.engine.get_processing_stats()
        logger.info(f"Engine Stats:")
        logger.info(f"  Total Tasks: {engine_stats['total_tasks']}")
        logger.info(f"  Completed Tasks: {engine_stats['completed_tasks']}")
        logger.info(f"  Failed Tasks: {engine_stats['failed_tasks']}")
        logger.info(f"  Active Tasks: {engine_stats['active_tasks']}")
        logger.info(f"  Queued Tasks: {engine_stats['queued_tasks']}")
        logger.info(f"  Average Processing Time: {engine_stats['average_processing_time']:.2f}s")
        logger.info(f"  Is Processing: {engine_stats['is_processing']}")
        
        # Cache stats
        cache_stats = self.cache_manager.get_stats()
        logger.info(f"\nCache Stats:")
        logger.info(f"  Hits: {cache_stats['hits']}")
        logger.info(f"  Misses: {cache_stats['misses']}")
        logger.info(f"  Hit Rate: {cache_stats['hit_rate']}%")
        logger.info(f"  Memory Entries: {cache_stats['memory_entries']}")
        
        # Webhook stats
        webhook_endpoints = self.webhook_manager.get_endpoints()
        logger.info(f"\nWebhook Stats:")
        logger.info(f"  Configured Endpoints: {len(webhook_endpoints)}")
        
        # Document processor stats
        doc_stats = self.engine.document_processor.get_statistics()
        logger.info(f"\nDocument Stats:")
        logger.info(f"  Total Documents: {doc_stats['total_documents']}")
        logger.info(f"  Total Size: {doc_stats['total_size_mb']} MB")
        logger.info(f"  Documents by Area: {doc_stats['documents_by_area']}")
    
    async def run_comprehensive_demo(self):
        """Run the comprehensive demonstration."""
        logger.info("🚀 Starting BUL Comprehensive Example")
        logger.info("=" * 60)
        
        try:
            # Setup components
            await self.setup_webhooks()
            await self.setup_cache_demo()
            
            # Demonstrate agents
            await self.demonstrate_agents()
            
            # Submit queries
            task_ids = await self.submit_sample_queries()
            
            # Monitor processing
            await self.monitor_processing(task_ids)
            
            # Show final stats
            await self.show_system_stats()
            
            logger.info("\n🎉 Comprehensive example completed successfully!")
            logger.info("Check the generated_documents directory for output files")
            logger.info("Visit http://localhost:8000/dashboard for the web interface")
            logger.info("API documentation available at http://localhost:8000/docs")
            
        except Exception as e:
            logger.error(f"Error in comprehensive demo: {e}")
            raise

async def main():
    """Main function to run the comprehensive example."""
    example = BULExample()
    await example.run_comprehensive_demo()

if __name__ == "__main__":
    asyncio.run(main())

