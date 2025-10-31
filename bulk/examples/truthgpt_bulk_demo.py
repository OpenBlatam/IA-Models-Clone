"""
TruthGPT Bulk Document Generation Demo
=====================================

Demonstration script showing how to use the TruthGPT-inspired bulk document
generation system to create multiple documents with a single request.
"""

import asyncio
import logging
import json
from datetime import datetime
from pathlib import Path
import sys
import os

# Add the parent directory to the path to import our modules
sys.path.append(str(Path(__file__).parent.parent))

from core.truthgpt_bulk_processor import TruthGPTBulkProcessor, get_global_truthgpt_processor
from config.bul_config import BULConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TruthGPTBulkDemo:
    """Demo class for TruthGPT bulk document generation."""
    
    def __init__(self):
        self.processor = get_global_truthgpt_processor()
        self.setup_callbacks()
    
    def setup_callbacks(self):
        """Setup callbacks for monitoring document generation."""
        
        async def on_document_generated(task, processed_doc):
            """Callback when a document is generated."""
            logger.info(f"✅ Document Generated: {task.document_type} for {task.business_area}")
            logger.info(f"   Task ID: {task.id}")
            logger.info(f"   Content Length: {len(task.content)} characters")
            
            # Save document to file
            await self.save_document_to_file(task, processed_doc)
        
        async def on_request_completed(request_id, result):
            """Callback when a request is completed."""
            logger.info(f"🎉 Request Completed: {request_id}")
            logger.info(f"   Documents Generated: {result.documents_generated}")
            logger.info(f"   Processing Time: {result.processing_time:.2f} seconds")
        
        async def on_error(task, error):
            """Callback when an error occurs."""
            logger.error(f"❌ Error in task {task.id}: {error}")
        
        # Set callbacks
        self.processor.set_document_callback(on_document_generated)
        self.processor.set_request_callback(on_request_completed)
        self.processor.set_error_callback(on_error)
    
    async def save_document_to_file(self, task, processed_doc):
        """Save generated document to file."""
        try:
            # Create output directory
            output_dir = Path("generated_documents")
            output_dir.mkdir(exist_ok=True)
            
            # Create filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{task.document_type}_{task.business_area}_{timestamp}.md"
            filepath = output_dir / filename
            
            # Write content to file
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"# {task.document_type.replace('_', ' ').title()}\n")
                f.write(f"**Business Area:** {task.business_area}\n")
                f.write(f"**Generated:** {datetime.now().isoformat()}\n")
                f.write(f"**Task ID:** {task.id}\n\n")
                f.write("---\n\n")
                f.write(task.content)
            
            logger.info(f"💾 Document saved: {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save document: {e}")
    
    async def demo_basic_bulk_generation(self):
        """Demo basic bulk document generation."""
        logger.info("🚀 Starting Basic Bulk Generation Demo")
        
        # Submit a bulk request
        request_id = await self.processor.submit_bulk_request(
            query="Digital transformation strategies for small businesses",
            document_types=["business_plan", "marketing_strategy", "operational_manual"],
            business_areas=["strategy", "marketing", "operations"],
            max_documents=9,  # 3 types × 3 areas
            continuous_mode=True,
            priority=1
        )
        
        logger.info(f"📝 Request submitted: {request_id}")
        
        # Monitor progress
        await self.monitor_request_progress(request_id)
        
        return request_id
    
    async def demo_continuous_generation(self):
        """Demo continuous document generation."""
        logger.info("🔄 Starting Continuous Generation Demo")
        
        # Submit a large bulk request
        request_id = await self.processor.submit_bulk_request(
            query="AI implementation in modern businesses",
            document_types=[
                "business_plan", "marketing_strategy", "technical_documentation",
                "financial_analysis", "hr_policy", "operational_manual"
            ],
            business_areas=[
                "strategy", "marketing", "technical", "finance", "hr", "operations"
            ],
            max_documents=50,  # Large number for continuous generation
            continuous_mode=True,
            priority=1
        )
        
        logger.info(f"📝 Large request submitted: {request_id}")
        
        # Monitor progress with periodic updates
        await self.monitor_request_progress(request_id, update_interval=10)
        
        return request_id
    
    async def demo_multiple_requests(self):
        """Demo multiple concurrent requests."""
        logger.info("🎯 Starting Multiple Requests Demo")
        
        # Submit multiple requests
        requests = []
        
        # Request 1: Marketing focus
        request1 = await self.processor.submit_bulk_request(
            query="Social media marketing strategies",
            document_types=["marketing_strategy", "content_strategy"],
            business_areas=["marketing", "content"],
            max_documents=10,
            continuous_mode=True,
            priority=1
        )
        requests.append(request1)
        
        # Request 2: Technical focus
        request2 = await self.processor.submit_bulk_request(
            query="Cloud infrastructure setup",
            document_types=["technical_documentation", "operational_manual"],
            business_areas=["technical", "operations"],
            max_documents=8,
            continuous_mode=True,
            priority=2
        )
        requests.append(request2)
        
        # Request 3: Business focus
        request3 = await self.processor.submit_bulk_request(
            query="Financial planning and budgeting",
            document_types=["financial_analysis", "business_plan"],
            business_areas=["finance", "strategy"],
            max_documents=6,
            continuous_mode=True,
            priority=1
        )
        requests.append(request3)
        
        logger.info(f"📝 Multiple requests submitted: {requests}")
        
        # Monitor all requests
        await self.monitor_multiple_requests(requests)
        
        return requests
    
    async def monitor_request_progress(self, request_id: str, update_interval: int = 5):
        """Monitor the progress of a single request."""
        logger.info(f"👀 Monitoring request: {request_id}")
        
        while True:
            try:
                status = await self.processor.get_request_status(request_id)
                if not status:
                    logger.warning(f"Request {request_id} not found")
                    break
                
                logger.info(f"📊 Progress: {status['documents_generated']}/{status['max_documents']} "
                          f"({status['progress_percentage']:.1f}%) - "
                          f"Active: {status['active_tasks']}, Queued: {status['queued_tasks']}")
                
                # Check if completed
                if status['documents_generated'] >= status['max_documents']:
                    logger.info(f"✅ Request {request_id} completed!")
                    break
                
                await asyncio.sleep(update_interval)
                
            except Exception as e:
                logger.error(f"Error monitoring request: {e}")
                break
    
    async def monitor_multiple_requests(self, request_ids: List[str]):
        """Monitor multiple requests concurrently."""
        logger.info(f"👀 Monitoring {len(request_ids)} requests")
        
        tasks = []
        for request_id in request_ids:
            task = asyncio.create_task(self.monitor_request_progress(request_id, 10))
            tasks.append(task)
        
        # Wait for all monitoring tasks to complete
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def get_final_results(self, request_id: str):
        """Get final results for a request."""
        try:
            # Get all documents
            documents = await self.processor.get_request_documents(request_id)
            
            logger.info(f"📋 Final Results for {request_id}:")
            logger.info(f"   Total Documents: {len(documents)}")
            
            # Group by document type
            by_type = {}
            for doc in documents:
                doc_type = doc['document_type']
                if doc_type not in by_type:
                    by_type[doc_type] = 0
                by_type[doc_type] += 1
            
            logger.info("   Documents by Type:")
            for doc_type, count in by_type.items():
                logger.info(f"     {doc_type}: {count}")
            
            # Group by business area
            by_area = {}
            for doc in documents:
                area = doc['business_area']
                if area not in by_area:
                    by_area[area] = 0
                by_area[area] += 1
            
            logger.info("   Documents by Business Area:")
            for area, count in by_area.items():
                logger.info(f"     {area}: {count}")
            
            return documents
            
        except Exception as e:
            logger.error(f"Error getting final results: {e}")
            return []
    
    async def run_comprehensive_demo(self):
        """Run a comprehensive demo of all features."""
        logger.info("🎬 Starting Comprehensive TruthGPT Bulk Generation Demo")
        logger.info("=" * 60)
        
        try:
            # Start the processor
            logger.info("🚀 Starting continuous processing...")
            asyncio.create_task(self.processor.start_continuous_processing())
            await asyncio.sleep(2)  # Give it time to start
            
            # Demo 1: Basic bulk generation
            logger.info("\n📝 Demo 1: Basic Bulk Generation")
            logger.info("-" * 40)
            request1 = await self.demo_basic_bulk_generation()
            await self.get_final_results(request1)
            
            await asyncio.sleep(5)  # Brief pause
            
            # Demo 2: Multiple concurrent requests
            logger.info("\n📝 Demo 2: Multiple Concurrent Requests")
            logger.info("-" * 40)
            requests = await self.demo_multiple_requests()
            
            # Get results for all requests
            for request_id in requests:
                await self.get_final_results(request_id)
            
            await asyncio.sleep(5)  # Brief pause
            
            # Demo 3: Large continuous generation
            logger.info("\n📝 Demo 3: Large Continuous Generation")
            logger.info("-" * 40)
            request3 = await self.demo_continuous_generation()
            await self.get_final_results(request3)
            
            # Final statistics
            logger.info("\n📊 Final Processing Statistics")
            logger.info("-" * 40)
            stats = self.processor.get_processing_stats()
            for key, value in stats.items():
                logger.info(f"   {key}: {value}")
            
            logger.info("\n🎉 Demo completed successfully!")
            
        except Exception as e:
            logger.error(f"Demo failed: {e}")
        finally:
            # Stop processing
            self.processor.stop_processing()
            logger.info("🛑 Processing stopped")

async def main():
    """Main function to run the demo."""
    demo = TruthGPTBulkDemo()
    await demo.run_comprehensive_demo()

if __name__ == "__main__":
    # Check if OpenRouter API key is configured
    if not os.getenv("OPENROUTER_API_KEY"):
        logger.error("❌ OPENROUTER_API_KEY environment variable not set!")
        logger.info("Please set your OpenRouter API key:")
        logger.info("export OPENROUTER_API_KEY='your-api-key-here'")
        sys.exit(1)
    
    # Run the demo
    asyncio.run(main())



























