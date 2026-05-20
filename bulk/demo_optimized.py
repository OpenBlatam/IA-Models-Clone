"""
BUL Optimized Demo Script
=========================

Demonstrates the optimized BUL system capabilities.
"""

import asyncio
import logging
import sys
from pathlib import Path
from datetime import datetime

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from modules.document_processor import DocumentProcessor
from modules.query_analyzer import QueryAnalyzer
from modules.business_agents import BusinessAgentManager
from modules.api_handler import APIHandler, DocumentRequest
from config_optimized import BULConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BULDemo:
    """BUL system demonstration class."""
    
    def __init__(self):
        self.config = BULConfig()
        self.processor = DocumentProcessor(self.config.to_dict())
        self.analyzer = QueryAnalyzer()
        self.agent_manager = BusinessAgentManager(self.config.to_dict())
        self.api_handler = APIHandler(
            self.processor, self.analyzer, self.agent_manager
        )
    
    async def demo_query_analysis(self):
        """Demonstrate query analysis capabilities."""
        print("\n🔍 Query Analysis Demo")
        print("=" * 40)
        
        demo_queries = [
            "Create a comprehensive marketing strategy for a new restaurant",
            "Develop a sales process for B2B software sales team",
            "Write an operational manual for customer service procedures",
            "Create HR policies for remote work and flexible schedules",
            "Generate a financial plan and budget for a startup"
        ]
        
        for i, query in enumerate(demo_queries, 1):
            print(f"\n{i}. Query: {query}")
            
            analysis = self.analyzer.analyze(query)
            
            print(f"   📊 Primary Area: {analysis.primary_area}")
            print(f"   📋 Document Types: {', '.join(analysis.document_types)}")
            print(f"   🎯 Complexity: {analysis.complexity.value}")
            print(f"   ⭐ Priority: {analysis.priority}")
            print(f"   🎯 Confidence: {analysis.confidence:.2f}")
            
            if analysis.secondary_areas:
                print(f"   🔗 Secondary Areas: {', '.join(analysis.secondary_areas)}")
    
    async def demo_business_agents(self):
        """Demonstrate business agent capabilities."""
        print("\n🤖 Business Agents Demo")
        print("=" * 40)
        
        available_areas = self.agent_manager.get_available_areas()
        print(f"Available Business Areas: {', '.join(available_areas)}")
        
        for area in available_areas:
            capabilities = self.agent_manager.get_agent_capabilities(area)
            if capabilities:
                print(f"\n{area.title()} Agent:")
                print(f"   📄 Document Types: {', '.join(capabilities['supported_document_types'])}")
                print(f"   ⭐ Priority: {capabilities['priority']}")
    
    async def demo_document_generation(self):
        """Demonstrate document generation."""
        print("\n📄 Document Generation Demo")
        print("=" * 40)
        
        demo_requests = [
            {
                "query": "Create a marketing strategy for a new coffee shop",
                "business_area": "marketing",
                "document_type": "strategy",
                "priority": 1
            },
            {
                "query": "Develop a sales proposal for enterprise software",
                "business_area": "sales", 
                "document_type": "proposal",
                "priority": 2
            },
            {
                "query": "Write an operational manual for customer support",
                "business_area": "operations",
                "document_type": "manual",
                "priority": 3
            }
        ]
        
        for i, request_data in enumerate(demo_requests, 1):
            print(f"\n{i}. Generating {request_data['business_area']} {request_data['document_type']}")
            
            try:
                # Create request
                request = DocumentRequest(**request_data)
                
                # Generate document
                response = await self.api_handler.generate_document(request)
                
                print(f"   ✅ Task ID: {response.task_id}")
                print(f"   📊 Status: {response.status}")
                print(f"   ⏱️  Estimated Time: {response.estimated_time}s")
                
                # Simulate processing time
                await asyncio.sleep(0.5)
                
                # Check task status
                status = await self.api_handler.get_task_status(response.task_id)
                print(f"   📈 Progress: {status.progress}%")
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
    
    async def demo_system_info(self):
        """Demonstrate system information."""
        print("\n📊 System Information Demo")
        print("=" * 40)
        
        # System info
        system_info = self.api_handler.get_system_info()
        print(f"System: {system_info['system']}")
        print(f"Version: {system_info['version']}")
        print(f"Status: {system_info['status']}")
        print(f"Available Areas: {', '.join(system_info['available_areas'])}")
        
        # Health status
        health = self.api_handler.get_health_status()
        print(f"\nHealth Status: {health['status']}")
        print("Components:")
        for component, status in health['components'].items():
            print(f"   {component}: {status}")
        
        # Configuration
        print(f"\nConfiguration:")
        print(f"   API Host: {self.config.api_host}")
        print(f"   API Port: {self.config.api_port}")
        print(f"   Debug Mode: {self.config.debug_mode}")
        print(f"   Max Concurrent Tasks: {self.config.max_concurrent_tasks}")
        print(f"   Output Directory: {self.config.output_directory}")
    
    async def demo_task_management(self):
        """Demonstrate task management."""
        print("\n📋 Task Management Demo")
        print("=" * 40)
        
        # List current tasks
        tasks = self.api_handler.list_tasks()
        print(f"Current Tasks: {len(tasks['tasks'])}")
        
        if tasks['tasks']:
            print("\nRecent Tasks:")
            for task in tasks['tasks'][-3:]:  # Show last 3 tasks
                print(f"   {task['task_id'][:12]}... - {task['status']} ({task['progress']}%)")
        else:
            print("   No tasks currently active")
    
    async def run_full_demo(self):
        """Run the complete demonstration."""
        print("🚀 BUL - Business Universal Language (Optimized)")
        print("=" * 60)
        print("Comprehensive System Demonstration")
        print("=" * 60)
        
        try:
            # Run all demos
            await self.demo_system_info()
            await self.demo_query_analysis()
            await self.demo_business_agents()
            await self.demo_document_generation()
            await self.demo_task_management()
            
            print("\n" + "=" * 60)
            print("🎉 Demo completed successfully!")
            print("=" * 60)
            
            print("\n🚀 Next Steps:")
            print("1. Start the system: python start_optimized.py")
            print("2. Access API docs: http://localhost:8000/docs")
            print("3. Run tests: python test_optimized.py")
            print("4. Validate system: python validate_system.py")
            
        except Exception as e:
            print(f"\n❌ Demo error: {e}")
            logger.error(f"Demo error: {e}")

async def main():
    """Main demo function."""
    demo = BULDemo()
    await demo.run_full_demo()

if __name__ == "__main__":
    asyncio.run(main())
