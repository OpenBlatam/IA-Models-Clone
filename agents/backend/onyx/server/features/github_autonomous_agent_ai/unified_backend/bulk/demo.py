"""
BUL Demo Script
===============

Demonstration script showing how to use the BUL system.
"""

import asyncio
import logging
from typing import List

from .core.bul_engine import BULEngine
from .config.bul_config import BULConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def demo_bul_system():
    """Demonstrate the BUL system capabilities."""
    
    print("🚀 BUL - Business Unlimited Demo")
    print("=" * 50)
    
    # Initialize system
    config = BULConfig()
    engine = BULEngine(config)
    
    # Demo queries for different business areas
    demo_queries = [
        {
            "query": "Create a comprehensive marketing strategy for a new coffee shop",
            "priority": 1,
            "description": "Marketing Strategy Generation"
        },
        {
            "query": "Develop a sales process for B2B software sales team",
            "priority": 2,
            "description": "Sales Process Development"
        },
        {
            "query": "Write an operational manual for customer service procedures",
            "priority": 3,
            "description": "Operations Manual Creation"
        },
        {
            "query": "Create HR policies for remote work and flexible schedules",
            "priority": 2,
            "description": "HR Policy Development"
        },
        {
            "query": "Generate a financial plan and budget for a startup",
            "priority": 1,
            "description": "Financial Planning"
        }
    ]
    
    print(f"📋 Submitting {len(demo_queries)} demo queries...")
    
    # Submit queries
    task_ids = []
    for i, query_data in enumerate(demo_queries, 1):
        print(f"\n{i}. {query_data['description']}")
        print(f"   Query: {query_data['query']}")
        
        try:
            task_id = await engine.submit_query(
                query=query_data['query'],
                priority=query_data['priority']
            )
            task_ids.append(task_id)
            print(f"   ✅ Task ID: {task_id}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print(f"\n📊 Processing Statistics:")
    stats = engine.get_processing_stats()
    print(f"   Total Tasks: {stats['total_tasks']}")
    print(f"   Active Tasks: {stats['active_tasks']}")
    print(f"   Queued Tasks: {stats['queued_tasks']}")
    print(f"   Enabled Areas: {', '.join(stats['enabled_areas'])}")
    
    # Monitor processing
    print(f"\n⏳ Monitoring task processing...")
    print("   (In a real scenario, the continuous processor would handle this)")
    
    # Simulate checking task status
    for task_id in task_ids:
        status = engine.get_task_status(task_id)
        if status:
            print(f"   Task {task_id[:8]}... - Status: {status['status']}")
    
    print(f"\n📁 Document Storage:")
    print(f"   Output Directory: {config.get_output_directory()}")
    print(f"   Supported Formats: {', '.join(config.document.supported_formats)}")
    
    print(f"\n🔧 System Configuration:")
    print(f"   Max Concurrent Tasks: {config.processing.max_concurrent_tasks}")
    print(f"   Task Timeout: {config.processing.task_timeout}s")
    print(f"   Continuous Mode: {config.processing.continuous_mode}")
    
    print(f"\n🎯 Available Business Areas:")
    for area in config.sme.enabled_areas:
        priority = config.get_area_priority(area)
        print(f"   - {area.title()} (Priority: {priority})")
    
    print(f"\n✨ Demo completed!")
    print(f"   The BUL system is ready to process business queries continuously.")
    print(f"   Use the API endpoints or Python interface to submit queries.")

def demo_query_analysis():
    """Demonstrate query analysis capabilities."""
    
    print("\n🔍 Query Analysis Demo")
    print("=" * 30)
    
    from .utils.query_analyzer import QueryAnalyzer
    
    analyzer = QueryAnalyzer()
    
    test_queries = [
        "Create a marketing strategy for a new restaurant",
        "Develop HR policies for remote work",
        "Write a technical manual for software deployment",
        "Generate a financial analysis report",
        "Create a customer service training program"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        analysis = analyzer.analyze(query)
        print(f"  Primary Area: {analysis.primary_area}")
        print(f"  Secondary Areas: {', '.join(analysis.secondary_areas)}")
        print(f"  Document Types: {', '.join(analysis.document_types)}")
        print(f"  Complexity: {analysis.complexity.value}")
        print(f"  Priority: {analysis.priority}")
        print(f"  Confidence: {analysis.confidence:.2f}")

def demo_agent_info():
    """Demonstrate agent information."""
    
    print("\n🤖 Business Agent Demo")
    print("=" * 30)
    
    from .agents.sme_agent_manager import SMEAgentManager
    from .config.bul_config import BULConfig
    
    config = BULConfig()
    agent_manager = SMEAgentManager(config)
    
    print("Available Agents:")
    agents_info = agent_manager.get_all_agents_info()
    
    for area, info in agents_info.items():
        if info:
            print(f"\n{area.title()} Agent:")
            print(f"  Document Types: {', '.join(info['supported_document_types'])}")
            print(f"  Expertise Areas: {', '.join(info['expertise_areas'][:3])}...")
            print(f"  Priority: {info['priority']}")

if __name__ == "__main__":
    print("Starting BUL Demo...")
    
    # Run query analysis demo
    demo_query_analysis()
    
    # Run agent info demo
    demo_agent_info()
    
    # Run main system demo
    asyncio.run(demo_bul_system())
    
    print("\n🎉 Demo completed successfully!")
    print("   The BUL system is ready for production use.")

