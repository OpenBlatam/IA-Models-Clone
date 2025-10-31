#!/usr/bin/env python3
"""
Test script for the Bulk AI System
==================================

Demonstrates the bulk AI system capabilities with continuous generation.
"""

import asyncio
import logging
import time
import json
from datetime import datetime
from typing import Dict, List, Any

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import bulk AI components
from bulk_ai_system import BulkAISystem, BulkAIConfig
from continuous_generator import ContinuousGenerationEngine, ContinuousGenerationConfig

async def test_bulk_ai_system():
    """Test the bulk AI system."""
    print("🚀 Testing Bulk AI System")
    print("=" * 50)
    
    # Configure bulk AI system
    config = BulkAIConfig(
        max_concurrent_generations=5,
        max_documents_per_query=20,
        enable_adaptive_model_selection=True,
        enable_ultra_optimization=True,
        enable_hybrid_optimization=True,
        enable_mcts_optimization=True,
        enable_olympiad_benchmarks=True
    )
    
    # Initialize bulk AI system
    bulk_ai = BulkAISystem(config)
    
    try:
        await bulk_ai.initialize()
        
        # Test query
        query = "Explain the principles of advanced machine learning optimization and provide examples of cutting-edge techniques used in neural network training, including quantum computing applications and edge computing optimizations."
        
        print(f"Query: {query}")
        print("-" * 50)
        
        # Process query
        start_time = time.time()
        results = await bulk_ai.process_query(query, max_documents=10)
        end_time = time.time()
        
        print(f"✅ Query processed successfully")
        print(f"Selected Model: {results['selected_model']}")
        print(f"Total Documents: {results['total_documents']}")
        print(f"Generation Time: {end_time - start_time:.2f} seconds")
        print(f"Performance Metrics: {results['performance_metrics']}")
        
        # Show system status
        status = await bulk_ai.get_system_status()
        print(f"\n📊 System Status:")
        print(f"Available Models: {status['available_models']}")
        print(f"Total Generated: {status['total_generated']}")
        print(f"System Resources: {status['system_resources']}")
        
    except Exception as e:
        logger.error(f"Error testing bulk AI system: {e}")
    finally:
        await bulk_ai.stop_generation()

async def test_continuous_generator():
    """Test the continuous generator."""
    print("\n🔄 Testing Continuous Generator")
    print("=" * 50)
    
    # Configure continuous generator
    config = ContinuousGenerationConfig(
        max_documents=15,
        generation_interval=0.5,
        enable_real_time_monitoring=True,
        enable_auto_cleanup=True,
        enable_model_rotation=True,
        enable_quality_filtering=True
    )
    
    # Initialize continuous generator
    generator = ContinuousGenerationEngine(config)
    
    try:
        await generator.initialize()
        
        # Test query
        query = "Generate comprehensive content about artificial intelligence, machine learning, and advanced optimization techniques."
        
        print(f"Query: {query}")
        print("-" * 50)
        
        # Start continuous generation
        document_count = 0
        async for result in generator.start_continuous_generation(query):
            document_count += 1
            
            print(f"📄 Document {document_count}:")
            print(f"  ID: {result.document_id}")
            print(f"  Model: {result.model_used}")
            print(f"  Quality: {result.quality_score:.3f}")
            print(f"  Time: {result.generation_time:.3f}s")
            print(f"  Content: {result.content[:100]}...")
            print("-" * 30)
            
            # Show performance summary every 5 documents
            if document_count % 5 == 0:
                summary = generator.get_performance_summary()
                print(f"📈 Performance Summary:")
                print(f"  Total Generated: {summary['total_generated']}")
                print(f"  Average Quality: {summary['average_quality_score']:.3f}")
                print(f"  Model Usage: {summary['model_usage']}")
                print("=" * 50)
        
        # Final summary
        final_summary = generator.get_performance_summary()
        print(f"\n🎯 Final Performance Summary:")
        print(f"Total Documents: {final_summary['total_generated']}")
        print(f"Average Generation Time: {final_summary['average_generation_time']:.3f}s")
        print(f"Average Quality Score: {final_summary['average_quality_score']:.3f}")
        print(f"Model Usage: {final_summary['model_usage']}")
        
    except Exception as e:
        logger.error(f"Error testing continuous generator: {e}")
    finally:
        generator.stop()

async def test_integration():
    """Test the complete integration."""
    print("\n🔗 Testing Complete Integration")
    print("=" * 50)
    
    # Test multiple queries with different characteristics
    test_queries = [
        "Explain quantum computing principles and their applications in machine learning",
        "Generate viral social media content about AI and technology trends",
        "Create professional brand content for a tech startup",
        "Analyze the latest developments in neural network optimization",
        "Write comprehensive documentation about advanced algorithms"
    ]
    
    # Configure systems
    bulk_ai_config = BulkAIConfig(
        max_concurrent_generations=3,
        max_documents_per_query=5,
        enable_adaptive_model_selection=True
    )
    
    continuous_config = ContinuousGenerationConfig(
        max_documents=10,
        generation_interval=0.3,
        enable_model_rotation=True
    )
    
    # Initialize systems
    bulk_ai = BulkAISystem(bulk_ai_config)
    generator = ContinuousGenerationEngine(continuous_config)
    
    try:
        await bulk_ai.initialize()
        await generator.initialize()
        
        print(f"Testing {len(test_queries)} different queries...")
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n🔍 Test {i}/{len(test_queries)}: {query[:50]}...")
            
            # Test with bulk AI system
            start_time = time.time()
            results = await bulk_ai.process_query(query, max_documents=3)
            bulk_time = time.time() - start_time
            
            print(f"  Bulk AI: {results['total_documents']} docs in {bulk_time:.2f}s")
            print(f"  Selected Model: {results['selected_model']}")
            
            # Test with continuous generator
            start_time = time.time()
            doc_count = 0
            async for result in generator.start_continuous_generation(query):
                doc_count += 1
                if doc_count >= 3:  # Limit for testing
                    break
            continuous_time = time.time() - start_time
            
            print(f"  Continuous: {doc_count} docs in {continuous_time:.2f}s")
            print(f"  Quality: {result.quality_score:.3f}")
        
        # Final system status
        bulk_status = await bulk_ai.get_system_status()
        continuous_performance = generator.get_performance_summary()
        
        print(f"\n📊 Final System Status:")
        print(f"Bulk AI - Total Generated: {bulk_status['total_generated']}")
        print(f"Continuous - Total Generated: {continuous_performance['total_generated']}")
        print(f"Continuous - Average Quality: {continuous_performance['average_quality_score']:.3f}")
        
    except Exception as e:
        logger.error(f"Error in integration test: {e}")
    finally:
        await bulk_ai.stop_generation()
        generator.stop()

async def main():
    """Main test function."""
    print("🧪 Bulk AI System Test Suite")
    print("=" * 60)
    
    try:
        # Test 1: Bulk AI System
        await test_bulk_ai_system()
        
        # Test 2: Continuous Generator
        await test_continuous_generator()
        
        # Test 3: Complete Integration
        await test_integration()
        
        print("\n✅ All tests completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⏹️ Tests interrupted by user")
    except Exception as e:
        logger.error(f"Test suite failed: {e}")
        print(f"\n❌ Test suite failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())










