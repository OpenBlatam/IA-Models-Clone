#!/usr/bin/env python3
"""
Bulk TruthGPT AI System - Main Runner
====================================

Complete integration and demonstration of the bulk AI system.
This script shows how to use the bulk AI system for continuous generation.
"""

import asyncio
import logging
import time
import json
from datetime import datetime
from typing import Dict, List, Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import bulk AI components
try:
    from bulk_ai_system import BulkAISystem, BulkAIConfig
    from continuous_generator import ContinuousGenerationEngine, ContinuousGenerationConfig
except ImportError as e:
    logger.error(f"Failed to import bulk AI components: {e}")
    logger.info("Please ensure all dependencies are installed and paths are correct")
    exit(1)

class BulkAIDemo:
    """Demonstration class for the Bulk AI System."""
    
    def __init__(self):
        self.bulk_ai = None
        self.continuous_generator = None
        self.demo_queries = [
            "Explain the principles of advanced machine learning optimization and provide examples of cutting-edge techniques used in neural network training.",
            "Generate viral social media content about artificial intelligence and technology trends that would engage a broad audience.",
            "Create comprehensive brand content for a tech startup focused on AI and machine learning solutions.",
            "Analyze the latest developments in quantum computing and their applications in artificial intelligence.",
            "Write detailed documentation about advanced algorithms used in modern AI systems.",
            "Generate educational content about deep learning architectures and their practical applications.",
            "Create marketing content for an AI-powered product that highlights its unique features and benefits.",
            "Explain the mathematical foundations of neural networks and optimization algorithms.",
            "Generate creative content about the future of artificial intelligence and its impact on society.",
            "Write technical documentation for implementing advanced AI optimization techniques."
        ]
    
    async def initialize_systems(self):
        """Initialize both bulk AI and continuous generation systems."""
        logger.info("🚀 Initializing Bulk AI Systems...")
        
        try:
            # Configure bulk AI system
            bulk_ai_config = BulkAIConfig(
                max_concurrent_generations=8,
                max_documents_per_query=50,
                generation_interval=0.2,
                enable_adaptive_model_selection=True,
                enable_ultra_optimization=True,
                enable_hybrid_optimization=True,
                enable_mcts_optimization=True,
                enable_olympiad_benchmarks=True,
                enable_quantum_optimization=True,
                enable_edge_computing=True,
                target_memory_usage=0.8,
                target_cpu_usage=0.7,
                enable_auto_scaling=True
            )
            
            # Initialize bulk AI system
            self.bulk_ai = BulkAISystem(bulk_ai_config)
            await self.bulk_ai.initialize()
            logger.info("✅ Bulk AI System initialized")
            
            # Configure continuous generator
            continuous_config = ContinuousGenerationConfig(
                max_documents=100,
                generation_interval=0.3,
                batch_size=1,
                max_concurrent_tasks=5,
                enable_model_rotation=True,
                model_rotation_interval=20,
                enable_adaptive_scheduling=True,
                memory_threshold=0.9,
                cpu_threshold=0.8,
                enable_auto_cleanup=True,
                cleanup_interval=25,
                enable_quality_filtering=True,
                min_content_length=100,
                max_content_length=1500,
                enable_content_diversity=True,
                enable_real_time_monitoring=True,
                metrics_collection_interval=2.0,
                enable_performance_profiling=True
            )
            
            # Initialize continuous generator
            self.continuous_generator = ContinuousGenerationEngine(continuous_config)
            await self.continuous_generator.initialize()
            logger.info("✅ Continuous Generator initialized")
            
            logger.info("🎉 All systems initialized successfully!")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize systems: {e}")
            raise
    
    async def demo_bulk_ai_processing(self):
        """Demonstrate bulk AI processing capabilities."""
        logger.info("\n🔍 Demonstrating Bulk AI Processing")
        logger.info("=" * 60)
        
        try:
            # Test with different types of queries
            for i, query in enumerate(self.demo_queries[:3], 1):
                logger.info(f"\n📝 Test {i}/3: {query[:80]}...")
                
                start_time = time.time()
                results = await self.bulk_ai.process_query(query, max_documents=5)
                processing_time = time.time() - start_time
                
                logger.info(f"✅ Processing completed in {processing_time:.2f}s")
                logger.info(f"   Selected Model: {results['selected_model']}")
                logger.info(f"   Documents Generated: {results['total_documents']}")
                logger.info(f"   Generation Time: {results['generation_time']:.2f}s")
                
                # Show performance metrics
                metrics = results['performance_metrics']
                logger.info(f"   Performance: {metrics}")
                
                # Show system status
                status = await self.bulk_ai.get_system_status()
                logger.info(f"   System Status: {status['total_generated']} total, "
                          f"{status['system_resources']['cpu_usage']:.1f}% CPU, "
                          f"{status['system_resources']['memory_usage']:.1f}% Memory")
                
                # Small delay between tests
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"❌ Error in bulk AI processing demo: {e}")
    
    async def demo_continuous_generation(self):
        """Demonstrate continuous generation capabilities."""
        logger.info("\n🔄 Demonstrating Continuous Generation")
        logger.info("=" * 60)
        
        try:
            # Select a comprehensive query for continuous generation
            query = "Generate comprehensive content about artificial intelligence, machine learning, and advanced optimization techniques with practical examples and real-world applications."
            
            logger.info(f"📝 Query: {query}")
            logger.info("🔄 Starting continuous generation...")
            
            document_count = 0
            quality_scores = []
            model_usage = {}
            
            async for result in self.continuous_generator.start_continuous_generation(query):
                document_count += 1
                quality_scores.append(result.quality_score)
                
                # Track model usage
                model_name = result.model_used
                if model_name not in model_usage:
                    model_usage[model_name] = 0
                model_usage[model_name] += 1
                
                logger.info(f"📄 Document {document_count}:")
                logger.info(f"   ID: {result.document_id}")
                logger.info(f"   Model: {result.model_used}")
                logger.info(f"   Quality: {result.quality_score:.3f}")
                logger.info(f"   Time: {result.generation_time:.3f}s")
                logger.info(f"   Content: {result.content[:100]}...")
                
                # Show performance summary every 10 documents
                if document_count % 10 == 0:
                    performance = self.continuous_generator.get_performance_summary()
                    avg_quality = sum(quality_scores) / len(quality_scores)
                    
                    logger.info(f"\n📊 Performance Summary (after {document_count} documents):")
                    logger.info(f"   Total Generated: {performance['total_generated']}")
                    logger.info(f"   Average Quality: {avg_quality:.3f}")
                    logger.info(f"   Average Generation Time: {performance['average_generation_time']:.3f}s")
                    logger.info(f"   Model Usage: {model_usage}")
                    logger.info(f"   Error Rate: {performance['error_rate']:.3f}")
                    logger.info("-" * 50)
                
                # Limit for demo purposes
                if document_count >= 30:
                    logger.info("🛑 Reached demo limit (30 documents)")
                    break
            
            # Final summary
            final_performance = self.continuous_generator.get_performance_summary()
            final_avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
            
            logger.info(f"\n🎯 Final Continuous Generation Summary:")
            logger.info(f"   Total Documents: {document_count}")
            logger.info(f"   Average Quality: {final_avg_quality:.3f}")
            logger.info(f"   Average Generation Time: {final_performance['average_generation_time']:.3f}s")
            logger.info(f"   Model Distribution: {model_usage}")
            logger.info(f"   Success Rate: {1 - final_performance['error_rate']:.3f}")
            
        except Exception as e:
            logger.error(f"❌ Error in continuous generation demo: {e}")
    
    async def demo_advanced_features(self):
        """Demonstrate advanced features and optimizations."""
        logger.info("\n⚡ Demonstrating Advanced Features")
        logger.info("=" * 60)
        
        try:
            # Test adaptive model selection
            logger.info("🧠 Testing Adaptive Model Selection...")
            
            test_queries = [
                "Simple technical question about algorithms",
                "Complex philosophical discussion about AI consciousness and ethics",
                "Marketing content for a new AI product launch",
                "Technical documentation for advanced neural network architectures"
            ]
            
            for query in test_queries:
                logger.info(f"   Query: {query[:50]}...")
                
                # Get system status before
                status_before = await self.bulk_ai.get_system_status()
                
                # Process query
                results = await self.bulk_ai.process_query(query, max_documents=2)
                
                # Get system status after
                status_after = await self.bulk_ai.get_system_status()
                
                logger.info(f"   Selected Model: {results['selected_model']}")
                logger.info(f"   Documents: {results['total_documents']}")
                logger.info(f"   Quality: {results['performance_metrics'].get('quality_score', 'N/A')}")
                
                # Show resource usage
                cpu_change = status_after['system_resources']['cpu_usage'] - status_before['system_resources']['cpu_usage']
                memory_change = status_after['system_resources']['memory_usage'] - status_before['system_resources']['memory_usage']
                
                logger.info(f"   Resource Impact: CPU {cpu_change:+.1f}%, Memory {memory_change:+.1f}%")
                logger.info("-" * 40)
            
            # Test performance optimization
            logger.info("\n⚡ Testing Performance Optimization...")
            
            # Get current performance
            performance = self.continuous_generator.get_performance_summary()
            logger.info(f"   Current Performance:")
            logger.info(f"   - Generation Rate: {performance.get('generation_rate', 0):.2f} docs/sec")
            logger.info(f"   - Average Quality: {performance.get('average_quality_score', 0):.3f}")
            logger.info(f"   - Model Usage: {performance.get('model_usage', {})}")
            
            # Test system resilience
            logger.info("\n🛡️ Testing System Resilience...")
            
            # Simulate high load
            logger.info("   Simulating high load scenario...")
            high_load_query = "Generate comprehensive content about advanced AI optimization techniques with detailed technical explanations and practical examples."
            
            start_time = time.time()
            doc_count = 0
            
            async for result in self.continuous_generator.start_continuous_generation(high_load_query):
                doc_count += 1
                if doc_count >= 10:  # Limit for demo
                    break
            
            load_time = time.time() - start_time
            logger.info(f"   High load test: {doc_count} documents in {load_time:.2f}s")
            logger.info(f"   Rate: {doc_count/load_time:.2f} docs/sec")
            
        except Exception as e:
            logger.error(f"❌ Error in advanced features demo: {e}")
    
    async def run_complete_demo(self):
        """Run the complete demonstration."""
        logger.info("🎬 Starting Bulk TruthGPT AI System Demo")
        logger.info("=" * 80)
        
        try:
            # Initialize systems
            await self.initialize_systems()
            
            # Demo 1: Bulk AI Processing
            await self.demo_bulk_ai_processing()
            
            # Demo 2: Continuous Generation
            await self.demo_continuous_generation()
            
            # Demo 3: Advanced Features
            await self.demo_advanced_features()
            
            # Final system status
            logger.info("\n📊 Final System Status")
            logger.info("=" * 60)
            
            bulk_status = await self.bulk_ai.get_system_status()
            continuous_performance = self.continuous_generator.get_performance_summary()
            
            logger.info(f"Bulk AI System:")
            logger.info(f"   Total Generated: {bulk_status['total_generated']}")
            logger.info(f"   Available Models: {bulk_status['available_models']}")
            logger.info(f"   System Resources: {bulk_status['system_resources']}")
            
            logger.info(f"\nContinuous Generator:")
            logger.info(f"   Total Generated: {continuous_performance['total_generated']}")
            logger.info(f"   Average Quality: {continuous_performance['average_quality_score']:.3f}")
            logger.info(f"   Generation Rate: {continuous_performance.get('generation_rate', 0):.2f} docs/sec")
            logger.info(f"   Model Usage: {continuous_performance['model_usage']}")
            
            logger.info("\n🎉 Demo completed successfully!")
            logger.info("The Bulk TruthGPT AI System is ready for production use!")
            
        except Exception as e:
            logger.error(f"❌ Demo failed: {e}")
        finally:
            # Cleanup
            if self.bulk_ai:
                await self.bulk_ai.stop_generation()
            if self.continuous_generator:
                self.continuous_generator.stop()
            
            logger.info("🧹 Cleanup completed")

async def main():
    """Main function to run the demo."""
    demo = BulkAIDemo()
    await demo.run_complete_demo()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n⏹️ Demo interrupted by user")
    except Exception as e:
        logger.error(f"❌ Demo failed with error: {e}")
        logger.info("Please check the logs and ensure all dependencies are installed")










