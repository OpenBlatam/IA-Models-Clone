"""
Perfect Integration Demo - TruthGPT with Ultra-Adaptive K/V Cache
Demonstrates seamless integration with existing TruthGPT architecture
"""

import asyncio
import logging
import time
import json
from typing import Dict, Any, List
from pathlib import Path
import sys

# Add the bulk API to the path
sys.path.append(str(Path(__file__).parent.parent))

from api.bul_api import (
    BULAPI,
    BULAPIConfig,
    create_bul_api,
    create_bul_api_config,
    create_truthgpt_bul_api,
    create_high_performance_bul_api,
    create_memory_efficient_bul_api
)

from api.truthgpt_bulk_api import (
    TruthGPTBulkAPI,
    TruthGPTBulkAPIConfig,
    create_truthgpt_bulk_api,
    create_truthgpt_bulk_api_config,
    create_truthgpt_api,
    create_high_performance_truthgpt_api,
    create_memory_efficient_truthgpt_api
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerfectIntegrationDemo:
    """
    Perfect Integration Demo for TruthGPT with Ultra-Adaptive K/V Cache.
    
    This demo shows:
    - Seamless integration with existing TruthGPT architecture
    - Ultra-adaptive K/V cache optimization
    - Bulk processing with automatic scaling
    - Real-time performance monitoring
    - Perfect adaptation to workload characteristics
    """
    
    def __init__(self):
        self.bul_api = None
        self.truthgpt_api = None
        self.demo_results = {}
        
        logger.info("Perfect Integration Demo initialized")
    
    def setup_apis(self):
        """Setup BUL and TruthGPT APIs."""
        logger.info("Setting up APIs...")
        
        # Create BUL API
        self.bul_api = create_truthgpt_bul_api(
            model_name="truthgpt-base",
            model_size="medium"
        )
        
        # Create TruthGPT API
        self.truthgpt_api = create_truthgpt_api(
            model_name="truthgpt-base",
            model_size="medium"
        )
        
        logger.info("APIs setup complete")
    
    async def demo_single_request_processing(self):
        """Demo single request processing."""
        logger.info("=== Demo: Single Request Processing ===")
        
        # Create test request
        request = {
            'text': 'What is the meaning of life?',
            'max_length': 100,
            'temperature': 0.7,
            'session_id': 'demo_session_1'
        }
        
        start_time = time.time()
        
        try:
            # Process through BUL API
            bul_response = await self.bul_api.process_single_request(request)
            
            # Process through TruthGPT API
            truthgpt_response = await self.truthgpt_api.process_single_request(request)
            
            processing_time = time.time() - start_time
            
            # Store results
            self.demo_results['single_request'] = {
                'bul_response': bul_response,
                'truthgpt_response': truthgpt_response,
                'processing_time': processing_time
            }
            
            logger.info(f"Single request processing completed in {processing_time:.2f}s")
            logger.info(f"BUL API success: {bul_response.get('success', False)}")
            logger.info(f"TruthGPT API success: {truthgpt_response.get('success', False)}")
            
        except Exception as e:
            logger.error(f"Error in single request processing: {e}")
    
    async def demo_bulk_request_processing(self):
        """Demo bulk request processing."""
        logger.info("=== Demo: Bulk Request Processing ===")
        
        # Create test requests
        requests = [
            {
                'text': 'What is artificial intelligence?',
                'max_length': 100,
                'temperature': 0.7,
                'session_id': 'demo_session_2'
            },
            {
                'text': 'Explain quantum computing',
                'max_length': 100,
                'temperature': 0.7,
                'session_id': 'demo_session_3'
            },
            {
                'text': 'What is machine learning?',
                'max_length': 100,
                'temperature': 0.7,
                'session_id': 'demo_session_4'
            },
            {
                'text': 'Describe neural networks',
                'max_length': 100,
                'temperature': 0.7,
                'session_id': 'demo_session_5'
            }
        ]
        
        start_time = time.time()
        
        try:
            # Process through BUL API
            bul_response = await self.bul_api.process_bulk_requests(requests)
            
            # Process through TruthGPT API
            truthgpt_response = await self.truthgpt_api.process_bulk_requests(requests)
            
            processing_time = time.time() - start_time
            
            # Store results
            self.demo_results['bulk_request'] = {
                'bul_response': bul_response,
                'truthgpt_response': truthgpt_response,
                'processing_time': processing_time
            }
            
            logger.info(f"Bulk request processing completed in {processing_time:.2f}s")
            logger.info(f"BUL API success: {bul_response.get('success', False)}")
            logger.info(f"TruthGPT API success: {truthgpt_response.get('success', False)}")
            
        except Exception as e:
            logger.error(f"Error in bulk request processing: {e}")
    
    async def demo_adaptive_scaling(self):
        """Demo adaptive scaling based on workload."""
        logger.info("=== Demo: Adaptive Scaling ===")
        
        # Create different workload scenarios
        workloads = [
            {
                'name': 'Light Workload',
                'requests': [
                    {'text': 'Hello world', 'max_length': 50, 'temperature': 0.7}
                ] * 2
            },
            {
                'name': 'Medium Workload',
                'requests': [
                    {'text': 'What is the meaning of life?', 'max_length': 100, 'temperature': 0.7}
                ] * 8
            },
            {
                'name': 'Heavy Workload',
                'requests': [
                    {'text': 'Explain the theory of relativity in detail', 'max_length': 200, 'temperature': 0.7}
                ] * 16
            }
        ]
        
        scaling_results = {}
        
        for workload in workloads:
            logger.info(f"Testing {workload['name']}...")
            
            start_time = time.time()
            
            try:
                # Process workload
                response = await self.bul_api.process_bulk_requests(workload['requests'])
                
                processing_time = time.time() - start_time
                
                # Store results
                scaling_results[workload['name']] = {
                    'num_requests': len(workload['requests']),
                    'processing_time': processing_time,
                    'success_rate': response.get('success_rate', 0.0),
                    'throughput': len(workload['requests']) / processing_time if processing_time > 0 else 0.0
                }
                
                logger.info(f"{workload['name']} completed in {processing_time:.2f}s")
                
            except Exception as e:
                logger.error(f"Error in {workload['name']}: {e}")
        
        self.demo_results['adaptive_scaling'] = scaling_results
    
    async def demo_performance_monitoring(self):
        """Demo performance monitoring."""
        logger.info("=== Demo: Performance Monitoring ===")
        
        # Get performance stats
        bul_stats = self.bul_api.get_bul_api_stats()
        truthgpt_stats = self.truthgpt_api.get_api_stats()
        
        # Store results
        self.demo_results['performance_monitoring'] = {
            'bul_api_stats': bul_stats,
            'truthgpt_api_stats': truthgpt_stats
        }
        
        logger.info("Performance monitoring data collected")
        logger.info(f"BUL API active sessions: {bul_stats.get('active_sessions', 0)}")
        logger.info(f"TruthGPT API active sessions: {truthgpt_stats.get('active_sessions', 0)}")
    
    async def demo_cache_optimization(self):
        """Demo cache optimization."""
        logger.info("=== Demo: Cache Optimization ===")
        
        # Test cache hit scenarios
        cache_test_requests = [
            {
                'text': 'What is the capital of France?',
                'max_length': 50,
                'temperature': 0.7,
                'session_id': 'cache_test_session'
            }
        ]
        
        # First request (cache miss)
        start_time = time.time()
        first_response = await self.bul_api.process_bulk_requests(cache_test_requests)
        first_processing_time = time.time() - start_time
        
        # Second request (cache hit)
        start_time = time.time()
        second_response = await self.bul_api.process_bulk_requests(cache_test_requests)
        second_processing_time = time.time() - start_time
        
        # Store results
        self.demo_results['cache_optimization'] = {
            'first_request_time': first_processing_time,
            'second_request_time': second_processing_time,
            'cache_improvement': (first_processing_time - second_processing_time) / first_processing_time if first_processing_time > 0 else 0.0
        }
        
        logger.info(f"First request time: {first_processing_time:.2f}s")
        logger.info(f"Second request time: {second_processing_time:.2f}s")
        logger.info(f"Cache improvement: {self.demo_results['cache_optimization']['cache_improvement']:.2%}")
    
    async def demo_memory_efficiency(self):
        """Demo memory efficiency."""
        logger.info("=== Demo: Memory Efficiency ===")
        
        # Create memory-efficient API
        memory_efficient_api = create_memory_efficient_bul_api()
        
        # Test memory usage
        memory_test_requests = [
            {
                'text': 'Test memory efficiency with long text that should trigger memory optimization',
                'max_length': 100,
                'temperature': 0.7,
                'session_id': f'memory_test_session_{i}'
            }
            for i in range(10)
        ]
        
        start_time = time.time()
        
        try:
            # Process memory test
            response = await memory_efficient_api.process_bulk_requests(memory_test_requests)
            
            processing_time = time.time() - start_time
            
            # Store results
            self.demo_results['memory_efficiency'] = {
                'num_requests': len(memory_test_requests),
                'processing_time': processing_time,
                'success_rate': response.get('success_rate', 0.0),
                'throughput': len(memory_test_requests) / processing_time if processing_time > 0 else 0.0
            }
            
            logger.info(f"Memory efficiency test completed in {processing_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Error in memory efficiency test: {e}")
        finally:
            # Cleanup
            memory_efficient_api.shutdown()
    
    async def demo_high_performance(self):
        """Demo high performance processing."""
        logger.info("=== Demo: High Performance Processing ===")
        
        # Create high-performance API
        high_performance_api = create_high_performance_bul_api()
        
        # Test high performance
        performance_test_requests = [
            {
                'text': f'High performance test request {i}',
                'max_length': 100,
                'temperature': 0.7,
                'session_id': f'performance_test_session_{i}'
            }
            for i in range(20)
        ]
        
        start_time = time.time()
        
        try:
            # Process performance test
            response = await high_performance_api.process_bulk_requests(performance_test_requests)
            
            processing_time = time.time() - start_time
            
            # Store results
            self.demo_results['high_performance'] = {
                'num_requests': len(performance_test_requests),
                'processing_time': processing_time,
                'success_rate': response.get('success_rate', 0.0),
                'throughput': len(performance_test_requests) / processing_time if processing_time > 0 else 0.0
            }
            
            logger.info(f"High performance test completed in {processing_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Error in high performance test: {e}")
        finally:
            # Cleanup
            high_performance_api.shutdown()
    
    def generate_demo_report(self):
        """Generate comprehensive demo report."""
        logger.info("=== Demo Report ===")
        
        report = {
            'demo_summary': {
                'total_demos': len(self.demo_results),
                'timestamp': time.time()
            },
            'results': self.demo_results
        }
        
        # Print summary
        logger.info("Demo Results Summary:")
        for demo_name, results in self.demo_results.items():
            logger.info(f"  {demo_name}: {results}")
        
        # Save report to file
        report_file = Path(__file__).parent / "perfect_integration_demo_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Demo report saved to {report_file}")
        
        return report
    
    async def run_complete_demo(self):
        """Run complete demo."""
        logger.info("Starting Perfect Integration Demo...")
        
        try:
            # Setup APIs
            self.setup_apis()
            
            # Run demos
            await self.demo_single_request_processing()
            await self.demo_bulk_request_processing()
            await self.demo_adaptive_scaling()
            await self.demo_performance_monitoring()
            await self.demo_cache_optimization()
            await self.demo_memory_efficiency()
            await self.demo_high_performance()
            
            # Generate report
            report = self.generate_demo_report()
            
            logger.info("Perfect Integration Demo completed successfully!")
            return report
            
        except Exception as e:
            logger.error(f"Error in demo: {e}")
            raise
        finally:
            # Cleanup
            if self.bul_api:
                self.bul_api.shutdown()
            if self.truthgpt_api:
                self.truthgpt_api.shutdown()

async def main():
    """Main demo function."""
    demo = PerfectIntegrationDemo()
    
    try:
        # Run complete demo
        report = await demo.run_complete_demo()
        
        logger.info("Demo completed successfully!")
        logger.info(f"Report: {report}")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise

if __name__ == "__main__":
    # Run the demo
    asyncio.run(main())




