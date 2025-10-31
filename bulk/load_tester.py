"""
BUL Load Tester
===============

Load testing tool for the BUL system to test performance under various loads.
"""

import asyncio
import aiohttp
import time
import logging
import sys
import statistics
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LoadTester:
    """Load testing tool for BUL system."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'response_times': [],
            'errors': []
        }
    
    async def single_request(self, session: aiohttp.ClientSession, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make a single request and measure performance."""
        start_time = time.time()
        
        try:
            async with session.post(
                f"{self.base_url}/documents/generate",
                json=request_data,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                end_time = time.time()
                response_time = end_time - start_time
                
                if response.status == 200:
                    result = await response.json()
                    return {
                        'success': True,
                        'response_time': response_time,
                        'status_code': response.status,
                        'task_id': result.get('task_id'),
                        'error': None
                    }
                else:
                    return {
                        'success': False,
                        'response_time': response_time,
                        'status_code': response.status,
                        'task_id': None,
                        'error': f"HTTP {response.status}"
                    }
        
        except asyncio.TimeoutError:
            end_time = time.time()
            return {
                'success': False,
                'response_time': end_time - start_time,
                'status_code': None,
                'task_id': None,
                'error': 'Timeout'
            }
        except Exception as e:
            end_time = time.time()
            return {
                'success': False,
                'response_time': end_time - start_time,
                'status_code': None,
                'task_id': None,
                'error': str(e)
            }
    
    async def concurrent_requests(self, num_requests: int, concurrency: int, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run concurrent requests with specified concurrency level."""
        print(f"🚀 Running {num_requests} requests with concurrency {concurrency}")
        
        semaphore = asyncio.Semaphore(concurrency)
        results = []
        
        async def bounded_request(session, request_id):
            async with semaphore:
                # Add unique identifier to request
                unique_request = request_data.copy()
                unique_request['metadata'] = {
                    'request_id': request_id,
                    'timestamp': datetime.now().isoformat()
                }
                
                result = await self.single_request(session, unique_request)
                result['request_id'] = request_id
                return result
        
        async with aiohttp.ClientSession() as session:
            tasks = [
                bounded_request(session, i) 
                for i in range(num_requests)
            ]
            
            # Execute all requests
            start_time = time.time()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            end_time = time.time()
        
        # Process results
        successful = [r for r in results if isinstance(r, dict) and r.get('success')]
        failed = [r for r in results if isinstance(r, dict) and not r.get('success')]
        exceptions = [r for r in results if isinstance(r, Exception)]
        
        response_times = [r['response_time'] for r in successful]
        
        return {
            'total_requests': num_requests,
            'successful_requests': len(successful),
            'failed_requests': len(failed) + len(exceptions),
            'exceptions': len(exceptions),
            'total_time': end_time - start_time,
            'response_times': response_times,
            'average_response_time': statistics.mean(response_times) if response_times else 0,
            'min_response_time': min(response_times) if response_times else 0,
            'max_response_time': max(response_times) if response_times else 0,
            'median_response_time': statistics.median(response_times) if response_times else 0,
            'requests_per_second': num_requests / (end_time - start_time),
            'success_rate': (len(successful) / num_requests) * 100,
            'errors': [r.get('error') for r in failed if r.get('error')]
        }
    
    async def stress_test(self, max_concurrency: int = 50, step: int = 5) -> Dict[str, Any]:
        """Run stress test with increasing concurrency levels."""
        print(f"💪 Running stress test (max concurrency: {max_concurrency})")
        
        request_data = {
            "query": "Create a marketing strategy for a new product",
            "business_area": "marketing",
            "document_type": "strategy",
            "priority": 1
        }
        
        results = {}
        
        for concurrency in range(step, max_concurrency + 1, step):
            print(f"   Testing concurrency level: {concurrency}")
            
            result = await self.concurrent_requests(
                num_requests=concurrency * 2,  # 2 requests per concurrent user
                concurrency=concurrency,
                request_data=request_data
            )
            
            results[f'concurrency_{concurrency}'] = result
            
            # Check if system is still responsive
            if result['success_rate'] < 80:
                print(f"   ⚠️  Success rate dropped to {result['success_rate']:.1f}%")
                break
            
            # Small delay between tests
            await asyncio.sleep(1)
        
        return results
    
    async def endurance_test(self, duration_minutes: int = 5, concurrency: int = 10) -> Dict[str, Any]:
        """Run endurance test for specified duration."""
        print(f"⏱️  Running endurance test ({duration_minutes} minutes, concurrency: {concurrency})")
        
        request_data = {
            "query": "Generate a business document",
            "business_area": "marketing",
            "document_type": "strategy",
            "priority": 1
        }
        
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        
        all_results = []
        
        while time.time() < end_time:
            result = await self.concurrent_requests(
                num_requests=concurrency,
                concurrency=concurrency,
                request_data=request_data
            )
            
            all_results.append(result)
            
            # Small delay between batches
            await asyncio.sleep(2)
        
        # Aggregate results
        total_requests = sum(r['total_requests'] for r in all_results)
        total_successful = sum(r['successful_requests'] for r in all_results)
        total_failed = sum(r['failed_requests'] for r in all_results)
        
        all_response_times = []
        for r in all_results:
            all_response_times.extend(r['response_times'])
        
        return {
            'test_duration_minutes': duration_minutes,
            'total_requests': total_requests,
            'successful_requests': total_successful,
            'failed_requests': total_failed,
            'success_rate': (total_successful / total_requests) * 100 if total_requests > 0 else 0,
            'average_response_time': statistics.mean(all_response_times) if all_response_times else 0,
            'min_response_time': min(all_response_times) if all_response_times else 0,
            'max_response_time': max(all_response_times) if all_response_times else 0,
            'median_response_time': statistics.median(all_response_times) if all_response_times else 0,
            'requests_per_minute': total_requests / duration_minutes,
            'batches_completed': len(all_results)
        }
    
    def generate_report(self, results: Dict[str, Any], test_type: str) -> str:
        """Generate load test report."""
        report = f"""
BUL Load Test Report
===================
Test Type: {test_type}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

"""
        
        if test_type == 'concurrent':
            report += f"""
CONCURRENT LOAD TEST RESULTS
----------------------------
Total Requests: {results['total_requests']}
Successful Requests: {results['successful_requests']}
Failed Requests: {results['failed_requests']}
Success Rate: {results['success_rate']:.1f}%

RESPONSE TIME STATISTICS
------------------------
Average Response Time: {results['average_response_time']:.3f}s
Min Response Time: {results['min_response_time']:.3f}s
Max Response Time: {results['max_response_time']:.3f}s
Median Response Time: {results['median_response_time']:.3f}s

PERFORMANCE METRICS
------------------
Requests per Second: {results['requests_per_second']:.1f}
Total Test Duration: {results['total_time']:.1f}s
"""
        
        elif test_type == 'stress':
            report += "STRESS TEST RESULTS\n"
            report += "------------------\n"
            
            for concurrency, data in results.items():
                if concurrency.startswith('concurrency_'):
                    level = concurrency.replace('concurrency_', '')
                    report += f"""
Concurrency Level {level}:
  Total Requests: {data['total_requests']}
  Success Rate: {data['success_rate']:.1f}%
  Average Response Time: {data['average_response_time']:.3f}s
  Requests per Second: {data['requests_per_second']:.1f}
"""
        
        elif test_type == 'endurance':
            report += f"""
ENDURANCE TEST RESULTS
---------------------
Test Duration: {results['test_duration_minutes']} minutes
Total Requests: {results['total_requests']}
Successful Requests: {results['successful_requests']}
Failed Requests: {results['failed_requests']}
Success Rate: {results['success_rate']:.1f}%

RESPONSE TIME STATISTICS
------------------------
Average Response Time: {results['average_response_time']:.3f}s
Min Response Time: {results['min_response_time']:.3f}s
Max Response Time: {results['max_response_time']:.3f}s
Median Response Time: {results['median_response_time']:.3f}s

PERFORMANCE METRICS
------------------
Requests per Minute: {results['requests_per_minute']:.1f}
Batches Completed: {results['batches_completed']}
"""
        
        # Add error analysis if any
        if 'errors' in results and results['errors']:
            error_counts = {}
            for error in results['errors']:
                error_counts[error] = error_counts.get(error, 0) + 1
            
            report += "\nERROR ANALYSIS\n"
            report += "--------------\n"
            for error, count in error_counts.items():
                report += f"{error}: {count} occurrences\n"
        
        return report

async def main():
    """Main load testing function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="BUL Load Tester")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL of BUL system")
    parser.add_argument("--type", choices=['concurrent', 'stress', 'endurance'], 
                       default='concurrent', help="Type of load test")
    parser.add_argument("--requests", type=int, default=100, help="Number of requests")
    parser.add_argument("--concurrency", type=int, default=10, help="Concurrency level")
    parser.add_argument("--duration", type=int, default=5, help="Duration in minutes (for endurance test)")
    parser.add_argument("--report", action="store_true", help="Generate detailed report")
    
    args = parser.parse_args()
    
    tester = LoadTester(args.url)
    
    print("🚀 BUL Load Tester")
    print("=" * 50)
    print(f"Target URL: {args.url}")
    print(f"Test Type: {args.type}")
    
    # Check if system is running
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{args.url}/health", timeout=aiohttp.ClientTimeout(total=5)) as response:
                if response.status == 200:
                    print("✅ System is running and accessible")
                else:
                    print(f"⚠️  System responded with status {response.status}")
    except Exception as e:
        print(f"❌ Cannot connect to system: {e}")
        print("   Make sure the BUL system is running on the specified URL")
        return 1
    
    # Run appropriate test
    if args.type == 'concurrent':
        results = await tester.concurrent_requests(
            num_requests=args.requests,
            concurrency=args.concurrency,
            request_data={
                "query": "Create a marketing strategy for a new product",
                "business_area": "marketing",
                "document_type": "strategy",
                "priority": 1
            }
        )
    elif args.type == 'stress':
        results = await tester.stress_test(max_concurrency=args.concurrency)
    elif args.type == 'endurance':
        results = await tester.endurance_test(
            duration_minutes=args.duration,
            concurrency=args.concurrency
        )
    
    # Display results
    print("\n" + "=" * 50)
    print("📊 LOAD TEST RESULTS")
    print("=" * 50)
    
    if args.type == 'concurrent':
        print(f"Total Requests: {results['total_requests']}")
        print(f"Successful: {results['successful_requests']}")
        print(f"Failed: {results['failed_requests']}")
        print(f"Success Rate: {results['success_rate']:.1f}%")
        print(f"Average Response Time: {results['average_response_time']:.3f}s")
        print(f"Requests per Second: {results['requests_per_second']:.1f}")
    
    elif args.type == 'stress':
        print("Stress Test Results:")
        for concurrency, data in results.items():
            if concurrency.startswith('concurrency_'):
                level = concurrency.replace('concurrency_', '')
                print(f"  Concurrency {level}: {data['success_rate']:.1f}% success, {data['average_response_time']:.3f}s avg")
    
    elif args.type == 'endurance':
        print(f"Duration: {results['test_duration_minutes']} minutes")
        print(f"Total Requests: {results['total_requests']}")
        print(f"Success Rate: {results['success_rate']:.1f}%")
        print(f"Average Response Time: {results['average_response_time']:.3f}s")
        print(f"Requests per Minute: {results['requests_per_minute']:.1f}")
    
    # Generate report if requested
    if args.report:
        report = tester.generate_report(results, args.type)
        report_file = f"load_test_report_{args.type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        print(f"\n📄 Detailed report saved to: {report_file}")
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
