"""
BUL Performance Analyzer
========================

Analyzes system performance and provides optimization recommendations.
"""

import asyncio
import time
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import statistics

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from modules.document_processor import DocumentProcessor
from modules.query_analyzer import QueryAnalyzer
from modules.business_agents import BusinessAgentManager
from config_optimized import BULConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerformanceAnalyzer:
    """Analyzes system performance and provides recommendations."""
    
    def __init__(self, config: Optional[BULConfig] = None):
        self.config = config or BULConfig()
        self.performance_data = {
            'query_analysis_times': [],
            'document_generation_times': [],
            'agent_processing_times': [],
            'total_request_times': []
        }
        self.benchmarks = {
            'query_analysis': {'good': 0.1, 'acceptable': 0.5, 'poor': 1.0},
            'document_generation': {'good': 2.0, 'acceptable': 5.0, 'poor': 10.0},
            'agent_processing': {'good': 0.5, 'acceptable': 2.0, 'poor': 5.0},
            'total_request': {'good': 3.0, 'acceptable': 8.0, 'poor': 15.0}
        }
    
    async def benchmark_query_analysis(self, num_tests: int = 10) -> Dict[str, Any]:
        """Benchmark query analysis performance."""
        print(f"🔍 Benchmarking query analysis ({num_tests} tests)...")
        
        analyzer = QueryAnalyzer()
        test_queries = [
            "Create a marketing strategy for a new product",
            "Develop a sales process for enterprise clients",
            "Write an operational manual for customer service",
            "Create HR policies for remote work",
            "Generate a financial plan for a startup",
            "Design a content strategy for social media",
            "Build a customer service training program",
            "Create a business development plan",
            "Develop a quality assurance process",
            "Write a technical documentation guide"
        ]
        
        times = []
        for i in range(num_tests):
            query = test_queries[i % len(test_queries)]
            start_time = time.time()
            
            analysis = analyzer.analyze(query)
            
            end_time = time.time()
            duration = end_time - start_time
            times.append(duration)
            
            print(f"   Test {i+1}: {duration:.3f}s - {analysis.primary_area}")
        
        return self._analyze_performance('query_analysis', times)
    
    async def benchmark_document_generation(self, num_tests: int = 5) -> Dict[str, Any]:
        """Benchmark document generation performance."""
        print(f"📄 Benchmarking document generation ({num_tests} tests)...")
        
        processor = DocumentProcessor(self.config.to_dict())
        test_cases = [
            {"query": "Marketing strategy", "area": "marketing", "type": "strategy"},
            {"query": "Sales proposal", "area": "sales", "type": "proposal"},
            {"query": "Operations manual", "area": "operations", "type": "manual"},
            {"query": "HR policy", "area": "hr", "type": "policy"},
            {"query": "Financial plan", "area": "finance", "type": "budget"}
        ]
        
        times = []
        for i in range(num_tests):
            case = test_cases[i % len(test_cases)]
            start_time = time.time()
            
            document = await processor.generate_document(
                query=case["query"],
                business_area=case["area"],
                document_type=case["type"]
            )
            
            end_time = time.time()
            duration = end_time - start_time
            times.append(duration)
            
            print(f"   Test {i+1}: {duration:.3f}s - {case['area']} {case['type']}")
        
        return self._analyze_performance('document_generation', times)
    
    async def benchmark_agent_processing(self, num_tests: int = 5) -> Dict[str, Any]:
        """Benchmark agent processing performance."""
        print(f"🤖 Benchmarking agent processing ({num_tests} tests)...")
        
        agent_manager = BusinessAgentManager(self.config.to_dict())
        test_cases = [
            {"area": "marketing", "query": "Create a campaign", "type": "campaign"},
            {"area": "sales", "query": "Develop a proposal", "type": "proposal"},
            {"area": "operations", "query": "Write a procedure", "type": "procedure"},
            {"area": "hr", "query": "Create a policy", "type": "policy"},
            {"area": "finance", "query": "Generate a budget", "type": "budget"}
        ]
        
        times = []
        for i in range(num_tests):
            case = test_cases[i % len(test_cases)]
            start_time = time.time()
            
            result = await agent_manager.process_with_agent(
                area=case["area"],
                query=case["query"],
                document_type=case["type"]
            )
            
            end_time = time.time()
            duration = end_time - start_time
            times.append(duration)
            
            print(f"   Test {i+1}: {duration:.3f}s - {case['area']} agent")
        
        return self._analyze_performance('agent_processing', times)
    
    async def benchmark_end_to_end(self, num_tests: int = 3) -> Dict[str, Any]:
        """Benchmark end-to-end request processing."""
        print(f"🔄 Benchmarking end-to-end processing ({num_tests} tests)...")
        
        analyzer = QueryAnalyzer()
        processor = DocumentProcessor(self.config.to_dict())
        agent_manager = BusinessAgentManager(self.config.to_dict())
        
        test_queries = [
            "Create a comprehensive marketing strategy for a new restaurant",
            "Develop a sales process for B2B software sales team",
            "Write an operational manual for customer service procedures"
        ]
        
        times = []
        for i in range(num_tests):
            query = test_queries[i % len(test_queries)]
            start_time = time.time()
            
            # Full workflow
            analysis = analyzer.analyze(query)
            agent_result = await agent_manager.process_with_agent(
                analysis.primary_area, query, analysis.document_types[0]
            )
            document = await processor.generate_document(
                query=query,
                business_area=analysis.primary_area,
                document_type=analysis.document_types[0]
            )
            
            end_time = time.time()
            duration = end_time - start_time
            times.append(duration)
            
            print(f"   Test {i+1}: {duration:.3f}s - {analysis.primary_area} workflow")
        
        return self._analyze_performance('total_request', times)
    
    def _analyze_performance(self, component: str, times: List[float]) -> Dict[str, Any]:
        """Analyze performance data for a component."""
        if not times:
            return {'error': 'No performance data'}
        
        avg_time = statistics.mean(times)
        min_time = min(times)
        max_time = max(times)
        median_time = statistics.median(times)
        
        # Determine performance level
        benchmark = self.benchmarks[component]
        if avg_time <= benchmark['good']:
            performance_level = 'excellent'
        elif avg_time <= benchmark['acceptable']:
            performance_level = 'good'
        elif avg_time <= benchmark['poor']:
            performance_level = 'acceptable'
        else:
            performance_level = 'poor'
        
        return {
            'component': component,
            'samples': len(times),
            'average_time': avg_time,
            'min_time': min_time,
            'max_time': max_time,
            'median_time': median_time,
            'performance_level': performance_level,
            'benchmark': benchmark,
            'recommendations': self._get_recommendations(component, avg_time, benchmark)
        }
    
    def _get_recommendations(self, component: str, avg_time: float, benchmark: Dict[str, float]) -> List[str]:
        """Get performance optimization recommendations."""
        recommendations = []
        
        if avg_time > benchmark['poor']:
            if component == 'query_analysis':
                recommendations.extend([
                    "Consider caching common query patterns",
                    "Optimize keyword matching algorithms",
                    "Use more efficient data structures for business area detection"
                ])
            elif component == 'document_generation':
                recommendations.extend([
                    "Implement document template caching",
                    "Use async file I/O operations",
                    "Consider parallel document generation for multiple requests"
                ])
            elif component == 'agent_processing':
                recommendations.extend([
                    "Cache agent responses for similar queries",
                    "Implement agent response streaming",
                    "Consider agent specialization optimization"
                ])
            elif component == 'total_request':
                recommendations.extend([
                    "Implement request queuing and batching",
                    "Use connection pooling for external services",
                    "Consider horizontal scaling for high load"
                ])
        
        return recommendations
    
    async def run_full_benchmark(self) -> Dict[str, Any]:
        """Run complete performance benchmark."""
        print("🚀 Starting BUL Performance Analysis")
        print("=" * 50)
        
        results = {}
        
        # Run all benchmarks
        results['query_analysis'] = await self.benchmark_query_analysis()
        results['document_generation'] = await self.benchmark_document_generation()
        results['agent_processing'] = await self.benchmark_agent_processing()
        results['end_to_end'] = await self.benchmark_end_to_end()
        
        # Generate summary
        results['summary'] = self._generate_summary(results)
        
        return results
    
    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate performance summary."""
        summary = {
            'overall_performance': 'excellent',
            'bottlenecks': [],
            'recommendations': [],
            'performance_score': 100
        }
        
        # Analyze each component
        for component, data in results.items():
            if component == 'summary':
                continue
            
            if data.get('performance_level') == 'poor':
                summary['bottlenecks'].append(component)
                summary['performance_score'] -= 20
            elif data.get('performance_level') == 'acceptable':
                summary['performance_score'] -= 10
            
            # Collect recommendations
            if 'recommendations' in data:
                summary['recommendations'].extend(data['recommendations'])
        
        # Determine overall performance
        if summary['performance_score'] >= 90:
            summary['overall_performance'] = 'excellent'
        elif summary['performance_score'] >= 70:
            summary['overall_performance'] = 'good'
        elif summary['performance_score'] >= 50:
            summary['overall_performance'] = 'acceptable'
        else:
            summary['overall_performance'] = 'poor'
        
        return summary
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate performance report."""
        report = f"""
BUL Performance Analysis Report
==============================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

OVERALL PERFORMANCE
------------------
Performance Level: {results['summary']['overall_performance'].upper()}
Performance Score: {results['summary']['performance_score']}/100

COMPONENT ANALYSIS
-----------------
"""
        
        for component, data in results.items():
            if component == 'summary':
                continue
            
            report += f"""
{component.replace('_', ' ').title()}:
  Average Time: {data['average_time']:.3f}s
  Min Time: {data['min_time']:.3f}s
  Max Time: {data['max_time']:.3f}s
  Performance Level: {data['performance_level']}
  Samples: {data['samples']}
"""
        
        # Bottlenecks
        if results['summary']['bottlenecks']:
            report += f"""
BOTTLENECKS IDENTIFIED
---------------------
{chr(10).join(f"- {bottleneck}" for bottleneck in results['summary']['bottlenecks'])}
"""
        
        # Recommendations
        if results['summary']['recommendations']:
            report += f"""
OPTIMIZATION RECOMMENDATIONS
---------------------------
{chr(10).join(f"- {rec}" for rec in results['summary']['recommendations'])}
"""
        
        return report

async def main():
    """Main performance analysis function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="BUL Performance Analyzer")
    parser.add_argument("--component", choices=['query', 'document', 'agent', 'end-to-end', 'all'], 
                       default='all', help="Component to benchmark")
    parser.add_argument("--tests", type=int, default=5, help="Number of tests to run")
    parser.add_argument("--report", action="store_true", help="Generate detailed report")
    
    args = parser.parse_args()
    
    analyzer = PerformanceAnalyzer()
    
    if args.component == 'all':
        results = await analyzer.run_full_benchmark()
    else:
        results = {}
        if args.component == 'query':
            results['query_analysis'] = await analyzer.benchmark_query_analysis(args.tests)
        elif args.component == 'document':
            results['document_generation'] = await analyzer.benchmark_document_generation(args.tests)
        elif args.component == 'agent':
            results['agent_processing'] = await analyzer.benchmark_agent_processing(args.tests)
        elif args.component == 'end-to-end':
            results['end_to_end'] = await analyzer.benchmark_end_to_end(args.tests)
    
    # Display results
    print("\n" + "=" * 50)
    print("📊 PERFORMANCE ANALYSIS RESULTS")
    print("=" * 50)
    
    for component, data in results.items():
        if component == 'summary':
            continue
        
        print(f"\n{component.replace('_', ' ').title()}:")
        print(f"  Average Time: {data['average_time']:.3f}s")
        print(f"  Performance Level: {data['performance_level']}")
        if data.get('recommendations'):
            print(f"  Recommendations: {len(data['recommendations'])} items")
    
    if 'summary' in results:
        print(f"\nOverall Performance: {results['summary']['overall_performance'].upper()}")
        print(f"Performance Score: {results['summary']['performance_score']}/100")
    
    # Generate report if requested
    if args.report:
        report = analyzer.generate_report(results)
        report_file = f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        print(f"\n📄 Detailed report saved to: {report_file}")

if __name__ == "__main__":
    asyncio.run(main())
