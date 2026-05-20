from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import asyncio
import sys
import os
from pathlib import Path
from profiler_optimizer import LinkedInPostsProfiler
        import uvloop
from typing import Any, List, Dict, Optional
import logging
#!/usr/bin/env python3
"""
LinkedIn Posts Profiling Runner
===============================

Simple script to run the comprehensive profiling system and generate optimization reports.
"""


# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))


async def main():
    """Run the profiling system"""
    print("🚀 LinkedIn Posts System Profiler")
    print("=" * 50)
    
    # Initialize profiler
    profiler = LinkedInPostsProfiler()
    
    try:
        print("📊 Running comprehensive profiling...")
        print("   - System performance analysis")
        print("   - Data loading bottlenecks")
        print("   - Model inference profiling")
        print("   - Cache performance analysis")
        print("   - Optimization suggestions")
        print()
        
        # Run profiling
        results = await profiler.run_comprehensive_profiling()
        
        # Export report
        profiler.export_profiling_report(results, "linkedin_posts_profiling_report.json")
        
        print("✅ Profiling completed successfully!")
        print("📄 Report saved to: linkedin_posts_profiling_report.json")
        
        # Show quick summary
        print("\n📈 Quick Summary:")
        print(f"   • System CPU Usage: {results['system_stats']['cpu_percent']:.1f}%")
        print(f"   • System Memory Usage: {results['system_stats']['memory_percent']:.1f}%")
        print(f"   • Cache Hit Ratio: {results['cache_performance']['hit_ratio']:.1%}")
        print(f"   • Optimization Suggestions: {len(results['optimization_suggestions'])}")
        
        # Show top bottlenecks
        bottlenecks = []
        for suggestion in results['optimization_suggestions']:
            if suggestion['impact'] == 'high':
                bottlenecks.append(suggestion['description'])
        
        if bottlenecks:
            print(f"\n⚠️  Top Bottlenecks:")
            for i, bottleneck in enumerate(bottlenecks[:3], 1):
                print(f"   {i}. {bottleneck}")
        
        print("\n🎯 Next Steps:")
        print("   1. Review the detailed report in 'linkedin_posts_profiling_report.json'")
        print("   2. Implement the suggested optimizations")
        print("   3. Re-run profiling to measure improvements")
        
    except Exception as e:
        print(f"❌ Profiling failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Configure uvloop for better performance
    try:
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    except ImportError:
        print("⚠️  uvloop not available, using standard event loop")
    
    # Run profiling
    asyncio.run(main()) 