#!/usr/bin/env python3
"""
CLI Tool for Ultra-Adaptive K/V Cache Engine Management
Provides command-line interface for monitoring, configuration, and operations
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import time

try:
    from ultra_adaptive_kv_cache_engine import (
        UltraAdaptiveKVCacheEngine,
        AdaptiveConfig,
        AdaptiveMode,
        TruthGPTIntegration
    )
    ENGINE_AVAILABLE = True
except ImportError:
    ENGINE_AVAILABLE = False
    print("Warning: Ultra-Adaptive K/V Cache Engine not available")

try:
    from ultra_adaptive_kv_cache_monitor import PerformanceMonitor
    MONITOR_AVAILABLE = True
except ImportError:
    MONITOR_AVAILABLE = False

try:
    from ultra_adaptive_kv_cache_health_checker import create_engine_health_checks
    HEALTH_AVAILABLE = True
except ImportError:
    HEALTH_AVAILABLE = False


class CLITool:
    """CLI tool for engine management."""
    
    def __init__(self):
        self.engine = None
    
    def create_engine(self, config_path: Optional[str] = None) -> UltraAdaptiveKVCacheEngine:
        """Create engine instance."""
        if not ENGINE_AVAILABLE:
            print("Error: Engine not available")
            sys.exit(1)
        
        if config_path and Path(config_path).exists():
            # Load config from file
            with open(config_path) as f:
                config_data = json.load(f)
            config = AdaptiveConfig(**config_data)
            return UltraAdaptiveKVCacheEngine(config)
        else:
            return TruthGPTIntegration.create_engine_for_truthgpt()
    
    async def cmd_stats(self, args):
        """Display engine statistics."""
        engine = self.create_engine()
        try:
            stats = engine.get_performance_stats()
            
            if args.json:
                print(json.dumps(stats, indent=2, default=str))
            else:
                self._print_stats(stats)
        finally:
            engine.shutdown()
    
    def _print_stats(self, stats: Dict[str, Any]):
        """Print stats in human-readable format."""
        print("=" * 80)
        print("ULTRA-ADAPTIVE K/V CACHE ENGINE - STATISTICS")
        print("=" * 80)
        
        engine_stats = stats.get('engine_stats', {})
        
        print("\n📊 PERFORMANCE METRICS:")
        print(f"  Total Requests: {engine_stats.get('total_requests', 0):,}")
        print(f"  Total Tokens: {engine_stats.get('total_tokens', 0):,}")
        print(f"  Average Response Time: {engine_stats.get('avg_response_time', 0)*1000:.2f} ms")
        print(f"  P50 Latency: {engine_stats.get('p50_response_time', 0)*1000:.2f} ms")
        print(f"  P95 Latency: {engine_stats.get('p95_response_time', 0)*1000:.2f} ms")
        print(f"  P99 Latency: {engine_stats.get('p99_response_time', 0)*1000:.2f} ms")
        print(f"  Throughput: {engine_stats.get('throughput', 0):.2f} req/s")
        print(f"  Cache Hit Rate: {engine_stats.get('cache_hit_rate', 0)*100:.2f}%")
        print(f"  Error Rate: {engine_stats.get('error_rate', 0)*100:.2f}%")
        
        print("\n💾 MEMORY:")
        print(f"  Memory Usage: {stats.get('memory_usage', 0)*100:.2f}%")
        mem_history = stats.get('memory_usage_history', {})
        if mem_history:
            print(f"  Mean: {mem_history.get('mean', 0)*100:.2f}%")
            print(f"  Min: {mem_history.get('min', 0)*100:.2f}%")
            print(f"  Max: {mem_history.get('max', 0)*100:.2f}%")
        
        print("\n🎮 GPU:")
        print(f"  Available GPUs: {stats.get('available_gpus', 0)}")
        gpu_workloads = stats.get('gpu_workloads', {})
        for gpu_id, workload in gpu_workloads.items():
            print(f"  {gpu_id}:")
            print(f"    Active Tasks: {workload.get('active_tasks', 0)}")
            print(f"    Memory Used: {workload.get('memory_used', 0):.2f} GB")
        
        print("\n📦 SESSIONS:")
        print(f"  Active Sessions: {stats.get('active_sessions', 0)}")
        
        print("\n🔧 CACHE:")
        cache_stats = stats.get('cache_stats', {})
        if cache_stats:
            print(f"  Cache Size: {cache_stats.get('size', 'N/A')}")
            print(f"  Hits: {cache_stats.get('hits', 0)}")
            print(f"  Misses: {cache_stats.get('misses', 0)}")
        
        cache_persistence = stats.get('cache_persistence', {})
        if cache_persistence.get('enabled'):
            print(f"  Cache Files: {cache_persistence.get('cache_files', 0)}")
            cache_size = cache_persistence.get('cache_size_bytes', 0)
            print(f"  Cache Size on Disk: {cache_size / (1024**2):.2f} MB")
    
    async def cmd_clear_cache(self, args):
        """Clear engine cache."""
        engine = self.create_engine()
        try:
            engine.clear_cache()
            print("✅ Cache cleared successfully")
        finally:
            engine.shutdown()
    
    async def cmd_cleanup_sessions(self, args):
        """Cleanup old sessions."""
        engine = self.create_engine()
        try:
            max_age = args.max_age if args.max_age else 3600
            engine.cleanup_sessions(max_age=max_age)
            print(f"✅ Sessions older than {max_age}s cleaned up")
        finally:
            engine.shutdown()
    
    async def cmd_health(self, args):
        """Check engine health."""
        if not HEALTH_AVAILABLE:
            print("Error: Health checker not available")
            return
        
        engine = self.create_engine()
        try:
            health_monitor = create_engine_health_checks(engine)
            results = await health_monitor.check_all()
            overall = health_monitor.get_overall_status(results)
            
            print("=" * 80)
            print("HEALTH CHECK REPORT")
            print("=" * 80)
            print(f"\nOverall Status: {overall.value.upper()}")
            print("\nComponent Status:")
            
            for name, result in results.items():
                status_icon = "✅" if result.status.value == "healthy" else "⚠️" if result.status.value == "degraded" else "❌"
                print(f"  {status_icon} {name}: {result.status.value}")
                print(f"      {result.message}")
            
        finally:
            engine.shutdown()
    
    async def cmd_monitor(self, args):
        """Start real-time monitoring."""
        if not MONITOR_AVAILABLE:
            print("Error: Monitor not available")
            return
        
        engine = self.create_engine()
        try:
            monitor = PerformanceMonitor(engine, check_interval=args.interval)
            
            # Start monitoring
            monitor_task = asyncio.create_task(monitor.start_monitoring())
            
            # Print dashboard
            if args.dashboard:
                from ultra_adaptive_kv_cache_monitor import print_dashboard
                dashboard_task = asyncio.create_task(
                    print_dashboard(monitor, refresh_interval=args.refresh)
                )
                try:
                    await dashboard_task
                except KeyboardInterrupt:
                    monitor.stop_monitoring()
                    dashboard_task.cancel()
            else:
                try:
                    await monitor_task
                except KeyboardInterrupt:
                    monitor.stop_monitoring()
                    monitor_task.cancel()
        finally:
            engine.shutdown()
    
    async def cmd_test(self, args):
        """Run test requests."""
        engine = self.create_engine()
        try:
            request = {
                'text': args.text if args.text else 'Test request',
                'max_length': args.max_length if args.max_length else 50,
                'temperature': args.temperature if args.temperature else 0.7,
                'session_id': args.session_id if args.session_id else 'cli_test'
            }
            
            print(f"Processing request: {request['text']}")
            start_time = time.time()
            result = await engine.process_request(request)
            duration = time.time() - start_time
            
            if result['success']:
                print(f"✅ Request processed in {duration*1000:.2f}ms")
                print(f"Response: {result['response'].get('text', '')[:200]}...")
            else:
                print(f"❌ Request failed: {result.get('error', 'Unknown error')}")
        finally:
            engine.shutdown()
    
    async def cmd_config(self, args):
        """Display or generate configuration."""
        if args.generate:
            config = AdaptiveConfig()
            config_dict = {
                'model_name': config.model_name,
                'model_size': config.model_size,
                'max_sequence_length': config.max_sequence_length,
                'cache_size': config.cache_size,
                'num_workers': config.num_workers,
                'enable_cache_persistence': config.enable_cache_persistence,
                'enable_checkpointing': config.enable_checkpointing,
                'use_multi_gpu': config.use_multi_gpu
            }
            
            output_file = args.output if args.output else 'engine_config.json'
            with open(output_file, 'w') as f:
                json.dump(config_dict, f, indent=2)
            
            print(f"✅ Configuration generated: {output_file}")
        else:
            config = AdaptiveConfig()
            print("=" * 80)
            print("CURRENT CONFIGURATION")
            print("=" * 80)
            print(json.dumps({
                'model_name': config.model_name,
                'model_size': config.model_size,
                'cache_size': config.cache_size,
                'num_workers': config.num_workers,
                'enable_cache_persistence': config.enable_cache_persistence,
                'enable_checkpointing': config.enable_checkpointing
            }, indent=2))


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Ultra-Adaptive K/V Cache Engine CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Display engine statistics')
    stats_parser.add_argument('--json', action='store_true', help='Output as JSON')
    stats_parser.add_argument('--config', type=str, help='Config file path')
    
    # Clear cache command
    clear_parser = subparsers.add_parser('clear-cache', help='Clear engine cache')
    clear_parser.add_argument('--config', type=str, help='Config file path')
    
    # Cleanup sessions command
    cleanup_parser = subparsers.add_parser('cleanup', help='Cleanup old sessions')
    cleanup_parser.add_argument('--max-age', type=int, help='Max age in seconds (default: 3600)')
    cleanup_parser.add_argument('--config', type=str, help='Config file path')
    
    # Health command
    health_parser = subparsers.add_parser('health', help='Check engine health')
    health_parser.add_argument('--config', type=str, help='Config file path')
    
    # Monitor command
    monitor_parser = subparsers.add_parser('monitor', help='Start monitoring')
    monitor_parser.add_argument('--dashboard', action='store_true', help='Show dashboard')
    monitor_parser.add_argument('--interval', type=float, default=5.0, help='Check interval (seconds)')
    monitor_parser.add_argument('--refresh', type=float, default=2.0, help='Dashboard refresh rate')
    monitor_parser.add_argument('--config', type=str, help='Config file path')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Run test request')
    test_parser.add_argument('--text', type=str, help='Request text')
    test_parser.add_argument('--max-length', type=int, help='Max length')
    test_parser.add_argument('--temperature', type=float, help='Temperature')
    test_parser.add_argument('--session-id', type=str, help='Session ID')
    test_parser.add_argument('--config', type=str, help='Config file path')
    
    # Config command
    config_parser = subparsers.add_parser('config', help='Display or generate configuration')
    config_parser.add_argument('--generate', action='store_true', help='Generate config file')
    config_parser.add_argument('--output', type=str, help='Output file path')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    cli = CLITool()
    
    # Route to command handler
    if args.command == 'stats':
        asyncio.run(cli.cmd_stats(args))
    elif args.command == 'clear-cache':
        asyncio.run(cli.cmd_clear_cache(args))
    elif args.command == 'cleanup':
        asyncio.run(cli.cmd_cleanup_sessions(args))
    elif args.command == 'health':
        asyncio.run(cli.cmd_health(args))
    elif args.command == 'monitor':
        asyncio.run(cli.cmd_monitor(args))
    elif args.command == 'test':
        asyncio.run(cli.cmd_test(args))
    elif args.command == 'config':
        asyncio.run(cli.cmd_config(args))


if __name__ == "__main__":
    main()

