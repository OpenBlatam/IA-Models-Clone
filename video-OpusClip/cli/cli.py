"""
Command Line Interface for Improved Video-OpusClip API

Comprehensive CLI tool with:
- API management commands
- Health monitoring
- Performance testing
- Configuration management
- Database operations
- Cache management
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Union
import asyncio
import click
import json
import time
import requests
from pathlib import Path
import structlog

from ..config import settings, validate_configuration
from ..database import db_manager, DatabaseMigrator
from ..cache import CacheManager, CacheConfig
from ..monitoring import PerformanceMonitor, HealthChecker, MonitoringConfig

logger = structlog.get_logger("cli")

# =============================================================================
# CLI CONFIGURATION
# =============================================================================

@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.option('--config', '-c', type=click.Path(), help='Configuration file path')
@click.pass_context
def cli(ctx, verbose, config):
    """Video-OpusClip API Management CLI"""
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose
    ctx.obj['config'] = config
    
    if verbose:
        click.echo("Verbose mode enabled")

# =============================================================================
# API MANAGEMENT COMMANDS
# =============================================================================

@cli.group()
def api():
    """API management commands"""
    pass

@api.command()
@click.option('--host', default='localhost', help='API host')
@click.option('--port', default=8000, help='API port')
@click.option('--timeout', default=30, help='Request timeout in seconds')
def health(host, port, timeout):
    """Check API health status"""
    try:
        url = f"http://{host}:{port}/health"
        response = requests.get(url, timeout=timeout)
        
        if response.status_code == 200:
            data = response.json()
            click.echo("✅ API is healthy")
            click.echo(f"Status: {data.get('status', 'unknown')}")
            click.echo(f"Timestamp: {data.get('timestamp', 'unknown')}")
            
            if data.get('issues'):
                click.echo("⚠️ Issues found:")
                for issue in data['issues']:
                    click.echo(f"  - {issue}")
        else:
            click.echo(f"❌ API health check failed: {response.status_code}")
            
    except requests.exceptions.RequestException as e:
        click.echo(f"❌ Failed to connect to API: {e}")

@api.command()
@click.option('--host', default='localhost', help='API host')
@click.option('--port', default=8000, help='API port')
def metrics(host, port):
    """Get API performance metrics"""
    try:
        url = f"http://{host}:{port}/metrics"
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            click.echo("📊 API Performance Metrics")
            click.echo("=" * 40)
            
            # Performance metrics
            if 'performance' in data:
                perf = data['performance']
                click.echo(f"Request Count: {perf.get('request_count', 0)}")
                click.echo(f"Average Response Time: {perf.get('response_time_avg', 0):.3f}s")
                click.echo(f"Error Count: {perf.get('error_count', 0)}")
                click.echo(f"Error Rate: {perf.get('error_rate', 0):.2f}%")
            
            # Database metrics
            if 'database' in data:
                db = data['database']
                click.echo(f"\nDatabase Queries: {db.get('queries_executed', 0)}")
                click.echo(f"Average Query Time: {db.get('average_query_time', 0):.3f}s")
                click.echo(f"Query Errors: {db.get('query_errors', 0)}")
            
            # Cache metrics
            if 'cache' in data:
                cache = data['cache']
                click.echo(f"\nCache Hit Rate: {cache.get('hit_rate_percent', 0):.1f}%")
                click.echo(f"Cache Hits: {cache.get('hits', 0)}")
                click.echo(f"Cache Misses: {cache.get('misses', 0)}")
        else:
            click.echo(f"❌ Failed to get metrics: {response.status_code}")
            
    except requests.exceptions.RequestException as e:
        click.echo(f"❌ Failed to connect to API: {e}")

@api.command()
@click.option('--host', default='localhost', help='API host')
@click.option('--port', default=8000, help='API port')
@click.option('--url', required=True, help='YouTube URL to process')
@click.option('--language', default='en', help='Video language')
@click.option('--quality', default='high', help='Video quality')
@click.option('--format', default='mp4', help='Video format')
def process_video(host, port, url, language, quality, format):
    """Process a single video"""
    try:
        api_url = f"http://{host}:{port}/api/v1/video/process"
        payload = {
            "youtube_url": url,
            "language": language,
            "max_clip_length": 60,
            "quality": quality,
            "format": format
        }
        
        click.echo(f"🎥 Processing video: {url}")
        response = requests.post(api_url, json=payload)
        
        if response.status_code == 200:
            data = response.json()
            click.echo("✅ Video processed successfully")
            click.echo(f"Title: {data.get('title', 'N/A')}")
            click.echo(f"Duration: {data.get('duration', 0)}s")
            click.echo(f"Processing Time: {data.get('processing_time', 0):.2f}s")
            click.echo(f"File Path: {data.get('file_path', 'N/A')}")
        else:
            click.echo(f"❌ Video processing failed: {response.status_code}")
            click.echo(f"Error: {response.text}")
            
    except requests.exceptions.RequestException as e:
        click.echo(f"❌ Failed to process video: {e}")

# =============================================================================
# DATABASE COMMANDS
# =============================================================================

@cli.group()
def db():
    """Database management commands"""
    pass

@db.command()
async def init():
    """Initialize database"""
    try:
        click.echo("🗄️ Initializing database...")
        await db_manager.initialize()
        click.echo("✅ Database initialized successfully")
    except Exception as e:
        click.echo(f"❌ Database initialization failed: {e}")

@db.command()
async def migrate():
    """Run database migrations"""
    try:
        click.echo("🔄 Running database migrations...")
        migrator = DatabaseMigrator(db_manager)
        await migrator.run_migrations()
        click.echo("✅ Database migrations completed successfully")
    except Exception as e:
        click.echo(f"❌ Database migration failed: {e}")

@db.command()
async def health():
    """Check database health"""
    try:
        click.echo("🔍 Checking database health...")
        health_status = await db_manager.health_check()
        
        if health_status.get('healthy'):
            click.echo("✅ Database is healthy")
            click.echo(f"Response Time: {health_status.get('response_time', 0):.3f}s")
            
            pool_info = health_status.get('pool_info', {})
            click.echo(f"Pool Size: {pool_info.get('size', 0)}")
            click.echo(f"Checked In: {pool_info.get('checked_in', 0)}")
            click.echo(f"Checked Out: {pool_info.get('checked_out', 0)}")
        else:
            click.echo("❌ Database is unhealthy")
            click.echo(f"Error: {health_status.get('error', 'Unknown error')}")
            
    except Exception as e:
        click.echo(f"❌ Database health check failed: {e}")

@db.command()
async def stats():
    """Get database statistics"""
    try:
        click.echo("📊 Database Statistics")
        click.echo("=" * 30)
        
        stats = db_manager.get_stats()
        click.echo(f"Queries Executed: {stats.get('queries_executed', 0)}")
        click.echo(f"Query Errors: {stats.get('query_errors', 0)}")
        click.echo(f"Average Query Time: {stats.get('average_query_time', 0):.3f}s")
        click.echo(f"Total Query Time: {stats.get('total_query_time', 0):.3f}s")
        
    except Exception as e:
        click.echo(f"❌ Failed to get database stats: {e}")

# =============================================================================
# CACHE COMMANDS
# =============================================================================

@cli.group()
def cache():
    """Cache management commands"""
    pass

@cache.command()
async def init():
    """Initialize cache"""
    try:
        click.echo("💾 Initializing cache...")
        cache_config = CacheConfig()
        cache_manager = CacheManager(cache_config)
        await cache_manager.initialize()
        click.echo("✅ Cache initialized successfully")
    except Exception as e:
        click.echo(f"❌ Cache initialization failed: {e}")

@cache.command()
async def stats():
    """Get cache statistics"""
    try:
        click.echo("📊 Cache Statistics")
        click.echo("=" * 25)
        
        cache_config = CacheConfig()
        cache_manager = CacheManager(cache_config)
        await cache_manager.initialize()
        
        stats = cache_manager.get_stats()
        click.echo(f"Hit Rate: {stats.get('hit_rate_percent', 0):.1f}%")
        click.echo(f"Hits: {stats.get('hits', 0)}")
        click.echo(f"Misses: {stats.get('misses', 0)}")
        click.echo(f"Total Requests: {stats.get('total_requests', 0)}")
        
        await cache_manager.close()
        
    except Exception as e:
        click.echo(f"❌ Failed to get cache stats: {e}")

@cache.command()
async def clear():
    """Clear cache"""
    try:
        click.echo("🗑️ Clearing cache...")
        cache_config = CacheConfig()
        cache_manager = CacheManager(cache_config)
        await cache_manager.initialize()
        await cache_manager.clear()
        click.echo("✅ Cache cleared successfully")
        await cache_manager.close()
    except Exception as e:
        click.echo(f"❌ Failed to clear cache: {e}")

# =============================================================================
# CONFIGURATION COMMANDS
# =============================================================================

@cli.group()
def config():
    """Configuration management commands"""
    pass

@config.command()
def validate():
    """Validate configuration"""
    try:
        click.echo("🔍 Validating configuration...")
        if validate_configuration():
            click.echo("✅ Configuration is valid")
        else:
            click.echo("❌ Configuration validation failed")
    except Exception as e:
        click.echo(f"❌ Configuration validation error: {e}")

@config.command()
def show():
    """Show current configuration"""
    try:
        click.echo("⚙️ Current Configuration")
        click.echo("=" * 30)
        click.echo(f"App Name: {settings.app_name}")
        click.echo(f"Version: {settings.app_version}")
        click.echo(f"Environment: {settings.environment}")
        click.echo(f"Debug: {settings.debug}")
        click.echo(f"Host: {settings.host}")
        click.echo(f"Port: {settings.port}")
        click.echo(f"Workers: {settings.workers}")
        click.echo(f"Log Level: {settings.log_level}")
        click.echo(f"Database URL: {settings.database_url}")
        click.echo(f"Redis Host: {settings.redis_host}")
        click.echo(f"Cache Enabled: {settings.cache_enabled}")
        click.echo(f"Performance Monitoring: {settings.enable_performance_monitoring}")
    except Exception as e:
        click.echo(f"❌ Failed to show configuration: {e}")

@config.command()
@click.option('--output', '-o', type=click.Path(), help='Output file path')
def export(output):
    """Export configuration to file"""
    try:
        config_data = {
            "app_name": settings.app_name,
            "app_version": settings.app_version,
            "environment": settings.environment,
            "debug": settings.debug,
            "host": settings.host,
            "port": settings.port,
            "workers": settings.workers,
            "log_level": settings.log_level,
            "database_url": settings.database_url,
            "redis_host": settings.redis_host,
            "cache_enabled": settings.cache_enabled,
            "enable_performance_monitoring": settings.enable_performance_monitoring
        }
        
        if output:
            with open(output, 'w') as f:
                json.dump(config_data, f, indent=2)
            click.echo(f"✅ Configuration exported to {output}")
        else:
            click.echo(json.dumps(config_data, indent=2))
            
    except Exception as e:
        click.echo(f"❌ Failed to export configuration: {e}")

# =============================================================================
# PERFORMANCE TESTING COMMANDS
# =============================================================================

@cli.group()
def test():
    """Performance testing commands"""
    pass

@test.command()
@click.option('--host', default='localhost', help='API host')
@click.option('--port', default=8000, help='API port')
@click.option('--requests', default=100, help='Number of requests')
@click.option('--concurrent', default=10, help='Concurrent requests')
def load(host, port, requests, concurrent):
    """Run load test against API"""
    try:
        click.echo(f"🚀 Running load test: {requests} requests, {concurrent} concurrent")
        
        url = f"http://{host}:{port}/health"
        start_time = time.time()
        
        # Simple load test
        success_count = 0
        error_count = 0
        response_times = []
        
        for i in range(requests):
            try:
                req_start = time.time()
                response = requests.get(url, timeout=10)
                req_time = time.time() - req_start
                response_times.append(req_time)
                
                if response.status_code == 200:
                    success_count += 1
                else:
                    error_count += 1
                    
            except Exception as e:
                error_count += 1
            
            if (i + 1) % 10 == 0:
                click.echo(f"Completed {i + 1}/{requests} requests")
        
        total_time = time.time() - start_time
        
        # Calculate statistics
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        min_response_time = min(response_times) if response_times else 0
        max_response_time = max(response_times) if response_times else 0
        requests_per_second = requests / total_time
        success_rate = (success_count / requests) * 100
        
        click.echo("\n📊 Load Test Results")
        click.echo("=" * 25)
        click.echo(f"Total Requests: {requests}")
        click.echo(f"Successful: {success_count}")
        click.echo(f"Errors: {error_count}")
        click.echo(f"Success Rate: {success_rate:.1f}%")
        click.echo(f"Total Time: {total_time:.2f}s")
        click.echo(f"Requests/Second: {requests_per_second:.1f}")
        click.echo(f"Average Response Time: {avg_response_time:.3f}s")
        click.echo(f"Min Response Time: {min_response_time:.3f}s")
        click.echo(f"Max Response Time: {max_response_time:.3f}s")
        
    except Exception as e:
        click.echo(f"❌ Load test failed: {e}")

# =============================================================================
# MAIN CLI ENTRY POINT
# =============================================================================

def main():
    """Main CLI entry point"""
    cli()

if __name__ == "__main__":
    main()






























