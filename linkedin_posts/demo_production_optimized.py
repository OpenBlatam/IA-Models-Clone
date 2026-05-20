from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import time
import json
import logging
from typing import Dict, List, Any
from datetime import datetime, timedelta
from uuid import uuid4
import aiohttp
import uvloop
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.panel import Panel
from rich.text import Text
from rich.live import Live
from rich.layout import Layout
from rich.align import Align
import statistics
import asyncio_throttle
from typing import Any, List, Dict, Optional
#!/usr/bin/env python3
"""
Production-optimized demo script for LinkedIn Posts API.
Showcases all advanced features with performance optimizations.
"""


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Rich console for beautiful output
console = Console()

# Performance metrics
class PerformanceMetrics:
    def __init__(self) -> Any:
        self.response_times = []
        self.success_count = 0
        self.error_count = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.start_time = time.time()
    
    def add_response_time(self, response_time: float):
        
    """add_response_time function."""
self.response_times.append(response_time)
    
    def add_success(self) -> Any:
        self.success_count += 1
    
    def add_error(self) -> Any:
        self.error_count += 1
    
    def add_cache_hit(self) -> Any:
        self.cache_hits += 1
    
    def add_cache_miss(self) -> Any:
        self.cache_misses += 1
    
    def get_stats(self) -> Dict[str, Any]:
        if not self.response_times:
            return {"error": "No response times recorded"}
        
        return {
            "total_requests": len(self.response_times),
            "success_count": self.success_count,
            "error_count": self.error_count,
            "success_rate": (self.success_count / (self.success_count + self.error_count)) * 100,
            "avg_response_time": statistics.mean(self.response_times),
            "min_response_time": min(self.response_times),
            "max_response_time": max(self.response_times),
            "median_response_time": statistics.median(self.response_times),
            "p95_response_time": statistics.quantiles(self.response_times, n=20)[18],
            "p99_response_time": statistics.quantiles(self.response_times, n=100)[98],
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": (self.cache_hits / (self.cache_hits + self.cache_misses)) * 100 if (self.cache_hits + self.cache_misses) > 0 else 0,
            "total_duration": time.time() - self.start_time,
            "requests_per_second": len(self.response_times) / (time.time() - self.start_time)
        }


class OptimizedAPIClient:
    """High-performance API client with connection pooling and optimizations."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        
    """__init__ function."""
self.base_url = base_url
        self.session = None
        self.metrics = PerformanceMetrics()
        self.throttle = asyncio_throttle.Throttler(rate_limit=100, period=1.0)  # 100 requests per second
    
    async def __aenter__(self) -> Any:
        # Optimized connector settings
        connector = aiohttp.TCPConnector(
            limit=100,  # Total connection pool size
            limit_per_host=30,  # Connections per host
            ttl_dns_cache=300,  # DNS cache TTL
            use_dns_cache=True,
            keepalive_timeout=30,
            enable_cleanup_closed=True
        )
        
        # Optimized timeout settings
        timeout = aiohttp.ClientTimeout(
            total=30,
            connect=5,
            sock_read=10
        )
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={
                "User-Agent": "LinkedIn-Posts-Demo/2.0",
                "Accept": "application/json",
                "Content-Type": "application/json"
            }
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> Any:
        if self.session:
            await self.session.close()
    
    async async def make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make an optimized HTTP request with metrics tracking."""
        async with self.throttle:
            start_time = time.time()
            
            try:
                url = f"{self.base_url}{endpoint}"
                
                async with self.session.request(method, url, **kwargs) as response:
                    response_time = time.time() - start_time
                    self.metrics.add_response_time(response_time)
                    
                    if response.status == 200:
                        self.metrics.add_success()
                        
                        # Check for cache headers
                        if response.headers.get('X-Cache-Status') == 'HIT':
                            self.metrics.add_cache_hit()
                        else:
                            self.metrics.add_cache_miss()
                        
                        return await response.json()
                    else:
                        self.metrics.add_error()
                        return {"error": f"HTTP {response.status}", "message": await response.text()}
                        
            except Exception as e:
                response_time = time.time() - start_time
                self.metrics.add_response_time(response_time)
                self.metrics.add_error()
                return {"error": str(e)}


class ProductionDemo:
    """Production-optimized demo showcasing all features."""
    
    def __init__(self) -> Any:
        self.client = OptimizedAPIClient()
        self.demo_data = self._generate_demo_data()
    
    def _generate_demo_data(self) -> Dict[str, Any]:
        """Generate realistic demo data."""
        return {
            "users": [
                {
                    "id": str(uuid4()),
                    "email": f"user{i}@example.com",
                    "name": f"Demo User {i}",
                    "industry": ["Technology", "Marketing", "Finance", "Healthcare"][i % 4],
                    "target_audience": ["Professionals", "Entrepreneurs", "Students", "Executives"][i % 4]
                }
                for i in range(10)
            ],
            "topics": [
                "AI and Machine Learning in Business",
                "Remote Work Best Practices",
                "Digital Marketing Trends 2024",
                "Sustainable Business Practices",
                "Leadership in the Digital Age",
                "Data-Driven Decision Making",
                "Customer Experience Innovation",
                "Cybersecurity for Businesses",
                "Future of Work",
                "Building High-Performance Teams"
            ],
            "templates": [
                {
                    "name": "Professional Insight",
                    "category": "business",
                    "content": "🔍 Insight: {insight}\n\n💡 Key takeaway: {takeaway}\n\n🚀 Action step: {action}\n\n#Professional #Business #Growth"
                },
                {
                    "name": "Industry Update",
                    "category": "marketing",
                    "content": "📈 Industry Update: {update}\n\n🎯 Impact: {impact}\n\n💭 What do you think? Share your perspective!\n\n#Industry #Marketing #Trends"
                },
                {
                    "name": "Leadership Tip",
                    "category": "educational",
                    "content": "🌟 Leadership Tip: {tip}\n\n✅ How to apply: {application}\n\n🤝 Tag someone who needs to see this!\n\n#Leadership #Management #Success"
                }
            ]
        }
    
    async def run_comprehensive_demo(self) -> Any:
        """Run comprehensive production demo."""
        console.print(Panel.fit(
            "[bold blue]LinkedIn Posts API - Production Demo[/bold blue]\n"
            "[green]Showcasing advanced features with performance optimizations[/green]",
            title="🚀 Production Demo",
            border_style="blue"
        ))
        
        async with self.client:
            # Demo sections
            await self._demo_health_check()
            await self._demo_bulk_operations()
            await self._demo_ai_features()
            await self._demo_analytics()
            await self._demo_templates()
            await self._demo_performance_testing()
            await self._demo_cache_performance()
            
            # Final performance report
            await self._show_performance_report()
    
    async def _demo_health_check(self) -> Any:
        """Demo health check and system status."""
        console.print("\n[bold yellow]1. Health Check & System Status[/bold yellow]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True
        ) as progress:
            task = progress.add_task("Checking system health...", total=None)
            
            # Health check
            health_result = await self.client.make_request("GET", "/health")
            
            # Metrics
            metrics_result = await self.client.make_request("GET", "/metrics")
            
            progress.update(task, completed=True)
        
        # Display results
        health_table = Table(title="System Health")
        health_table.add_column("Component", style="cyan")
        health_table.add_column("Status", style="green")
        health_table.add_column("Response Time", style="yellow")
        
        if "error" not in health_result:
            health_table.add_row("API", "✅ Healthy", f"{health_result.get('response_time', 0):.2f}ms")
            health_table.add_row("Database", "✅ Connected", f"{health_result.get('db_response_time', 0):.2f}ms")
            health_table.add_row("Cache", "✅ Active", f"{health_result.get('cache_response_time', 0):.2f}ms")
        else:
            health_table.add_row("API", "❌ Error", health_result.get("error", "Unknown"))
        
        console.print(health_table)
    
    async def _demo_bulk_operations(self) -> Any:
        """Demo bulk operations with parallel processing."""
        console.print("\n[bold yellow]2. Bulk Operations & Parallel Processing[/bold yellow]")
        
        # Prepare bulk data
        bulk_posts = []
        for i, topic in enumerate(self.demo_data["topics"][:5]):
            bulk_posts.append({
                "title": f"Post {i+1}",
                "content": f"Exploring {topic}. This is a comprehensive look at current trends and future implications.",
                "hashtags": ["#Professional", "#Business", f"#Topic{i+1}"],
                "tone": "professional",
                "enable_ai_optimization": True
            })
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task("Creating bulk posts...", total=len(bulk_posts))
            
            # Create posts in parallel
            tasks = []
            for post_data in bulk_posts:
                task_coro = self.client.make_request("POST", "/api/v1/posts", json=post_data)
                tasks.append(task_coro)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            progress.update(task, completed=len(bulk_posts))
        
        # Display results
        bulk_table = Table(title="Bulk Operations Results")
        bulk_table.add_column("Operation", style="cyan")
        bulk_table.add_column("Count", style="green")
        bulk_table.add_column("Success Rate", style="yellow")
        
        successful = sum(1 for r in results if isinstance(r, dict) and "error" not in r)
        success_rate = (successful / len(results)) * 100
        
        bulk_table.add_row("Bulk Post Creation", str(len(bulk_posts)), f"{success_rate:.1f}%")
        
        console.print(bulk_table)
    
    async def _demo_ai_features(self) -> Any:
        """Demo AI features and optimizations."""
        console.print("\n[bold yellow]3. AI Features & Content Optimization[/bold yellow]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True
        ) as progress:
            task = progress.add_task("Testing AI features...", total=None)
            
            # AI content generation
            ai_generation = await self.client.make_request("POST", "/api/v1/ai/generate", json={
                "topic": "Future of Remote Work",
                "tone": "professional",
                "target_audience": "Business Leaders",
                "include_hashtags": True,
                "include_call_to_action": True
            })
            
            # Content optimization
            optimization = await self.client.make_request("POST", "/api/v1/ai/optimize", json={
                "content": "Remote work is changing how we do business.",
                "target_audience": "Professionals",
                "industry": "Technology"
            })
            
            # Hashtag suggestions
            hashtags = await self.client.make_request("POST", "/api/v1/ai/hashtags", json={
                "content": "Exploring the future of artificial intelligence in business",
                "industry": "Technology"
            })
            
            progress.update(task, completed=True)
        
        # Display AI results
        ai_table = Table(title="AI Features Performance")
        ai_table.add_column("Feature", style="cyan")
        ai_table.add_column("Status", style="green")
        ai_table.add_column("Quality Score", style="yellow")
        
        ai_table.add_row(
            "Content Generation",
            "✅ Success" if "error" not in ai_generation else "❌ Error",
            f"{ai_generation.get('quality_score', 0):.2f}" if "error" not in ai_generation else "N/A"
        )
        
        ai_table.add_row(
            "Content Optimization",
            "✅ Success" if "error" not in optimization else "❌ Error",
            f"{optimization.get('optimization_score', 0):.2f}" if "error" not in optimization else "N/A"
        )
        
        ai_table.add_row(
            "Hashtag Suggestions",
            "✅ Success" if "error" not in hashtags else "❌ Error",
            f"{len(hashtags.get('hashtags', []))}" if "error" not in hashtags else "N/A"
        )
        
        console.print(ai_table)
    
    async def _demo_analytics(self) -> Any:
        """Demo analytics and insights."""
        console.print("\n[bold yellow]4. Analytics & Performance Insights[/bold yellow]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True
        ) as progress:
            task = progress.add_task("Generating analytics...", total=None)
            
            # Get analytics data
            analytics = await self.client.make_request("GET", "/api/v1/analytics/dashboard")
            
            # Performance metrics
            performance = await self.client.make_request("GET", "/api/v1/analytics/performance")
            
            # Engagement insights
            engagement = await self.client.make_request("GET", "/api/v1/analytics/engagement")
            
            progress.update(task, completed=True)
        
        # Display analytics
        analytics_table = Table(title="Analytics Dashboard")
        analytics_table.add_column("Metric", style="cyan")
        analytics_table.add_column("Value", style="green")
        analytics_table.add_column("Trend", style="yellow")
        
        if "error" not in analytics:
            analytics_table.add_row("Total Posts", str(analytics.get("total_posts", 0)), "📈 +15%")
            analytics_table.add_row("Avg Engagement", f"{analytics.get('avg_engagement', 0):.1f}%", "📈 +8%")
            analytics_table.add_row("Reach", f"{analytics.get('total_reach', 0):,}", "📈 +22%")
        else:
            analytics_table.add_row("Analytics", "❌ Error", analytics.get("error", "Unknown"))
        
        console.print(analytics_table)
    
    async def _demo_templates(self) -> Any:
        """Demo template system."""
        console.print("\n[bold yellow]5. Template System & Content Automation[/bold yellow]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True
        ) as progress:
            task = progress.add_task("Testing templates...", total=None)
            
            # Create template
            template_data = {
                "name": "Demo Template",
                "category": "business",
                "content_template": "💡 {insight}\n\n🎯 Key takeaway: {takeaway}\n\n#Business #Growth",
                "variables": [
                    {"name": "insight", "description": "Main insight", "required": True},
                    {"name": "takeaway", "description": "Key takeaway", "required": True}
                ]
            }
            
            template_result = await self.client.make_request("POST", "/api/v1/templates", json=template_data)
            
            # Use template
            if "error" not in template_result:
                template_id = template_result.get("id")
                post_from_template = await self.client.make_request("POST", f"/api/v1/templates/{template_id}/generate", json={
                    "variables": {
                        "insight": "Remote work productivity has increased by 25% in 2024",
                        "takeaway": "Flexible work arrangements drive better results"
                    }
                })
            
            progress.update(task, completed=True)
        
        # Display template results
        template_table = Table(title="Template System")
        template_table.add_column("Operation", style="cyan")
        template_table.add_column("Status", style="green")
        template_table.add_column("Result", style="yellow")
        
        template_table.add_row(
            "Template Creation",
            "✅ Success" if "error" not in template_result else "❌ Error",
            template_result.get("name", "Error") if "error" not in template_result else template_result.get("error")
        )
        
        if "error" not in template_result:
            template_table.add_row(
                "Content Generation",
                "✅ Success" if "error" not in post_from_template else "❌ Error",
                f"{len(post_from_template.get('content', ''))} chars" if "error" not in post_from_template else "Error"
            )
        
        console.print(template_table)
    
    async def _demo_performance_testing(self) -> Any:
        """Demo performance testing with concurrent requests."""
        console.print("\n[bold yellow]6. Performance Testing & Load Simulation[/bold yellow]")
        
        concurrent_requests = 50
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task("Running load test...", total=concurrent_requests)
            
            # Create concurrent requests
            tasks = []
            for i in range(concurrent_requests):
                task_coro = self.client.make_request("GET", "/health")
                tasks.append(task_coro)
            
            # Execute with progress updates
            completed = 0
            for coro in asyncio.as_completed(tasks):
                await coro
                completed += 1
                progress.update(task, completed=completed)
        
        # Performance results
        stats = self.client.metrics.get_stats()
        
        perf_table = Table(title="Performance Test Results")
        perf_table.add_column("Metric", style="cyan")
        perf_table.add_column("Value", style="green")
        perf_table.add_column("Target", style="yellow")
        
        perf_table.add_row("Concurrent Requests", str(concurrent_requests), "50")
        perf_table.add_row("Success Rate", f"{stats.get('success_rate', 0):.1f}%", ">95%")
        perf_table.add_row("Avg Response Time", f"{stats.get('avg_response_time', 0)*1000:.1f}ms", "<100ms")
        perf_table.add_row("P95 Response Time", f"{stats.get('p95_response_time', 0)*1000:.1f}ms", "<200ms")
        perf_table.add_row("Requests/Second", f"{stats.get('requests_per_second', 0):.1f}", ">100")
        
        console.print(perf_table)
    
    async def _demo_cache_performance(self) -> Any:
        """Demo cache performance and optimization."""
        console.print("\n[bold yellow]7. Cache Performance & Optimization[/bold yellow]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True
        ) as progress:
            task = progress.add_task("Testing cache performance...", total=None)
            
            # Test cache with repeated requests
            cache_test_requests = 20
            endpoint = "/api/v1/posts"
            
            for _ in range(cache_test_requests):
                await self.client.make_request("GET", endpoint)
            
            progress.update(task, completed=True)
        
        # Cache performance results
        stats = self.client.metrics.get_stats()
        
        cache_table = Table(title="Cache Performance")
        cache_table.add_column("Metric", style="cyan")
        cache_table.add_column("Value", style="green")
        cache_table.add_column("Target", style="yellow")
        
        cache_table.add_row("Cache Hit Rate", f"{stats.get('cache_hit_rate', 0):.1f}%", ">80%")
        cache_table.add_row("Cache Hits", str(stats.get('cache_hits', 0)), "High")
        cache_table.add_row("Cache Misses", str(stats.get('cache_misses', 0)), "Low")
        
        console.print(cache_table)
    
    async def _show_performance_report(self) -> Any:
        """Show comprehensive performance report."""
        console.print("\n[bold green]📊 Final Performance Report[/bold green]")
        
        stats = self.client.metrics.get_stats()
        
        # Create performance summary
        report_table = Table(title="Performance Summary", title_style="bold green")
        report_table.add_column("Category", style="cyan", width=20)
        report_table.add_column("Metric", style="white", width=25)
        report_table.add_column("Value", style="green", width=15)
        report_table.add_column("Status", style="yellow", width=10)
        
        # Response times
        avg_time = stats.get('avg_response_time', 0) * 1000
        report_table.add_row("Response Times", "Average", f"{avg_time:.1f}ms", "✅" if avg_time < 100 else "⚠️")
        
        p95_time = stats.get('p95_response_time', 0) * 1000
        report_table.add_row("", "95th Percentile", f"{p95_time:.1f}ms", "✅" if p95_time < 200 else "⚠️")
        
        # Throughput
        rps = stats.get('requests_per_second', 0)
        report_table.add_row("Throughput", "Requests/Second", f"{rps:.1f}", "✅" if rps > 50 else "⚠️")
        
        # Reliability
        success_rate = stats.get('success_rate', 0)
        report_table.add_row("Reliability", "Success Rate", f"{success_rate:.1f}%", "✅" if success_rate > 95 else "⚠️")
        
        # Cache
        cache_hit_rate = stats.get('cache_hit_rate', 0)
        report_table.add_row("Caching", "Hit Rate", f"{cache_hit_rate:.1f}%", "✅" if cache_hit_rate > 70 else "⚠️")
        
        console.print(report_table)
        
        # Performance grade
        grade = self._calculate_performance_grade(stats)
        grade_color = "green" if grade in ["A", "B"] else "yellow" if grade == "C" else "red"
        
        console.print(Panel.fit(
            f"[bold {grade_color}]Performance Grade: {grade}[/bold {grade_color}]\n"
            f"Total Requests: {stats.get('total_requests', 0)}\n"
            f"Total Duration: {stats.get('total_duration', 0):.1f}s",
            title="🏆 Final Score",
            border_style=grade_color
        ))
    
    def _calculate_performance_grade(self, stats: Dict[str, Any]) -> str:
        """Calculate performance grade based on metrics."""
        score = 0
        
        # Response time score (40%)
        avg_time = stats.get('avg_response_time', 0) * 1000
        if avg_time < 50:
            score += 40
        elif avg_time < 100:
            score += 30
        elif avg_time < 200:
            score += 20
        else:
            score += 10
        
        # Success rate score (30%)
        success_rate = stats.get('success_rate', 0)
        if success_rate > 99:
            score += 30
        elif success_rate > 95:
            score += 25
        elif success_rate > 90:
            score += 20
        else:
            score += 10
        
        # Throughput score (20%)
        rps = stats.get('requests_per_second', 0)
        if rps > 100:
            score += 20
        elif rps > 50:
            score += 15
        elif rps > 25:
            score += 10
        else:
            score += 5
        
        # Cache performance score (10%)
        cache_hit_rate = stats.get('cache_hit_rate', 0)
        if cache_hit_rate > 90:
            score += 10
        elif cache_hit_rate > 70:
            score += 8
        elif cache_hit_rate > 50:
            score += 6
        else:
            score += 3
        
        # Convert to grade
        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"


async def main():
    """Main demo function."""
    # Use uvloop for better performance
    if hasattr(asyncio, 'set_event_loop_policy'):
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    
    demo = ProductionDemo()
    await demo.run_comprehensive_demo()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[red]Demo interrupted by user[/red]")
    except Exception as e:
        console.print(f"\n[red]Demo failed with error: {e}[/red]")
        logger.exception("Demo failed") 