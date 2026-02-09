#!/usr/bin/env python3
"""
API Dashboard
=============
Real-time dashboard for API monitoring and debugging.

⚠️ DEPRECATED: This file is deprecated. Consider migrating to the new tools structure.

For new code, use:
    from tools.manager import ToolManager
    manager = ToolManager()
    # Tools can be extended with dashboard capabilities
"""
import warnings

warnings.warn(
    "api_dashboard.py is deprecated. Consider migrating to the new tools structure in tools/.",
    DeprecationWarning,
    stacklevel=2
)

import time
import json
import requests
from datetime import datetime, timedelta
from typing import Dict, Any, List
from collections import deque
import threading
from pathlib import Path


class APIDashboard:
    """Real-time API dashboard."""
    
    def __init__(self, base_url: str = "http://localhost:8000", history_size: int = 1000):
        self.base_url = base_url
        self.history_size = history_size
        self.metrics = {
            "requests": deque(maxlen=history_size),
            "errors": deque(maxlen=history_size),
            "response_times": deque(maxlen=history_size),
            "status_codes": {},
            "endpoints": {},
            "start_time": datetime.now(),
            "last_update": datetime.now()
        }
        self.running = False
        self.monitor_thread = None
        self.update_interval = 1.0
    
    def check_health(self) -> Dict[str, Any]:
        """Check API health."""
        start_time = time.time()
        
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            response_time = (time.time() - start_time) * 1000
            
            result = {
                "timestamp": datetime.now().isoformat(),
                "status": response.status_code,
                "response_time": response_time,
                "healthy": response.status_code == 200,
                "data": response.json() if response.status_code == 200 else None
            }
            
            self._record_metric(result, response_time)
            return result
        
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            result = {
                "timestamp": datetime.now().isoformat(),
                "status": "error",
                "response_time": response_time,
                "healthy": False,
                "error": str(e)
            }
            self._record_metric(result, response_time)
            return result
    
    def _record_metric(self, result: Dict[str, Any], response_time: float):
        """Record metric."""
        self.metrics["requests"].append(result)
        self.metrics["response_times"].append(response_time)
        self.metrics["last_update"] = datetime.now()
        
        status = str(result.get("status", "error"))
        self.metrics["status_codes"][status] = self.metrics["status_codes"].get(status, 0) + 1
        
        if not result.get("healthy", False):
            self.metrics["errors"].append(result)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        response_times = list(self.metrics["response_times"])
        requests_list = list(self.metrics["requests"])
        
        stats = {
            "total_requests": len(requests_list),
            "total_errors": len(self.metrics["errors"]),
            "uptime_seconds": (datetime.now() - self.metrics["start_time"]).total_seconds(),
            "last_update": self.metrics["last_update"].isoformat(),
            "status_codes": dict(self.metrics["status_codes"]),
        }
        
        if response_times:
            sorted_times = sorted(response_times)
            stats["response_time"] = {
                "min": min(response_times),
                "max": max(response_times),
                "avg": sum(response_times) / len(response_times),
                "median": sorted_times[len(sorted_times) // 2],
                "p95": sorted_times[int(len(sorted_times) * 0.95)] if len(sorted_times) > 0 else 0,
                "p99": sorted_times[int(len(sorted_times) * 0.99)] if len(sorted_times) > 0 else 0,
                "current": response_times[-1] if response_times else 0
            }
        
        # Calculate rates
        if len(requests_list) > 0:
            stats["error_rate"] = (len(self.metrics["errors"]) / len(requests_list)) * 100
            healthy_count = sum(1 for r in requests_list if r.get("healthy", False))
            stats["availability"] = (healthy_count / len(requests_list)) * 100
        else:
            stats["error_rate"] = 0
            stats["availability"] = 0
        
        # Calculate requests per second
        uptime = stats["uptime_seconds"]
        if uptime > 0:
            stats["requests_per_second"] = len(requests_list) / uptime
        else:
            stats["requests_per_second"] = 0
        
        # Recent activity (last minute)
        one_minute_ago = datetime.now() - timedelta(minutes=1)
        recent_requests = [
            r for r in requests_list
            if datetime.fromisoformat(r["timestamp"]) > one_minute_ago
        ]
        stats["recent_requests"] = len(recent_requests)
        stats["recent_errors"] = len([
            r for r in recent_requests if not r.get("healthy", False)
        ])
        
        return stats
    
    def monitor_continuous(self, interval: float = 1.0):
        """Start continuous monitoring."""
        self.running = True
        self.update_interval = interval
        
        def monitor_loop():
            while self.running:
                self.check_health()
                time.sleep(interval)
        
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop continuous monitoring."""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)
    
    def print_dashboard(self):
        """Print formatted dashboard."""
        stats = self.get_statistics()
        
        # Clear screen (works on most terminals)
        print("\033[2J\033[H", end="")
        
        # Header
        print("=" * 80)
        print("📊 API Dashboard - Real-time Monitoring".center(80))
        print("=" * 80)
        print()
        
        # Status
        status_emoji = "🟢" if stats["availability"] > 99 else "🟡" if stats["availability"] > 95 else "🔴"
        print(f"{status_emoji} Status: {'HEALTHY' if stats['availability'] > 99 else 'DEGRADED' if stats['availability'] > 95 else 'UNHEALTHY'}")
        print(f"   Availability: {stats['availability']:.2f}%")
        print(f"   Error Rate: {stats['error_rate']:.2f}%")
        print()
        
        # Requests
        print("📈 Requests:")
        print(f"   Total: {stats['total_requests']}")
        print(f"   Recent (1min): {stats['recent_requests']}")
        print(f"   Rate: {stats['requests_per_second']:.2f} req/s")
        print()
        
        # Response Time
        if stats.get("response_time"):
            rt = stats["response_time"]
            print("⏱️  Response Time:")
            print(f"   Current: {rt['current']:.2f}ms")
            print(f"   Average: {rt['avg']:.2f}ms")
            print(f"   Median: {rt['median']:.2f}ms")
            print(f"   P95: {rt['p95']:.2f}ms")
            print(f"   P99: {rt['p99']:.2f}ms")
            print(f"   Min: {rt['min']:.2f}ms")
            print(f"   Max: {rt['max']:.2f}ms")
            print()
        
        # Status Codes
        if stats["status_codes"]:
            print("📊 Status Codes:")
            for code, count in sorted(stats["status_codes"].items()):
                emoji = "✅" if code == "200" else "⚠️" if code.startswith("4") else "❌"
                print(f"   {emoji} {code}: {count}")
            print()
        
        # Errors
        if stats["total_errors"] > 0:
            print(f"❌ Errors: {stats['total_errors']}")
            print(f"   Recent (1min): {stats['recent_errors']}")
            print()
        
        # Uptime
        uptime = stats["uptime_seconds"]
        hours = int(uptime // 3600)
        minutes = int((uptime % 3600) // 60)
        seconds = int(uptime % 60)
        print(f"⏰ Uptime: {hours:02d}:{minutes:02d}:{seconds:02d}")
        print(f"   Last Update: {stats['last_update']}")
        print()
        
        print("=" * 80)
        print("Press Ctrl+C to stop")
        print("=" * 80)
    
    def run_dashboard(self, refresh_interval: float = 2.0):
        """Run live dashboard."""
        self.monitor_continuous(interval=1.0)
        
        try:
            while True:
                self.print_dashboard()
                time.sleep(refresh_interval)
        except KeyboardInterrupt:
            self.stop_monitoring()
            print("\n\n👋 Dashboard stopped")
    
    def export_data(self, file_path: Path):
        """Export dashboard data."""
        data = {
            "statistics": self.get_statistics(),
            "recent_requests": list(self.metrics["requests"])[-100:],
            "recent_errors": list(self.metrics["errors"])[-50:],
            "exported_at": datetime.now().isoformat()
        }
        
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)
        
        print(f"✅ Dashboard data exported to {file_path}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="API Dashboard")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--interval", type=float, default=2.0, help="Refresh interval (seconds)")
    parser.add_argument("--export", help="Export data to file")
    
    args = parser.parse_args()
    
    dashboard = APIDashboard(base_url=args.url)
    
    if args.export:
        # Run for a bit then export
        dashboard.monitor_continuous(interval=1.0)
        time.sleep(10)  # Collect data for 10 seconds
        dashboard.stop_monitoring()
        dashboard.export_data(Path(args.export))
    else:
        # Run live dashboard
        dashboard.run_dashboard(refresh_interval=args.interval)


if __name__ == "__main__":
    main()



