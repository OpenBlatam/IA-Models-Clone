#!/usr/bin/env python3
"""
API Monitor
===========
Real-time monitoring tool for the API.

⚠️ DEPRECATED: This file is deprecated. Consider migrating to the new tools structure.

For new code, use:
    from tools.manager import ToolManager
    manager = ToolManager()
    # Tools can be extended with monitoring capabilities
"""
import warnings

warnings.warn(
    "api_monitor.py is deprecated. Consider migrating to the new tools structure in tools/.",
    DeprecationWarning,
    stacklevel=2
)

import time
import json
import requests
from datetime import datetime
from typing import Dict, Any, List
from pathlib import Path
from collections import deque
import threading


class APIMonitor:
    """Real-time API monitoring."""
    
    def __init__(self, base_url: str = "http://localhost:8000", history_size: int = 100):
        self.base_url = base_url
        self.history_size = history_size
        self.metrics = {
            "requests": deque(maxlen=history_size),
            "errors": deque(maxlen=history_size),
            "response_times": deque(maxlen=history_size),
            "status_codes": {},
            "endpoints": {},
            "start_time": datetime.now()
        }
        self.running = False
        self.monitor_thread = None
    
    def check_health(self) -> Dict[str, Any]:
        """Check API health and record metrics."""
        start_time = time.time()
        
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            response_time = (time.time() - start_time) * 1000  # ms
            
            result = {
                "timestamp": datetime.now().isoformat(),
                "status": response.status_code,
                "response_time": response_time,
                "healthy": response.status_code == 200,
                "data": response.json() if response.status_code == 200 else None
            }
            
            # Record metrics
            self.metrics["requests"].append(result)
            self.metrics["response_times"].append(response_time)
            
            # Update status codes
            status = str(response.status_code)
            self.metrics["status_codes"][status] = self.metrics["status_codes"].get(status, 0) + 1
            
            if response.status_code >= 400:
                self.metrics["errors"].append(result)
            
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
            
            self.metrics["requests"].append(result)
            self.metrics["errors"].append(result)
            
            return result
    
    def monitor_continuous(self, interval: float = 1.0):
        """Start continuous monitoring."""
        self.running = True
        
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
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get monitoring statistics."""
        response_times = list(self.metrics["response_times"])
        
        stats = {
            "total_requests": len(self.metrics["requests"]),
            "total_errors": len(self.metrics["errors"]),
            "uptime_seconds": (datetime.now() - self.metrics["start_time"]).total_seconds(),
            "status_codes": dict(self.metrics["status_codes"]),
        }
        
        if response_times:
            stats["response_time"] = {
                "min": min(response_times),
                "max": max(response_times),
                "avg": sum(response_times) / len(response_times),
                "current": response_times[-1] if response_times else 0
            }
        
        # Calculate error rate
        if len(self.metrics["requests"]) > 0:
            stats["error_rate"] = len(self.metrics["errors"]) / len(self.metrics["requests"]) * 100
        else:
            stats["error_rate"] = 0
        
        # Calculate availability
        if len(self.metrics["requests"]) > 0:
            healthy_count = sum(1 for r in self.metrics["requests"] if r.get("healthy", False))
            stats["availability"] = (healthy_count / len(self.metrics["requests"])) * 100
        else:
            stats["availability"] = 0
        
        return stats
    
    def print_statistics(self):
        """Print current statistics."""
        stats = self.get_statistics()
        
        print("\n" + "=" * 60)
        print("📊 API Monitoring Statistics")
        print("=" * 60)
        print(f"Total Requests: {stats['total_requests']}")
        print(f"Total Errors: {stats['total_errors']}")
        print(f"Error Rate: {stats['error_rate']:.2f}%")
        print(f"Availability: {stats['availability']:.2f}%")
        print(f"Uptime: {stats['uptime_seconds']:.1f}s")
        
        if stats.get("response_time"):
            rt = stats["response_time"]
            print(f"\nResponse Time:")
            print(f"  Current: {rt['current']:.2f}ms")
            print(f"  Average: {rt['avg']:.2f}ms")
            print(f"  Min: {rt['min']:.2f}ms")
            print(f"  Max: {rt['max']:.2f}ms")
        
        if stats["status_codes"]:
            print(f"\nStatus Codes:")
            for code, count in stats["status_codes"].items():
                print(f"  {code}: {count}")
        
        print("=" * 60)
    
    def save_metrics(self, file_path: Path):
        """Save metrics to file."""
        data = {
            "statistics": self.get_statistics(),
            "recent_requests": list(self.metrics["requests"])[-20:],  # Last 20
            "recent_errors": list(self.metrics["errors"])[-20:],  # Last 20
            "exported_at": datetime.now().isoformat()
        }
        
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)
        
        print(f"✅ Metrics saved to {file_path}")


def interactive_monitor():
    """Interactive monitoring session."""
    monitor = APIMonitor()
    
    print("=" * 60)
    print("📊 API Monitor - Real-time Monitoring")
    print("=" * 60)
    print("Commands:")
    print("  check          - Check health once")
    print("  start          - Start continuous monitoring")
    print("  stop           - Stop monitoring")
    print("  stats          - Show statistics")
    print("  save <file>     - Save metrics to file")
    print("  quit/exit       - Exit")
    print("=" * 60)
    print()
    
    while True:
        try:
            command = input("monitor> ").strip()
            
            if not command:
                continue
            
            cmd = command.split()[0].lower()
            
            if cmd in ["quit", "exit", "q"]:
                monitor.stop_monitoring()
                print("👋 Goodbye!")
                break
            
            elif cmd == "check":
                result = monitor.check_health()
                status_emoji = "✅" if result.get("healthy") else "❌"
                print(f"{status_emoji} Health: {result.get('status')} - {result.get('response_time', 0):.2f}ms")
            
            elif cmd == "start":
                interval = float(command.split()[1]) if len(command.split()) > 1 else 1.0
                monitor.monitor_continuous(interval=interval)
                print(f"✅ Monitoring started (interval: {interval}s)")
                print("   Use 'stats' to view statistics, 'stop' to stop")
            
            elif cmd == "stop":
                monitor.stop_monitoring()
                print("🛑 Monitoring stopped")
            
            elif cmd == "stats":
                monitor.print_statistics()
            
            elif cmd == "save" and len(command.split()) > 1:
                file_path = Path(command.split()[1])
                monitor.save_metrics(file_path)
            
            else:
                print(f"❌ Unknown command: {command}")
            
            print()
        
        except KeyboardInterrupt:
            monitor.stop_monitoring()
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def main():
    """Main entry point."""
    import sys
    
    if len(sys.argv) > 1:
        monitor = APIMonitor()
        
        if sys.argv[1] == "check":
            result = monitor.check_health()
            print(json.dumps(result, indent=2))
        
        elif sys.argv[1] == "stats":
            monitor.print_statistics()
        
        elif sys.argv[1] == "monitor" and len(sys.argv) > 2:
            interval = float(sys.argv[2])
            monitor.monitor_continuous(interval=interval)
            try:
                while True:
                    time.sleep(5)
                    monitor.print_statistics()
            except KeyboardInterrupt:
                monitor.stop_monitoring()
        else:
            print("Usage:")
            print("  python api_monitor.py                    - Interactive mode")
            print("  python api_monitor.py check              - Check health once")
            print("  python api_monitor.py stats             - Show statistics")
            print("  python api_monitor.py monitor [interval] - Continuous monitoring")
    else:
        interactive_monitor()


if __name__ == "__main__":
    main()



