#!/usr/bin/env python3
"""
API Logger
==========
Advanced logging tool for API requests and responses.

⚠️ DEPRECATED: This file is deprecated. Consider migrating to the new tools structure.

For new code, use:
    from tools.manager import ToolManager
    manager = ToolManager()
    # Tools can be extended with logging capabilities
"""
import warnings

warnings.warn(
    "api_logger.py is deprecated. Consider migrating to the new tools structure in tools/.",
    DeprecationWarning,
    stacklevel=2
)

import json
import requests
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
import time


@dataclass
class LogEntry:
    """Log entry."""
    timestamp: str
    method: str
    url: str
    status_code: int
    response_time: float
    request_headers: Dict[str, str]
    response_headers: Dict[str, str]
    request_body: Optional[Any] = None
    response_body: Optional[Any] = None
    error: Optional[str] = None


class APILogger:
    """Advanced API logger."""
    
    def __init__(self, base_url: str = "http://localhost:8000", log_file: Optional[Path] = None):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json"
        })
        self.logs: List[LogEntry] = []
        self.log_file = log_file
        self.auto_save = log_file is not None
    
    def log_request(
        self,
        method: str,
        endpoint: str,
        **kwargs
    ) -> LogEntry:
        """Make request and log it."""
        start_time = time.time()
        url = f"{self.base_url}{endpoint}"
        
        request_headers = dict(self.session.headers)
        request_headers.update(kwargs.get("headers", {}))
        
        request_body = kwargs.get("json") or kwargs.get("data")
        
        try:
            if method.upper() == "GET":
                response = self.session.get(url, **kwargs)
            elif method.upper() == "POST":
                response = self.session.post(url, **kwargs)
            elif method.upper() == "PUT":
                response = self.session.put(url, **kwargs)
            elif method.upper() == "DELETE":
                response = self.session.delete(url, **kwargs)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            response_time = (time.time() - start_time) * 1000
            
            # Try to parse response
            try:
                response_body = response.json()
            except:
                response_body = response.text[:1000]  # First 1000 chars
            
            entry = LogEntry(
                timestamp=datetime.now().isoformat(),
                method=method.upper(),
                url=url,
                status_code=response.status_code,
                response_time=response_time,
                request_headers=request_headers,
                response_headers=dict(response.headers),
                request_body=request_body,
                response_body=response_body
            )
            
            self.logs.append(entry)
            
            if self.auto_save:
                self.save_logs()
            
            return entry
        
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            
            entry = LogEntry(
                timestamp=datetime.now().isoformat(),
                method=method.upper(),
                url=url,
                status_code=0,
                response_time=response_time,
                request_headers=request_headers,
                response_headers={},
                request_body=request_body,
                error=str(e)
            )
            
            self.logs.append(entry)
            
            if self.auto_save:
                self.save_logs()
            
            return entry
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get logging statistics."""
        if not self.logs:
            return {"total": 0}
        
        status_codes = {}
        methods = {}
        total_time = 0
        errors = 0
        
        for log in self.logs:
            status_codes[log.status_code] = status_codes.get(log.status_code, 0) + 1
            methods[log.method] = methods.get(log.method, 0) + 1
            total_time += log.response_time
            if log.error:
                errors += 1
        
        avg_time = total_time / len(self.logs) if self.logs else 0
        
        return {
            "total": len(self.logs),
            "errors": errors,
            "status_codes": status_codes,
            "methods": methods,
            "avg_response_time": avg_time,
            "total_time": total_time
        }
    
    def print_statistics(self):
        """Print logging statistics."""
        stats = self.get_statistics()
        
        print("\n" + "=" * 60)
        print("📊 Logging Statistics")
        print("=" * 60)
        print(f"Total Requests: {stats['total']}")
        print(f"Errors: {stats['errors']}")
        print(f"Average Response Time: {stats['avg_response_time']:.2f}ms")
        print(f"Total Time: {stats['total_time']:.2f}ms")
        
        if stats.get("status_codes"):
            print("\nStatus Codes:")
            for code, count in sorted(stats["status_codes"].items()):
                print(f"  {code}: {count}")
        
        if stats.get("methods"):
            print("\nMethods:")
            for method, count in sorted(stats["methods"].items()):
                print(f"  {method}: {count}")
        
        print("=" * 60)
    
    def save_logs(self, file_path: Optional[Path] = None):
        """Save logs to file."""
        path = file_path or self.log_file
        if not path:
            raise ValueError("No log file specified")
        
        data = {
            "logs": [asdict(log) for log in self.logs],
            "statistics": self.get_statistics(),
            "exported_at": datetime.now().isoformat()
        }
        
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        
        print(f"✅ Logs saved to {path}")
    
    def print_recent_logs(self, count: int = 10):
        """Print recent logs."""
        recent = self.logs[-count:] if len(self.logs) > count else self.logs
        
        print("\n" + "=" * 60)
        print(f"📋 Recent Logs (last {len(recent)})")
        print("=" * 60)
        
        for log in recent:
            status_emoji = "✅" if 200 <= log.status_code < 300 else "❌" if log.status_code >= 400 else "⚠️"
            print(f"\n{status_emoji} {log.method} {log.url}")
            print(f"   Status: {log.status_code}")
            print(f"   Time: {log.response_time:.2f}ms")
            print(f"   Timestamp: {log.timestamp}")
            if log.error:
                print(f"   Error: {log.error}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="API Logger")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--log-file", help="Log file path")
    parser.add_argument("--stats", action="store_true", help="Show statistics")
    parser.add_argument("--recent", type=int, help="Show recent logs")
    parser.add_argument("--export", help="Export logs to file")
    
    args = parser.parse_args()
    
    logger = APILogger(
        base_url=args.url,
        log_file=Path(args.log_file) if args.log_file else None
    )
    
    # Example: log some requests
    if not args.stats and not args.recent and not args.export:
        print("📝 API Logger - Example Usage")
        print("=" * 60)
        print("Making sample requests...")
        
        logger.log_request("GET", "/health")
        logger.log_request("GET", "/")
        logger.log_request("GET", "/docs")
        
        print("✅ Sample requests logged")
    
    if args.stats:
        logger.print_statistics()
    
    if args.recent:
        logger.print_recent_logs(count=args.recent)
    
    if args.export:
        logger.save_logs(Path(args.export))


if __name__ == "__main__":
    main()



