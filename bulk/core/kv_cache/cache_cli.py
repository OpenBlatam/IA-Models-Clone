"""
Cache CLI interface.

Provides command-line interface for cache operations.
"""
from __future__ import annotations

import logging
import argparse
import json
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class CacheCLI:
    """
    Cache CLI interface.
    
    Provides command-line interface.
    """
    
    def __init__(self, cache: Any):
        """
        Initialize CLI.
        
        Args:
            cache: Cache instance
        """
        self.cache = cache
        self.parser = argparse.ArgumentParser(description="KV Cache CLI")
        self._setup_commands()
    
    def _setup_commands(self) -> None:
        """Setup CLI commands."""
        subparsers = self.parser.add_subparsers(dest="command", help="Commands")
        
        # Get command
        get_parser = subparsers.add_parser("get", help="Get value from cache")
        get_parser.add_argument("position", type=int, help="Cache position")
        get_parser.add_argument("--json", action="store_true", help="Output as JSON")
        
        # Put command
        put_parser = subparsers.add_parser("put", help="Put value in cache")
        put_parser.add_argument("position", type=int, help="Cache position")
        put_parser.add_argument("value", help="Value to cache")
        put_parser.add_argument("--json", action="store_true", help="Output as JSON")
        
        # Stats command
        stats_parser = subparsers.add_parser("stats", help="Get cache statistics")
        stats_parser.add_argument("--json", action="store_true", help="Output as JSON")
        
        # Clear command
        clear_parser = subparsers.add_parser("clear", help="Clear cache")
        clear_parser.add_argument("--json", action="store_true", help="Output as JSON")
        
        # Health command
        health_parser = subparsers.add_parser("health", help="Health check")
        health_parser.add_argument("--json", action="store_true", help="Output as JSON")
    
    def execute(self, args: Optional[list] = None) -> str:
        """
        Execute CLI command.
        
        Args:
            args: Optional command arguments
            
        Returns:
            Command output
        """
        parsed_args = self.parser.parse_args(args)
        
        if not parsed_args.command:
            self.parser.print_help()
            return ""
        
        output_json = parsed_args.json
        
        if parsed_args.command == "get":
            return self._cmd_get(parsed_args.position, output_json)
        elif parsed_args.command == "put":
            return self._cmd_put(parsed_args.position, parsed_args.value, output_json)
        elif parsed_args.command == "stats":
            return self._cmd_stats(output_json)
        elif parsed_args.command == "clear":
            return self._cmd_clear(output_json)
        elif parsed_args.command == "health":
            return self._cmd_health(output_json)
        
        return ""
    
    def _cmd_get(self, position: int, output_json: bool) -> str:
        """Execute get command."""
        value = self.cache.get(position)
        
        if output_json:
            return json.dumps({
                "position": position,
                "value": str(value) if value is not None else None,
                "found": value is not None
            })
        
        if value is None:
            return f"Cache miss at position {position}"
        
        return f"Position {position}: {value}"
    
    def _cmd_put(self, position: int, value: str, output_json: bool) -> str:
        """Execute put command."""
        try:
            self.cache.put(position, value)
            
            if output_json:
                return json.dumps({
                    "success": True,
                    "position": position,
                    "message": "Value cached"
                })
            
            return f"Cached value at position {position}"
        except Exception as e:
            if output_json:
                return json.dumps({
                    "success": False,
                    "error": str(e)
                })
            return f"Error: {e}"
    
    def _cmd_stats(self, output_json: bool) -> str:
        """Execute stats command."""
        stats = self.cache.get_stats()
        
        if output_json:
            return json.dumps(stats, indent=2)
        
        lines = ["Cache Statistics:"]
        for key, value in stats.items():
            lines.append(f"  {key}: {value}")
        
        return "\n".join(lines)
    
    def _cmd_clear(self, output_json: bool) -> str:
        """Execute clear command."""
        try:
            self.cache.clear()
            
            if output_json:
                return json.dumps({
                    "success": True,
                    "message": "Cache cleared"
                })
            
            return "Cache cleared"
        except Exception as e:
            if output_json:
                return json.dumps({
                    "success": False,
                    "error": str(e)
                })
            return f"Error: {e}"
    
    def _cmd_health(self, output_json: bool) -> str:
        """Execute health command."""
        try:
            stats = self.cache.get_stats()
            healthy = stats.get("cache_size", 0) >= 0
            
            if output_json:
                return json.dumps({
                    "status": "healthy" if healthy else "unhealthy",
                    "cache_size": stats.get("cache_size", 0),
                    "memory_mb": stats.get("memory_mb", 0.0)
                })
            
            status = "healthy" if healthy else "unhealthy"
            return f"Cache status: {status}"
        except Exception as e:
            if output_json:
                return json.dumps({
                    "status": "unhealthy",
                    "error": str(e)
                })
            return f"Error: {e}"


def main():
    """CLI entry point."""
    # In production: would initialize cache from config
    from kv_cache import BaseKVCache, KVCacheConfig
    
    config = KVCacheConfig()
    cache = BaseKVCache(config)
    
    cli = CacheCLI(cache)
    output = cli.execute()
    print(output)


if __name__ == "__main__":
    main()

