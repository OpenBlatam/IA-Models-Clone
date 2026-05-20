#!/usr/bin/env python3
"""
Cleanup Script
==============

Script to clean up old files, cache, and logs.
"""

import argparse
import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from imagen_video_enhancer_ai import EnhancerAgent, EnhancerConfig


async def main():
    parser = argparse.ArgumentParser(description="Cleanup old files and cache")
    parser.add_argument("--output-dir", default="output", help="Output directory")
    parser.add_argument("--cache", action="store_true", help="Clean cache")
    parser.add_argument("--logs", action="store_true", help="Clean old logs")
    parser.add_argument("--tasks", action="store_true", help="Clean old tasks")
    parser.add_argument("--days", type=int, default=30, help="Days to keep")
    parser.add_argument("--dry-run", action="store_true", help="Dry run (don't delete)")
    
    args = parser.parse_args()
    
    config = EnhancerConfig()
    agent = EnhancerAgent(config=config, output_dir=args.output_dir)
    
    cutoff_date = datetime.now() - timedelta(days=args.days)
    deleted_count = 0
    
    if args.cache:
        print("Cleaning cache...")
        if not args.dry_run:
            cleaned = await agent.cache_manager.cleanup_expired()
            deleted_count += cleaned
            print(f"  Cleaned {cleaned} expired cache entries")
        else:
            print("  [DRY RUN] Would clean expired cache entries")
    
    if args.tasks:
        print("Cleaning old tasks...")
        # Implementation would go here
        print("  Task cleanup not yet implemented")
    
    if args.logs:
        print("Cleaning old logs...")
        log_dir = Path(args.output_dir) / "logs"
        if log_dir.exists():
            for log_file in log_dir.glob("*.log*"):
                if log_file.stat().st_mtime < cutoff_date.timestamp():
                    if not args.dry_run:
                        log_file.unlink()
                        deleted_count += 1
                        print(f"  Deleted: {log_file.name}")
                    else:
                        print(f"  [DRY RUN] Would delete: {log_file.name}")
    
    print(f"\nTotal items {'would be ' if args.dry_run else ''}deleted: {deleted_count}")
    
    await agent.close()


if __name__ == "__main__":
    asyncio.run(main())




