#!/usr/bin/env python3
"""
Export Statistics Script
========================

Script to export statistics and metrics.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from imagen_video_enhancer_ai import EnhancerAgent, EnhancerConfig


async def main():
    parser = argparse.ArgumentParser(description="Export statistics and metrics")
    parser.add_argument("--output-dir", default="output", help="Output directory")
    parser.add_argument("--output-file", help="Output file (defaults to stats_YYYYMMDD_HHMMSS.json)")
    parser.add_argument("--format", choices=["json", "csv"], default="json", help="Output format")
    parser.add_argument("--hours", type=int, help="Hours of history to include")
    
    args = parser.parse_args()
    
    config = EnhancerConfig()
    agent = EnhancerAgent(config=config, output_dir=args.output_dir)
    
    # Get stats
    stats = agent.get_stats()
    
    # Get metrics if requested
    if args.hours:
        start_time = datetime.now() - timedelta(hours=args.hours)
        metrics_data = {}
        for metric_name in agent.metrics_collector.get_all_metrics():
            points = agent.metrics_collector.get_metric(metric_name, start_time=start_time)
            metrics_data[metric_name] = [p.to_dict() for p in points]
        stats["metrics_history"] = metrics_data
    
    # Determine output file
    if args.output_file:
        output_file = Path(args.output_file)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = Path(args.output_dir) / f"stats_{timestamp}.{args.format}"
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Export
    if args.format == "json":
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False, default=str)
    elif args.format == "csv":
        import csv
        # Flatten stats for CSV
        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["key", "value"])
            for key, value in stats.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        writer.writerow([f"{key}.{sub_key}", sub_value])
                else:
                    writer.writerow([key, value])
    
    print(f"Statistics exported to: {output_file}")
    
    await agent.close()


if __name__ == "__main__":
    asyncio.run(main())




