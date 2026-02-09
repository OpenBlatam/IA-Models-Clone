#!/usr/bin/env python3
"""
API Utilities
=============
Collection of utility functions for API operations.

⚠️ DEPRECATED: This file is deprecated. Use `tools.utils` instead.

For new code, use:
    from tools.utils import *
    # Utility functions are available in tools.utils
"""
import warnings

warnings.warn(
    "api_utils.py is deprecated. Use 'tools.utils' instead.",
    DeprecationWarning,
    stacklevel=2
)

import json
import requests
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime


def format_response_time(ms: float) -> str:
    """Format response time in human-readable format."""
    if ms < 1:
        return f"{ms*1000:.2f}μs"
    elif ms < 1000:
        return f"{ms:.2f}ms"
    else:
        return f"{ms/1000:.2f}s"


def format_bytes(bytes: int) -> str:
    """Format bytes in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes < 1024.0:
            return f"{bytes:.2f} {unit}"
        bytes /= 1024.0
    return f"{bytes:.2f} TB"


def validate_json_file(file_path: Path) -> bool:
    """Validate JSON file."""
    try:
        with open(file_path, "r") as f:
            json.load(f)
        return True
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON: {e}")
        return False
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return False


def merge_json_files(files: List[Path], output_file: Path):
    """Merge multiple JSON files."""
    merged_data = {
        "merged_at": datetime.now().isoformat(),
        "sources": [],
        "data": {}
    }
    
    for file_path in files:
        if not file_path.exists():
            print(f"⚠️  File not found: {file_path}")
            continue
        
        with open(file_path, "r") as f:
            data = json.load(f)
        
        merged_data["sources"].append(str(file_path))
        merged_data["data"][file_path.stem] = data
    
    with open(output_file, "w") as f:
        json.dump(merged_data, f, indent=2)
    
    print(f"✅ Merged {len(files)} files into {output_file}")


def clean_old_files(directory: Path, pattern: str = "*.json", days: int = 7):
    """Clean old files in directory."""
    from datetime import timedelta
    
    cutoff_date = datetime.now() - timedelta(days=days)
    deleted = 0
    
    for file_path in directory.glob(pattern):
        if file_path.is_file():
            file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
            if file_time < cutoff_date:
                file_path.unlink()
                deleted += 1
    
    print(f"✅ Deleted {deleted} old files (older than {days} days)")


def export_to_csv(json_file: Path, csv_file: Path):
    """Export JSON data to CSV."""
    import csv
    
    with open(json_file, "r") as f:
        data = json.load(f)
    
    # Simple CSV export (would need to be customized based on data structure)
    if isinstance(data, list) and len(data) > 0:
        with open(csv_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)
        
        print(f"✅ Exported to CSV: {csv_file}")
    else:
        print("❌ Data structure not suitable for CSV export")


def check_api_connectivity(base_url: str, timeout: int = 5) -> bool:
    """Check if API is reachable."""
    try:
        response = requests.get(f"{base_url}/health", timeout=timeout)
        return response.status_code == 200
    except:
        return False


def get_api_info(base_url: str) -> Dict[str, Any]:
    """Get API information."""
    info = {
        "base_url": base_url,
        "reachable": False,
        "health_status": "unknown",
        "version": None,
        "timestamp": datetime.now().isoformat()
    }
    
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        info["reachable"] = True
        
        if response.status_code == 200:
            data = response.json()
            info["health_status"] = data.get("status", "unknown")
            info["version"] = data.get("version")
    except:
        pass
    
    return info


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="API Utilities")
    parser.add_argument("--validate", help="Validate JSON file")
    parser.add_argument("--merge", nargs="+", help="Merge JSON files")
    parser.add_argument("--output", help="Output file")
    parser.add_argument("--clean", help="Clean old files in directory")
    parser.add_argument("--days", type=int, default=7, help="Days threshold for cleaning")
    parser.add_argument("--export-csv", help="Export JSON to CSV")
    parser.add_argument("--check", help="Check API connectivity")
    parser.add_argument("--info", help="Get API information")
    
    args = parser.parse_args()
    
    if args.validate:
        validate_json_file(Path(args.validate))
    
    if args.merge and args.output:
        merge_json_files([Path(f) for f in args.merge], Path(args.output))
    
    if args.clean:
        clean_old_files(Path(args.clean), days=args.days)
    
    if args.export_csv:
        json_file = Path(args.export_csv)
        csv_file = json_file.with_suffix(".csv")
        export_to_csv(json_file, csv_file)
    
    if args.check:
        if check_api_connectivity(args.check):
            print(f"✅ API is reachable: {args.check}")
        else:
            print(f"❌ API is not reachable: {args.check}")
    
    if args.info:
        info = get_api_info(args.info)
        print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()



