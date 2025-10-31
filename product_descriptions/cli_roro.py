from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import sys
import json
import yaml
import logging
from typing import Optional
import typer
from pathlib import Path
from roro_models import ScanRequestObj, ScanResponseObj, ScanStatus
from rate_limiting_network_scans import NetworkScanner, RateLimitConfig, RateLimitType, ScanType, BackoffStrategy, NetworkTarget
from decorators.centralized import centralized_logging_metrics_exception
from datetime import datetime

    import asyncio
from typing import Any, List, Dict, Optional
app = typer.Typer()

logging.basicConfig(level=logging.INFO)

@centralized_logging_metrics_exception
def run_scan_from_obj(scan_request: ScanRequestObj) -> ScanResponseObj:
    config = RateLimitConfig()
    scanner = NetworkScanner(config, RateLimitType[scan_request.rate_limit_type.upper()])
    for target in scan_request.targets:
        asyncio.run(scanner.add_target(NetworkTarget(host=target, scan_type=scan_request.scan_type)))
    asyncio.run(scanner.run_scan(scan_request.max_concurrent))
    results = [
        {
            "target": r.target.host,
            "success": r.success,
            "response_time": r.response_time,
            "data": r.data,
            "error": r.error_message,
            "retry_count": r.retry_count
        } for r in scanner.results
    ]
    stats = scanner.get_scan_stats()
    return ScanResponseObj(
        scan_id=f"scan_{int(datetime.utcnow().timestamp())}",
        status=ScanStatus.COMPLETED,
        results=results,
        stats=stats,
        timestamp=datetime.utcnow()
    )

@app.command()
def scan(
    input_file: Optional[Path] = typer.Option(None, help="Input JSON/YAML file for scan request. If omitted, reads from stdin."),
    output_file: Optional[Path] = typer.Option(None, help="Output JSON/YAML file for scan response. If omitted, writes to stdout."),
    format: str = typer.Option("json", help="Output format: json or yaml.")
):
    """Run a network scan using the RORO pattern (CLI interface)."""
    if input_file:
        with open(input_file, "r") as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            if input_file.suffix in [".yaml", ".yml"]:
                data = yaml.safe_load(f)
            else:
                data = json.load(f)
    else:
        data = json.load(sys.stdin)
    scan_request = ScanRequestObj(**data)
    response = run_scan_from_obj(scan_request)
    output = json.dumps(response.__dict__, default=str, indent=2) if format == "json" else yaml.safe_dump(response.__dict__, default_flow_style=False)
    if output_file:
        with open(output_file, "w") as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write(output)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
    else:
        print(output)

match __name__:
    case "__main__":
    app() 