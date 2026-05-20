"""
MCP Prometheus Export - Exportación de métricas Prometheus
==========================================================

Exporta métricas del servidor MCP en formato Prometheus.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class PrometheusExporter:
    """Exportador de métricas en formato Prometheus"""
    
    def __init__(self, metrics: Optional[Any] = None):
        self.metrics = metrics
    
    def export(self) -> str:
        """Exportar métricas en formato Prometheus"""
        if not self.metrics:
            return "# No metrics available\n"
        
        lines = []
        stats = self.metrics.get_stats()
        
        lines.append("# HELP mcp_requests_total Total number of requests")
        lines.append("# TYPE mcp_requests_total counter")
        lines.append(f"mcp_requests_total {stats.get('total_requests', 0)}")
        lines.append("")
        
        lines.append("# HELP mcp_errors_total Total number of errors")
        lines.append("# TYPE mcp_errors_total counter")
        lines.append(f"mcp_errors_total {stats.get('total_errors', 0)}")
        lines.append("")
        
        lines.append("# HELP mcp_commands_executed_total Total commands executed")
        lines.append("# TYPE mcp_commands_executed_total counter")
        lines.append(f"mcp_commands_executed_total {stats.get('commands_executed', 0)}")
        lines.append("")
        
        lines.append("# HELP mcp_commands_failed_total Total commands failed")
        lines.append("# TYPE mcp_commands_failed_total counter")
        lines.append(f"mcp_commands_failed_total {stats.get('commands_failed', 0)}")
        lines.append("")
        
        lines.append("# HELP mcp_websocket_connections Current WebSocket connections")
        lines.append("# TYPE mcp_websocket_connections gauge")
        lines.append(f"mcp_websocket_connections {stats.get('websocket_connections', 0)}")
        lines.append("")
        
        lines.append("# HELP mcp_response_time_seconds Response time in seconds")
        lines.append("# TYPE mcp_response_time_seconds histogram")
        avg_time = stats.get('average_response_time_ms', 0) / 1000.0
        lines.append(f"mcp_response_time_seconds_bucket{{le=\"0.1\"}} {1 if avg_time < 0.1 else 0}")
        lines.append(f"mcp_response_time_seconds_bucket{{le=\"0.5\"}} {1 if avg_time < 0.5 else 0}")
        lines.append(f"mcp_response_time_seconds_bucket{{le=\"1.0\"}} {1 if avg_time < 1.0 else 0}")
        lines.append(f"mcp_response_time_seconds_bucket{{le=\"5.0\"}} {1 if avg_time < 5.0 else 0}")
        lines.append(f"mcp_response_time_seconds_bucket{{le=\"+Inf\"}} 1")
        lines.append(f"mcp_response_time_seconds_sum {avg_time * stats.get('total_requests', 0)}")
        lines.append(f"mcp_response_time_seconds_count {stats.get('total_requests', 0)}")
        lines.append("")
        
        lines.append("# HELP mcp_requests_per_second Requests per second")
        lines.append("# TYPE mcp_requests_per_second gauge")
        lines.append(f"mcp_requests_per_second {stats.get('requests_per_second', 0)}")
        lines.append("")
        
        lines.append("# HELP mcp_error_rate Error rate (0-1)")
        lines.append("# TYPE mcp_error_rate gauge")
        lines.append(f"mcp_error_rate {stats.get('error_rate', 0)}")
        lines.append("")
        
        lines.append("# HELP mcp_uptime_seconds Server uptime in seconds")
        lines.append("# TYPE mcp_uptime_seconds gauge")
        lines.append(f"mcp_uptime_seconds {stats.get('uptime_seconds', 0)}")
        lines.append("")
        
        percentiles = stats.get('response_time_percentiles', {})
        if percentiles:
            lines.append("# HELP mcp_response_time_p50 50th percentile response time (ms)")
            lines.append("# TYPE mcp_response_time_p50 gauge")
            lines.append(f"mcp_response_time_p50 {percentiles.get('p50', 0) / 1000.0}")
            lines.append("")
            
            lines.append("# HELP mcp_response_time_p95 95th percentile response time (ms)")
            lines.append("# TYPE mcp_response_time_p95 gauge")
            lines.append(f"mcp_response_time_p95 {percentiles.get('p95', 0) / 1000.0}")
            lines.append("")
            
            lines.append("# HELP mcp_response_time_p99 99th percentile response time (ms)")
            lines.append("# TYPE mcp_response_time_p99 gauge")
            lines.append(f"mcp_response_time_p99 {percentiles.get('p99', 0) / 1000.0}")
            lines.append("")
        
        error_types = stats.get('error_types', {})
        if error_types:
            lines.append("# HELP mcp_errors_by_type Errors by type")
            lines.append("# TYPE mcp_errors_by_type counter")
            for error_type, count in error_types.items():
                lines.append(f'mcp_errors_by_type{{type="{error_type}"}} {count}')
            lines.append("")
        
        endpoint_stats = stats.get('endpoint_stats', {})
        if endpoint_stats:
            lines.append("# HELP mcp_endpoint_requests_total Requests per endpoint")
            lines.append("# TYPE mcp_endpoint_requests_total counter")
            for endpoint, ep_stats in endpoint_stats.items():
                count = ep_stats.get('request_count', 0)
                lines.append(f'mcp_endpoint_requests_total{{endpoint="{endpoint}"}} {count}')
            lines.append("")
        
        return "\n".join(lines)

