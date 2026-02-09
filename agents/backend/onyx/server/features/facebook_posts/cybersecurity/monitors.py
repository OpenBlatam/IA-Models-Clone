from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
BUFFER_SIZE = 1024

import psutil
import asyncio
import time
import json
import os
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import aiofiles
from datetime import datetime, timedelta
import threading
import queue
from typing import Any, List, Dict, Optional
import logging
"""
System monitoring utilities with proper async/def distinction.
Async for I/O operations, def for CPU-bound analysis.
"""


@dataclass
class MonitoringConfig:
    """Configuration for monitoring operations."""
    interval: float = 1.0
    log_file: str = "security_events.log"
    max_log_size: int = 10 * 1024 * 1024  # 10MB
    alert_thresholds: Dict[str, float] = None
    enable_file_monitoring: bool = True
    enable_network_monitoring: bool = True
    enable_process_monitoring: bool = True

@dataclass
class SystemMetrics:
    """System resource metrics."""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    disk_usage_percent: float
    network_bytes_sent: int
    network_bytes_recv: int
    active_connections: int
    running_processes: int

@dataclass
class SecurityEvent:
    """Security event record."""
    timestamp: datetime
    event_type: str
    severity: str
    description: str
    source: str
    details: Dict[str, Any]

async def monitor_system_resources(config: MonitoringConfig) -> SystemMetrics:
    """Monitor system resources asynchronously."""
    try:
        # Get CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # Get memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        # Get disk usage
        disk = psutil.disk_usage('/')
        disk_usage_percent = disk.percent
        
        # Get network statistics
        network = psutil.net_io_counters()
        network_bytes_sent = network.bytes_sent
        network_bytes_recv = network.bytes_recv
        
        # Get active connections
        connections = psutil.net_connections()
        active_connections = len([conn for conn in connections if conn.status == 'ESTABLISHED'])
        
        # Get running processes
        running_processes = len(psutil.pids())
        
        return SystemMetrics(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            disk_usage_percent=disk_usage_percent,
            network_bytes_sent=network_bytes_sent,
            network_bytes_recv=network_bytes_recv,
            active_connections=active_connections,
            running_processes=running_processes
        )
        
    except Exception as e:
        # Return default metrics on error
        return SystemMetrics(
            timestamp=time.time(),
            cpu_percent=0.0,
            memory_percent=0.0,
            disk_usage_percent=0.0,
            network_bytes_sent=0,
            network_bytes_recv=0,
            active_connections=0,
            running_processes=0
        )

def detect_anomalies(metrics: List[SystemMetrics], config: MonitoringConfig) -> List[Dict[str, Any]]:
    """Detect anomalies in system metrics."""
    if len(metrics) < 2:
        return []
    
    anomalies = []
    thresholds = config.alert_thresholds or {
        'cpu_percent': 80.0,
        'memory_percent': 85.0,
        'disk_usage_percent': 90.0,
        'active_connections': 1000,
        'network_bytes_sent': 100 * 1024 * 1024,  # 100MB
        'network_bytes_recv': 100 * 1024 * 1024   # 100MB
    }
    
    # Get latest metrics
    latest = metrics[-1]
    previous = metrics[-2] if len(metrics) > 1 else latest
    
    # Check CPU usage
    if latest.cpu_percent > thresholds['cpu_percent']:
        anomalies.append({
            'type': 'high_cpu_usage',
            'severity': 'warning',
            'current_value': latest.cpu_percent,
            'threshold': thresholds['cpu_percent'],
            'description': f"CPU usage is {latest.cpu_percent:.1f}% (threshold: {thresholds['cpu_percent']}%)"
        })
    
    # Check memory usage
    if latest.memory_percent > thresholds['memory_percent']:
        anomalies.append({
            'type': 'high_memory_usage',
            'severity': 'warning',
            'current_value': latest.memory_percent,
            'threshold': thresholds['memory_percent'],
            'description': f"Memory usage is {latest.memory_percent:.1f}% (threshold: {thresholds['memory_percent']}%)"
        })
    
    # Check disk usage
    if latest.disk_usage_percent > thresholds['disk_usage_percent']:
        anomalies.append({
            'type': 'high_disk_usage',
            'severity': 'critical',
            'current_value': latest.disk_usage_percent,
            'threshold': thresholds['disk_usage_percent'],
            'description': f"Disk usage is {latest.disk_usage_percent:.1f}% (threshold: {thresholds['disk_usage_percent']}%)"
        })
    
    # Check for sudden spikes
    cpu_spike = abs(latest.cpu_percent - previous.cpu_percent) > 20
    memory_spike = abs(latest.memory_percent - previous.memory_percent) > 10
    
    if cpu_spike:
        anomalies.append({
            'type': 'cpu_spike',
            'severity': 'info',
            'current_value': latest.cpu_percent,
            'previous_value': previous.cpu_percent,
            'description': f"CPU usage spiked from {previous.cpu_percent:.1f}% to {latest.cpu_percent:.1f}%"
        })
    
    if memory_spike:
        anomalies.append({
            'type': 'memory_spike',
            'severity': 'info',
            'current_value': latest.memory_percent,
            'previous_value': previous.memory_percent,
            'description': f"Memory usage spiked from {previous.memory_percent:.1f}% to {latest.memory_percent:.1f}%"
        })
    
    return anomalies

async def log_security_events(events: List[SecurityEvent], config: MonitoringConfig):
    """Log security events to file asynchronously."""
    try:
        async with aiofiles.open(config.log_file, 'a', encoding='utf-8') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            for event in events:
                log_entry = {
                    'timestamp': event.timestamp.isoformat(),
                    'event_type': event.event_type,
                    'severity': event.severity,
                    'description': event.description,
                    'source': event.source,
                    'details': event.details
                }
                
                await f.write(json.dumps(log_entry) + '\n')
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        
        # Check log file size and rotate if necessary
        await rotate_log_file(config.log_file, config.max_log_size)
        
    except Exception as e:
        print(f"Error logging security events: {e}")

async def rotate_log_file(log_file: str, max_size: int):
    """Rotate log file if it exceeds maximum size."""
    try:
        if os.path.exists(log_file):
            file_size = os.path.getsize(log_file)
            
            if file_size > max_size:
                # Create backup file
                backup_file = f"{log_file}.{int(time.time())}"
                async with aiofiles.open(log_file, 'r') as src:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                    async with aiofiles.open(backup_file, 'w') as dst:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                        await dst.write(await src.read())
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                
                # Clear original file
                async with aiofiles.open(log_file, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                    await f.write('')
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                    
    except Exception as e:
        print(f"Error rotating log file: {e}")

async def track_user_activity(user_id: str, action: str, details: Dict[str, Any], 
                            config: MonitoringConfig) -> SecurityEvent:
    """Track user activity and detect suspicious behavior."""
    timestamp = datetime.now()
    
    # Analyze activity for suspicious patterns
    suspicious_patterns = detect_suspicious_activity(user_id, action, details)
    
    # Determine severity
    severity = 'info'
    if suspicious_patterns:
        severity = 'warning' if len(suspicious_patterns) < 3 else 'critical'
    
    event = SecurityEvent(
        timestamp=timestamp,
        event_type='user_activity',
        severity=severity,
        description=f"User {user_id} performed {action}",
        source='user_tracking',
        details={
            'user_id': user_id,
            'action': action,
            'suspicious_patterns': suspicious_patterns,
            'additional_details': details
        }
    )
    
    # Log the event
    await log_security_events([event], config)
    
    return event

def detect_suspicious_activity(user_id: str, action: str, details: Dict[str, Any]) -> List[str]:
    """Detect suspicious user activity patterns."""
    suspicious_patterns = []
    
    # Check for rapid actions
    if 'timestamp' in details:
        # This would typically check against user's action history
        # For demo purposes, we'll use simple heuristics
        pass
    
    # Check for unusual actions
    unusual_actions = ['delete_all', 'export_data', 'change_permissions', 'login_from_new_location']
    if action in unusual_actions:
        suspicious_patterns.append(f"Unusual action: {action}")
    
    # Check for bulk operations
    if 'count' in details and details['count'] > 100:
        suspicious_patterns.append(f"Bulk operation: {details['count']} items")
    
    # Check for sensitive data access
    sensitive_patterns = ['password', 'credit_card', 'ssn', 'private']
    for pattern in sensitive_patterns:
        if pattern in action.lower() or any(pattern in str(v).lower() for v in details.values()):
            suspicious_patterns.append(f"Sensitive data access: {pattern}")
    
    return suspicious_patterns

async def monitor_file_changes(directory: str, config: MonitoringConfig) -> List[Dict[str, Any]]:
    """Monitor file system changes asynchronously."""
    if not config.enable_file_monitoring:
        return []
    
    changes = []
    
    try:
        # Get current file list
        current_files = {}
        for root, dirs, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    stat = os.stat(file_path)
                    current_files[file_path] = {
                        'size': stat.st_size,
                        'mtime': stat.st_mtime,
                        'ctime': stat.st_ctime
                    }
                except OSError:
                    continue
        
        # Compare with previous state (this would typically be stored)
        # For demo purposes, we'll return empty list
        return changes
        
    except Exception as e:
        return [{'error': str(e)}]

async def monitor_network_connections(config: MonitoringConfig) -> List[Dict[str, Any]]:
    """Monitor network connections asynchronously."""
    if not config.enable_network_monitoring:
        return []
    
    try:
        connections = psutil.net_connections()
        suspicious_connections = []
        
        for conn in connections:
            # Check for suspicious patterns
            if conn.status == 'ESTABLISHED':
                # Check for connections to known malicious IPs (demo)
                suspicious_ips = ['192.168.1.100', '10.0.0.50']  # Example
                
                if conn.raddr and conn.raddr.ip in suspicious_ips:
                    suspicious_connections.append({
                        'type': 'suspicious_ip',
                        'local_address': f"{conn.laddr.ip}:{conn.laddr.port}",
                        'remote_address': f"{conn.raddr.ip}:{conn.raddr.port}",
                        'status': conn.status,
                        'pid': conn.pid
                    })
                
                # Check for unusual ports
                unusual_ports = [22, 23, 3389, 5900]  # SSH, Telnet, RDP, VNC
                if conn.raddr and conn.raddr.port in unusual_ports:
                    suspicious_connections.append({
                        'type': 'unusual_port',
                        'local_address': f"{conn.laddr.ip}:{conn.laddr.port}",
                        'remote_address': f"{conn.raddr.ip}:{conn.raddr.port}",
                        'port': conn.raddr.port,
                        'status': conn.status,
                        'pid': conn.pid
                    })
        
        return suspicious_connections
        
    except Exception as e:
        return [{'error': str(e)}]

def analyze_process_behavior(processes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Analyze process behavior for security issues."""
    suspicious_processes = []
    
    for process in processes:
        issues = []
        
        # Check for high resource usage
        if process.get('cpu_percent', 0) > 80:
            issues.append('High CPU usage')
        
        if process.get('memory_percent', 0) > 50:
            issues.append('High memory usage')
        
        # Check for suspicious process names
        suspicious_names = ['crypto', 'miner', 'botnet', 'keylogger', 'backdoor']
        process_name = process.get('name', '').lower()
        
        for suspicious_name in suspicious_names:
            if suspicious_name in process_name:
                issues.append(f'Suspicious process name: {suspicious_name}')
        
        # Check for unusual parent-child relationships
        if process.get('parent_pid') and process.get('parent_pid') != 1:
            # This would typically check against known process trees
            pass
        
        if issues:
            suspicious_processes.append({
                'pid': process.get('pid'),
                'name': process.get('name'),
                'issues': issues,
                'cpu_percent': process.get('cpu_percent'),
                'memory_percent': process.get('memory_percent')
            })
    
    return suspicious_processes

# Named exports for main functionality
__all__ = [
    'monitor_system_resources',
    'detect_anomalies',
    'log_security_events',
    'track_user_activity',
    'monitor_file_changes',
    'monitor_network_connections',
    'analyze_process_behavior',
    'MonitoringConfig',
    'SystemMetrics',
    'SecurityEvent'
] 