from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from typing import List, Dict, Optional, Callable, Any, Union
from functools import reduce, partial, wraps
import asyncio
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime
import hashlib
import json

    import re
from typing import Any, List, Dict, Optional
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enums for type safety
class ScanType(Enum):
    PORT = "port"
    VULNERABILITY = "vulnerability"
    NETWORK = "network"
    WEB = "web"

class SecurityLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

# Data models using dataclasses
@dataclass
class ScanTarget:
    host: str
    ports: List[int] = field(default_factory=list)
    scan_type: ScanType = ScanType.PORT
    timeout: int = 30
    retries: int = 3

@dataclass
class ScanResult:
    target: str
    port: Optional[int] = None
    is_open: bool = False
    service: Optional[str] = None
    banner: Optional[str] = None
    vulnerabilities: List[str] = field(default_factory=list)
    scan_time: datetime = field(default_factory=datetime.now)
    security_level: SecurityLevel = SecurityLevel.LOW

@dataclass
class SecurityReport:
    target: str
    scan_results: List[ScanResult]
    total_vulnerabilities: int
    risk_score: float
    recommendations: List[str]
    generated_at: datetime = field(default_factory=datetime.now)

# Pure functions for data validation
def validate_target(target: Dict[str, Any]) -> Optional[str]:
    """Validate scan target data"""
    if not target.get('host'):
        return "Host is required"
    if not isinstance(target.get('ports', []), list):
        return "Ports must be a list"
    if target.get('timeout', 0) <= 0:
        return "Timeout must be positive"
    return None

def validate_scan_config(config: Dict[str, Any]) -> Optional[str]:
    """Validate scan configuration"""
    if not config.get('scan_type'):
        return "Scan type is required"
    if config.get('max_concurrent_scans', 0) <= 0:
        return "Max concurrent scans must be positive"
    return None

# Pure functions for data transformation
def extract_host_info(target: ScanTarget) -> Dict[str, Any]:
    """Extract host information from target"""
    return {
        'host': target.host,
        'port_count': len(target.ports),
        'scan_type': target.scan_type.value,
        'has_timeout': target.timeout > 0
    }

def calculate_risk_score(results: List[ScanResult]) -> float:
    """Calculate risk score based on scan results"""
    if not results:
        return 0.0
    
    risk_factors = {
        SecurityLevel.LOW: 1.0,
        SecurityLevel.MEDIUM: 2.0,
        SecurityLevel.HIGH: 3.0,
        SecurityLevel.CRITICAL: 4.0
    }
    
    total_score = sum(
        risk_factors[result.security_level] * len(result.vulnerabilities)
        for result in results
    )
    
    return min(total_score / len(results), 10.0)

def filter_critical_vulnerabilities(results: List[ScanResult]) -> List[ScanResult]:
    """Filter results with critical vulnerabilities"""
    return [
        result for result in results 
        if result.security_level in [SecurityLevel.HIGH, SecurityLevel.CRITICAL]
    ]

# Higher-order functions
def with_retry(max_retries: int = 3, delay: float = 1.0):
    """Decorator for retry logic"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        await asyncio.sleep(delay * (2 ** attempt))
                    logger.warning(f"Attempt {attempt + 1} failed: {e}")
            
            raise last_exception
        return wrapper
    return decorator

def with_logging(func: Callable) -> Callable:
    """Decorator for logging function calls"""
    @wraps(func)
    async def wrapper(*args, **kwargs) -> Any:
        start_time = datetime.now()
        logger.info(f"Starting {func.__name__} with args: {args}")
        
        try:
            result = await func(*args, **kwargs)
            duration = (datetime.now() - start_time).total_seconds()
            logger.info(f"Completed {func.__name__} in {duration:.2f}s")
            return result
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}")
            raise
    
    return wrapper

# Functional composition
def compose(*functions: Callable) -> Callable:
    """Compose multiple functions"""
    def inner(arg) -> Any:
        return reduce(lambda acc, f: f(acc), reversed(functions), arg)
    return inner

def pipe(*functions: Callable) -> Callable:
    """Pipe data through multiple functions"""
    def inner(arg) -> Any:
        return reduce(lambda acc, f: f(acc), functions, arg)
    return inner

# Async functional utilities
async def map_async(func: Callable, items: List[Any]) -> List[Any]:
    """Apply async function to all items"""
    tasks = [func(item) for item in items]
    return await asyncio.gather(*tasks)

async def filter_async(func: Callable, items: List[Any]) -> List[Any]:
    """Filter items using async function"""
    results = await map_async(func, items)
    return [item for item, result in zip(items, results) if result]

async def reduce_async(func: Callable, items: List[Any], initial: Any = None) -> Any:
    """Reduce items using async function"""
    if not items:
        return initial
    
    if initial is None:
        result = items[0]
        items = items[1:]
    else:
        result = initial
    
    for item in items:
        result = await func(result, item)
    
    return result

# Cybersecurity-specific functional utilities
def hash_sensitive_data(data: str) -> str:
    """Hash sensitive data"""
    return hashlib.sha256(data.encode()).hexdigest()

def sanitize_input(input_data: str) -> str:
    """Sanitize user input"""
    dangerous_chars = ['<', '>', '"', "'", '&', ';', '|', '`', '$', '(', ')']
    sanitized = input_data
    for char in dangerous_chars:
        sanitized = sanitized.replace(char, f'\\{char}')
    return sanitized

def validate_ip_address(ip: str) -> bool:
    """Validate IP address format"""
    pattern = r'^(\d{1,3}\.){3}\d{1,3}$'
    if not re.match(pattern, ip):
        return False
    
    parts = ip.split('.')
    return all(0 <= int(part) <= 255 for part in parts)

def validate_port_range(ports: List[int]) -> bool:
    """Validate port range"""
    return all(1 <= port <= 65535 for port in ports)

# Main functional pipeline
class SecurityScanner:
    """Functional security scanner using composition"""
    
    def __init__(self, config: Dict[str, Any]):
        
    """__init__ function."""
self.config = config
        self.validate_config()
    
    def validate_config(self) -> None:
        """Validate configuration using pure function"""
        error = validate_scan_config(self.config)
        if error:
            raise ValueError(f"Invalid config: {error}")
    
    @with_logging
    @with_retry(max_retries=3)
    async def scan_target(self, target: ScanTarget) -> ScanResult:
        """Scan a single target"""
        # Validate target first
        target_dict = {
            'host': target.host,
            'ports': target.ports,
            'timeout': target.timeout
        }
        error = validate_target(target_dict)
        if error:
            raise ValueError(f"Invalid target: {error}")
        
        # Simulate scan logic
        await asyncio.sleep(0.1)  # Simulate network delay
        
        return ScanResult(
            target=target.host,
            port=target.ports[0] if target.ports else None,
            is_open=True,  # Simulated result
            service="http",
            banner="Apache/2.4.41",
            vulnerabilities=["CVE-2021-41773"],
            security_level=SecurityLevel.HIGH
        )
    
    async def scan_multiple_targets(self, targets: List[ScanTarget]) -> List[ScanResult]:
        """Scan multiple targets concurrently"""
        # Use functional composition
        pipeline = compose(
            lambda t: self.scan_target(t),
            lambda results: [r for r in results if r.is_open],
            lambda results: sorted(results, key=lambda r: r.security_level.value, reverse=True)
        )
        
        return await map_async(pipeline, targets)
    
    def generate_report(self, results: List[ScanResult]) -> SecurityReport:
        """Generate security report using functional composition"""
        # Extract target from first result
        target = results[0].target if results else "unknown"
        
        # Calculate metrics using pure functions
        total_vulnerabilities = sum(len(r.vulnerabilities) for r in results)
        risk_score = calculate_risk_score(results)
        critical_results = filter_critical_vulnerabilities(results)
        
        # Generate recommendations
        recommendations = self.generate_recommendations(results, risk_score)
        
        return SecurityReport(
            target=target,
            scan_results=results,
            total_vulnerabilities=total_vulnerabilities,
            risk_score=risk_score,
            recommendations=recommendations
        )
    
    def generate_recommendations(self, results: List[ScanResult], risk_score: float) -> List[str]:
        """Generate security recommendations"""
        recommendations = []
        
        if risk_score > 7.0:
            recommendations.append("Immediate action required - critical vulnerabilities detected")
        
        if any(r.security_level == SecurityLevel.CRITICAL for r in results):
            recommendations.append("Patch critical vulnerabilities immediately")
        
        if len(results) > 10:
            recommendations.append("Consider implementing network segmentation")
        
        return recommendations

# Functional API handlers
async async def handle_scan_request(request_data: Dict[str, Any]) -> Dict[str, Any]:
    """Handle scan request using functional approach"""
    # Validate input
    if not request_data.get('targets'):
        return {'error': 'No targets provided'}
    
    # Transform input to domain objects
    targets = [
        ScanTarget(
            host=target['host'],
            ports=target.get('ports', [80, 443]),
            scan_type=ScanType(target.get('scan_type', 'port')),
            timeout=target.get('timeout', 30)
        )
        for target in request_data['targets']
    ]
    
    # Create scanner and execute scan
    config = request_data.get('config', {'max_concurrent_scans': 10})
    scanner = SecurityScanner(config)
    
    try:
        results = await scanner.scan_multiple_targets(targets)
        report = scanner.generate_report(results)
        
        # Transform to response format
        return {
            'success': True,
            'report': {
                'target': report.target,
                'total_vulnerabilities': report.total_vulnerabilities,
                'risk_score': report.risk_score,
                'recommendations': report.recommendations,
                'scan_results': [
                    {
                        'port': r.port,
                        'service': r.service,
                        'vulnerabilities': r.vulnerabilities,
                        'security_level': r.security_level.value
                    }
                    for r in report.scan_results
                ]
            }
        }
    except Exception as e:
        logger.error(f"Scan failed: {e}")
        return {'error': str(e)}

# Example usage
async def main():
    """Example usage of functional security scanner"""
    # Sample scan request
    request_data = {
        'targets': [
            {
                'host': '192.168.1.1',
                'ports': [80, 443, 22],
                'scan_type': 'port',
                'timeout': 30
            },
            {
                'host': '192.168.1.2',
                'ports': [80, 443],
                'scan_type': 'vulnerability',
                'timeout': 45
            }
        ],
        'config': {
            'max_concurrent_scans': 5,
            'scan_type': 'comprehensive'
        }
    }
    
    # Execute scan using functional pipeline
    result = await handle_scan_request(request_data)
    
    # Output results
    print(json.dumps(result, indent=2, default=str))

match __name__:
    case "__main__":
    asyncio.run(main()) 