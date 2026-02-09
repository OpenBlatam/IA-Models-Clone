from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from urllib.parse import urlparse
import aiohttp
import numpy as np
import structlog
from fastapi import BackgroundTasks, Depends, HTTPException, status
from pydantic import BaseModel, Field, validator
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Any, List, Dict, Optional
"""
Enhanced Optimized Scan Engine with ML-based False Positive Detection
Production-ready cybersecurity scanning with advanced features
"""



# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


class ScanStatus(str, Enum):
    """Scan status enumeration"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ScanPriority(str, Enum):
    """Scan priority levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SecurityLevel(str, Enum):
    """Security level enumeration"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ScanType(str, Enum):
    """Scan type enumeration"""
    VULNERABILITY = "vulnerability"
    PENETRATION = "penetration"
    COMPLIANCE = "compliance"
    MALWARE = "malware"
    NETWORK = "network"


@dataclass
class ScanTarget:
    """Scan target configuration"""
    url: str
    port: Optional[int] = None
    protocol: str = "https"
    timeout: int = 30
    retries: int = 3
    priority: ScanPriority = ScanPriority.MEDIUM
    security_level: SecurityLevel = SecurityLevel.MEDIUM
    
    def __post_init__(self) -> Any:
        """Validate target configuration"""
        if not self.url:
            raise ValueError("URL cannot be empty")
        if self.timeout <= 0:
            raise ValueError("Timeout must be positive")
        if self.retries < 0:
            raise ValueError("Retries cannot be negative")


@dataclass
class ScanResult:
    """Scan result data"""
    target: str
    status: ScanStatus
    findings: List[Dict[str, Any]] = field(default_factory=list)
    false_positives: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    scan_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    error_message: Optional[str] = None


@dataclass
class SecurityMetrics:
    """Security metrics collection"""
    scan_id: str
    start_time: float
    end_time: Optional[float] = None
    total_targets: int = 0
    completed_targets: int = 0
    failed_targets: int = 0
    total_findings: int = 0
    false_positives: int = 0
    true_positives: int = 0
    scan_duration: float = 0.0
    throughput: float = 0.0
    efficiency_score: float = 0.0
    ml_confidence: float = 0.0
    resource_usage: Dict[str, float] = field(default_factory=dict)
    
    def calculate_metrics(self) -> Any:
        """Calculate derived metrics"""
        if self.end_time:
            self.scan_duration = self.end_time - self.start_time
            if self.scan_duration > 0:
                self.throughput = self.completed_targets / self.scan_duration
                self.efficiency_score = self.true_positives / max(self.total_findings, 1)
                self.ml_confidence = 1.0 - (self.false_positives / max(self.total_findings, 1))


class ScanConfiguration(BaseModel):
    """Scan configuration model"""
    scan_type: ScanType = ScanType.VULNERABILITY
    max_concurrent_scans: int = Field(default=10, ge=1, le=100)
    timeout_per_target: int = Field(default=30, ge=5, le=300)
    retry_attempts: int = Field(default=3, ge=0, le=10)
    enable_ml_detection: bool = True
    ml_confidence_threshold: float = Field(default=0.8, ge=0.1, le=1.0)
    enable_connection_multiplexing: bool = True
    max_connections_per_host: int = Field(default=10, ge=1, le=50)
    enable_structured_logging: bool = True
    log_correlation_id: bool = True
    enable_chaos_engineering: bool = False
    chaos_failure_rate: float = Field(default=0.01, ge=0.0, le=0.1)
    target_deduplication: bool = True
    scan_priority: ScanPriority = ScanPriority.MEDIUM
    security_level: SecurityLevel = SecurityLevel.MEDIUM
    key_rotation_interval: int = Field(default=3600, ge=300, le=86400)
    
    @validator('ml_confidence_threshold')
    def validate_ml_threshold(cls, v) -> bool:
        if not 0.1 <= v <= 1.0:
            raise ValueError('ML confidence threshold must be between 0.1 and 1.0')
        return v


class MLFalsePositiveDetector:
    """ML-based false positive detection"""
    
    def __init__(self, confidence_threshold: float = 0.8):
        
    """__init__ function."""
self.confidence_threshold = confidence_threshold
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self._is_trained = False
        
    def _extract_features(self, findings: List[Dict[str, Any]]) -> np.ndarray:
        """Extract features from findings for ML analysis"""
        if not findings:
            return np.array([])
        
        # Extract text features from findings
        texts = []
        for finding in findings:
            text_parts = [
                finding.get('title', ''),
                finding.get('description', ''),
                finding.get('severity', ''),
                finding.get('category', ''),
                str(finding.get('cvss_score', '')),
                finding.get('cve_id', '')
            ]
            texts.append(' '.join(filter(None, text_parts)))
        
        if not texts:
            return np.array([])
        
        # Vectorize text features
        try:
            features = self.vectorizer.fit_transform(texts).toarray()
            return features
        except Exception as e:
            logger.warning(f"Feature extraction failed: {e}")
            return np.array([])
    
    def _train_model(self, features: np.ndarray):
        """Train the isolation forest model"""
        if len(features) == 0:
            return
        
        try:
            self.isolation_forest.fit(features)
            self._is_trained = True
            logger.info("ML model trained successfully", 
                       sample_count=len(features), 
                       feature_count=features.shape[1] if len(features) > 0 else 0)
        except Exception as e:
            logger.error(f"ML model training failed: {e}")
    
    def detect_false_positives(self, findings: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Detect false positives using ML"""
        if not self.enable_ml_detection or not findings:
            return findings, []
        
        features = self._extract_features(findings)
        if len(features) == 0:
            return findings, []
        
        # Train model if not trained
        if not self._is_trained:
            self._train_model(features)
        
        if not self._is_trained:
            return findings, []
        
        try:
            # Predict anomalies (potential false positives)
            predictions = self.isolation_forest.predict(features)
            anomaly_scores = self.isolation_forest.decision_function(features)
            
            true_positives = []
            false_positives = []
            
            for i, finding in enumerate(findings):
                # Higher anomaly score indicates more likely to be false positive
                confidence = 1.0 - (anomaly_scores[i] + 0.5)  # Normalize to 0-1
                
                if confidence >= self.confidence_threshold:
                    true_positives.append(finding)
                else:
                    finding['ml_confidence'] = confidence
                    finding['ml_prediction'] = 'false_positive'
                    false_positives.append(finding)
            
            logger.info("ML false positive detection completed",
                       total_findings=len(findings),
                       true_positives=len(true_positives),
                       false_positives=len(false_positives),
                       avg_confidence=np.mean([1.0 - (score + 0.5) for score in anomaly_scores]))
            
            return true_positives, false_positives
            
        except Exception as e:
            logger.error(f"ML prediction failed: {e}")
            return findings, []


class ConnectionMultiplexer:
    """Connection multiplexing for improved performance"""
    
    def __init__(self, max_connections_per_host: int = 10):
        
    """__init__ function."""
self.max_connections_per_host = max_connections_per_host
        self._connectors: Dict[str, aiohttp.TCPConnector] = {}
        self._sessions: Dict[str, aiohttp.ClientSession] = {}
        
    async def _get_session(self, host: str) -> aiohttp.ClientSession:
        """Get or create session for host"""
        if host not in self._sessions:
            connector = aiohttp.TCPConnector(
                limit=self.max_connections_per_host,
                limit_per_host=self.max_connections_per_host,
                ttl_dns_cache=300,
                use_dns_cache=True,
                keepalive_timeout=30
            )
            self._connectors[host] = connector
            self._sessions[host] = aiohttp.ClientSession(connector=connector)
        
        return self._sessions[host]
    
    async async def request(self, url: str, method: str = "GET", **kwargs) -> aiohttp.ClientResponse:
        """Make HTTP request with connection multiplexing"""
        parsed_url = urlparse(url)
        host = parsed_url.netloc
        session = await self._get_session(host)
        
        return await session.request(method, url, **kwargs)
    
    async def close_all(self) -> Any:
        """Close all sessions and connectors"""
        for session in self._sessions.values():
            await session.close()
        for connector in self._connectors.values():
            await connector.close()
        self._sessions.clear()
        self._connectors.clear()


class ChaosEngineering:
    """Chaos engineering for testing system resilience"""
    
    def __init__(self, failure_rate: float = 0.01):
        
    """__init__ function."""
self.failure_rate = failure_rate
        
    def should_fail(self) -> bool:
        """Determine if operation should fail"""
        return np.random.random() < self.failure_rate
    
    async def inject_failure(self, operation_name: str):
        """Inject controlled failure"""
        if self.should_fail():
            logger.warning(f"Chaos engineering: Injecting failure in {operation_name}")
            raise Exception(f"Chaos engineering failure: {operation_name}")


class OptimizedScanEngine:
    """Enhanced optimized scan engine with advanced features"""
    
    def __init__(self, config: ScanConfiguration):
        
    """__init__ function."""
self.config = config
        self.ml_detector = MLFalsePositiveDetector(config.ml_confidence_threshold)
        self.connection_multiplexer = ConnectionMultiplexer(config.max_connections_per_host)
        self.chaos_engineering = ChaosEngineering(config.chaos_failure_rate)
        self.active_scans: Dict[str, asyncio.Task] = {}
        self.scan_metrics: Dict[str, SecurityMetrics] = {}
        self._scan_semaphore = asyncio.Semaphore(config.max_concurrent_scans)
        self._correlation_ids: Set[str] = set()
        
    def _generate_correlation_id(self) -> str:
        """Generate unique correlation ID for request tracking"""
        correlation_id = str(uuid.uuid4())
        self._correlation_ids.add(correlation_id)
        return correlation_id
    
    def _setup_logging_context(self, correlation_id: str, scan_id: str):
        """Setup structured logging context"""
        if self.config.enable_structured_logging:
            structlog.contextvars.clear_contextvars()
            structlog.contextvars.bind_contextvars(
                correlation_id=correlation_id,
                scan_id=scan_id,
                scan_type=self.config.scan_type.value,
                priority=self.config.scan_priority.value
            )
    
    def _deduplicate_targets(self, targets: List[ScanTarget]) -> List[ScanTarget]:
        """Remove duplicate targets"""
        if not self.config.target_deduplication:
            return targets
        
        seen = set()
        unique_targets = []
        
        for target in targets:
            target_key = f"{target.url}:{target.port}"
            if target_key not in seen:
                seen.add(target_key)
                unique_targets.append(target)
        
        logger.info("Target deduplication completed",
                   original_count=len(targets),
                   unique_count=len(unique_targets))
        
        return unique_targets
    
    async def _scan_single_target(self, target: ScanTarget, scan_id: str) -> ScanResult:
        """Scan a single target with comprehensive error handling"""
        correlation_id = self._generate_correlation_id()
        self._setup_logging_context(correlation_id, scan_id)
        
        result = ScanResult(
            target=target.url,
            status=ScanStatus.RUNNING,
            scan_id=scan_id
        )
        
        start_time = time.time()
        
        try:
            # Chaos engineering injection
            if self.config.enable_chaos_engineering:
                await self.chaos_engineering.inject_failure(f"scan_target_{target.url}")
            
            # Perform actual scan
            findings = await self._perform_scan(target)
            
            # ML-based false positive detection
            if self.config.enable_ml_detection:
                true_positives, false_positives = self.ml_detector.detect_false_positives(findings)
                result.findings = true_positives
                result.false_positives = false_positives
            else:
                result.findings = findings
            
            # Calculate metrics
            end_time = time.time()
            result.metrics = {
                'scan_duration': end_time - start_time,
                'findings_count': len(result.findings),
                'false_positives_count': len(result.false_positives),
                'target_timeout': target.timeout,
                'target_retries': target.retries
            }
            
            result.status = ScanStatus.COMPLETED
            
            logger.info("Target scan completed successfully",
                       target=target.url,
                       duration=result.metrics['scan_duration'],
                       findings=len(result.findings),
                       false_positives=len(result.false_positives))
            
        except asyncio.TimeoutError:
            result.status = ScanStatus.FAILED
            result.error_message = f"Scan timeout after {target.timeout} seconds"
            logger.error("Target scan timeout", target=target.url, timeout=target.timeout)
            
        except Exception as e:
            result.status = ScanStatus.FAILED
            result.error_message = str(e)
            logger.error("Target scan failed", target=target.url, error=str(e), exc_info=True)
        
        return result
    
    async def _perform_scan(self, target: ScanTarget) -> List[Dict[str, Any]]:
        """Perform actual security scan on target"""
        findings = []
        
        # Simulate various security checks
        checks = [
            self._check_ssl_security,
            self._check_headers_security,
            self._check_open_ports,
            self._check_vulnerabilities
        ]
        
        for check_func in checks:
            try:
                check_findings = await check_func(target)
                findings.extend(check_findings)
            except Exception as e:
                logger.warning(f"Check {check_func.__name__} failed for {target.url}: {e}")
        
        return findings
    
    async def _check_ssl_security(self, target: ScanTarget) -> List[Dict[str, Any]]:
        """Check SSL/TLS security"""
        findings = []
        
        try:
            url = f"{target.protocol}://{target.url}"
            async with self.connection_multiplexer.request(url, timeout=aiohttp.ClientTimeout(total=target.timeout)) as response:
                if response.conn and hasattr(response.conn, 'transport'):
                    ssl_info = response.conn.transport.get_extra_info('ssl_object')
                    if ssl_info:
                        # Check SSL version
                        version = ssl_info.version()
                        if version in ['SSLv2', 'SSLv3', 'TLSv1.0', 'TLSv1.1']:
                            findings.append({
                                'title': 'Weak SSL/TLS Version',
                                'description': f'Target uses weak SSL/TLS version: {version}',
                                'severity': 'high',
                                'category': 'ssl_security',
                                'cvss_score': 7.5,
                                'cve_id': 'CVE-2016-2183'
                            })
        except Exception as e:
            logger.debug(f"SSL check failed for {target.url}: {e}")
        
        return findings
    
    async def _check_headers_security(self, target: ScanTarget) -> List[Dict[str, Any]]:
        """Check security headers"""
        findings = []
        
        try:
            url = f"{target.protocol}://{target.url}"
            async with self.connection_multiplexer.request(url, timeout=aiohttp.ClientTimeout(total=target.timeout)) as response:
                headers = response.headers
                
                # Check for missing security headers
                security_headers = {
                    'X-Frame-Options': 'Missing X-Frame-Options header',
                    'X-Content-Type-Options': 'Missing X-Content-Type-Options header',
                    'X-XSS-Protection': 'Missing X-XSS-Protection header',
                    'Strict-Transport-Security': 'Missing HSTS header'
                }
                
                for header, message in security_headers.items():
                    if header not in headers:
                        findings.append({
                            'title': 'Missing Security Header',
                            'description': message,
                            'severity': 'medium',
                            'category': 'security_headers',
                            'cvss_score': 5.0,
                            'cve_id': None
                        })
        except Exception as e:
            logger.debug(f"Headers check failed for {target.url}: {e}")
        
        return findings
    
    async def _check_open_ports(self, target: ScanTarget) -> List[Dict[str, Any]]:
        """Check for open ports"""
        findings = []
        
        if target.port:
            # Check specific port
            try:
                reader, writer = await asyncio.wait_for(
                    asyncio.open_connection(target.url, target.port),
                    timeout=target.timeout
                )
                writer.close()
                await writer.wait_closed()
                
                # Port is open - check if it's a risky port
                risky_ports = {21, 22, 23, 25, 53, 80, 110, 143, 443, 993, 995}
                if target.port in risky_ports:
                    findings.append({
                        'title': 'Risky Port Open',
                        'description': f'Port {target.port} is open and accessible',
                        'severity': 'medium',
                        'category': 'open_ports',
                        'cvss_score': 4.0,
                        'cve_id': None
                    })
            except Exception:
                pass  # Port is closed or unreachable
        
        return findings
    
    async def _check_vulnerabilities(self, target: ScanTarget) -> List[Dict[str, Any]]:
        """Check for common vulnerabilities"""
        findings = []
        
        # Simulate vulnerability checks
        try:
            url = f"{target.protocol}://{target.url}"
            async with self.connection_multiplexer.request(url, timeout=aiohttp.ClientTimeout(total=target.timeout)) as response:
                # Check for directory traversal
                test_url = f"{url}/../../../etc/passwd"
                async with self.connection_multiplexer.request(test_url, timeout=aiohttp.ClientTimeout(total=5)) as test_response:
                    if test_response.status == 200:
                        content = await test_response.text()
                        if 'root:' in content:
                            findings.append({
                                'title': 'Directory Traversal Vulnerability',
                                'description': 'Target is vulnerable to directory traversal attacks',
                                'severity': 'critical',
                                'category': 'directory_traversal',
                                'cvss_score': 9.8,
                                'cve_id': 'CVE-2021-41773'
                            })
        except Exception as e:
            logger.debug(f"Vulnerability check failed for {target.url}: {e}")
        
        return findings
    
    async def scan_targets(self, targets: List[ScanTarget], scan_id: str) -> List[ScanResult]:
        """Scan multiple targets with concurrency control"""
        correlation_id = self._generate_correlation_id()
        self._setup_logging_context(correlation_id, scan_id)
        
        # Initialize metrics
        metrics = SecurityMetrics(
            scan_id=scan_id,
            start_time=time.time(),
            total_targets=len(targets)
        )
        self.scan_metrics[scan_id] = metrics
        
        # Deduplicate targets
        unique_targets = self._deduplicate_targets(targets)
        metrics.total_targets = len(unique_targets)
        
        logger.info("Starting scan operation",
                   scan_id=scan_id,
                   total_targets=len(unique_targets),
                   max_concurrent=self.config.max_concurrent_scans)
        
        try:
            # Create scan tasks with semaphore control
            tasks = []
            for target in unique_targets:
                task = asyncio.create_task(
                    self._scan_with_semaphore(target, scan_id)
                )
                tasks.append(task)
                self.active_scans[scan_id] = task
            
            # Wait for all scans to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            scan_results = []
            for result in results:
                if isinstance(result, Exception):
                    logger.error("Scan task failed", error=str(result))
                    continue
                scan_results.append(result)
                
                # Update metrics
                if result.status == ScanStatus.COMPLETED:
                    metrics.completed_targets += 1
                    metrics.total_findings += len(result.findings)
                    metrics.false_positives += len(result.false_positives)
                    metrics.true_positives += len(result.findings)
                else:
                    metrics.failed_targets += 1
            
            # Finalize metrics
            metrics.end_time = time.time()
            metrics.calculate_metrics()
            
            logger.info("Scan operation completed",
                       scan_id=scan_id,
                       completed=metrics.completed_targets,
                       failed=metrics.failed_targets,
                       duration=metrics.scan_duration,
                       throughput=metrics.throughput,
                       efficiency=metrics.efficiency_score)
            
            return scan_results
            
        except Exception as e:
            logger.error("Scan operation failed", scan_id=scan_id, error=str(e), exc_info=True)
            raise
        finally:
            # Cleanup
            if scan_id in self.active_scans:
                del self.active_scans[scan_id]
    
    async def _scan_with_semaphore(self, target: ScanTarget, scan_id: str) -> ScanResult:
        """Scan target with semaphore control"""
        async with self._scan_semaphore:
            return await self._scan_single_target(target, scan_id)
    
    async def cancel_scan(self, scan_id: str) -> bool:
        """Cancel active scan"""
        if scan_id in self.active_scans:
            task = self.active_scans[scan_id]
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            del self.active_scans[scan_id]
            
            # Update metrics
            if scan_id in self.scan_metrics:
                self.scan_metrics[scan_id].end_time = time.time()
                self.scan_metrics[scan_id].calculate_metrics()
            
            logger.info("Scan cancelled", scan_id=scan_id)
            return True
        
        return False
    
    def get_scan_metrics(self, scan_id: str) -> Optional[SecurityMetrics]:
        """Get scan metrics"""
        return self.scan_metrics.get(scan_id)
    
    async def cleanup(self) -> Any:
        """Cleanup resources"""
        # Cancel all active scans
        for scan_id in list(self.active_scans.keys()):
            await self.cancel_scan(scan_id)
        
        # Close connection multiplexer
        await self.connection_multiplexer.close_all()
        
        logger.info("Scan engine cleanup completed")


# FastAPI Integration
class ScanRequest(BaseModel):
    """Scan request model"""
    targets: List[ScanTarget]
    configuration: Optional[ScanConfiguration] = None
    
    @validator('targets')
    def validate_targets(cls, v) -> Optional[Dict[str, Any]]:
        if not v:
            raise ValueError('At least one target is required')
        return v


class ScanResponse(BaseModel):
    """Scan response model"""
    scan_id: str
    status: ScanStatus
    results: List[ScanResult]
    metrics: SecurityMetrics
    message: str


# Dependency injection
_scan_engine: Optional[OptimizedScanEngine] = None


def get_scan_engine() -> OptimizedScanEngine:
    """Get scan engine instance"""
    global _scan_engine
    if _scan_engine is None:
        config = ScanConfiguration()
        _scan_engine = OptimizedScanEngine(config)
    return _scan_engine


async def cleanup_scan_engine():
    """Cleanup scan engine on shutdown"""
    global _scan_engine
    if _scan_engine:
        await _scan_engine.cleanup()
        _scan_engine = None


# FastAPI routes
async def start_scan(
    request: ScanRequest,
    background_tasks: BackgroundTasks,
    scan_engine: OptimizedScanEngine = Depends(get_scan_engine)
) -> ScanResponse:
    """Start a new scan operation"""
    scan_id = str(uuid.uuid4())
    
    # Configure scan
    config = request.configuration or ScanConfiguration()
    
    # Start scan in background
    background_tasks.add_task(
        scan_engine.scan_targets,
        request.targets,
        scan_id
    )
    
    return ScanResponse(
        scan_id=scan_id,
        status=ScanStatus.PENDING,
        results=[],
        metrics=SecurityMetrics(scan_id=scan_id, start_time=time.time()),
        message="Scan started successfully"
    )


async def get_scan_status(
    scan_id: str,
    scan_engine: OptimizedScanEngine = Depends(get_scan_engine)
) -> ScanResponse:
    """Get scan status and results"""
    metrics = scan_engine.get_scan_metrics(scan_id)
    if not metrics:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Scan {scan_id} not found"
        )
    
    # In a real implementation, you would retrieve results from storage
    results = []  # Placeholder
    
    return ScanResponse(
        scan_id=scan_id,
        status=ScanStatus.COMPLETED if metrics.end_time else ScanStatus.RUNNING,
        results=results,
        metrics=metrics,
        message="Scan status retrieved successfully"
    )


async def cancel_scan(
    scan_id: str,
    scan_engine: OptimizedScanEngine = Depends(get_scan_engine)
) -> dict:
    """Cancel an active scan"""
    success = await scan_engine.cancel_scan(scan_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Active scan {scan_id} not found"
        )
    
    return {"message": f"Scan {scan_id} cancelled successfully"}


async def health_check(
    scan_engine: OptimizedScanEngine = Depends(get_scan_engine)
) -> dict:
    """Health check endpoint"""
    active_scans = len(scan_engine.active_scans)
    total_metrics = len(scan_engine.scan_metrics)
    
    return {
        "status": "healthy",
        "active_scans": active_scans,
        "total_metrics": total_metrics,
        "timestamp": time.time()
    } 