"""
Tracing Types and Definitions
=============================

Type definitions for distributed tracing and APM.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timezone
import uuid
import struct

class SpanKind(Enum):
    """Span kind enumeration."""
    INTERNAL = "internal"
    SERVER = "server"
    CLIENT = "client"
    PRODUCER = "producer"
    CONSUMER = "consumer"

class SpanStatus(Enum):
    """Span status enumeration."""
    UNSET = "unset"
    OK = "ok"
    ERROR = "error"

class TraceFlags(Enum):
    """Trace flags enumeration."""
    SAMPLED = 0x01
    DEBUG = 0x02

@dataclass
class TraceId:
    """Trace ID representation."""
    value: str
    
    def __init__(self, value: str = None):
        if value is None:
            # Generate 128-bit trace ID
            self.value = format(uuid.uuid4().int, '032x')
        else:
            self.value = value
    
    def __str__(self) -> str:
        return self.value
    
    def __eq__(self, other) -> bool:
        return isinstance(other, TraceId) and self.value == other.value

@dataclass
class SpanId:
    """Span ID representation."""
    value: str
    
    def __init__(self, value: str = None):
        if value is None:
            # Generate 64-bit span ID
            self.value = format(uuid.uuid4().int & 0xFFFFFFFFFFFFFFFF, '016x')
        else:
            self.value = value
    
    def __str__(self) -> str:
        return self.value
    
    def __eq__(self, other) -> bool:
        return isinstance(other, SpanId) and self.value == other.value

@dataclass
class TraceState:
    """Trace state representation."""
    entries: Dict[str, str] = field(default_factory=dict)
    
    def add(self, key: str, value: str):
        """Add trace state entry."""
        self.entries[key] = value
    
    def get(self, key: str) -> Optional[str]:
        """Get trace state entry."""
        return self.entries.get(key)
    
    def remove(self, key: str):
        """Remove trace state entry."""
        self.entries.pop(key, None)
    
    def to_string(self) -> str:
        """Convert to W3C trace state string."""
        if not self.entries:
            return ""
        return ",".join([f"{k}={v}" for k, v in self.entries.items()])
    
    @classmethod
    def from_string(cls, trace_state_str: str) -> 'TraceState':
        """Create from W3C trace state string."""
        entries = {}
        if trace_state_str:
            for entry in trace_state_str.split(","):
                if "=" in entry:
                    key, value = entry.split("=", 1)
                    entries[key.strip()] = value.strip()
        return cls(entries=entries)

@dataclass
class SpanContext:
    """Span context representation."""
    trace_id: TraceId
    span_id: SpanId
    trace_flags: int = 0
    trace_state: TraceState = field(default_factory=TraceState)
    is_remote: bool = False
    
    def is_valid(self) -> bool:
        """Check if span context is valid."""
        return (self.trace_id.value != "0" * 32 and 
                self.span_id.value != "0" * 16)
    
    def is_sampled(self) -> bool:
        """Check if span is sampled."""
        return bool(self.trace_flags & TraceFlags.SAMPLED.value)
    
    def is_debug(self) -> bool:
        """Check if span is in debug mode."""
        return bool(self.trace_flags & TraceFlags.DEBUG.value)

@dataclass
class SpanAttribute:
    """Span attribute definition."""
    key: str
    value: Union[str, int, float, bool, List[Union[str, int, float, bool]]]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {self.key: self.value}

@dataclass
class SpanEvent:
    """Span event definition."""
    name: str
    timestamp: datetime
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    def add_attribute(self, key: str, value: Any):
        """Add event attribute."""
        self.attributes[key] = value

@dataclass
class SpanLink:
    """Span link definition."""
    context: SpanContext
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    def add_attribute(self, key: str, value: Any):
        """Add link attribute."""
        self.attributes[key] = value

@dataclass
class Span:
    """Span representation."""
    name: str
    context: SpanContext
    parent_context: Optional[SpanContext] = None
    kind: SpanKind = SpanKind.INTERNAL
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: Optional[datetime] = None
    status: SpanStatus = SpanStatus.UNSET
    status_message: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[SpanEvent] = field(default_factory=list)
    links: List[SpanLink] = field(default_factory=list)
    resource: Dict[str, Any] = field(default_factory=dict)
    
    def add_attribute(self, key: str, value: Any):
        """Add span attribute."""
        self.attributes[key] = value
    
    def add_event(self, name: str, attributes: Dict[str, Any] = None):
        """Add span event."""
        event = SpanEvent(
            name=name,
            timestamp=datetime.now(timezone.utc),
            attributes=attributes or {}
        )
        self.events.append(event)
    
    def add_link(self, context: SpanContext, attributes: Dict[str, Any] = None):
        """Add span link."""
        link = SpanLink(
            context=context,
            attributes=attributes or {}
        )
        self.links.append(link)
    
    def set_status(self, status: SpanStatus, message: Optional[str] = None):
        """Set span status."""
        self.status = status
        self.status_message = message
    
    def finish(self):
        """Finish the span."""
        self.end_time = datetime.now(timezone.utc)
    
    def duration(self) -> Optional[float]:
        """Get span duration in seconds."""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None
    
    def is_finished(self) -> bool:
        """Check if span is finished."""
        return self.end_time is not None

@dataclass
class Trace:
    """Trace representation."""
    trace_id: TraceId
    spans: List[Span] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    def add_span(self, span: Span):
        """Add span to trace."""
        self.spans.append(span)
        
        if not self.start_time or span.start_time < self.start_time:
            self.start_time = span.start_time
        
        if span.end_time and (not self.end_time or span.end_time > self.end_time):
            self.end_time = span.end_time
    
    def get_root_spans(self) -> List[Span]:
        """Get root spans (spans without parent)."""
        root_spans = []
        span_ids = {span.context.span_id for span in self.spans}
        
        for span in self.spans:
            if (not span.parent_context or 
                span.parent_context.span_id not in span_ids):
                root_spans.append(span)
        
        return root_spans
    
    def get_span_tree(self) -> Dict[str, List[Span]]:
        """Get span tree structure."""
        tree = {}
        
        for span in self.spans:
            parent_id = (span.parent_context.span_id.value 
                        if span.parent_context else "root")
            
            if parent_id not in tree:
                tree[parent_id] = []
            tree[parent_id].append(span)
        
        return tree

@dataclass
class SamplingDecision:
    """Sampling decision."""
    decision: bool
    attributes: Dict[str, Any] = field(default_factory=dict)
    trace_state: Optional[TraceState] = None

@dataclass
class Sampler:
    """Sampler configuration."""
    name: str
    type: str  # always_on, always_off, trace_id_ratio, parent_based
    config: Dict[str, Any] = field(default_factory=dict)
    
    def should_sample(
        self, 
        trace_id: TraceId, 
        parent_context: Optional[SpanContext] = None
    ) -> SamplingDecision:
        """Determine if span should be sampled."""
        if self.type == "always_on":
            return SamplingDecision(True)
        elif self.type == "always_off":
            return SamplingDecision(False)
        elif self.type == "trace_id_ratio":
            ratio = self.config.get("ratio", 1.0)
            # Use trace ID to determine sampling
            trace_id_int = int(trace_id.value[:8], 16)
            return SamplingDecision(trace_id_int < (ratio * 0xFFFFFFFF))
        elif self.type == "parent_based":
            if parent_context:
                return SamplingDecision(parent_context.is_sampled())
            else:
                # Use default sampler for root spans
                return SamplingDecision(True)
        
        return SamplingDecision(False)

@dataclass
class ExporterConfig:
    """Exporter configuration."""
    name: str
    type: str  # jaeger, zipkin, otlp, console, file
    endpoint: Optional[str] = None
    headers: Dict[str, str] = field(default_factory=dict)
    timeout: int = 30
    batch_size: int = 512
    export_interval: int = 5
    config: Dict[str, Any] = field(default_factory=dict)

@dataclass
class InstrumentationConfig:
    """Instrumentation configuration."""
    enabled: bool = True
    capture_parameters: bool = True
    capture_return_values: bool = False
    capture_exceptions: bool = True
    max_attributes: int = 128
    max_events: int = 128
    max_links: int = 128
    max_attribute_length: int = 1024

@dataclass
class TracingConfig:
    """Tracing configuration."""
    service_name: str
    service_version: str = "1.0.0"
    environment: str = "production"
    sampler: Sampler = field(default_factory=lambda: Sampler("default", "always_on"))
    exporters: List[ExporterConfig] = field(default_factory=list)
    instrumentation: InstrumentationConfig = field(default_factory=InstrumentationConfig)
    resource: Dict[str, Any] = field(default_factory=dict)
    max_span_count: int = 1000
    max_trace_duration: int = 300  # seconds

@dataclass
class APMMetrics:
    """APM metrics."""
    request_count: int = 0
    request_duration_ms: float = 0.0
    error_count: int = 0
    error_rate: float = 0.0
    throughput_rps: float = 0.0
    cpu_usage_percent: float = 0.0
    memory_usage_mb: float = 0.0
    database_query_count: int = 0
    database_query_duration_ms: float = 0.0
    cache_hit_rate: float = 0.0
    external_api_calls: int = 0
    external_api_duration_ms: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

@dataclass
class PerformanceProfile:
    """Performance profile."""
    operation_name: str
    duration_ms: float
    cpu_time_ms: float
    memory_allocated_mb: float
    database_queries: int = 0
    external_calls: int = 0
    cache_operations: int = 0
    bottlenecks: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

@dataclass
class ErrorTracking:
    """Error tracking information."""
    error_id: str
    error_type: str
    error_message: str
    stack_trace: str
    context: Dict[str, Any] = field(default_factory=dict)
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    severity: str = "error"  # error, warning, info
    resolved: bool = False
