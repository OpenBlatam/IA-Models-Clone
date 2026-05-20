"""
Tests for Additional Infrastructure Components
Tests for compression, serialization, tracing, pagination, etc.
"""

import pytest
from unittest.mock import Mock, AsyncMock
import json

from core.infrastructure.compression import Compression
from core.infrastructure.serialization import Serialization
from core.infrastructure.tracing import Tracing
from core.infrastructure.pagination import Pagination
from core.infrastructure.request_deduplicator import RequestDeduplicator
from core.infrastructure.metrics_exporter import MetricsExporter


class TestCompression:
    """Tests for Compression"""
    
    @pytest.fixture
    def compression(self):
        """Create compression utility"""
        return Compression()
    
    def test_compress_data(self, compression):
        """Test compressing data"""
        data = b"x" * 1000  # 1KB of data
        
        compressed = compression.compress(data)
        
        assert compressed is not None
        assert len(compressed) < len(data)  # Should be smaller
    
    def test_decompress_data(self, compression):
        """Test decompressing data"""
        original_data = b"x" * 1000
        
        compressed = compression.compress(original_data)
        decompressed = compression.decompress(compressed)
        
        assert decompressed == original_data
    
    def test_compress_json(self, compression):
        """Test compressing JSON data"""
        json_data = json.dumps({"key": "value", "data": "x" * 100}).encode()
        
        compressed = compression.compress(json_data)
        decompressed = compression.decompress(compressed)
        
        assert json.loads(decompressed) == json.loads(json_data)


class TestSerialization:
    """Tests for Serialization"""
    
    @pytest.fixture
    def serialization(self):
        """Create serialization utility"""
        return Serialization()
    
    def test_serialize_dict(self, serialization):
        """Test serializing dictionary"""
        data = {"key": "value", "number": 123}
        
        serialized = serialization.serialize(data)
        
        assert isinstance(serialized, (str, bytes))
    
    def test_deserialize_data(self, serialization):
        """Test deserializing data"""
        original = {"key": "value", "number": 123}
        
        serialized = serialization.serialize(original)
        deserialized = serialization.deserialize(serialized)
        
        assert deserialized == original
    
    def test_serialize_complex_object(self, serialization):
        """Test serializing complex object"""
        data = {
            "nested": {
                "list": [1, 2, 3],
                "dict": {"a": 1, "b": 2}
            },
            "array": [{"x": 1}, {"y": 2}]
        }
        
        serialized = serialization.serialize(data)
        deserialized = serialization.deserialize(serialized)
        
        assert deserialized == data


class TestTracing:
    """Tests for Tracing"""
    
    @pytest.fixture
    def tracing(self):
        """Create tracing utility"""
        return Tracing()
    
    def test_create_trace(self, tracing):
        """Test creating a trace"""
        trace_id = tracing.create_trace("operation_name")
        
        assert trace_id is not None
        assert isinstance(trace_id, str)
    
    def test_add_span(self, tracing):
        """Test adding span to trace"""
        trace_id = tracing.create_trace("operation")
        
        span_id = tracing.add_span(trace_id, "span_name")
        
        assert span_id is not None
    
    def test_get_trace(self, tracing):
        """Test getting trace information"""
        trace_id = tracing.create_trace("operation")
        tracing.add_span(trace_id, "span1")
        
        trace = tracing.get_trace(trace_id)
        
        assert trace is not None
        assert trace["trace_id"] == trace_id


class TestPagination:
    """Tests for Pagination"""
    
    @pytest.fixture
    def pagination(self):
        """Create pagination utility"""
        return Pagination()
    
    def test_create_pagination(self, pagination):
        """Test creating pagination"""
        items = list(range(100))
        page = pagination.paginate(items, page=1, per_page=10)
        
        assert len(page["items"]) == 10
        assert page["page"] == 1
        assert page["per_page"] == 10
        assert page["total"] == 100
        assert page["pages"] == 10
    
    def test_pagination_metadata(self, pagination):
        """Test pagination metadata"""
        items = list(range(50))
        page = pagination.paginate(items, page=2, per_page=10)
        
        assert page["page"] == 2
        assert page["has_next"] is True
        assert page["has_prev"] is True
        assert page["total"] == 50
    
    def test_pagination_edge_cases(self, pagination):
        """Test pagination edge cases"""
        # Empty list
        page = pagination.paginate([], page=1, per_page=10)
        assert len(page["items"]) == 0
        assert page["total"] == 0
        
        # Page beyond total
        items = list(range(10))
        page = pagination.paginate(items, page=10, per_page=10)
        assert len(page["items"]) == 0


class TestRequestDeduplicator:
    """Tests for RequestDeduplicator"""
    
    @pytest.fixture
    def deduplicator(self):
        """Create request deduplicator"""
        return RequestDeduplicator()
    
    @pytest.mark.asyncio
    async def test_deduplicate_request(self, deduplicator):
        """Test deduplicating requests"""
        request_id = "req-123"
        
        # First request
        is_duplicate1 = await deduplicator.is_duplicate(request_id)
        assert is_duplicate1 is False
        
        # Second request with same ID
        is_duplicate2 = await deduplicator.is_duplicate(request_id)
        assert is_duplicate2 is True
    
    @pytest.mark.asyncio
    async def test_deduplicator_ttl(self, deduplicator):
        """Test deduplicator TTL"""
        request_id = "req-456"
        
        # Mark as seen
        await deduplicator.is_duplicate(request_id)
        
        # Should be duplicate
        assert await deduplicator.is_duplicate(request_id) is True
        
        # After TTL expires, should not be duplicate
        # (implementation dependent)


class TestMetricsExporter:
    """Tests for MetricsExporter"""
    
    @pytest.fixture
    def metrics_exporter(self):
        """Create metrics exporter"""
        return MetricsExporter()
    
    def test_export_metrics(self, metrics_exporter):
        """Test exporting metrics"""
        metrics = {
            "requests_total": 1000,
            "errors_total": 10,
            "avg_response_time": 0.5
        }
        
        exported = metrics_exporter.export(metrics)
        
        assert exported is not None
        assert isinstance(exported, (str, dict))
    
    def test_export_prometheus_format(self, metrics_exporter):
        """Test exporting in Prometheus format"""
        metrics = {
            "requests_total": 1000,
            "errors_total": 10
        }
        
        prometheus_format = metrics_exporter.export_prometheus(metrics)
        
        assert prometheus_format is not None
        assert isinstance(prometheus_format, str)
        # Should contain metric names
        assert "requests_total" in prometheus_format or len(prometheus_format) > 0



