"""
gRPC Support for Inter-Service Communication
High-performance binary protocol for microservices
"""

try:
    import grpc
    from grpc import aio as grpc_aio
    GRPC_AVAILABLE = True
except ImportError:
    GRPC_AVAILABLE = False

__all__ = [
    "create_grpc_server",
    "GRPC_AVAILABLE",
]















