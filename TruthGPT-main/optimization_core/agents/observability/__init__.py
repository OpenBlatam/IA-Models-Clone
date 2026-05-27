"""
OpenClaw -- Agent Observability & Tracing.
"""
from .tracer import Tracer

# Singleton tracer instance for the entire application
global_tracer = Tracer()
