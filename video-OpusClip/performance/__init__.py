#!/usr/bin/env python3
"""
Performance Package

Advanced performance optimization and auto-scaling system for the Video-OpusClip API.
"""

from .auto_scaling import (
    ScalingStrategy,
    ScalingAction,
    ResourceMetrics,
    ScalingDecision,
    ScalingConfig,
    AutoScalingEngine,
    LoadBalancer,
    auto_scaling_engine,
    load_balancer
)

__all__ = [
    'ScalingStrategy',
    'ScalingAction',
    'ResourceMetrics',
    'ScalingDecision',
    'ScalingConfig',
    'AutoScalingEngine',
    'LoadBalancer',
    'auto_scaling_engine',
    'load_balancer'
]





























