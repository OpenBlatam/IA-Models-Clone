"""
Utility Functions for Ultra-Adaptive K/V Cache Engine
Helper functions for common operations, optimizations, and integrations
"""

import time
import hashlib
import json
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


def generate_session_id(user_id: Optional[str] = None, context: Optional[str] = None) -> str:
    """
    Generate a session ID based on user and context.
    
    Args:
        user_id: Optional user identifier
        context: Optional context identifier
        
    Returns:
        Session ID string
    """
    components = []
    
    if user_id:
        components.append(f"user_{user_id}")
    if context:
        components.append(f"ctx_{hashlib.md5(context.encode()).hexdigest()[:8]}")
    
    if not components:
        components.append(f"session_{int(time.time())}")
    
    return "_".join(components) + f"_{int(time.time())}"


def calculate_request_hash(request: Dict[str, Any]) -> str:
    """
    Calculate hash for a request to use as cache key.
    
    Args:
        request: Request dictionary
        
    Returns:
        MD5 hash string
    """
    # Include relevant fields for hashing
    hash_input = json.dumps({
        'text': request.get('text', ''),
        'max_length': request.get('max_length', 100),
        'temperature': request.get('temperature', 1.0),
        'session_id': request.get('session_id', '')
    }, sort_keys=True)
    
    return hashlib.md5(hash_input.encode()).hexdigest()


def estimate_request_complexity(request: Dict[str, Any]) -> Dict[str, Any]:
    """
    Estimate complexity of a request to help with load balancing.
    
    Args:
        request: Request dictionary
        
    Returns:
        Complexity metrics dictionary
    """
    text = request.get('text', '')
    max_length = request.get('max_length', 100)
    temperature = request.get('temperature', 1.0)
    
    # Simple complexity estimation
    text_length = len(text)
    word_count = len(text.split())
    estimated_tokens = text_length // 4  # Rough estimate
    
    complexity = {
        'text_length': text_length,
        'word_count': word_count,
        'estimated_tokens': estimated_tokens,
        'max_generation_length': max_length,
        'temperature': temperature,
        'complexity_score': (text_length * 0.3 + max_length * 0.7) / 100.0,
        'estimated_processing_time': estimated_tokens * 0.01 + max_length * 0.05  # Rough estimate
    }
    
    return complexity


def optimize_batch_requests(requests: List[Dict[str, Any]], 
                            max_batch_size: int = 20) -> List[List[Dict[str, Any]]]:
    """
    Optimize requests into batches based on complexity and session affinity.
    
    Args:
        requests: List of requests
        max_batch_size: Maximum size per batch
        
    Returns:
        List of optimized batches
    """
    if len(requests) <= max_batch_size:
        return [requests]
    
    # Group by session for better cache utilization
    session_groups = defaultdict(list)
    other_requests = []
    
    for req in requests:
        session_id = req.get('session_id')
        if session_id:
            session_groups[session_id].append(req)
        else:
            other_requests.append(req)
    
    batches = []
    
    # Add session-based batches
    for session_id, session_requests in session_groups.items():
        # Split large session batches
        for i in range(0, len(session_requests), max_batch_size):
            batches.append(session_requests[i:i + max_batch_size])
    
    # Add remaining requests in batches
    for i in range(0, len(other_requests), max_batch_size):
        batches.append(other_requests[i:i + max_batch_size])
    
    return batches


def validate_request(request: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    Validate a request before processing.
    
    Args:
        request: Request dictionary
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check required fields
    if 'text' not in request:
        return False, "Missing required field: 'text'"
    
    text = request.get('text', '')
    if not isinstance(text, str):
        return False, "Field 'text' must be a string"
    
    if len(text) == 0:
        return False, "Field 'text' cannot be empty"
    
    # Check optional fields
    max_length = request.get('max_length', 100)
    if not isinstance(max_length, int) or max_length <= 0:
        return False, "Field 'max_length' must be a positive integer"
    
    if max_length > 10000:
        return False, "Field 'max_length' cannot exceed 10000"
    
    temperature = request.get('temperature', 1.0)
    if not isinstance(temperature, (int, float)) or temperature < 0 or temperature > 2:
        return False, "Field 'temperature' must be between 0 and 2"
    
    return True, None


def format_performance_stats(stats: Dict[str, Any]) -> str:
    """
    Format performance statistics as a human-readable string.
    
    Args:
        stats: Performance statistics dictionary
        
    Returns:
        Formatted string
    """
    engine_stats = stats.get('engine_stats', {})
    
    lines = [
        "Performance Statistics",
        "=" * 60,
        f"Total Requests: {engine_stats.get('total_requests', 0):,}",
        f"Total Tokens: {engine_stats.get('total_tokens', 0):,}",
        f"Average Response Time: {engine_stats.get('avg_response_time', 0)*1000:.2f} ms",
        f"P50 Latency: {engine_stats.get('p50_response_time', 0)*1000:.2f} ms",
        f"P95 Latency: {engine_stats.get('p95_response_time', 0)*1000:.2f} ms",
        f"P99 Latency: {engine_stats.get('p99_response_time', 0)*1000:.2f} ms",
        f"Throughput: {engine_stats.get('throughput', 0):.2f} req/s",
        f"Cache Hit Rate: {engine_stats.get('cache_hit_rate', 0)*100:.2f}%",
        f"Error Rate: {engine_stats.get('error_rate', 0)*100:.2f}%",
        f"Memory Usage: {stats.get('memory_usage', 0)*100:.2f}%",
        f"Active Sessions: {stats.get('active_sessions', 0)}",
        f"Available GPUs: {stats.get('available_gpus', 0)}",
    ]
    
    return "\n".join(lines)


def export_stats_to_file(stats: Dict[str, Any], output_file: str):
    """
    Export performance statistics to a JSON file.
    
    Args:
        stats: Performance statistics dictionary
        output_file: Output file path
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    export_data = {
        'timestamp': time.time(),
        'datetime': time.strftime('%Y-%m-%d %H:%M:%S'),
        'stats': stats
    }
    
    with open(output_file, 'w') as f:
        json.dump(export_data, f, indent=2, default=str)
    
    logger.info(f"Exported stats to {output_file}")


def create_workload_profile(requests: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Create a workload profile from a list of requests.
    
    Args:
        requests: List of requests
        
    Returns:
        Workload profile dictionary
    """
    if not requests:
        return {}
    
    text_lengths = [len(r.get('text', '')) for r in requests]
    max_lengths = [r.get('max_length', 100) for r in requests]
    temperatures = [r.get('temperature', 1.0) for r in requests]
    
    session_ids = [r.get('session_id') for r in requests if r.get('session_id')]
    unique_sessions = len(set(session_ids))
    
    return {
        'batch_size': len(requests),
        'avg_text_length': sum(text_lengths) / len(text_lengths),
        'max_text_length': max(text_lengths),
        'min_text_length': min(text_lengths),
        'avg_max_length': sum(max_lengths) / len(max_lengths),
        'avg_temperature': sum(temperatures) / len(temperatures),
        'unique_sessions': unique_sessions,
        'session_reuse_rate': unique_sessions / len(requests) if requests else 0,
        'estimated_complexity': sum(len(t) for t in text_lengths) / len(text_lengths) / 100.0
    }


def recommend_config_optimizations(workload_profile: Dict[str, Any], 
                                  current_stats: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recommend configuration optimizations based on workload and current stats.
    
    Args:
        workload_profile: Workload profile dictionary
        current_stats: Current performance statistics
        
    Returns:
        Recommended optimizations dictionary
    """
    recommendations = {
        'recommended_changes': [],
        'reasoning': []
    }
    
    engine_stats = current_stats.get('engine_stats', {})
    
    # Check memory usage
    memory_usage = current_stats.get('memory_usage', 0)
    if memory_usage > 0.9:
        recommendations['recommended_changes'].append({
            'setting': 'compression_ratio',
            'current': 0.3,
            'recommended': 0.5,
            'reason': 'High memory usage detected'
        })
        recommendations['reasoning'].append(
            f"Memory usage is {memory_usage*100:.1f}%, consider increasing compression"
        )
    
    # Check error rate
    error_rate = engine_stats.get('error_rate', 0)
    if error_rate > 0.05:
        recommendations['recommended_changes'].append({
            'setting': 'num_workers',
            'current': 4,
            'recommended': 2,
            'reason': 'High error rate, reduce concurrency'
        })
        recommendations['reasoning'].append(
            f"Error rate is {error_rate*100:.1f}%, consider reducing workers"
        )
    
    # Check throughput
    throughput = engine_stats.get('throughput', 0)
    if throughput < 1.0 and workload_profile.get('batch_size', 0) > 5:
        recommendations['recommended_changes'].append({
            'setting': 'dynamic_batching',
            'current': True,
            'recommended': True,
            'reason': 'Low throughput, batching may help'
        })
        recommendations['reasoning'].append(
            "Throughput is low, ensure dynamic batching is enabled"
        )
    
    # Check cache hit rate
    cache_hit_rate = engine_stats.get('cache_hit_rate', 1.0)
    if cache_hit_rate < 0.5:
        recommendations['recommended_changes'].append({
            'setting': 'cache_size',
            'current': 8192,
            'recommended': 16384,
            'reason': 'Low cache hit rate'
        })
        recommendations['reasoning'].append(
            f"Cache hit rate is {cache_hit_rate*100:.1f}%, consider increasing cache size"
        )
    
    return recommendations


def sanitize_session_data(session_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitize session data for logging/storage.
    
    Args:
        session_data: Session data dictionary
        
    Returns:
        Sanitized session data
    """
    sanitized = session_data.copy()
    
    # Remove or truncate large fields
    if 'text' in sanitized and len(str(sanitized['text'])) > 1000:
        sanitized['text'] = str(sanitized['text'])[:1000] + "..."
    
    # Remove sensitive fields
    sensitive_fields = ['api_key', 'password', 'token', 'secret']
    for field in sensitive_fields:
        if field in sanitized:
            sanitized[field] = "***REDACTED***"
    
    return sanitized


def calculate_cost_estimate(requests: List[Dict[str, Any]], 
                            tokens_per_request: float = 100.0,
                            cost_per_1k_tokens: float = 0.01) -> Dict[str, Any]:
    """
    Calculate cost estimate for processing requests.
    
    Args:
        requests: List of requests
        tokens_per_request: Estimated tokens per request
        cost_per_1k_tokens: Cost per 1000 tokens
        
    Returns:
        Cost estimate dictionary
    """
    total_requests = len(requests)
    estimated_tokens = total_requests * tokens_per_request
    estimated_cost = (estimated_tokens / 1000.0) * cost_per_1k_tokens
    
    return {
        'total_requests': total_requests,
        'estimated_tokens': estimated_tokens,
        'estimated_cost': estimated_cost,
        'cost_per_request': estimated_cost / total_requests if total_requests > 0 else 0,
        'tokens_per_request': tokens_per_request,
        'cost_per_1k_tokens': cost_per_1k_tokens
    }

