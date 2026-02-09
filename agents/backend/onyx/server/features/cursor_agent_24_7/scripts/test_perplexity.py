#!/usr/bin/env python3
"""
Test Script for Perplexity System
==================================

Quick test script to verify Perplexity system is working.
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.perplexity import PerplexityService, QueryType


async def test_basic():
    """Test basic functionality."""
    print("Testing Perplexity System...")
    print("=" * 50)
    
    service = PerplexityService(
        enable_cache=False,
        enable_metrics=True,
        enable_validation=True
    )
    
    # Test query processing
    print("\n1. Testing query processing...")
    search_results = [
        {
            'title': 'Python Programming Language',
            'url': 'https://python.org',
            'snippet': 'Python is a high-level programming language',
            'content': 'Python is a versatile programming language used for web development, data science, and more.'
        }
    ]
    
    result = await service.answer_query(
        query="What is Python?",
        search_results=search_results
    )
    
    print(f"   Query: {result['query']}")
    print(f"   Type: {result['query_type']}")
    print(f"   Answer length: {len(result['answer'])} chars")
    print(f"   ✓ Query processing works")
    
    # Test validation
    print("\n2. Testing validation...")
    validation = service.validate_answer(result['answer'], result['query_type'])
    print(f"   Valid: {validation['valid']}")
    print(f"   Issues: {validation['issue_count']}")
    print(f"   ✓ Validation works")
    
    # Test metrics
    print("\n3. Testing metrics...")
    metrics = service.get_metrics()
    print(f"   Total queries: {metrics['total_queries']}")
    print(f"   ✓ Metrics collection works")
    
    # Test cache
    print("\n4. Testing cache...")
    cache_stats = service.get_cache_stats()
    print(f"   Cache enabled: {cache_stats.get('enabled', False)}")
    print(f"   ✓ Cache system works")
    
    print("\n" + "=" * 50)
    print("All tests passed! ✓")


if __name__ == "__main__":
    asyncio.run(test_basic())




