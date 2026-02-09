"""
Perplexity Examples - Usage examples
=====================================

Example code demonstrating how to use the Perplexity system.
"""

import asyncio
from typing import List, Dict, Any
from .service import PerplexityService
from .processor import PerplexityProcessor
from .config import PerplexityConfig


# Example 1: Basic query processing
async def example_basic_query():
    """Basic example of processing a query."""
    service = PerplexityService()
    
    search_results = [
        {
            'title': 'Python Programming',
            'url': 'https://python.org',
            'snippet': 'Python is a high-level programming language',
            'content': 'Python is a high-level, interpreted programming language...'
        }
    ]
    
    result = await service.answer_query(
        query="What is Python?",
        search_results=search_results
    )
    
    print(f"Query: {result['query']}")
    print(f"Type: {result['query_type']}")
    print(f"Answer: {result['answer']}")


# Example 2: With LLM provider
async def example_with_llm():
    """Example using LLM provider."""
    import openai
    
    # Initialize LLM provider
    llm_client = openai.AsyncOpenAI(api_key="your-api-key")
    
    service = PerplexityService()
    
    result = await service.answer_query(
        query="Explain quantum computing",
        search_results=[...],
        llm_provider=llm_client,
        use_llm=True
    )
    
    print(result['answer'])


# Example 3: Batch processing
async def example_batch_processing():
    """Example of batch processing multiple queries."""
    service = PerplexityService()
    
    queries = [
        {
            'query': 'What is Python?',
            'search_results': [...]
        },
        {
            'query': 'What is JavaScript?',
            'search_results': [...]
        }
    ]
    
    results = await service.batch_process(queries)
    
    for result in results:
        if result['success']:
            print(f"Query: {result['query']}")
            print(f"Answer: {result['answer']}")
        else:
            print(f"Error: {result['error']}")


# Example 4: Custom configuration
def example_custom_config():
    """Example with custom configuration."""
    config = PerplexityConfig(
        enable_validation=True,
        enable_cache=True,
        enable_metrics=True,
        cache_ttl=7200,  # 2 hours
        cache_max_size=5000
    )
    
    service = PerplexityService(config=config)
    return service


# Example 5: Using processor directly
async def example_direct_processor():
    """Example using processor directly."""
    processor = PerplexityProcessor(
        enable_validation=True,
        enable_cache=True,
        enable_metrics=True
    )
    
    # Process query
    processed = processor.process_query(
        query="What is the weather?",
        search_results=[...]
    )
    
    # Generate answer
    answer = await processor.generate_answer(processed, None)
    print(answer)
    
    # Get prompt
    prompt = processor.build_llm_prompt(processed)
    print(prompt)


# Example 6: Validation
def example_validation():
    """Example of answer validation."""
    service = PerplexityService()
    
    answer = "This is a test answer[1][2]."
    result = service.validate_answer(answer, "general")
    
    if result['valid']:
        print("Answer is valid!")
    else:
        print(f"Found {result['issue_count']} issues:")
        for issue in result['issues']:
            print(f"  - {issue['message']}: {issue['suggestion']}")


# Example 7: Cache and metrics
def example_cache_metrics():
    """Example of using cache and metrics."""
    service = PerplexityService()
    
    # Get cache stats
    cache_stats = service.get_cache_stats()
    print(f"Cache size: {cache_stats['size']}/{cache_stats['max_size']}")
    
    # Get metrics
    metrics = service.get_metrics()
    print(f"Total queries: {metrics['total_queries']}")
    print(f"Average processing time: {metrics['average_processing_time_ms']}ms")
    
    # Clear if needed
    # service.clear_cache()
    # service.clear_metrics()


# Example 8: Different query types
async def example_query_types():
    """Example showing different query types."""
    service = PerplexityService()
    
    queries = [
        ("What is the weather today?", "weather"),
        ("Latest news about AI", "recent_news"),
        ("Who is Albert Einstein?", "people"),
        ("Python code example", "coding"),
        ("Chocolate cake recipe", "cooking_recipes"),
    ]
    
    for query, expected_type in queries:
        processed = service.processor.process_query(query, [])
        print(f"Query: {query}")
        print(f"Detected type: {processed.query_type.value}")
        print(f"Expected: {expected_type}")
        print()


if __name__ == "__main__":
    # Run examples
    asyncio.run(example_basic_query())




