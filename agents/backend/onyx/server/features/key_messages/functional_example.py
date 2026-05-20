from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import asyncio
from typing import List, Dict, Any, Callable
from functools import partial, reduce, compose
from datetime import datetime
from .models import KeyMessageRequest, MessageType, MessageTone
from .service import (
from .utils import (
from typing import Any, List, Dict, Optional
import logging
"""
Functional programming examples for Key Messages feature.
"""

    startup_service, shutdown_service, generate_response, analyze_message,
    generate_batch, ServiceConfig
)
    pipe, compose, filter_with_predicate, map_with_index, group_by,
    sort_by, chunk_list, flatten_list, unique_items, create_pipeline,
    create_conditional_pipeline, create_error_handler, safe_execute
)

# Example 1: Pure Functions for Message Processing
def create_marketing_message(message: str) -> str:
    """Create marketing message."""
    return f"🚀 {message} - Don't miss out!"

def create_educational_message(message: str) -> str:
    """Create educational message."""
    return f"📚 Learn more: {message}"

def create_promotional_message(message: str) -> str:
    """Create promotional message."""
    return f"🎉 Special offer: {message}"

def add_hashtags(message: str, hashtags: List[str]) -> str:
    """Add hashtags to message."""
    hashtag_string = " ".join(f"#{tag}" for tag in hashtags)
    return f"{message} {hashtag_string}"

def add_call_to_action(message: str, cta: str) -> str:
    """Add call to action to message."""
    return f"{message} {cta}"

# Example 2: Function Composition
def create_message_pipeline(message_type: MessageType, hashtags: List[str], cta: str):
    """Create message processing pipeline."""
    # Define message creators based on type
    message_creators = {
        MessageType.MARKETING: create_marketing_message,
        MessageType.EDUCATIONAL: create_educational_message,
        MessageType.PROMOTIONAL: create_promotional_message
    }
    
    # Get the appropriate creator
    creator = message_creators.get(message_type, lambda x: x)
    
    # Compose the pipeline
    pipeline = compose(
        creator,
        partial(add_hashtags, hashtags=hashtags),
        partial(add_call_to_action, cta=cta)
    )
    
    return pipeline

# Example 3: Functional Data Processing
def process_message_batch(messages: List[str], pipeline: Callable[[str], str]) -> List[str]:
    """Process batch of messages using functional approach."""
    return list(map(pipeline, messages))

def filter_messages_by_length(messages: List[str], min_length: int = 10) -> List[str]:
    """Filter messages by minimum length."""
    return filter_with_predicate(lambda msg: len(msg) >= min_length, messages)

def sort_messages_by_length(messages: List[str], reverse: bool = False) -> List[str]:
    """Sort messages by length."""
    return sort_by(messages, key_func=len, reverse=reverse)

def group_messages_by_type(messages: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group messages by type."""
    return group_by(messages, key_func=lambda msg: msg.get('type', 'unknown'))

# Example 4: Error Handling with Functional Approach
def safe_message_processing(message: str, processor: Callable[[str], str], default: str = "") -> str:
    """Safely process message with error handling."""
    return safe_execute(processor, default, message)

def create_resilient_pipeline(*processors: Callable[[str], str]) -> Callable[[str], str]:
    """Create pipeline that continues on errors."""
    def resilient_processor(message: str) -> str:
        result = message
        for processor in processors:
            result = safe_message_processing(result, processor, result)
        return result
    return resilient_processor

# Example 5: Async Functional Processing
async def process_messages_async(messages: List[str], processor: Callable[[str], str]) -> List[str]:
    """Process messages asynchronously."""
    async def process_single(message: str) -> str:
        # Simulate async processing
        await asyncio.sleep(0.1)
        return processor(message)
    
    tasks = [process_single(msg) for msg in messages]
    return await asyncio.gather(*tasks)

async def batch_process_with_chunks(messages: List[str], chunk_size: int = 5) -> List[str]:
    """Process messages in chunks."""
    chunks = chunk_list(messages, chunk_size)
    
    async def process_chunk(chunk: List[str]) -> List[str]:
        return await process_messages_async(chunk, lambda x: x.upper())
    
    chunk_results = await asyncio.gather(*[process_chunk(chunk) for chunk in chunks])
    return flatten_list(chunk_results)

# Example 6: Conditional Processing
def create_conditional_processor(
    condition: Callable[[str], bool],
    true_processor: Callable[[str], str],
    false_processor: Callable[[str], str]
) -> Callable[[str], str]:
    """Create conditional processor."""
    return create_conditional_pipeline(condition, true_processor, false_processor)

def is_long_message(message: str) -> bool:
    """Check if message is long."""
    return len(message) > 50

def shorten_message(message: str) -> str:
    """Shorten long message."""
    return message[:50] + "..."

def enhance_short_message(message: str) -> str:
    """Enhance short message."""
    return f"💡 {message}"

# Example 7: Service Integration with Functional Approach
async def functional_message_generation_example():
    """Example of functional message generation."""
    # Initialize service
    config = ServiceConfig(
        model_name="gpt2",
        max_concurrent_requests=5,
        cache_size=100,
        timeout_seconds=30
    )
    
    try:
        await startup_service(config)
        
        # Create message requests
        requests = [
            KeyMessageRequest(
                message="Our new product is amazing",
                message_type=MessageType.MARKETING,
                tone=MessageTone.ENTHUSIASTIC,
                keywords=["innovation", "amazing", "product"]
            ),
            KeyMessageRequest(
                message="Learn about machine learning",
                message_type=MessageType.EDUCATIONAL,
                tone=MessageTone.PROFESSIONAL,
                keywords=["learning", "machine learning", "education"]
            ),
            KeyMessageRequest(
                message="Special discount available",
                message_type=MessageType.PROMOTIONAL,
                tone=MessageTone.URGENT,
                keywords=["discount", "special", "offer"]
            )
        ]
        
        # Process requests functionally
        results = await asyncio.gather(*[
            generate_response(req) for req in requests
        ], return_exceptions=True)
        
        # Filter successful results
        successful_results = filter_with_predicate(
            lambda r: isinstance(r, dict) and r.get('success', False),
            results
        )
        
        # Extract response data
        response_data = list(map(
            lambda r: r.get('data', {}),
            successful_results
        ))
        
        # Group by message type
        grouped_data = group_by(
            response_data,
            key_func=lambda d: d.get('message_type', 'unknown')
        )
        
        print("Functional processing results:")
        for msg_type, responses in grouped_data.items():
            print(f"\n{msg_type}:")
            for response in responses:
                print(f"  - {response.get('response', '')[:100]}...")
        
        return grouped_data
        
    finally:
        await shutdown_service()

# Example 8: Advanced Functional Patterns
def create_message_analyzer() -> Callable[[str], Dict[str, Any]]:
    """Create message analyzer using functional composition."""
    def count_words(text: str) -> int:
        return len(text.split())
    
    def count_characters(text: str) -> int:
        return len(text)
    
    def calculate_avg_word_length(text: str) -> float:
        words = text.split()
        return sum(len(word) for word in words) / len(words) if words else 0
    
    def detect_sentiment(text: str) -> str:
        positive_words = ['amazing', 'great', 'excellent', 'wonderful']
        negative_words = ['terrible', 'awful', 'bad', 'horrible']
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            return 'positive'
        elif negative_count > positive_count:
            return 'negative'
        else:
            return 'neutral'
    
    # Compose the analyzer
    def analyze_message(text: str) -> Dict[str, Any]:
        return {
            'word_count': count_words(text),
            'character_count': count_characters(text),
            'avg_word_length': calculate_avg_word_length(text),
            'sentiment': detect_sentiment(text),
            'analyzed_at': datetime.now().isoformat()
        }
    
    return analyze_message

# Example 9: Pipeline with Side Effects
def create_logging_pipeline(base_pipeline: Callable[[str], str]) -> Callable[[str], str]:
    """Add logging to pipeline."""
    def logging_pipeline(message: str) -> str:
        print(f"Processing message: {message[:50]}...")
        result = base_pipeline(message)
        print(f"Result: {result[:50]}...")
        return result
    return logging_pipeline

# Example 10: Functional Testing
def create_message_validator() -> Callable[[str], bool]:
    """Create message validator."""
    def has_minimum_length(text: str) -> bool:
        return len(text) >= 10
    
    def has_maximum_length(text: str) -> bool:
        return len(text) <= 1000
    
    def has_no_profanity(text: str) -> bool:
        profanity_words = ['bad_word1', 'bad_word2']  # Example
        return not any(word in text.lower() for word in profanity_words)
    
    def validate_message(text: str) -> bool:
        return all([
            has_minimum_length(text),
            has_maximum_length(text),
            has_no_profanity(text)
        ])
    
    return validate_message

# Usage Examples
async def run_functional_examples():
    """Run all functional examples."""
    print("=== Functional Programming Examples ===\n")
    
    # Example 1: Basic function composition
    print("1. Basic Function Composition:")
    pipeline = create_message_pipeline(
        MessageType.MARKETING,
        hashtags=["innovation", "product"],
        cta="Learn more!"
    )
    result = pipeline("Our new AI solution")
    print(f"   Input: 'Our new AI solution'")
    print(f"   Output: '{result}'\n")
    
    # Example 2: Data processing
    print("2. Data Processing:")
    messages = [
        "Short message",
        "This is a longer message that should be processed",
        "Another short one",
        "This is a very long message that exceeds the typical length limit"
    ]
    
    filtered_messages = filter_messages_by_length(messages, min_length=20)
    sorted_messages = sort_messages_by_length(filtered_messages, reverse=True)
    
    print(f"   Original messages: {len(messages)}")
    print(f"   Filtered messages: {len(filtered_messages)}")
    print(f"   Sorted messages: {sorted_messages}\n")
    
    # Example 3: Conditional processing
    print("3. Conditional Processing:")
    conditional_processor = create_conditional_processor(
        is_long_message,
        shorten_message,
        enhance_short_message
    )
    
    test_messages = [
        "Short message",
        "This is a very long message that should be shortened because it exceeds the length limit"
    ]
    
    for msg in test_messages:
        processed = conditional_processor(msg)
        print(f"   '{msg}' -> '{processed}'\n")
    
    # Example 4: Message analysis
    print("4. Message Analysis:")
    analyzer = create_message_analyzer()
    analysis = analyzer("This is an amazing product that will revolutionize the industry!")
    print(f"   Analysis: {analysis}\n")
    
    # Example 5: Service integration
    print("5. Service Integration:")
    try:
        results = await functional_message_generation_example()
        print(f"   Generated {sum(len(responses) for responses in results.values())} messages")
        print(f"   Grouped by type: {list(results.keys())}\n")
    except Exception as e:
        print(f"   Service example failed: {e}\n")

match __name__:
    case "__main__":
    asyncio.run(run_functional_examples()) 