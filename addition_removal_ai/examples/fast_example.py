"""
Fast Usage Example - Addition Removal AI
"""

import time
from addition_removal_ai import (
    create_fast_editor,
    create_fast_ai_engine
)


def main():
    """Demonstrate fast operations"""
    
    print("=== Fast Addition Removal AI ===\n")
    
    # Create fast editor
    print("Initializing Fast Editor...")
    editor = create_fast_editor(batch_size=32)
    
    # Create fast AI engine
    print("Initializing Fast AI Engine...")
    ai_engine = create_fast_ai_engine(batch_size=32)
    
    # Example content
    content = "Artificial intelligence is transforming technology."
    
    print("\n1. Fast Single Operations:")
    print("-" * 50)
    
    # Fast add
    start = time.time()
    result = editor.add(content, "Machine learning is a key component.", "end")
    add_time = (time.time() - start) * 1000
    print(f"Add operation: {add_time:.2f}ms")
    
    # Fast analyze
    start = time.time()
    analysis = ai_engine.analyze_content_fast(content)
    analyze_time = (time.time() - start) * 1000
    print(f"Analyze operation: {analyze_time:.2f}ms")
    
    print("\n2. Fast Batch Operations:")
    print("-" * 50)
    
    # Prepare batch
    contents = [f"Content {i}: AI is amazing." for i in range(100)]
    additions = [f"Addition {i}" for i in range(100)]
    
    # Batch add
    operations = [
        {"content": c, "addition": a, "position": "end"}
        for c, a in zip(contents, additions)
    ]
    
    start = time.time()
    results = editor.add_batch(operations)
    batch_add_time = (time.time() - start) * 1000
    print(f"Batch add (100 items): {batch_add_time:.2f}ms")
    print(f"Average per item: {batch_add_time/100:.2f}ms")
    
    # Batch analyze
    start = time.time()
    analyses = ai_engine.analyze_batch(contents)
    batch_analyze_time = (time.time() - start) * 1000
    print(f"Batch analyze (100 items): {batch_analyze_time:.2f}ms")
    print(f"Average per item: {batch_analyze_time/100:.2f}ms")
    
    # Batch generate
    prompts = [f"Write about AI topic {i}" for i in range(50)]
    start = time.time()
    generated = ai_engine.generate_batch(prompts, max_length=50)
    batch_generate_time = (time.time() - start) * 1000
    print(f"Batch generate (50 items): {batch_generate_time:.2f}ms")
    print(f"Average per item: {batch_generate_time/50:.2f}ms")
    
    print("\n3. Performance Summary:")
    print("-" * 50)
    print(f"Single add: {add_time:.2f}ms")
    print(f"Single analyze: {analyze_time:.2f}ms")
    print(f"Batch add (100): {batch_add_time:.2f}ms ({batch_add_time/100:.2f}ms/item)")
    print(f"Batch analyze (100): {batch_analyze_time:.2f}ms ({batch_analyze_time/100:.2f}ms/item)")
    print(f"Batch generate (50): {batch_generate_time:.2f}ms ({batch_generate_time/50:.2f}ms/item)")
    
    print("\n=== Example Complete ===")


if __name__ == "__main__":
    main()

