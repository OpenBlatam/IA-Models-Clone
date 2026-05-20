"""
Example: Using Deep Learning Models with Addition Removal AI
"""

import torch
from addition_removal_ai import (
    ContentEditor,
    EnhancedAIEngine,
    create_transformer_analyzer,
    create_text_generator,
    create_gradio_app
)


def main():
    """Example usage of deep learning features"""
    
    print("=== Addition Removal AI with Deep Learning ===\n")
    
    # Initialize editor
    editor = ContentEditor()
    
    # Initialize enhanced AI engine
    print("Initializing Enhanced AI Engine...")
    ai_engine = EnhancedAIEngine(
        config={
            "use_transformer_analyzer": True,
            "use_sentiment_analyzer": True,
            "use_ner_analyzer": True,
            "use_text_generator": True,
            "use_t5_generator": False,  # Optional
            "transformer_model": "bert-base-uncased",
            "text_model": "gpt2"
        },
        use_gpu=torch.cuda.is_available()
    )
    
    # Example content
    content = """
    Artificial intelligence is transforming the way we work and live.
    Machine learning algorithms can now understand and generate human-like text.
    This technology has applications in many fields including healthcare, education, and business.
    """
    
    print("\n1. Content Analysis with Transformers:")
    print("-" * 50)
    analysis = ai_engine.analyze_content(content)
    print(f"Sentiment: {analysis.get('sentiment', {})}")
    print(f"Entities: {analysis.get('entities', [])}")
    
    print("\n2. Semantic Similarity:")
    print("-" * 50)
    text1 = "AI is revolutionizing technology"
    text2 = "Artificial intelligence is changing tech"
    similarity = ai_engine.calculate_similarity(text1, text2)
    print(f"Similarity between '{text1}' and '{text2}': {similarity:.3f}")
    
    print("\n3. Content Generation:")
    print("-" * 50)
    prompt = "The future of artificial intelligence"
    generated = ai_engine.generate_content(prompt, max_length=50)
    print(f"Generated: {generated}")
    
    print("\n4. Content Suggestions:")
    print("-" * 50)
    suggestions = ai_engine.suggest_additions(content)
    for i, suggestion in enumerate(suggestions[:3], 1):
        print(f"Suggestion {i}: {suggestion[:100]}...")
    
    print("\n5. Content Optimization:")
    print("-" * 50)
    optimized = ai_engine.optimize_content(content)
    print(f"Optimized: {optimized[:200]}...")
    
    print("\n6. Summarization:")
    print("-" * 50)
    summary = ai_engine.summarize(content)
    print(f"Summary: {summary}")
    
    print("\n7. Using Editor with AI:")
    print("-" * 50)
    result = editor.add(
        content=content,
        addition="Deep learning is a subset of machine learning.",
        position="end"
    )
    print(f"Added content successfully: {result.get('success', False)}")
    
    print("\n=== Example Complete ===")


def launch_gradio():
    """Launch Gradio interface"""
    editor = ContentEditor()
    ai_engine = EnhancedAIEngine(use_gpu=torch.cuda.is_available())
    
    app = create_gradio_app(editor, ai_engine)
    app.launch(server_port=7860)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "gradio":
        launch_gradio()
    else:
        main()

