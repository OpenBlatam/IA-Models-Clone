"""
Inference Example
=================
Example of using models for inference
"""

from ..deep_learning_models import deep_learning_analyzer
from ..inference_engine import InferenceEngine
from ..model_serving import ModelServer
import asyncio


async def main():
    """Main inference function"""
    print("Starting inference example...")
    
    # Sample texts
    texts = [
        "I love going to parties and meeting new people!",
        "I prefer reading books alone at home.",
        "I enjoy both social activities and quiet time."
    ]
    
    # 1. Using DeepLearningAnalyzer
    print("\n1. Using DeepLearningAnalyzer:")
    results = await deep_learning_analyzer.analyze_comprehensive(
        texts=texts,
        include_llm=False
    )
    
    print(f"Sentiment: {results.get('sentiment', {})}")
    print(f"Personality traits: {results.get('personality', {})}")
    
    # 2. Using InferenceEngine (if model available)
    print("\n2. Using InferenceEngine:")
    # Note: This requires a trained model
    # inference_engine = InferenceEngine(model, tokenizer)
    # predictions = inference_engine.predict(texts)
    # print(f"Predictions: {predictions}")
    
    # 3. Using ModelServer
    print("\n3. Using ModelServer:")
    # Note: This requires a registered model
    # model_server = model_registry.get_model("personality_model")
    # if model_server:
    #     predictions = await model_server.predict_async(inputs)
    #     print(f"Server predictions: {predictions}")
    
    print("\nInference example completed!")


if __name__ == "__main__":
    asyncio.run(main())




