"""
Example Script: Using CustomLLMTrainer
======================================

This script demonstrates how to use the CustomLLMTrainer class
to fine-tune a language model on a custom dataset.

Usage:
    python example_llm_training.py
"""

import json
from pathlib import Path
from llm_trainer import CustomLLMTrainer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_example_dataset(output_path: str = "training_dataset.json") -> None:
    """Create an example dataset for training."""
    dataset = [
        {
            "prompt": "What is artificial intelligence?",
            "response": "Artificial intelligence (AI) is the simulation of human intelligence in machines that are programmed to think and learn like humans."
        },
        {
            "prompt": "Explain deep learning.",
            "response": "Deep learning is a subset of machine learning that uses neural networks with multiple layers to learn and make decisions from data."
        },
        {
            "prompt": "What is natural language processing?",
            "response": "Natural language processing (NLP) is a branch of AI that helps computers understand, interpret, and manipulate human language."
        },
        {
            "prompt": "Describe computer vision.",
            "response": "Computer vision is a field of AI that trains computers to interpret and understand the visual world using digital images and videos."
        },
        {
            "prompt": "What is reinforcement learning?",
            "response": "Reinforcement learning is a type of machine learning where an agent learns to make decisions by taking actions in an environment to maximize cumulative reward."
        }
    ]
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Example dataset created at: {output_path}")
    return output_path


def main():
    """Main function to demonstrate CustomLLMTrainer usage."""
    
    # Create example dataset
    dataset_path = create_example_dataset("training_dataset.json")
    
    # Initialize trainer
    logger.info("Initializing CustomLLMTrainer...")
    trainer = CustomLLMTrainer(
        model_name="gpt2",  # Small model for demonstration
        dataset_path=dataset_path,
        output_dir="./llm_checkpoints",
        learning_rate=3e-5,
        num_train_epochs=3,
        batch_size=8,
        max_length=512,
        logging_steps=5,
        save_steps=50,
        gradient_accumulation_steps=1,
    )
    
    # Train the model
    logger.info("Starting training...")
    try:
        trainer.train()
        logger.info("Training completed successfully!")
        
        # Example: Generate predictions
        logger.info("\nGenerating sample predictions...")
        test_prompts = [
            "What is machine learning?",
            "Explain neural networks."
        ]
        
        predictions = trainer.predict(test_prompts, max_new_tokens=50)
        for prompt, prediction in zip(test_prompts, predictions):
            logger.info(f"\nPrompt: {prompt}")
            logger.info(f"Response: {prediction}")
            
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()

