"""
Modular Usage Example
=====================

Example showing how to use components independently in a modular way.

Author: BUL System
Date: 2024
"""

from llm_trainer import (
    DeviceManager,
    DatasetLoader,
    TokenizerUtils,
    ModelFactory,
    TrainingConfig,
    DatasetValidator,
    DatasetProcessor,
)

# Example: Build trainer manually using individual components
def build_trainer_manually():
    """Build trainer step by step using modular components."""
    
    # 1. Setup device
    device_manager = DeviceManager()
    print(f"Device: {device_manager.get_device_summary()}")
    
    # 2. Validate and load dataset
    validator = DatasetValidator()
    is_valid, errors, data = validator.validate_file("data/training.json")
    if not is_valid:
        raise ValueError(f"Dataset invalid: {errors}")
    
    processor = DatasetProcessor()
    processed_data = processor.clean_dataset(data)
    
    dataset_loader = DatasetLoader("data/training.json")
    dataset = dataset_loader.prepare_dataset(processed_data)
    
    # 3. Setup tokenizer
    tokenizer_utils = TokenizerUtils("gpt2", model_type="causal")
    tokenizer = tokenizer_utils.get_tokenizer()
    
    # 4. Create model
    model_factory = ModelFactory(device_manager)
    model = model_factory.create_causal_model("gpt2", tokenizer_vocab_size=len(tokenizer))
    
    # 5. Setup training config
    training_config = TrainingConfig(
        output_dir="./checkpoints",
        learning_rate=3e-5,
        num_train_epochs=3,
        batch_size=8,
        device_manager=device_manager
    )
    
    # 6. Now you can create your own trainer or use transformers Trainer directly
    from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
    
    trainer = Trainer(
        model=model,
        args=training_config.get_training_args(),
        train_dataset=dataset["train"],
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    
    return trainer


if __name__ == "__main__":
    trainer = build_trainer_manually()
    trainer.train()

