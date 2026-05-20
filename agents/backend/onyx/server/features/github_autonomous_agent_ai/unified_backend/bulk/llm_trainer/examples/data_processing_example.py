"""
Data Processing Example
======================

Example showing how to use data validators and processors independently.

Author: BUL System
Date: 2024
"""

import json
from llm_trainer.data import DatasetValidator, DatasetProcessor

# Example: Validate and process dataset before training
def prepare_dataset(file_path: str):
    """Prepare dataset with validation and processing."""
    
    # 1. Validate file
    validator = DatasetValidator()
    is_valid, errors, data = validator.validate_file(file_path)
    
    if not is_valid:
        print("Validation errors:")
        for error in errors:
            print(f"  - {error}")
        return None
    
    print("✓ Dataset structure is valid")
    
    # 2. Validate quality
    quality = validator.validate_quality(data)
    print(f"Quality score: {quality['quality_score']:.1f}/100")
    if quality['warnings']:
        for warning in quality['warnings']:
            print(f"  Warning: {warning}")
    
    # 3. Process dataset
    processor = DatasetProcessor()
    
    # Clean
    cleaned = processor.clean_dataset(data)
    
    # Filter by length
    filtered = processor.filter_by_length(
        cleaned,
        min_prompt_length=10,
        max_prompt_length=500,
        min_response_length=20,
        max_response_length=1000
    )
    
    print(f"Processed: {len(data)} -> {len(filtered)} examples")
    
    # Save processed dataset
    with open("processed_dataset.json", "w") as f:
        json.dump(filtered, f, indent=2, ensure_ascii=False)
    
    return filtered


if __name__ == "__main__":
    dataset = prepare_dataset("raw_data.json")
    if dataset:
        print("Dataset ready for training!")

