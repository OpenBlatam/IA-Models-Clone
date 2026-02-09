"""
Recommender Module

Intelligent configuration recommendations based on use case.
"""

from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class ConfigRecommender:
    """
    Recommends configurations based on use case and requirements.
    """
    
    @staticmethod
    def recommend_for_use_case(
        use_case: str,
        requirements: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Recommend configuration for a specific use case.
        
        Args:
            use_case: Use case (e.g., "text_classification", "image_generation")
            requirements: Additional requirements
            
        Returns:
            Recommended configuration
        """
        requirements = requirements or {}
        
        use_case_configs = {
            "text_classification": {
                "framework": "pytorch",
                "model_type": "transformer",
                "batch_size": 32,
                "learning_rate": 1e-4,
                "num_epochs": 10,
                "mixed_precision": True,
                "early_stopping": True
            },
            "text_generation": {
                "framework": "pytorch",
                "model_type": "llm",
                "batch_size": 8,
                "learning_rate": 5e-5,
                "num_epochs": 3,
                "mixed_precision": True,
                "gradient_clipping": True,
                "gradient_accumulation_steps": 4
            },
            "image_classification": {
                "framework": "pytorch",
                "model_type": "cnn",
                "batch_size": 64,
                "learning_rate": 1e-3,
                "num_epochs": 20,
                "mixed_precision": True,
                "data_augmentation": True
            },
            "image_generation": {
                "framework": "pytorch",
                "model_type": "diffusion",
                "batch_size": 4,
                "learning_rate": 1e-4,
                "num_epochs": 50,
                "mixed_precision": True,
                "gradient_checkpointing": True
            },
            "sentiment_analysis": {
                "framework": "pytorch",
                "model_type": "transformer",
                "batch_size": 32,
                "learning_rate": 2e-5,
                "num_epochs": 5,
                "mixed_precision": True
            },
            "named_entity_recognition": {
                "framework": "pytorch",
                "model_type": "transformer",
                "batch_size": 16,
                "learning_rate": 3e-5,
                "num_epochs": 10,
                "mixed_precision": True
            },
            "question_answering": {
                "framework": "pytorch",
                "model_type": "transformer",
                "batch_size": 16,
                "learning_rate": 3e-5,
                "num_epochs": 5,
                "mixed_precision": True,
                "gradient_clipping": True
            },
            "machine_translation": {
                "framework": "pytorch",
                "model_type": "transformer",
                "batch_size": 32,
                "learning_rate": 1e-4,
                "num_epochs": 10,
                "mixed_precision": True,
                "label_smoothing": 0.1
            },
            "speech_recognition": {
                "framework": "pytorch",
                "model_type": "rnn",
                "batch_size": 16,
                "learning_rate": 1e-3,
                "num_epochs": 20,
                "mixed_precision": True
            },
            "recommendation_system": {
                "framework": "pytorch",
                "model_type": "transformer",
                "batch_size": 64,
                "learning_rate": 1e-3,
                "num_epochs": 15,
                "mixed_precision": True
            }
        }
        
        if use_case not in use_case_configs:
            logger.warning(f"Unknown use case: {use_case}, using defaults")
            base_config = {
                "framework": "pytorch",
                "model_type": "transformer",
                "batch_size": 32,
                "learning_rate": 1e-4,
                "num_epochs": 10
            }
        else:
            base_config = use_case_configs[use_case].copy()
        
        # Override with requirements
        base_config.update(requirements)
        
        return base_config
    
    @staticmethod
    def recommend_based_on_data(
        data_size: str,
        data_type: str,
        task_type: str
    ) -> Dict[str, Any]:
        """
        Recommend configuration based on data characteristics.
        
        Args:
            data_size: Size of dataset (small, medium, large, very_large)
            data_type: Type of data (text, image, audio, video)
            task_type: Type of task (classification, generation, regression)
            
        Returns:
            Recommended configuration
        """
        config = {
            "framework": "pytorch",
            "mixed_precision": True
        }
        
        # Adjust batch size based on data size
        batch_sizes = {
            "small": 64,
            "medium": 32,
            "large": 16,
            "very_large": 8
        }
        config["batch_size"] = batch_sizes.get(data_size, 32)
        
        # Adjust model type based on data type
        model_types = {
            "text": "transformer",
            "image": "cnn",
            "audio": "rnn",
            "video": "cnn"
        }
        config["model_type"] = model_types.get(data_type, "transformer")
        
        # Adjust epochs based on data size
        epochs = {
            "small": 50,
            "medium": 20,
            "large": 10,
            "very_large": 5
        }
        config["num_epochs"] = epochs.get(data_size, 10)
        
        # Adjust learning rate based on task
        learning_rates = {
            "classification": 1e-4,
            "generation": 5e-5,
            "regression": 1e-3
        }
        config["learning_rate"] = learning_rates.get(task_type, 1e-4)
        
        return config
    
    @staticmethod
    def recommend_for_budget(
        budget_type: str,
        time_budget: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Recommend configuration based on budget constraints.
        
        Args:
            budget_type: Type of budget (compute, time, memory)
            time_budget: Time budget (fast, medium, slow)
            
        Returns:
            Recommended configuration
        """
        config = {
            "framework": "pytorch"
        }
        
        if budget_type == "compute":
            # Minimize compute
            config.update({
                "batch_size": 8,
                "mixed_precision": True,
                "gradient_checkpointing": True,
                "num_epochs": 5
            })
        elif budget_type == "time":
            # Minimize time
            if time_budget == "fast":
                config.update({
                    "batch_size": 64,
                    "mixed_precision": True,
                    "num_epochs": 3,
                    "early_stopping": True
                })
            else:
                config.update({
                    "batch_size": 32,
                    "mixed_precision": True,
                    "num_epochs": 10
                })
        elif budget_type == "memory":
            # Minimize memory
            config.update({
                "batch_size": 4,
                "mixed_precision": True,
                "gradient_checkpointing": True,
                "gradient_accumulation_steps": 8
            })
        
        return config


def recommend_config(
    use_case: Optional[str] = None,
    data_size: Optional[str] = None,
    data_type: Optional[str] = None,
    task_type: Optional[str] = None,
    budget_type: Optional[str] = None,
    requirements: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Get configuration recommendation based on multiple factors.
    
    Args:
        use_case: Use case name
        data_size: Size of dataset
        data_type: Type of data
        task_type: Type of task
        budget_type: Budget constraint
        requirements: Additional requirements
        
    Returns:
        Recommended configuration
    """
    recommender = ConfigRecommender()
    config = {}
    
    if use_case:
        config = recommender.recommend_for_use_case(use_case, requirements)
    elif data_size and data_type and task_type:
        config = recommender.recommend_based_on_data(data_size, data_type, task_type)
    elif budget_type:
        config = recommender.recommend_for_budget(budget_type)
    else:
        # Default recommendation
        config = {
            "framework": "pytorch",
            "model_type": "transformer",
            "batch_size": 32,
            "learning_rate": 1e-4,
            "num_epochs": 10
        }
    
    if requirements:
        config.update(requirements)
    
    return config















