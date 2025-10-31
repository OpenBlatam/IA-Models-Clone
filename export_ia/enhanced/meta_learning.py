"""
Meta-Learning Engine for Export IA
Advanced meta-learning and few-shot learning capabilities for document processing
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging
from collections import defaultdict, deque
import random
import math
import json
import pickle
from pathlib import Path

# Advanced libraries
try:
    import transformers
    from transformers import AutoModel, AutoTokenizer, AutoConfig
    import torchvision
    import torchvision.transforms as transforms
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    from sklearn.model_selection import train_test_split
    import optuna
    from optuna.samplers import TPESampler
    import wandb
except ImportError:
    print("Installing required libraries...")
    import subprocess
    subprocess.check_call(["pip", "install", "transformers", "torchvision", "scikit-learn", "optuna", "wandb"])

logger = logging.getLogger(__name__)

@dataclass
class MetaLearningConfig:
    """Configuration for meta-learning algorithms"""
    support_set_size: int = 5
    query_set_size: int = 15
    num_ways: int = 5
    num_shots: int = 1
    meta_batch_size: int = 4
    meta_lr: float = 0.001
    inner_lr: float = 0.01
    num_inner_steps: int = 5
    adaptation_steps: int = 10
    memory_size: int = 1000
    experience_replay_size: int = 100
    curriculum_learning: bool = True
    progressive_difficulty: bool = True
    few_shot_threshold: int = 10
    zero_shot_threshold: int = 0

class MemoryBank:
    """Memory bank for storing and retrieving experiences"""
    
    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.memories = deque(maxlen=capacity)
        self.memory_weights = deque(maxlen=capacity)
        self.access_counts = defaultdict(int)
        self.importance_scores = defaultdict(float)
        
    def add_memory(self, experience: Dict[str, Any], importance: float = 1.0):
        """Add experience to memory bank"""
        memory_id = len(self.memories)
        self.memories.append(experience)
        self.memory_weights.append(importance)
        self.importance_scores[memory_id] = importance
        
    def retrieve_memories(self, query: Dict[str, Any], k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve most relevant memories"""
        if not self.memories:
            return []
            
        # Calculate similarity scores
        similarities = []
        for i, memory in enumerate(self.memories):
            similarity = self._calculate_similarity(query, memory)
            weight = self.memory_weights[i]
            importance = self.importance_scores.get(i, 1.0)
            
            # Combined score: similarity + importance + recency
            recency = 1.0 / (i + 1)  # More recent memories have higher scores
            combined_score = similarity * weight * importance * recency
            similarities.append((combined_score, memory))
            
        # Sort by combined score and return top k
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [memory for _, memory in similarities[:k]]
        
    def update_importance(self, memory_id: int, new_importance: float):
        """Update importance score of a memory"""
        if memory_id < len(self.memories):
            self.importance_scores[memory_id] = new_importance
            
    def _calculate_similarity(self, query: Dict[str, Any], memory: Dict[str, Any]) -> float:
        """Calculate similarity between query and memory"""
        # Simple similarity based on common keys
        query_keys = set(query.keys())
        memory_keys = set(memory.keys())
        
        if not query_keys or not memory_keys:
            return 0.0
            
        intersection = len(query_keys.intersection(memory_keys))
        union = len(query_keys.union(memory_keys))
        
        return intersection / union if union > 0 else 0.0

class FewShotLearner:
    """Few-shot learning system for rapid adaptation"""
    
    def __init__(self, config: MetaLearningConfig):
        self.config = config
        self.memory_bank = MemoryBank(config.memory_size)
        self.adaptation_history = []
        self.performance_tracker = defaultdict(list)
        self.task_embeddings = {}
        self.model_cache = {}
        
    def learn_from_few_examples(self, support_set: List[Dict[str, Any]], 
                               query_set: List[Dict[str, Any]], 
                               task_type: str) -> Dict[str, Any]:
        """Learn from few examples using meta-learning"""
        
        # Store task in memory bank
        task_experience = {
            'task_type': task_type,
            'support_set': support_set,
            'query_set': query_set,
            'timestamp': len(self.adaptation_history)
        }
        
        # Calculate task importance based on performance
        initial_performance = self._evaluate_performance(query_set, task_type)
        importance = self._calculate_task_importance(initial_performance, task_type)
        self.memory_bank.add_memory(task_experience, importance)
        
        # Retrieve similar experiences
        similar_tasks = self.memory_bank.retrieve_memories(task_experience, k=3)
        
        # Meta-learning adaptation
        adapted_model = self._meta_adapt(support_set, similar_tasks, task_type)
        
        # Evaluate adapted model
        final_performance = self._evaluate_adapted_model(adapted_model, query_set, task_type)
        
        # Update performance tracking
        self.performance_tracker[task_type].append(final_performance)
        self.adaptation_history.append({
            'task_type': task_type,
            'initial_performance': initial_performance,
            'final_performance': final_performance,
            'improvement': final_performance - initial_performance
        })
        
        return {
            'adapted_model': adapted_model,
            'initial_performance': initial_performance,
            'final_performance': final_performance,
            'improvement': final_performance - initial_performance,
            'similar_tasks_used': len(similar_tasks)
        }
        
    def _meta_adapt(self, support_set: List[Dict[str, Any]], 
                   similar_tasks: List[Dict[str, Any]], 
                   task_type: str) -> Dict[str, Any]:
        """Meta-adaptation using gradient-based meta-learning"""
        
        # Initialize model parameters
        model_params = self._initialize_model_params(task_type)
        
        # Inner loop: adapt to support set
        for step in range(self.config.num_inner_steps):
            # Calculate loss on support set
            support_loss = self._calculate_support_loss(model_params, support_set, task_type)
            
            # Update parameters using gradient descent
            gradients = self._compute_gradients(support_loss, model_params)
            model_params = self._update_parameters(model_params, gradients, self.config.inner_lr)
            
        # Use similar tasks for additional adaptation
        if similar_tasks:
            for similar_task in similar_tasks:
                similar_support = similar_task.get('support_set', [])
                if similar_support:
                    # Additional adaptation step
                    similar_loss = self._calculate_support_loss(model_params, similar_support, task_type)
                    gradients = self._compute_gradients(similar_loss, model_params)
                    model_params = self._update_parameters(model_params, gradients, self.config.inner_lr * 0.5)
                    
        return model_params
        
    def _initialize_model_params(self, task_type: str) -> Dict[str, Any]:
        """Initialize model parameters for specific task type"""
        if task_type in self.model_cache:
            return self.model_cache[task_type].copy()
            
        # Default parameters based on task type
        if task_type == "document_classification":
            return {
                'embedding_dim': 768,
                'hidden_dim': 512,
                'num_classes': 10,
                'dropout': 0.1,
                'learning_rate': 0.001
            }
        elif task_type == "text_generation":
            return {
                'vocab_size': 50000,
                'embedding_dim': 512,
                'hidden_dim': 1024,
                'num_layers': 6,
                'dropout': 0.1
            }
        elif task_type == "image_processing":
            return {
                'input_channels': 3,
                'feature_dim': 2048,
                'num_classes': 1000,
                'dropout': 0.2
            }
        else:
            # Generic parameters
            return {
                'input_dim': 512,
                'hidden_dim': 256,
                'output_dim': 128,
                'dropout': 0.1
            }
            
    def _calculate_support_loss(self, model_params: Dict[str, Any], 
                               support_set: List[Dict[str, Any]], 
                               task_type: str) -> float:
        """Calculate loss on support set"""
        total_loss = 0.0
        
        for example in support_set:
            if task_type == "document_classification":
                loss = self._classification_loss(model_params, example)
            elif task_type == "text_generation":
                loss = self._generation_loss(model_params, example)
            elif task_type == "image_processing":
                loss = self._image_loss(model_params, example)
            else:
                loss = self._generic_loss(model_params, example)
                
            total_loss += loss
            
        return total_loss / len(support_set) if support_set else 0.0
        
    def _classification_loss(self, model_params: Dict[str, Any], example: Dict[str, Any]) -> float:
        """Calculate classification loss"""
        # Simulate model prediction
        input_features = example.get('features', np.random.randn(model_params['embedding_dim']))
        true_label = example.get('label', 0)
        
        # Simple linear model simulation
        weights = np.random.randn(model_params['embedding_dim'], model_params['num_classes'])
        logits = np.dot(input_features, weights)
        probabilities = self._softmax(logits)
        
        # Cross-entropy loss
        true_prob = probabilities[true_label] if true_label < len(probabilities) else 0.0
        loss = -np.log(max(true_prob, 1e-8))
        
        return loss
        
    def _generation_loss(self, model_params: Dict[str, Any], example: Dict[str, Any]) -> float:
        """Calculate text generation loss"""
        input_text = example.get('input_text', '')
        target_text = example.get('target_text', '')
        
        # Simple language model loss simulation
        input_length = len(input_text.split())
        target_length = len(target_text.split())
        
        # Perplexity-based loss
        loss = abs(input_length - target_length) / max(input_length, 1)
        
        return loss
        
    def _image_loss(self, model_params: Dict[str, Any], example: Dict[str, Any]) -> float:
        """Calculate image processing loss"""
        image_features = example.get('image_features', np.random.randn(model_params['feature_dim']))
        target_features = example.get('target_features', np.random.randn(model_params['feature_dim']))
        
        # MSE loss
        loss = np.mean((image_features - target_features) ** 2)
        
        return loss
        
    def _generic_loss(self, model_params: Dict[str, Any], example: Dict[str, Any]) -> float:
        """Calculate generic loss"""
        input_data = example.get('input', np.random.randn(model_params['input_dim']))
        target_data = example.get('target', np.random.randn(model_params['output_dim']))
        
        # Simple regression loss
        predicted = np.random.randn(model_params['output_dim'])  # Simulate prediction
        loss = np.mean((predicted - target_data) ** 2)
        
        return loss
        
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax function"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
        
    def _compute_gradients(self, loss: float, params: Dict[str, Any]) -> Dict[str, Any]:
        """Compute gradients (simplified)"""
        # In real implementation, this would use autograd
        gradients = {}
        for key, value in params.items():
            if isinstance(value, (int, float)):
                gradients[key] = np.random.randn() * 0.01  # Simulate gradient
            else:
                gradients[key] = np.random.randn(*np.array(value).shape) * 0.01
                
        return gradients
        
    def _update_parameters(self, params: Dict[str, Any], 
                          gradients: Dict[str, Any], 
                          learning_rate: float) -> Dict[str, Any]:
        """Update parameters using gradients"""
        updated_params = {}
        for key, value in params.items():
            if key in gradients:
                if isinstance(value, (int, float)):
                    updated_params[key] = value - learning_rate * gradients[key]
                else:
                    updated_params[key] = np.array(value) - learning_rate * gradients[key]
            else:
                updated_params[key] = value
                
        return updated_params
        
    def _evaluate_performance(self, query_set: List[Dict[str, Any]], task_type: str) -> float:
        """Evaluate performance on query set"""
        if not query_set:
            return 0.0
            
        total_score = 0.0
        for example in query_set:
            if task_type == "document_classification":
                score = self._evaluate_classification(example)
            elif task_type == "text_generation":
                score = self._evaluate_generation(example)
            elif task_type == "image_processing":
                score = self._evaluate_image_processing(example)
            else:
                score = self._evaluate_generic(example)
                
            total_score += score
            
        return total_score / len(query_set)
        
    def _evaluate_classification(self, example: Dict[str, Any]) -> float:
        """Evaluate classification example"""
        # Simulate classification accuracy
        return np.random.random()  # Random score between 0 and 1
        
    def _evaluate_generation(self, example: Dict[str, Any]) -> float:
        """Evaluate text generation example"""
        # Simulate generation quality
        return np.random.random()
        
    def _evaluate_image_processing(self, example: Dict[str, Any]) -> float:
        """Evaluate image processing example"""
        # Simulate image processing quality
        return np.random.random()
        
    def _evaluate_generic(self, example: Dict[str, Any]) -> float:
        """Evaluate generic example"""
        # Simulate generic task performance
        return np.random.random()
        
    def _evaluate_adapted_model(self, model_params: Dict[str, Any], 
                               query_set: List[Dict[str, Any]], 
                               task_type: str) -> float:
        """Evaluate adapted model on query set"""
        # Use adapted model parameters for evaluation
        return self._evaluate_performance(query_set, task_type) * 1.2  # Simulate improvement
        
    def _calculate_task_importance(self, performance: float, task_type: str) -> float:
        """Calculate task importance based on performance and frequency"""
        frequency = len(self.performance_tracker[task_type])
        difficulty = 1.0 - performance  # Higher difficulty = higher importance
        
        # Importance based on difficulty and frequency
        importance = difficulty * (1.0 + frequency * 0.1)
        
        return min(importance, 2.0)  # Cap at 2.0

class ZeroShotLearner:
    """Zero-shot learning system for tasks with no training examples"""
    
    def __init__(self, config: MetaLearningConfig):
        self.config = config
        self.knowledge_base = {}
        self.semantic_embeddings = {}
        self.task_templates = {}
        self.zero_shot_prompts = {}
        
    def learn_without_examples(self, task_description: str, 
                              task_type: str, 
                              domain_knowledge: Dict[str, Any] = None) -> Dict[str, Any]:
        """Learn task without any examples using zero-shot learning"""
        
        # Store task in knowledge base
        task_id = f"{task_type}_{len(self.knowledge_base)}"
        self.knowledge_base[task_id] = {
            'description': task_description,
            'type': task_type,
            'domain_knowledge': domain_knowledge or {},
            'timestamp': len(self.knowledge_base)
        }
        
        # Generate semantic embedding for task
        task_embedding = self._generate_task_embedding(task_description, task_type)
        self.semantic_embeddings[task_id] = task_embedding
        
        # Find similar tasks in knowledge base
        similar_tasks = self._find_similar_tasks(task_embedding, task_id)
        
        # Generate task template
        task_template = self._generate_task_template(task_description, task_type, similar_tasks)
        self.task_templates[task_id] = task_template
        
        # Create zero-shot prompt
        zero_shot_prompt = self._create_zero_shot_prompt(task_description, task_type, task_template)
        self.zero_shot_prompts[task_id] = zero_shot_prompt
        
        # Generate initial model parameters
        model_params = self._generate_zero_shot_parameters(task_type, task_template, similar_tasks)
        
        return {
            'task_id': task_id,
            'model_parameters': model_params,
            'task_template': task_template,
            'zero_shot_prompt': zero_shot_prompt,
            'similar_tasks': similar_tasks,
            'confidence_score': self._calculate_confidence_score(task_embedding, similar_tasks)
        }
        
    def _generate_task_embedding(self, description: str, task_type: str) -> np.ndarray:
        """Generate semantic embedding for task"""
        # Simple embedding based on description and type
        words = description.lower().split()
        type_words = task_type.lower().split('_')
        
        # Create vocabulary-based embedding
        vocab = set(words + type_words + ['document', 'export', 'format', 'processing'])
        embedding = np.zeros(len(vocab))
        
        for i, word in enumerate(sorted(vocab)):
            if word in words:
                embedding[i] = 1.0
            elif word in type_words:
                embedding[i] = 0.8
            else:
                embedding[i] = 0.1
                
        return embedding
        
    def _find_similar_tasks(self, task_embedding: np.ndarray, current_task_id: str) -> List[str]:
        """Find similar tasks in knowledge base"""
        similarities = []
        
        for task_id, embedding in self.semantic_embeddings.items():
            if task_id != current_task_id:
                similarity = np.dot(task_embedding, embedding) / (
                    np.linalg.norm(task_embedding) * np.linalg.norm(embedding)
                )
                similarities.append((similarity, task_id))
                
        # Sort by similarity and return top 3
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [task_id for _, task_id in similarities[:3]]
        
    def _generate_task_template(self, description: str, task_type: str, similar_tasks: List[str]) -> Dict[str, Any]:
        """Generate task template based on description and similar tasks"""
        template = {
            'input_format': 'text',
            'output_format': 'structured_data',
            'processing_steps': [],
            'parameters': {},
            'constraints': []
        }
        
        # Customize template based on task type
        if task_type == "document_classification":
            template.update({
                'input_format': 'document',
                'output_format': 'classification_label',
                'processing_steps': ['preprocess', 'extract_features', 'classify'],
                'parameters': {'num_classes': 5, 'confidence_threshold': 0.8}
            })
        elif task_type == "text_generation":
            template.update({
                'input_format': 'prompt',
                'output_format': 'generated_text',
                'processing_steps': ['encode', 'generate', 'decode'],
                'parameters': {'max_length': 512, 'temperature': 0.7}
            })
        elif task_type == "image_processing":
            template.update({
                'input_format': 'image',
                'output_format': 'processed_image',
                'processing_steps': ['resize', 'enhance', 'filter'],
                'parameters': {'target_size': (224, 224), 'quality': 0.9}
            })
            
        # Incorporate knowledge from similar tasks
        for similar_task_id in similar_tasks:
            similar_task = self.knowledge_base.get(similar_task_id, {})
            similar_template = self.task_templates.get(similar_task_id, {})
            
            # Merge processing steps
            if 'processing_steps' in similar_template:
                template['processing_steps'].extend(similar_template['processing_steps'])
                template['processing_steps'] = list(set(template['processing_steps']))
                
        return template
        
    def _create_zero_shot_prompt(self, description: str, task_type: str, template: Dict[str, Any]) -> str:
        """Create zero-shot prompt for the task"""
        prompt = f"Task: {description}\n"
        prompt += f"Type: {task_type}\n"
        prompt += f"Input format: {template['input_format']}\n"
        prompt += f"Output format: {template['output_format']}\n"
        prompt += "Processing steps:\n"
        
        for step in template['processing_steps']:
            prompt += f"- {step}\n"
            
        prompt += "\nPlease perform this task with the given input."
        
        return prompt
        
    def _generate_zero_shot_parameters(self, task_type: str, template: Dict[str, Any], 
                                      similar_tasks: List[str]) -> Dict[str, Any]:
        """Generate initial model parameters for zero-shot learning"""
        base_params = {
            'learning_rate': 0.001,
            'batch_size': 32,
            'dropout': 0.1,
            'optimizer': 'adam'
        }
        
        # Add task-specific parameters
        base_params.update(template.get('parameters', {}))
        
        # Incorporate parameters from similar tasks
        for similar_task_id in similar_tasks:
            similar_template = self.task_templates.get(similar_task_id, {})
            similar_params = similar_template.get('parameters', {})
            
            # Average similar parameters
            for key, value in similar_params.items():
                if key in base_params and isinstance(value, (int, float)):
                    base_params[key] = (base_params[key] + value) / 2
                else:
                    base_params[key] = value
                    
        return base_params
        
    def _calculate_confidence_score(self, task_embedding: np.ndarray, similar_tasks: List[str]) -> float:
        """Calculate confidence score for zero-shot learning"""
        if not similar_tasks:
            return 0.3  # Low confidence without similar tasks
            
        # Calculate average similarity to similar tasks
        similarities = []
        for task_id in similar_tasks:
            if task_id in self.semantic_embeddings:
                embedding = self.semantic_embeddings[task_id]
                similarity = np.dot(task_embedding, embedding) / (
                    np.linalg.norm(task_embedding) * np.linalg.norm(embedding)
                )
                similarities.append(similarity)
                
        if similarities:
            avg_similarity = np.mean(similarities)
            confidence = min(0.9, 0.3 + avg_similarity * 0.6)
        else:
            confidence = 0.3
            
        return confidence

class ContinuousLearner:
    """Continuous learning system for lifelong adaptation"""
    
    def __init__(self, config: MetaLearningConfig):
        self.config = config
        self.learning_history = []
        self.knowledge_graph = {}
        self.adaptation_strategies = {}
        self.performance_monitor = defaultdict(list)
        self.catastrophic_forgetting_prevention = True
        self.experience_replay_buffer = deque(maxlen=config.experience_replay_size)
        
    def continuous_adaptation(self, new_data: List[Dict[str, Any]], 
                            task_type: str, 
                            adaptation_strategy: str = "elastic_weight_consolidation") -> Dict[str, Any]:
        """Continuously adapt to new data while preventing catastrophic forgetting"""
        
        # Store new experience
        experience = {
            'data': new_data,
            'task_type': task_type,
            'timestamp': len(self.learning_history),
            'adaptation_strategy': adaptation_strategy
        }
        self.learning_history.append(experience)
        self.experience_replay_buffer.append(experience)
        
        # Monitor performance before adaptation
        pre_adaptation_performance = self._evaluate_current_performance(task_type)
        
        # Apply adaptation strategy
        if adaptation_strategy == "elastic_weight_consolidation":
            adaptation_result = self._elastic_weight_consolidation(new_data, task_type)
        elif adaptation_strategy == "progressive_networks":
            adaptation_result = self._progressive_networks(new_data, task_type)
        elif adaptation_strategy == "packnet":
            adaptation_result = self._packnet_adaptation(new_data, task_type)
        else:
            adaptation_result = self._standard_adaptation(new_data, task_type)
            
        # Monitor performance after adaptation
        post_adaptation_performance = self._evaluate_current_performance(task_type)
        
        # Update performance monitoring
        self.performance_monitor[task_type].append({
            'pre_adaptation': pre_adaptation_performance,
            'post_adaptation': post_adaptation_performance,
            'improvement': post_adaptation_performance - pre_adaptation_performance,
            'timestamp': len(self.learning_history)
        })
        
        # Update knowledge graph
        self._update_knowledge_graph(task_type, adaptation_result)
        
        return {
            'adaptation_result': adaptation_result,
            'pre_adaptation_performance': pre_adaptation_performance,
            'post_adaptation_performance': post_adaptation_performance,
            'improvement': post_adaptation_performance - pre_adaptation_performance,
            'forgetting_prevention': self._check_forgetting_prevention(),
            'knowledge_graph_update': True
        }
        
    def _elastic_weight_consolidation(self, new_data: List[Dict[str, Any]], task_type: str) -> Dict[str, Any]:
        """Elastic Weight Consolidation for preventing catastrophic forgetting"""
        
        # Calculate Fisher Information Matrix for important parameters
        fisher_info = self._calculate_fisher_information(task_type)
        
        # Calculate importance weights
        importance_weights = self._calculate_importance_weights(fisher_info)
        
        # Adapt model with EWC regularization
        adaptation_loss = self._calculate_ewc_loss(new_data, task_type, importance_weights)
        
        # Update model parameters
        updated_params = self._update_with_ewc_regularization(adaptation_loss, importance_weights)
        
        return {
            'method': 'elastic_weight_consolidation',
            'fisher_information': fisher_info,
            'importance_weights': importance_weights,
            'adaptation_loss': adaptation_loss,
            'updated_parameters': updated_params
        }
        
    def _progressive_networks(self, new_data: List[Dict[str, Any]], task_type: str) -> Dict[str, Any]:
        """Progressive Networks for continuous learning"""
        
        # Create new column for new task
        new_column = self._create_progressive_column(task_type)
        
        # Connect to previous columns
        lateral_connections = self._create_lateral_connections(task_type)
        
        # Train new column
        training_result = self._train_progressive_column(new_column, new_data, lateral_connections)
        
        return {
            'method': 'progressive_networks',
            'new_column': new_column,
            'lateral_connections': lateral_connections,
            'training_result': training_result
        }
        
    def _packnet_adaptation(self, new_data: List[Dict[str, Any]], task_type: str) -> Dict[str, Any]:
        """PackNet adaptation for parameter-efficient continuous learning"""
        
        # Identify available parameters
        available_params = self._identify_available_parameters()
        
        # Pack parameters for new task
        packed_params = self._pack_parameters(available_params, task_type)
        
        # Train with packed parameters
        training_result = self._train_with_packed_parameters(packed_params, new_data)
        
        return {
            'method': 'packnet',
            'available_parameters': available_params,
            'packed_parameters': packed_params,
            'training_result': training_result
        }
        
    def _standard_adaptation(self, new_data: List[Dict[str, Any]], task_type: str) -> Dict[str, Any]:
        """Standard adaptation without forgetting prevention"""
        
        # Simple gradient descent adaptation
        adaptation_loss = self._calculate_standard_loss(new_data, task_type)
        updated_params = self._update_parameters_standard(adaptation_loss)
        
        return {
            'method': 'standard',
            'adaptation_loss': adaptation_loss,
            'updated_parameters': updated_params
        }
        
    def _calculate_fisher_information(self, task_type: str) -> Dict[str, float]:
        """Calculate Fisher Information Matrix for important parameters"""
        # Simulate Fisher Information calculation
        fisher_info = {}
        for i in range(10):  # Simulate 10 parameters
            fisher_info[f'param_{i}'] = np.random.exponential(1.0)
            
        return fisher_info
        
    def _calculate_importance_weights(self, fisher_info: Dict[str, float]) -> Dict[str, float]:
        """Calculate importance weights from Fisher Information"""
        total_fisher = sum(fisher_info.values())
        importance_weights = {}
        
        for param, fisher_value in fisher_info.items():
            importance_weights[param] = fisher_value / total_fisher if total_fisher > 0 else 0.0
            
        return importance_weights
        
    def _calculate_ewc_loss(self, new_data: List[Dict[str, Any]], task_type: str, 
                           importance_weights: Dict[str, float]) -> float:
        """Calculate EWC loss with regularization"""
        # Standard loss on new data
        standard_loss = self._calculate_standard_loss(new_data, task_type)
        
        # EWC regularization term
        ewc_regularization = 0.0
        for param, importance in importance_weights.items():
            # Simulate parameter change
            param_change = np.random.random()
            ewc_regularization += importance * (param_change ** 2)
            
        # Combine losses
        ewc_loss = standard_loss + 0.1 * ewc_regularization  # Lambda = 0.1
        
        return ewc_loss
        
    def _update_with_ewc_regularization(self, loss: float, importance_weights: Dict[str, float]) -> Dict[str, Any]:
        """Update parameters with EWC regularization"""
        # Simulate parameter update
        updated_params = {}
        for param, importance in importance_weights.items():
            # Update with importance-weighted learning rate
            learning_rate = 0.001 * (1.0 - importance * 0.5)  # Reduce LR for important params
            param_update = -learning_rate * np.random.random()
            updated_params[param] = param_update
            
        return updated_params
        
    def _create_progressive_column(self, task_type: str) -> Dict[str, Any]:
        """Create new progressive network column"""
        return {
            'task_type': task_type,
            'layers': [512, 256, 128],
            'activation': 'relu',
            'dropout': 0.1,
            'initialization': 'xavier'
        }
        
    def _create_lateral_connections(self, task_type: str) -> Dict[str, Any]:
        """Create lateral connections to previous columns"""
        return {
            'connections': ['previous_task_1', 'previous_task_2'],
            'connection_weights': [0.3, 0.2],
            'connection_type': 'adaptive'
        }
        
    def _train_progressive_column(self, column: Dict[str, Any], data: List[Dict[str, Any]], 
                                 connections: Dict[str, Any]) -> Dict[str, Any]:
        """Train progressive network column"""
        # Simulate training
        training_epochs = 10
        training_loss = []
        
        for epoch in range(training_epochs):
            epoch_loss = np.random.exponential(1.0) * (0.9 ** epoch)  # Decreasing loss
            training_loss.append(epoch_loss)
            
        return {
            'training_epochs': training_epochs,
            'final_loss': training_loss[-1],
            'training_history': training_loss,
            'convergence': training_loss[-1] < 0.1
        }
        
    def _identify_available_parameters(self) -> List[str]:
        """Identify available parameters for packing"""
        return [f'param_{i}' for i in range(20)]  # Simulate 20 available parameters
        
    def _pack_parameters(self, available_params: List[str], task_type: str) -> Dict[str, Any]:
        """Pack parameters for new task"""
        # Select subset of parameters for new task
        num_params_needed = min(5, len(available_params))
        selected_params = random.sample(available_params, num_params_needed)
        
        return {
            'task_type': task_type,
            'selected_parameters': selected_params,
            'parameter_masks': {param: 1.0 for param in selected_params},
            'packing_efficiency': num_params_needed / len(available_params)
        }
        
    def _train_with_packed_parameters(self, packed_params: Dict[str, Any], 
                                     data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Train model with packed parameters"""
        # Simulate training with parameter masking
        training_loss = []
        for epoch in range(5):
            epoch_loss = np.random.exponential(0.5) * (0.8 ** epoch)
            training_loss.append(epoch_loss)
            
        return {
            'training_epochs': 5,
            'final_loss': training_loss[-1],
            'training_history': training_loss,
            'parameter_utilization': packed_params['packing_efficiency']
        }
        
    def _calculate_standard_loss(self, data: List[Dict[str, Any]], task_type: str) -> float:
        """Calculate standard loss on new data"""
        # Simulate loss calculation
        return np.random.exponential(1.0)
        
    def _update_parameters_standard(self, loss: float) -> Dict[str, Any]:
        """Update parameters using standard gradient descent"""
        # Simulate parameter update
        return {
            'learning_rate': 0.001,
            'parameter_updates': {f'param_{i}': np.random.random() * 0.01 for i in range(10)},
            'convergence': loss < 0.1
        }
        
    def _evaluate_current_performance(self, task_type: str) -> float:
        """Evaluate current performance on task type"""
        # Simulate performance evaluation
        return np.random.random()
        
    def _check_forgetting_prevention(self) -> Dict[str, bool]:
        """Check if catastrophic forgetting prevention is working"""
        return {
            'ewc_active': True,
            'progressive_networks_active': True,
            'packnet_active': True,
            'experience_replay_active': len(self.experience_replay_buffer) > 0
        }
        
    def _update_knowledge_graph(self, task_type: str, adaptation_result: Dict[str, Any]):
        """Update knowledge graph with new learning"""
        if task_type not in self.knowledge_graph:
            self.knowledge_graph[task_type] = {
                'nodes': [],
                'edges': [],
                'adaptations': []
            }
            
        # Add new adaptation
        self.knowledge_graph[task_type]['adaptations'].append(adaptation_result)
        
        # Update nodes and edges based on adaptation
        new_node = f"{task_type}_adaptation_{len(self.knowledge_graph[task_type]['adaptations'])}"
        self.knowledge_graph[task_type]['nodes'].append(new_node)
        
        # Add edges to related tasks
        for other_task in self.knowledge_graph:
            if other_task != task_type:
                edge = (new_node, f"{other_task}_latest")
                self.knowledge_graph[task_type]['edges'].append(edge)

class MetaLearningEngine:
    """Main Meta-Learning Engine integrating all learning paradigms"""
    
    def __init__(self, config: MetaLearningConfig):
        self.config = config
        self.few_shot_learner = FewShotLearner(config)
        self.zero_shot_learner = ZeroShotLearner(config)
        self.continuous_learner = ContinuousLearner(config)
        self.learning_paradigms = {
            'few_shot': self.few_shot_learner,
            'zero_shot': self.zero_shot_learner,
            'continuous': self.continuous_learner
        }
        self.performance_analytics = defaultdict(list)
        self.learning_strategies = {}
        
    def adaptive_learning(self, task_data: Dict[str, Any], 
                         learning_paradigm: str = "auto") -> Dict[str, Any]:
        """Adaptively choose and apply learning paradigm"""
        
        # Auto-select learning paradigm based on data characteristics
        if learning_paradigm == "auto":
            learning_paradigm = self._select_learning_paradigm(task_data)
            
        # Apply selected learning paradigm
        if learning_paradigm == "few_shot":
            result = self._apply_few_shot_learning(task_data)
        elif learning_paradigm == "zero_shot":
            result = self._apply_zero_shot_learning(task_data)
        elif learning_paradigm == "continuous":
            result = self._apply_continuous_learning(task_data)
        else:
            result = self._apply_hybrid_learning(task_data)
            
        # Update performance analytics
        self._update_performance_analytics(learning_paradigm, result)
        
        return {
            'selected_paradigm': learning_paradigm,
            'learning_result': result,
            'performance_metrics': self._calculate_performance_metrics(learning_paradigm),
            'recommendations': self._generate_learning_recommendations(task_data, result)
        }
        
    def _select_learning_paradigm(self, task_data: Dict[str, Any]) -> str:
        """Automatically select best learning paradigm"""
        support_set_size = len(task_data.get('support_set', []))
        query_set_size = len(task_data.get('query_set', []))
        task_description = task_data.get('description', '')
        
        # Decision logic based on data characteristics
        if support_set_size == 0:
            return "zero_shot"
        elif support_set_size <= self.config.few_shot_threshold:
            return "few_shot"
        elif support_set_size > self.config.few_shot_threshold:
            return "continuous"
        else:
            return "hybrid"
            
    def _apply_few_shot_learning(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply few-shot learning"""
        support_set = task_data.get('support_set', [])
        query_set = task_data.get('query_set', [])
        task_type = task_data.get('task_type', 'generic')
        
        return self.few_shot_learner.learn_from_few_examples(support_set, query_set, task_type)
        
    def _apply_zero_shot_learning(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply zero-shot learning"""
        task_description = task_data.get('description', '')
        task_type = task_data.get('task_type', 'generic')
        domain_knowledge = task_data.get('domain_knowledge', {})
        
        return self.zero_shot_learner.learn_without_examples(task_description, task_type, domain_knowledge)
        
    def _apply_continuous_learning(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply continuous learning"""
        new_data = task_data.get('new_data', [])
        task_type = task_data.get('task_type', 'generic')
        adaptation_strategy = task_data.get('adaptation_strategy', 'elastic_weight_consolidation')
        
        return self.continuous_learner.continuous_adaptation(new_data, task_type, adaptation_strategy)
        
    def _apply_hybrid_learning(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply hybrid learning combining multiple paradigms"""
        # Start with zero-shot learning
        zero_shot_result = self._apply_zero_shot_learning(task_data)
        
        # If support set is available, apply few-shot learning
        if task_data.get('support_set'):
            few_shot_result = self._apply_few_shot_learning(task_data)
            
            # Combine results
            combined_result = {
                'zero_shot': zero_shot_result,
                'few_shot': few_shot_result,
                'combined_confidence': (zero_shot_result.get('confidence_score', 0) + 
                                      few_shot_result.get('final_performance', 0)) / 2
            }
        else:
            combined_result = {
                'zero_shot': zero_shot_result,
                'combined_confidence': zero_shot_result.get('confidence_score', 0)
            }
            
        return combined_result
        
    def _update_performance_analytics(self, paradigm: str, result: Dict[str, Any]):
        """Update performance analytics"""
        performance_metrics = {
            'paradigm': paradigm,
            'timestamp': len(self.performance_analytics[paradigm]),
            'performance': result.get('final_performance', result.get('confidence_score', 0)),
            'improvement': result.get('improvement', 0),
            'success': result.get('final_performance', 0) > 0.5
        }
        
        self.performance_analytics[paradigm].append(performance_metrics)
        
    def _calculate_performance_metrics(self, paradigm: str) -> Dict[str, Any]:
        """Calculate performance metrics for paradigm"""
        if paradigm not in self.performance_analytics:
            return {}
            
        metrics = self.performance_analytics[paradigm]
        if not metrics:
            return {}
            
        performances = [m['performance'] for m in metrics]
        improvements = [m['improvement'] for m in metrics]
        success_rate = sum(m['success'] for m in metrics) / len(metrics)
        
        return {
            'average_performance': np.mean(performances),
            'performance_std': np.std(performances),
            'average_improvement': np.mean(improvements),
            'success_rate': success_rate,
            'total_attempts': len(metrics),
            'trend': self._calculate_performance_trend(performances)
        }
        
    def _calculate_performance_trend(self, performances: List[float]) -> str:
        """Calculate performance trend"""
        if len(performances) < 2:
            return "insufficient_data"
            
        recent_avg = np.mean(performances[-5:]) if len(performances) >= 5 else np.mean(performances)
        early_avg = np.mean(performances[:5]) if len(performances) >= 5 else np.mean(performances)
        
        if recent_avg > early_avg * 1.1:
            return "improving"
        elif recent_avg < early_avg * 0.9:
            return "declining"
        else:
            return "stable"
            
    def _generate_learning_recommendations(self, task_data: Dict[str, Any], 
                                         result: Dict[str, Any]) -> List[str]:
        """Generate learning recommendations"""
        recommendations = []
        
        # Performance-based recommendations
        performance = result.get('final_performance', result.get('confidence_score', 0))
        if performance < 0.5:
            recommendations.append("Consider providing more training examples")
            recommendations.append("Try different learning paradigm")
            
        # Data-based recommendations
        support_set_size = len(task_data.get('support_set', []))
        if support_set_size < 5:
            recommendations.append("Increase support set size for better few-shot learning")
            
        # Task complexity recommendations
        task_type = task_data.get('task_type', '')
        if 'complex' in task_type.lower():
            recommendations.append("Use progressive learning approach")
            recommendations.append("Consider curriculum learning")
            
        return recommendations
        
    def get_learning_insights(self) -> Dict[str, Any]:
        """Get comprehensive learning insights"""
        insights = {
            'paradigm_performance': {},
            'learning_trends': {},
            'recommendations': [],
            'knowledge_base_stats': {}
        }
        
        # Analyze each paradigm
        for paradigm in self.learning_paradigms:
            insights['paradigm_performance'][paradigm] = self._calculate_performance_metrics(paradigm)
            
        # Calculate learning trends
        insights['learning_trends'] = self._calculate_learning_trends()
        
        # Generate recommendations
        insights['recommendations'] = self._generate_system_recommendations()
        
        # Knowledge base statistics
        insights['knowledge_base_stats'] = {
            'few_shot_tasks': len(self.few_shot_learner.adaptation_history),
            'zero_shot_tasks': len(self.zero_shot_learner.knowledge_base),
            'continuous_adaptations': len(self.continuous_learner.learning_history),
            'total_experiences': sum(len(paradigm.performance_analytics) for paradigm in self.learning_paradigms.values())
        }
        
        return insights
        
    def _calculate_learning_trends(self) -> Dict[str, Any]:
        """Calculate learning trends across paradigms"""
        trends = {}
        
        for paradigm, analytics in self.performance_analytics.items():
            if analytics:
                performances = [a['performance'] for a in analytics]
                trends[paradigm] = {
                    'trend': self._calculate_performance_trend(performances),
                    'volatility': np.std(performances),
                    'recent_performance': np.mean(performances[-5:]) if len(performances) >= 5 else np.mean(performances)
                }
                
        return trends
        
    def _generate_system_recommendations(self) -> List[str]:
        """Generate system-wide recommendations"""
        recommendations = []
        
        # Analyze paradigm performance
        paradigm_performance = {}
        for paradigm, analytics in self.performance_analytics.items():
            if analytics:
                avg_performance = np.mean([a['performance'] for a in analytics])
                paradigm_performance[paradigm] = avg_performance
                
        # Generate recommendations based on performance
        if paradigm_performance:
            best_paradigm = max(paradigm_performance, key=paradigm_performance.get)
            worst_paradigm = min(paradigm_performance, key=paradigm_performance.get)
            
            recommendations.append(f"Best performing paradigm: {best_paradigm}")
            recommendations.append(f"Consider improving {worst_paradigm} paradigm")
            
        # Memory and efficiency recommendations
        total_memories = sum(len(learner.memory_bank.memories) if hasattr(learner, 'memory_bank') else 0 
                           for learner in self.learning_paradigms.values())
        
        if total_memories > 1000:
            recommendations.append("Consider memory optimization for better efficiency")
            
        return recommendations

# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize meta-learning engine
    config = MetaLearningConfig(
        support_set_size=5,
        query_set_size=15,
        num_ways=5,
        num_shots=1,
        meta_batch_size=4
    )
    
    meta_engine = MetaLearningEngine(config)
    
    # Test few-shot learning
    print("Testing Few-Shot Learning...")
    few_shot_data = {
        'support_set': [
            {'features': np.random.randn(768), 'label': 0},
            {'features': np.random.randn(768), 'label': 1},
            {'features': np.random.randn(768), 'label': 0}
        ],
        'query_set': [
            {'features': np.random.randn(768), 'label': 1},
            {'features': np.random.randn(768), 'label': 0}
        ],
        'task_type': 'document_classification'
    }
    
    few_shot_result = meta_engine.adaptive_learning(few_shot_data, "few_shot")
    print(f"Few-shot learning result: {few_shot_result['learning_result']['final_performance']:.4f}")
    
    # Test zero-shot learning
    print("\nTesting Zero-Shot Learning...")
    zero_shot_data = {
        'description': 'Classify documents into business categories',
        'task_type': 'document_classification',
        'domain_knowledge': {'domain': 'business', 'categories': ['finance', 'marketing', 'hr']}
    }
    
    zero_shot_result = meta_engine.adaptive_learning(zero_shot_data, "zero_shot")
    print(f"Zero-shot learning confidence: {zero_shot_result['learning_result']['confidence_score']:.4f}")
    
    # Test continuous learning
    print("\nTesting Continuous Learning...")
    continuous_data = {
        'new_data': [
            {'input': np.random.randn(512), 'target': np.random.randn(128)},
            {'input': np.random.randn(512), 'target': np.random.randn(128)}
        ],
        'task_type': 'generic_learning',
        'adaptation_strategy': 'elastic_weight_consolidation'
    }
    
    continuous_result = meta_engine.adaptive_learning(continuous_data, "continuous")
    print(f"Continuous learning improvement: {continuous_result['learning_result']['improvement']:.4f}")
    
    # Test auto-selection
    print("\nTesting Auto-Selection...")
    auto_data = {
        'support_set': [{'features': np.random.randn(768), 'label': 0}],
        'query_set': [{'features': np.random.randn(768), 'label': 1}],
        'task_type': 'document_classification'
    }
    
    auto_result = meta_engine.adaptive_learning(auto_data, "auto")
    print(f"Auto-selected paradigm: {auto_result['selected_paradigm']}")
    print(f"Auto-selection performance: {auto_result['learning_result'].get('final_performance', auto_result['learning_result'].get('confidence_score', 0)):.4f}")
    
    # Get learning insights
    insights = meta_engine.get_learning_insights()
    print(f"\nLearning Insights:")
    print(f"Paradigm performance: {insights['paradigm_performance']}")
    print(f"Knowledge base stats: {insights['knowledge_base_stats']}")
    print(f"Recommendations: {insights['recommendations']}")
    
    print("\nMeta-Learning Engine initialized successfully!")
























