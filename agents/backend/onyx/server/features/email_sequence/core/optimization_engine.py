from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import optuna
from ..models.sequence import EmailSequence, SequenceStep
from ..models.subscriber import Subscriber
from ..models.template import EmailTemplate
from typing import Any, List, Dict, Optional
"""
Optimization Engine for Email Sequence System

Advanced optimization engine using PyTorch, transformers, and deep learning techniques
for email sequence optimization and personalization.
"""


    AutoTokenizer, 
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    pipeline
)


logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """Configuration for optimization engine"""
    learning_rate: float = 1e-4
    batch_size: int = 32
    max_epochs: int = 100
    early_stopping_patience: int = 10
    use_mixed_precision: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    model_name: str = "microsoft/DialoGPT-medium"
    max_length: int = 512
    num_trials: int = 50


class EmailSequenceDataset(Dataset):
    """Custom dataset for email sequence optimization"""
    
    def __init__(self, sequences: List[EmailSequence], subscribers: List[Subscriber], 
                 templates: List[EmailTemplate], tokenizer):
        
    """__init__ function."""
self.sequences = sequences
        self.subscribers = subscribers
        self.templates = templates
        self.tokenizer = tokenizer
        
    def __len__(self) -> Any:
        return len(self.sequences)
    
    def __getitem__(self, idx) -> Optional[Dict[str, Any]]:
        sequence = self.sequences[idx]
        subscriber = self.subscribers[idx % len(self.subscribers)]
        template = self.templates[idx % len(self.templates)]
        
        # Create input text
        input_text = self._create_input_text(sequence, subscriber, template)
        
        # Tokenize
        encoding = self.tokenizer(
            input_text,
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(1.0)  # Placeholder for optimization target
        }
    
    def _create_input_text(self, sequence: EmailSequence, subscriber: Subscriber, 
                          template: EmailTemplate) -> str:
        """Create input text for model"""
        return f"""
        Sequence: {sequence.name}
        Subscriber: {subscriber.email} - {subscriber.first_name} {subscriber.last_name}
        Company: {subscriber.company}
        Interests: {', '.join(subscriber.interests)}
        Template: {template.name}
        Content: {template.html_content[:200]}...
        """


class EmailOptimizationModel(nn.Module):
    """PyTorch model for email sequence optimization"""
    
    def __init__(self, model_name: str, num_classes: int = 1):
        
    """__init__ function."""
super().__init__()
        self.transformer = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=num_classes
        )
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.transformer.config.hidden_size, num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self) -> Any:
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, input_ids, attention_mask, labels=None) -> Any:
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.MSELoss()
            loss = loss_fct(logits.squeeze(), labels)
        
        return {"loss": loss, "logits": logits}


class OptimizationEngine:
    """Advanced optimization engine for email sequences"""
    
    def __init__(self, config: OptimizationConfig):
        
    """__init__ function."""
self.config = config
        self.device = torch.device(config.device)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = None
        self.optimizer = None
        self.scheduler = None
        
        # Initialize optimization components
        self._setup_model()
        self._setup_optimization_tools()
        
        logger.info(f"Optimization Engine initialized on {self.device}")
    
    def _setup_model(self) -> Any:
        """Setup PyTorch model with mixed precision"""
        self.model = EmailOptimizationModel(self.config.model_name)
        self.model.to(self.device)
        
        if self.config.use_mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
    
    def _setup_optimization_tools(self) -> Any:
        """Setup optimization tools"""
        # Optuna for hyperparameter optimization
        self.study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler()
        )
        
        # Performance tracking
        self.performance_history = []
    
    async def optimize_sequence_timing(
        self, 
        sequence: EmailSequence, 
        subscribers: List[Subscriber]
    ) -> Dict[str, Any]:
        """Optimize email sequence timing using ML"""
        
        def objective(trial) -> Any:
            # Hyperparameters to optimize
            delay_hours = trial.suggest_int("delay_hours", 1, 168)
            send_time = trial.suggest_int("send_hour", 0, 23)
            day_of_week = trial.suggest_int("day_of_week", 0, 6)
            
            # Simulate performance
            performance = self._simulate_timing_performance(
                sequence, subscribers, delay_hours, send_time, day_of_week
            )
            
            return performance
        
        # Run optimization
        self.study.optimize(objective, n_trials=self.config.num_trials)
        
        best_params = self.study.best_params
        best_value = self.study.best_value
        
        return {
            "optimal_delay_hours": best_params["delay_hours"],
            "optimal_send_hour": best_params["send_hour"],
            "optimal_day_of_week": best_params["day_of_week"],
            "expected_performance": best_value,
            "optimization_history": self.study.trials_dataframe().to_dict()
        }
    
    async def optimize_content_personalization(
        self, 
        template: EmailTemplate, 
        subscribers: List[Subscriber]
    ) -> Dict[str, Any]:
        """Optimize content personalization using transformers"""
        
        # Prepare dataset
        dataset = EmailSequenceDataset(
            sequences=[EmailSequence(name="test")] * len(subscribers),
            subscribers=subscribers,
            templates=[template] * len(subscribers),
            tokenizer=self.tokenizer
        )
        
        dataloader = DataLoader(
            dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True
        )
        
        # Training loop with mixed precision
        self.model.train()
        total_loss = 0
        
        for batch in dataloader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            if self.config.use_mixed_precision:
                with torch.cuda.amp.autocast():
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs["loss"]
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs["loss"]
                
                loss.backward()
                self.optimizer.step()
            
            self.optimizer.zero_grad()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        
        return {
            "training_loss": avg_loss,
            "model_performance": self._evaluate_model(dataloader),
            "personalization_improvement": self._calculate_improvement()
        }
    
    async def optimize_subject_lines(
        self, 
        email_content: str, 
        subscriber_data: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """Generate optimized subject lines using transformers"""
        
        # Use transformers pipeline for text generation
        generator = pipeline(
            "text-generation",
            model="microsoft/DialoGPT-medium",
            device=0 if torch.cuda.is_available() else -1
        )
        
        # Create prompt for subject line generation
        prompt = f"""
        Email Content: {email_content[:200]}...
        Subscriber: {subscriber_data.get('first_name', '')} {subscriber_data.get('last_name', '')}
        Company: {subscriber_data.get('company', '')}
        Interests: {', '.join(subscriber_data.get('interests', []))}
        
        Generate 5 compelling email subject lines:
        """
        
        # Generate subject lines
        generated_text = generator(
            prompt, 
            max_length=100, 
            num_return_sequences=5,
            temperature=0.8,
            do_sample=True
        )
        
        # Extract and clean subject lines
        subject_lines = []
        for i, generation in enumerate(generated_text):
            text = generation['generated_text']
            # Extract subject line from generated text
            subject_line = self._extract_subject_line(text)
            subject_lines.append({
                "variant_id": f"variant_{i+1}",
                "subject_line": subject_line,
                "confidence": generation.get('score', 0.0)
            })
        
        return subject_lines
    
    async def optimize_sequence_structure(
        self, 
        sequence: EmailSequence
    ) -> Dict[str, Any]:
        """Optimize sequence structure using ML"""
        
        # Analyze current sequence
        current_metrics = self._analyze_sequence_metrics(sequence)
        
        # Generate optimization suggestions
        suggestions = []
        
        # Optimize step order
        if len(sequence.steps) > 1:
            optimal_order = self._optimize_step_order(sequence.steps)
            suggestions.append({
                "type": "step_order",
                "current_order": [step.order for step in sequence.steps],
                "optimal_order": optimal_order,
                "expected_improvement": 0.15
            })
        
        # Optimize delays
        optimal_delays = self._optimize_delays(sequence.steps)
        suggestions.append({
            "type": "delays",
            "current_delays": [step.delay_hours for step in sequence.steps],
            "optimal_delays": optimal_delays,
            "expected_improvement": 0.12
        })
        
        # Optimize content length
        content_optimization = self._optimize_content_length(sequence.steps)
        suggestions.append({
            "type": "content_length",
            "current_lengths": [len(step.content or "") for step in sequence.steps],
            "optimal_lengths": content_optimization,
            "expected_improvement": 0.08
        })
        
        return {
            "current_metrics": current_metrics,
            "optimization_suggestions": suggestions,
            "overall_improvement_potential": sum(s["expected_improvement"] for s in suggestions)
        }
    
    def _simulate_timing_performance(
        self, 
        sequence: EmailSequence, 
        subscribers: List[Subscriber], 
        delay_hours: int, 
        send_hour: int, 
        day_of_week: int
    ) -> float:
        """Simulate performance for timing optimization"""
        # Simplified simulation - in production, use real data
        base_performance = 0.25  # Base open rate
        
        # Time-based adjustments
        time_factor = 1.0
        if 9 <= send_hour <= 17:  # Business hours
            time_factor = 1.2
        elif 18 <= send_hour <= 20:  # Evening
            time_factor = 1.1
        
        # Day-based adjustments
        day_factor = 1.0
        if day_of_week in [1, 2, 3]:  # Tuesday-Thursday
            day_factor = 1.15
        elif day_of_week == 0:  # Monday
            day_factor = 0.9
        
        # Delay-based adjustments
        delay_factor = 1.0
        if 24 <= delay_hours <= 72:  # 1-3 days
            delay_factor = 1.1
        elif delay_hours > 168:  # > 1 week
            delay_factor = 0.8
        
        return base_performance * time_factor * day_factor * delay_factor
    
    def _evaluate_model(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate model performance"""
        self.model.eval()
        total_loss = 0
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                loss = F.mse_loss(outputs["logits"].squeeze(), labels)
                total_loss += loss.item()
                
                predictions.extend(outputs["logits"].squeeze().cpu().numpy())
                actuals.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(dataloader)
        
        return {
            "loss": avg_loss,
            "rmse": np.sqrt(np.mean((np.array(predictions) - np.array(actuals)) ** 2))
        }
    
    def _calculate_improvement(self) -> float:
        """Calculate improvement from optimization"""
        if len(self.performance_history) < 2:
            return 0.0
        
        recent_performance = np.mean(self.performance_history[-5:])
        baseline_performance = np.mean(self.performance_history[:5])
        
        return (recent_performance - baseline_performance) / baseline_performance
    
    def _extract_subject_line(self, text: str) -> str:
        """Extract subject line from generated text"""
        lines = text.split('\n')
        for line in lines:
            if line.strip() and len(line.strip()) < 100:
                return line.strip()
        return "Optimized Subject Line"
    
    def _analyze_sequence_metrics(self, sequence: EmailSequence) -> Dict[str, float]:
        """Analyze current sequence metrics"""
        return {
            "total_steps": len(sequence.steps),
            "avg_delay": np.mean([step.delay_hours or 0 for step in sequence.steps]),
            "avg_content_length": np.mean([len(step.content or "") for step in sequence.steps]),
            "complexity_score": self._calculate_complexity_score(sequence)
        }
    
    def _calculate_complexity_score(self, sequence: EmailSequence) -> float:
        """Calculate sequence complexity score"""
        score = 0.0
        
        # Step count factor
        score += min(len(sequence.steps) * 0.1, 1.0)
        
        # Delay complexity
        delays = [step.delay_hours or 0 for step in sequence.steps]
        score += np.std(delays) * 0.01
        
        # Content complexity
        content_lengths = [len(step.content or "") for step in sequence.steps]
        score += np.std(content_lengths) * 0.001
        
        return min(score, 1.0)
    
    def _optimize_step_order(self, steps: List[SequenceStep]) -> List[int]:
        """Optimize step order using optimization algorithms"""
        # Simplified optimization - in production, use more sophisticated algorithms
        current_order = [step.order for step in steps]
        
        # Simple heuristic: put shorter delays first
        optimized_order = sorted(current_order, key=lambda x: steps[x-1].delay_hours or 0)
        
        return optimized_order
    
    def _optimize_delays(self, steps: List[SequenceStep]) -> List[int]:
        """Optimize delays between steps"""
        optimized_delays = []
        
        for i, step in enumerate(steps):
            if i == 0:
                optimized_delays.append(0)  # First step
            elif i == 1:
                optimized_delays.append(24)  # 1 day after first
            else:
                # Progressive delays
                optimized_delays.append(24 * (i + 1))
        
        return optimized_delays
    
    def _optimize_content_length(self, steps: List[SequenceStep]) -> List[int]:
        """Optimize content length for each step"""
        optimized_lengths = []
        
        for i, step in enumerate(steps):
            if i == 0:
                optimized_lengths.append(500)  # Welcome email
            elif i == len(steps) - 1:
                optimized_lengths.append(800)  # Final call-to-action
            else:
                optimized_lengths.append(600)  # Middle emails
        
        return optimized_lengths
    
    async def get_optimization_report(self) -> Dict[str, Any]:
        """Generate comprehensive optimization report"""
        return {
            "optimization_metrics": {
                "total_optimizations": len(self.performance_history),
                "average_improvement": np.mean(self.performance_history) if self.performance_history else 0,
                "best_improvement": max(self.performance_history) if self.performance_history else 0,
                "optimization_trend": self._calculate_trend()
            },
            "model_performance": {
                "device": str(self.device),
                "mixed_precision": self.config.use_mixed_precision,
                "model_name": self.config.model_name
            },
            "recommendations": self._generate_recommendations()
        }
    
    def _calculate_trend(self) -> str:
        """Calculate optimization trend"""
        if len(self.performance_history) < 3:
            return "insufficient_data"
        
        recent = np.mean(self.performance_history[-3:])
        earlier = np.mean(self.performance_history[:3])
        
        if recent > earlier * 1.1:
            return "improving"
        elif recent < earlier * 0.9:
            return "declining"
        else:
            return "stable"
    
    def _generate_recommendations(self) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []
        
        if self.config.use_mixed_precision:
            recommendations.append("Mixed precision training is enabled for optimal performance")
        
        if torch.cuda.is_available():
            recommendations.append("GPU acceleration is active")
        else:
            recommendations.append("Consider using GPU for faster optimization")
        
        if len(self.performance_history) > 10:
            recommendations.append("Sufficient data for reliable optimization trends")
        
        return recommendations 