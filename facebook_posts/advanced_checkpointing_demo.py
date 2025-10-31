#!/usr/bin/env python3
"""
🔒 Advanced Checkpoint Management System Demonstration

This script demonstrates the comprehensive checkpointing capabilities including:
- Advanced checkpoint setup and configuration
- Automatic checkpointing with different strategies
- Checkpoint validation and integrity checking
- Checkpoint comparison and analysis
- Checkpoint management and cleanup
- Integration with experiment tracking
"""

import sys
import os
import time
import random
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import logging

# Add modular structure to path
sys.path.append(str(Path(__file__).parent / "modular_structure"))

# Import checkpointing components
from utils.checkpoint_manager import CheckpointManager, CheckpointConfig, CheckpointMetadata
from utils.logger import Logger, LogConfig

# Import experiment tracking components
from experiment_tracking import (
    ExperimentConfig, create_experiment_tracker,
    ProblemDefinition, DatasetAnalysis
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DemoModel:
    """Simple demo model for checkpointing demonstration."""
    
    def __init__(self, input_size: int = 10, output_size: int = 1):
        import torch
        import torch.nn as nn
        
        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_size)
        )
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.9)
        
        self.input_size = input_size
        self.output_size = output_size
    
    def get_state_dict(self):
        """Get model state dict."""
        return self.model.state_dict()
    
    def load_state_dict(self, state_dict):
        """Load model state dict."""
        self.model.load_state_dict(state_dict)
    
    def get_optimizer_state_dict(self):
        """Get optimizer state dict."""
        return self.optimizer.state_dict()
    
    def load_optimizer_state_dict(self, state_dict):
        """Load optimizer state dict."""
        self.optimizer.load_state_dict(state_dict)
    
    def get_scheduler_state_dict(self):
        """Get scheduler state dict."""
        return self.scheduler.state_dict()
    
    def load_scheduler_state_dict(self, state_dict):
        """Load scheduler state dict."""
        self.scheduler.load_state_dict(state_dict)

def create_demo_problem_definition() -> ProblemDefinition:
    """Create a demo problem definition."""
    return ProblemDefinition(
        problem_title="Advanced Checkpointing Demonstration",
        problem_description="Demonstrate comprehensive checkpointing capabilities for ML experiments",
        problem_type="regression",
        domain="general",
        primary_objective="Showcase advanced checkpointing features",
        success_metrics=["checkpoint_integrity", "recovery_success", "performance_tracking"],
        baseline_performance=0.5,
        target_performance=0.9,
        computational_constraints="Single GPU, reasonable memory usage",
        time_constraints="Demonstration time: 5-10 minutes",
        accuracy_requirements="Demonstrate all checkpointing features successfully",
        interpretability_requirements="Clear logging and status reporting",
        business_value="Improved experiment reliability and model management",
        stakeholders=["ml_engineers", "researchers", "devops_team"],
        deployment_context="Development and production environments"
    )

def create_demo_dataset_analysis() -> DatasetAnalysis:
    """Create a demo dataset analysis."""
    return DatasetAnalysis(
        dataset_name="checkpointing_demo_data",
        dataset_source="synthetic_generation",
        dataset_version="1.0.0",
        dataset_size=1000,
        input_shape=(10,),
        output_shape=(1,),
        feature_count=10,
        class_count=1,
        data_types=["numerical"],
        missing_values_pct=0.0,
        duplicate_records_pct=0.0,
        outlier_pct=0.05,
        class_imbalance_ratio=1.0,
        train_size=700,
        val_size=150,
        test_size=150,
        data_split_strategy="random_70_15_15",
        normalization_needed=True,
        encoding_needed=False,
        augmentation_strategy="none",
        preprocessing_steps=["normalize", "split"]
    )

def demonstrate_basic_checkpointing():
    """Demonstrate basic checkpointing functionality."""
    logger.info("🔧 Demonstrating Basic Checkpointing")
    logger.info("=" * 60)
    
    # Create checkpoint configuration
    config = CheckpointConfig(
        checkpoint_dir="demo_checkpoints",
        save_interval=2,  # Save every 2 epochs
        max_checkpoints=5,
        save_best_only=False,
        monitor_metric="val_loss",
        monitor_mode="min",
        save_optimizer=True,
        save_scheduler=True,
        save_metadata=True,
        backup_checkpoints=True,
        validate_checkpoints=True
    )
    
    # Initialize checkpoint manager
    checkpoint_manager = CheckpointManager(config, "basic_demo")
    logger.info(f"✅ CheckpointManager initialized: {checkpoint_manager.checkpoint_dir}")
    
    # Create demo model
    model = DemoModel()
    logger.info("✅ Demo model created")
    
    # Simulate training and save checkpoints
    for epoch in range(10):
        # Simulate training metrics
        train_loss = 1.0 / (epoch + 1) + random.uniform(-0.1, 0.1)
        val_loss = train_loss + random.uniform(0, 0.2)
        accuracy = min(0.95, 0.5 + epoch * 0.05 + random.uniform(-0.02, 0.02))
        
        metrics = {
            'train_loss': train_loss,
            'val_loss': val_loss,
            'accuracy': accuracy,
            'learning_rate': 0.001 * (0.9 ** epoch)
        }
        
        # Save checkpoint
        checkpoint_id = checkpoint_manager.save_checkpoint(
            model=model.model,
            optimizer=model.optimizer,
            scheduler=model.scheduler,
            epoch=epoch,
            step=epoch * 100,
            metrics=metrics,
            tags=['demo', 'basic', f'epoch_{epoch}'],
            description=f"Basic demo checkpoint from epoch {epoch}"
        )
        
        if checkpoint_id:
            logger.info(f"✅ Checkpoint saved: {checkpoint_id}")
            logger.info(f"   Epoch {epoch}: Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Acc={accuracy:.4f}")
        else:
            logger.info(f"⚠️ Checkpoint not saved for epoch {epoch}")
        
        time.sleep(0.1)  # Simulate training time
    
    # List checkpoints
    checkpoints = checkpoint_manager.list_checkpoints()
    logger.info(f"📋 Total checkpoints saved: {len(checkpoints)}")
    
    # Get checkpoint summary
    summary = checkpoint_manager.get_checkpoint_summary()
    logger.info(f"📊 Checkpoint summary: {summary['total_checkpoints']} checkpoints, {summary['total_size_mb']:.2f} MB")
    
    return checkpoint_manager, model

def demonstrate_advanced_checkpointing():
    """Demonstrate advanced checkpointing features."""
    logger.info("\n🚀 Demonstrating Advanced Checkpointing Features")
    logger.info("=" * 60)
    
    # Create checkpoint configuration for best-only saving
    config = CheckpointConfig(
        checkpoint_dir="demo_checkpoints_advanced",
        save_interval=1,
        max_checkpoints=3,
        save_best_only=True,  # Only save best checkpoints
        monitor_metric="val_loss",
        monitor_mode="min",
        save_optimizer=True,
        save_scheduler=True,
        save_metadata=True,
        backup_checkpoints=True,
        validate_checkpoints=True
    )
    
    # Initialize checkpoint manager
    checkpoint_manager = CheckpointManager(config, "advanced_demo")
    logger.info(f"✅ Advanced CheckpointManager initialized: {checkpoint_manager.checkpoint_dir}")
    
    # Create demo model
    model = DemoModel()
    logger.info("✅ Demo model created")
    
    # Simulate training with varying performance
    val_losses = [0.8, 0.7, 0.9, 0.6, 0.8, 0.5, 0.7, 0.4, 0.6, 0.3]  # Some improvements, some not
    
    for epoch, val_loss in enumerate(val_losses):
        train_loss = val_loss + random.uniform(-0.1, 0.1)
        accuracy = 1.0 - val_loss + random.uniform(-0.05, 0.05)
        
        metrics = {
            'train_loss': train_loss,
            'val_loss': val_loss,
            'accuracy': accuracy,
            'learning_rate': 0.001 * (0.9 ** epoch)
        }
        
        # Save checkpoint (only saves if val_loss improves)
        checkpoint_id = checkpoint_manager.save_checkpoint(
            model=model.model,
            optimizer=model.optimizer,
            scheduler=model.scheduler,
            epoch=epoch,
            step=epoch * 100,
            metrics=metrics,
            tags=['demo', 'advanced', f'epoch_{epoch}'],
            description=f"Advanced demo checkpoint from epoch {epoch} (val_loss: {val_loss:.4f})"
        )
        
        if checkpoint_id:
            logger.info(f"✅ Best checkpoint saved: {checkpoint_id}")
            logger.info(f"   Epoch {epoch}: Val Loss={val_loss:.4f} (improved!)")
        else:
            logger.info(f"⚠️ Checkpoint not saved for epoch {epoch} (val_loss: {val_loss:.4f} - no improvement)")
        
        time.sleep(0.1)
    
    # Show best checkpoint
    best_checkpoint = checkpoint_manager.get_best_checkpoint()
    if best_checkpoint:
        logger.info(f"🏆 Best checkpoint: {best_checkpoint.checkpoint_id}")
        logger.info(f"   Val Loss: {best_checkpoint.metrics.get('val_loss', 'N/A')}")
    
    return checkpoint_manager, model

def demonstrate_checkpoint_validation_and_comparison(checkpoint_manager: CheckpointManager):
    """Demonstrate checkpoint validation and comparison features."""
    logger.info("\n🔍 Demonstrating Checkpoint Validation and Comparison")
    logger.info("=" * 60)
    
    # List all checkpoints
    checkpoints = checkpoint_manager.list_checkpoints()
    logger.info(f"📋 Available checkpoints: {len(checkpoints)}")
    
    if len(checkpoints) < 2:
        logger.warning("⚠️ Need at least 2 checkpoints for comparison demo")
        return
    
    # Validate checkpoints
    logger.info("🔍 Validating checkpoints...")
    for checkpoint in checkpoints[:3]:  # Validate first 3 checkpoints
        is_valid = checkpoint_manager.validate_checkpoint(checkpoint.checkpoint_id)
        status = "✅ Valid" if is_valid else "❌ Invalid"
        logger.info(f"   {checkpoint.checkpoint_id}: {status}")
    
    # Compare checkpoints
    if len(checkpoints) >= 2:
        checkpoint_ids = [cp.checkpoint_id for cp in checkpoints[:3]]
        logger.info(f"🔍 Comparing checkpoints: {checkpoint_ids}")
        
        comparison = checkpoint_manager.compare_checkpoints(checkpoint_ids)
        
        if comparison and 'error' not in comparison:
            logger.info("📊 Checkpoint comparison results:")
            for metric, values in comparison.get('metrics_comparison', {}).items():
                logger.info(f"   {metric}:")
                for cp_id, value in values.items():
                    logger.info(f"     {cp_id}: {value:.4f}")
        else:
            logger.error(f"❌ Comparison failed: {comparison.get('error', 'Unknown error')}")

def demonstrate_checkpoint_management(checkpoint_manager: CheckpointManager):
    """Demonstrate checkpoint management features."""
    logger.info("\n🗂️ Demonstrating Checkpoint Management")
    logger.info("=" * 60)
    
    # Get checkpoint summary
    summary = checkpoint_manager.get_checkpoint_summary()
    logger.info("📊 Current checkpoint summary:")
    logger.info(f"   Total checkpoints: {summary['total_checkpoints']}")
    logger.info(f"   Total size: {summary['total_size_mb']:.2f} MB")
    logger.info(f"   Best checkpoint: {summary['best_checkpoint']}")
    logger.info(f"   Latest checkpoint: {summary['latest_checkpoint']}")
    
    # Export a checkpoint
    if summary['latest_checkpoint']:
        export_path = "exported_checkpoints/demo_export.pt"
        success = checkpoint_manager.export_checkpoint(summary['latest_checkpoint'], export_path)
        if success:
            logger.info(f"📤 Checkpoint exported to: {export_path}")
        else:
            logger.error("❌ Failed to export checkpoint")
    
    # Get checkpoint info
    if summary['best_checkpoint']:
        info = checkpoint_manager.get_checkpoint_info(summary['best_checkpoint'])
        if info:
            logger.info(f"📋 Best checkpoint info:")
            logger.info(f"   ID: {info.checkpoint_id}")
            logger.info(f"   Epoch: {info.epoch}, Step: {info.step}")
            logger.info(f"   File size: {info.file_size} bytes")
            logger.info(f"   Tags: {info.tags}")
            logger.info(f"   Description: {info.description}")

def demonstrate_experiment_tracker_integration():
    """Demonstrate integration with experiment tracking."""
    logger.info("\n🔗 Demonstrating Experiment Tracker Integration")
    logger.info("=" * 60)
    
    # Create experiment configuration
    config = ExperimentConfig(
        experiment_name="checkpointing_demo_experiment",
        project_name="advanced_checkpointing_demo",
        run_name="demo_run_001",
        tags=["demo", "checkpointing", "advanced"],
        notes="Demonstration of advanced checkpointing capabilities",
        enable_tensorboard=True,
        enable_wandb=False,  # Disable W&B for demo
        log_interval=1,
        save_interval=2,
        log_gradients=True,
        log_images=False,
        log_text=True,
        log_gradient_norms=True,
        log_nan_inf_counts=True,
        tensorboard_dir="runs/tensorboard",
        model_save_dir="checkpoints/experiment_tracker",
        config_save_dir="configs",
        problem_definition=create_demo_problem_definition(),
        dataset_analysis=create_demo_dataset_analysis()
    )
    
    # Create experiment tracker
    tracker = create_experiment_tracker(config)
    logger.info(f"✅ Experiment tracker created: {tracker.config.experiment_name}")
    
    # Setup advanced checkpointing
    tracker.setup_advanced_checkpointing()
    logger.info("✅ Advanced checkpointing setup completed")
    
    # Create demo model
    model = DemoModel()
    
    # Simulate training with experiment tracking
    for epoch in range(5):
        # Simulate training metrics
        train_loss = 1.0 / (epoch + 1) + random.uniform(-0.1, 0.1)
        val_loss = train_loss + random.uniform(0, 0.2)
        accuracy = min(0.95, 0.5 + epoch * 0.1 + random.uniform(-0.02, 0.02))
        
        # Log training step
        tracker.log_training_step(
            loss=train_loss,
            accuracy=accuracy,
            learning_rate=0.001 * (0.9 ** epoch),
            gradient_norm=random.uniform(0.5, 2.0),
            nan_count=random.randint(0, 2),
            inf_count=random.randint(0, 1),
            clipping_applied=random.choice([True, False]),
            clipping_threshold=1.0 if random.choice([True, False]) else None,
            training_time=random.uniform(0.1, 0.5)
        )
        
        # Save checkpoint through experiment tracker
        metrics = {
            'train_loss': train_loss,
            'val_loss': val_loss,
            'accuracy': accuracy,
            'learning_rate': 0.001 * (0.9 ** epoch)
        }
        
        checkpoint_id = tracker.save_checkpoint(
            model=model.model,
            optimizer=model.optimizer,
            epoch=epoch,
            step=epoch * 100,
            loss=train_loss,
            metrics=metrics,
            tags=['experiment_tracker', f'epoch_{epoch}'],
            description=f"Experiment tracker checkpoint from epoch {epoch}"
        )
        
        if checkpoint_id:
            logger.info(f"✅ Experiment tracker checkpoint saved: {checkpoint_id}")
        
        time.sleep(0.1)
    
    # Get experiment summary
    summary = tracker.get_experiment_summary()
    logger.info(f"📊 Experiment summary: {summary['total_steps']} steps, {summary['checkpoints']} checkpoints")
    
    # Close tracker
    tracker.close()
    logger.info("✅ Experiment tracker closed")
    
    return tracker

def demonstrate_logging_integration():
    """Demonstrate logging integration."""
    logger.info("\n📝 Demonstrating Logging Integration")
    logger.info("=" * 60)
    
    # Create logger configuration
    log_config = LogConfig(
        log_level="INFO",
        log_file="checkpointing_demo.log",
        log_dir="logs",
        json_format=False,
        console_output=True,
        file_output=True
    )
    
    # Create logger
    demo_logger = Logger("checkpointing_demo", log_config, "checkpointing_demo")
    logger.info("✅ Logger initialized")
    
    # Log various events
    demo_logger.log_event("demo_started", {"version": "1.0.0", "features": ["checkpointing", "validation", "comparison"]})
    demo_logger.log_metric("demo_progress", 0.25, step=1)
    demo_logger.log_metric("demo_progress", 0.50, step=2)
    demo_logger.log_metric("demo_progress", 0.75, step=3)
    demo_logger.log_metric("demo_progress", 1.00, step=4)
    demo_logger.log_event("demo_completed", {"status": "success", "duration": "5 minutes"})
    
    # Get experiment summary
    summary = demo_logger.get_experiment_summary()
    logger.info(f"📊 Logging summary: {summary['total_events']} events, {summary['total_metrics']} metrics")
    
    # Save experiment log
    demo_logger.save_experiment_log()
    logger.info("✅ Experiment log saved")
    
    # Close logger
    demo_logger.close()
    logger.info("✅ Logger closed")

def main():
    """Main demonstration function."""
    logger.info("🚀 Advanced Checkpoint Management System Demonstration")
    logger.info("=" * 80)
    logger.info("This demonstration showcases comprehensive checkpointing capabilities")
    logger.info("including automatic checkpointing, validation, comparison, and management.")
    logger.info("=" * 80)
    
    try:
        # 1. Basic checkpointing demonstration
        basic_manager, basic_model = demonstrate_basic_checkpointing()
        
        # 2. Advanced checkpointing demonstration
        advanced_manager, advanced_model = demonstrate_advanced_checkpointing()
        
        # 3. Checkpoint validation and comparison
        demonstrate_checkpoint_validation_and_comparison(basic_manager)
        
        # 4. Checkpoint management
        demonstrate_checkpoint_management(advanced_manager)
        
        # 5. Experiment tracker integration
        experiment_tracker = demonstrate_experiment_tracker_integration()
        
        # 6. Logging integration
        demonstrate_logging_integration()
        
        logger.info("\n🎉 Demonstration completed successfully!")
        logger.info("=" * 80)
        logger.info("Key features demonstrated:")
        logger.info("✅ Basic checkpointing with interval-based saving")
        logger.info("✅ Advanced checkpointing with best-only saving")
        logger.info("✅ Checkpoint validation and integrity checking")
        logger.info("✅ Checkpoint comparison and analysis")
        logger.info("✅ Checkpoint management and export")
        logger.info("✅ Experiment tracker integration")
        logger.info("✅ Comprehensive logging integration")
        logger.info("=" * 80)
        logger.info("Check the following directories for generated files:")
        logger.info("📁 demo_checkpoints/ - Basic checkpointing demo")
        logger.info("📁 demo_checkpoints_advanced/ - Advanced checkpointing demo")
        logger.info("📁 checkpoints/experiment_tracker/ - Experiment tracker checkpoints")
        logger.info("📁 logs/ - Logging output")
        logger.info("📁 runs/tensorboard/ - TensorBoard logs")
        logger.info("📁 exported_checkpoints/ - Exported checkpoints")
        
    except Exception as e:
        logger.error(f"❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)






