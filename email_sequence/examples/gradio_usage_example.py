from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, List, Any
import sys
from gradio_app import GradioEmailSequenceApp
from models.sequence import EmailSequence, SequenceStep
from models.subscriber import Subscriber
from models.template import EmailTemplate
from typing import Any, List, Dict, Optional
"""
Gradio Interface Usage Example

This example demonstrates how to use the Gradio interface
programmatically and integrate it with other systems.
"""


# Add the parent directory to the path
sys.path.append(str(Path(__file__).parent.parent))


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GradioInterfaceExample:
    """Example class demonstrating Gradio interface usage"""
    
    def __init__(self) -> Any:
        self.app = GradioEmailSequenceApp()
        logger.info("Gradio Interface Example initialized")
    
    async def example_sequence_generation(self) -> Any:
        """Example of sequence generation workflow"""
        
        logger.info("=== Sequence Generation Example ===")
        
        # Example 1: Generate a welcome sequence
        welcome_sequence = await self._generate_welcome_sequence()
        logger.info(f"Generated welcome sequence with {len(welcome_sequence.steps)} steps")
        
        # Example 2: Generate a conversion sequence
        conversion_sequence = await self._generate_conversion_sequence()
        logger.info(f"Generated conversion sequence with {len(conversion_sequence.steps)} steps")
        
        # Example 3: Generate a re-engagement sequence
        reengagement_sequence = await self._generate_reengagement_sequence()
        logger.info(f"Generated re-engagement sequence with {len(reengagement_sequence.steps)} steps")
        
        return [welcome_sequence, conversion_sequence, reengagement_sequence]
    
    async def _generate_welcome_sequence(self) -> EmailSequence:
        """Generate a welcome email sequence"""
        
        # Create target subscriber
        subscriber = Subscriber(
            id="welcome_example",
            email="newuser@example.com",
            name="New User",
            company="Example Corp",
            interests=["productivity", "automation"],
            industry="Technology"
        )
        
        # Generate sequence using the app's generator
        sequence = await self.app.sequence_generator.generate_sequence(
            target_audience=subscriber,
            templates=self.app.sample_templates,
            config={
                "model_type": "GPT-3.5",
                "sequence_length": 3,
                "creativity_level": 0.7,
                "industry_focus": "Technology"
            }
        )
        
        return sequence
    
    async def _generate_conversion_sequence(self) -> EmailSequence:
        """Generate a conversion email sequence"""
        
        subscriber = Subscriber(
            id="conversion_example",
            email="prospect@example.com",
            name="Prospect User",
            company="Prospect Corp",
            interests=["marketing", "growth"],
            industry="Marketing"
        )
        
        sequence = await self.app.sequence_generator.generate_sequence(
            target_audience=subscriber,
            templates=self.app.sample_templates,
            config={
                "model_type": "GPT-4",
                "sequence_length": 5,
                "creativity_level": 0.8,
                "industry_focus": "Marketing"
            }
        )
        
        return sequence
    
    async def _generate_reengagement_sequence(self) -> EmailSequence:
        """Generate a re-engagement email sequence"""
        
        subscriber = Subscriber(
            id="reengagement_example",
            email="inactive@example.com",
            name="Inactive User",
            company="Inactive Corp",
            interests=["recovery", "reactivation"],
            industry="General"
        )
        
        sequence = await self.app.sequence_generator.generate_sequence(
            target_audience=subscriber,
            templates=self.app.sample_templates,
            config={
                "model_type": "Claude",
                "sequence_length": 4,
                "creativity_level": 0.6,
                "industry_focus": "General"
            }
        )
        
        return sequence
    
    async def example_evaluation_workflow(self, sequences: List[EmailSequence]):
        """Example of evaluation workflow"""
        
        logger.info("=== Evaluation Workflow Example ===")
        
        evaluation_results = []
        
        for i, sequence in enumerate(sequences, 1):
            logger.info(f"Evaluating sequence {i}: {sequence.name}")
            
            # Evaluate the sequence
            results = await self.app.evaluator.evaluate_sequence(
                sequence=sequence,
                subscribers=self.app.sample_subscribers,
                templates=self.app.sample_templates,
                config={
                    "enable_content_quality": True,
                    "enable_engagement": True,
                    "enable_business_impact": True,
                    "enable_technical": True,
                    "content_weight": 0.3,
                    "engagement_weight": 0.3,
                    "business_impact_weight": 0.2,
                    "technical_weight": 0.2
                }
            )
            
            evaluation_results.append({
                "sequence_id": sequence.id,
                "sequence_name": sequence.name,
                "results": results
            })
            
            # Log key metrics
            overall_score = results.get("overall_metrics", {}).get("overall_score", 0)
            logger.info(f"Sequence {i} overall score: {overall_score:.3f}")
        
        return evaluation_results
    
    async def example_training_workflow(self) -> Any:
        """Example of training workflow"""
        
        logger.info("=== Training Workflow Example ===")
        
        # Configure training parameters
        training_config = {
            "early_stopping": {
                "patience": 10,
                "min_delta": 0.001
            },
            "learning_rate_scheduler": {
                "scheduler_type": "cosine",
                "initial_lr": 0.001
            },
            "gradient_management": {
                "max_grad_norm": 1.0,
                "enable_gradient_clipping": True,
                "enable_nan_inf_check": True
            },
            "training": {
                "max_epochs": 50,
                "batch_size": 32
            }
        }
        
        # Simulate training process
        logger.info("Starting training simulation...")
        
        # This would normally train an actual model
        # For this example, we'll simulate the process
        training_log = self._simulate_training_process(training_config)
        
        logger.info("Training simulation completed")
        logger.info(f"Training log length: {len(training_log)} lines")
        
        return {
            "config": training_config,
            "log": training_log,
            "final_metrics": {
                "final_loss": 0.123,
                "best_loss": 0.098,
                "epochs_trained": 50,
                "early_stopping_triggered": False
            }
        }
    
    def _simulate_training_process(self, config: Dict) -> str:
        """Simulate a training process"""
        
        log_lines = []
        max_epochs = config["training"]["max_epochs"]
        
        for epoch in range(max_epochs):
            # Simulate training metrics
            train_loss = 0.5 * (0.95 ** epoch) + 0.1
            val_loss = 0.6 * (0.93 ** epoch) + 0.12
            
            log_lines.append(f"Epoch {epoch+1}/{max_epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Simulate early stopping check
            if epoch > 20 and val_loss < 0.15:
                log_lines.append("Early stopping triggered")
                break
        
        return "\n".join(log_lines)
    
    async def example_gradient_management(self) -> Any:
        """Example of gradient management workflow"""
        
        logger.info("=== Gradient Management Example ===")
        
        # Configure gradient management
        gradient_config = {
            "max_grad_norm": 1.0,
            "clip_type": "norm",
            "enable_nan_inf_check": True,
            "replace_nan_with": 0.0,
            "replace_inf_with": 1e6,
            "enable_gradient_monitoring": True,
            "verbose_logging": False,
            "adaptive_clipping": False,
            "adaptive_window_size": 100
        }
        
        # Simulate gradient management process
        logger.info("Testing gradient management...")
        
        gradient_log = self._simulate_gradient_management(gradient_config)
        
        logger.info("Gradient management test completed")
        
        return {
            "config": gradient_config,
            "log": gradient_log,
            "summary": {
                "total_steps": 50,
                "health_issues": {"unhealthy_steps": 2},
                "nan_inf_summary": {"total_replacements": 1},
                "clipping_summary": {"total_clipped": 5}
            }
        }
    
    def _simulate_gradient_management(self, config: Dict) -> str:
        """Simulate gradient management process"""
        
        log_lines = []
        total_steps = 50
        
        for step in range(total_steps):
            # Simulate gradient statistics
            grad_norm = 0.8 + (step % 10) * 0.1
            is_healthy = grad_norm < config["max_grad_norm"]
            is_clipped = grad_norm > config["max_grad_norm"]
            
            if step % 10 == 0:
                log_lines.append(f"Step {step}: GradNorm={grad_norm:.3f}, Healthy={is_healthy}, Clipped={is_clipped}")
        
        return "\n".join(log_lines)
    
    async def example_complete_workflow(self) -> Any:
        """Complete workflow example combining all features"""
        
        logger.info("=== Complete Workflow Example ===")
        
        # Step 1: Generate sequences
        logger.info("Step 1: Generating sequences...")
        sequences = await self.example_sequence_generation()
        
        # Step 2: Evaluate sequences
        logger.info("Step 2: Evaluating sequences...")
        evaluation_results = await self.example_evaluation_workflow(sequences)
        
        # Step 3: Train model (if needed)
        logger.info("Step 3: Training model...")
        training_results = await self.example_training_workflow()
        
        # Step 4: Test gradient management
        logger.info("Step 4: Testing gradient management...")
        gradient_results = await self.example_gradient_management()
        
        # Step 5: Generate final report
        logger.info("Step 5: Generating final report...")
        final_report = self._generate_final_report(
            sequences, evaluation_results, training_results, gradient_results
        )
        
        return final_report
    
    def _generate_final_report(
        self,
        sequences: List[EmailSequence],
        evaluation_results: List[Dict],
        training_results: Dict,
        gradient_results: Dict
    ) -> Dict:
        """Generate a comprehensive final report"""
        
        # Calculate summary statistics
        total_sequences = len(sequences)
        avg_sequence_length = sum(len(s.steps) for s in sequences) / total_sequences
        
        # Calculate average evaluation scores
        avg_overall_score = 0
        if evaluation_results:
            scores = [
                result["results"].get("overall_metrics", {}).get("overall_score", 0)
                for result in evaluation_results
            ]
            avg_overall_score = sum(scores) / len(scores)
        
        report = {
            "summary": {
                "total_sequences_generated": total_sequences,
                "average_sequence_length": avg_sequence_length,
                "average_evaluation_score": avg_overall_score,
                "training_completed": True,
                "gradient_management_tested": True
            },
            "sequences": [
                {
                    "id": seq.id,
                    "name": seq.name,
                    "length": len(seq.steps),
                    "description": seq.description
                }
                for seq in sequences
            ],
            "evaluations": evaluation_results,
            "training": training_results,
            "gradient_management": gradient_results,
            "recommendations": self._generate_recommendations(
                sequences, evaluation_results, training_results
            )
        }
        
        return report
    
    def _generate_recommendations(
        self,
        sequences: List[EmailSequence],
        evaluation_results: List[Dict],
        training_results: Dict
    ) -> List[str]:
        """Generate recommendations based on results"""
        
        recommendations = []
        
        # Analyze sequence performance
        if evaluation_results:
            scores = [
                result["results"].get("overall_metrics", {}).get("overall_score", 0)
                for result in evaluation_results
            ]
            
            if min(scores) < 0.6:
                recommendations.append("Consider improving sequence quality - some sequences scored below 0.6")
            
            if max(scores) > 0.8:
                recommendations.append("Excellent sequence quality achieved - consider scaling successful patterns")
        
        # Analyze training results
        if training_results.get("final_metrics", {}).get("early_stopping_triggered"):
            recommendations.append("Early stopping was triggered - consider adjusting learning rate or patience")
        
        # General recommendations
        recommendations.extend([
            "Monitor sequence performance in production",
            "Regularly update training data with new examples",
            "Consider A/B testing different sequence variations",
            "Implement feedback loops for continuous improvement"
        ])
        
        return recommendations
    
    def save_results(self, results: Dict, filename: str = "workflow_results.json"):
        """Save results to a JSON file"""
        
        output_path = Path("./outputs") / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to: {output_path}")
        return output_path


async def main():
    """Main example function"""
    
    logger.info("Starting Gradio Interface Usage Example")
    
    # Create example instance
    example = GradioInterfaceExample()
    
    try:
        # Run complete workflow
        results = await example.example_complete_workflow()
        
        # Save results
        output_path = example.save_results(results)
        
        # Display summary
        summary = results["summary"]
        logger.info("=" * 60)
        logger.info("WORKFLOW COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        logger.info(f"Sequences Generated: {summary['total_sequences_generated']}")
        logger.info(f"Average Sequence Length: {summary['average_sequence_length']:.1f}")
        logger.info(f"Average Evaluation Score: {summary['average_evaluation_score']:.3f}")
        logger.info(f"Training Completed: {summary['training_completed']}")
        logger.info(f"Gradient Management Tested: {summary['gradient_management_tested']}")
        logger.info(f"Results saved to: {output_path}")
        logger.info("=" * 60)
        
        # Display recommendations
        logger.info("RECOMMENDATIONS:")
        for i, rec in enumerate(results["recommendations"], 1):
            logger.info(f"{i}. {rec}")
        
    except Exception as e:
        logger.error(f"Error in workflow: {e}")
        raise


if __name__ == "__main__":
    # Run the example
    asyncio.run(main()) 