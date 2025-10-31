import torch
import torch.nn as nn
import numpy as np
import unittest
from typing import Dict, Any
import sys
import os

# Add the current directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from custom_nn_modules import (
    FacebookContentAnalysisTransformer,
    MultiModalFacebookAnalyzer,
    TemporalEngagementPredictor,
    AdaptiveContentOptimizer,
    FacebookDiffusionUNet
)

from forward_reverse_diffusion import (
    DiffusionConfig,
    BetaSchedule,
    ForwardDiffusionProcess,
    ReverseDiffusionProcess,
    DiffusionTraining,
    DiffusionVisualizer
)


class TestCustomNNModules(unittest.TestCase):
    """Test cases for custom nn.Module implementations"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 2
        self.seq_length = 128
        self.feature_dim = 768
    
    def test_facebook_content_analysis_transformer(self):
        """Test FacebookContentAnalysisTransformer"""
        model = FacebookContentAnalysisTransformer().to(self.device)
        
        # Test input
        input_ids = torch.randint(0, 1000, (self.batch_size, self.seq_length)).to(self.device)
        attention_mask = torch.ones(self.batch_size, self.seq_length).to(self.device)
        
        # Forward pass
        outputs = model(input_ids, attention_mask)
        
        # Check outputs
        self.assertIn('engagement_score', outputs)
        self.assertIn('viral_potential', outputs)
        self.assertIn('content_quality', outputs)
        self.assertIn('hidden_states', outputs)
        self.assertIn('pooled_output', outputs)
        
        # Check shapes
        self.assertEqual(outputs['engagement_score'].shape, (self.batch_size, 10))
        self.assertEqual(outputs['viral_potential'].shape, (self.batch_size, 1))
        self.assertEqual(outputs['content_quality'].shape, (self.batch_size, 1))
        self.assertEqual(outputs['hidden_states'].shape, (self.batch_size, self.seq_length, 768))
        self.assertEqual(outputs['pooled_output'].shape, (self.batch_size, 768))
        
        # Check value ranges
        self.assertTrue(torch.all(outputs['viral_potential'] >= 0) and torch.all(outputs['viral_potential'] <= 1))
        self.assertTrue(torch.all(outputs['content_quality'] >= 0) and torch.all(outputs['content_quality'] <= 1))
        
        print("✓ FacebookContentAnalysisTransformer test passed")
    
    def test_multimodal_facebook_analyzer(self):
        """Test MultiModalFacebookAnalyzer"""
        model = MultiModalFacebookAnalyzer().to(self.device)
        
        # Test text-only input
        text_inputs = {
            'input_ids': torch.randint(0, 1000, (self.batch_size, self.seq_length)).to(self.device),
            'attention_mask': torch.ones(self.batch_size, self.seq_length).to(self.device)
        }
        
        # Forward pass
        outputs = model(text_inputs)
        
        # Check outputs
        self.assertIn('engagement_score', outputs)
        self.assertIn('viral_potential', outputs)
        self.assertIn('content_quality', outputs)
        self.assertIn('estimated_reach', outputs)
        self.assertIn('fused_features', outputs)
        
        # Check shapes
        self.assertEqual(outputs['engagement_score'].shape, (self.batch_size, 10))
        self.assertEqual(outputs['viral_potential'].shape, (self.batch_size, 1))
        self.assertEqual(outputs['content_quality'].shape, (self.batch_size, 1))
        self.assertEqual(outputs['estimated_reach'].shape, (self.batch_size, 1))
        
        # Check value ranges
        self.assertTrue(torch.all(outputs['viral_potential'] >= 0) and torch.all(outputs['viral_potential'] <= 1))
        self.assertTrue(torch.all(outputs['content_quality'] >= 0) and torch.all(outputs['content_quality'] <= 1))
        self.assertTrue(torch.all(outputs['estimated_reach'] >= 0) and torch.all(outputs['estimated_reach'] <= 1))
        
        print("✓ MultiModalFacebookAnalyzer test passed")
    
    def test_temporal_engagement_predictor(self):
        """Test TemporalEngagementPredictor"""
        model = TemporalEngagementPredictor().to(self.device)
        
        # Test input
        content_features = torch.randn(self.batch_size, self.feature_dim).to(self.device)
        
        # Forward pass
        outputs = model(content_features)
        
        # Check outputs
        self.assertIn('engagement_score', outputs)
        self.assertIn('temporal_pattern', outputs)
        self.assertIn('peak_time_probability', outputs)
        self.assertIn('attention_weights', outputs)
        self.assertIn('temporal_features', outputs)
        
        # Check shapes
        self.assertEqual(outputs['engagement_score'].shape, (self.batch_size, 10))
        self.assertEqual(outputs['temporal_pattern'].shape, (self.batch_size, 24))
        self.assertEqual(outputs['peak_time_probability'].shape, (self.batch_size, 1))
        self.assertEqual(outputs['temporal_features'].shape, (self.batch_size, 24, 512))
        
        # Check value ranges
        self.assertTrue(torch.all(outputs['peak_time_probability'] >= 0) and torch.all(outputs['peak_time_probability'] <= 1))
        
        print("✓ TemporalEngagementPredictor test passed")
    
    def test_adaptive_content_optimizer(self):
        """Test AdaptiveContentOptimizer"""
        model = AdaptiveContentOptimizer().to(self.device)
        
        # Test input
        content_features = torch.randn(self.batch_size, self.feature_dim).to(self.device)
        
        # Forward pass
        outputs = model(content_features)
        
        # Check outputs
        self.assertIn('predicted_performance', outputs)
        self.assertIn('content_type_probabilities', outputs)
        self.assertIn('optimization_suggestions', outputs)
        self.assertIn('optimization_trajectory', outputs)
        self.assertIn('analyzed_features', outputs)
        
        # Check shapes
        self.assertEqual(outputs['predicted_performance'].shape, (self.batch_size, 1))
        self.assertEqual(outputs['content_type_probabilities'].shape, (self.batch_size, 5))
        self.assertEqual(outputs['optimization_suggestions'].shape, (self.batch_size, self.feature_dim))
        self.assertEqual(outputs['optimization_trajectory'].shape, (self.batch_size, 5, 256))
        
        # Check value ranges
        self.assertTrue(torch.all(outputs['predicted_performance'] >= 0) and torch.all(outputs['predicted_performance'] <= 1))
        self.assertTrue(torch.all(outputs['content_type_probabilities'] >= 0) and torch.all(outputs['content_type_probabilities'] <= 1))
        self.assertTrue(torch.all(torch.sum(outputs['content_type_probabilities'], dim=1) - 1.0 < 1e-6))
        
        print("✓ AdaptiveContentOptimizer test passed")
    
    def test_facebook_diffusion_unet(self):
        """Test FacebookDiffusionUNet"""
        model = FacebookDiffusionUNet().to(self.device)
        
        # Test input
        x = torch.randn(self.batch_size, 3, 64, 64).to(self.device)
        timesteps = torch.randint(0, 1000, (self.batch_size,)).to(self.device)
        context = torch.randn(self.batch_size, 77, 768).to(self.device)
        
        # Forward pass
        output = model(x, timesteps, context)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, 3, 64, 64))
        
        print("✓ FacebookDiffusionUNet test passed")


class TestDiffusionProcesses(unittest.TestCase):
    """Test cases for diffusion processes"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 2
        self.image_size = 32
        
        # Create configuration
        self.config = DiffusionConfig(
            num_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule=BetaSchedule.COSINE,
            prediction_type="epsilon",
            loss_type="mse"
        )
        
        # Create a simple UNet model for testing
        class SimpleUNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
                self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
                self.time_embed = nn.Linear(1, 64)
                self.final = nn.Conv2d(64, 3, 3, padding=1)
            
            def forward(self, x, t, context=None):
                t_emb = self.time_embed(t.float().unsqueeze(-1))
                t_emb = t_emb.view(t_emb.shape[0], t_emb.shape[1], 1, 1)
                h = torch.relu(self.conv1(x))
                h = h + t_emb
                h = torch.relu(self.conv2(h))
                return self.final(h)
        
        self.model = SimpleUNet().to(self.device)
    
    def test_forward_diffusion_process(self):
        """Test ForwardDiffusionProcess"""
        forward_process = ForwardDiffusionProcess(self.config)
        
        # Test input
        x_start = torch.randn(self.batch_size, 3, self.image_size, self.image_size).to(self.device)
        t = torch.randint(0, self.config.num_timesteps, (self.batch_size,)).to(self.device)
        
        # Test q_sample
        x_t, noise = forward_process.q_sample(x_start, t)
        
        # Check shapes
        self.assertEqual(x_t.shape, x_start.shape)
        self.assertEqual(noise.shape, x_start.shape)
        
        # Test q_posterior_mean_variance
        posterior_mean, posterior_variance, posterior_log_variance = forward_process.q_posterior_mean_variance(
            x_start, x_t, t
        )
        
        # Check shapes
        self.assertEqual(posterior_mean.shape, x_start.shape)
        self.assertEqual(posterior_variance.shape, x_start.shape)
        self.assertEqual(posterior_log_variance.shape, x_start.shape)
        
        print("✓ ForwardDiffusionProcess test passed")
    
    def test_reverse_diffusion_process(self):
        """Test ReverseDiffusionProcess"""
        reverse_process = ReverseDiffusionProcess(self.config, self.model)
        
        # Test input
        x_t = torch.randn(self.batch_size, 3, self.image_size, self.image_size).to(self.device)
        t = torch.randint(0, self.config.num_timesteps, (self.batch_size,)).to(self.device)
        context = torch.randn(self.batch_size, 77, 768).to(self.device)
        
        # Test p_mean_variance
        out = reverse_process.p_mean_variance(x_t, t, context)
        
        # Check outputs
        self.assertIn('mean', out)
        self.assertIn('variance', out)
        self.assertIn('log_variance', out)
        self.assertIn('pred_x_start', out)
        self.assertIn('pred_epsilon', out)
        
        # Check shapes
        self.assertEqual(out['mean'].shape, x_t.shape)
        self.assertEqual(out['variance'].shape, x_t.shape)
        self.assertEqual(out['log_variance'].shape, x_t.shape)
        self.assertEqual(out['pred_x_start'].shape, x_t.shape)
        self.assertEqual(out['pred_epsilon'].shape, x_t.shape)
        
        # Test p_sample
        sample = reverse_process.p_sample(x_t, t, context, return_dict=False)
        self.assertEqual(sample.shape, x_t.shape)
        
        print("✓ ReverseDiffusionProcess test passed")
    
    def test_diffusion_training(self):
        """Test DiffusionTraining"""
        forward_process = ForwardDiffusionProcess(self.config)
        trainer = DiffusionTraining(self.model, forward_process, self.config)
        
        # Test input
        x_start = torch.randn(self.batch_size, 3, self.image_size, self.image_size).to(self.device)
        context = torch.randn(self.batch_size, 77, 768).to(self.device)
        
        # Test compute_loss
        loss = trainer.compute_loss(x_start, context)
        
        # Check loss
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.shape, ())
        self.assertTrue(loss.item() >= 0)
        
        print("✓ DiffusionTraining test passed")
    
    def test_diffusion_visualizer(self):
        """Test DiffusionVisualizer"""
        forward_process = ForwardDiffusionProcess(self.config)
        visualizer = DiffusionVisualizer(forward_process)
        
        # Test input
        x_start = torch.randn(self.batch_size, 3, self.image_size, self.image_size).to(self.device)
        
        # Test visualize_forward_process
        forward_samples = visualizer.visualize_forward_process(x_start, num_steps=5)
        
        # Check samples
        self.assertEqual(len(forward_samples), 6)  # Original + 5 steps
        for i, sample in enumerate(forward_samples):
            self.assertEqual(sample.shape, x_start.shape)
        
        # Test visualize_reverse_process
        reverse_process = ReverseDiffusionProcess(self.config, self.model)
        reverse_samples = visualizer.visualize_reverse_process(
            reverse_process, (self.batch_size, 3, self.image_size, self.image_size), num_steps=5
        )
        
        # Check samples
        self.assertEqual(len(reverse_samples), 6)  # Noise + 5 steps
        for i, sample in enumerate(reverse_samples):
            self.assertEqual(sample.shape, (self.batch_size, 3, self.image_size, self.image_size))
        
        print("✓ DiffusionVisualizer test passed")
    
    def test_beta_schedules(self):
        """Test different beta schedules"""
        schedules = [BetaSchedule.LINEAR, BetaSchedule.COSINE, BetaSchedule.SIGMOID, BetaSchedule.QUADRATIC]
        
        for schedule in schedules:
            config = DiffusionConfig(beta_schedule=schedule)
            forward_process = ForwardDiffusionProcess(config)
            
            # Check that betas are properly bounded
            self.assertTrue(torch.all(forward_process.betas > 0))
            self.assertTrue(torch.all(forward_process.betas < 1))
            
            # Check that alphas are properly bounded
            self.assertTrue(torch.all(forward_process.alphas > 0))
            self.assertTrue(torch.all(forward_process.alphas < 1))
            
            print(f"✓ Beta schedule {schedule.value} test passed")


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 2
        self.seq_length = 128
        self.feature_dim = 768
        self.image_size = 32
    
    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow"""
        # 1. Create content analysis transformer
        transformer = FacebookContentAnalysisTransformer().to(self.device)
        
        # 2. Create multimodal analyzer
        multimodal = MultiModalFacebookAnalyzer().to(self.device)
        
        # 3. Create temporal predictor
        temporal = TemporalEngagementPredictor().to(self.device)
        
        # 4. Create adaptive optimizer
        adaptive = AdaptiveContentOptimizer().to(self.device)
        
        # 5. Create diffusion model
        unet = FacebookDiffusionUNet().to(self.device)
        
        # 6. Create diffusion processes
        config = DiffusionConfig(num_timesteps=50)  # Shorter for testing
        forward_process = ForwardDiffusionProcess(config)
        reverse_process = ReverseDiffusionProcess(config, unet)
        trainer = DiffusionTraining(unet, forward_process, config)
        
        # 7. Test complete pipeline
        # Input data
        input_ids = torch.randint(0, 1000, (self.batch_size, self.seq_length)).to(self.device)
        attention_mask = torch.ones(self.batch_size, self.seq_length).to(self.device)
        text_inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
        
        # Content analysis
        transformer_outputs = transformer(input_ids, attention_mask)
        multimodal_outputs = multimodal(text_inputs)
        
        # Temporal analysis
        content_features = transformer_outputs['pooled_output']
        temporal_outputs = temporal(content_features)
        
        # Adaptive optimization
        adaptive_outputs = adaptive(content_features)
        
        # Diffusion generation
        x_start = torch.randn(self.batch_size, 3, self.image_size, self.image_size).to(self.device)
        context = torch.randn(self.batch_size, 77, 768).to(self.device)
        
        # Training
        loss = trainer.compute_loss(x_start, context)
        
        # Sampling
        samples = reverse_process.p_sample_loop(
            (self.batch_size, 3, self.image_size, self.image_size),
            context,
            return_dict=False
        )
        
        # Verify all outputs
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(samples.shape, (self.batch_size, 3, self.image_size, self.image_size))
        
        print("✓ End-to-end workflow test passed")
    
    def test_memory_efficiency(self):
        """Test memory efficiency of models"""
        import gc
        
        # Test memory usage for large batch
        large_batch_size = 8
        
        models = [
            FacebookContentAnalysisTransformer().to(self.device),
            MultiModalFacebookAnalyzer().to(self.device),
            TemporalEngagementPredictor().to(self.device),
            AdaptiveContentOptimizer().to(self.device),
            FacebookDiffusionUNet().to(self.device)
        ]
        
        for model in models:
            # Clear memory
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Test with large batch
            if isinstance(model, FacebookContentAnalysisTransformer):
                input_ids = torch.randint(0, 1000, (large_batch_size, self.seq_length)).to(self.device)
                attention_mask = torch.ones(large_batch_size, self.seq_length).to(self.device)
                _ = model(input_ids, attention_mask)
            elif isinstance(model, MultiModalFacebookAnalyzer):
                text_inputs = {
                    'input_ids': torch.randint(0, 1000, (large_batch_size, self.seq_length)).to(self.device),
                    'attention_mask': torch.ones(large_batch_size, self.seq_length).to(self.device)
                }
                _ = model(text_inputs)
            elif isinstance(model, TemporalEngagementPredictor):
                content_features = torch.randn(large_batch_size, self.feature_dim).to(self.device)
                _ = model(content_features)
            elif isinstance(model, AdaptiveContentOptimizer):
                content_features = torch.randn(large_batch_size, self.feature_dim).to(self.device)
                _ = model(content_features)
            elif isinstance(model, FacebookDiffusionUNet):
                x = torch.randn(large_batch_size, 3, self.image_size, self.image_size).to(self.device)
                timesteps = torch.randint(0, 1000, (large_batch_size,)).to(self.device)
                context = torch.randn(large_batch_size, 77, 768).to(self.device)
                _ = model(x, timesteps, context)
            
            print(f"✓ Memory efficiency test passed for {model.__class__.__name__}")


def run_performance_benchmark():
    """Run performance benchmark"""
    print("\n" + "="*50)
    print("PERFORMANCE BENCHMARK")
    print("="*50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Benchmark parameters
    batch_sizes = [1, 2, 4, 8]
    seq_lengths = [64, 128, 256]
    
    models = {
        'FacebookContentAnalysisTransformer': FacebookContentAnalysisTransformer().to(device),
        'MultiModalFacebookAnalyzer': MultiModalFacebookAnalyzer().to(device),
        'TemporalEngagementPredictor': TemporalEngagementPredictor().to(device),
        'AdaptiveContentOptimizer': AdaptiveContentOptimizer().to(device),
    }
    
    for model_name, model in models.items():
        print(f"\n{model_name}:")
        print("-" * 30)
        
        for batch_size in batch_sizes:
            for seq_length in seq_lengths:
                try:
                    # Prepare input
                    if model_name == 'FacebookContentAnalysisTransformer':
                        input_ids = torch.randint(0, 1000, (batch_size, seq_length)).to(device)
                        attention_mask = torch.ones(batch_size, seq_length).to(device)
                        
                        # Warmup
                        for _ in range(3):
                            _ = model(input_ids, attention_mask)
                        
                        # Benchmark
                        torch.cuda.synchronize() if torch.cuda.is_available() else None
                        start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
                        end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
                        
                        if torch.cuda.is_available():
                            start_time.record()
                        else:
                            import time
                            start_time = time.time()
                        
                        for _ in range(10):
                            _ = model(input_ids, attention_mask)
                        
                        if torch.cuda.is_available():
                            end_time.record()
                            torch.cuda.synchronize()
                            elapsed_time = start_time.elapsed_time(end_time) / 10
                        else:
                            import time
                            end_time = time.time()
                            elapsed_time = (end_time - start_time) / 10
                        
                        print(f"  Batch {batch_size}, Seq {seq_length}: {elapsed_time:.2f}ms")
                        
                    elif model_name == 'MultiModalFacebookAnalyzer':
                        text_inputs = {
                            'input_ids': torch.randint(0, 1000, (batch_size, seq_length)).to(device),
                            'attention_mask': torch.ones(batch_size, seq_length).to(device)
                        }
                        
                        # Warmup
                        for _ in range(3):
                            _ = model(text_inputs)
                        
                        # Benchmark
                        torch.cuda.synchronize() if torch.cuda.is_available() else None
                        start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
                        end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
                        
                        if torch.cuda.is_available():
                            start_time.record()
                        else:
                            import time
                            start_time = time.time()
                        
                        for _ in range(10):
                            _ = model(text_inputs)
                        
                        if torch.cuda.is_available():
                            end_time.record()
                            torch.cuda.synchronize()
                            elapsed_time = start_time.elapsed_time(end_time) / 10
                        else:
                            import time
                            end_time = time.time()
                            elapsed_time = (end_time - start_time) / 10
                        
                        print(f"  Batch {batch_size}, Seq {seq_length}: {elapsed_time:.2f}ms")
                        
                    elif model_name in ['TemporalEngagementPredictor', 'AdaptiveContentOptimizer']:
                        content_features = torch.randn(batch_size, 768).to(device)
                        
                        # Warmup
                        for _ in range(3):
                            _ = model(content_features)
                        
                        # Benchmark
                        torch.cuda.synchronize() if torch.cuda.is_available() else None
                        start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
                        end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
                        
                        if torch.cuda.is_available():
                            start_time.record()
                        else:
                            import time
                            start_time = time.time()
                        
                        for _ in range(10):
                            _ = model(content_features)
                        
                        if torch.cuda.is_available():
                            end_time.record()
                            torch.cuda.synchronize()
                            elapsed_time = start_time.elapsed_time(end_time) / 10
                        else:
                            import time
                            end_time = time.time()
                            elapsed_time = (end_time - start_time) / 10
                        
                        print(f"  Batch {batch_size}: {elapsed_time:.2f}ms")
                        
                except Exception as e:
                    print(f"  Batch {batch_size}, Seq {seq_length}: Error - {str(e)}")


if __name__ == "__main__":
    print("Running Custom NN.Module Tests...")
    print("="*50)
    
    # Run unit tests
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run performance benchmark
    run_performance_benchmark()
    
    print("\n" + "="*50)
    print("ALL TESTS COMPLETED!")
    print("="*50)


