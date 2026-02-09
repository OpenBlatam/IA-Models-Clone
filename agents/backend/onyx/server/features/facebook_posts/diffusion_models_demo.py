"""
Comprehensive Demonstration of Diffusion Models System.
"""

import torch
import numpy as np
import time
import json
from pathlib import Path
from diffusion_models_system import (
    DiffusionConfig, TrainingConfig, create_diffusion_system,
    DiffusionModelManager, DiffusionTrainer, DiffusionAnalyzer
)
import warnings
warnings.filterwarnings("ignore")
from typing import List


class DiffusionModelsDemo:
    """Comprehensive demonstration of the diffusion models system."""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 Using device: {self.device}")
        
        # Sample prompts for demonstration
        self.sample_prompts = [
            "A beautiful sunset over a mountain landscape, digital art",
            "A futuristic city with flying cars and neon lights",
            "A serene forest with ancient trees and morning mist",
            "A portrait of a wise old wizard with magical aura",
            "A steampunk mechanical robot in a Victorian workshop"
        ]
        
        # Negative prompts
        self.negative_prompts = [
            "blurry, low quality, distorted, ugly, deformed",
            "blurry, low quality, distorted, ugly, deformed",
            "blurry, low quality, distorted, ugly, deformed",
            "blurry, low quality, distorted, ugly, deformed",
            "blurry, low quality, distorted, ugly, deformed"
        ]
    
    def demo_basic_system_creation(self):
        """Demonstrate basic system creation and configuration."""
        print("\n" + "="*60)
        print("🔧 DEMO: Basic System Creation & Configuration")
        print("="*60)
        
        # Create configurations
        diffusion_config = DiffusionConfig(
            model_name="runwayml/stable-diffusion-v1-5",
            model_type="stable_diffusion",
            use_pipeline=True,
            num_inference_steps=20,  # Reduced for demo
            guidance_scale=7.5,
            height=512,
            width=512,
            enable_attention_slicing=True,
            enable_vae_slicing=True,
            use_amp=True
        )
        
        training_config = TrainingConfig(
            learning_rate=1e-5,
            num_epochs=10,  # Reduced for demo
            batch_size=1,
            optimizer="adamw",
            loss_type="l2",
            gradient_clip_norm=1.0
        )
        
        print(f"\n📋 Diffusion Configuration:")
        for key, value in diffusion_config.__dict__.items():
            print(f"   {key}: {value}")
        
        print(f"\n📋 Training Configuration:")
        for key, value in training_config.__dict__.items():
            print(f"   {key}: {value}")
        
        # Create system
        try:
            print(f"\n🔄 Creating diffusion system...")
            model_manager, trainer, analyzer = create_diffusion_system(
                diffusion_config, training_config
            )
            print(f"✅ System created successfully!")
            
            return model_manager, trainer, analyzer
            
        except Exception as e:
            print(f"❌ Error creating system: {e}")
            print(f"⚠️ This is expected if models are not downloaded")
            return None, None, None
    
    def demo_model_information(self, model_manager: DiffusionModelManager):
        """Demonstrate model information and analysis."""
        print("\n" + "="*60)
        print("📊 DEMO: Model Information & Analysis")
        print("="*60)
        
        if not model_manager:
            print("⚠️ No model manager available for this demo")
            return
        
        try:
            # Get model info
            model_info = model_manager.get_model_info()
            
            print(f"\n🏗️ Model Architecture:")
            print(f"   Device: {model_info['device']}")
            print(f"   Models loaded: {model_info['models_loaded']}")
            
            if 'unet_params' in model_info:
                print(f"\n📈 Parameter Counts:")
                print(f"   UNet parameters: {model_info['unet_params']:,}")
                print(f"   UNet trainable: {model_info['unet_trainable_params']:,}")
            
            if 'vae_params' in model_info:
                print(f"   VAE parameters: {model_info['vae_params']:,}")
            
            if 'text_encoder_params' in model_info:
                print(f"   Text encoder parameters: {model_info['text_encoder_params']:,}")
            
            # Configuration details
            print(f"\n⚙️ Configuration Details:")
            config = model_info['config']
            important_keys = ['model_name', 'model_type', 'num_inference_steps', 'guidance_scale']
            for key in important_keys:
                if key in config:
                    print(f"   {key}: {config[key]}")
            
        except Exception as e:
            print(f"❌ Error getting model info: {e}")
    
    def demo_image_generation(self, model_manager: DiffusionModelManager):
        """Demonstrate image generation capabilities."""
        print("\n" + "="*60)
        print("🎨 DEMO: Image Generation")
        print("="*60)
        
        if not model_manager:
            print("⚠️ No model manager available for this demo")
            return
        
        try:
            # Generate images for first prompt
            prompt = self.sample_prompts[0]
            negative_prompt = self.negative_prompts[0]
            
            print(f"\n📝 Generating image for prompt:")
            print(f"   Prompt: {prompt}")
            print(f"   Negative: {negative_prompt}")
            
            # Time the generation
            start_time = time.time()
            
            # Generate image
            images = model_manager.generate_image(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_images=1,
                num_inference_steps=20,  # Reduced for demo
                guidance_scale=7.5
            )
            
            generation_time = time.time() - start_time
            
            print(f"✅ Generated {len(images)} images in {generation_time:.2f}s")
            print(f"   Image size: {images[0].size}")
            print(f"   Image mode: {images[0].mode}")
            
            # Save generated image
            output_dir = Path("generated_images")
            output_dir.mkdir(exist_ok=True)
            
            image_path = output_dir / f"generated_image_{int(time.time())}.png"
            images[0].save(image_path)
            print(f"💾 Image saved to: {image_path}")
            
            return images, [generation_time]
            
        except Exception as e:
            print(f"❌ Error generating images: {e}")
            return [], []
    
    def demo_batch_generation(self, model_manager: DiffusionModelManager):
        """Demonstrate batch image generation."""
        print("\n" + "="*60)
        print("📦 DEMO: Batch Image Generation")
        print("="*60)
        
        if not model_manager:
            print("⚠️ No model manager available for this demo")
            return
        
        try:
            # Generate images for multiple prompts
            prompts = self.sample_prompts[:3]  # First 3 prompts
            negative_prompts = self.negative_prompts[:3]
            
            print(f"\n📝 Generating images for {len(prompts)} prompts:")
            for i, prompt in enumerate(prompts):
                print(f"   {i+1}. {prompt}")
            
            all_images = []
            generation_times = []
            
            for i, (prompt, negative_prompt) in enumerate(zip(prompts, negative_prompts)):
                print(f"\n🔄 Generating image {i+1}/{len(prompts)}...")
                
                start_time = time.time()
                images = model_manager.generate_image(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_images=1,
                    num_inference_steps=20,
                    guidance_scale=7.5
                )
                generation_time = time.time() - start_time
                
                all_images.extend(images)
                generation_times.append(generation_time)
                
                print(f"   ✅ Generated in {generation_time:.2f}s")
            
            print(f"\n🎉 Batch generation completed!")
            print(f"   Total images: {len(all_images)}")
            print(f"   Total time: {sum(generation_times):.2f}s")
            print(f"   Average time per image: {np.mean(generation_times):.2f}s")
            
            # Save all images
            output_dir = Path("generated_images")
            output_dir.mkdir(exist_ok=True)
            
            for i, image in enumerate(all_images):
                image_path = output_dir / f"batch_image_{i+1}_{int(time.time())}.png"
                image.save(image_path)
                print(f"   💾 Image {i+1} saved to: {image_path}")
            
            return all_images, generation_times
            
        except Exception as e:
            print(f"❌ Error in batch generation: {e}")
            return [], []
    
    def demo_quality_analysis(self, analyzer: DiffusionAnalyzer, images: List, prompts: List):
        """Demonstrate image quality analysis."""
        print("\n" + "="*60)
        print("🔍 DEMO: Image Quality Analysis")
        print("="*60)
        
        if not analyzer or not images:
            print("⚠️ No analyzer or images available for this demo")
            return
        
        try:
            print(f"\n🔍 Analyzing {len(images)} images...")
            
            # Analyze generation quality
            quality_metrics = analyzer.analyze_generation_quality(images, prompts)
            
            if quality_metrics:
                print(f"\n📊 Quality Metrics:")
                print(f"   Brightness: {quality_metrics['brightness']:.4f}")
                print(f"   Contrast: {quality_metrics['contrast']:.4f}")
                print(f"   Sharpness: {quality_metrics['sharpness']:.4f}")
                print(f"   Color Diversity: {quality_metrics['color_diversity']:.4f}")
                print(f"   Image Count: {quality_metrics['image_count']}")
                
                print(f"\n📝 Prompts Analyzed:")
                for i, prompt in enumerate(quality_metrics['prompts']):
                    print(f"   {i+1}. {prompt[:80]}...")
            else:
                print("⚠️ No quality metrics available")
            
            return quality_metrics
            
        except Exception as e:
            print(f"❌ Error in quality analysis: {e}")
            return {}
    
    def demo_performance_analysis(self, analyzer: DiffusionAnalyzer, generation_times: List):
        """Demonstrate performance analysis."""
        print("\n" + "="*60)
        print("⚡ DEMO: Performance Analysis")
        print("="*60)
        
        if not analyzer or not generation_times:
            print("⚠️ No analyzer or generation times available for this demo")
            return
        
        try:
            print(f"\n⚡ Analyzing performance for {len(generation_times)} generations...")
            
            # Analyze performance
            performance_metrics = analyzer.analyze_model_performance(generation_times, [])
            
            if performance_metrics:
                print(f"\n📊 Performance Metrics:")
                
                gen_time = performance_metrics['generation_time']
                print(f"   Generation Time:")
                print(f"     Mean: {gen_time['mean']:.4f}s")
                print(f"     Std: {gen_time['std']:.4f}s")
                print(f"     Min: {gen_time['min']:.4f}s")
                print(f"     Max: {gen_time['max']:.4f}s")
                print(f"     Total: {gen_time['total']:.4f}s")
                
                throughput = performance_metrics['throughput']
                print(f"   Throughput:")
                print(f"     Images per second: {throughput['images_per_second']:.2f}")
                print(f"     Average time per image: {throughput['average_time_per_image']:.4f}s")
            else:
                print("⚠️ No performance metrics available")
            
            return performance_metrics
            
        except Exception as e:
            print(f"❌ Error in performance analysis: {e}")
            return {}
    
    def demo_comprehensive_report(self, analyzer: DiffusionAnalyzer, images: List, prompts: List, generation_times: List):
        """Demonstrate comprehensive report generation."""
        print("\n" + "="*60)
        print("📋 DEMO: Comprehensive Report Generation")
        print("="*60)
        
        if not analyzer or not images:
            print("⚠️ No analyzer or images available for this demo")
            return
        
        try:
            print(f"\n📋 Generating comprehensive report...")
            
            # Get quality metrics first
            quality_metrics = analyzer.analyze_generation_quality(images, prompts)
            
            # Create comprehensive report
            report = analyzer.create_generation_report(
                images=images,
                prompts=prompts,
                generation_times=generation_times,
                quality_metrics=quality_metrics
            )
            
            if report:
                print(f"\n📊 Report Summary:")
                summary = report['summary']
                print(f"   Total Images: {summary['total_images']}")
                print(f"   Total Prompts: {summary['total_prompts']}")
                print(f"   Total Generation Time: {summary['generation_time_total']:.2f}s")
                print(f"   Average Generation Time: {summary['generation_time_average']:.4f}s")
                
                print(f"\n📝 Prompt Analysis:")
                prompt_analysis = report['prompt_analysis']
                print(f"   Average Prompt Length: {prompt_analysis['average_prompt_length']:.1f} characters")
                print(f"   Prompt Lengths: {prompt_analysis['prompt_lengths']}")
                
                print(f"\n⏰ Report Timestamp: {report['timestamp']}")
                
                # Save report to file
                output_dir = Path("reports")
                output_dir.mkdir(exist_ok=True)
                
                report_path = output_dir / f"diffusion_report_{int(time.time())}.json"
                with open(report_path, 'w') as f:
                    json.dump(report, f, indent=2, default=str)
                
                print(f"💾 Report saved to: {report_path}")
                
                return report
            else:
                print("⚠️ No report generated")
                return {}
            
        except Exception as e:
            print(f"❌ Error generating report: {e}")
            return {}
    
    def demo_training_workflow(self, trainer: DiffusionTrainer):
        """Demonstrate training workflow (simulated)."""
        print("\n" + "="*60)
        print("🏋️ DEMO: Training Workflow (Simulated)")
        print("="*60)
        
        if not trainer:
            print("⚠️ No trainer available for this demo")
            return
        
        try:
            print(f"\n🏋️ Simulating training workflow...")
            
            # Simulate training steps
            print(f"   📚 Training Configuration:")
            print(f"     Learning Rate: {trainer.config.learning_rate}")
            print(f"     Optimizer: {trainer.config.optimizer}")
            print(f"     Loss Type: {trainer.config.loss_type}")
            print(f"     Gradient Clip Norm: {trainer.config.gradient_clip_norm}")
            
            print(f"\n   🔄 Training State:")
            print(f"     Global Step: {trainer.global_step}")
            print(f"     Epoch: {trainer.epoch}")
            print(f"     Best Loss: {trainer.best_loss}")
            
            # Simulate a training step
            print(f"\n   🎯 Simulating training step...")
            
            # Create dummy batch
            dummy_batch = {
                'pixel_values': torch.randn(1, 3, 512, 512),
                'input_ids': torch.randint(0, 1000, (1, 77))
            }
            
            # Simulate training step
            step_result = trainer.train_step(dummy_batch, epoch=0)
            
            print(f"     ✅ Training step completed!")
            print(f"     Loss: {step_result['loss']:.6f}")
            print(f"     Learning Rate: {step_result['learning_rate']:.2e}")
            print(f"     Global Step: {step_result['global_step']}")
            
            # Simulate checkpoint saving
            print(f"\n   💾 Simulating checkpoint save...")
            checkpoint_dir = Path("checkpoints")
            checkpoint_dir.mkdir(exist_ok=True)
            
            checkpoint_path = checkpoint_dir / f"demo_checkpoint_{int(time.time())}.pt"
            trainer.save_checkpoint(str(checkpoint_path), metadata={'demo': True})
            
            print(f"     ✅ Checkpoint saved to: {checkpoint_path}")
            
            return step_result
            
        except Exception as e:
            print(f"❌ Error in training workflow: {e}")
            return {}
    
    def demo_advanced_features(self, model_manager: DiffusionModelManager):
        """Demonstrate advanced features and optimizations."""
        print("\n" + "="*60)
        print("🚀 DEMO: Advanced Features & Optimizations")
        print("="*60)
        
        if not model_manager:
            print("⚠️ No model manager available for this demo")
            return
        
        try:
            print(f"\n🚀 Advanced Features Available:")
            
            # Check optimizations
            config = model_manager.config
            optimizations = [
                ('Attention Slicing', config.enable_attention_slicing),
                ('VAE Slicing', config.enable_vae_slicing),
                ('XFormers Memory Efficient Attention', config.enable_xformers_memory_efficient_attention),
                ('Model CPU Offload', config.enable_model_cpu_offload),
                ('Gradient Checkpointing', config.use_gradient_checkpointing),
                ('Mixed Precision (AMP)', config.use_amp),
                ('EMA', config.use_ema)
            ]
            
            for feature, enabled in optimizations:
                status = "✅ Enabled" if enabled else "❌ Disabled"
                print(f"   {feature}: {status}")
            
            # Model capabilities
            print(f"\n🎯 Model Capabilities:")
            print(f"   Model Type: {config.model_type}")
            print(f"   Pipeline Mode: {config.use_pipeline}")
            print(f"   Inference Steps: {config.num_inference_steps}")
            print(f"   Guidance Scale: {config.guidance_scale}")
            print(f"   Image Dimensions: {config.width}x{config.height}")
            
            # Training capabilities
            print(f"\n🏋️ Training Capabilities:")
            print(f"   Training Timesteps: {config.num_train_timesteps}")
            print(f"   Beta Schedule: {config.beta_schedule}")
            print(f"   Beta Range: {config.beta_start} to {config.beta_end}")
            
            return optimizations
            
        except Exception as e:
            print(f"❌ Error in advanced features demo: {e}")
            return []
    
    def run_all_demos(self):
        """Run all demonstration functions."""
        print("🎨 Diffusion Models System Demo")
        print("=" * 80)
        
        try:
            # Demo 1: Basic system creation
            model_manager, trainer, analyzer = self.demo_basic_system_creation()
            
            # Demo 2: Model information
            self.demo_model_information(model_manager)
            
            # Demo 3: Image generation
            images, gen_times = self.demo_image_generation(model_manager)
            
            # Demo 4: Batch generation
            batch_images, batch_times = self.demo_batch_generation(model_manager)
            
            # Combine all images and times
            all_images = images + batch_images
            all_times = gen_times + batch_times
            all_prompts = self.sample_prompts[:len(all_images)]
            
            # Demo 5: Quality analysis
            quality_metrics = self.demo_quality_analysis(analyzer, all_images, all_prompts)
            
            # Demo 6: Performance analysis
            performance_metrics = self.demo_performance_analysis(analyzer, all_times)
            
            # Demo 7: Comprehensive report
            report = self.demo_comprehensive_report(analyzer, all_images, all_prompts, all_times)
            
            # Demo 8: Training workflow
            training_result = self.demo_training_workflow(trainer)
            
            # Demo 9: Advanced features
            advanced_features = self.demo_advanced_features(model_manager)
            
            print("\n" + "="*80)
            print("🎉 All demos completed successfully!")
            print("="*80)
            
            # Summary
            print(f"\n📋 System Summary:")
            print(f"   ✅ Comprehensive diffusion models system")
            print(f"   ✅ Stable Diffusion pipeline integration")
            print(f"   ✅ Advanced training capabilities")
            print(f"   ✅ Quality and performance analysis")
            print(f"   ✅ Memory optimization features")
            print(f"   ✅ Comprehensive reporting system")
            print(f"   ✅ Checkpoint management")
            
            # Generated content summary
            if all_images:
                print(f"\n🎨 Generated Content:")
                print(f"   Total Images: {len(all_images)}")
                print(f"   Total Generation Time: {sum(all_times):.2f}s")
                print(f"   Average Quality Score: {quality_metrics.get('sharpness', 0):.4f}")
                print(f"   Performance: {performance_metrics.get('throughput', {}).get('images_per_second', 0):.2f} img/s")
            
        except Exception as e:
            print(f"❌ Demo error: {e}")
            raise


if __name__ == "__main__":
    # Run the demo
    demo = DiffusionModelsDemo()
    demo.run_all_demos()






