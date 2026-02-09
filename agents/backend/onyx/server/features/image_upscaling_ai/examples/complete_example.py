"""
Complete Example
================

Complete example demonstrating all features of the image upscaling system.
"""

import asyncio
import logging
from pathlib import Path
from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def example_basic_upscaling():
    """Example 1: Basic upscaling."""
    print("\n" + "="*60)
    print("Example 1: Basic Upscaling")
    print("="*60)
    
    from image_upscaling_ai.core.upscaling_service import UpscalingService
    from image_upscaling_ai.config.upscaling_config import UpscalingConfig
    
    # Initialize service
    config = UpscalingConfig.from_env()
    service = UpscalingService(config=config)
    service.initialize_model()
    
    # Upscale image
    result = await service.upscale_image(
        image="input.jpg",
        scale_factor=2.0,
        use_ai=True
    )
    
    print(f"✅ Upscaled: {result['original_size']} -> {result['upscaled_size']}")
    print(f"Quality: {result.get('quality_score', 'N/A')}")


async def example_enhanced_upscaling():
    """Example 2: Enhanced upscaling with all features."""
    print("\n" + "="*60)
    print("Example 2: Enhanced Upscaling")
    print("="*60)
    
    from image_upscaling_ai.core.enhanced_service import EnhancedUpscalingService
    
    # Initialize enhanced service
    service = EnhancedUpscalingService()
    
    # Load image
    image = Image.open("input.jpg").convert("RGB")
    
    # Upscale with all features
    result = await service.upscale_image_enhanced(
        image,
        scale_factor=4.0,
        use_recommendations=True,
        validate_quality=True
    )
    
    if result["success"]:
        print(f"✅ Success!")
        print(f"Method: {result['method_used']}")
        print(f"Quality: {result['quality_score']:.2f}")
        print(f"Time: {result['processing_time']:.2f}s")
        
        # Save result
        result["upscaled_image"].save("output_enhanced.jpg")
        print("Saved: output_enhanced.jpg")
        
        # Submit feedback
        service.submit_feedback(
            result["operation_id"],
            satisfaction=0.95,
            quality_rating=0.92,
            speed_rating=0.85
        )
        print("✅ Feedback submitted")
    else:
        print(f"❌ Error: {result['error']}")


async def example_realesrgan():
    """Example 3: Real-ESRGAN upscaling."""
    print("\n" + "="*60)
    print("Example 3: Real-ESRGAN Upscaling")
    print("="*60)
    
    try:
        from image_upscaling_ai.models import RealESRGANModelManager
        
        # Initialize manager
        manager = RealESRGANModelManager(
            auto_download=False,
            device="cuda"  # or "cpu"
        )
        
        # Load image
        image = Image.open("input.jpg").convert("RGB")
        
        # Detect image type and select model
        image_type = manager.detect_image_type(image)
        print(f"Detected image type: {image_type}")
        
        # Select best model
        best_model = manager.select_best_model(image, 4.0)
        print(f"Selected model: {best_model}")
        
        # Upscale
        upscaled = await manager.upscale_async(
            image,
            scale_factor=4.0,
            model_name=best_model,
            auto_select=True
        )
        
        upscaled.save("output_realesrgan.jpg")
        print(f"✅ Upscaled: {image.size} -> {upscaled.size}")
        print("Saved: output_realesrgan.jpg")
        
    except ImportError:
        print("⚠️ Real-ESRGAN not available. Install with: pip install realesrgan basicsr")


async def example_smart_recommendations():
    """Example 4: Smart recommendations."""
    print("\n" + "="*60)
    print("Example 4: Smart Recommendations")
    print("="*60)
    
    from image_upscaling_ai.models import SmartRecommender, AdaptiveLearner, AdvancedImageDetector
    
    # Initialize components
    learner = AdaptiveLearner()
    detector = AdvancedImageDetector()
    recommender = SmartRecommender(learner=learner, detector=detector)
    
    # Load image
    image = Image.open("input.jpg").convert("RGB")
    
    # Get recommendation
    recommendation = recommender.recommend(
        image,
        target_scale=4.0,
        prioritize_speed=False,
        min_quality=0.7
    )
    
    print(f"Recommended Method: {recommendation.method}")
    print(f"Expected Quality: {recommendation.expected_quality:.2f}")
    print(f"Expected Time: {recommendation.expected_time:.2f}s")
    print(f"Confidence: {recommendation.confidence:.2f}")
    print(f"Preprocessing: {recommendation.preprocessing_mode}")
    print(f"Postprocessing: {recommendation.postprocessing_mode}")
    print("\nReasoning:")
    for reason in recommendation.reasoning:
        print(f"  - {reason}")


async def example_batch_processing():
    """Example 5: Batch processing."""
    print("\n" + "="*60)
    print("Example 5: Batch Processing")
    print("="*60)
    
    from image_upscaling_ai.models import BatchOptimizer, RealESRGANModelManager
    
    # Initialize
    manager = RealESRGANModelManager()
    optimizer = BatchOptimizer(initial_batch_size=2, adaptive=True)
    
    # Load images
    image_paths = ["input1.jpg", "input2.jpg", "input3.jpg"]
    images = [Image.open(path).convert("RGB") for path in image_paths]
    
    # Process function
    async def upscale_image(img):
        return await manager.upscale_async(img, 4.0)
    
    # Process batch
    result = await optimizer.process_batch_optimized(
        images,
        upscale_image,
        progress_callback=lambda current, total: print(f"Progress: {current}/{total}")
    )
    
    print(f"✅ Batch complete!")
    print(f"Total: {result.total_items}")
    print(f"Successful: {result.successful}")
    print(f"Failed: {result.failed}")
    print(f"Total time: {result.total_time:.2f}s")
    print(f"Avg time/item: {result.avg_time_per_item:.2f}s")
    print(f"Optimal batch size: {optimizer.get_optimal_batch_size()}")
    
    # Save results
    for idx, res in enumerate(result.results):
        if res.get("success"):
            res["result"].save(f"output_batch_{idx}.jpg")


async def example_quality_validation():
    """Example 6: Quality validation."""
    print("\n" + "="*60)
    print("Example 6: Quality Validation")
    print("="*60)
    
    from image_upscaling_ai.models import (
        QualityValidator,
        AdaptivePreprocessor,
        AdaptivePostprocessor,
        RealESRGANModelManager
    )
    
    # Initialize
    validator = QualityValidator(min_score=0.7, strict_mode=False)
    preprocessor = AdaptivePreprocessor()
    postprocessor = AdaptivePostprocessor()
    manager = RealESRGANModelManager()
    
    # Load image
    original = Image.open("input.jpg").convert("RGB")
    
    # Process
    preprocessed = preprocessor.preprocess(original, mode="auto")
    upscaled = await manager.upscale_async(preprocessed, 4.0)
    final = postprocessor.postprocess(upscaled, original=original, mode="auto")
    
    # Validate
    report = validator.validate(final, original, 4.0)
    
    print(f"Quality Score: {report.overall_score:.2f}")
    print(f"Passed: {'✅' if report.passed else '❌'}")
    print(f"\nMetrics:")
    for key, value in report.metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    if report.issues:
        print(f"\nIssues:")
        for issue in report.issues:
            print(f"  - {issue}")
    
    if report.recommendations:
        print(f"\nRecommendations:")
        for rec in report.recommendations:
            print(f"  - {rec}")


async def example_real_time_monitoring():
    """Example 7: Real-time monitoring."""
    print("\n" + "="*60)
    print("Example 7: Real-Time Monitoring")
    print("="*60)
    
    from image_upscaling_ai.models import RealtimeAnalyzer, RealESRGANModelManager
    
    # Initialize
    analyzer = RealtimeAnalyzer()
    manager = RealESRGANModelManager()
    
    # Setup callbacks
    def on_progress(metrics):
        print(f"  {metrics.stage}: {metrics.progress:.1%} "
              f"(Quality: {metrics.quality_estimate:.2f})")
        if metrics.time_remaining:
            print(f"    Time remaining: {metrics.time_remaining:.1f}s")
    
    analyzer.add_progress_callback(on_progress)
    
    # Load image
    image = Image.open("input.jpg").convert("RGB")
    
    # Start operation
    analyzer.start_operation()
    
    # Process with monitoring
    analyzer.update_stage("preprocessing")
    analyzer.update_progress(0.1, quality_estimate=0.75)
    await asyncio.sleep(0.1)  # Simulate processing
    
    analyzer.update_stage("upscaling")
    analyzer.update_progress(0.5, quality_estimate=0.85)
    upscaled = await manager.upscale_async(image, 4.0)
    
    analyzer.update_stage("postprocessing")
    analyzer.update_progress(0.9, quality_estimate=0.90)
    await asyncio.sleep(0.1)  # Simulate processing
    
    # Finish
    summary = analyzer.finish_operation()
    print(f"\n✅ Complete!")
    print(f"Total time: {summary['total_time']:.2f}s")


async def example_system_metrics():
    """Example 8: System metrics."""
    print("\n" + "="*60)
    print("Example 8: System Metrics")
    print("="*60)
    
    from image_upscaling_ai.models import AdvancedMetricsCollector
    from image_upscaling_ai.core.enhanced_service import EnhancedUpscalingService
    
    # Initialize
    service = EnhancedUpscalingService()
    
    # Process some images
    for i in range(3):
        result = await service.upscale_image_enhanced(
            f"input{i+1}.jpg" if Path(f"input{i+1}.jpg").exists() else "input.jpg",
            scale_factor=2.0
        )
    
    # Get system status
    status = service.get_system_status()
    
    print("System Status:")
    print(f"  Total operations: {status['system_metrics']['total_operations']}")
    print(f"  Success rate: {status['system_metrics']['success_rate']:.2%}")
    print(f"  Avg quality: {status['system_metrics']['avg_quality']:.2f}")
    print(f"  Throughput: {status['system_metrics']['throughput']:.2f} ops/s")
    print(f"  Cache hit rate: {status['cache']['hit_rate']:.2%}")
    
    # Get metrics
    metrics = service.metrics.get_system_metrics()
    print(f"\nMetrics:")
    print(f"  Avg processing time: {metrics.avg_processing_time:.2f}s")
    print(f"  Cache hit rate: {metrics.cache_hit_rate:.2%}")
    
    # Method performance
    print(f"\nMethod Performance:")
    for method in ["RealESRGAN_x4plus", "opencv_edsr"]:
        perf = service.metrics.get_method_performance(method)
        if perf["count"] > 0:
            print(f"  {method}:")
            print(f"    Count: {perf['count']}")
            print(f"    Avg quality: {perf['avg_quality']:.2f}")
            print(f"    Avg time: {perf['avg_time']:.2f}s")


async def example_complete_workflow():
    """Example 9: Complete workflow."""
    print("\n" + "="*60)
    print("Example 9: Complete Workflow")
    print("="*60)
    
    from image_upscaling_ai.core.enhanced_service import EnhancedUpscalingService
    from image_upscaling_ai.models import WorkflowStage
    
    # Initialize
    service = EnhancedUpscalingService()
    
    # Load image
    image = Image.open("input.jpg").convert("RGB")
    
    # Complete workflow
    result = await service.upscale_image_enhanced(
        image,
        scale_factor=4.0,
        use_recommendations=True,
        validate_quality=True,
        collect_feedback=True
    )
    
    if result["success"]:
        print("✅ Complete Workflow Success!")
        print(f"\nStages:")
        print(f"  1. Analysis: {result['analysis']['image_type']}")
        print(f"  2. Recommendation: {result['recommendation']['method']}")
        print(f"  3. Preprocessing: {result['recommendation']['preprocessing_mode']}")
        print(f"  4. Upscaling: {result['method_used']}")
        print(f"  5. Postprocessing: {result['recommendation']['postprocessing_mode']}")
        print(f"  6. Validation: {'✅ Passed' if result['quality_passed'] else '❌ Failed'}")
        
        print(f"\nResults:")
        print(f"  Quality: {result['quality_score']:.2f}")
        print(f"  Time: {result['processing_time']:.2f}s")
        print(f"  Size: {result['original_size']} -> {result['upscaled_size']}")
        
        # Save
        result["upscaled_image"].save("output_complete.jpg")
        print("\n✅ Saved: output_complete.jpg")


async def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("Image Upscaling AI - Complete Examples")
    print("="*60)
    
    examples = [
        ("Basic Upscaling", example_basic_upscaling),
        ("Enhanced Upscaling", example_enhanced_upscaling),
        ("Real-ESRGAN", example_realesrgan),
        ("Smart Recommendations", example_smart_recommendations),
        ("Batch Processing", example_batch_processing),
        ("Quality Validation", example_quality_validation),
        ("Real-Time Monitoring", example_real_time_monitoring),
        ("System Metrics", example_system_metrics),
        ("Complete Workflow", example_complete_workflow),
    ]
    
    for name, example_func in examples:
        try:
            await example_func()
        except FileNotFoundError as e:
            print(f"⚠️ Skipping {name}: {e}")
        except Exception as e:
            print(f"❌ Error in {name}: {e}")
            logger.exception(e)
    
    print("\n" + "="*60)
    print("Examples Complete!")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())


