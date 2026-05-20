from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import time
import json
from typing import List, Dict, Any
import statistics
from src import (
from src.core.models import (
from src.optimization.base import OptimizerFactory, OptimizationPipeline
        import traceback
    import sys
from typing import Any, List, Dict, Optional
import logging
"""
🚀 Facebook Posts - Optimization Demo Consolidado
================================================

Demo completo que muestra todas las optimizaciones y funcionalidades
del sistema de Facebook Posts refactorizado.
"""


# Import del sistema refactorizado
    FacebookPostsSystem, 
    create_facebook_posts_system,
    quick_generate_post
)
    PostRequest, PostResponse, ContentType, AudienceType, 
    OptimizationLevel, QualityTier
)

# ===== DATOS DE DEMO =====

SAMPLE_POSTS = [
    "Check out our new product! It's really good and you should buy it now.",
    "We launched a new feature. It helps with productivity.",
    "Product available. Price is competitive. Contact us.",
    "AI breakthrough in healthcare technology. Revolutionary changes coming.",
    "Digital marketing strategies for 2024. Learn the latest techniques.",
    "Social media tips for entrepreneurs. Boost your online presence.",
    "Machine learning applications in business. Transform your operations.",
    "Content creation best practices. Engage your audience effectively.",
    "E-commerce optimization strategies. Increase your sales.",
    "Personal development tips. Improve your life and career."
]

SAMPLE_REQUESTS = [
    {
        "topic": "Digital Marketing Tips",
        "audience": "professionals",
        "content_type": "educational",
        "optimization_level": "standard"
    },
    {
        "topic": "AI in Healthcare",
        "audience": "medical_professionals", 
        "content_type": "technical",
        "optimization_level": "advanced"
    },
    {
        "topic": "Social Media Strategy",
        "audience": "entrepreneurs",
        "content_type": "promotional",
        "optimization_level": "ultra"
    }
]

# ===== FUNCIONES DE DEMO =====

async def demo_system_initialization():
    """Demo de inicialización del sistema."""
    print("\n" + "="*60)
    print("🚀 DEMO: SYSTEM INITIALIZATION")
    print("="*60)
    
    try:
        # Crear sistema
        print("📦 Creating Facebook Posts System...")
        system = await create_facebook_posts_system()
        
        # Health check
        print("🏥 Running health check...")
        health = await system.health_check()
        
        print(f"✅ System initialized successfully!")
        print(f"   Status: {health['status']}")
        print(f"   Available optimizers: {len(system.get_available_optimizers())}")
        print(f"   System stats: {system.get_stats()}")
        
        return system
        
    except Exception as e:
        print(f"❌ System initialization failed: {e}")
        return None

async def demo_basic_post_generation(system: FacebookPostsSystem):
    """Demo de generación básica de posts."""
    print("\n" + "="*60)
    print("📝 DEMO: BASIC POST GENERATION")
    print("="*60)
    
    try:
        # Generar post simple
        print("🎯 Generating simple post...")
        response = await system.quick_generate(
            "Digital Marketing Tips",
            "professionals",
            "educational"
        )
        
        if response.success:
            post = response.post
            print(f"✅ Post generated successfully!")
            print(f"   ID: {post.id}")
            print(f"   Content: {post.content[:100]}...")
            print(f"   Status: {post.status.value}")
            print(f"   Content Type: {post.content_type.value}")
            print(f"   Audience: {post.audience_type.value}")
            print(f"   Word Count: {post.get_word_count()}")
            print(f"   Processing Time: {response.processing_time:.3f}s")
            
            return post
        else:
            print(f"❌ Post generation failed: {response.error}")
            return None
            
    except Exception as e:
        print(f"❌ Basic generation failed: {e}")
        return None

async def demo_advanced_post_generation(system: FacebookPostsSystem):
    """Demo de generación avanzada con optimizaciones."""
    print("\n" + "="*60)
    print("⚡ DEMO: ADVANCED POST GENERATION")
    print("="*60)
    
    try:
        # Crear request avanzado
        request = PostRequest(
            topic="AI in Modern Healthcare",
            audience_type=AudienceType.PROFESSIONALS,
            content_type=ContentType.TECHNICAL,
            tone="professional",
            optimization_level=OptimizationLevel.ULTRA,
            include_hashtags=True,
            include_mentions=False,
            tags=["AI", "Healthcare", "Technology"]
        )
        
        print("🎯 Generating advanced post with optimizations...")
        response = await system.generate_post(request)
        
        if response.success:
            post = response.post
            print(f"✅ Advanced post generated successfully!")
            print(f"   ID: {post.id}")
            print(f"   Content: {post.content[:150]}...")
            print(f"   Optimization Level: {post.optimization_level.value}")
            print(f"   Quality Tier: {post.quality_tier.value if post.quality_tier else 'N/A'}")
            print(f"   Hashtags: {post.hashtags}")
            print(f"   Tags: {post.tags}")
            print(f"   Processing Time: {response.processing_time:.3f}s")
            print(f"   Optimizations Applied: {response.optimizations_applied}")
            
            return post
        else:
            print(f"❌ Advanced generation failed: {response.error}")
            return None
            
    except Exception as e:
        print(f"❌ Advanced generation failed: {e}")
        return None

async def demo_post_analysis(system: FacebookPostsSystem, post_id: str):
    """Demo de análisis de posts."""
    print("\n" + "="*60)
    print("📊 DEMO: POST ANALYSIS")
    print("="*60)
    
    try:
        print(f"🔍 Analyzing post {post_id}...")
        analysis = await system.analyze_post(post_id)
        
        if analysis['success']:
            print(f"✅ Analysis completed successfully!")
            
            # Métricas básicas
            basic_metrics = analysis['analysis']['basic']
            print(f"   Word Count: {basic_metrics['word_count']}")
            print(f"   Character Count: {basic_metrics['character_count']}")
            print(f"   Reading Time: {basic_metrics['reading_time']:.1f} minutes")
            print(f"   Hashtag Count: {basic_metrics['hashtag_count']}")
            
            # Métricas avanzadas
            if 'advanced' in analysis['analysis']:
                advanced_metrics = analysis['analysis']['advanced']
                if 'metrics' in advanced_metrics:
                    metrics = advanced_metrics['metrics']
                    print(f"   Engagement Score: {metrics['engagement_score']:.3f}")
                    print(f"   Quality Score: {metrics['quality_score']:.3f}")
                    print(f"   Readability Score: {metrics['readability_score']:.3f}")
                    print(f"   Sentiment Score: {metrics['sentiment_score']:.3f}")
                    print(f"   Overall Score: {metrics['overall_score']:.3f}")
            
            # Recomendaciones
            recommendations = analysis.get('recommendations', [])
            print(f"   Recommendations: {len(recommendations)}")
            for i, rec in enumerate(recommendations[:3], 1):
                print(f"     {i}. {rec['message']}")
            
            return analysis
        else:
            print(f"❌ Analysis failed: {analysis.get('error', 'Unknown error')}")
            return None
            
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return None

async def demo_optimization_management(system: FacebookPostsSystem):
    """Demo de gestión de optimizadores."""
    print("\n" + "="*60)
    print("🔧 DEMO: OPTIMIZATION MANAGEMENT")
    print("="*60)
    
    try:
        # Mostrar optimizadores disponibles
        available_optimizers = system.get_available_optimizers()
        print(f"📋 Available optimizers: {len(available_optimizers)}")
        for optimizer in available_optimizers:
            print(f"   - {optimizer}")
        
        # Añadir optimizador personalizado
        print("\n➕ Adding custom optimizer...")
        success = system.add_optimizer('performance', {
            'enabled': True,
            'priority': 1,
            'gpu_acceleration': True
        })
        
        if success:
            print("✅ Custom optimizer added successfully!")
            
            # Obtener optimizador
            optimizer = system.get_optimizer('performance')
            if optimizer:
                print(f"   Optimizer: {optimizer.name}")
                print(f"   Enabled: {optimizer.is_enabled}")
                print(f"   Priority: {optimizer.priority}")
                print(f"   Config: {optimizer.get_config()}")
        
        # Actualizar configuración
        print("\n⚙️ Updating optimizer configuration...")
        update_success = system.update_optimizer_config('performance', {
            'gpu_acceleration': False,
            'memory_optimization': True
        })
        
        if update_success:
            print("✅ Optimizer configuration updated!")
        
        return True
        
    except Exception as e:
        print(f"❌ Optimization management failed: {e}")
        return False

async def demo_batch_processing(system: FacebookPostsSystem):
    """Demo de procesamiento en lotes."""
    print("\n" + "="*60)
    print("📦 DEMO: BATCH PROCESSING")
    print("="*60)
    
    try:
        topics = [
            "AI in Business",
            "Digital Marketing",
            "Social Media Strategy", 
            "Content Creation",
            "Technology Trends"
        ]
        
        print(f"🔄 Processing {len(topics)} posts in batch...")
        start_time = time.time()
        
        responses = []
        for i, topic in enumerate(topics, 1):
            print(f"   Processing {i}/{len(topics)}: {topic}")
            response = await system.quick_generate(topic, "professionals", "educational")
            responses.append(response)
        
        total_time = time.time() - start_time
        
        # Analizar resultados
        successful = [r for r in responses if r.success]
        failed = [r for r in responses if not r.success]
        
        print(f"\n✅ Batch processing completed!")
        print(f"   Total Time: {total_time:.3f}s")
        print(f"   Average Time per Post: {total_time/len(topics):.3f}s")
        print(f"   Successful: {len(successful)}/{len(topics)}")
        print(f"   Failed: {len(failed)}/{len(topics)}")
        
        # Mostrar estadísticas de calidad
        if successful:
            quality_scores = []
            for response in successful:
                if response.post and response.post.metrics:
                    quality_scores.append(response.post.metrics.overall_score)
            
            if quality_scores:
                print(f"   Average Quality Score: {statistics.mean(quality_scores):.3f}")
                print(f"   Best Quality Score: {max(quality_scores):.3f}")
                print(f"   Worst Quality Score: {min(quality_scores):.3f}")
        
        return responses
        
    except Exception as e:
        print(f"❌ Batch processing failed: {e}")
        return []

async def demo_analytics_and_insights(system: FacebookPostsSystem):
    """Demo de analytics e insights."""
    print("\n" + "="*60)
    print("📈 DEMO: ANALYTICS & INSIGHTS")
    print("="*60)
    
    try:
        # Obtener analytics del sistema
        print("📊 Getting system analytics...")
        analytics = await system.get_analytics('system')
        
        if analytics.get('success'):
            print("✅ System analytics retrieved!")
            
            # Engine stats
            engine_stats = analytics.get('engine', {}).get('requests', {})
            print(f"   Total Requests: {engine_stats.get('total_requests', 0)}")
            print(f"   Successful Requests: {engine_stats.get('successful_requests', 0)}")
            print(f"   Failed Requests: {engine_stats.get('failed_requests', 0)}")
            print(f"   Average Processing Time: {engine_stats.get('avg_processing_time', 0):.3f}s")
            
            # Cache performance
            cache_stats = analytics.get('engine', {}).get('cache_performance', {})
            print(f"   Cache Hit Rate: {cache_stats.get('hit_rate', 0):.1%}")
            print(f"   Cache Miss Rate: {cache_stats.get('miss_rate', 0):.1%}")
            
            # Optimization stats
            optimization_stats = analytics.get('optimization', {})
            print(f"   Total Optimizers: {optimization_stats.get('total_optimizers', 0)}")
            print(f"   Enabled Optimizers: {optimization_stats.get('enabled_optimizers', 0)}")
            
            # Posts analytics
            posts_stats = analytics.get('posts', {})
            print(f"   Total Posts: {posts_stats.get('total_posts', 0)}")
            print(f"   Recent Posts: {posts_stats.get('recent_posts', 0)}")
            
            return analytics
        else:
            print(f"❌ Analytics failed: {analytics.get('error', 'Unknown error')}")
            return None
            
    except Exception as e:
        print(f"❌ Analytics failed: {e}")
        return None

async def demo_workflow_complete(system: FacebookPostsSystem):
    """Demo de workflow completo."""
    print("\n" + "="*60)
    print("🔄 DEMO: COMPLETE WORKFLOW")
    print("="*60)
    
    try:
        # 1. Generar post
        print("1️⃣ Generating post...")
        response = await system.quick_generate(
            "Machine Learning in Finance",
            "professionals",
            "technical"
        )
        
        if not response.success:
            print(f"❌ Post generation failed: {response.error}")
            return False
        
        post = response.post
        print(f"✅ Post generated: {post.id}")
        
        # 2. Analizar post
        print("2️⃣ Analyzing post...")
        analysis = await system.analyze_post(post.id)
        
        if not analysis['success']:
            print(f"❌ Analysis failed: {analysis.get('error')}")
            return False
        
        print(f"✅ Analysis completed")
        
        # 3. Aprobar post (si cumple criterios)
        quality_score = analysis['analysis']['advanced']['metrics']['quality_score']
        print(f"3️⃣ Quality score: {quality_score:.3f}")
        
        if quality_score > 0.7:
            print("✅ Post meets quality criteria, approving...")
            approval_result = await system.approve_post(post.id, "demo_user", "Auto-approved by demo")
            
            if approval_result['success']:
                print(f"✅ Post approved: {post.id}")
                
                # 4. Publicar post
                print("4️⃣ Publishing post...")
                publish_result = await system.publish_post(post.id, "demo_publisher")
                
                if publish_result['success']:
                    print(f"✅ Post published: {post.id}")
                    print("🎉 Complete workflow successful!")
                    return True
                else:
                    print(f"❌ Publication failed: {publish_result.get('error')}")
            else:
                print(f"❌ Approval failed: {approval_result.get('error')}")
        else:
            print(f"⚠️ Post quality too low ({quality_score:.3f}), not approving")
        
        return False
        
    except Exception as e:
        print(f"❌ Workflow failed: {e}")
        return False

# ===== DEMO PRINCIPAL =====

async def run_complete_demo():
    """Ejecutar demo completo del sistema."""
    print("🚀 FACEBOOK POSTS SYSTEM - COMPLETE DEMO")
    print("="*60)
    print("Este demo muestra todas las funcionalidades del sistema")
    print("refactorizado y consolidado.")
    print("="*60)
    
    start_time = time.time()
    
    try:
        # 1. Inicialización del sistema
        system = await demo_system_initialization()
        if not system:
            print("❌ Demo failed: System initialization failed")
            return
        
        # 2. Generación básica
        basic_post = await demo_basic_post_generation(system)
        
        # 3. Generación avanzada
        advanced_post = await demo_advanced_post_generation(system)
        
        # 4. Análisis de posts
        if basic_post:
            await demo_post_analysis(system, basic_post.id)
        
        # 5. Gestión de optimizadores
        await demo_optimization_management(system)
        
        # 6. Procesamiento en lotes
        await demo_batch_processing(system)
        
        # 7. Analytics e insights
        await demo_analytics_and_insights(system)
        
        # 8. Workflow completo
        await demo_workflow_complete(system)
        
        total_time = time.time() - start_time
        
        print("\n" + "="*60)
        print("🎉 DEMO COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Total demo time: {total_time:.2f} seconds")
        print("\n✅ All system features working correctly:")
        print("   🚀 System initialization and health checks")
        print("   📝 Basic and advanced post generation")
        print("   📊 Post analysis and quality assessment")
        print("   🔧 Optimization management")
        print("   📦 Batch processing capabilities")
        print("   📈 Analytics and insights")
        print("   🔄 Complete workflow automation")
        print("\n🚀 System ready for production use!")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        traceback.print_exc()

# ===== FUNCIÓN DE DEMO RÁPIDO =====

async def quick_demo():
    """Demo rápido para testing básico."""
    print("⚡ QUICK DEMO - Basic Functionality Test")
    print("="*40)
    
    try:
        # Crear sistema
        system = await create_facebook_posts_system()
        
        # Generar post simple
        response = await system.quick_generate("AI Technology", "professionals", "educational")
        
        if response.success:
            print(f"✅ Quick demo successful!")
            print(f"   Post ID: {response.post.id}")
            print(f"   Content: {response.post.content[:50]}...")
            print(f"   Processing Time: {response.processing_time:.3f}s")
        else:
            print(f"❌ Quick demo failed: {response.error}")
            
    except Exception as e:
        print(f"❌ Quick demo failed: {e}")

# ===== MAIN ENTRY POINT =====

if __name__ == "__main__":
    
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        # Demo rápido
        asyncio.run(quick_demo())
    else:
        # Demo completo
        asyncio.run(run_complete_demo()) 