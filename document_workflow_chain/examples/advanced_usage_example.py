"""
Advanced Usage Example for Workflow Chain Engine
===============================================

This example demonstrates the advanced capabilities of the enhanced workflow chain engine,
including comprehensive document analysis, multi-model support, and performance tracking.

Features demonstrated:
- Advanced document analysis
- Multi-model workflow chains
- Performance metrics
- Context optimization
- Quality assessment
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, Any

# Import the enhanced workflow chain engine
from ..workflow_chain_engine import (
    WorkflowChainEngine, 
    MultiModelWorkflowChain,
    DocumentProcessor,
    ModelContextLimits
)
from ..advanced_analysis import AdvancedWorkflowChainEngine


async def demonstrate_basic_workflow():
    """Demonstrate basic workflow chain functionality"""
    print("🚀 Basic Workflow Chain Demo")
    print("=" * 50)
    
    # Initialize the basic engine
    engine = WorkflowChainEngine()
    await engine.initialize()
    
    # Create a workflow chain
    chain = await engine.create_workflow_chain(
        name="AI Content Series",
        description="A series of blog posts about AI and machine learning",
        initial_prompt="Write an introduction to artificial intelligence, covering the basics and current applications."
    )
    
    print(f"✅ Created chain: {chain.id}")
    print(f"📄 Initial document: {chain.nodes[chain.root_node_id].title}")
    print(f"📊 Quality score: {chain.nodes[chain.root_node_id].metadata.get('quality_score', 0):.2f}")
    
    # Continue the chain
    next_doc = await engine.continue_workflow_chain(chain.id)
    print(f"📄 Next document: {next_doc.title}")
    print(f"📊 Quality score: {next_doc.metadata.get('quality_score', 0):.2f}")
    
    return chain


async def demonstrate_advanced_analysis():
    """Demonstrate advanced document analysis capabilities"""
    print("\n🔍 Advanced Document Analysis Demo")
    print("=" * 50)
    
    # Initialize the advanced engine
    base_engine = WorkflowChainEngine()
    await base_engine.initialize()
    advanced_engine = AdvancedWorkflowChainEngine(base_engine)
    
    # Create a chain with comprehensive analysis
    chain = await advanced_engine.create_workflow_chain_with_analysis(
        name="Technical Documentation Series",
        description="Comprehensive technical documentation with analysis",
        initial_prompt="Write a detailed guide about machine learning algorithms, including supervised and unsupervised learning methods.",
        enable_analysis=True
    )
    
    root_node = chain.nodes[chain.root_node_id]
    analysis = root_node.metadata.get('comprehensive_analysis', {})
    
    print(f"✅ Created chain with analysis: {chain.id}")
    print(f"📄 Document: {root_node.title}")
    
    # Display analysis results
    if 'basic_stats' in analysis:
        stats = analysis['basic_stats']
        print(f"📊 Word count: {stats.get('word_count', 0)}")
        print(f"📄 Estimated pages: {stats.get('estimated_pages', 0)}")
        print(f"⏱️ Reading time: {stats.get('reading_time_minutes', 0):.1f} minutes")
    
    if 'readability' in analysis:
        readability = analysis['readability']
        print(f"📖 Flesch Reading Ease: {readability.get('flesch_reading_ease', 0):.1f}")
        print(f"🎓 Grade Level: {readability.get('flesch_kincaid_grade', 0):.1f}")
    
    if 'sentiment' in analysis:
        sentiment = analysis['sentiment']
        print(f"😊 Sentiment: {sentiment.get('sentiment', 'neutral')} (score: {sentiment.get('score', 0):.2f})")
    
    if 'topics' in analysis:
        topics = analysis['topics'][:5]  # Top 5 topics
        print(f"🏷️ Top topics: {', '.join([t['topic'] for t in topics])}")
    
    print(f"⭐ Overall quality: {analysis.get('overall_quality', 0):.2f}")
    
    return chain


async def demonstrate_multi_model_support():
    """Demonstrate multi-model workflow chain capabilities"""
    print("\n🤖 Multi-Model Support Demo")
    print("=" * 50)
    
    # Create a multi-model workflow chain
    chain = MultiModelWorkflowChain(
        id="multi-model-demo",
        name="Multi-Model Content Chain",
        description="Demonstrates automatic model switching",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        settings={'ai_model': 'claude-3-5-sonnet-20241022'}
    )
    
    # Test different content sizes
    test_contents = [
        ("Small content", 1000),  # 1K tokens
        ("Medium content", 50000),  # 50K tokens
        ("Large content", 150000),  # 150K tokens
        ("Very large content", 500000),  # 500K tokens
    ]
    
    print("📊 Model Selection for Different Content Sizes:")
    for content_name, token_count in test_contents:
        optimal_model = chain.get_optimal_model(token_count)
        model_limit = ModelContextLimits.get_limit(optimal_model)
        
        print(f"  {content_name} ({token_count:,} tokens):")
        print(f"    → Optimal model: {optimal_model}")
        print(f"    → Model limit: {model_limit:,} tokens")
        print(f"    → Utilization: {(token_count/model_limit)*100:.1f}%")
        
        # Check if model switch is needed
        if chain.switch_model_if_needed(token_count):
            print(f"    ✅ Model switched to: {optimal_model}")
        print()


async def demonstrate_context_optimization():
    """Demonstrate context optimization capabilities"""
    print("\n⚡ Context Optimization Demo")
    print("=" * 50)
    
    # Test document processing capabilities
    sample_content = """
    Artificial Intelligence (AI) has revolutionized numerous industries and continues to shape our world in unprecedented ways. 
    From healthcare to finance, transportation to entertainment, AI technologies are transforming how we work, live, and interact.
    
    Machine learning, a subset of AI, enables computers to learn and improve from experience without being explicitly programmed. 
    This capability has led to breakthroughs in image recognition, natural language processing, and predictive analytics.
    
    Deep learning, which uses neural networks with multiple layers, has been particularly successful in complex tasks such as 
    computer vision and speech recognition. These technologies power everything from autonomous vehicles to virtual assistants.
    
    The future of AI holds immense promise, with ongoing research in areas like artificial general intelligence (AGI) and 
    quantum machine learning. However, it also presents challenges related to ethics, privacy, and the future of work.
    
    As we continue to advance AI capabilities, it's crucial to ensure these technologies are developed and deployed responsibly, 
    with consideration for their societal impact and potential risks.
    """ * 50  # Repeat to create a longer document
    
    # Analyze the document
    stats = DocumentProcessor.get_document_statistics(sample_content)
    
    print(f"📄 Document Statistics:")
    print(f"  Word count: {stats['word_count']:,}")
    print(f"  Estimated pages: {stats['estimated_pages']}")
    print(f"  Estimated tokens: {stats['estimated_tokens']:,}")
    print(f"  Reading time: {stats['reading_time_minutes']:.1f} minutes")
    
    # Test compatibility with different models
    models_to_test = [
        "gpt-4-turbo",
        "claude-3-5-sonnet-20241022", 
        "gemini-1.5-pro"
    ]
    
    print(f"\n🔍 Model Compatibility Check:")
    for model in models_to_test:
        can_process, message = DocumentProcessor.can_process_with_model(sample_content, model)
        status = "✅" if can_process else "❌"
        print(f"  {status} {model}: {message}")


async def demonstrate_performance_tracking():
    """Demonstrate performance tracking capabilities"""
    print("\n📈 Performance Tracking Demo")
    print("=" * 50)
    
    # Initialize advanced engine
    base_engine = WorkflowChainEngine()
    await base_engine.initialize()
    advanced_engine = AdvancedWorkflowChainEngine(base_engine)
    
    # Create multiple documents to track performance
    chain = await advanced_engine.create_workflow_chain_with_analysis(
        name="Performance Test Chain",
        description="Testing performance tracking capabilities",
        initial_prompt="Write about the benefits of renewable energy sources.",
        enable_analysis=True
    )
    
    # Generate several more documents
    for i in range(3):
        await advanced_engine.continue_workflow_chain_with_analysis(
            chain.id,
            f"Continue with topic {i+2} about renewable energy applications.",
            enable_analysis=True
        )
    
    # Get performance summary
    performance_summary = advanced_engine.get_performance_summary()
    
    print("📊 Performance Summary:")
    session_summary = performance_summary['session_summary']
    print(f"  Documents processed: {session_summary['documents_processed']}")
    print(f"  Total tokens used: {session_summary['total_tokens_used']:,}")
    print(f"  Average quality: {session_summary['average_quality']:.2f}")
    print(f"  Tokens per minute: {session_summary['tokens_per_minute']:.1f}")
    print(f"  Documents per hour: {session_summary['documents_per_hour']:.1f}")
    
    # Get chain analytics
    analytics = advanced_engine.get_chain_analytics(chain.id)
    
    print(f"\n📈 Chain Analytics:")
    print(f"  Total documents: {analytics['total_documents']}")
    print(f"  Total words: {analytics['total_words']:,}")
    print(f"  Quality trend: {analytics['quality_trend']}")
    print(f"  Content evolution: {analytics['content_evolution']['evolution_type']}")
    print(f"  Generation efficiency: {analytics['generation_efficiency']['efficiency_score']:.2f}")


async def demonstrate_quality_assessment():
    """Demonstrate comprehensive quality assessment"""
    print("\n⭐ Quality Assessment Demo")
    print("=" * 50)
    
    # Test different quality documents
    test_documents = [
        {
            "name": "High Quality Document",
            "content": """
            Artificial Intelligence represents a paradigm shift in how we approach problem-solving and automation. 
            The field encompasses machine learning, natural language processing, computer vision, and robotics.
            
            Machine learning algorithms can identify patterns in data that would be impossible for humans to detect manually. 
            This capability has revolutionized industries from healthcare to finance, enabling more accurate diagnoses and 
            better risk assessment.
            
            However, the rapid advancement of AI also raises important ethical considerations. Issues such as algorithmic bias, 
            privacy concerns, and the potential displacement of human workers must be carefully addressed as we continue to 
            develop these technologies.
            
            The future of AI holds tremendous promise, but it requires responsible development and deployment to ensure 
            that these powerful tools benefit all of humanity rather than exacerbating existing inequalities.
            """
        },
        {
            "name": "Medium Quality Document", 
            "content": """
            AI is good. It helps people. Machine learning is part of AI. It uses data. 
            AI can do many things. It is used in many places. Some people like AI. 
            Others are worried about AI. AI will change the world. We need to be careful with AI.
            """
        },
        {
            "name": "Low Quality Document",
            "content": """
            AI bad good maybe. Machine learning data. AI everywhere. People think AI. 
            AI change world. Be careful. AI help. AI problem. AI future. AI now.
            """
        }
    ]
    
    # Analyze each document
    for doc in test_documents:
        print(f"\n📄 Analyzing: {doc['name']}")
        
        stats = DocumentProcessor.get_document_statistics(doc['content'])
        print(f"  Word count: {stats['word_count']}")
        print(f"  Estimated pages: {stats['estimated_pages']}")
        print(f"  Reading time: {stats['reading_time_minutes']:.1f} minutes")
        
        # Test with different models
        for model in ["gpt-4-turbo", "claude-3-5-sonnet-20241022"]:
            can_process, message = DocumentProcessor.can_process_with_model(doc['content'], model)
            status = "✅" if can_process else "❌"
            print(f"  {status} {model}: Compatible")


async def main():
    """Run all demonstrations"""
    print("🎯 Advanced Workflow Chain Engine Demonstrations")
    print("=" * 60)
    
    try:
        # Run all demonstrations
        await demonstrate_basic_workflow()
        await demonstrate_advanced_analysis()
        await demonstrate_multi_model_support()
        await demonstrate_context_optimization()
        await demonstrate_performance_tracking()
        await demonstrate_quality_assessment()
        
        print("\n✅ All demonstrations completed successfully!")
        print("\n📋 Summary of Capabilities Demonstrated:")
        print("  • Basic workflow chain creation and continuation")
        print("  • Advanced document analysis with multiple metrics")
        print("  • Multi-model support with automatic model selection")
        print("  • Context optimization and token management")
        print("  • Performance tracking and analytics")
        print("  • Comprehensive quality assessment")
        print("  • Readability analysis and sentiment detection")
        print("  • Topic extraction and entity recognition")
        print("  • Content evolution tracking")
        print("  • Generation efficiency metrics")
        
    except Exception as e:
        print(f"❌ Error during demonstration: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run the demonstrations
    asyncio.run(main())



























