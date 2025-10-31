"""
Document Workflow Chain Demo
============================

This demo shows how to use the Document Workflow Chain system to create
continuous document generation workflows where each document's output
becomes the input for the next document.
"""

import asyncio
import json
from datetime import datetime
from workflow_chain_engine import WorkflowChainEngine

class MockAIClient:
    """Mock AI client for demonstration purposes"""
    
    def __init__(self):
        self.call_count = 0
    
    async def generate_text(self, prompt: str) -> str:
        """Generate mock text based on prompt"""
        self.call_count += 1
        
        # Simple mock responses based on prompt content
        if "introduction" in prompt.lower():
            return """
            Artificial Intelligence has revolutionized the way we approach content creation. 
            From automated writing assistants to intelligent content optimization, AI tools 
            are transforming the creative landscape. This comprehensive guide explores the 
            various ways AI can enhance your content creation process, improve efficiency, 
            and deliver higher quality results.
            
            The benefits of AI-powered content creation are numerous. Not only does it 
            accelerate the writing process, but it also helps maintain consistency across 
            different pieces of content. AI can analyze audience preferences, optimize 
            for search engines, and even suggest improvements to enhance readability and engagement.
            """
        
        elif "benefits" in prompt.lower() or "advantages" in prompt.lower():
            return """
            The advantages of implementing AI in content creation workflows extend far 
            beyond simple automation. One of the most significant benefits is the ability 
            to scale content production without proportionally increasing human resources. 
            AI can generate multiple variations of content, test different approaches, 
            and identify what resonates most with your target audience.
            
            Additionally, AI-powered content creation enables personalization at scale. 
            By analyzing user data and preferences, AI can create customized content 
            that speaks directly to individual readers. This level of personalization 
            was previously impossible to achieve manually, especially for large audiences.
            """
        
        elif "tools" in prompt.lower() or "platforms" in prompt.lower():
            return """
            The market is flooded with AI content creation tools, each offering unique 
            capabilities and features. Popular platforms like GPT-4, Claude, and other 
            language models provide powerful text generation capabilities. These tools 
            can create everything from blog posts and articles to marketing copy and 
            social media content.
            
            When selecting AI content creation tools, consider factors such as output 
            quality, customization options, integration capabilities, and cost. Some 
            tools specialize in specific content types, while others offer comprehensive 
            solutions for various content needs. The key is to choose tools that align 
            with your specific requirements and workflow.
            """
        
        elif "future" in prompt.lower() or "trends" in prompt.lower():
            return """
            The future of AI content creation looks incredibly promising, with emerging 
            trends pointing toward even more sophisticated and integrated solutions. 
            We're seeing the development of multimodal AI systems that can create 
            content across different formats - text, images, videos, and audio - 
            all from a single prompt or concept.
            
            Another exciting trend is the integration of AI content creation with 
            real-time data and analytics. This allows for dynamic content that 
            adapts based on current events, user behavior, and market conditions. 
            The result is more relevant, timely, and engaging content that 
            automatically stays current with changing circumstances.
            """
        
        else:
            return f"""
            This is a continuation of our discussion about AI-powered content creation. 
            The previous content has laid the foundation for understanding how artificial 
            intelligence can transform your content strategy. Now, let's explore additional 
            aspects and considerations that are crucial for successful implementation.
            
            As we continue this exploration, it's important to remember that AI is a tool 
            that enhances human creativity rather than replacing it. The most effective 
            content creation strategies combine the efficiency and capabilities of AI 
            with human insight, creativity, and strategic thinking.
            """

async def demo_basic_workflow():
    """Demonstrate basic workflow chain creation and continuation"""
    print("🚀 Starting Document Workflow Chain Demo")
    print("=" * 50)
    
    # Initialize the engine with mock AI client
    ai_client = MockAIClient()
    engine = WorkflowChainEngine(ai_client=ai_client)
    
    # Create initial workflow chain
    print("\n📝 Creating initial workflow chain...")
    chain = await engine.create_workflow_chain(
        name="AI Content Creation Guide",
        description="A comprehensive guide to AI-powered content creation",
        initial_prompt="Write an introduction to AI-powered content creation, covering the basics and benefits."
    )
    
    print(f"✅ Created chain: {chain.id}")
    print(f"📄 Initial document: '{chain.nodes[chain.root_node_id].title}'")
    print(f"📊 Content preview: {chain.nodes[chain.root_node_id].content[:100]}...")
    
    # Continue the chain multiple times
    print("\n🔄 Continuing workflow chain...")
    
    for i in range(3):
        print(f"\n--- Continuation {i+1} ---")
        next_doc = await engine.continue_workflow_chain(chain.id)
        print(f"📄 New document: '{next_doc.title}'")
        print(f"📊 Content preview: {next_doc.content[:100]}...")
        print(f"🔗 Parent: {next_doc.parent_id}")
        print(f"👶 Children: {len(next_doc.children_ids)}")
    
    # Get chain history
    print("\n📚 Chain History:")
    history = engine.get_chain_history(chain.id)
    for i, doc in enumerate(history, 1):
        print(f"{i}. {doc.title} (Generated: {doc.generated_at.strftime('%H:%M:%S')})")
    
    # Generate a blog title
    print("\n🏷️  Generating blog title...")
    sample_content = "This is a sample blog post about the future of artificial intelligence in content creation and how it's transforming the way we approach digital marketing and content strategy."
    title = await engine.generate_blog_title(sample_content)
    print(f"📰 Generated title: '{title}'")
    
    # Export workflow chain
    print("\n💾 Exporting workflow chain...")
    export_data = engine.export_workflow_chain(chain.id)
    print(f"📦 Exported {len(export_data['nodes'])} documents")
    
    print(f"\n🎯 Demo completed! AI client made {ai_client.call_count} calls.")
    return chain

async def demo_workflow_management():
    """Demonstrate workflow management features"""
    print("\n🔧 Workflow Management Demo")
    print("=" * 30)
    
    engine = WorkflowChainEngine()
    
    # Create multiple chains
    chains = []
    for i in range(3):
        chain = await engine.create_workflow_chain(
            name=f"Test Chain {i+1}",
            description=f"Test workflow chain number {i+1}",
            initial_prompt=f"Write about topic {i+1} in detail."
        )
        chains.append(chain)
        print(f"✅ Created chain {i+1}: {chain.id}")
    
    # Get all active chains
    active_chains = engine.get_all_active_chains()
    print(f"\n📊 Total active chains: {len(active_chains)}")
    
    # Pause and resume a chain
    chain_to_pause = chains[0]
    print(f"\n⏸️  Pausing chain: {chain_to_pause.id}")
    engine.pause_workflow_chain(chain_to_pause.id)
    
    # Check status
    paused_chain = engine.get_workflow_chain(chain_to_pause.id)
    print(f"📊 Chain status: {paused_chain.status}")
    
    # Resume chain
    print(f"\n▶️  Resuming chain: {chain_to_pause.id}")
    engine.resume_workflow_chain(chain_to_pause.id)
    
    resumed_chain = engine.get_workflow_chain(chain_to_pause.id)
    print(f"📊 Chain status: {resumed_chain.status}")
    
    # Complete a chain
    chain_to_complete = chains[1]
    print(f"\n✅ Completing chain: {chain_to_complete.id}")
    engine.complete_workflow_chain(chain_to_complete.id)
    
    # Check remaining active chains
    remaining_chains = engine.get_all_active_chains()
    print(f"📊 Remaining active chains: {len(remaining_chains)}")
    
    print("\n🎯 Workflow management demo completed!")

async def demo_api_simulation():
    """Simulate API usage patterns"""
    print("\n🌐 API Simulation Demo")
    print("=" * 25)
    
    # This would typically be done through HTTP requests
    # For demo purposes, we'll simulate the API calls directly
    
    engine = WorkflowChainEngine(MockAIClient())
    
    # Simulate POST /create
    print("📡 Simulating API: POST /create")
    chain = await engine.create_workflow_chain(
        name="API Test Chain",
        description="Testing API functionality",
        initial_prompt="Create a test document for API demonstration."
    )
    print(f"✅ Chain created via API simulation: {chain.id}")
    
    # Simulate POST /continue
    print("\n📡 Simulating API: POST /continue")
    next_doc = await engine.continue_workflow_chain(chain.id)
    print(f"✅ Document continued via API simulation: {next_doc.id}")
    
    # Simulate GET /chain/{id}
    print("\n📡 Simulating API: GET /chain/{id}")
    retrieved_chain = engine.get_workflow_chain(chain.id)
    print(f"✅ Chain retrieved via API simulation: {retrieved_chain.name}")
    
    # Simulate GET /chain/{id}/history
    print("\n📡 Simulating API: GET /chain/{id}/history")
    history = engine.get_chain_history(chain.id)
    print(f"✅ History retrieved via API simulation: {len(history)} documents")
    
    print("\n🎯 API simulation demo completed!")

async def main():
    """Run all demos"""
    print("🎬 Document Workflow Chain - Complete Demo")
    print("=" * 50)
    
    try:
        # Run basic workflow demo
        await demo_basic_workflow()
        
        # Run workflow management demo
        await demo_workflow_management()
        
        # Run API simulation demo
        await demo_api_simulation()
        
        print("\n🎉 All demos completed successfully!")
        print("\n💡 Key Features Demonstrated:")
        print("   • Continuous document generation with chaining")
        print("   • Workflow state management (pause/resume/complete)")
        print("   • Chain history tracking")
        print("   • Blog title generation")
        print("   • Workflow export functionality")
        print("   • API endpoint simulation")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {str(e)}")
        raise

if __name__ == "__main__":
    # Run the demo
    asyncio.run(main())


