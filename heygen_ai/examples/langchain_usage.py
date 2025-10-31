from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import asyncio
import logging
from pathlib import Path
from ..core import HeyGenAI, VideoRequest
from typing import Any, List, Dict, Optional
"""
LangChain and OpenRouter usage examples for HeyGen AI equivalent.
"""



# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def langchain_script_generation():
    """Example of script generation using LangChain and OpenRouter."""
    try:
        # Initialize HeyGen AI with OpenRouter API key
        # Replace with your actual OpenRouter API key
        openrouter_api_key = "your_openrouter_api_key_here"
        heygen = HeyGenAI(openrouter_api_key=openrouter_api_key)
        
        # Generate script using LangChain
        logger.info("Generating script using LangChain...")
        script = await heygen.generate_script(
            topic="The Future of Artificial Intelligence in Healthcare",
            language="en",
            style="educational",
            duration="3 minutes",
            context="Target audience: medical professionals and healthcare administrators"
        )
        
        logger.info(f"Generated script:\n{script}")
        
        # Analyze the generated script
        logger.info("Analyzing generated script...")
        analysis = await heygen.analyze_script(script)
        
        logger.info(f"Script analysis:")
        logger.info(f"  Word count: {analysis['word_count']}")
        logger.info(f"  Estimated duration: {analysis['estimated_duration']:.2f} minutes")
        logger.info(f"  Readability score: {analysis['readability_score']}")
        logger.info(f"  Sentiment: {analysis['sentiment']['overall']}")
        logger.info(f"  Suggestions: {analysis['suggestions']}")
        
    except Exception as e:
        logger.error(f"Error in LangChain script generation: {e}")


async def langchain_translation_example():
    """Example of script translation using LangChain."""
    try:
        # Initialize HeyGen AI with OpenRouter API key
        openrouter_api_key = "your_openrouter_api_key_here"
        heygen = HeyGenAI(openrouter_api_key=openrouter_api_key)
        
        # Original script in English
        original_script = """
        Hello! Welcome to our presentation about artificial intelligence in healthcare.
        Today, we'll explore how AI is transforming patient care, diagnosis, and treatment.
        We'll discuss the benefits, challenges, and future prospects of AI in medicine.
        """
        
        # Translate to Spanish
        logger.info("Translating script to Spanish using LangChain...")
        spanish_script = await heygen.translate_script(
            script=original_script,
            target_language="es",
            source_language="en",
            preserve_style=True
        )
        
        logger.info(f"Original (English):\n{original_script}")
        logger.info(f"Translated (Spanish):\n{spanish_script}")
        
        # Translate to French
        logger.info("Translating script to French using LangChain...")
        french_script = await heygen.translate_script(
            script=original_script,
            target_language="fr",
            source_language="en",
            preserve_style=True
        )
        
        logger.info(f"Translated (French):\n{french_script}")
        
    except Exception as e:
        logger.error(f"Error in LangChain translation: {e}")


async def langchain_agent_chat():
    """Example of chatting with LangChain agent."""
    try:
        # Initialize HeyGen AI with OpenRouter API key
        openrouter_api_key = "your_openrouter_api_key_here"
        heygen = HeyGenAI(openrouter_api_key=openrouter_api_key)
        
        # Chat with the agent
        logger.info("Chatting with LangChain agent...")
        
        messages = [
            "I need to create a video about renewable energy. Can you help me generate a script?",
            "The script should be educational and target high school students.",
            "Can you also suggest some visual elements that would work well with this topic?",
            "What would be the ideal duration for this type of content?"
        ]
        
        for message in messages:
            logger.info(f"User: {message}")
            response = await heygen.chat_with_agent(message)
            logger.info(f"Agent: {response}")
            logger.info("-" * 50)
        
    except Exception as e:
        logger.error(f"Error in LangChain agent chat: {e}")


async def knowledge_base_example():
    """Example of knowledge base creation and search using LangChain."""
    try:
        # Initialize HeyGen AI with OpenRouter API key
        openrouter_api_key = "your_openrouter_api_key_here"
        heygen = HeyGenAI(openrouter_api_key=openrouter_api_key)
        
        # Sample documents for knowledge base
        documents = [
            "Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines.",
            "Machine Learning is a subset of AI that enables computers to learn and improve from experience.",
            "Deep Learning uses neural networks with multiple layers to process complex patterns in data.",
            "Natural Language Processing (NLP) helps computers understand and generate human language.",
            "Computer Vision enables machines to interpret and understand visual information from the world."
        ]
        
        # Create knowledge base
        logger.info("Creating knowledge base using LangChain...")
        await heygen.create_knowledge_base(documents, "ai_concepts")
        
        # Search knowledge base
        queries = [
            "What is machine learning?",
            "How does deep learning work?",
            "Explain natural language processing"
        ]
        
        for query in queries:
            logger.info(f"Searching for: {query}")
            results = await heygen.search_knowledge_base(query, "ai_concepts", k=3)
            
            logger.info("Search results:")
            for i, result in enumerate(results, 1):
                logger.info(f"  {i}. {result}")
            logger.info("-" * 50)
        
    except Exception as e:
        logger.error(f"Error in knowledge base example: {e}")


async def complete_video_workflow():
    """Complete video generation workflow using LangChain."""
    try:
        # Initialize HeyGen AI with OpenRouter API key
        openrouter_api_key = "your_openrouter_api_key_here"
        heygen = HeyGenAI(openrouter_api_key=openrouter_api_key)
        
        # Step 1: Generate script using LangChain
        logger.info("Step 1: Generating script using LangChain...")
        script = await heygen.generate_script(
            topic="The Impact of Climate Change on Global Agriculture",
            language="en",
            style="educational",
            duration="2 minutes",
            context="Target audience: university students studying environmental science"
        )
        
        logger.info(f"Generated script:\n{script}")
        
        # Step 2: Optimize script
        logger.info("Step 2: Optimizing script...")
        optimized_script = await heygen.optimize_script(
            script=script,
            duration="2 minutes",
            style="educational",
            language="en"
        )
        
        logger.info(f"Optimized script:\n{optimized_script}")
        
        # Step 3: Analyze script
        logger.info("Step 3: Analyzing script...")
        analysis = await heygen.analyze_script(optimized_script)
        
        logger.info(f"Analysis results:")
        logger.info(f"  Word count: {analysis['word_count']}")
        logger.info(f"  Estimated duration: {analysis['estimated_duration']:.2f} minutes")
        logger.info(f"  Readability score: {analysis['readability_score']}")
        
        # Step 4: Create video (if components are available)
        logger.info("Step 4: Creating video...")
        request = VideoRequest(
            script=optimized_script,
            avatar_id="professional_male_01",
            voice_id="en_us_01",
            language="en",
            output_format="mp4",
            resolution="1080p"
        )
        
        response = await heygen.create_video(request)
        
        if response.status == "completed":
            logger.info(f"Video created successfully!")
            logger.info(f"Video ID: {response.video_id}")
            logger.info(f"Output URL: {response.output_url}")
            logger.info(f"Duration: {response.duration} seconds")
            logger.info(f"File size: {response.file_size} bytes")
        else:
            logger.error(f"Video creation failed: {response.metadata.get('error')}")
        
    except Exception as e:
        logger.error(f"Error in complete video workflow: {e}")


async def langchain_status_check():
    """Check LangChain integration status."""
    try:
        # Initialize HeyGen AI with OpenRouter API key
        openrouter_api_key = "your_openrouter_api_key_here"
        heygen = HeyGenAI(openrouter_api_key=openrouter_api_key)
        
        # Check overall health
        health_status = heygen.health_check()
        logger.info("System health status:")
        for component, status in health_status.items():
            logger.info(f"  {component}: {'✅' if status else '❌'}")
        
        # Check LangChain status
        langchain_status = heygen.get_langchain_status()
        logger.info("\nLangChain integration status:")
        for key, value in langchain_status.items():
            if isinstance(value, bool):
                logger.info(f"  {key}: {'✅' if value else '❌'}")
            else:
                logger.info(f"  {key}: {value}")
        
    except Exception as e:
        logger.error(f"Error in status check: {e}")


async def main():
    """Run all LangChain examples."""
    logger.info("Starting LangChain and OpenRouter examples...")
    
    # Check if OpenRouter API key is available
    openrouter_api_key = "your_openrouter_api_key_here"
    if openrouter_api_key == "your_openrouter_api_key_here":
        logger.warning("Please set your OpenRouter API key to run these examples")
        logger.info("You can get an API key from: https://openrouter.ai/")
        return
    
    # Run examples
    await langchain_status_check()
    await langchain_script_generation()
    await langchain_translation_example()
    await langchain_agent_chat()
    await knowledge_base_example()
    await complete_video_workflow()
    
    logger.info("All LangChain examples completed!")


match __name__:
    case "__main__":
    asyncio.run(main()) 