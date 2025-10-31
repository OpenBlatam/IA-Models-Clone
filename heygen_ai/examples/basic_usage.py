from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import asyncio
import logging
from pathlib import Path
from ..core import HeyGenAI, VideoRequest
from typing import Any, List, Dict, Optional
"""
Basic usage example for HeyGen AI equivalent.
"""



# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def basic_video_generation():
    """Basic example of video generation."""
    try:
        # Initialize HeyGen AI
        heygen = HeyGenAI()
        
        # Create video request
        request = VideoRequest(
            script="Hello! Welcome to our AI-powered video generation system. "
                   "This is a demonstration of how we can create professional "
                   "videos with AI avatars and natural speech synthesis.",
            avatar_id="professional_male_01",
            voice_id="en_us_01",
            language="en",
            output_format="mp4",
            resolution="1080p"
        )
        
        # Generate video
        logger.info("Starting video generation...")
        response = await heygen.create_video(request)
        
        if response.status == "completed":
            logger.info(f"Video generated successfully!")
            logger.info(f"Video ID: {response.video_id}")
            logger.info(f"Output URL: {response.output_url}")
            logger.info(f"Duration: {response.duration} seconds")
            logger.info(f"File size: {response.file_size} bytes")
        else:
            logger.error(f"Video generation failed: {response.metadata.get('error')}")
            
    except Exception as e:
        logger.error(f"Error in video generation: {e}")


async def script_generation_example():
    """Example of AI script generation."""
    try:
        # Initialize HeyGen AI
        heygen = HeyGenAI()
        
        # Generate script
        logger.info("Generating AI script...")
        script = await heygen.generate_script(
            topic="The Future of Artificial Intelligence",
            language="en",
            style="educational"
        )
        
        logger.info(f"Generated script:\n{script}")
        
        # Analyze script
        logger.info("Analyzing script...")
        analysis = await heygen.script_generator.analyze_script(script)
        
        logger.info(f"Script analysis:")
        logger.info(f"  Word count: {analysis['word_count']}")
        logger.info(f"  Estimated duration: {analysis['estimated_duration']:.2f} minutes")
        logger.info(f"  Readability score: {analysis['readability_score']}")
        logger.info(f"  Sentiment: {analysis['sentiment']['overall']}")
        
    except Exception as e:
        logger.error(f"Error in script generation: {e}")


async def voice_cloning_example():
    """Example of voice cloning."""
    try:
        # Initialize HeyGen AI
        heygen = HeyGenAI()
        
        # Example audio sample URLs (these would be real URLs in practice)
        audio_samples = [
            "https://example.com/audio/sample1.wav",
            "https://example.com/audio/sample2.wav",
            "https://example.com/audio/sample3.wav"
        ]
        
        # Clone voice
        logger.info("Cloning voice...")
        voice_id = await heygen.voice_engine.clone_voice(
            audio_samples=audio_samples,
            voice_name="My Custom Voice"
        )
        
        logger.info(f"Voice cloned successfully! Voice ID: {voice_id}")
        
        # Get voice information
        voice = await heygen.get_voice(voice_id)
        logger.info(f"Voice details: {voice}")
        
    except Exception as e:
        logger.error(f"Error in voice cloning: {e}")


async def avatar_creation_example():
    """Example of custom avatar creation."""
    try:
        # Initialize HeyGen AI
        heygen = HeyGenAI()
        
        # Create custom avatar
        logger.info("Creating custom avatar...")
        avatar_id = await heygen.avatar_manager.create_custom_avatar(
            image_path="https://example.com/avatar_source.jpg",
            name="My Custom Avatar",
            style="professional"
        )
        
        logger.info(f"Avatar created successfully! Avatar ID: {avatar_id}")
        
        # Get avatar information
        avatar = await heygen.get_avatar(avatar_id)
        logger.info(f"Avatar details: {avatar}")
        
    except Exception as e:
        logger.error(f"Error in avatar creation: {e}")


async def batch_processing_example():
    """Example of batch video processing."""
    try:
        # Initialize HeyGen AI
        heygen = HeyGenAI()
        
        # Create multiple video requests
        requests = [
            VideoRequest(
                script="Welcome to our first video in the series.",
                avatar_id="professional_male_01",
                voice_id="en_us_01",
                language="en"
            ),
            VideoRequest(
                script="This is our second video with different content.",
                avatar_id="professional_female_01", 
                voice_id="en_us_02",
                language="en"
            ),
            VideoRequest(
                script="And here's our third video in Spanish.",
                avatar_id="professional_male_01",
                voice_id="es_es_01",
                language="es"
            )
        ]
        
        # Process videos in batch
        logger.info("Starting batch video processing...")
        responses = await heygen.batch_create_videos(requests)
        
        # Process results
        for i, response in enumerate(responses):
            if response.status == "completed":
                logger.info(f"Video {i+1} completed: {response.video_id}")
            else:
                logger.error(f"Video {i+1} failed: {response.metadata.get('error')}")
                
    except Exception as e:
        logger.error(f"Error in batch processing: {e}")


async def main():
    """Run all examples."""
    logger.info("Starting HeyGen AI examples...")
    
    # Run examples
    await basic_video_generation()
    await script_generation_example()
    await voice_cloning_example()
    await avatar_creation_example()
    await batch_processing_example()
    
    logger.info("All examples completed!")


match __name__:
    case "__main__":
    asyncio.run(main()) 