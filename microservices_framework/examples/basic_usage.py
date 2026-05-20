"""
Basic Usage Examples
Demonstrates how to use the deep learning microservices.
"""

import asyncio
import httpx
from pathlib import Path


async def example_text_generation():
    """Example: Generate text using LLM service."""
    print("=" * 50)
    print("Example 1: Text Generation")
    print("=" * 50)
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            "http://localhost:8001/generate",
            json={
                "prompt": "The future of artificial intelligence",
                "model_name": "gpt2",
                "max_length": 100,
                "temperature": 0.8,
                "top_p": 0.9,
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"Prompt: {result['prompt']}")
            print(f"Generated: {result['generated_text']}")
        else:
            print(f"Error: {response.status_code} - {response.text}")


async def example_image_generation():
    """Example: Generate image using Diffusion service."""
    print("\n" + "=" * 50)
    print("Example 2: Image Generation")
    print("=" * 50)
    
    async with httpx.AsyncClient(timeout=300.0) as client:
        response = await client.post(
            "http://localhost:8002/text-to-image",
            json={
                "prompt": "A beautiful landscape with mountains and lakes, sunset, photorealistic",
                "model_name": "runwayml/stable-diffusion-v1-5",
                "num_inference_steps": 30,  # Reduced for faster demo
                "guidance_scale": 7.5,
                "width": 512,
                "height": 512,
            }
        )
        
        if response.status_code == 200:
            output_path = Path("generated_image.png")
            output_path.write_bytes(response.content)
            print(f"Image saved to: {output_path.absolute()}")
        else:
            print(f"Error: {response.status_code} - {response.text}")


async def example_embeddings():
    """Example: Generate text embeddings."""
    print("\n" + "=" * 50)
    print("Example 3: Text Embeddings")
    print("=" * 50)
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            "http://localhost:8001/embeddings",
            json={
                "texts": [
                    "The future of AI",
                    "Machine learning is fascinating",
                    "Deep learning models are powerful",
                ],
                "model_name": "sentence-transformers/all-MiniLM-L6-v2",
                "normalize": True,
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"Generated {len(result['embeddings'])} embeddings")
            print(f"Embedding dimension: {result['dimension']}")
            print(f"First embedding (first 10 values): {result['embeddings'][0][:10]}")
        else:
            print(f"Error: {response.status_code} - {response.text}")


async def example_training_job():
    """Example: Start a training job."""
    print("\n" + "=" * 50)
    print("Example 4: Training Job")
    print("=" * 50)
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        # Start training
        response = await client.post(
            "http://localhost:8003/train",
            json={
                "model_name": "gpt2",
                "task_type": "causal_lm",
                "dataset_path": "wikitext",
                "num_epochs": 1,  # Short demo
                "batch_size": 2,
                "learning_rate": 5e-5,
                "use_lora": True,
                "lora_r": 8,
                "lora_alpha": 16,
                "max_length": 128,
            }
        )
        
        if response.status_code == 200:
            job = response.json()
            job_id = job["job_id"]
            print(f"Training job started: {job_id}")
            print(f"Model: {job['model_name']}")
            print(f"Output directory: {job['output_dir']}")
            
            # Check status
            print("\nChecking job status...")
            status_response = await client.get(f"http://localhost:8003/jobs/{job_id}/status")
            if status_response.status_code == 200:
                status = status_response.json()
                print(f"Status: {status['status']}")
                print(f"Progress: {status['progress']*100:.1f}%")
        else:
            print(f"Error: {response.status_code} - {response.text}")


async def example_gradio_interface():
    """Example: Launch Gradio interface."""
    print("\n" + "=" * 50)
    print("Example 5: Gradio Interface")
    print("=" * 50)
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            "http://localhost:8004/interfaces/text-generation",
            json={
                "model_name": "gpt2",
                "port": 7860,
                "share": False,
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"Gradio interface launched!")
            print(f"URL: {result['url']}")
            print(f"Interface ID: {result['interface_id']}")
            print(f"\nOpen {result['url']} in your browser to use the interface.")
        else:
            print(f"Error: {response.status_code} - {response.text}")


async def check_services_health():
    """Check if all services are running."""
    print("Checking service health...")
    services = {
        "LLM Service": "http://localhost:8001/health",
        "Diffusion Service": "http://localhost:8002/health",
        "Training Service": "http://localhost:8003/health",
        "Gradio Service": "http://localhost:8004/health",
    }
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        for name, url in services.items():
            try:
                response = await client.get(url)
                if response.status_code == 200:
                    status = response.json()
                    print(f"✓ {name}: {status.get('status', 'unknown')}")
                else:
                    print(f"✗ {name}: HTTP {response.status_code}")
            except Exception as e:
                print(f"✗ {name}: Not reachable - {str(e)}")


async def main():
    """Run all examples."""
    print("\n" + "=" * 50)
    print("Deep Learning Microservices - Usage Examples")
    print("=" * 50)
    print("\nMake sure all services are running before running examples.")
    print("Start services with:")
    print("  python services/llm_service/main.py")
    print("  python services/diffusion_service/main.py")
    print("  python services/training_service/main.py")
    print("  python services/gradio_service/main.py")
    print("\n")
    
    # Check service health first
    await check_services_health()
    print("\n")
    
    # Run examples
    try:
        await example_text_generation()
        await asyncio.sleep(1)
        
        # Uncomment to test image generation (requires GPU)
        # await example_image_generation()
        # await asyncio.sleep(1)
        
        await example_embeddings()
        await asyncio.sleep(1)
        
        # Uncomment to test training (takes longer)
        # await example_training_job()
        # await asyncio.sleep(1)
        
        # Uncomment to test Gradio interface
        # await example_gradio_interface()
        
    except httpx.ConnectError:
        print("\nError: Could not connect to services.")
        print("Please make sure all services are running.")
    except Exception as e:
        print(f"\nError: {str(e)}")
    
    print("\n" + "=" * 50)
    print("Examples completed!")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())



