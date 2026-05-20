import asyncio
import time
import sys
import os
import httpx

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend_ads.llm_interface import generate_ads_lcel, DEEPSEEK_API_KEY, DEEPSEEK_API_URL

async def test_api_key():
    print("Testing API key...")
    
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Simple test message
    data = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": "Hello, this is a test message."}],
        "temperature": 0.7
    }
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{DEEPSEEK_API_URL}/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                print("✅ API Key is valid and working!")
                print(f"Response status: {response.status_code}")
                response_data = response.json()
                print(f"Model response: {response_data['choices'][0]['message']['content']}")
            else:
                print(f"❌ API Key test failed!")
                print(f"Status code: {response.status_code}")
                print(f"Error message: {response.text}")
                
    except Exception as e:
        print(f"❌ Error testing API key: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_api_key()) 