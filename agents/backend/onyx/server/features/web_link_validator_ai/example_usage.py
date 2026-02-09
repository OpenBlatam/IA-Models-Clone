"""Example usage of Web Link Validator AI"""
import asyncio
import httpx


async def example_validate_single_link():
    """Example: Validate a single link"""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8025/validate",
            json={
                "url": "https://docs.python.org",
                "query": "Python documentation"
            }
        )
        print("Single Link Validation:")
        print(response.json())
        print()


async def example_validate_multiple_links():
    """Example: Validate multiple links"""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8025/validate/batch",
            json={
                "urls": [
                    "https://docs.python.org",
                    "https://example.com",
                    "https://fake-link-that-does-not-exist-12345.com"
                ],
                "query": "Python programming resources"
            }
        )
        print("Multiple Links Validation:")
        for result in response.json():
            print(f"URL: {result['url']}")
            print(f"  Valid: {result['valid']}")
            print(f"  Exists: {result['exists']}")
            print(f"  Relevant: {result['relevant']}")
            print(f"  Score: {result['relevance_score']}")
            print()


async def example_quick_check():
    """Example: Quick existence check"""
    async with httpx.AsyncClient() as client:
        response = await client.get(
            "http://localhost:8025/check/https://docs.python.org"
        )
        print("Quick Check:")
        print(response.json())
        print()


async def main():
    """Run all examples"""
    print("=" * 60)
    print("Web Link Validator AI - Example Usage")
    print("=" * 60)
    print()
    
    # Make sure the server is running on localhost:8025
    try:
        await example_validate_single_link()
        await example_validate_multiple_links()
        await example_quick_check()
    except httpx.ConnectError:
        print("Error: Could not connect to server.")
        print("Make sure the server is running on http://localhost:8025")
        print("Start it with: python main.py")


if __name__ == "__main__":
    asyncio.run(main())

