import asyncio
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent.parent.parent.parent
sys.path.append(str(project_root))

from agents.backend.onyx.server.features.lovable_contabilidad_mexicana_sam3.api.ai_controller import (
    analyze_accessibility,
    generate_alt_text,
    generate_meta_tags,
    suggest_keywords,
    generate_ui_component,
    generate_page_layout,
    AccessibilityRequest,
    ImageContextRequest,
    SEORequest,
    UIRequest
)

async def main():
    print("==================================================")
    print("TESTING LOVABLE CONTABILIDAD AI FEATURES")
    print("==================================================\n")

    # 1. Test Web Accessibility
    print("[Test 1] Web Accessibility Analysis")
    html_snippet = '<img src="logo.png"> <button>Click me</button>'
    print(f"Input HTML: {html_snippet}")
    try:
        result = await analyze_accessibility(AccessibilityRequest(html_snippet=html_snippet))
        print(f"Analysis: {result['analysis'][:100]}...\n")
    except Exception as e:
        print(f"Error: {e}\n")

    print("[Test 2] Alt Text Generation")
    context = "A photo of a happy dog running in a park with a frisbee."
    print(f"Context: {context}")
    try:
        result = await generate_alt_text(ImageContextRequest(context=context))
        print(f"Alt Text: {result['alt_text']}\n")
    except Exception as e:
        print(f"Error: {e}\n")

    # 2. Test SEO
    print("[Test 3] SEO Meta Tags")
    content = "Lovable Contabilidad is the best accounting software for Mexican businesses. We automate taxes, invoicing, and reporting."
    print(f"Content: {content}")
    try:
        result = await generate_meta_tags(SEORequest(content=content))
        print(f"Title: {result['title']}")
        print(f"Description: {result['description']}\n")
    except Exception as e:
        print(f"Error: {e}\n")

    print("[Test 4] SEO Keywords")
    try:
        result = await suggest_keywords(SEORequest(content=content))
        print(f"Keywords: {result['keywords']}\n")
    except Exception as e:
        print(f"Error: {e}\n")

    # 3. Test UI Generation
    print("[Test 5] UI Component Generation")
    description = "A blue primary button with rounded corners and white text."
    print(f"Description: {description}")
    try:
        result = await generate_ui_component(UIRequest(description=description))
        print(f"Code: {result['code'][:100]}...\n")
    except Exception as e:
        print(f"Error: {e}\n")

    print("[Test 6] Page Layout Generation")
    page_desc = "A landing page for an accounting firm with a hero section and services list."
    print(f"Description: {page_desc}")
    try:
        result = await generate_page_layout(UIRequest(description=page_desc))
        print(f"Layout: {result['layout'][:100]}...\n")
    except Exception as e:
        print(f"Error: {e}\n")

if __name__ == "__main__":
    asyncio.run(main())
