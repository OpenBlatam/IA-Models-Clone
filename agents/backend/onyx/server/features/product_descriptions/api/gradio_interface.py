from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import gradio as gr
import asyncio
import logging
from typing import List, Tuple, Dict, Any
import json
from ..core.generator import ProductDescriptionGenerator
from ..core.config import ProductDescriptionConfig
    import asyncio
from typing import Any, List, Dict, Optional
"""
Gradio Interface for Product Description Generator
=================================================

Interactive web interface using Gradio for easy testing and demonstration.
"""



logger = logging.getLogger(__name__)


class ProductDescriptionGradioApp:
    """Gradio interface for Product Description Generator."""
    
    def __init__(self, generator: ProductDescriptionGenerator):
        
    """__init__ function."""
self.generator = generator
        self.app = None
        
    def create_interface(self) -> gr.Interface:
        """Create Gradio interface."""
        
        def generate_description(
            product_name: str,
            features: str,
            category: str,
            brand: str,
            style: str,
            tone: str,
            temperature: float,
            max_length: int,
            num_variations: int
        ) -> Tuple[str, str, str]:
            """Generate description through Gradio interface."""
            
            try:
                # Parse features
                feature_list = [f.strip() for f in features.split(',') if f.strip()]
                
                if not feature_list:
                    return "Error: Please provide at least one feature", "", ""
                
                # Generate
                results = self.generator.generate(
                    product_name=product_name,
                    features=feature_list,
                    category=category,
                    brand=brand,
                    style=style,
                    tone=tone,
                    temperature=temperature,
                    max_length=max_length,
                    num_variations=num_variations
                )
                
                # Format output
                descriptions = []
                metadata_info = []
                
                for i, result in enumerate(results):
                    descriptions.append(f"**Variation {i+1}:**\n{result['description']}\n")
                    
                    metadata_info.append(
                        f"**Variation {i+1} Stats:**\n"
                        f"- Quality Score: {result['quality_score']:.2f}\n"
                        f"- SEO Score: {result['seo_score']:.2f}\n"
                        f"- Word Count: {result['metadata']['word_count']}\n"
                        f"- Character Count: {result['metadata']['char_count']}\n"
                    )
                
                descriptions_text = "\n".join(descriptions)
                metadata_text = "\n".join(metadata_info)
                
                # JSON output
                json_output = json.dumps(results, indent=2)
                
                return descriptions_text, metadata_text, json_output
                
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                return error_msg, "", ""
        
        # Define interface
        interface = gr.Interface(
            fn=generate_description,
            inputs=[
                gr.Textbox(
                    label="Product Name",
                    placeholder="e.g., iPhone 15 Pro",
                    value="Wireless Bluetooth Headphones"
                ),
                gr.Textbox(
                    label="Features (comma-separated)",
                    placeholder="e.g., noise cancellation, 20-hour battery, wireless charging",
                    value="Active noise cancellation, 30-hour battery life, Quick charge, Premium leather"
                ),
                gr.Dropdown(
                    choices=["electronics", "clothing", "home", "sports", "beauty", "books", "general"],
                    label="Category",
                    value="electronics"
                ),
                gr.Textbox(
                    label="Brand",
                    placeholder="e.g., Apple, Samsung, Sony",
                    value="TechPro"
                ),
                gr.Dropdown(
                    choices=["professional", "casual", "luxury", "technical", "creative"],
                    label="Writing Style",
                    value="professional"
                ),
                gr.Dropdown(
                    choices=["friendly", "formal", "enthusiastic", "informative", "persuasive"],
                    label="Writing Tone",
                    value="friendly"
                ),
                gr.Slider(
                    minimum=0.1,
                    maximum=2.0,
                    step=0.1,
                    label="Temperature (creativity)",
                    value=0.7
                ),
                gr.Slider(
                    minimum=50,
                    maximum=500,
                    step=10,
                    label="Max Length",
                    value=300
                ),
                gr.Slider(
                    minimum=1,
                    maximum=3,
                    step=1,
                    label="Number of Variations",
                    value=1
                )
            ],
            outputs=[
                gr.Textbox(
                    label="Generated Descriptions",
                    lines=10,
                    max_lines=20
                ),
                gr.Textbox(
                    label="Metadata & Scores",
                    lines=8
                ),
                gr.Textbox(
                    label="JSON Output",
                    lines=15,
                    max_lines=25
                )
            ],
            title="🛍️ AI Product Description Generator",
            description="Generate compelling product descriptions using advanced AI models. Customize style, tone, and parameters to match your brand.",
            examples=[
                [
                    "Premium Coffee Maker",
                    "Programmable, Built-in grinder, Thermal carafe, Auto-shutoff",
                    "home",
                    "BrewMaster",
                    "professional",
                    "enthusiastic",
                    0.8,
                    250,
                    2
                ],
                [
                    "Gaming Laptop",
                    "RTX 4080, 16GB RAM, 1TB SSD, 144Hz display, RGB keyboard",
                    "electronics",
                    "GameForce",
                    "technical",
                    "informative",
                    0.6,
                    400,
                    1
                ],
                [
                    "Designer Handbag",
                    "Genuine leather, Multiple compartments, Adjustable strap, Luxury finish",
                    "clothing",
                    "LuxeStyle",
                    "luxury",
                    "persuasive",
                    0.9,
                    300,
                    2
                ]
            ],
            cache_examples=True,
            theme=gr.themes.Soft()
        )
        
        self.app = interface
        return interface
    
    def launch(self, **kwargs) -> Any:
        """Launch the Gradio interface."""
        if not self.app:
            self.create_interface()
        
        return self.app.launch(**kwargs)


# Standalone function to create and launch app
def create_gradio_app(config: ProductDescriptionConfig = None) -> ProductDescriptionGradioApp:
    """Create and initialize Gradio app."""
    
    # Initialize generator
    generator = ProductDescriptionGenerator(config)
    
    # Run initialization in event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(generator.initialize())
    
    # Create app
    return ProductDescriptionGradioApp(generator) 