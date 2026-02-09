"""Script to generate API documentation."""

import json
from pathlib import Path
from typing import Dict, Any


def generate_openapi_spec(app) -> Dict[str, Any]:
    """
    Generate OpenAPI specification.
    
    Args:
        app: FastAPI application
        
    Returns:
        OpenAPI specification dictionary
    """
    return app.openapi()


def save_openapi_spec(spec: Dict[str, Any], output_path: Path) -> None:
    """
    Save OpenAPI specification to file.
    
    Args:
        spec: OpenAPI specification
        output_path: Output file path
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(spec, f, indent=2)
    
    print(f"OpenAPI spec saved to {output_path}")


def generate_api_docs():
    """Generate API documentation."""
    from main import app
    
    spec = generate_openapi_spec(app)
    output_path = Path("./docs/openapi.json")
    save_openapi_spec(spec, output_path)
    
    print("API documentation generated successfully")


if __name__ == "__main__":
    generate_api_docs()

