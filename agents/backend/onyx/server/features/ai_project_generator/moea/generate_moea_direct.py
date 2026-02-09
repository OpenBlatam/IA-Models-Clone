"""Direct script to generate MOEA project using ProjectGenerator"""
import asyncio
import sys
import logging
from pathlib import Path

# Add the features directory to path to import the module
features_dir = Path(__file__).parent.parent
sys.path.insert(0, str(features_dir))

from ai_project_generator.core.project_generator import ProjectGenerator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def main():
    """Generate MOEA project"""
    print("🚀 Generating MOEA Optimization System...")
    
    # Create generator
    generator = ProjectGenerator(
        base_output_dir="generated_projects",
        backend_framework="fastapi",
        frontend_framework="react"
    )
    
    # Generate project
    description = (
        "A Multi-Objective Evolutionary Algorithm (MOEA) system for solving optimization problems "
        "with multiple conflicting objectives. The system should support various MOEA algorithms like "
        "NSGA-II, NSGA-III, MOEA/D, and SPEA2. It should include visualization of Pareto fronts, "
        "performance metrics calculation (hypervolume, IGD, GD), comparison tools, and interactive "
        "parameter tuning. The system should handle real-time optimization, batch processing, and "
        "export results in various formats."
    )
    
    try:
        result = await generator.generate_project(
            description=description,
            project_name="moea_optimization_system",
            author="Blatam Academy",
            version="1.0.0"
        )
        
        print("\n✅ Project generated successfully!")
        print(f"   Project directory: {result.get('project_dir', 'N/A')}")
        print(f"   Backend path: {result.get('backend_path', 'N/A')}")
        print(f"   Frontend path: {result.get('frontend_path', 'N/A')}")
        print(f"   Project info: {result.get('project_info', {})}")
        
    except Exception as e:
        print(f"\n❌ Error generating project: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

