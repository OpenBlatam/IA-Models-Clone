#!/usr/bin/env python3
"""
🔍 COMPLETE SYSTEM VALIDATION - Blaze AI Optimized
Comprehensive validation of all system components and features
"""

import asyncio
import time
import sys
import os
from pathlib import Path
import logging
import yaml
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('system_validation.log')
    ]
)
logger = logging.getLogger(__name__)

class CompleteSystemValidator:
    """Complete system validation for Blaze AI."""
    
    def __init__(self):
        self.validation_results = {}
        self.start_time = time.time()
        self.system_health = {}
    
    async def validate_file_structure(self):
        """Validate complete file structure."""
        logger.info("📁 VALIDATING: File Structure")
        logger.info("=" * 50)
        
        required_files = {
            "Core Application": [
                "optimized_main.py",
                "main.py",
                "main_enhanced.py"
            ],
            "Configuration": [
                "config-optimized.yaml",
                "config-enhanced.yaml",
                "config.yaml"
            ],
            "Dependencies": [
                "requirements-optimized.txt",
                "requirements-enhanced.txt",
                "requirements.txt"
            ],
            "Docker": [
                "Dockerfile.optimized",
                "docker-compose.optimized.yml",
                "deploy_optimized.sh"
            ],
            "Documentation": [
                "QUICK_START_OPTIMIZED.md",
                "OPTIMIZATION_SUMMARY.md",
                "README.md"
            ],
            "Enhanced Features": [
                "enhanced_features/security.py",
                "enhanced_features/monitoring.py",
                "enhanced_features/rate_limiting.py",
                "enhanced_features/error_handling.py"
            ]
        }
        
        structure_score = 0
        total_files = 0
        
        for category, files in required_files.items():
            logger.info(f"\n{category}:")
            category_score = 0
            
            for file_path in files:
                total_files += 1
                if Path(file_path).exists():
                    logger.info(f"  ✅ {file_path}")
                    category_score += 1
                    structure_score += 1
                else:
                    logger.warning(f"  ❌ {file_path} - MISSING")
            
            self.validation_results[f"structure_{category.lower().replace(' ', '_')}"] = f"{category_score}/{len(files)}"
        
        self.system_health['file_structure'] = f"{structure_score}/{total_files}"
        logger.info(f"\n📊 File Structure Score: {structure_score}/{total_files}")
    
    async def validate_configuration_files(self):
        """Validate configuration files syntax and content."""
        logger.info("\n🔧 VALIDATING: Configuration Files")
        logger.info("=" * 50)
        
        config_files = [
            "config-optimized.yaml",
            "config-enhanced.yaml"
        ]
        
        config_score = 0
        
        for config_file in config_files:
            try:
                if Path(config_file).exists():
                    with open(config_file, 'r', encoding='utf-8') as f:
                        config_content = yaml.safe_load(f)
                    
                    # Validate key sections
                    required_sections = ['app', 'api', 'security']
                    section_score = 0
                    
                    for section in required_sections:
                        if section in config_content:
                            section_score += 1
                            logger.info(f"  ✅ {config_file} - {section} section")
                        else:
                            logger.warning(f"  ⚠️  {config_file} - Missing {section} section")
                    
                    if section_score == len(required_sections):
                        config_score += 1
                        logger.info(f"✅ {config_file} - VALID")
                    else:
                        logger.warning(f"⚠️  {config_file} - INCOMPLETE")
                
            except Exception as e:
                logger.error(f"❌ {config_file} - ERROR: {e}")
        
        self.system_health['configuration'] = f"{config_score}/{len(config_files)}"
        logger.info(f"\n📊 Configuration Score: {config_score}/{len(config_files)}")
    
    async def validate_python_code(self):
        """Validate Python code syntax and imports."""
        logger.info("\n🐍 VALIDATING: Python Code Quality")
        logger.info("=" * 50)
        
        python_files = [
            "optimized_main.py",
            "main.py",
            "main_enhanced.py"
        ]
        
        code_score = 0
        
        for py_file in python_files:
            try:
                if Path(py_file).exists():
                    # Try to compile the Python file
                    with open(py_file, 'r', encoding='utf-8') as f:
                        source_code = f.read()
                    
                    compile(source_code, py_file, 'exec')
                    logger.info(f"  ✅ {py_file} - Syntax OK")
                    
                    # Check for key optimizations
                    optimization_features = [
                        "async def", "await", "@lru_cache", "FastAPI",
                        "class", "def", "import", "from"
                    ]
                    
                    feature_count = sum(1 for feature in optimization_features if feature in source_code)
                    if feature_count >= 3:
                        code_score += 1
                        logger.info(f"  ✅ {py_file} - Features OK")
                    else:
                        logger.warning(f"  ⚠️  {py_file} - Limited features")
                
            except Exception as e:
                logger.error(f"❌ {py_file} - SYNTAX ERROR: {e}")
        
        self.system_health['python_code'] = f"{code_score}/{len(python_files)}"
        logger.info(f"\n📊 Python Code Score: {code_score}/{len(python_files)}")
    
    async def validate_docker_configuration(self):
        """Validate Docker configuration files."""
        logger.info("\n🐳 VALIDATING: Docker Configuration")
        logger.info("=" * 50)
        
        docker_score = 0
        
        # Validate Dockerfile
        if Path("Dockerfile.optimized").exists():
            try:
                with open("Dockerfile.optimized", 'r') as f:
                    dockerfile_content = f.read()
                
                required_docker_features = [
                    "FROM", "COPY", "RUN", "EXPOSE", "CMD"
                ]
                
                feature_count = sum(1 for feature in required_docker_features if feature in dockerfile_content)
                if feature_count >= 4:
                    logger.info("  ✅ Dockerfile.optimized - VALID")
                    docker_score += 1
                else:
                    logger.warning("  ⚠️  Dockerfile.optimized - INCOMPLETE")
                
            except Exception as e:
                logger.error(f"❌ Dockerfile.optimized - ERROR: {e}")
        
        # Validate docker-compose
        if Path("docker-compose.optimized.yml").exists():
            try:
                with open("docker-compose.optimized.yml", 'r') as f:
                    compose_content = yaml.safe_load(f)
                
                if 'services' in compose_content and 'volumes' in compose_content:
                    logger.info("  ✅ docker-compose.optimized.yml - VALID")
                    docker_score += 1
                else:
                    logger.warning("  ⚠️  docker-compose.optimized.yml - INCOMPLETE")
                
            except Exception as e:
                logger.error(f"❌ docker-compose.optimized.yml - ERROR: {e}")
        
        self.system_health['docker'] = f"{docker_score}/2"
        logger.info(f"\n📊 Docker Score: {docker_score}/2")
    
    async def validate_dependencies(self):
        """Validate dependency files."""
        logger.info("\n📦 VALIDATING: Dependencies")
        logger.info("=" * 50)
        
        dep_score = 0
        
        req_files = [
            "requirements-optimized.txt",
            "requirements-enhanced.txt",
            "requirements.txt"
        ]
        
        for req_file in req_files:
            if Path(req_file).exists():
                try:
                    with open(req_file, 'r') as f:
                        dependencies = f.readlines()
                    
                    # Check for key dependencies
                    key_deps = ["fastapi", "uvicorn", "pydantic"]
                    dep_count = sum(1 for dep in dependencies if any(key in dep.lower() for key in key_deps))
                    
                    if dep_count >= 2:
                        logger.info(f"  ✅ {req_file} - Key dependencies found")
                        dep_score += 1
                    else:
                        logger.warning(f"  ⚠️  {req_file} - Limited dependencies")
                
                except Exception as e:
                    logger.error(f"❌ {req_file} - ERROR: {e}")
        
        self.system_health['dependencies'] = f"{dep_score}/{len(req_files)}"
        logger.info(f"\n📊 Dependencies Score: {dep_score}/{len(req_files)}")
    
    async def validate_documentation(self):
        """Validate documentation completeness."""
        logger.info("\n📚 VALIDATING: Documentation")
        logger.info("=" * 50)
        
        doc_score = 0
        
        doc_files = [
            "QUICK_START_OPTIMIZED.md",
            "OPTIMIZATION_SUMMARY.md",
            "README.md"
        ]
        
        for doc_file in doc_files:
            if Path(doc_file).exists():
                try:
                    with open(doc_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Check content length and key sections
                    if len(content) > 1000 and ("#" in content or "##" in content):
                        logger.info(f"  ✅ {doc_file} - Comprehensive")
                        doc_score += 1
                    else:
                        logger.warning(f"  ⚠️  {doc_file} - Basic content")
                
                except Exception as e:
                    logger.error(f"❌ {doc_file} - ERROR: {e}")
        
        self.system_health['documentation'] = f"{doc_score}/{len(doc_files)}"
        logger.info(f"\n📊 Documentation Score: {doc_score}/{len(doc_files)}")
    
    async def run_complete_validation(self):
        """Run complete system validation."""
        logger.info("🔍 STARTING COMPLETE SYSTEM VALIDATION")
        logger.info("=" * 60)
        
        # Run all validation steps
        await self.validate_file_structure()
        await self.validate_configuration_files()
        await self.validate_python_code()
        await self.validate_docker_configuration()
        await self.validate_dependencies()
        await self.validate_documentation()
        
        # Generate comprehensive report
        self.generate_validation_report()
    
    def generate_validation_report(self):
        """Generate comprehensive validation report."""
        end_time = time.time()
        duration = end_time - self.start_time
        
        logger.info("\n" + "=" * 60)
        logger.info("🔍 COMPLETE SYSTEM VALIDATION REPORT")
        logger.info("=" * 60)
        
        # Calculate overall scores
        total_score = 0
        max_score = 0
        
        for category, score in self.system_health.items():
            if '/' in str(score):
                current, maximum = map(int, str(score).split('/'))
                total_score += current
                max_score += maximum
                
                percentage = (current / maximum) * 100
                status = "✅" if percentage >= 80 else "⚠️" if percentage >= 60 else "❌"
                
                logger.info(f"{status} {category.replace('_', ' ').title()}: {score} ({percentage:.1f}%)")
        
        overall_percentage = (total_score / max_score) * 100 if max_score > 0 else 0
        
        logger.info("-" * 60)
        logger.info(f"📊 OVERALL SYSTEM HEALTH: {total_score}/{max_score} ({overall_percentage:.1f}%)")
        logger.info(f"⏱️  Validation duration: {duration:.2f} seconds")
        
        # System status
        if overall_percentage >= 90:
            logger.info("🎉 EXCELLENT! System is production-ready!")
        elif overall_percentage >= 80:
            logger.info("✅ GOOD! System is well-optimized!")
        elif overall_percentage >= 70:
            logger.info("⚠️  FAIR! Some improvements needed.")
        else:
            logger.warning("❌ NEEDS WORK! Significant improvements required.")
        
        # Recommendations
        logger.info("\n🚀 RECOMMENDATIONS:")
        if overall_percentage >= 90:
            logger.info("   - Deploy to production immediately")
            logger.info("   - Monitor performance metrics")
            logger.info("   - Scale as needed")
        elif overall_percentage >= 80:
            logger.info("   - Address minor issues")
            logger.info("   - Run performance tests")
            logger.info("   - Deploy to staging first")
        else:
            logger.info("   - Fix critical issues first")
            logger.info("   - Complete missing components")
            logger.info("   - Re-run validation")
        
        logger.info("=" * 60)
        
        # Save detailed results
        self.save_detailed_results()
    
    def save_detailed_results(self):
        """Save detailed validation results to file."""
        results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "system_health": self.system_health,
            "validation_results": self.validation_results
        }
        
        try:
            with open("validation_results.json", 'w') as f:
                json.dump(results, f, indent=2)
            logger.info("💾 Detailed results saved to validation_results.json")
        except Exception as e:
            logger.error(f"❌ Failed to save results: {e}")

async def main():
    """Main validation function."""
    validator = CompleteSystemValidator()
    await validator.run_complete_validation()

if __name__ == "__main__":
    asyncio.run(main())
