"""
Migration Guide Script
Helps users migrate from the old scattered architecture to the new unified system
"""

import os
import shutil
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MigrationGuide:
    """Migration helper for transitioning to the new architecture"""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.backup_dir = self.project_root / "backup_old_architecture"
        
    def create_backup(self):
        """Create backup of old architecture files"""
        logger.info("📦 Creating backup of old architecture...")
        
        if self.backup_dir.exists():
            shutil.rmtree(self.backup_dir)
        self.backup_dir.mkdir(exist_ok=True)
        
        # Backup old optimization files
        old_optimization_dir = self.project_root / "optimization_core"
        if old_optimization_dir.exists():
            backup_opt_dir = self.backup_dir / "optimization_core"
            shutil.copytree(old_optimization_dir, backup_opt_dir)
            logger.info("✅ Backed up optimization_core directory")
        
        # Backup old test files
        test_files = list(self.project_root.glob("test_*.py"))
        if test_files:
            backup_tests_dir = self.backup_dir / "old_tests"
            backup_tests_dir.mkdir(exist_ok=True)
            for test_file in test_files:
                shutil.copy2(test_file, backup_tests_dir / test_file.name)
            logger.info(f"✅ Backed up {len(test_files)} old test files")
        
        # Backup old variant directories
        variant_dirs = ["variant", "variant_optimized", "qwen_variant", "qwen_qwq_variant"]
        for variant_dir in variant_dirs:
            old_dir = self.project_root / variant_dir
            if old_dir.exists():
                backup_variant_dir = self.backup_dir / variant_dir
                shutil.copytree(old_dir, backup_variant_dir)
                logger.info(f"✅ Backed up {variant_dir} directory")
        
        logger.info("✅ Backup completed")
    
    def generate_migration_report(self):
        """Generate a migration report showing what was consolidated"""
        logger.info("📊 Generating migration report...")
        
        report = {
            "old_architecture": {
                "optimization_files": 0,
                "test_files": 0,
                "variant_directories": 0,
                "total_files": 0
            },
            "new_architecture": {
                "core_files": 0,
                "test_files": 0,
                "example_files": 0,
                "total_files": 0
            },
            "consolidation": {
                "files_reduced": 0,
                "directories_consolidated": 0,
                "maintainability_improvement": "High"
            }
        }
        
        # Count old files
        old_opt_dir = self.project_root / "optimization_core"
        if old_opt_dir.exists():
            report["old_architecture"]["optimization_files"] = len(list(old_opt_dir.glob("*.py")))
        
        report["old_architecture"]["test_files"] = len(list(self.project_root.glob("test_*.py")))
        
        variant_dirs = ["variant", "variant_optimized", "qwen_variant", "qwen_qwq_variant"]
        report["old_architecture"]["variant_directories"] = len([d for d in variant_dirs if (self.project_root / d).exists()])
        
        # Count new files
        core_dir = self.project_root / "core"
        if core_dir.exists():
            report["new_architecture"]["core_files"] = len(list(core_dir.glob("*.py")))
        
        tests_dir = self.project_root / "tests"
        if tests_dir.exists():
            report["new_architecture"]["test_files"] = len(list(tests_dir.glob("*.py")))
        
        examples_dir = self.project_root / "examples"
        if examples_dir.exists():
            report["new_architecture"]["example_files"] = len(list(examples_dir.glob("*.py")))
        
        # Calculate consolidation
        old_total = (report["old_architecture"]["optimization_files"] + 
                    report["old_architecture"]["test_files"] + 
                    report["old_architecture"]["variant_directories"])
        
        new_total = (report["new_architecture"]["core_files"] + 
                    report["new_architecture"]["test_files"] + 
                    report["new_architecture"]["example_files"])
        
        report["consolidation"]["files_reduced"] = old_total - new_total
        report["consolidation"]["directories_consolidated"] = report["old_architecture"]["variant_directories"]
        
        return report
    
    def create_migration_script(self):
        """Create a script to help users migrate their code"""
        migration_script = '''"""
Migration Script for TruthGPT
Helps migrate from old architecture to new unified system
"""

# OLD IMPORTS (what you used to import)
# from optimization_core.enhanced_optimization_core import EnhancedOptimizationCore
# from optimization_core.supreme_optimization_core import SupremeOptimizationCore
# from optimization_core.transcendent_optimization_core import TranscendentOptimizationCore
# from optimization_core.mega_enhanced_optimization_core import MegaEnhancedOptimizationCore
# from optimization_core.ultra_enhanced_optimization_core import UltraEnhancedOptimizationCore

# NEW IMPORTS (what you should use now)
from core import (
    OptimizationEngine, OptimizationConfig, OptimizationLevel,
    ModelManager, ModelConfig, ModelType,
    TrainingManager, TrainingConfig,
    InferenceEngine, InferenceConfig,
    MonitoringSystem
)

def migrate_optimization_code():
    """Example of how to migrate optimization code"""
    
    # OLD WAY (multiple imports and inconsistent APIs)
    # enhanced_core = EnhancedOptimizationCore()
    # supreme_core = SupremeOptimizationCore()
    # transcendent_core = TranscendentOptimizationCore()
    
    # NEW WAY (unified API)
    # Choose your optimization level
    config = OptimizationConfig(level=OptimizationLevel.ENHANCED)
    engine = OptimizationEngine(config)
    
    # All optimization levels use the same API
    optimized_model = engine.optimize_model(your_model)
    
    return optimized_model

def migrate_model_loading():
    """Example of how to migrate model loading"""
    
    # OLD WAY (scattered model files)
    # from variant.qwen_model import QwenModel
    # from variant_optimized.advanced_model import AdvancedModel
    
    # NEW WAY (unified model management)
    config = ModelConfig(
        model_type=ModelType.TRANSFORMER,
        model_name="my_model",
        hidden_size=768,
        num_layers=12
    )
    manager = ModelManager(config)
    model = manager.load_model()
    
    return model

def migrate_training_code():
    """Example of how to migrate training code"""
    
    # OLD WAY (multiple training scripts)
    # from test_enhanced_optimization_core import train_enhanced
    # from test_supreme_optimization_core import train_supreme
    
    # NEW WAY (unified training management)
    config = TrainingConfig(
        epochs=10,
        batch_size=32,
        learning_rate=1e-3
    )
    trainer = TrainingManager(config)
    trainer.setup_training(model, train_dataset, val_dataset)
    results = trainer.train()
    
    return results

def migrate_inference_code():
    """Example of how to migrate inference code"""
    
    # OLD WAY (basic inference)
    # output = model(input_tokens)
    
    # NEW WAY (optimized inference with caching)
    config = InferenceConfig(
        batch_size=1,
        max_length=512,
        temperature=0.8
    )
    inferencer = InferenceEngine(config)
    inferencer.load_model(model)
    
    result = inferencer.generate("Hello, world!", max_length=100)
    return result

def migrate_monitoring_code():
    """Example of how to migrate monitoring code"""
    
    # OLD WAY (no monitoring)
    # print("Training completed")
    
    # NEW WAY (comprehensive monitoring)
    monitor = MonitoringSystem()
    monitor.start_monitoring()
    
    # Add alert callbacks
    def alert_callback(alert_type, data):
        print(f"Alert: {alert_type} - {data}")
    
    monitor.add_alert_callback(alert_callback)
    
    # Get comprehensive metrics
    report = monitor.get_comprehensive_report()
    return report

# MIGRATION CHECKLIST
migration_checklist = [
    "✅ Replace old optimization imports with unified core imports",
    "✅ Update optimization code to use OptimizationEngine with configurable levels",
    "✅ Replace scattered model loading with ModelManager",
    "✅ Update training code to use TrainingManager",
    "✅ Add InferenceEngine for optimized inference",
    "✅ Add MonitoringSystem for performance tracking",
    "✅ Update test files to use new unified test suite",
    "✅ Remove old scattered files after migration",
    "✅ Update documentation to reference new architecture"
]

if __name__ == "__main__":
    print("🚀 TruthGPT Migration Guide")
    print("=" * 50)
    print("This script shows how to migrate from the old architecture to the new unified system.")
    print("\\nKey benefits of migration:")
    print("- Unified API across all optimization levels")
    print("- Better performance and memory management")
    print("- Comprehensive monitoring and metrics")
    print("- Cleaner, more maintainable code")
    print("- Extensive test coverage")
    print("\\nMigration checklist:")
    for item in migration_checklist:
        print(f"  {item}")
'''
        
        script_path = self.project_root / "migration_script.py"
        with open(script_path, 'w') as f:
            f.write(migration_script)
        
        logger.info(f"✅ Migration script created: {script_path}")
    
    def run_migration(self):
        """Run the complete migration process"""
        logger.info("🔄 Starting TruthGPT migration process...")
        
        # 1. Create backup
        self.create_backup()
        
        # 2. Generate migration report
        report = self.generate_migration_report()
        logger.info(f"📊 Migration report: {report}")
        
        # 3. Create migration script
        self.create_migration_script()
        
        logger.info("✅ Migration process completed!")
        logger.info("📁 Old files backed up to: backup_old_architecture/")
        logger.info("📝 Migration script created: migration_script.py")
        logger.info("📖 See REFACTORED_README.md for detailed usage instructions")

def main():
    """Main migration function"""
    project_root = "."
    migrator = MigrationGuide(project_root)
    migrator.run_migration()

if __name__ == "__main__":
    main()

