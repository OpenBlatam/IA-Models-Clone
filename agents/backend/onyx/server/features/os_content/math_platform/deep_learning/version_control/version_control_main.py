from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import os
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import argparse
from datetime import datetime
from git_manager import GitManager, GitConfig
from experiment_tracker import ExperimentTracker, ExperimentConfig
from model_checkpointer import ModelCheckpointer, CheckpointConfig
        import zipfile
from typing import Any, List, Dict, Optional
import asyncio
#!/usr/bin/env python3
"""
Version Control Main
Complete version control system integration for deep learning projects.
"""


# Import version control components

logger = logging.getLogger(__name__)


class VersionControlSystem:
    """Complete version control system for deep learning projects."""
    
    def __init__(self, project_root: str = ".", config_file: str = None):
        
    """__init__ function."""
self.project_root = Path(project_root)
        self.config_file = config_file
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize components
        self.git_manager = None
        self.experiment_tracker = None
        self.model_checkpointer = None
        
        self._initialize_components()
        
        logger.info(f"Initialized version control system: {self.project_root}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load version control configuration."""
        if self.config_file and Path(self.config_file).exists():
            with open(self.config_file, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                return yaml.safe_load(f)
        
        # Default configuration
        return {
            'git': {
                'enabled': True,
                'auto_commit': True,
                'track_configs': True,
                'track_models': False,
                'author_name': 'Deep Learning Team',
                'author_email': 'team@example.com'
            },
            'experiments': {
                'enabled': True,
                'track_code': True,
                'track_configs': True,
                'track_models': True,
                'track_metrics': True,
                'auto_commit': True
            },
            'checkpoints': {
                'enabled': True,
                'save_best_only': True,
                'save_last': True,
                'save_top_k': 3,
                'git_enabled': True,
                'git_auto_commit': True
            },
            'project': {
                'name': 'deep_learning_project',
                'description': 'Deep learning project with version control',
                'created_by': 'Deep Learning Team'
            }
        }
    
    def _initialize_components(self) -> Any:
        """Initialize version control components."""
        # Initialize Git manager
        if self.config['git']['enabled']:
            git_config = GitConfig(
                repo_path=str(self.project_root),
                author_name=self.config['git']['author_name'],
                author_email=self.config['git']['author_email'],
                track_configs=self.config['git']['track_configs'],
                track_models=self.config['git']['track_models'],
                auto_commit=self.config['git']['auto_commit']
            )
            self.git_manager = GitManager(git_config)
            logger.info("Initialized Git manager")
        
        # Initialize experiment tracker
        if self.config['experiments']['enabled']:
            exp_config = ExperimentConfig(
                experiment_name=self.config['project']['name'],
                description=self.config['project']['description'],
                created_by=self.config['project']['created_by'],
                track_code=self.config['experiments']['track_code'],
                track_configs=self.config['experiments']['track_configs'],
                track_models=self.config['experiments']['track_models'],
                track_metrics=self.config['experiments']['track_metrics'],
                git_enabled=self.config['experiments']['auto_commit']
            )
            self.experiment_tracker = ExperimentTracker(exp_config)
            logger.info("Initialized experiment tracker")
        
        # Initialize model checkpointer
        if self.config['checkpoints']['enabled']:
            checkpoint_config = CheckpointConfig(
                model_name=self.config['project']['name'],
                checkpoint_dir=str(self.project_root / "checkpoints"),
                save_best_only=self.config['checkpoints']['save_best_only'],
                save_last=self.config['checkpoints']['save_last'],
                save_top_k=self.config['checkpoints']['save_top_k'],
                git_enabled=self.config['checkpoints']['git_enabled'],
                git_auto_commit=self.config['checkpoints']['git_auto_commit']
            )
            self.model_checkpointer = ModelCheckpointer(checkpoint_config)
            logger.info("Initialized model checkpointer")
    
    def initialize_project(self) -> Any:
        """Initialize version control for a new project."""
        logger.info("Initializing version control for new project...")
        
        # Create project structure
        directories = [
            "configs",
            "models",
            "data",
            "experiments",
            "checkpoints",
            "logs",
            "results",
            "tests",
            "docs"
        ]
        
        for directory in directories:
            (self.project_root / directory).mkdir(parents=True, exist_ok=True)
        
        # Create .gitignore
        gitignore_content = self._get_gitignore_content()
        gitignore_path = self.project_root / ".gitignore"
        with open(gitignore_path, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write(gitignore_content)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        
        # Create README
        readme_content = self._get_readme_content()
        readme_path = self.project_root / "README.md"
        with open(readme_path, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write(readme_content)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        
        # Create initial configuration
        config_content = self._get_initial_config_content()
        config_path = self.project_root / "configs" / "project_config.yaml"
        with open(config_path, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            yaml.dump(config_content, f, default_flow_style=False, indent=2)
        
        # Initialize Git repository
        if self.git_manager:
            self.git_manager.add_file(str(gitignore_path), "Add .gitignore")
            self.git_manager.add_file(str(readme_path), "Add README")
            self.git_manager.add_file(str(config_path), "Add initial configuration")
            self.git_manager.commit("Initial project setup", commit_type="init")
        
        logger.info("Project initialization completed")
    
    def start_experiment(self, experiment_name: str, config_file: str = None) -> str:
        """Start a new experiment with version control."""
        if not self.experiment_tracker:
            raise RuntimeError("Experiment tracker not initialized")
        
        # Start experiment run
        run_id = self.experiment_tracker.start_run(experiment_name, config_file)
        
        # Git commit if enabled
        if self.git_manager and self.config['git']['auto_commit']:
            self.git_manager.commit(f"Start experiment: {experiment_name}", commit_type="experiment")
        
        logger.info(f"Started experiment: {experiment_name} (run_id: {run_id})")
        return run_id
    
    def end_experiment(self, status: str = "completed", final_metrics: Dict[str, Any] = None):
        """End the current experiment."""
        if not self.experiment_tracker:
            raise RuntimeError("Experiment tracker not initialized")
        
        self.experiment_tracker.end_run(status, final_metrics)
        
        # Git commit if enabled
        if self.git_manager and self.config['git']['auto_commit']:
            self.git_manager.commit(f"End experiment: {status}", commit_type="experiment")
        
        logger.info(f"Ended experiment: {status}")
    
    def log_metric(self, name: str, value: float, step: int = None):
        """Log a metric for the current experiment."""
        if not self.experiment_tracker:
            raise RuntimeError("Experiment tracker not initialized")
        
        self.experiment_tracker.log_metric(name, value, step)
    
    def save_checkpoint(self, model, optimizer=None, scheduler=None, scaler=None,
                       epoch: int = 0, step: int = 0, metrics: Dict[str, float] = None,
                       description: str = "", tags: List[str] = None) -> str:
        """Save a model checkpoint with version control."""
        if not self.model_checkpointer:
            raise RuntimeError("Model checkpointer not initialized")
        
        checkpoint_id = self.model_checkpointer.save_checkpoint(
            model, optimizer, scheduler, scaler, epoch, step, metrics, description, tags
        )
        
        logger.info(f"Saved checkpoint: {checkpoint_id}")
        return checkpoint_id
    
    def load_checkpoint(self, checkpoint_id: str, model=None, optimizer=None,
                       scheduler=None, scaler=None, map_location: str = "cpu") -> Dict[str, Any]:
        """Load a model checkpoint."""
        if not self.model_checkpointer:
            raise RuntimeError("Model checkpointer not initialized")
        
        return self.model_checkpointer.load_checkpoint(
            checkpoint_id, model, optimizer, scheduler, scaler, map_location
        )
    
    def commit_changes(self, message: str, commit_type: str = "update"):
        """Commit changes to Git."""
        if not self.git_manager:
            raise RuntimeError("Git manager not initialized")
        
        self.git_manager.add_all_files()
        self.git_manager.commit(message, commit_type)
        logger.info(f"Committed changes: {message}")
    
    def create_tag(self, tag_name: str, message: str = None):
        """Create a Git tag."""
        if not self.git_manager:
            raise RuntimeError("Git manager not initialized")
        
        self.git_manager.create_tag(tag_name, message)
        logger.info(f"Created tag: {tag_name}")
    
    def get_project_status(self) -> Dict[str, Any]:
        """Get comprehensive project status."""
        status = {
            'project_root': str(self.project_root),
            'git_status': None,
            'experiment_status': None,
            'checkpoint_status': None
        }
        
        if self.git_manager:
            status['git_status'] = self.git_manager.get_status()
        
        if self.experiment_tracker:
            status['experiment_status'] = {
                'current_run': self.experiment_tracker.current_run.to_dict() if self.experiment_tracker.current_run else None,
                'total_runs': len(self.experiment_tracker.list_runs())
            }
        
        if self.model_checkpointer:
            status['checkpoint_status'] = {
                'total_checkpoints': len(self.model_checkpointer.list_checkpoints()),
                'best_checkpoints': {k: v.checkpoint_id for k, v in self.model_checkpointer.best_checkpoints.items()}
            }
        
        return status
    
    def export_project(self, export_path: str = None) -> str:
        """Export the entire project with version control history."""
        if export_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_path = f"{self.config['project']['name']}_{timestamp}.zip"
        
        
        with zipfile.ZipFile(export_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add all project files
            for file_path in self.project_root.rglob('*'):
                if file_path.is_file() and '.git' not in str(file_path):
                    arcname = file_path.relative_to(self.project_root)
                    zipf.write(file_path, arcname)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        
        logger.info(f"Exported project to: {export_path}")
        return export_path
    
    def _get_gitignore_content(self) -> str:
        """Get content for .gitignore file."""
        return """# Deep Learning Project .gitignore

# Python
__pycache__/
*.py[cod]
*.so
.Python
build/
dist/
*.egg-info/

# Deep Learning specific
*.pth
*.pt
*.ckpt
*.h5
*.pkl
checkpoints/
logs/
runs/
wandb/
data/raw/
data/processed/
*.csv
*.json
*.npy
*.npz

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Keep important files
!config/*.yaml
!config/*.yml
!requirements.txt
!README.md
"""
    
    def _get_readme_content(self) -> str:
        """Get content for README.md file."""
        return f"""# {self.config['project']['name']}

{self.config['project']['description']}

## Project Structure

```
{self.config['project']['name']}/
├── configs/          # Configuration files
├── models/           # Model definitions
├── data/             # Data files
├── experiments/      # Experiment tracking
├── checkpoints/      # Model checkpoints
├── logs/             # Log files
├── results/          # Results and outputs
├── tests/            # Test files
└── docs/             # Documentation
```

## Getting Started

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Initialize version control:
```bash
python version_control_main.py init
```

3. Start an experiment:
```bash
python version_control_main.py start-experiment experiment_name
```

## Version Control

This project uses comprehensive version control for:
- Code changes (Git)
- Experiment tracking
- Model checkpointing
- Configuration management

## Contributing

1. Create a new branch for your changes
2. Make your changes
3. Commit with descriptive messages
4. Create a pull request

## License

MIT License
"""
    
    def _get_initial_config_content(self) -> Dict[str, Any]:
        """Get initial configuration content."""
        return {
            'project': self.config['project'],
            'git': self.config['git'],
            'experiments': self.config['experiments'],
            'checkpoints': self.config['checkpoints']
        }


class VersionControlCLI:
    """Command-line interface for version control system."""
    
    def __init__(self, vc_system: VersionControlSystem):
        
    """__init__ function."""
self.vc_system = vc_system
    
    def create_parser(self) -> Any:
        """Create command-line argument parser."""
        parser = argparse.ArgumentParser(description="Version Control System CLI")
        subparsers = parser.add_subparsers(dest='command', help='Available commands')
        
        # Initialize project
        init_parser = subparsers.add_parser('init', help='Initialize version control for new project')
        
        # Start experiment
        start_parser = subparsers.add_parser('start-experiment', help='Start a new experiment')
        start_parser.add_argument('experiment_name', help='Name of the experiment')
        start_parser.add_argument('--config', help='Configuration file path')
        
        # End experiment
        end_parser = subparsers.add_parser('end-experiment', help='End the current experiment')
        end_parser.add_argument('--status', default='completed', help='Experiment status')
        
        # Log metric
        metric_parser = subparsers.add_parser('log-metric', help='Log a metric')
        metric_parser.add_argument('name', help='Metric name')
        metric_parser.add_argument('value', type=float, help='Metric value')
        metric_parser.add_argument('--step', type=int, help='Step number')
        
        # Save checkpoint
        checkpoint_parser = subparsers.add_parser('save-checkpoint', help='Save model checkpoint')
        checkpoint_parser.add_argument('--epoch', type=int, default=0, help='Epoch number')
        checkpoint_parser.add_argument('--step', type=int, default=0, help='Step number')
        checkpoint_parser.add_argument('--description', help='Checkpoint description')
        
        # Load checkpoint
        load_parser = subparsers.add_parser('load-checkpoint', help='Load model checkpoint')
        load_parser.add_argument('checkpoint_id', help='Checkpoint ID')
        
        # Commit changes
        commit_parser = subparsers.add_parser('commit', help='Commit changes')
        commit_parser.add_argument('message', help='Commit message')
        commit_parser.add_argument('--type', default='update', help='Commit type')
        
        # Create tag
        tag_parser = subparsers.add_parser('tag', help='Create Git tag')
        tag_parser.add_argument('tag_name', help='Tag name')
        tag_parser.add_argument('--message', help='Tag message')
        
        # Status
        status_parser = subparsers.add_parser('status', help='Show project status')
        
        # Export project
        export_parser = subparsers.add_parser('export', help='Export project')
        export_parser.add_argument('--output', help='Output file path')
        
        return parser
    
    def run(self, args=None) -> Any:
        """Run the CLI."""
        parser = self.create_parser()
        parsed_args = parser.parse_args(args)
        
        if parsed_args.command == 'init':
            self.vc_system.initialize_project()
            print("Project initialized successfully")
        
        elif parsed_args.command == 'start-experiment':
            run_id = self.vc_system.start_experiment(parsed_args.experiment_name, parsed_args.config)
            print(f"Started experiment: {run_id}")
        
        elif parsed_args.command == 'end-experiment':
            self.vc_system.end_experiment(parsed_args.status)
            print("Experiment ended")
        
        elif parsed_args.command == 'log-metric':
            self.vc_system.log_metric(parsed_args.name, parsed_args.value, parsed_args.step)
            print(f"Logged metric: {parsed_args.name} = {parsed_args.value}")
        
        elif parsed_args.command == 'save-checkpoint':
            # This would typically be called from within training code
            print("Checkpoint saving requires model object - use from code")
        
        elif parsed_args.command == 'load-checkpoint':
            try:
                result = self.vc_system.load_checkpoint(parsed_args.checkpoint_id)
                print("Checkpoint loaded successfully")
            except Exception as e:
                print(f"Error loading checkpoint: {e}")
        
        elif parsed_args.command == 'commit':
            self.vc_system.commit_changes(parsed_args.message, parsed_args.type)
            print("Changes committed")
        
        elif parsed_args.command == 'tag':
            self.vc_system.create_tag(parsed_args.tag_name, parsed_args.message)
            print(f"Created tag: {parsed_args.tag_name}")
        
        elif parsed_args.command == 'status':
            status = self.vc_system.get_project_status()
            print(json.dumps(status, indent=2))
        
        elif parsed_args.command == 'export':
            export_path = self.vc_system.export_project(parsed_args.output)
            print(f"Exported project to: {export_path}")
        
        else:
            parser.print_help()


# Example usage
if __name__ == "__main__":
    # Create version control system
    vc_system = VersionControlSystem("./deep_learning_project")
    
    # Create CLI
    cli = VersionControlCLI(vc_system)
    
    # Run CLI
    cli.run() 