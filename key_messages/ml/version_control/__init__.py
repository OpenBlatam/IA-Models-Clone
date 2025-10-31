from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

from .git_manager import (
from .config_versioning import (
from .model_versioning import (
from .change_tracking import (
from ml.version_control import create_git_manager
from ml.config import get_config
from ml.version_control import create_config_version_manager
from ml.version_control import create_model_version_manager
from ml.version_control import create_change_tracker
from ml.version_control import (
from ml.config import get_config
from ml.version_control import (
from ml.version_control import VersionControlWorkflow
from ml.experiment_tracking import create_tracker
from ml.version_control import ModelRegistry
from ml.version_control import ConfigDiff
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Version Control Module for Key Messages ML Pipeline
Provides Git integration, configuration versioning, and model versioning
"""

    GitManager,
    GitConfig,
    GitCommit,
    GitBranch,
    GitTag
)
    ConfigVersionManager,
    ConfigSnapshot,
    ConfigDiff,
    ConfigHistory
)
    ModelVersionManager,
    ModelVersion,
    ModelRegistry,
    ModelMetadata
)
    ChangeTracker,
    ChangeLog,
    ChangeEntry,
    ChangeType
)

# Version information
__version__ = "1.0.0"

# Module exports
__all__ = [
    # Git management
    "GitManager",
    "GitConfig", 
    "GitCommit",
    "GitBranch",
    "GitTag",
    
    # Configuration versioning
    "ConfigVersionManager",
    "ConfigSnapshot",
    "ConfigDiff",
    "ConfigHistory",
    
    # Model versioning
    "ModelVersionManager",
    "ModelVersion",
    "ModelRegistry",
    "ModelMetadata",
    
    # Change tracking
    "ChangeTracker",
    "ChangeLog",
    "ChangeEntry",
    "ChangeType"
]

# Convenience functions
def create_git_manager(repo_path: str = None, config: dict = None) -> GitManager:
    """
    Create Git manager from configuration.
    
    Args:
        repo_path: Path to Git repository (defaults to current directory)
        config: Configuration dictionary with Git settings
        
    Returns:
        Configured Git manager
    """
    if config is None:
        config = {}
    
    git_config = GitConfig(
        repo_path=repo_path or config.get("repo_path", "."),
        user_name=config.get("user_name", "ML Pipeline"),
        user_email=config.get("user_email", "ml-pipeline@example.com"),
        auto_commit=config.get("auto_commit", True),
        auto_push=config.get("auto_push", False),
        commit_message_template=config.get("commit_message_template", 
                                         "Auto-commit: {change_type} - {description}")
    )
    
    return GitManager(git_config)

def create_config_version_manager(config: dict = None) -> ConfigVersionManager:
    """
    Create configuration version manager from configuration.
    
    Args:
        config: Configuration dictionary with versioning settings
        
    Returns:
        Configured configuration version manager
    """
    if config is None:
        config = {}
    
    return ConfigVersionManager(
        config_dir=config.get("config_dir", "./config_versions"),
        auto_snapshot=config.get("auto_snapshot", True),
        max_history=config.get("max_history", 50),
        compression=config.get("compression", True)
    )

def create_model_version_manager(config: dict = None) -> ModelVersionManager:
    """
    Create model version manager from configuration.
    
    Args:
        config: Configuration dictionary with model versioning settings
        
    Returns:
        Configured model version manager
    """
    if config is None:
        config = {}
    
    return ModelVersionManager(
        registry_path=config.get("registry_path", "./model_registry"),
        auto_version=config.get("auto_version", True),
        version_scheme=config.get("version_scheme", "semantic"),
        metadata_schema=config.get("metadata_schema", {}),
        compression=config.get("compression", True)
    )

def create_change_tracker(config: dict = None) -> ChangeTracker:
    """
    Create change tracker from configuration.
    
    Args:
        config: Configuration dictionary with change tracking settings
        
    Returns:
        Configured change tracker
    """
    if config is None:
        config = {}
    
    return ChangeTracker(
        log_file=config.get("log_file", "./change_log.json"),
        auto_log=config.get("auto_log", True),
        include_metadata=config.get("include_metadata", True),
        max_entries=config.get("max_entries", 1000)
    )

# Example usage
def example_usage():
    """
    Example usage of the version control system.
    """
    print("""
# Version Control Usage Examples

## 1. Basic Git Integration
```python

# Load configuration
config = get_config("production")
git_config = config["version_control"]["git"]

# Create Git manager
git_manager = create_git_manager(config=git_config)

# Initialize repository if needed
if not git_manager.is_repo():
    git_manager.init_repo()

# Stage and commit changes
git_manager.stage_all()
commit_hash = git_manager.commit("Update model configuration")

# Create tag for release
git_manager.create_tag("v1.0.0", "Release version 1.0.0")

# Push changes
git_manager.push()
```

## 2. Configuration Versioning
```python

# Create config version manager
config_manager = create_config_version_manager()

# Create snapshot of current configuration
snapshot = config_manager.create_snapshot(
    config=current_config,
    description="Updated model hyperparameters",
    tags=["production", "gpt2"]
)

# Get configuration history
history = config_manager.get_history(limit=10)
for entry in history:
    print(f"Version {entry.version}: {entry.description}")

# Compare configurations
diff = config_manager.compare_versions("v1.0.0", "v1.1.0")
print("Changes:", diff.changes)

# Restore previous configuration
config_manager.restore_version("v1.0.0")
```

## 3. Model Versioning
```python

# Create model version manager
model_manager = create_model_version_manager()

# Register a new model version
model_version = model_manager.register_model(
    model_path="./models/gpt2_key_messages.pt",
    model_name="gpt2_key_messages",
    version="1.0.0",
    metadata={
        "architecture": "gpt2",
        "dataset": "key_messages_v1",
        "accuracy": 0.85,
        "training_time": "2h 30m"
    },
    tags=["production", "gpt2"]
)

# List all model versions
versions = model_manager.list_versions("gpt2_key_messages")
for version in versions:
    print(f"Version {version.version}: {version.metadata['accuracy']}")

# Get model metadata
metadata = model_manager.get_metadata("gpt2_key_messages", "1.0.0")
print("Model accuracy:", metadata["accuracy"])

# Load specific model version
model = model_manager.load_model("gpt2_key_messages", "1.0.0")
```

## 4. Change Tracking
```python

# Create change tracker
change_tracker = create_change_tracker()

# Log configuration changes
change_tracker.log_change(
    change_type="config_update",
    description="Updated learning rate from 1e-4 to 5e-5",
    affected_files=["config.yaml"],
    metadata={"old_value": 1e-4, "new_value": 5e-5}
)

# Log model changes
change_tracker.log_change(
    change_type="model_training",
    description="Trained new GPT-2 model with improved dataset",
    affected_files=["models/gpt2_v2.pt"],
    metadata={"accuracy": 0.87, "training_time": "3h 15m"}
)

# Get change history
changes = change_tracker.get_changes(
    change_type="config_update",
    limit=10
)

for change in changes:
    print(f"{change.timestamp}: {change.description}")
```

## 5. Integrated Workflow
```python
    create_git_manager, 
    create_config_version_manager,
    create_model_version_manager,
    create_change_tracker
)

class VersionControlWorkflow:
    def __init__(self, config) -> Any:
        self.git_manager = create_git_manager(config=config["git"])
        self.config_manager = create_config_version_manager(config["config_versioning"])
        self.model_manager = create_model_version_manager(config["model_versioning"])
        self.change_tracker = create_change_tracker(config["change_tracking"])
        
    def update_configuration(self, new_config, description) -> Any:
        # Create config snapshot
        snapshot = self.config_manager.create_snapshot(
            config=new_config,
            description=description
        )
        
        # Log change
        self.change_tracker.log_change(
            change_type="config_update",
            description=description,
            affected_files=["config.yaml"]
        )
        
        # Git operations
        self.git_manager.stage_all()
        self.git_manager.commit(f"Config update: {description}")
        
        return snapshot.version
        
    def register_model(self, model_path, model_name, version, metadata) -> Any:
        # Register model version
        model_version = self.model_manager.register_model(
            model_path=model_path,
            model_name=model_name,
            version=version,
            metadata=metadata
        )
        
        # Log change
        self.change_tracker.log_change(
            change_type="model_registration",
            description=f"Registered {model_name} v{version}",
            affected_files=[model_path]
        )
        
        # Git operations
        self.git_manager.stage_all()
        self.git_manager.commit(f"Model registration: {model_name} v{version}")
        
        return model_version

# Usage
workflow = VersionControlWorkflow(config["version_control"])

# Update configuration
config_version = workflow.update_configuration(
    new_config=updated_config,
    description="Optimized hyperparameters for production"
)

# Register model
model_version = workflow.register_model(
    model_path="./models/gpt2_final.pt",
    model_name="gpt2_key_messages",
    version="1.1.0",
    metadata={"accuracy": 0.89, "training_time": "4h 30m"}
)
```

## 6. Configuration-Driven Setup
```yaml
# config.yaml
version_control:
  git:
    repo_path: "./ml_pipeline"
    user_name: "ML Pipeline"
    user_email: "ml-pipeline@example.com"
    auto_commit: true
    auto_push: false
    commit_message_template: "Auto-commit: {change_type} - {description}"
    
  config_versioning:
    config_dir: "./config_versions"
    auto_snapshot: true
    max_history: 50
    compression: true
    
  model_versioning:
    registry_path: "./model_registry"
    auto_version: true
    version_scheme: "semantic"
    compression: true
    metadata_schema:
      required_fields: ["architecture", "dataset", "accuracy"]
      optional_fields: ["training_time", "parameters", "description"]
    
  change_tracking:
    log_file: "./change_log.json"
    auto_log: true
    include_metadata: true
    max_entries: 1000
```

```python
# Usage with configuration
    create_git_manager,
    create_config_version_manager,
    create_model_version_manager,
    create_change_tracker
)

config = get_config("production")
vc_config = config["version_control"]

git_manager = create_git_manager(config=vc_config["git"])
config_manager = create_config_version_manager(vc_config["config_versioning"])
model_manager = create_model_version_manager(vc_config["model_versioning"])
change_tracker = create_change_tracker(vc_config["change_tracking"])
```

## 7. Automated Version Control
```python

class AutomatedVersionControl:
    def __init__(self, config) -> Any:
        self.workflow = VersionControlWorkflow(config["version_control"])
        self.tracker = create_tracker(config["experiment_tracking"])
        
    def on_config_change(self, old_config, new_config) -> Any:
        """Automatically version configuration changes."""
        diff = self._compute_config_diff(old_config, new_config)
        
        if diff.has_changes():
            version = self.workflow.update_configuration(
                new_config=new_config,
                description=f"Config update: {diff.summary()}"
            )
            
            # Log to experiment tracker
            self.tracker.log_metrics({
                "config_version": version,
                "config_changes": len(diff.changes)
            })
            
    def on_model_save(self, model_path, model_name, metadata) -> Any:
        """Automatically version model saves."""
        version = self._generate_version(model_name)
        
        model_version = self.workflow.register_model(
            model_path=model_path,
            model_name=model_name,
            version=version,
            metadata=metadata
        )
        
        # Log to experiment tracker
        self.tracker.log_metrics({
            "model_version": version,
            "model_accuracy": metadata.get("accuracy", 0)
        })
        
        return model_version
```

## 8. CI/CD Integration
```python
# .github/workflows/version_control.yml
name: Version Control

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  version_control:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
        with:
          fetch-depth: 0
          
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.9
          
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          
      - name: Run version control checks
        run: |
          python -m ml.version_control.ci_checks
          
      - name: Update version tags
        run: |
          python -m ml.version_control.update_tags
```

## 9. Model Registry API
```python

# Initialize registry
registry = ModelRegistry("./model_registry")

# Register model
registry.register(
    name="gpt2_key_messages",
    version="1.0.0",
    path="./models/gpt2.pt",
    metadata={
        "architecture": "gpt2",
        "dataset": "key_messages_v1",
        "accuracy": 0.85,
        "training_time": "2h 30m",
        "framework": "pytorch",
        "python_version": "3.9"
    }
)

# List models
models = registry.list_models()
for model in models:
    print(f"{model.name}: {model.latest_version}")

# Get model info
model_info = registry.get_model("gpt2_key_messages")
print(f"Versions: {model_info.versions}")
print(f"Latest: {model_info.latest_version}")

# Load model
model = registry.load_model("gpt2_key_messages", "1.0.0")
```

## 10. Configuration Diff Visualization
```python

# Create diff
diff = ConfigDiff(old_config, new_config)

# Print human-readable diff
print("Configuration Changes:")
for change in diff.changes:
    print(f"  {change.path}: {change.old_value} -> {change.new_value}")

# Generate diff report
report = diff.generate_report()
print(f"Total changes: {report.total_changes}")
print(f"Breaking changes: {report.breaking_changes}")

# Export diff to file
diff.export_to_file("config_diff.json")
```
""")

match __name__:
    case "__main__":
    example_usage() 