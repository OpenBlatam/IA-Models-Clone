# Version Control System Guide

## Overview

This guide covers the comprehensive version control system designed for deep learning projects. The system provides Git integration along with specialized versioning for configurations, models, and experiments, ensuring complete reproducibility and change tracking.

## Key Features

### 1. Git Integration
- **Automatic Git operations** for code versioning
- **Commit and tag management** for releases
- **Branch management** for feature development
- **Diff tracking** for change visualization

### 2. Configuration Versioning
- **YAML configuration tracking** with diff generation
- **Version history** with metadata preservation
- **Rollback capabilities** to previous configurations
- **Configuration comparison** and validation

### 3. Model Versioning
- **Model file versioning** with metadata preservation
- **Performance metrics tracking** for each version
- **Model comparison** and selection
- **Artifact management** for large model files

### 4. Experiment Versioning
- **Complete experiment tracking** with results
- **Reproducibility assurance** through metadata
- **Performance comparison** across versions
- **Artifact preservation** including plots and logs

## Architecture Components

### Core Classes

1. **VersionMetadata**
   - Version identification and metadata
   - Change tracking and diff information
   - Performance metrics and dependencies
   - Timestamp and author information

2. **GitManager**
   - Git repository operations
   - Commit and tag management
   - Status tracking and history
   - Branch management

3. **ConfigurationVersioner**
   - YAML configuration versioning
   - Diff generation and comparison
   - Version history management
   - Rollback capabilities

4. **ModelVersioner**
   - Model file versioning
   - Metadata preservation
   - Performance tracking
   - Artifact management

5. **ExperimentVersioner**
   - Experiment result versioning
   - Complete metadata preservation
   - Performance comparison
   - Reproducibility tracking

6. **VersionControlSystem**
   - Main orchestrator for all versioning
   - Git integration coordination
   - Snapshot management
   - Project status tracking

## Configuration

### Basic Configuration
```yaml
# .version_control.yaml
project_info:
  name: "deep-learning-project"
  description: "Production deep learning project"
  version: "1.0.0"

versioning:
  auto_commit: true
  auto_tag: true
  snapshot_frequency: "weekly"
  
git:
  remote_url: "https://github.com/user/repo.git"
  default_branch: "main"
  protected_branches: ["main", "production"]
  
storage:
  config_dir: "configs"
  model_dir: "models"
  experiment_dir: "experiments"
  max_versions_per_item: 50
```

### Advanced Configuration
```yaml
# Advanced version control configuration
versioning:
  auto_commit: true
  auto_tag: true
  snapshot_frequency: "daily"
  compression: "gzip"
  encryption: false
  
git:
  remote_url: "https://github.com/user/repo.git"
  default_branch: "main"
  protected_branches: ["main", "production"]
  commit_message_template: "[{type}] {description}"
  tag_naming: "{type}-{name}-{version}"
  
storage:
  config_dir: "configs"
  model_dir: "models"
  experiment_dir: "experiments"
  max_versions_per_item: 100
  cleanup_old_versions: true
  compression_threshold: "100MB"
  
security:
  encrypt_sensitive_configs: true
  access_control: true
  audit_logging: true
```

## Usage Patterns

### 1. Basic Version Control Operations

```python
from version_control import version_control

# Initialize version control system
with version_control("my_project", auto_commit=True) as vc:
    # Version configuration
    config_version = vc.version_configuration(
        config_name="transformer_config",
        config_data={
            "model": {"type": "transformer", "layers": 12},
            "training": {"epochs": 10, "lr": 2e-5}
        },
        description="Initial transformer configuration",
        tags=["transformer", "v1.0"]
    )
    
    # Version model
    model_version = vc.version_model(
        model_name="transformer_model",
        model_path="models/transformer.pt",
        metadata={
            "architecture": "transformer",
            "parameters": 110000000,
            "training_epochs": 10
        },
        description="Trained transformer model",
        tags=["transformer", "trained"]
    )
    
    # Version experiment
    experiment_version = vc.version_experiment(
        experiment_id="transformer_training",
        experiment_data={
            "config": config_data,
            "metadata": {"start_time": "2023-12-01T10:00:00"}
        },
        results={
            "metrics": {"final_loss": 0.25, "final_accuracy": 0.92},
            "plots": {"loss_curve": "plots/loss.png"}
        },
        description="Transformer training experiment",
        tags=["transformer", "experiment"]
    )
```

### 2. Git Integration

```python
from version_control import GitManager

# Initialize Git manager
git_manager = GitManager("my_project")

# Get repository status
status = git_manager.get_status()
print(f"Current branch: {status['branch']}")
print(f"Commit hash: {status['commit_hash']}")
print(f"Is dirty: {status['is_dirty']}")

# Commit changes
commit_hash = git_manager.commit_changes(
    message="Update transformer configuration",
    files=["configs/transformer.yaml"]
)

# Create tag
tag_name = git_manager.create_tag(
    tag_name="v1.0.0",
    message="First stable release"
)

# Get commit history
history = git_manager.get_commit_history(max_count=10)
for commit in history:
    print(f"{commit['hash']}: {commit['message']}")
```

### 3. Configuration Versioning

```python
from version_control import ConfigurationVersioner

# Initialize configuration versioner
config_versioner = ConfigurationVersioner("configs")

# Version configuration
version_id = config_versioner.version_config(
    config_name="transformer_config",
    config_data={
        "model": {"type": "transformer", "layers": 12},
        "training": {"epochs": 10, "lr": 2e-5}
    },
    description="Initial configuration",
    tags=["transformer", "v1.0"]
)

# Get specific version
config_data = config_versioner.get_config_version("transformer_config", version_id)

# Get latest version
latest_config = config_versioner.get_latest_config("transformer_config")

# List all versions
versions = config_versioner.list_config_versions("transformer_config")
for version in versions:
    print(f"Version {version['version_id']}: {version['description']}")

# Rollback to previous version
success = config_versioner.rollback_config("transformer_config", version_id)
```

### 4. Model Versioning

```python
from version_control import ModelVersioner

# Initialize model versioner
model_versioner = ModelVersioner("models")

# Version model
version_id = model_versioner.version_model(
    model_name="transformer_model",
    model_path="models/transformer.pt",
    metadata={
        "architecture": "transformer",
        "parameters": 110000000,
        "training_epochs": 10,
        "performance": {"accuracy": 0.92, "loss": 0.25}
    },
    description="Trained transformer model",
    tags=["transformer", "trained"]
)

# Get model version
model_data = model_versioner.get_model_version("transformer_model", version_id)
print(f"Model path: {model_data['model_path']}")
print(f"Architecture: {model_data['metadata']['architecture']}")

# List model versions
versions = model_versioner.list_model_versions("transformer_model")
for version in versions:
    print(f"Version {version['version_id']}: {version['description']}")
```

### 5. Experiment Versioning

```python
from version_control import ExperimentVersioner

# Initialize experiment versioner
experiment_versioner = ExperimentVersioner("experiments")

# Version experiment
version_id = experiment_versioner.version_experiment(
    experiment_id="transformer_training",
    experiment_data={
        "config": config_data,
        "metadata": {"start_time": "2023-12-01T10:00:00"}
    },
    results={
        "metrics": {"final_loss": 0.25, "final_accuracy": 0.92},
        "plots": {"loss_curve": "plots/loss.png"}
    },
    description="Transformer training experiment",
    tags=["transformer", "experiment"]
)

# Get experiment version
experiment_data = experiment_versioner.get_experiment_version("transformer_training", version_id)
print(f"Experiment: {experiment_data['experiment']['experiment_name']}")
print(f"Final accuracy: {experiment_data['results']['metrics']['final_accuracy']}")

# List experiment versions
versions = experiment_versioner.list_experiment_versions("transformer_training")
for version in versions:
    print(f"Version {version['version_id']}: {version['description']}")
```

## Integration with Experiment Tracking

### Versioned Experiment Tracker

```python
from version_control import VersionedExperimentTracker

# Initialize versioned experiment tracker
config = {
    "experiment_name": "transformer_training",
    "tracking": {"use_tensorboard": True},
    "checkpoint_dir": "checkpoints"
}

tracker = VersionedExperimentTracker(
    experiment_name="transformer_training",
    config=config,
    project_root="my_project",
    auto_commit=True
)

# Training loop with versioning
for epoch in range(10):
    for step, batch in enumerate(train_loader):
        loss = train_step(model, optimizer, batch)
        tracker.log_metrics({"loss": loss}, step=step)
        
        # Save checkpoint with versioning
        if step % 100 == 0:
            tracker.save_checkpoint(
                model=model,
                epoch=epoch,
                step=step,
                optimizer=optimizer,
                train_loss=loss,
                val_loss=validate_model(model, val_loader)
            )

# Finish experiment with versioning
tracker.finish("completed")
```

## Snapshot Management

### Creating Snapshots

```python
# Create project snapshot
snapshot_name = vc.create_snapshot(
    snapshot_name="v1.0.0",
    description="First stable release with transformer model"
)

# Create development snapshot
dev_snapshot = vc.create_snapshot(
    snapshot_name="dev-2023-12-01",
    description="Development snapshot for feature testing"
)
```

### Rollback to Snapshots

```python
# Rollback to specific snapshot
success = vc.rollback_to_snapshot("v1.0.0")
if success:
    print("Successfully rolled back to v1.0.0")
else:
    print("Failed to rollback")
```

## Project Status and Monitoring

### Get Project Status

```python
# Get comprehensive project status
status = vc.get_project_status()

print(f"Project: {status['project_info']['name']}")
print(f"Git branch: {status['git_status']['branch']}")
print(f"Git commit: {status['git_status']['commit_hash']}")
print(f"Config versions: {status['version_counts']['configs']}")
print(f"Model versions: {status['version_counts']['models']}")
print(f"Experiment versions: {status['version_counts']['experiments']}")

# Recent versions
print("\nRecent versions:")
for version in status['recent_versions']:
    print(f"- {version['item_type']}: {version['item_name']} ({version['version_id']})")
```

## Best Practices

### 1. Version Naming Conventions

```python
# Use descriptive version names
config_version = vc.version_configuration(
    config_name="transformer_classification",
    config_data=config,
    description="Transformer model for text classification",
    tags=["transformer", "classification", "v1.0"]
)

# Use semantic versioning for releases
model_version = vc.version_model(
    model_name="bert_classifier",
    model_path="models/bert_classifier.pt",
    metadata=metadata,
    description="BERT classifier for sentiment analysis",
    tags=["bert", "classifier", "v1.0.0"]
)
```

### 2. Commit Message Standards

```python
# Use structured commit messages
commit_messages = {
    "config": "[CONFIG] Update transformer hyperparameters",
    "model": "[MODEL] Add new BERT classifier model",
    "experiment": "[EXPERIMENT] Complete sentiment analysis training",
    "fix": "[FIX] Resolve data loading issue",
    "docs": "[DOCS] Update README with new features"
}
```

### 3. Tag Management

```python
# Use consistent tag naming
tag_patterns = {
    "release": "v{major}.{minor}.{patch}",
    "config": "config-{name}-{version}",
    "model": "model-{name}-{version}",
    "experiment": "experiment-{id}-{version}",
    "snapshot": "snapshot-{name}"
}
```

### 4. Branch Strategy

```python
# Use feature branches for development
git_manager.checkout_branch("feature/transformer-improvements", create=True)

# Work on feature
# ... make changes ...

# Merge back to main
git_manager.checkout_branch("main")
git_manager.repo.git.merge("feature/transformer-improvements")
```

## Error Handling

### 1. Git Operation Failures

```python
try:
    commit_hash = git_manager.commit_changes("Update configuration")
except GitCommandError as e:
    print(f"Git commit failed: {e}")
    # Handle gracefully - maybe save to temporary location
except Exception as e:
    print(f"Unexpected error: {e}")
    # Log error and continue
```

### 2. File System Issues

```python
try:
    version_id = vc.version_model(
        model_name="transformer",
        model_path="models/transformer.pt",
        metadata=metadata
    )
except FileNotFoundError:
    print("Model file not found")
    # Handle missing file
except PermissionError:
    print("Permission denied")
    # Handle permission issues
```

### 3. Version Conflicts

```python
# Check for version conflicts
existing_versions = vc.list_versions(item_name="transformer_config")
if len(existing_versions) > 0:
    latest_version = existing_versions[0]
    print(f"Latest version: {latest_version['version_id']}")
    
    # Compare with current configuration
    current_config = load_current_config()
    latest_config = vc.get_version_info("config", "transformer_config", latest_version['version_id'])
    
    if configs_are_identical(current_config, latest_config['data']):
        print("Configuration unchanged, skipping versioning")
    else:
        # Version the changes
        vc.version_configuration("transformer_config", current_config, "Configuration updated")
```

## Performance Optimization

### 1. Efficient Versioning

```python
# Use compression for large files
vc_system = VersionControlSystem(
    project_root="my_project",
    auto_commit=True,
    compression="gzip"
)

# Batch versioning operations
configs_to_version = [
    ("config1", config1_data),
    ("config2", config2_data),
    ("config3", config3_data)
]

for config_name, config_data in configs_to_version:
    vc.version_configuration(config_name, config_data, f"Version {config_name}")
```

### 2. Cleanup Old Versions

```python
# Automatic cleanup of old versions
config_versioner = ConfigurationVersioner(
    config_dir="configs",
    max_versions=10  # Keep only 10 versions per config
)

# Manual cleanup
old_versions = config_versioner.list_config_versions("old_config")
if len(old_versions) > 20:
    # Remove oldest versions
    for version in old_versions[20:]:
        config_versioner.delete_version("old_config", version['version_id'])
```

### 3. Parallel Versioning

```python
import concurrent.futures

def version_config_parallel(config_name, config_data):
    return vc.version_configuration(config_name, config_data, f"Version {config_name}")

# Version multiple configurations in parallel
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    futures = [
        executor.submit(version_config_parallel, name, data)
        for name, data in configs_to_version
    ]
    
    version_ids = [future.result() for future in futures]
```

## Security Considerations

### 1. Sensitive Data Handling

```python
# Encrypt sensitive configurations
from cryptography.fernet import Fernet

def encrypt_config(config_data, key):
    f = Fernet(key)
    encrypted_data = f.encrypt(json.dumps(config_data).encode())
    return encrypted_data

def decrypt_config(encrypted_data, key):
    f = Fernet(key)
    decrypted_data = f.decrypt(encrypted_data)
    return json.loads(decrypted_data.decode())

# Version encrypted configuration
key = Fernet.generate_key()
encrypted_config = encrypt_config(sensitive_config, key)

vc.version_configuration(
    config_name="sensitive_config",
    config_data={"encrypted_data": encrypted_config},
    description="Encrypted sensitive configuration"
)
```

### 2. Access Control

```python
# Implement access control for versioning
class SecureVersionControlSystem(VersionControlSystem):
    def __init__(self, project_root, authorized_users=None):
        super().__init__(project_root)
        self.authorized_users = authorized_users or []
    
    def check_authorization(self, user):
        if user not in self.authorized_users:
            raise PermissionError(f"User {user} not authorized")
    
    def version_configuration(self, config_name, config_data, **kwargs):
        current_user = os.getenv("USER")
        self.check_authorization(current_user)
        return super().version_configuration(config_name, config_data, **kwargs)
```

## Monitoring and Logging

### 1. Version Control Logging

```python
import structlog

logger = structlog.get_logger()

# Log versioning operations
logger.info("Configuration versioned",
           config_name="transformer_config",
           version_id=version_id,
           user=os.getenv("USER"),
           timestamp=datetime.now().isoformat())

# Log errors
try:
    vc.version_model("transformer", "model.pt", metadata)
except Exception as e:
    logger.error("Model versioning failed",
                model_name="transformer",
                error=str(e),
                user=os.getenv("USER"))
```

### 2. Version Analytics

```python
def analyze_version_history(vc_system):
    """Analyze version history for insights."""
    all_versions = vc_system.list_versions()
    
    # Version frequency by type
    type_counts = {}
    for version in all_versions:
        v_type = version['item_type']
        type_counts[v_type] = type_counts.get(v_type, 0) + 1
    
    # Version frequency by user
    user_counts = {}
    for version in all_versions:
        user = version['author']
        user_counts[user] = user_counts.get(user, 0) + 1
    
    # Version frequency over time
    time_series = {}
    for version in all_versions:
        date = version['timestamp'][:10]  # YYYY-MM-DD
        time_series[date] = time_series.get(date, 0) + 1
    
    return {
        "type_counts": type_counts,
        "user_counts": user_counts,
        "time_series": time_series
    }
```

## Conclusion

The version control system provides comprehensive versioning capabilities for deep learning projects, ensuring reproducibility, change tracking, and collaboration. Key benefits include:

- **Complete Reproducibility**: Every configuration, model, and experiment is versioned
- **Change Tracking**: Detailed diff information and change history
- **Git Integration**: Seamless integration with Git for code versioning
- **Performance Monitoring**: Track performance across versions
- **Collaboration**: Support for team-based development
- **Security**: Secure handling of sensitive data
- **Scalability**: Efficient handling of large files and many versions

This system is essential for production-grade deep learning projects requiring robust versioning and reproducibility. 