# Version Control Guide for ML Projects

## Overview

This guide covers the comprehensive version control system for ML projects in the Onyx Ads Backend. The system provides:

- **Git Integration**: Automated git operations for code and configuration tracking
- **Configuration Versioning**: Version control for all configuration files
- **Experiment Reproducibility**: Complete experiment reproduction through git commits
- **Branch Management**: Experiment-specific branches for isolation
- **Integration**: Seamless integration with configuration management and experiment tracking

## Table of Contents

1. [Version Control Manager](#version-control-manager)
2. [Integrated Version Control](#integrated-version-control)
3. [Git Operations](#git-operations)
4. [Configuration Versioning](#configuration-versioning)
5. [Experiment Reproducibility](#experiment-reproducibility)
6. [Best Practices](#best-practices)
7. [Troubleshooting](#troubleshooting)

## Version Control Manager

### Core Features

The `VersionControlManager` provides comprehensive git integration for ML projects:

#### Git Repository Management
```python
from onyx.server.features.ads.version_control_manager import VersionControlManager

# Create version control manager
vc_manager = VersionControlManager("./my_project")

# Check git status
status = vc_manager.get_git_status()
print(f"Git status: {status.value}")

# Get current branch
branch = vc_manager.get_current_branch()
print(f"Current branch: {branch}")

# Get current commit
commit = vc_manager.get_current_commit()
print(f"Current commit: {commit.hash} - {commit.message}")
```

#### File Version Information
```python
# Get version info for a specific file
version_info = vc_manager.get_file_version_info("config_manager.py")
print(f"File: {version_info.file_path}")
print(f"Git hash: {version_info.git_hash}")
print(f"Commit date: {version_info.commit_date}")
print(f"Branch: {version_info.branch}")
print(f"Is dirty: {version_info.is_dirty}")
```

### Experiment Version Control

#### Starting Experiments with Version Control
```python
from onyx.server.features.ads.version_control_manager import ExperimentVersionControl

# Create experiment version control
exp_vc = ExperimentVersionControl("./my_project")

# Start experiment with version control
configs = {
    "model": {"name": "bert_model", "type": "transformer"},
    "training": {"batch_size": 32, "learning_rate": 2e-5}
}

commit_hash = exp_vc.start_experiment(
    experiment_id="exp_001",
    experiment_name="BERT Classification",
    configs=configs,
    create_branch=True
)

print(f"Started experiment with commit: {commit_hash}")
```

#### Committing Experiment Changes
```python
# Commit changes during experiment
commit_hash = exp_vc.commit_experiment_changes(
    experiment_id="exp_001",
    experiment_name="BERT Classification",
    configs=updated_configs,
    message="Updated learning rate and batch size"
)

print(f"Committed changes: {commit_hash}")
```

#### Ending Experiments
```python
# End experiment with final metrics
final_metrics = {"accuracy": 0.95, "loss": 0.05}
final_commit = exp_vc.end_experiment(
    experiment_id="exp_001",
    experiment_name="BERT Classification",
    configs=configs,
    final_metrics=final_metrics,
    tag_name="v1.0"
)

print(f"Ended experiment with commit: {final_commit}")
```

## Integrated Version Control

### Complete Integration

The `IntegratedVersionControl` combines version control, configuration management, and experiment tracking:

#### Starting Integrated Experiments
```python
from onyx.server.features.ads.integrated_version_control import IntegratedVersionControl

# Create integrated version control
integrated_vc = IntegratedVersionControl("./my_project")

# Start integrated experiment
configs = {
    "model": {
        "name": "bert_classifier",
        "type": "transformer",
        "architecture": "bert-base-uncased",
        "input_size": 768,
        "output_size": 10
    },
    "training": {
        "batch_size": 32,
        "learning_rate": 2e-5,
        "epochs": 10
    },
    "experiment": {
        "experiment_name": "bert_classification",
        "tracking_backend": "wandb"
    }
}

commit_hash, tracker = integrated_vc.start_integrated_experiment(
    experiment_id="exp_001",
    experiment_name="BERT Classification",
    configs=configs,
    create_branch=True,
    tracking_backend="wandb"
)

print(f"Started integrated experiment with commit: {commit_hash}")
```

#### Training with Integrated Tracking
```python
# Training loop with integrated version control
for epoch in range(10):
    for batch_idx, (data, target) in enumerate(train_loader):
        # Training step
        loss = train_step(model, data, target, optimizer)
        
        # Log metrics
        tracker.log_metrics({
            "loss": loss.item(),
            "accuracy": accuracy.item()
        }, step=batch_idx, epoch=epoch)
        
        # Commit changes periodically
        if batch_idx % 100 == 0:
            integrated_vc.commit_experiment_changes(
                "exp_001", "BERT Classification", configs,
                f"Training step {batch_idx}", {"loss": loss.item()}
            )
```

#### Ending Integrated Experiments
```python
# End integrated experiment
final_metrics = {"test_accuracy": 0.95, "test_loss": 0.05}
final_commit = integrated_vc.end_integrated_experiment(
    "exp_001", "BERT Classification", configs, final_metrics, "v1.0"
)

print(f"Ended integrated experiment with commit: {final_commit}")
```

### Context Manager Usage
```python
from onyx.server.features.ads.integrated_version_control import integrated_experiment_context

# Use context manager for automatic cleanup
with integrated_experiment_context("exp_002", "Test Experiment", configs) as (vc, tracker, commit_hash):
    # Your training code here
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            loss = train_step(model, data, target, optimizer)
            tracker.log_metrics({"loss": loss.item()}, step=batch_idx)
    
    # Context manager handles cleanup automatically
```

## Git Operations

### Branch Management

#### Creating Experiment Branches
```python
# Create a new branch for an experiment
branch_name = vc_manager.create_experiment_branch("exp_001", base_branch="main")
print(f"Created branch: {branch_name}")

# Switch to a specific branch
success = vc_manager.switch_to_branch("experiment/exp_001")
if success:
    print("Successfully switched to experiment branch")
```

#### Branch Operations
```python
# Get current branch
current_branch = vc_manager.get_current_branch()

# List all branches
import subprocess
result = subprocess.run(["git", "branch"], capture_output=True, text=True)
branches = result.stdout.splitlines()

# Merge experiment branch back to main
vc_manager.switch_to_branch("main")
subprocess.run(["git", "merge", "experiment/exp_001"])
```

### Commit Management

#### Automatic Commits
```python
# Commit experiment changes
commit_hash = vc_manager.commit_experiment(
    experiment_id="exp_001",
    experiment_name="BERT Classification",
    configs=configs,
    message="Updated hyperparameters"
)

print(f"Committed experiment: {commit_hash}")
```

#### Manual Commits
```python
# Add all changes
subprocess.run(["git", "add", "."], cwd="./my_project")

# Commit with message
subprocess.run(
    ["git", "commit", "-m", "Manual commit: Updated model architecture"],
    cwd="./my_project"
)
```

### Tagging Experiments
```python
# Create a tag for an experiment
success = vc_manager.tag_experiment(
    experiment_id="exp_001",
    tag_name="v1.0",
    message="First successful version"
)

if success:
    print("Created experiment tag")

# Get all tags for an experiment
tags = vc_manager.get_experiment_tags("exp_001")
print(f"Experiment tags: {tags}")
```

## Configuration Versioning

### Configuration Snapshots
```python
# Create a snapshot of current configurations
snapshot_file = vc_manager.create_config_snapshot(configs)
print(f"Created config snapshot: {snapshot_file}")

# Restore configurations from snapshot
success = vc_manager.restore_config_snapshot(snapshot_file)
if success:
    print("Restored configurations from snapshot")
```

### Configuration Changes Tracking
```python
# Get changes made to a configuration file
changes = vc_manager.get_config_changes("configs/model_config.yaml", since_commit="HEAD~1")

for change in changes:
    print(f"Change type: {change['type']}")
    if 'additions' in change:
        print(f"Additions: {change['additions']}")
    if 'deletions' in change:
        print(f"Deletions: {change['deletions']}")
```

### Configuration History
```python
# Get commit history for configurations
config_commits = vc_manager.get_experiment_history("exp_001")

for commit in config_commits:
    print(f"Commit: {commit.hash} - {commit.message}")
    print(f"Date: {commit.date}")
```

## Experiment Reproducibility

### Reproducing Experiments
```python
# Reproduce an experiment from its exact version
success = vc_manager.reproduce_experiment("exp_001")
if success:
    print("Successfully reproduced experiment")
else:
    print("Failed to reproduce experiment")
```

### Integrated Experiment Reproduction
```python
# Reproduce with integrated tracking
tracker = integrated_vc.reproduce_integrated_experiment("exp_001")
if tracker:
    print("Successfully reproduced integrated experiment")
    
    # Continue training from the reproduced state
    for epoch in range(5):  # Continue for 5 more epochs
        for batch_idx, (data, target) in enumerate(train_loader):
            loss = train_step(model, data, target, optimizer)
            tracker.log_metrics({"loss": loss.item()}, step=batch_idx)
```

### Experiment Snapshots
```python
# Create a complete snapshot of an experiment
snapshot_file = integrated_vc.create_experiment_snapshot("exp_001")
print(f"Created experiment snapshot: {snapshot_file}")

# The snapshot includes:
# - Experiment configuration
# - Git commit information
# - Final metrics
# - Reproduction script
```

## Experiment Comparison

### Comparing Multiple Experiments
```python
# Compare multiple experiments
comparison = integrated_vc.compare_experiments(["exp_001", "exp_002", "exp_003"])

# View configuration differences
config_diffs = comparison["differences"]["configurations"]
for config_type, differences in config_diffs.items():
    print(f"Configuration differences in {config_type}:")
    for key, values in differences.items():
        print(f"  {key}: {values}")

# View metric differences
metric_diffs = comparison["differences"]["metrics"]
for metric, info in metric_diffs.items():
    print(f"{metric}: min={info['min']}, max={info['max']}, range={info['range']}")

# View summary
summary = comparison["summary"]
print(f"Total experiments: {summary['total_experiments']}")
print(f"Best experiment: {summary['best_experiment']['experiment_id']}")
```

### Getting Experiment Information
```python
# Get comprehensive experiment information
info = integrated_vc.get_integrated_experiment_info("exp_001")

if info:
    print(f"Experiment: {info.experiment_name}")
    print(f"Git hash: {info.git_hash}")
    print(f"Branch: {info.branch}")
    print(f"Reproducible: {info.is_reproducible}")
    print(f"Final metrics: {info.final_metrics}")
    print(f"Tags: {info.tags}")
```

## Git Configuration

### .gitignore for ML Projects

The system automatically creates a comprehensive `.gitignore` file:

```gitignore
# ML Project .gitignore

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
env/
ENV/
.venv/
.env/

# Jupyter Notebook
.ipynb_checkpoints

# PyTorch
*.pth
*.pt
*.ckpt

# TensorFlow
*.h5
*.hdf5
*.pb
*.pkl

# Checkpoints and models
checkpoints/
models/
saved_models/
*.model

# Data
data/
datasets/
*.csv
*.json
*.parquet
*.h5
*.hdf5

# Logs
logs/
*.log
tensorboard_logs/
wandb/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Temporary files
tmp/
temp/
*.tmp

# Environment variables
.env
.env.local
.env.production

# Config files with secrets
config/secrets.yaml
config/local.yaml

# Large files
*.zip
*.tar.gz
*.rar

# Profiling outputs
profiles/
*.prof
```

### Git Configuration
```bash
# Set up git configuration
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# Configure git for ML projects
git config --global core.autocrlf input
git config --global core.safecrlf warn
git config --global core.ignorecase false
```

## Best Practices

### Version Control Best Practices

1. **Regular Commits**: Commit changes frequently during experiments
2. **Meaningful Messages**: Use descriptive commit messages
3. **Branch Isolation**: Use separate branches for different experiments
4. **Tag Important Versions**: Tag successful experiments with version numbers
5. **Configuration Tracking**: Always track configuration changes

### Experiment Organization

1. **Consistent Naming**: Use consistent naming conventions for experiments
2. **Branch Naming**: Use descriptive branch names (e.g., `experiment/bert_classification_v1`)
3. **Tag Naming**: Use semantic versioning for tags (e.g., `v1.0.0`, `v1.1.0`)
4. **Documentation**: Document experiment purpose and results in commit messages

### Reproducibility

1. **Complete Snapshots**: Create complete snapshots for important experiments
2. **Dependency Tracking**: Track all dependencies and versions
3. **Environment Consistency**: Ensure consistent environments across reproductions
4. **Documentation**: Document reproduction steps and requirements

### Integration Best Practices

1. **Automatic Commits**: Use automatic commits for experiment changes
2. **Integrated Tracking**: Use integrated version control for complex experiments
3. **Regular Backups**: Regularly backup experiment data and configurations
4. **Version Compatibility**: Ensure compatibility between different versions

## Troubleshooting

### Common Issues

#### Git Repository Not Found
```python
# Problem: Git repository not initialized
try:
    vc_manager = VersionControlManager("./my_project")
except Exception as e:
    print(f"Failed to initialize git repository: {e}")
    # The system will automatically initialize a new repository
```

#### Branch Conflicts
```python
# Problem: Branch conflicts during merge
try:
    vc_manager.switch_to_branch("main")
    # Resolve conflicts manually
    subprocess.run(["git", "status"])
    # Edit conflicted files
    subprocess.run(["git", "add", "."])
    subprocess.run(["git", "commit", "-m", "Resolve conflicts"])
except Exception as e:
    print(f"Failed to resolve conflicts: {e}")
```

#### Experiment Reproduction Fails
```python
# Problem: Experiment reproduction fails
try:
    tracker = integrated_vc.reproduce_integrated_experiment("exp_001")
    if not tracker:
        print("Failed to reproduce experiment")
        # Check if dependencies are available
        # Check if git commit exists
        # Check if configurations are valid
except Exception as e:
    print(f"Reproduction failed: {e}")
```

### Debugging Tips

1. **Check Git Status**: Always check git status before operations
2. **Verify Dependencies**: Ensure all dependencies are available
3. **Check File Permissions**: Verify file permissions for git operations
4. **Review Logs**: Check logs for detailed error information
5. **Test Reproducibility**: Regularly test experiment reproduction

### Performance Optimization

1. **Selective Commits**: Only commit essential changes
2. **Batch Operations**: Batch multiple changes in single commits
3. **Efficient Branching**: Use lightweight branches for experiments
4. **Regular Cleanup**: Clean up old branches and tags

## Advanced Features

### Custom Git Operations
```python
# Custom git operations
import subprocess

# Stash changes
subprocess.run(["git", "stash"], cwd="./my_project")

# Apply stashed changes
subprocess.run(["git", "stash", "pop"], cwd="./my_project")

# Reset to specific commit
subprocess.run(["git", "reset", "--hard", "commit_hash"], cwd="./my_project")

# Cherry-pick commits
subprocess.run(["git", "cherry-pick", "commit_hash"], cwd="./my_project")
```

### Remote Repository Integration
```python
# Add remote repository
subprocess.run(["git", "remote", "add", "origin", "https://github.com/user/repo.git"])

# Push to remote
subprocess.run(["git", "push", "-u", "origin", "main"])

# Pull from remote
subprocess.run(["git", "pull", "origin", "main"])
```

### Automated Workflows
```python
# Automated experiment workflow
def automated_experiment_workflow(experiment_id, configs):
    # Start experiment
    commit_hash, tracker = integrated_vc.start_integrated_experiment(
        experiment_id, "Automated Experiment", configs
    )
    
    # Training loop
    for epoch in range(10):
        # Training code here
        pass
    
    # End experiment
    final_metrics = {"accuracy": 0.95}
    integrated_vc.end_integrated_experiment(
        experiment_id, "Automated Experiment", configs, final_metrics
    )
    
    # Create snapshot
    integrated_vc.create_experiment_snapshot(experiment_id)
```

This comprehensive guide provides everything needed to effectively use version control in ML projects, ensuring reproducibility, traceability, and collaboration. 