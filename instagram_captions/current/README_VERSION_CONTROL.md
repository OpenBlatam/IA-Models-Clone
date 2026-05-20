# Version Control System

## Overview

The Version Control System implements **Key Convention 5: "Use version control (e.g., git) for tracking changes in code and configurations"** from the NLP system requirements. This system provides comprehensive git-based version control for tracking changes in code, configurations, and experiments.

## Key Features

### 🔧 Git Repository Management
- **Automatic Repository Initialization**: Automatically initializes git repository if none exists
- **Git Configuration**: Sets up user credentials, default branch, and .gitignore patterns
- **Branch Management**: Create, checkout, and manage feature and experiment branches
- **Status Tracking**: Monitor staged, unstaged, and tracked files

### 📁 Configuration Version Tracking
- **Configuration Backups**: Automatic backup creation with timestamps
- **Change History**: Track all configuration file modifications
- **Version Comparison**: Compare different versions of configuration files
- **Metadata Tracking**: Store configuration change descriptions and timestamps

### 💻 Code Version Tracking
- **Multi-language Support**: Track changes in Python, TypeScript, JavaScript, C++, and more
- **Feature Branches**: Create dedicated branches for feature development
- **Change Descriptions**: Associate meaningful descriptions with code changes
- **File History**: Complete version history for individual files

### 🧪 Experiment Version Tracking
- **Experiment Branches**: Dedicated branches for different experiments
- **Results Tracking**: Track experiment results and metadata
- **Experiment Metadata**: Store experiment descriptions, creation dates, and status
- **Results Versioning**: Version control for experiment outputs

### 🏷️ Release Management
- **Release Tags**: Create semantic version tags for releases
- **Commit History**: Track all commits with metadata
- **Change Logging**: Comprehensive logging of all repository changes

## System Architecture

```
VersionControlSystem
├── GitRepositoryManager      # Core git operations
├── ConfigurationVersionTracker # Configuration file tracking
├── CodeVersionTracker        # Code file tracking
└── ExperimentVersionTracker  # Experiment tracking
```

### Core Components

#### GitRepositoryManager
- Repository initialization and configuration
- File staging and committing
- Branch creation and management
- Status monitoring and commit history

#### ConfigurationVersionTracker
- Configuration file change tracking
- Automatic backup creation
- Version comparison and history
- Metadata management

#### CodeVersionTracker
- Multi-language code file tracking
- Feature branch management
- Change description association
- File version history

#### ExperimentVersionTracker
- Experiment branch creation
- Results tracking and versioning
- Experiment metadata management
- Experiment lifecycle tracking

## Installation

### Prerequisites
1. **Git Installation**: Ensure git is installed on your system
   - Windows: Install Git for Windows
   - Linux: `sudo apt-get install git` (Ubuntu/Debian)
   - macOS: `brew install git` (Homebrew)

2. **Python Dependencies**: Install required Python packages
   ```bash
   pip install -r requirements_version_control.txt
   ```

### Setup
```python
from version_control_system import VersionControlSystem, GitConfig

# Create git configuration
git_config = GitConfig(
    repo_path=".",
    author_name="Your Name",
    author_email="your.email@example.com"
)

# Initialize version control system
vcs = VersionControlSystem(git_config)
```

## Usage Examples

### Basic Repository Operations

```python
# Get repository status
status = vcs.get_repository_info()
print(f"Current branch: {status['current_branch']}")
print(f"Staged files: {status['total_staged']}")

# Track all changes
vcs.track_changes("Update NLP model architecture", "feature")

# Create release tag
vcs.create_release_tag("v1.0.0", "Initial release")
```

### Configuration Tracking

```python
# Track configuration changes
success = vcs.config_tracker.track_config_changes(
    "config/model_config.yaml",
    "Update learning rate and batch size"
)

# Get configuration history
history = vcs.config_tracker.get_config_history("config/model_config.yaml")
for change in history:
    print(f"{change['date']}: {change['message']}")

# Compare configuration versions
diff = vcs.config_tracker.compare_config_versions(
    "config/model_config.yaml",
    "abc1234",  # commit hash 1
    "def5678"   # commit hash 2
)
print(diff)
```

### Code Tracking

```python
# Track specific code files
vcs.code_tracker.track_code_changes(
    ["models/transformer.py", "training/trainer.py"],
    "Implement attention mechanism improvements"
)

# Create feature branch
vcs.code_tracker.create_feature_branch(
    "attention-optimization",
    "Optimize attention computation for large models"
)

# Get file history
history = vcs.code_tracker.get_file_history("models/transformer.py")
```

### Experiment Tracking

```python
# Create experiment branch
vcs.experiment_tracker.create_experiment_branch(
    "bert-fine-tuning",
    "Fine-tune BERT model on domain-specific data"
)

# Track experiment results
vcs.experiment_tracker.track_experiment_results(
    "bert-fine-tuning",
    "results/bert_results.json",
    "Training completed with 95% accuracy"
)
```

## Configuration

### GitConfig Options

```python
@dataclass
class GitConfig:
    repo_path: str = "."                    # Repository path
    author_name: str = "NLP System"         # Git author name
    author_email: str = "nlp@system.com"    # Git author email
    default_branch: str = "main"            # Default branch name
    commit_message_template: str = "[{type}] {description}"  # Commit message format
    ignore_patterns: List[str] = [...]      # .gitignore patterns
```

### Custom .gitignore Patterns

The system automatically creates a `.gitignore` file with common patterns:

```
*.pyc, *.pyo, __pycache__/
*.log, *.tmp
*.pth, *.ckpt, *.pt, *.bin, *.h5
data/, cache/, outputs/, logs/, runs/
.ipynb_checkpoints/, .DS_Store
*.swp, *.swo
```

## Best Practices

### 1. Commit Message Convention
Use the built-in commit message template:
- `[config] Update model parameters`
- `[code] Implement new attention mechanism`
- `[experiment] Add fine-tuning results`
- `[feature] Start optimization implementation`

### 2. Branch Naming
- **Feature branches**: `feature/feature-name`
- **Experiment branches**: `experiment/experiment-name`
- **Bug fixes**: `fix/issue-description`

### 3. Configuration Tracking
- Always provide meaningful descriptions for configuration changes
- Use the configuration tracker for all config file modifications
- Review configuration history before making changes

### 4. Experiment Management
- Create dedicated branches for each experiment
- Track all experiment results and metadata
- Use descriptive experiment names and descriptions

## Advanced Features

### Custom Git Operations

```python
# Access underlying git repository manager
repo_manager = vcs.repo_manager

# Create custom branch
repo_manager.create_branch("custom-branch")

# Checkout specific branch
repo_manager.checkout_branch("main")

# Get detailed commit history
commits = repo_manager.get_commit_history(max_count=20)
```

### Integration with Other Systems

The version control system integrates seamlessly with:
- **Configuration Management System**: Track all configuration changes
- **Modular Code Structure**: Version control for all code modules
- **Experiment Tracking**: Git-based experiment versioning
- **Model Checkpointing**: Version control for model files

## Error Handling

The system includes comprehensive error handling:
- **Git Command Failures**: Graceful handling of git command errors
- **File Not Found**: Validation of file existence before operations
- **Permission Issues**: Clear error messages for permission problems
- **Network Issues**: Handling of remote repository connection problems

## Performance Considerations

- **Lazy Loading**: Git operations are performed only when needed
- **Efficient File Scanning**: Optimized file discovery for tracking
- **Minimal Git Calls**: Reduced number of git subprocess calls
- **Caching**: Repository status caching for improved performance

## Troubleshooting

### Common Issues

1. **Git Not Installed**
   ```
   Error: git command not found
   Solution: Install git for your operating system
   ```

2. **Permission Denied**
   ```
   Error: Permission denied for .git directory
   Solution: Check file permissions and ownership
   ```

3. **Repository Already Exists**
   ```
   Error: Repository already exists
   Solution: System automatically detects and uses existing repository
   ```

### Debug Mode

Enable debug logging for troubleshooting:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Initialize system with debug logging
vcs = VersionControlSystem(git_config)
```

## Future Enhancements

### Planned Features
- **Remote Repository Integration**: GitHub, GitLab, Bitbucket support
- **Automated Merging**: Intelligent conflict resolution
- **Branch Protection**: Prevent accidental changes to main branches
- **Web Interface**: Web-based repository management
- **CI/CD Integration**: Automated testing and deployment

### Extension Points
- **Custom Git Hooks**: Pre-commit and post-commit hooks
- **Plugin System**: Extensible version control features
- **API Integration**: REST API for external tools
- **Metrics Dashboard**: Repository analytics and insights

## Contributing

### Development Setup
1. Clone the repository
2. Install development dependencies
3. Create feature branch
4. Make changes and commit
5. Submit pull request

### Code Style
- Follow PEP 8 for Python code
- Use descriptive variable names
- Include comprehensive docstrings
- Add type hints for all functions

## License

This version control system is part of the NLP System and follows the same licensing terms.

## Conclusion

The Version Control System provides a robust, git-based solution for tracking changes in code and configurations. It implements all requirements from Key Convention 5 and integrates seamlessly with the broader NLP system architecture.

Key benefits:
- **Automated Git Management**: No manual git setup required
- **Comprehensive Tracking**: Track all types of changes
- **Experiment Versioning**: Dedicated experiment management
- **Best Practices**: Built-in conventions and patterns
- **Extensible Architecture**: Easy to extend and customize

For questions and support, refer to the main NLP system documentation.


