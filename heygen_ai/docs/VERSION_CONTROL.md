# Version Control Guide for HeyGen AI

This document outlines the version control practices and workflows for the HeyGen AI equivalent system using Git.

## Table of Contents

1. [Git Setup](#git-setup)
2. [Branching Strategy](#branching-strategy)
3. [Commit Message Convention](#commit-message-convention)
4. [Workflow Guidelines](#workflow-guidelines)
5. [Code Review Process](#code-review-process)
6. [Release Management](#release-management)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)

## Git Setup

### Initial Setup

Use the automated setup script to initialize the repository:

```bash
cd agents/backend/onyx/server/features/heygen_ai
python scripts/setup_git.py --name "Your Name" --email "your.email@example.com"
```

### Manual Setup

If you prefer manual setup:

```bash
# Initialize repository
git init

# Configure user
git config user.name "Your Name"
git config user.email "your.email@example.com"

# Add remote (if applicable)
git remote add origin <repository-url>

# Create initial commit
git add .
git commit -m "feat: initial commit - HeyGen AI equivalent system"
```

### Git Configuration

The setup script configures the following git settings:

```bash
# Core settings
git config core.autocrlf input
git config core.filemode false
git config core.ignorecase false

# Branch settings
git config init.defaultBranch main
git config branch.autosetupmerge true
git config branch.autosetuprebase always

# Push settings
git config push.default simple

# Color settings
git config color.ui auto
git config color.branch auto
git config color.diff auto
git config color.status auto

# Useful aliases
git config alias.st status
git config alias.co checkout
git config alias.br branch
git config alias.ci commit
git config alias.lg "log --oneline --graph --decorate"
```

## Branching Strategy

We follow a **Git Flow** branching strategy with the following branches:

### Main Branches

- **`main`**: Production-ready code
- **`develop`**: Integration branch for features

### Supporting Branches

- **`feature/*`**: New features
- **`hotfix/*`**: Critical production fixes
- **`release/*`**: Release preparation
- **`bugfix/*`**: Bug fixes

### Branch Naming Convention

```
feature/transformer-models
feature/diffusion-models
feature/training-system
feature/gradio-interfaces
feature/performance-optimization
feature/error-handling
feature/experiment-tracking
feature/api-endpoints
feature/documentation
feature/testing
feature/deployment
hotfix/critical-fixes
release/v1.0.0
```

### Branch Workflow

```bash
# Start a new feature
git checkout develop
git pull origin develop
git checkout -b feature/new-feature

# Work on feature
# ... make changes ...
git add .
git commit -m "feat(new-feature): implement core functionality"

# Finish feature
git checkout develop
git pull origin develop
git checkout feature/new-feature
git rebase develop
git checkout develop
git merge feature/new-feature
git push origin develop
git branch -d feature/new-feature
```

## Commit Message Convention

We follow the **Conventional Commits** specification:

### Format

```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

### Types

- **`feat`**: New features
- **`fix`**: Bug fixes
- **`docs`**: Documentation changes
- **`style`**: Code style changes (formatting, etc.)
- **`refactor`**: Code refactoring
- **`test`**: Adding or updating tests
- **`chore`**: Maintenance tasks
- **`perf`**: Performance improvements
- **`ci`**: CI/CD changes
- **`build`**: Build system changes
- **`revert`**: Reverting previous commits

### Scopes

- **`transformer`**: Transformer model changes
- **`diffusion`**: Diffusion model changes
- **`training`**: Training system changes
- **`api`**: API endpoint changes
- **`ui`**: User interface changes
- **`config`**: Configuration changes
- **`docs`**: Documentation changes
- **`test`**: Test-related changes

### Examples

```bash
# Feature commits
git commit -m "feat(transformer): add attention mechanism implementation"
git commit -m "feat(diffusion): implement StableDiffusionXL pipeline"
git commit -m "feat(training): add experiment tracking with W&B"

# Bug fixes
git commit -m "fix(api): resolve memory leak in video generation"
git commit -m "fix(training): correct gradient clipping implementation"

# Documentation
git commit -m "docs(api): update API documentation with new endpoints"
git commit -m "docs(setup): add installation guide for Windows"

# Refactoring
git commit -m "refactor(core): restructure model architecture for better modularity"
git commit -m "refactor(utils): extract common functions to shared module"

# Performance
git commit -m "perf(training): optimize data loading with prefetching"
git commit -m "perf(inference): implement batch processing for faster generation"

# Tests
git commit -m "test(transformer): add unit tests for attention mechanisms"
git commit -m "test(integration): add end-to-end tests for video generation"
```

## Workflow Guidelines

### Daily Workflow

1. **Start of day**:
   ```bash
   git checkout develop
   git pull origin develop
   ```

2. **Create feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Work on feature**:
   ```bash
   # Make changes
   git add .
   git commit -m "feat(scope): description"
   ```

4. **End of day**:
   ```bash
   git push origin feature/your-feature-name
   ```

### Feature Completion

1. **Update develop branch**:
   ```bash
   git checkout develop
   git pull origin develop
   ```

2. **Rebase feature branch**:
   ```bash
   git checkout feature/your-feature-name
   git rebase develop
   ```

3. **Merge to develop**:
   ```bash
   git checkout develop
   git merge feature/your-feature-name
   git push origin develop
   ```

4. **Clean up**:
   ```bash
   git branch -d feature/your-feature-name
   git push origin --delete feature/your-feature-name
   ```

### Hotfix Workflow

1. **Create hotfix branch**:
   ```bash
   git checkout main
   git pull origin main
   git checkout -b hotfix/critical-fix
   ```

2. **Fix the issue**:
   ```bash
   # Make critical fix
   git add .
   git commit -m "fix(critical): resolve production issue"
   ```

3. **Merge to main and develop**:
   ```bash
   git checkout main
   git merge hotfix/critical-fix
   git tag -a v1.0.1 -m "Release v1.0.1"
   git push origin main --tags
   
   git checkout develop
   git merge hotfix/critical-fix
   git push origin develop
   ```

## Code Review Process

### Pull Request Guidelines

1. **Create PR** from feature branch to develop
2. **Add description** with:
   - What was changed
   - Why it was changed
   - How to test
   - Screenshots (if applicable)

3. **Request review** from team members
4. **Address feedback** and update PR
5. **Merge** after approval

### Review Checklist

- [ ] Code follows project style guidelines
- [ ] Tests are included and passing
- [ ] Documentation is updated
- [ ] No sensitive data is committed
- [ ] Performance impact is considered
- [ ] Error handling is implemented
- [ ] Logging is appropriate

## Release Management

### Release Process

1. **Create release branch**:
   ```bash
   git checkout develop
   git pull origin develop
   git checkout -b release/v1.0.0
   ```

2. **Update version**:
   - Update version in `__init__.py`
   - Update `CHANGELOG.md`
   - Update documentation

3. **Final testing**:
   - Run all tests
   - Test in staging environment
   - Update release notes

4. **Merge to main**:
   ```bash
   git checkout main
   git merge release/v1.0.0
   git tag -a v1.0.0 -m "Release v1.0.0"
   git push origin main --tags
   ```

5. **Merge to develop**:
   ```bash
   git checkout develop
   git merge release/v1.0.0
   git push origin develop
   ```

6. **Clean up**:
   ```bash
   git branch -d release/v1.0.0
   git push origin --delete release/v1.0.0
   ```

### Version Numbering

We follow **Semantic Versioning** (SemVer):

- **MAJOR.MINOR.PATCH**
- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

Examples:
- `v1.0.0`: Initial release
- `v1.1.0`: New features added
- `v1.1.1`: Bug fixes
- `v2.0.0`: Breaking changes

## Best Practices

### General Guidelines

1. **Commit frequently**: Small, focused commits
2. **Write clear messages**: Descriptive commit messages
3. **Pull before push**: Always sync with remote
4. **Use branches**: Never work directly on main/develop
5. **Review your changes**: `git diff` before committing

### File Management

1. **Use .gitignore**: Keep repository clean
2. **Don't commit large files**: Use Git LFS for large files
3. **Don't commit secrets**: Use environment variables
4. **Don't commit generated files**: Add to .gitignore

### Code Quality

1. **Run tests**: Before committing
2. **Check formatting**: Use black, flake8
3. **Type checking**: Use mypy
4. **Documentation**: Update docs with changes

### Security

1. **No secrets in code**: Use environment variables
2. **Review sensitive changes**: Extra scrutiny for security-related code
3. **Use signed commits**: For critical changes
4. **Regular audits**: Review access and permissions

## Troubleshooting

### Common Issues

#### Merge Conflicts

```bash
# During merge conflict
git status  # See conflicted files
# Edit conflicted files
git add .   # Mark as resolved
git commit  # Complete merge
```

#### Rebase Conflicts

```bash
# During rebase conflict
git status  # See conflicted files
# Edit conflicted files
git add .   # Mark as resolved
git rebase --continue
```

#### Undo Last Commit

```bash
# Undo last commit (keep changes)
git reset --soft HEAD~1

# Undo last commit (discard changes)
git reset --hard HEAD~1
```

#### Revert Changes

```bash
# Revert specific commit
git revert <commit-hash>

# Revert last commit
git revert HEAD
```

#### Stash Changes

```bash
# Stash current changes
git stash

# List stashes
git stash list

# Apply last stash
git stash pop

# Apply specific stash
git stash apply stash@{n}
```

### Useful Commands

```bash
# View commit history
git log --oneline --graph --decorate

# View file history
git log --follow filename

# View changes in commit
git show <commit-hash>

# View current status
git status

# View branch information
git branch -a

# View remote information
git remote -v

# Clean up local branches
git branch --merged | grep -v "\*" | xargs -n 1 git branch -d

# View git configuration
git config --list
```

### Git Hooks

The project includes pre-commit hooks that:

1. **Check file sizes**: Prevent large files
2. **Check for secrets**: Prevent sensitive data
3. **Run linting**: flake8, black, mypy
4. **Validate commit messages**: Conventional commits format

To bypass hooks (emergency only):
```bash
git commit --no-verify -m "emergency: critical fix"
```

## Resources

- [Conventional Commits](https://www.conventionalcommits.org/)
- [Git Flow](https://nvie.com/posts/a-successful-git-branching-model/)
- [Semantic Versioning](https://semver.org/)
- [Git Best Practices](https://git-scm.com/book/en/v2)
- [Git Hooks](https://git-scm.com/docs/githooks)

## Support

For version control questions or issues:

1. Check this documentation
2. Review git logs and history
3. Consult team members
4. Create issue in project repository

---

*This document is maintained by the HeyGen AI development team.* 