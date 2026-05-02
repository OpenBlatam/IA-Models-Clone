# 🚀 MCP Server for Code Improvement — Blatam Academy

> Part of the [Blatam Academy Integrated Platform](README.md)

## 📋 Description

This MCP (Model Context Protocol) server provides advanced tools to analyze, improve, and refactor the Blatam Academy project code. It includes 16 specialized tools for different aspects of code improvement.

## 🛠️ Available Tools

### 1. **analyze_code_quality**
Analyzes code quality by detecting style issues, cyclomatic complexity, and best practice violations.

**Usage:**
```json
{
  "path": "features/suno_clone_ai/core/",
  "language": "python",
  "checks": ["complexity", "security", "performance"]
}
```

### 2. **detect_code_duplication**
Detects duplicate code in the project, identifying repetitive patterns.

**Usage:**
```json
{
  "minLines": 5,
  "threshold": 0.8,
  "excludePatterns": ["__pycache__", "node_modules"]
}
```

### 3. **suggest_refactoring**
Suggests specific refactorings based on identified patterns.

**Usage:**
```json
{
  "filePath": "features/suno_clone_ai/core/optimizer.py",
  "refactoringType": "extract_method",
  "apply": false
}
```

### 4. **analyze_architecture**
Analyzes project architecture, detecting violations of SOLID principles and Clean Architecture.

**Usage:**
```json
{
  "featurePath": "features/facebook_posts",
  "checkCleanArchitecture": true,
  "checkSOLID": true,
  "checkDependencies": true
}
```

### 5. **optimize_performance**
Identifies performance optimization opportunities.

**Usage:**
```json
{
  "filePath": "features/music_analyzer_ai/api.py",
  "optimizationTypes": ["cache", "async", "database"],
  "profile": true
}
```

### 6. **check_security**
Analyzes code for security vulnerabilities.

**Usage:**
```json
{
  "path": "features/",
  "severity": "medium",
  "checkSecrets": true,
  "checkInjection": true
}
```

### 7. **improve_documentation**
Analyzes and improves code documentation.

**Usage:**
```json
{
  "filePath": "features/dermatology_ai/services.py",
  "addDocstrings": true,
  "improveComments": true,
  "generateREADME": false
}
```

### 8. **analyze_dependencies**
Analyzes project dependencies: versions, conflicts, vulnerabilities.

**Usage:**
```json
{
  "checkVulnerabilities": true,
  "checkUnused": true,
  "checkConflicts": true,
  "suggestUpdates": true
}
```

### 9. **standardize_code_style**
Standardizes code style according to best practices.

**Usage:**
```json
{
  "filePath": "features/blog_posts/generator.py",
  "language": "python",
  "apply": false
}
```

### 10. **detect_anti_patterns**
Detects common anti-patterns in code.

**Usage:**
```json
{
  "path": "features/copywriting/",
  "antiPatterns": ["dead_code", "god_object", "long_method"]
}
```

### 11. **optimize_imports**
Optimizes and organizes imports.

**Usage:**
```json
{
  "filePath": "features/seo/optimizer.py",
  "removeUnused": true,
  "organize": true,
  "apply": false
}
```

### 12. **generate_tests**
Generates unit and integration tests.

**Usage:**
```json
{
  "filePath": "features/ai_video/processor.py",
  "testType": "both",
  "framework": "pytest",
  "coverage": 80
}
```

### 13. **migrate_to_clean_architecture**
Helps migrate existing code to Clean Architecture.

**Usage:**
```json
{
  "featurePath": "features/ads",
  "createStructure": true,
  "migrateCode": true,
  "generateInterfaces": true
}
```

### 14. **analyze_feature_consistency**
Analyzes consistency between features.

**Usage:**
```json
{
  "compareWith": "instagram_captions",
  "features": ["facebook_posts", "blog_posts"]
}
```

### 15. **optimize_config_files**
Optimizes and unifies configuration files.

**Usage:**
```json
{
  "configType": "all",
  "consolidate": false
}
```

### 16. **detect_unified_patterns**
Detects patterns that can be unified (like optimizers).

**Usage:**
```json
{
  "patternType": "optimizers",
  "minOccurrences": 3
}
```

## 📚 Available Resources

The server provides access to the following resources:

- **codebase://best-practices** — Project best practices
- **codebase://architecture-patterns** — Architectural patterns
- **codebase://refactoring-opportunities** — Refactoring opportunities
- **codebase://anti-patterns** — Anti-patterns to avoid
- **codebase://code-quality-standards** — Quality standards

## 🎯 Available Prompts

1. **improve_code_quality** — Improves general code quality
2. **refactor_code** — Refactors code following best practices
3. **optimize_performance** — Optimizes code performance

## 🚀 Configuration

### Installation

1. Ensure Python 3.8+ is installed
2. Install necessary dependencies:

```bash
pip install mcp pylint black isort mypy bandit safety
```

### Cursor Configuration

1. Open MCP settings in Cursor
2. Add the server using `mcp_code_improvement_server.json`
3. The server will connect automatically

### Environment Variables

```bash
export PYTHONPATH="${workspace}/agents/backend/onyx/server/features"
```

## 💡 Usage Examples

### Full Feature Analysis

```python
# Use analyze_code_quality + analyze_architecture + check_security
{
  "tool": "analyze_code_quality",
  "arguments": {
    "path": "features/suno_clone_ai",
    "language": "python",
    "checks": ["all"]
  }
}
```

### Optimizer Refactoring

```python
# Detect unified patterns
{
  "tool": "detect_unified_patterns",
  "arguments": {
    "patternType": "optimizers",
    "minOccurrences": 3
  }
}

# Then apply refactoring
{
  "tool": "suggest_refactoring",
  "arguments": {
    "filePath": "features/suno_clone_ai/core/optimizers/",
    "refactoringType": "extract_class",
    "apply": true
  }
}
```

### Migration to Clean Architecture

```python
{
  "tool": "migrate_to_clean_architecture",
  "arguments": {
    "featurePath": "features/facebook_posts",
    "createStructure": true,
    "migrateCode": true,
    "generateInterfaces": true
  }
}
```

## 📊 Metrics and Reports

The server generates detailed reports including:

- **Code Quality** — Overall score and category breakdown
- **Duplication** — Percentage of duplicate code and locations
- **Security** — Vulnerabilities found and severity
- **Performance** — Identification of optimization opportunities
- **Architecture** — Compliance with Clean Architecture and SOLID
- **Documentation** — Documentation coverage and suggested improvements

## 🔧 Customization

You can customize the server by editing `mcp_code_improvement_server.json`:

- Add new tools
- Modify default parameters
- Add additional resources
- Configure project-specific rules

## 🎓 Best Practices

1. **Use incremental analysis** — Start with quality analysis before refactoring
2. **Review suggestions** — Always review suggestions before applying automatic changes
3. **Tests first** — Generate tests before refactoring critical code
4. **Document changes** — Use `improve_documentation` after major refactorings
5. **Maintain consistency** — Use `analyze_feature_consistency` to maintain uniformity

## 🐛 Troubleshooting

### Server not connecting
- Verify Python is in PATH
- Check environment variables
- Ensure dependencies are installed

### Tools not working
- Check read/write permissions
- Check server logs
- Ensure paths are correct

### Slow performance
- Limit analysis scope
- Use appropriate exclusions
- Consider incremental analysis

## 📝 Notes

- The server is specifically designed for the Blatam Academy project
- Tools are optimized for Python but support multiple languages
- Automatic changes can always be reverted (use version control)
- It is recommended to run analysis in separate branches before merging

## 🤝 Contribution

To improve the MCP server:

1. Identify improvement areas
2. Add new tools as needed
3. Update documentation
4. Test with different project features

## 📄 License

MIT License — Blatam Academy

---

[← Back to Main README](README.md)
