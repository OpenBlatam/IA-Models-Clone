# MCP Code Improvement

<div align="center">

![Status](https://img.shields.io/badge/status-beta-yellow.svg)
![Version](https://img.shields.io/badge/version-0.5-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![MCP](https://img.shields.io/badge/Protocol-MCP-purple.svg)

**A specialized Model Context Protocol (MCP) server for automated code refactoring, linting, and optimization.**

[Overview](#-overview) •
[Features](#-key-features) •
[Architecture](#-architecture) •
[Installation](#-installation) •
[Usage](#-usage) •
[Contributing](#-contributing)

</div>

---

## 📋 Overview

**MCP Code Improvement** connects LLMs directly to your codebase's quality tools. It exposes tools for static analysis, auto-formatting, and intelligent refactoring as MCP resources and tools, allowing AI assistants (like Cursor or Claude Desktop) to proactively improve code quality.

### Why MCP Code Improvement?

- **Contextual Awareness**: The AI sees the linter errors and the code simultaneously.
- **Safe Refactoring**: Changes are applied via standard tooling, reducing hallucinated syntax.
- **Language Agnostic**: Modular architecture supports Python, TypeScript, Rust, and Go.

## 🚀 Key Features

| Feature | Description |
|---------|-------------|
| **Linter Integration** | Native support for Ruff, ESLint, Pylint, and GolangCI-Lint. |
| **Auto-Fixer** | Apply standard fixes (isort, black, prettier) automatically. |
| **Complexity Analysis** | Calculate Cyclomatic Complexity and suggest simplifications. |
| **Test Generation** | Analyze code and generate unit tests using pytest or Jest. |

## 🏗 Architecture

```mermaid
graph TD
    A[AI Client (Cursor)] -->|MCP Protocol| B(MCP Server)
    
    subgraph "Toolchain"
    B --> C[Linter Runner]
    B --> D[Formatter Runner]
    B --> E[Static Analyzer]
    end
    
    subgraph "File System"
    C <--> F[(Source Code)]
    D <--> F
    end
```

## 💻 Installation

```bash
pip install -r requirements.txt
```

## ⚡ Usage

### Start Server
```bash
python -m mcp_code_improvement.server
```

### Connect to Cursor
Add to your `mcp_config.json`:

```json
{
  "mcpServers": {
    "code-improver": {
      "command": "python",
      "args": ["-m", "mcp_code_improvement.server"]
    }
  }
}
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

---

<div align="center">
  <b>Built with ❤️ by Blatam Academy</b><br>
  Part of the Onyx Server Architecture<br>
  <a href="../README.md">← Back to Main README</a>
</div>
