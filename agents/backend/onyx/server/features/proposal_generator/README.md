# Proposal Generator

<div align="center">

![Status](https://img.shields.io/badge/status-active-success.svg)
![Version](https://img.shields.io/badge/version-1.5-blue.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Business](https://img.shields.io/badge/domain-Business-orange.svg)

**An intelligent document automation platform for crafting winning business proposals and contracts at scale.**

[Overview](#-overview) •
[Features](#-key-features) •
[Architecture](#-architecture) •
[Installation](#-installation) •
[Usage](#-usage) •
[Templates](#-template-system) •
[Contributing](#-contributing)

</div>

---

## 📋 Overview

**Proposal Generator** is a business-critical tool designed to streamline the sales cycle. By combining dynamic templates with AI-driven content generation, it allows sales teams to produce personalized, legally compliant, and visually professional proposals in seconds rather than hours.

It integrates seamlessly with CRM data to auto-populate client details while ensuring that all legal terms and conditions are up-to-date and consistent across documents.

### Why Proposal Generator?

- **Speed to Lead**: Generate a full proposal during a discovery call.
- **Consistency**: Ensure brand voice and legal compliance across all documents.
- **Personalization**: AI injects client-specific insights and value propositions.

## 🚀 Key Features

| Feature | Description |
|---------|-------------|
| **Smart Templating** | Jinja2-based template engine supporting conditional logic and dynamic variables. |
| **CRM Integration** | Direct connectors for Salesforce, HubSpot, and Pipedrive to fetch client data. |
| **Legal Guardrails** | Clause library with version control to manage standard terms and conditions. |
| **Multi-Format Export** | Generate PDFs, DOCX, and HTML5 web proposals. |
| **AI Content** | GPT-4 integration to draft executive summaries and scope descriptions. |
| **E-Signature Ready** | Integration hooks for DocuSign and HelloSign. |

## 🏗 Architecture

The system uses a layered approach to separate content generation from document rendering.

```mermaid
graph TD
    A[User Input / CRM Data] --> B(Validation Layer)
    B --> C{Proposal Engine}
    
    subgraph "Content Generation"
    C --> D[Template Loader]
    C --> E[AI Writer]
    C --> F[Data Merger]
    end
    
    subgraph "Asset Management"
    D --> G[Template Store]
    F --> H[Legal Clause DB]
    end
    
    subgraph "Rendering"
    F --> I[HTML/Markdown]
    I --> J[PDF Renderer (WeasyPrint)]
    I --> K[DOCX Renderer]
    end
    
    J --> L[Final Document]
```

## 💻 Installation

### Prerequisites

- Python 3.10+
- `wkhtmltopdf` or `WeasyPrint` dependencies (for PDF generation)

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/blatam-academy/proposal_generator.git
   cd proposal_generator
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Templates**
   ```bash
   # Copy example templates
   cp -r examples/templates/ templates/
   ```

## ⚡ Usage

### Python SDK

```python
from proposal_generator import ProposalBuilder, Client

# Define client
client = Client(
    name="Acme Corp",
    contact_person="Jane Doe",
    industry="Logistics"
)

# Initialize builder
builder = ProposalBuilder(template="saas_enterprise")

# Add sections
builder.add_section("Executive Summary", ai_generated=True)
builder.add_section("Scope of Work", items=["Implementation", "Training"])
builder.add_pricing_table([
    {"item": "License", "price": 5000, "qty": 1},
    {"item": "Onboarding", "price": 1500, "qty": 1}
])

# Generate
pdf_path = builder.render_pdf("output/acme_proposal.pdf")
print(f"Proposal generated: {pdf_path}")
```

### CLI Command

```bash
# Generate a standard proposal from a JSON data file
python -m proposal_generator generate \
    --client-data client.json \
    --template standard_services \
    --output proposal.pdf
```

## 🎨 Template System

Templates are built using standard **Markdown** mixed with **Jinja2** syntax, allowing for powerful logic.

**Example `template.md`:**

```markdown
# Proposal for {{ client_name }}

Dear {{ contact_person }},

We are excited to submit this proposal for {{ project_name }}.

## Scope of Work

{% for item in scope_items %}
- **{{ item.title }}**: {{ item.description }}
{% endfor %}

## Pricing

Total Investment: ${{ total_price }}
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">
  <b>Built with ❤️ by Blatam Academy</b><br>
  Part of the Onyx Server Architecture<br>
  <a href="../README.md">← Back to Main README</a>
</div>
