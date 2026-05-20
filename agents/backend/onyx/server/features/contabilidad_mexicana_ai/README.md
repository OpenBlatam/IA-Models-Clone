# Mexican Accounting AI

> Part of the [Blatam Academy Integrated Platform](../README.md)

AI system to solve Mexican accounting and tax problems based on [Contarely](https://contarely.com/). Provides tax advice, tax calculation, guides, and support for SAT procedures using OpenRouter.

## 🎯 Features

- **Tax Calculation**: ISR, IVA, IEPS for different tax regimes
- **Tax Advice**: Personalized consultations on tax topics
- **Tax Guides**: Comprehensive guides on specific topics
- **SAT Procedures**: Detailed information on SAT procedures
- **Tax Returns**: Help in preparing and filing tax returns
- **Multiple Regimes**: RESICO, PFAE, Wages and Salaries, Individuals, Legal Entities

## 📋 Requirements

- Python 3.8+
- OpenRouter API Key
- FastAPI (for endpoints)

## 🚀 Installation

```bash
# Install dependencies
pip install fastapi httpx uvicorn pydantic

# Configure environment variable
export OPENROUTER_API_KEY="your-api-key"
```

## 📖 Usage

### Basic Usage

```python
from contabilidad_mexicana_ai import ContadorAI, ContadorConfig

# Initialize
config = ContadorConfig()
contador = ContadorAI(config)

# Calculate taxes
resultado = await contador.calcular_impuestos(
    regimen="RESICO",
    tipo_impuesto="ISR",
    datos={
        "ingresos_mensuales": 50000,
        "gastos_deducibles": 10000
    }
)

# Tax advice
asesoria = await contador.asesoria_fiscal(
    pregunta="What deductions can I apply in RESICO?",
    contexto={"regimen": "RESICO", "actividad": "Professional services"}
)

# Generate guide
guia = await contador.guia_fiscal(
    tema="Deductions for entrepreneurs in RESICO",
    nivel_detalle="completo"
)
```

### API Endpoints

#### Calculate Taxes
```bash
POST /api/contador/calcular-impuestos
{
    "regimen": "RESICO",
    "tipo_impuesto": "ISR",
    "datos": {
        "ingresos_mensuales": 50000
    }
}
```

#### Tax Advice
```bash
POST /api/contador/asesoria-fiscal
{
    "pregunta": "What deductions can I apply?",
    "contexto": {
        "regimen": "RESICO"
    }
}
```

#### Tax Guide
```bash
POST /api/contador/guia-fiscal
{
    "tema": "Electronic invoicing",
    "nivel_detalle": "completo"
}
```

#### SAT Procedure
```bash
POST /api/contador/tramite-sat
{
    "tipo_tramite": "RFC Registration",
    "detalles": {
        "tipo_persona": "Persona Física"
    }
}
```

#### Tax Return Help
```bash
POST /api/contador/ayuda-declaracion
{
    "tipo_declaracion": "mensual",
    "periodo": "2024-01",
    "datos": {
        "regimen": "RESICO",
        "ingresos": 50000
    }
}
```

## 🏗️ Architecture

```
contabilidad_mexicana_ai/
├── core/
│   └── contador_ai.py          # Main class
├── config/
│   └── contador_config.py     # Configuration
├── infrastructure/
│   └── openrouter/
│       └── openrouter_client.py  # OpenRouter client
├── api/
│   └── contador_api.py        # FastAPI endpoints
├── services/
│   └── calculadora_impuestos.py  # Specialized calculations
└── utils/
    └── formatters.py          # Formatting utilities
```

## 🔧 Configuration

### Environment Variables

```bash
OPENROUTER_API_KEY=your-api-key
```

### Advanced Configuration

```python
from contabilidad_mexicana_ai import ContadorConfig, OpenRouterConfig

config = ContadorConfig(
    openrouter=OpenRouterConfig(
        default_model="anthropic/claude-3.5-sonnet",
        temperature=0.3,
        max_tokens=4000
    )
)
```

## 📚 Supported Tax Regimes

- **RESICO**: Simplified Trust Regime
- **PFAE**: Individuals with Business Activities
- **Platforms**: Regime for digital platforms
- **Wages and Salaries**: Employees
- **Individuals**: Other regimes
- **Legal Entities**: Companies

## 🔒 Security

- API keys are handled via environment variables
- No sensitive taxpayer data is stored
- All responses are generated in real-time

## 📝 Important Notes

- This system provides AI-based advice and does not replace consultation with a certified accountant
- Calculations must be verified against current tax legislation
- The information provided is for informational purposes only

## 🤝 Contributing

Contributions are welcome. Please:

1. Fork the project
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📄 License

Proprietary - Blatam Academy

## 🔗 References

- [Contarely](https://contarely.com/) - Inspiration and reference
- [SAT](https://www.sat.gob.mx/) - Mexican Tax Administration Service
- [OpenRouter](https://openrouter.ai/) - AI Provider

---

[← Back to Main README](../README.md)
