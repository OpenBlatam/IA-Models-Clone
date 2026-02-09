# Contabilidad Mexicana AI

Sistema de IA para resolver problemas contables y fiscales mexicanos basado en [Contarely](https://contarely.com/). Proporciona asesoría fiscal, cálculo de impuestos, guías y soporte para trámites del SAT usando OpenRouter.

## 🎯 Características

- **Cálculo de Impuestos**: ISR, IVA, IEPS para diferentes regímenes fiscales
- **Asesoría Fiscal**: Consultas personalizadas sobre temas fiscales
- **Guías Fiscales**: Guías completas sobre temas específicos
- **Trámites SAT**: Información detallada sobre trámites del SAT
- **Declaraciones**: Ayuda para preparar y presentar declaraciones
- **Múltiples Regímenes**: RESICO, PFAE, Sueldos y Salarios, Personas Físicas, Personas Morales

## 📋 Requisitos

- Python 3.8+
- OpenRouter API Key
- FastAPI (para endpoints)

## 🚀 Instalación

```bash
# Instalar dependencias
pip install fastapi httpx uvicorn pydantic

# Configurar variable de entorno
export OPENROUTER_API_KEY="tu-api-key"
```

## 📖 Uso

### Uso Básico

```python
from contabilidad_mexicana_ai import ContadorAI, ContadorConfig

# Inicializar
config = ContadorConfig()
contador = ContadorAI(config)

# Calcular impuestos
resultado = await contador.calcular_impuestos(
    regimen="RESICO",
    tipo_impuesto="ISR",
    datos={
        "ingresos_mensuales": 50000,
        "gastos_deducibles": 10000
    }
)

# Asesoría fiscal
asesoria = await contador.asesoria_fiscal(
    pregunta="¿Qué deducciones puedo aplicar en RESICO?",
    contexto={"regimen": "RESICO", "actividad": "Servicios profesionales"}
)

# Generar guía
guia = await contador.guia_fiscal(
    tema="Deducciones para emprendedores en RESICO",
    nivel_detalle="completo"
)
```

### API Endpoints

#### Calcular Impuestos
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

#### Asesoría Fiscal
```bash
POST /api/contador/asesoria-fiscal
{
    "pregunta": "¿Qué deducciones puedo aplicar?",
    "contexto": {
        "regimen": "RESICO"
    }
}
```

#### Guía Fiscal
```bash
POST /api/contador/guia-fiscal
{
    "tema": "Facturación electrónica",
    "nivel_detalle": "completo"
}
```

#### Trámite SAT
```bash
POST /api/contador/tramite-sat
{
    "tipo_tramite": "Alta en RFC",
    "detalles": {
        "tipo_persona": "Persona Física"
    }
}
```

#### Ayuda con Declaración
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

## 🏗️ Arquitectura

```
contabilidad_mexicana_ai/
├── core/
│   └── contador_ai.py          # Clase principal
├── config/
│   └── contador_config.py     # Configuración
├── infrastructure/
│   └── openrouter/
│       └── openrouter_client.py  # Cliente OpenRouter
├── api/
│   └── contador_api.py        # Endpoints FastAPI
├── services/
│   └── calculadora_impuestos.py  # Cálculos especializados
└── utils/
    └── formatters.py          # Utilidades de formato
```

## 🔧 Configuración

### Variables de Entorno

```bash
OPENROUTER_API_KEY=tu-api-key
```

### Configuración Avanzada

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

## 📚 Regímenes Fiscales Soportados

- **RESICO**: Régimen Simplificado de Confianza
- **PFAE**: Personas Físicas con Actividades Empresariales
- **Plataformas**: Régimen para plataformas digitales
- **Sueldos y Salarios**: Asalariados
- **Personas Físicas**: Otros regímenes
- **Personas Morales**: Empresas

## 🎓 Ejemplos

### Ejemplo 1: Cálculo ISR RESICO

```python
resultado = await contador.calcular_impuestos(
    regimen="RESICO",
    tipo_impuesto="ISR",
    datos={
        "ingresos_mensuales": 50000
    }
)
```

### Ejemplo 2: Asesoría sobre Deducciones

```python
asesoria = await contador.asesoria_fiscal(
    pregunta="¿Qué gastos puedo deducir en RESICO?",
    contexto={
        "regimen": "RESICO",
        "actividad": "Consultoría"
    }
)
```

### Ejemplo 3: Guía de Facturación

```python
guia = await contador.guia_fiscal(
    tema="Facturación electrónica CFDI 4.0",
    nivel_detalle="completo"
)
```

## 🔒 Seguridad

- Las API keys se manejan mediante variables de entorno
- No se almacenan datos sensibles de contribuyentes
- Todas las respuestas son generadas en tiempo real

## 📝 Notas Importantes

- Este sistema proporciona asesoría basada en IA y no reemplaza la consulta con un contador certificado
- Los cálculos deben verificarse con la legislación fiscal vigente
- La información proporcionada es de carácter informativo

## 🤝 Contribuir

Las contribuciones son bienvenidas. Por favor:

1. Fork el proyecto
2. Crea una rama para tu feature
3. Commit tus cambios
4. Push a la rama
5. Abre un Pull Request

## 📄 Licencia

Este proyecto es parte de Blatam Academy.

## 🔗 Referencias

- [Contarely](https://contarely.com/) - Inspiración y referencia
- [SAT](https://www.sat.gob.mx/) - Sistema de Administración Tributaria
- [OpenRouter](https://openrouter.ai/) - Proveedor de IA
