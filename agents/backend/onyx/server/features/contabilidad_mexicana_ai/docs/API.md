# API Documentation - Contabilidad Mexicana AI

## Endpoints

### POST /api/contador/calcular-impuestos

Calcula impuestos para un régimen fiscal específico.

**Request Body:**
```json
{
    "regimen": "RESICO",
    "tipo_impuesto": "ISR",
    "datos": {
        "ingresos_mensuales": 50000,
        "gastos_deducibles": 10000
    }
}
```

**Response:**
```json
{
    "success": true,
    "regimen": "RESICO",
    "tipo_impuesto": "ISR",
    "datos_entrada": {...},
    "resultado": "Cálculo detallado...",
    "tiempo_calculo": 1.23,
    "timestamp": "2024-01-15T10:30:00"
}
```

### POST /api/contador/asesoria-fiscal

Obtiene asesoría fiscal personalizada.

**Request Body:**
```json
{
    "pregunta": "¿Qué deducciones puedo aplicar?",
    "contexto": {
        "regimen": "RESICO"
    }
}
```

### POST /api/contador/guia-fiscal

Genera una guía fiscal sobre un tema específico.

**Request Body:**
```json
{
    "tema": "Facturación electrónica",
    "nivel_detalle": "completo"
}
```

### POST /api/contador/tramite-sat

Obtiene información sobre un trámite del SAT.

**Request Body:**
```json
{
    "tipo_tramite": "Alta en RFC",
    "detalles": {
        "tipo_persona": "Persona Física"
    }
}
```

### POST /api/contador/ayuda-declaracion

Ayuda con la preparación de una declaración fiscal.

**Request Body:**
```json
{
    "tipo_declaracion": "mensual",
    "periodo": "2024-01",
    "datos": {
        "regimen": "RESICO",
        "ingresos": 50000
    }
}
```

### GET /api/contador/regimenes

Lista los regímenes fiscales soportados.

**Response:**
```json
[
    "RESICO",
    "PFAE",
    "Plataformas",
    "Sueldos y Salarios",
    "Personas Físicas",
    "Personas Morales"
]
```

### GET /api/contador/tipos-impuestos

Lista los tipos de impuestos soportados.

**Response:**
```json
[
    "ISR",
    "IVA",
    "IEPS"
]
```

### GET /api/contador/servicios

Lista los servicios disponibles.

**Response:**
```json
[
    "calculo_impuestos",
    "asesoria_fiscal",
    "guias_fiscales",
    "tramites_sat",
    "declaraciones",
    "devoluciones",
    "regularizacion"
]
```

### GET /api/contador/health

Health check endpoint.

**Response:**
```json
{
    "status": "healthy",
    "service": "Contador AI"
}
```
